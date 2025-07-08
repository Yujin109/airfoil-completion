import datetime
import os

# config.py をインポートする前に環境変数を反映
import config

config.MASK_TYPE = os.getenv("MASK_TYPE", config.MASK_TYPE)
config.MISSING_POINTS_RATIO = float(os.getenv("MISSING_POINTS_RATIO", config.MISSING_POINTS_RATIO))
config.BLEND_RATIO = float(os.getenv("BLEND_RATIO", config.BLEND_RATIO))
config.BLEND_TYPE = os.getenv("BLEND_TYPE", config.BLEND_TYPE)
config.BLEND_VALUE = float(os.getenv("BLEND_VALUE", config.BLEND_VALUE))
cl_shift_env = os.getenv("CL_SHIFT", "")
config.CL_SHIFT = None if cl_shift_env == "" else float(cl_shift_env)
config.NUM_RESAMPLING = int(os.getenv("NUM_RESAMPLING", config.NUM_RESAMPLING))
config.JUMP_LENGTH = int(os.getenv("JUMP_LENGTH", config.JUMP_LENGTH))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    BLEND_RATIO,
    BLEND_TYPE,
    BLEND_VALUE,
    CL_SHIFT,
    CLUSTERS,
    DIFFUSION_PARAMS,
    FIG_SIZE,
    GUIDANCE_SCALE,
    JUMP_LENGTH,
    MASK_TYPE,
    MISSING_POINTS,
    MISSING_POINTS_RATIO,
    MODEL_NAME,
    NUM_RESAMPLING,
    REPAINT_DIR,
    SAMPLES_PER_CLUSTER,
    SEED,
    SHOW_KNOWN_MISSING,
    WEIGHT_PATH,
)
from src.data import AirfoilDataset
from src.diffusion_blend_mask import RePaintDiffuser
from src.metrics import smoothness_phi
from src.models.model_registry import MODEL_REGISTRY
from src.xfoil_utils import get_cl

# def mask_manual(pattern="head", length=248, missing=10):
#     if pattern == "head":
#         assert missing % 2 == 0
#         m = np.ones(length, dtype=bool)
#         start = (length - missing) // 2
#         m[start : start + missing] = False
#     elif pattern == "tail":
#         assert missing % 2 == 0
#         m = np.ones(length, dtype=bool)
#         miss_len = missing // 2
#         rest = length - missing
#         m[:miss_len] = False
#         m[miss_len + rest :] = False
#     elif pattern == "middle":
#         assert missing % 4 == 0
#         m = np.ones(length, dtype=bool)
#         miss_len = missing // 4
#         m[62 - miss_len : 62 + miss_len] = False
#         m[186 - miss_len : 186 + miss_len] = False
#     return m


def mask_manual(
    pattern: str = "head",
    coords: np.ndarray = None,
    missing_ratio: float = 0.7,
    blend_ratio: float = 0.0,
    blend_type: str = "constant",
    blend_value: float = 0.5,
) -> np.ndarray:
    """
    x 軸の範囲に対して欠損領域およびブレンド領域を指定するマスク生成関数。
    pattern: "head", "tail", "middle"
    coords: numpy array shape (2, L) または 1D array of x (length L)
    missing_ratio: ドメイン全体に対する欠損割合 (0.0–1.0)
    blend_ratio: ドメイン全体に対するブレンド領域割合 (0.0–1.0)、missing_ratio + blend_ratio <= 1
    blend_type: "constant" or "linear"
    blend_value: constant 時に使うマスク値 (0.0–1.0)
    戻り値: 長さ L の float 配列 (1=known, 0=missing, (0,1)=blend)
    """
    if coords is None:
        raise ValueError("coords must be provided")
    if pattern not in {"head", "tail", "middle", "top", "bottom"}:
        raise ValueError(f"Unknown pattern: {pattern}")
    if pattern not in {"top", "bottom"}:
        if not (0.0 <= missing_ratio <= 1.0 and 0.0 <= blend_ratio <= 1.0 and missing_ratio + blend_ratio <= 1.0):
            raise ValueError("missing_ratio and blend_ratio must be in [0,1] and sum <= 1")

    # x 座標のみ取り出し
    xs = coords[0] if coords.ndim == 2 else coords
    x_min, x_max = xs.min(), xs.max()
    range_x = x_max - x_min

    # インデックス配列
    n = xs.shape[0]
    half_n = n // 2
    blend_len = int(blend_ratio * n)

    # ゼロから始めて最後に known 部分を1で埋める
    mask = np.zeros_like(xs, dtype=float)

    # missing 範囲の定義
    if pattern == "head":
        miss_start = x_min
        miss_end = x_min + missing_ratio * range_x
        blend_start = miss_end
        blend_end = miss_end + blend_ratio * range_x

        mask[xs > blend_end] = 1.0
        mask[(xs >= miss_start) & (xs <= miss_end)] = 0.0

        blend_idx = (xs > blend_start) & (xs <= blend_end)
        if blend_type == "constant":
            mask[blend_idx] = blend_value
        elif blend_type == "linear":
            mask[blend_idx] = (xs[blend_idx] - blend_start) / (blend_end - blend_start)
        else:
            raise ValueError(f"Unknown blend_type: {blend_type}")

    elif pattern == "tail":
        miss_end = x_max
        miss_start = x_max - missing_ratio * range_x
        blend_end = miss_start
        blend_start = miss_start - blend_ratio * range_x

        mask[xs < blend_start] = 1.0
        mask[(xs >= miss_start) & (xs <= miss_end)] = 0.0

        blend_idx = (xs >= blend_start) & (xs < blend_end)
        if blend_type == "constant":
            mask[blend_idx] = blend_value
        elif blend_type == "linear":
            mask[blend_idx] = 1.0 - (xs[blend_idx] - blend_start) / (blend_end - blend_start)
        else:
            raise ValueError(f"Unknown blend_type: {blend_type}")

    elif pattern == "middle":
        half_known = (1.0 - missing_ratio) * range_x / 2
        miss_start = x_min + half_known
        miss_end = x_max - half_known
        blend_half = blend_ratio * range_x / 2
        blend_left_start = miss_start - blend_half
        blend_left_end = miss_start
        blend_right_start = miss_end
        blend_right_end = miss_end + blend_half

        # known
        mask[xs <= blend_left_start] = 1.0
        mask[xs >= blend_right_end] = 1.0
        # missing
        mask[(xs > miss_start) & (xs < miss_end)] = 0.0

        # left blend
        left_idx = (xs > blend_left_start) & (xs <= blend_left_end)
        if blend_type == "constant":
            mask[left_idx] = blend_value
        else:  # linear
            mask[left_idx] = (xs[left_idx] - blend_left_start) / (blend_left_end - blend_left_start)

        # right blend
        right_idx = (xs >= blend_right_start) & (xs < blend_right_end)
        if blend_type == "constant":
            mask[right_idx] = blend_value
        else:  # linear
            mask[right_idx] = (xs[right_idx] - blend_right_start) / (blend_right_end - blend_right_start)

    elif pattern == "top":
        # 「上半面」（idx half_n～n-1）を欠損
        miss_idx = np.arange(half_n, n)
        known_idx = np.arange(0, half_n)

        mask[miss_idx] = 0.0
        mask[known_idx] = 1.0

        # ブレンド幅（x軸割合の左右それぞれ blend_ratio/2）
        bw = blend_ratio * range_x / 2

        # 左側ブレンド (x_min -> x_min + bw)
        if bw > 0:
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            if blend_type == "constant":
                mask[idxs] = blend_value
            else:
                mask[idxs] = (xs[idxs] - bs) / (be - bs)

        # 右側ブレンド (x_max - bw -> x_max)
        if bw > 0:
            bs, be = x_max - bw, x_max
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            if blend_type == "constant":
                mask[idxs] = blend_value
            else:
                mask[idxs] = (be - xs[idxs]) / (be - bs)

    elif pattern == "bottom":
        # 「下半面」（idx 0～half_n-1）を欠損
        miss_idx = np.arange(0, half_n)
        known_idx = np.arange(half_n, n)

        mask[miss_idx] = 0.0
        mask[known_idx] = 1.0

        # ブレンド幅（x軸割合の左右それぞれ blend_ratio/2）
        bw = blend_ratio * range_x / 2

        # 左側ブレンド (x_min -> x_min + bw)
        if bw > 0:
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            if blend_type == "constant":
                mask[idxs] = blend_value
            else:
                mask[idxs] = (xs[idxs] - bs) / (be - bs)

        # 右側ブレンド (x_max - bw -> x_max)
        if bw > 0:
            bs, be = x_max - bw, x_max
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            if blend_type == "constant":
                mask[idxs] = blend_value
            else:
                mask[idxs] = (be - xs[idxs]) / (be - bs)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return mask


blue_rgb = mcolors.to_rgb("tab:blue")
orange_rgb = mcolors.to_rgb("tab:orange")
mid_rgb = tuple((b + o) / 2 for b, o in zip(blue_rgb, orange_rgb))


def plot_inpaint(orig, rep, mask, ax, title="", show_known_missing=False):
    # mask は float になったので、known/missing/blend を切り分け
    m = mask
    known = m == 1.0
    missing = m == 0.0
    blend = (m > 0.0) & (m < 1.0)

    # known 部分
    ax.plot(
        np.ma.array(orig[0], mask=~known),
        np.ma.array(orig[1], mask=~known),
        label="known",
        linestyle="-",
        color="tab:blue",
    )
    # blend 部分 (補間値)
    if blend.any():
        ax.plot(
            np.ma.array(rep[0], mask=~blend),
            np.ma.array(rep[1], mask=~blend),
            label="blend",
            linestyle="-",
            color=mid_rgb,
        )
    # inpaint 部分
    ax.plot(
        np.ma.array(rep[0], mask=~missing),
        np.ma.array(rep[1], mask=~missing),
        label="inpaint",
        linestyle="-",
        color="tab:orange",
    )
    if show_known_missing:
        ax.plot(
            np.ma.array(orig[0], mask=~missing),
            np.ma.array(orig[1], mask=~missing),
            label="known_missing",
            linestyle="--",
            color="tab:blue",
        )
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 0.35)
    ax.tick_params(labelsize=9)
    ax.grid(True)


def main():
    # prepare directories
    os.makedirs(REPAINT_DIR, exist_ok=True)

    # load dataset and model
    dataset = AirfoilDataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_REGISTRY[MODEL_NAME]
    model = model_cls().to(device)
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # initialize RePaint diffuser
    diffuser = RePaintDiffuser(
        **DIFFUSION_PARAMS,
        device=device,
        guidance_scale=GUIDANCE_SCALE,
        num_resampling=NUM_RESAMPLING,
        jump_length=JUMP_LENGTH,
    )

    # select sample indices per cluster
    np.random.seed(SEED)
    original_cls = dataset.denormalize_cl(dataset.cls_tensor).numpy().flatten()
    selected = []
    for lo, hi in CLUSTERS:
        idxs = np.where((original_cls >= lo) & (original_cls < hi))[0]
        if len(idxs) < SAMPLES_PER_CLUSTER:
            raise ValueError(f"Not enough samples in cluster [{lo},{hi}): {len(idxs)} found")
        sel = np.random.choice(idxs, SAMPLES_PER_CLUSTER, replace=False)
        selected.extend(sel)

    # prepare inputs
    coords_batch = dataset.coords_tensor[selected].to(device)
    cls_batch = dataset.cls_tensor[selected].to(device)

    # create masks
    coords_batch_denorm = dataset.denormalize_coord(coords_batch).cpu().numpy()
    masks_np = []
    for coords in coords_batch_denorm:
        mask = mask_manual(
            pattern=MASK_TYPE,
            coords=coords,
            missing_ratio=MISSING_POINTS_RATIO,
            blend_ratio=BLEND_RATIO,  # 例: 10% を blend に使う
            blend_type=BLEND_TYPE,  # "linear" or "constant"
            blend_value=BLEND_VALUE,  # constant 時に使う値
        )
        masks_np.append(mask)
    # masks = torch.stack([torch.from_numpy(m) for m in masks_np], dim=0).to(device)
    # numpy→torch の際に float32 に変換してスタック
    masks = torch.stack([torch.from_numpy(m).float() for m in masks_np], dim=0).to(device)

    if CL_SHIFT is not None:
        cls_batch += CL_SHIFT / dataset.cl_std
        for idx in selected:
            original_cls[idx] = original_cls[idx] + CL_SHIFT

    # run RePaint inpainting
    repainted = diffuser.repaint(model, coords_batch, masks, cls_batch)

    # denormalize for plotting
    orig_denorm = dataset.denormalize_coord(coords_batch).cpu().numpy()
    rep_denorm = dataset.denormalize_coord(repainted).cpu().numpy()

    # compute CL on inpainted
    cl_out_list = []
    for sample in rep_denorm:
        cl_out_list.append(get_cl(sample))

    # plot results
    fig, axes = plt.subplots(SAMPLES_PER_CLUSTER, len(CLUSTERS), figsize=FIG_SIZE)
    for i, idx in enumerate(selected):
        row = i // len(CLUSTERS)
        col = i % len(CLUSTERS)
        ax = axes[row, col]
        if np.isnan(cl_out_list[i]):
            cl_ape = np.nan
        else:
            cl_ape = abs((cl_out_list[i] - original_cls[idx]) / original_cls[idx] * 100)
        title = f"CL_in: {original_cls[idx]:.3f}\nCL_out: {cl_out_list[i]:.3f}\nAPE: {cl_ape:.1f}"
        plot_inpaint(orig_denorm[i], rep_denorm[i], masks[i].cpu().numpy(), ax, title, SHOW_KNOWN_MISSING)
    plt.tight_layout()

    # save figure
    date_str = datetime.datetime.now().strftime("%y%m%d%H%M")
    fname = (
        f"{date_str}_R:{NUM_RESAMPLING}_J:{JUMP_LENGTH}_M:{MASK_TYPE}_MPR:{MISSING_POINTS_RATIO}_SHIFT:{CL_SHIFT}.png"
    )
    if BLEND_RATIO > 0.0:
        if BLEND_TYPE == "linear":
            blend_str = f"BLEND:{BLEND_RATIO:.2f}_{BLEND_TYPE}"
        elif BLEND_TYPE == "constant":
            blend_str = f"BLEND:{BLEND_RATIO:.2f}_{BLEND_TYPE}_{BLEND_VALUE:.2f}"
        else:
            raise ValueError(f"Unknown BLEND_TYPE: {BLEND_TYPE}")
        fname = fname.replace(".png", f"_{blend_str}.png")

    fpath = os.path.join(REPAINT_DIR, fname)
    fig.savefig(fpath, dpi=300)
    print(f"[Saved sample images] {fpath}")

    # compute metrics

    metrics = dict()
    # MSE on cls
    cl_se_list = []
    cl_ape_list = []
    convergence_count = 0
    smoothness_phi_list = []
    for i, idx in enumerate(selected):
        cl_out = cl_out_list[i]
        cl_in = original_cls[idx]
        coord = dataset.coords_tensor[idx].cpu().numpy()
        if np.isnan(cl_out):
            cl_se_list.append(np.nan)
            cl_ape_list.append(np.nan)
        else:
            cl_se_list.append((cl_out - cl_in) ** 2)
            cl_ape_list.append(abs((cl_out - cl_in) / cl_in * 100))
            convergence_count += 1
        smoothness_phi_list.append(smoothness_phi(coord))
    metrics["cl_mse_mean"] = np.nanmean(cl_se_list)
    metrics["cl_mape_mean"] = np.nanmean(cl_ape_list)
    metrics["cl_convergence_ratio"] = convergence_count / len(selected)
    metrics["cl_smoothness_phi_mean"] = np.nanmean(smoothness_phi_list)

    for i, idx in enumerate(selected):
        cl_out = cl_out_list[i]
        cl_in = original_cls[idx]
        cl_ape_list.append(abs((cl_out - cl_in) / cl_in * 100))
    metrics["cl_ape_mean"] = np.mean(cl_ape_list)

    # .txtファイルとして保存
    txt_fname = (
        f"{date_str}_R:{NUM_RESAMPLING}_J:{JUMP_LENGTH}_M:{MASK_TYPE}_MPR:{MISSING_POINTS_RATIO}_SHIFT:{CL_SHIFT}.txt"
    )
    if BLEND_RATIO > 0.0:
        if BLEND_TYPE == "linear":
            txt_fname = txt_fname.replace(".txt", f"_BLEND:{BLEND_RATIO:.2f}_{BLEND_TYPE}.txt")
        elif BLEND_TYPE == "constant":
            txt_fname = txt_fname.replace(".txt", f"_BLEND:{BLEND_RATIO:.2f}_{BLEND_TYPE}_{BLEND_VALUE:.2f}.txt")
        else:
            raise ValueError(f"Unknown BLEND_TYPE: {BLEND_TYPE}")

    with open(
        os.path.join(
            REPAINT_DIR,
            txt_fname,
        ),
        "w",
    ) as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()

import datetime
import os

# config.py をインポートする前に環境変数を反映
import config

config.MASK_TYPE = os.getenv("MASK_TYPE", config.MASK_TYPE)
config.MISSING_POINTS_RATIO = float(os.getenv("MISSING_POINTS_RATIO", config.MISSING_POINTS_RATIO))
cl_shift_env = os.getenv("CL_SHIFT", "")
config.CL_SHIFT = None if cl_shift_env == "" else float(cl_shift_env)
config.NUM_RESAMPLING = int(os.getenv("NUM_RESAMPLING", config.NUM_RESAMPLING))
config.JUMP_LENGTH = int(os.getenv("JUMP_LENGTH", config.JUMP_LENGTH))

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
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
from src.diffusion import RePaintDiffuser
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


def mask_manual(pattern="head", coords=None, missing_ratio=0.7):
    """
    x 軸の範囲に対して欠損領域を指定するマスク生成関数。
    pattern: "head", "tail", "middle"
    coords: numpy array shape (2, L) または 1D array of x (length L)
    missing_ratio: ドメイン全体に対する欠損割合 (0.0–1.0)
    戻り値: 長さ L の bool 配列 (True=known, False=missing)
    """
    # x 座標のみ取り出し
    if coords.ndim == 2:
        xs = coords[0]  # shape (L,)
    else:
        xs = coords  # shape (L,)

    x_min, x_max = xs.min(), xs.max()
    range_x = x_max - x_min

    if pattern == "head":
        # ドメインの先端側を missing_ratio 分だけ欠損
        mask = np.ones_like(xs, dtype=bool)
        end = x_min + missing_ratio * range_x
        mask[(xs >= x_min) & (xs <= end)] = False

    elif pattern == "tail":
        # ドメインの末端側を missing_ratio 分だけ欠損
        mask = np.ones_like(xs, dtype=bool)
        start = x_max - missing_ratio * range_x
        mask[(xs >= start) & (xs <= x_max)] = False

    elif pattern == "middle":
        # ドメイン両端に 1-missing_ratioをequally split した既知領域を配置
        mask = np.zeros_like(xs, dtype=bool)
        half = (1 - missing_ratio) * range_x / 2
        # 先端側欠損
        mask[(xs >= x_min) & (xs <= x_min + half)] = True
        # 末端側欠損
        mask[(xs >= x_max - half) & (xs <= x_max)] = True

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return mask


def plot_inpaint(orig, rep, mask, ax, title="", show_known_missing=False):
    m = mask
    ax.plot(
        np.ma.array(orig[0], mask=~m), np.ma.array(orig[1], mask=~m), label="known", linestyle="-", color="tab:blue"
    )
    ax.plot(
        np.ma.array(rep[0], mask=m), np.ma.array(rep[1], mask=m), label="inpaint", linestyle="-", color="tab:orange"
    )
    if show_known_missing:
        ax.plot(
            np.ma.array(orig[0], mask=m),
            np.ma.array(orig[1], mask=m),
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
        mask = mask_manual(pattern=MASK_TYPE, coords=coords, missing_ratio=MISSING_POINTS_RATIO)
        masks_np.append(mask)
    # Tensor に変換してデバイスへ
    masks = torch.stack([torch.from_numpy(m) for m in masks_np], dim=0).to(device)

    # masks = torch.stack(
    #     [torch.from_numpy(mask_manual(pattern=MASK_TYPE, missing=MISSING_POINTS)) for _ in selected], dim=0
    # ).to(device)

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
    with open(
        os.path.join(
            REPAINT_DIR,
            f"{date_str}_R:{NUM_RESAMPLING}_J:{JUMP_LENGTH}_M:{MASK_TYPE}_MPR:{MISSING_POINTS_RATIO}_SHIFT:{CL_SHIFT}.txt",
        ),
        "w",
    ) as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()

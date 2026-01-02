#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
500_morphing.py

目的:
- 学習データから CL≈0.5 付近の候補を取り、CD ターゲットに最も近い 7 翼型を選ぶ（欠損前形状）
- 各翼型に対し CL_SHIFT を変えながら RePaint inpainting を実行
- 各生成結果について XFoil で CL/CD を計算
- 生成結果の可視化を保存:
  1) 横一列（base + CL_SHIFT 別）
  2) 生成結果の全重ね（CL_SHIFT 別に色）
  3) 学習データ + 生成結果の CL–CD 散布図（CL_SHIFT を色で識別、colorbar 付き）
- 生成座標・メトリクス（CSV/JSON）も保存

保存先:
results/{EXECUTION_NAME}/morphing/{tag}/
  ├─ base01_idx.../
  │    ├─ row.png
  │    ├─ overlay.png
  │    ├─ cl_cd_scatter.png
  │    ├─ generated.npy
  │    ├─ metrics.csv
  │    └─ meta.json
  └─ ...
キャッシュ:
results/{EXECUTION_NAME}/morphing/_cache/train_clcd_cache_angleX.npz
"""

import argparse
import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import DIFFUSION_PARAMS, EXECUTION_NAME, GUIDANCE_SCALE, MODEL_NAME, WEIGHT_PATH
from src.data import AirfoilDataset
from src.diffusion_blend_mask import RePaintDiffuser
from src.models.model_registry import MODEL_REGISTRY
from src.xfoil_utils import get_xf_instance


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_shifts(s: str) -> List[Optional[float]]:
    """
    例: "None,0.1,0.2,0.3"
    """
    out: List[Optional[float]] = []
    for tok in [x.strip() for x in s.split(",") if x.strip() != ""]:
        if tok.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(float(tok))
    return out


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


# ---------------------------
# XFoil: CL/CD (single call)
# ---------------------------
def get_cl_cd(coord: np.ndarray, angle: float = 5.0) -> Tuple[float, float]:
    """
    coord: (2, L)
    戻り: (cl, cd) 失敗時は (nan, nan)
    """
    try:
        from xfoil.model import Airfoil as _Airfoil

        xf = get_xf_instance()
        xf.Re = 3e6
        xf.max_iter = 100
        x, y = coord.reshape(2, -1)
        xf.airfoil = _Airfoil(x=x, y=y)
        c = xf.a(angle)
        if c is None or len(c) < 2:
            return (np.nan, np.nan)
        return (float(np.round(c[0], 10)), float(np.round(c[1], 10)))
    except Exception:
        return (np.nan, np.nan)


# ---------------------------
# Mask (304 と同系; head を主用途)
# ---------------------------
def mask_manual(
    pattern: str,
    coords: np.ndarray,
    missing_ratio: float,
    blend_ratio: float,
    blend_type: str,
    blend_value: float,
) -> np.ndarray:
    if coords is None:
        raise ValueError("coords must be provided")
    if pattern not in {"head", "tail", "middle", "top", "bottom"}:
        raise ValueError(f"Unknown pattern: {pattern}")

    xs = coords[0]
    x_min, x_max = xs.min(), xs.max()
    range_x = x_max - x_min
    n = xs.shape[0]
    half_n = n // 2

    if pattern not in {"top", "bottom"}:
        if not (0.0 <= missing_ratio <= 1.0 and 0.0 <= blend_ratio <= 1.0 and (missing_ratio + blend_ratio) <= 1.0):
            raise ValueError("missing_ratio and blend_ratio must be in [0,1] and sum <= 1")

    mask = np.zeros_like(xs, dtype=float)

    if pattern == "head":
        miss_start = x_min
        miss_end = x_min + missing_ratio * range_x
        blend_start = miss_end
        blend_end = miss_end + blend_ratio * range_x

        mask[xs > blend_end] = 1.0
        mask[(xs >= miss_start) & (xs <= miss_end)] = 0.0

        idx = (xs > blend_start) & (xs <= blend_end)
        if blend_ratio > 0:
            if blend_type == "constant":
                mask[idx] = blend_value
            elif blend_type == "linear":
                mask[idx] = (xs[idx] - blend_start) / max((blend_end - blend_start), 1e-12)
            else:
                raise ValueError(f"Unknown blend_type: {blend_type}")

    elif pattern == "tail":
        miss_end = x_max
        miss_start = x_max - missing_ratio * range_x
        blend_end = miss_start
        blend_start = miss_start - blend_ratio * range_x

        mask[xs < blend_start] = 1.0
        mask[(xs >= miss_start) & (xs <= miss_end)] = 0.0

        idx = (xs >= blend_start) & (xs < blend_end)
        if blend_ratio > 0:
            if blend_type == "constant":
                mask[idx] = blend_value
            elif blend_type == "linear":
                mask[idx] = 1.0 - (xs[idx] - blend_start) / max((blend_end - blend_start), 1e-12)
            else:
                raise ValueError(f"Unknown blend_type: {blend_type}")

    elif pattern == "middle":
        half_known = (1.0 - missing_ratio) * range_x / 2
        miss_start = x_min + half_known
        miss_end = x_max - half_known
        blend_half = blend_ratio * range_x / 2

        bls, ble = miss_start - blend_half, miss_start
        brs, bre = miss_end, miss_end + blend_half

        mask[xs <= bls] = 1.0
        mask[xs >= bre] = 1.0
        mask[(xs > miss_start) & (xs < miss_end)] = 0.0

        left_idx = (xs > bls) & (xs <= ble)
        right_idx = (xs >= brs) & (xs < bre)
        if blend_ratio > 0:
            if blend_type == "constant":
                mask[left_idx] = blend_value
                mask[right_idx] = blend_value
            elif blend_type == "linear":
                mask[left_idx] = (xs[left_idx] - bls) / max((ble - bls), 1e-12)
                mask[right_idx] = (xs[right_idx] - brs) / max((bre - brs), 1e-12)
            else:
                raise ValueError(f"Unknown blend_type: {blend_type}")

    elif pattern == "top":
        miss_idx = np.arange(half_n, n)
        known_idx = np.arange(0, half_n)
        mask[miss_idx] = 0.0
        mask[known_idx] = 1.0
        bw = blend_ratio * range_x / 2
        if bw > 0:
            # left
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (xs[idxs] - bs) / max((be - bs), 1e-12)
            # right
            bs, be = x_max - bw, x_max
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (be - xs[idxs]) / max((be - bs), 1e-12)

    elif pattern == "bottom":
        miss_idx = np.arange(0, half_n)
        known_idx = np.arange(half_n, n)
        mask[miss_idx] = 0.0
        mask[known_idx] = 1.0
        bw = blend_ratio * range_x / 2
        if bw > 0:
            # left
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (xs[idxs] - bs) / max((be - bs), 1e-12)
            # right
            bs, be = x_max - bw, x_max
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (be - xs[idxs]) / max((be - bs), 1e-12)

    return mask.astype(np.float32)


# ---------------------------
# Plot helpers (304 ベース)
# ---------------------------
blue_rgb = mcolors.to_rgb("tab:blue")
orange_rgb = mcolors.to_rgb("tab:orange")
mid_rgb = tuple((b + o) / 2 for b, o in zip(blue_rgb, orange_rgb))


def plot_inpaint(orig: np.ndarray, rep: np.ndarray, mask: np.ndarray, ax, title: str = "") -> None:
    """
    orig, rep: (2,L)
    mask: (L,) float (1=known, 0=missing, (0,1)=blend)
    """
    m = mask
    known = m == 1.0
    missing = m == 0.0
    blend = (m > 0.0) & (m < 1.0)

    ax.plot(
        np.ma.array(orig[0], mask=~known),
        np.ma.array(orig[1], mask=~known),
        label="known",
        linestyle="-",
        color="tab:blue",
    )
    if blend.any():
        ax.plot(
            np.ma.array(rep[0], mask=~blend),
            np.ma.array(rep[1], mask=~blend),
            label="blend",
            linestyle="-",
            color=mid_rgb,
        )
    ax.plot(
        np.ma.array(rep[0], mask=~missing),
        np.ma.array(rep[1], mask=~missing),
        label="inpaint",
        linestyle="-",
        color="tab:orange",
    )

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.25, 0.4)
    ax.grid(True)
    ax.tick_params(labelsize=8)


def plot_base_with_mask(base: np.ndarray, mask: np.ndarray, ax, title: str = "") -> None:
    """
    欠損前形状（base）を表示し、mask の known/missing を見えるようにする。
    """
    m = mask
    known = m == 1.0
    missing = m == 0.0
    blend = (m > 0.0) & (m < 1.0)

    # full base
    ax.plot(base[0], base[1], color="k", linewidth=1.5, label="base(full)")
    # highlight known/missing/blend along base
    ax.plot(
        np.ma.array(base[0], mask=~known),
        np.ma.array(base[1], mask=~known),
        color="tab:blue",
        linewidth=2.0,
        label="known",
    )
    ax.plot(
        np.ma.array(base[0], mask=~missing),
        np.ma.array(base[1], mask=~missing),
        color="tab:orange",
        linewidth=2.0,
        label="missing(region)",
    )
    if blend.any():
        ax.plot(
            np.ma.array(base[0], mask=~blend),
            np.ma.array(base[1], mask=~blend),
            color=mid_rgb,
            linewidth=2.0,
            label="blend(region)",
        )

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.25, 0.4)
    ax.grid(True)
    ax.tick_params(labelsize=8)


# ---------------------------
# Train cache for CL/CD
# ---------------------------
def get_or_build_train_clcd_cache(dataset: AirfoilDataset, cache_dir: str, angle: float) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"train_clcd_cache_angle{angle:.3f}.npz")
    if os.path.exists(cache_path):
        return cache_path

    coords = dataset.denormalize_coord(dataset.coords_tensor).cpu().numpy()
    cls = np.empty(coords.shape[0], dtype=float)
    cds = np.empty(coords.shape[0], dtype=float)

    for i in tqdm(range(coords.shape[0]), desc=f"Build train CL/CD cache (angle={angle:.3f})"):
        cl, cd = get_cl_cd(coords[i], angle=angle)
        cls[i] = cl
        cds[i] = cd

    np.savez_compressed(cache_path, cls=cls, cds=cds, dataset_size=coords.shape[0], angle=angle)
    return cache_path


# ---------------------------
# Base selection: CL≈target, CD≈targets (unique)
# ---------------------------
def select_bases_by_cl_cd(
    cls_labels: np.ndarray,
    cds_train: np.ndarray,
    target_cl: float,
    cl_window_init: float,
    cd_targets: List[float],
) -> List[int]:
    """
    - cls_labels: dataset のラベルCL（denorm）
    - cds_train: XFoil 計算済み CD（train cache）
    - target_cl 付近で候補を絞り、各 cd_target に一番近いものを重複なく選ぶ
    """
    windows = [cl_window_init, cl_window_init * 2, cl_window_init * 4, max(0.2, cl_window_init * 8)]
    cd_targets = list(cd_targets)

    for w in windows:
        cand = np.where(np.abs(cls_labels - target_cl) <= w)[0]
        # CD が NaN の候補は除外
        cand = cand[~np.isnan(cds_train[cand])]

        if len(cand) < len(cd_targets):
            continue

        remaining = cand.copy()
        chosen: List[int] = []
        for cd_t in cd_targets:
            if len(remaining) == 0:
                chosen = []
                break
            d = np.abs(cds_train[remaining] - cd_t)
            pick = int(remaining[int(np.argmin(d))])
            chosen.append(pick)
            remaining = remaining[remaining != pick]

        if len(chosen) == len(cd_targets):
            return chosen

    raise RuntimeError(
        f"Failed to pick {len(cd_targets)} bases near CL={target_cl} (init window={cl_window_init}). "
        "Try increasing --cl_window or relaxing CD targets."
    )


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    # selection
    parser.add_argument("--target_cl", type=float, default=0.5)
    parser.add_argument("--cl_window", type=float, default=0.05)
    parser.add_argument(
        "--cd_targets",
        type=str,
        default="0.005,0.0010,0.0015,0.0020,0.0025,0.0030,0.0035",
        help="comma-separated",
    )

    # repaint common settings (as requested)
    parser.add_argument("--mask_type", type=str, default="head", choices=["head", "tail", "middle", "top", "bottom"])
    parser.add_argument("--missing_points_ratio", type=float, default=0.6)
    parser.add_argument("--num_resampling", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=10)
    parser.add_argument("--blend_ratio", type=float, default=0.1)
    parser.add_argument("--blend_type", type=str, default="linear", choices=["linear", "constant"])
    parser.add_argument("--blend_value", type=float, default=0.5)  # constant 用（linear では基本未使用）

    # shifts
    parser.add_argument(
        "--cl_shifts",
        type=str,
        default="None,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="comma-separated; use None for no-shift",
    )
    parser.add_argument("--n_per_shift", type=int, default=1, help="samples per CL_SHIFT (default 1)")

    # xfoil
    parser.add_argument("--angle", type=float, default=5.0)

    # scatter
    parser.add_argument("--train_max_points", type=int, default=8000, help="max train points to draw on scatter")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # dirs
    base_dir = os.path.join("results", EXECUTION_NAME, "morphing")
    cache_dir = os.path.join(base_dir, "_cache")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    tag = (
        f"{date_str}_morphing"
        f"_CL{args.target_cl}"
        f"_M{args.mask_type}_MPR{args.missing_points_ratio}"
        f"_R{args.num_resampling}_J{args.jump_length}"
        f"_BR{args.blend_ratio}_{args.blend_type}"
    )
    run_dir = os.path.join(base_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    # load dataset/model
    dataset = AirfoilDataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = MODEL_REGISTRY[MODEL_NAME]
    model = model_cls().to(device)
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    diffuser = RePaintDiffuser(
        **DIFFUSION_PARAMS,
        device=device,
        guidance_scale=GUIDANCE_SCALE,
        num_resampling=args.num_resampling,
        jump_length=args.jump_length,
        model_name=MODEL_NAME,  # denoise 分岐の安定化
    )

    # prepare train CL/CD cache
    train_cache_path = get_or_build_train_clcd_cache(dataset, cache_dir, args.angle)
    train_cache = np.load(train_cache_path)
    train_cls_xfoil = train_cache["cls"]
    train_cds_xfoil = train_cache["cds"]

    # labels (dataset's conditioning CL)
    cls_labels = dataset.denormalize_cl(dataset.cls_tensor).numpy().flatten()

    # base selection
    cd_targets = parse_float_list(args.cd_targets)
    base_indices = select_bases_by_cl_cd(
        cls_labels=cls_labels,
        cds_train=train_cds_xfoil,
        target_cl=args.target_cl,
        cl_window_init=args.cl_window,
        cd_targets=cd_targets,
    )

    shifts = parse_shifts(args.cl_shifts)

    # scatter: downsample train points for plotting
    rng = np.random.default_rng(args.seed)
    valid_train = (~np.isnan(train_cls_xfoil)) & (~np.isnan(train_cds_xfoil))
    valid_idx = np.where(valid_train)[0]
    if args.train_max_points > 0 and len(valid_idx) > args.train_max_points:
        plot_train_idx = rng.choice(valid_idx, size=args.train_max_points, replace=False)
    else:
        plot_train_idx = valid_idx

    # save run meta
    run_meta = {
        "date": date_str,
        "execution_name": EXECUTION_NAME,
        "model_name": MODEL_NAME,
        "weight_path": WEIGHT_PATH,
        "tag": tag,
        "args": vars(args),
        "train_cache_path": train_cache_path,
        "picked_base_indices": [int(x) for x in base_indices],
    }
    with open(os.path.join(run_dir, f"{tag}_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    print(f"[RUN_DIR] {run_dir}")
    print("[BASE INDICES]", base_indices)

    # loop bases
    for b_i, idx in enumerate(base_indices, start=1):
        base_cl_label = float(cls_labels[idx])
        base_cd = float(train_cds_xfoil[idx]) if not np.isnan(train_cds_xfoil[idx]) else np.nan

        base_subdir = os.path.join(run_dir, f"base{b_i:02d}_idx{idx}_CL{base_cl_label:.3f}_CD{base_cd:.4f}")
        os.makedirs(base_subdir, exist_ok=True)

        # base tensors (normalized)
        coord_std = dataset.coords_tensor[idx : idx + 1].to(device)
        cls_std = dataset.cls_tensor[idx : idx + 1].to(device)

        # mask computed on denorm coords
        coord_denorm = dataset.denormalize_coord(coord_std).cpu().numpy()[0]  # (2,L)
        mask_np = mask_manual(
            pattern=args.mask_type,
            coords=coord_denorm,
            missing_ratio=args.missing_points_ratio,
            blend_ratio=args.blend_ratio,
            blend_type=args.blend_type,
            blend_value=args.blend_value,
        )
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(device).float()  # (1,L)

        # run shifts
        # collect: coords[S, N, 2, L], metrics rows
        S = len(shifts)
        N = args.n_per_shift
        L = coord_denorm.shape[1]
        gen_all = np.zeros((S, N, 2, L), dtype=np.float32)
        metrics_rows: List[Dict] = []

        for s_i, shift in enumerate(shifts):
            shift_val = 0.0 if shift is None else float(shift)
            cl_target = base_cl_label + shift_val

            # conditioning shift in normalized space
            cls_batch = cls_std.repeat(N, 1)
            if shift is not None and shift_val != 0.0:
                cls_batch = cls_batch + (shift_val / dataset.cl_std)

            coord_batch = coord_std.repeat(N, 1, 1)
            mask_batch = mask.repeat(N, 1)

            # reproducibility: base seed + base index + shift index
            set_seed(args.seed + b_i * 1000 + s_i * 10)

            with torch.no_grad():
                repainted_std = diffuser.repaint(model, coord_batch, mask_batch, cls_batch)

            rep_denorm = dataset.denormalize_coord(repainted_std).cpu().numpy()  # (N,2,L)
            gen_all[s_i] = rep_denorm.astype(np.float32)

            # xfoil CL/CD
            for n_i in range(N):
                cl_out, cd_out = get_cl_cd(rep_denorm[n_i], angle=args.angle)
                metrics_rows.append(
                    {
                        "base_order": b_i,
                        "base_index": int(idx),
                        "base_cl_label": base_cl_label,
                        "base_cd_train": base_cd,
                        "shift": None if shift is None else shift_val,
                        "shift_numeric": shift_val,  # None も 0 扱いで色付けできるように
                        "cl_target": cl_target,
                        "sample_id": int(n_i),
                        "cl_out": cl_out,
                        "cd_out": cd_out,
                        "cl_err": (cl_out - cl_target) if not np.isnan(cl_out) else np.nan,
                        "cl_ape_pct": (
                            abs((cl_out - cl_target) / max(abs(cl_target), 1e-12)) * 100.0
                            if not np.isnan(cl_out)
                            else np.nan
                        ),
                    }
                )

        # save generated coords
        gen_path = os.path.join(base_subdir, "generated.npy")
        np.save(gen_path, gen_all)

        # save metrics csv
        import csv

        csv_path = os.path.join(base_subdir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            for r in metrics_rows:
                writer.writerow(r)

        # per-base meta
        base_meta = {
            "base_order": b_i,
            "base_index": int(idx),
            "base_cl_label": base_cl_label,
            "base_cd_train": base_cd,
            "outputs": {
                "generated_npy": gen_path,
                "metrics_csv": csv_path,
            },
        }
        with open(os.path.join(base_subdir, "meta.json"), "w") as f:
            json.dump(base_meta, f, indent=2, ensure_ascii=False)

        # ---------------------------
        # Plot 1) row plot
        # ---------------------------
        cols = 1 + S
        fig_w = max(16, 3.2 * cols)
        fig, axes = plt.subplots(1, cols, figsize=(fig_w, 4.5))

        # base panel
        axes[0]
        base_title = f"BASE\nCL(label): {base_cl_label:.3f}\nCD(train): {base_cd:.4f}"
        plot_base_with_mask(coord_denorm, mask_np, axes[0], title=base_title)

        # each shift: use first sample (n=0) for the row visualization
        # (n_per_shift>1 の場合は、代表として 0 を表示)
        metrics_by_shift = {}
        for r in metrics_rows:
            if r["sample_id"] == 0:
                metrics_by_shift[r["shift_numeric"]] = r

        for s_i, shift in enumerate(shifts):
            shift_val = 0.0 if shift is None else float(shift)
            rep = gen_all[s_i, 0]  # (2,L)
            r0 = metrics_by_shift.get(shift_val, None)
            if r0 is None:
                title = f"SHIFT {shift}\n(no-metrics)"
            else:
                title = (
                    f"SHIFT {shift}\n"
                    f"CL_tgt: {r0['cl_target']:.3f}\n"
                    f"CL_out: {r0['cl_out']:.3f}\n"
                    f"CD_out: {r0['cd_out']:.4f}"
                )
            plot_inpaint(coord_denorm, rep, mask_np, axes[1 + s_i], title=title)

        plt.tight_layout()
        row_png = os.path.join(base_subdir, "row.png")
        fig.savefig(row_png, dpi=250)
        plt.close(fig)

        # ---------------------------
        # Plot 2) overlay plot (all shifts)
        # ---------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(coord_denorm[0], coord_denorm[1], color="k", linewidth=2.0, label="base")

        cmap = plt.get_cmap("viridis")
        for s_i, shift in enumerate(shifts):
            rep = gen_all[s_i, 0]
            c = cmap(s_i / max(S - 1, 1))
            lab = "shift=None" if shifts[s_i] is None else f"shift={float(shifts[s_i]):.1f}"
            ax.plot(rep[0], rep[1], linewidth=1.6, color=c, label=lab)

        ax.set_aspect("equal", "box")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.25, 0.4)
        ax.grid(True)
        ax.set_title("Overlay (base + generated for each CL_SHIFT)")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        overlay_png = os.path.join(base_subdir, "overlay.png")
        fig.savefig(overlay_png, dpi=250)
        plt.close(fig)

        # ---------------------------
        # Plot 3) CL–CD scatter (train + generated colored by shift)
        # ---------------------------
        # gather generated points (all N, all shifts)
        cl_gen = np.array([r["cl_out"] for r in metrics_rows], dtype=float)
        cd_gen = np.array([r["cd_out"] for r in metrics_rows], dtype=float)
        shift_gen = np.array([r["shift_numeric"] for r in metrics_rows], dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # train scatter
        ax.scatter(
            train_cls_xfoil[plot_train_idx],
            train_cds_xfoil[plot_train_idx],
            s=10,
            alpha=0.25,
            label="train",
        )

        # generated scatter (color by shift)
        valid_gen = (~np.isnan(cl_gen)) & (~np.isnan(cd_gen))
        sc = ax.scatter(
            cl_gen[valid_gen],
            cd_gen[valid_gen],
            c=shift_gen[valid_gen],
            s=35,
        )
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("CL_SHIFT (None treated as 0.0)")

        # base point
        if not (np.isnan(base_cl_label) or np.isnan(base_cd)):
            ax.scatter(
                [base_cl_label],
                [base_cd],
                s=180,
                marker="*",
                edgecolors="k",
                linewidths=1.0,
                label="base",
            )

        ax.set_xlabel("CL")
        ax.set_ylabel("CD")
        ax.set_title("CL–CD Scatter (train + generated)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        scatter_png = os.path.join(base_subdir, "cl_cd_scatter.png")
        fig.savefig(scatter_png, dpi=250)
        plt.close(fig)

        # update base meta with plots
        base_meta["outputs"].update(
            {
                "row_png": row_png,
                "overlay_png": overlay_png,
                "cl_cd_scatter_png": scatter_png,
            }
        )
        with open(os.path.join(base_subdir, "meta.json"), "w") as f:
            json.dump(base_meta, f, indent=2, ensure_ascii=False)

        print(f"[DONE base {b_i:02d}] {base_subdir}")

    print("\n[ALL DONE]")
    print(f"Outputs in: {run_dir}")


if __name__ == "__main__":
    main()

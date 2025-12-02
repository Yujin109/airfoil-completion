#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
400_repeat_config_analysis.py  (with base train CL/CD and overlay point)

- RePaint で n 回生成 → 分布/統計/生成 .npy 保存
- 学習データ CL/CD を angle ごとにキャッシュ（npz）
- CL–CD 散布図：学習(サブセット)・生成に加え、基準サンプル(1点)を強調表示
- 基準サンプルの CL/CD を stats と meta に保存
- 保存先: results/{EXECUTION_NAME}/analysis/
"""

import argparse
import datetime
import json
import os
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import config
from config import (
    DIFFUSION_PARAMS,
    EXECUTION_NAME,
    GUIDANCE_SCALE,
    MODEL_NAME,
    WEIGHT_PATH,
)
from src.data import AirfoilDataset
from src.diffusion_blend_mask import RePaintDiffuser
from src.models.model_registry import MODEL_REGISTRY
from src.xfoil_utils import get_xf_instance  # 共有インスタンスで高速化


# ---------------------------
# XFoil: 一回で CL/CD 取得
# ---------------------------
def get_cl_cd(coord: np.ndarray, angle: float = 5.0) -> Tuple[float, float]:
    try:
        import numpy as _np
        from xfoil.model import Airfoil as _Airfoil

        xf = get_xf_instance()
        xf.Re = 3e6
        xf.max_iter = 100
        x, y = coord.reshape(2, -1)
        xf.airfoil = _Airfoil(x=x, y=y)
        c = xf.a(angle)
        if c is None or len(c) < 2:
            return (np.nan, np.nan)
        return (float(_np.round(c[0], 10)), float(_np.round(c[1], 10)))
    except Exception:
        return (np.nan, np.nan)


# ---------------------------
# マスク生成（305_* と同仕様）
# ---------------------------
def mask_manual(pattern, coords, missing_ratio, blend_ratio, blend_type, blend_value) -> np.ndarray:
    if coords is None:
        raise ValueError("coords must be provided")
    if pattern not in {"head", "tail", "middle", "top", "bottom"}:
        raise ValueError(f"Unknown pattern: {pattern}")

    xs = coords[0] if coords.ndim == 2 else coords
    x_min, x_max = xs.min(), xs.max()
    range_x = x_max - x_min
    n = xs.shape[0]
    half_n = n // 2

    if pattern not in {"top", "bottom"}:
        if not (0.0 <= missing_ratio <= 1.0 and 0.0 <= blend_ratio <= 1.0 and (missing_ratio + blend_ratio) <= 1.0):
            raise ValueError("missing_ratio/blend_ratio invalid")

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
                mask[idx] = (xs[idx] - blend_start) / (blend_end - blend_start)

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
                mask[idx] = 1.0 - (xs[idx] - blend_start) / (blend_end - blend_start)

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
                mask[left_idx] = mask[right_idx] = blend_value
            else:
                mask[left_idx] = (xs[left_idx] - bls) / (ble - bls)
                mask[right_idx] = (xs[right_idx] - brs) / (bre - brs)

    elif pattern == "top":
        miss_idx = np.arange(half_n, n)
        known_idx = np.arange(0, half_n)
        mask[miss_idx] = 0.0
        mask[known_idx] = 1.0
        bw = blend_ratio * range_x / 2
        if bw > 0:
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (xs[idxs] - bs) / (be - bs)
            bs, be = x_max - bw, x_max
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (be - xs[idxs]) / (be - bs)

    elif pattern == "bottom":
        miss_idx = np.arange(0, half_n)
        known_idx = np.arange(half_n, n)
        mask[miss_idx] = 0.0
        mask[known_idx] = 1.0
        bw = blend_ratio * range_x / 2
        if bw > 0:
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (xs[idxs] - bs) / (be - bs)
            bs, be = x_max - bw, x_max
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (be - xs[idxs]) / (be - bs)

    return mask


# ---------------------------
# 図ユーティリティ
# ---------------------------
def save_hist(data: np.ndarray, title: str, xlabel: str, out_path: str, vline: float = None, bins: int = 30):
    arr = data[~np.isnan(data)]
    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins)
    if vline is not None:
        plt.axvline(vline, color="red", linestyle="--", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cl_cd_from_paths(
    paths: List[str],
    labels: List[str],
    out_path: str,
    angle_deg: float,
    base_point: Optional[Tuple[float, float]] = None,
    base_point_label: str = "train(base)",
    target_cl_line: Optional[float] = None,  # ADDED: オレンジ点線
    cl_plus_line: Optional[float] = None,  # ADDED: 赤点線
):
    """
    paths の各要素:
      - .npz: 'cls','cds' を読み込んで散布（XFoil無し）
      - .npy: 形状から XFoil で CL/CD を計算（遅い）
    base_point を与えると、その1点を大きいマーカーで重ね描き。
    """
    assert len(paths) == len(labels)
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 6))
    for i, (p, lab) in enumerate(zip(paths, labels)):
        if p.endswith(".npz"):
            cache = np.load(p)
            cls = cache["cls"]
            cds = cache["cds"]
        else:
            arr = np.load(p, allow_pickle=True)
            if arr.ndim == 2:
                arr = arr.reshape(1, *arr.shape)
            cls, cds = [], []
            for k in tqdm(range(arr.shape[0]), desc=f"Compute CL/CD for {lab}"):
                cl, cd = get_cl_cd(arr[k], angle=angle_deg)
                if not (np.isnan(cl) or np.isnan(cd)):
                    cls.append(cl)
                    cds.append(cd)
            cls = np.array(cls)
            cds = np.array(cds)
        if len(cls) > 0:
            plt.scatter(cls, cds, label=lab, alpha=0.6, c=[cmap(i % 10)])

    # ADDED: 縦線（指定CL: オレンジ点線, 指定CL+シフト: 赤点線）
    if target_cl_line is not None:
        plt.axvline(target_cl_line, color="orange", linestyle="--", linewidth=2, label="base CL")
    if cl_plus_line is not None:
        plt.axvline(cl_plus_line, color="red", linestyle="--", linewidth=2, label="target CL")

    # 基準点の重ね描き
    if base_point is not None and not (np.isnan(base_point[0]) or np.isnan(base_point[1])):
        plt.scatter(
            [base_point[0]],
            [base_point[1]],
            s=150,
            marker="*",
            edgecolors="k",
            linewidths=1.0,
            label=base_point_label,
            c="blue",
        )  # CHANGED

    plt.xlabel("CL")
    plt.ylabel("CD")
    plt.title("CL–CD Scatter")
    plt.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# 学習 CL/CD キャッシュ
# ---------------------------
def get_or_build_train_clcd_cache(dataset: AirfoilDataset, analysis_dir: str, angle: float) -> str:
    cache_path = os.path.join(analysis_dir, f"train_clcd_cache_angle{angle:.3f}.npz")
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


def make_subset_clcd_cache(full_cache_npz: str, idx_sel: np.ndarray, out_npz: str):
    data = np.load(full_cache_npz)
    cls_full, cds_full = data["cls"], data["cds"]
    np.savez_compressed(out_npz, cls=cls_full[idx_sel], cds=cds_full[idx_sel])


# ---------------------------
# メイン
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_cl", type=float, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--mask_type", type=str, default="top", choices=["head", "tail", "middle", "top", "bottom"])
    parser.add_argument("--missing_points_ratio", type=float, default=0.5)
    parser.add_argument("--blend_ratio", type=float, default=0.1)
    parser.add_argument("--blend_type", type=str, default="linear", choices=["linear", "constant"])
    parser.add_argument("--blend_value", type=float, default=1.0)
    parser.add_argument("--num_resampling", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=10)
    parser.add_argument("--cl_shift", type=float, default=0.1)
    parser.add_argument("--angle", type=float, default=5.0)
    parser.add_argument("--train_fraction", type=float, default=1.0, help="0-1 (既定=1.0=全学習データ)")
    parser.add_argument("--scatter_labels", type=str, default="train(sample),generated")
    parser.add_argument("--base_point_label", type=str, default="train(base)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ANALYSIS_DIR = os.path.join("results", EXECUTION_NAME, "analysis")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%y%m%d%H%M%S")  # 先頭に timestamp
    tag = (
        f"{date_str}_N{args.n}_R{args.num_resampling}_J{args.jump_length}"
        f"_M:{args.mask_type}_MPR:{args.missing_points_ratio}"
        f"_BR:{args.blend_ratio}_{args.blend_type}_{args.blend_value}"
        f"_SHIFT:{args.cl_shift}_CLIN:{args.target_cl}"
    )
    # ADDED
    RUN_DIR = os.path.join(ANALYSIS_DIR, tag)
    os.makedirs(RUN_DIR, exist_ok=True)

    # データ/モデル
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
    )

    # 指定CLに最も近い学習サンプル
    cls_denorm_all = dataset.denormalize_cl(dataset.cls_tensor).numpy().flatten()
    idx_closest = int(np.argmin(np.abs(cls_denorm_all - args.target_cl)))
    cl_in = float(cls_denorm_all[idx_closest])
    cl_target = cl_in + args.cl_shift

    coord_std = dataset.coords_tensor[idx_closest : idx_closest + 1].to(device)
    cls_std = dataset.cls_tensor[idx_closest : idx_closest + 1].to(device)

    # 基準サンプルの CL/CD（キャッシュが角度一致ならそれを利用）
    full_cache_npz = get_or_build_train_clcd_cache(dataset, ANALYSIS_DIR, args.angle)
    base_cl = base_cd = np.nan
    try:
        cache = np.load(full_cache_npz)
        # キャッシュが全学習順序に対応している前提
        base_cl = float(cache["cls"][idx_closest])
        base_cd = float(cache["cds"][idx_closest])
    except Exception:
        coord_denorm_single_tmp = dataset.denormalize_coord(coord_std).cpu().numpy()[0]
        base_cl, base_cd = get_cl_cd(coord_denorm_single_tmp, angle=args.angle)

    # マスク（標準化前）
    coord_denorm_single = dataset.denormalize_coord(coord_std).cpu().numpy()[0]
    mask_np = mask_manual(
        args.mask_type,
        coord_denorm_single,
        args.missing_points_ratio,
        args.blend_ratio,
        args.blend_type,
        args.blend_value,
    ).astype(np.float32)
    mask = torch.from_numpy(mask_np).unsqueeze(0).to(device).float()

    # バッチ
    coord_std_batch = coord_std.repeat(args.n, 1, 1)
    cls_std_batch = cls_std.repeat(args.n, 1)
    mask_batch = mask.repeat(args.n, 1)
    if args.cl_shift:
        cls_std_batch = cls_std_batch + (args.cl_shift / dataset.cl_std)

    # 乱数
    if args.seed is not None and args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # RePaint（内部 tqdm あり）
    with torch.no_grad():
        repainted_std = diffuser.repaint(model, coord_std_batch, mask_batch, cls_std_batch)

    # 生成 .npy（標準化前）
    repainted_denorm = dataset.denormalize_coord(repainted_std).cpu().numpy()
    gen_npy_path = os.path.join(RUN_DIR, f"{tag}_generated.npy")
    np.save(gen_npy_path, repainted_denorm)

    # 生成 CL（進捗）
    cl_out = np.empty(repainted_denorm.shape[0], dtype=float)
    for i in tqdm(range(repainted_denorm.shape[0]), desc="Compute CL for generated"):
        cl, _ = get_cl_cd(repainted_denorm[i], angle=args.angle)
        cl_out[i] = cl
    err = cl_out - cl_target
    valid_mask = ~np.isnan(cl_out)
    convergence_ratio = float(valid_mask.mean())
    cl_mean = float(np.nanmean(cl_out))
    cl_std = float(np.nanstd(cl_out))
    err_mean = float(np.nanmean(err))
    err_std = float(np.nanstd(err))
    mape = float(np.nanmean(np.abs((cl_out - cl_target) / max(abs(cl_target), 1e-12)) * 100.0))

    # 分布図（赤線）
    cl_hist_path = os.path.join(RUN_DIR, f"{tag}_hist_cl.png")
    err_hist_path = os.path.join(RUN_DIR, f"{tag}_hist_err.png")
    save_hist(cl_out, f"CL distribution (target={cl_target:.3f})", "CL", cl_hist_path, vline=cl_target)
    save_hist(err, "Error distribution (CL_out - CL_target)", "Error", err_hist_path, vline=0.0)

    # 統計（基準サンプルの CL/CD を追記）
    stats_path = os.path.join(RUN_DIR, f"{tag}_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"EXECUTION_NAME: {EXECUTION_NAME}\n")
        f.write(f"MODEL_NAME: {MODEL_NAME}\n")
        f.write(f"WEIGHT_PATH: {WEIGHT_PATH}\n")
        f.write(f"n: {args.n}\n")
        f.write(f"mask_type: {args.mask_type}\n")
        f.write(f"missing_points_ratio: {args.missing_points_ratio}\n")
        f.write(f"blend_ratio: {args.blend_ratio}\n")
        f.write(f"blend_type: {args.blend_type}\n")
        f.write(f"blend_value: {args.blend_value}\n")
        f.write(f"num_resampling: {args.num_resampling}\n")
        f.write(f"jump_length: {args.jump_length}\n")
        f.write(f"angle(deg): {args.angle}\n")
        f.write(f"train_fraction: {args.train_fraction}\n")
        f.write(f"picked_train_index: {idx_closest}\n")
        f.write(f"cl_in (picked): {cl_in:.6f}\n")
        f.write(f"base_cl: {base_cl:.10f}\n")
        f.write(f"base_cd: {base_cd:.10f}\n")
        f.write(f"cl_shift: {args.cl_shift}\n")
        f.write(f"cl_target = cl_in + cl_shift: {cl_target:.6f}\n\n")
        f.write(f"convergence_ratio: {convergence_ratio:.4f}\n")
        f.write(f"cl_mean: {cl_mean:.6f}\n")
        f.write(f"cl_std: {cl_std:.6f}\n")
        f.write(f"err_mean: {err_mean:.6f}\n")
        f.write(f"err_std: {err_std:.6f}\n")
        f.write(f"MAPE(%): {mape:.3f}\n")

    # 学習 CL/CD キャッシュ → サブセット npz → 散布
    # （full_cache_npz は上で作成済み）
    total_N = dataset.coords_tensor.shape[0]
    use_N = int(round(total_N * max(0.0, min(args.train_fraction, 1.0))))
    if use_N < total_N:
        rng = np.random.default_rng(args.seed if (args.seed is not None and args.seed >= 0) else None)
        idx_sel = np.sort(rng.choice(total_N, size=use_N, replace=False))
    else:
        idx_sel = np.arange(total_N)

    train_subset_npz = os.path.join(RUN_DIR, f"{tag}_train_clcd_subset.npz")
    make_subset_clcd_cache(full_cache_npz, idx_sel, train_subset_npz)

    train_coords_denorm = dataset.denormalize_coord(dataset.coords_tensor[idx_sel]).cpu().numpy()
    train_coords_npy = os.path.join(RUN_DIR, f"{tag}_train_coords_for_scatter.npy")
    np.save(train_coords_npy, train_coords_denorm)

    labels = [s.strip() for s in args.scatter_labels.split(",")]
    if len(labels) < 2:
        labels = labels + ["generated"]
    elif len(labels) > 2:
        labels = labels[:2]

    clcd_png_path = os.path.join(RUN_DIR, f"{tag}_cl_cd_scatter.png")
    plot_cl_cd_from_paths(
        paths=[train_subset_npz, gen_npy_path],
        labels=labels,
        out_path=clcd_png_path,
        angle_deg=args.angle,
        base_point=(base_cl, base_cd),
        base_point_label=args.base_point_label,
        target_cl_line=args.target_cl,  # ADDED: オレンジ点線
        cl_plus_line=(args.target_cl + args.cl_shift),  # ADDED: 赤実線
    )

    # メタ
    meta = {
        "date": date_str,
        "execution_name": EXECUTION_NAME,
        "model_name": MODEL_NAME,
        "weight_path": WEIGHT_PATH,
        "tag": tag,
        "args": vars(args),
        "picked_train_index": int(idx_closest),
        "cl_in": cl_in,
        "cl_target": cl_target,
        "base_cl": base_cl,
        "base_cd": base_cd,
        "outputs": {
            "generated_npy": gen_npy_path,
            "train_coords_npy": train_coords_npy,
            "train_subset_clcd_npz": train_subset_npz,
            "hist_cl_png": cl_hist_path,
            "hist_err_png": err_hist_path,
            "stats_txt": stats_path,
            "cl_cd_png": clcd_png_path,
        },
    }
    meta_path = os.path.join(RUN_DIR, f"{tag}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[Analysis Saved]")
    for k, v in meta["outputs"].items():
        print(f"{k}: {v}")
    print(f"meta_json: {meta_path}")


if __name__ == "__main__":
    main()

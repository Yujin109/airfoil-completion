#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
400_repeat_config_analysis.py

1つのconfigで、同一CLの学習サンプルから n 回 RePaint 生成し、
- 出力 CL の分布
- 誤差 (cl_out - cl_target) の分布
- 統計（平均, 標準偏差, 収束率, MAPE など）
- CL–CD 散布図（.npy を複数受け取って色分け）

を保存する分析スクリプト。

保存先: results/{MODEL_NAME}/analysis/
ファイル名: 実験条件（mask, ratios, R, J, shift, n, cl_inなど）を含む。

使い方（例）:
    python 400_repeat_config_analysis.py \
        --target_cl 0.682 \
        --n 100 \
        --mask_type top \
        --missing_points_ratio 0.5 \
        --blend_ratio 0.1 \
        --blend_type linear \
        --blend_value 1.0 \
        --num_resampling 10 \
        --jump_length 10 \
        --cl_shift 0.1

※ config.py の MODEL_NAME / WEIGHT_PATH / DIFFUSION_PARAMS 等を使用します。
"""

import os
import sys
import math
import json
import argparse
import datetime
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 既存プロジェクトの依存
import config
from config import (
    MODEL_NAME,
    WEIGHT_PATH,
    DIFFUSION_PARAMS,
    GUIDANCE_SCALE,
)
from src.data import AirfoilDataset
from src.diffusion_blend_mask import RePaintDiffuser
from src.models.model_registry import MODEL_REGISTRY
from src.xfoil_utils import get_cl, get_cd

# ---------------------------
# マスク生成（標準化前座標を入力; 305_* と同仕様）
# ---------------------------
def mask_manual(
    pattern: str,
    coords: np.ndarray,
    missing_ratio: float,
    blend_ratio: float,
    blend_type: str,
    blend_value: float,
) -> np.ndarray:
    """
    pattern: "head", "tail", "middle", "top", "bottom"
    coords: shape (2, L) or (L,) の x座標含む配列（標準化前）
    戻り値: 長さ L の float 配列 (1=known, 0=missing, (0,1)=blend)
    """
    if coords is None:
        raise ValueError("coords must be provided")
    if pattern not in {"head", "tail", "middle", "top", "bottom"}:
        raise ValueError(f"Unknown pattern: {pattern}")

    xs = coords[0] if coords.ndim == 2 else coords
    x_min, x_max = xs.min(), xs.max()
    range_x = x_max - x_min
    n = xs.shape[0]
    half_n = n // 2

    # 非 top / bottom の場合は比率チェック
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

        blend_idx = (xs > blend_start) & (xs <= blend_end)
        if blend_ratio > 0:
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
        if blend_ratio > 0:
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

        mask[xs <= blend_left_start] = 1.0
        mask[xs >= blend_right_end] = 1.0
        mask[(xs > miss_start) & (xs < miss_end)] = 0.0

        left_idx = (xs > blend_left_start) & (xs <= blend_left_end)
        right_idx = (xs >= blend_right_start) & (xs < blend_right_end)

        if blend_ratio > 0:
            if blend_type == "constant":
                mask[left_idx] = blend_value
                mask[right_idx] = blend_value
            elif blend_type == "linear":
                mask[left_idx] = (xs[left_idx] - blend_left_start) / (blend_left_end - blend_left_start)
                mask[right_idx] = (xs[right_idx] - blend_right_start) / (blend_right_end - blend_right_start)
            else:
                raise ValueError(f"Unknown blend_type: {blend_type}")

    elif pattern == "top":
        miss_idx = np.arange(half_n, n)
        known_idx = np.arange(0, half_n)
        mask[miss_idx] = 0.0
        mask[known_idx] = 1.0

        bw = blend_ratio * range_x / 2
        if bw > 0:
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            if blend_type == "constant":
                mask[idxs] = blend_value
            elif blend_type == "linear":
                mask[idxs] = (xs[idxs] - bs) / (be - bs)
            else:
                raise ValueError(f"Unknown blend_type: {blend_type}")

            bs, be = x_max - bw, x_max
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            if blend_type == "constant":
                mask[idxs] = blend_value
            elif blend_type == "linear":
                mask[idxs] = (be - xs[idxs]) / (be - bs)
            else:
                raise ValueError(f"Unknown blend_type: {blend_type}")

    elif pattern == "bottom":
        miss_idx = np.arange(0, half_n)
        known_idx = np.arange(half_n, n)
        mask[miss_idx] = 0.0
        mask[known_idx] = 1.0

        bw = blend_ratio * range_x / 2
        if bw > 0:
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            if blend_type == "constant":
                mask[idxs] = blend_value
            elif blend_type == "linear":
                mask[idxs] = (xs[idxs] - bs) / (be - bs)
            else:
                raise ValueError(f"Unknown blend_type: {blend_type}")

            bs, be = x_max - bw, x_max
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            if blend_type == "constant":
                mask[idxs] = blend_value
            elif blend_type == "linear":
                mask[idxs] = (be - xs[idxs]) / (be - bs)
            else:
                raise ValueError(f"Unknown blend_type: {blend_type}")

    return mask


# ---------------------------
# 図ユーティリティ
# ---------------------------
def save_hist(data: np.ndarray, title: str, xlabel: str, out_path: str, bins: int = 30):
    plt.figure(figsize=(8, 5))
    plt.hist(data[~np.isnan(data)], bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cl_cd_from_npy_list(npy_paths: List[str], labels: List[str], out_path: str, angle_deg: float = 5.0):
    """
    各 .npy（shape (N,2,L) or (2,L) を許容）について cl, cd を計算し、色分け散布図を描画。
    XFoil 計算なので枚数・点数によっては時間がかかる点に注意。
    """
    assert len(npy_paths) == len(labels)
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(8, 6))
    for i, (p, lab) in enumerate(zip(npy_paths, labels)):
        arr = np.load(p, allow_pickle=True)
        # arr shape: (N,2,L) or (2,L)
        if arr.ndim == 2:
            arr = arr.reshape(1, *arr.shape)
        cls, cds = [], []
        for k in range(arr.shape[0]):
            cl = get_cl(arr[k], angle=angle_deg)
            cd = get_cd(arr[k], angle=angle_deg)
            if not (np.isnan(cl) or np.isnan(cd)):
                cls.append(cl)
                cds.append(cd)
        if len(cls) > 0:
            plt.scatter(cls, cds, label=lab, alpha=0.6, c=[cmap(i % 10)])

    plt.xlabel("CL (α={}°)".format(angle_deg))
    plt.ylabel("CD (α={}°)".format(angle_deg))
    plt.title("CL–CD Scatter")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# メイン処理
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_cl", type=float, required=True, help="学習データで狙うCL（例: 0.682）")
    parser.add_argument("--n", type=int, default=100, help="同一サンプルからの生成回数")
    parser.add_argument("--mask_type", type=str, default="top", choices=["head","tail","middle","top","bottom"])
    parser.add_argument("--missing_points_ratio", type=float, default=0.5)
    parser.add_argument("--blend_ratio", type=float, default=0.1)
    parser.add_argument("--blend_type", type=str, default="linear", choices=["linear","constant"])
    parser.add_argument("--blend_value", type=float, default=1.0)
    parser.add_argument("--num_resampling", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=10)
    parser.add_argument("--cl_shift", type=float, default=0.1)
    parser.add_argument("--angle", type=float, default=5.0, help="XFoil計算の迎角[deg]")
    parser.add_argument("--dataset_sample_for_scatter", type=int, default=500, help="学習データからCL-CD散布用に抽出する件数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 解析出力ディレクトリ（モデル名ベース）
    ANALYSIS_DIR = os.path.join("results", MODEL_NAME, "analysis")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # タイムスタンプ & 実験名（ファイル名共通プレフィックス）
    date_str = datetime.datetime.now().strftime("%y%m%d%H%M")
    tag = (
        f"{date_str}_N{args.n}_R{args.num_resampling}_J{args.jump_length}"
        f"_M:{args.mask_type}_MPR:{args.missing_points_ratio}"
        f"_BR:{args.blend_ratio}_{args.blend_type}_{args.blend_value}"
        f"_SHIFT:{args.cl_shift}_CLIN:{args.target_cl}"
    )

    # 1) データとモデルのロード
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

    # 2) 指定CLに最も近い学習サンプルを1つ選択
    cls_denorm_all = dataset.denormalize_cl(dataset.cls_tensor).numpy().flatten()
    idx_closest = int(np.argmin(np.abs(cls_denorm_all - args.target_cl)))
    cl_in = float(cls_denorm_all[idx_closest])  # 実際に使う入力CL
    cl_target = cl_in + args.cl_shift

    coord_std = dataset.coords_tensor[idx_closest:idx_closest+1].to(device)  # shape (1,2,L) 標準化済
    cls_std = dataset.cls_tensor[idx_closest:idx_closest+1].to(device)       # shape (1,1)

    # 3) マスク生成（標準化前の座標で）
    coord_denorm_single = dataset.denormalize_coord(coord_std).cpu().numpy()[0]  # (2,L)
    mask_np = mask_manual(
        pattern=args.mask_type,
        coords=coord_denorm_single,
        missing_ratio=args.missing_points_ratio,
        blend_ratio=args.blend_ratio,
        blend_type=args.blend_type,
        blend_value=args.blend_value,
    ).astype(np.float32)
    mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)  # (1,L)
    mask = mask.float()

    # 4) n 回分のバッチを構成（同一サンプルを n 回複製）
    coord_std_batch = coord_std.repeat(args.n, 1, 1)  # (n,2,L)
    cls_std_batch = cls_std.repeat(args.n, 1)         # (n,1)
    mask_batch = mask.repeat(args.n, 1)               # (n,L)

    # CL shift を正規化空間に反映
    if args.cl_shift is not None and args.cl_shift != 0.0:
        cls_std_batch = cls_std_batch + (args.cl_shift / dataset.cl_std)

    # 5) RePaint 実行
    # 乱数の再現性（必要なら固定 / デフォルトはばらつきを出すため固定しない）
    if args.seed is not None and args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # 上記で初期化した後、各サンプルごとに追加の乱数を使うため敢えて固定しすぎない

    with torch.no_grad():
        repainted_std = diffuser.repaint(model, coord_std_batch, mask_batch, cls_std_batch)  # (n,2,L)

    # 6) 標準化前へ戻して .npy 保存
    repainted_denorm = dataset.denormalize_coord(repainted_std).cpu().numpy()  # (n,2,L)
    gen_npy_path = os.path.join(ANALYSIS_DIR, f"generated_{tag}.npy")
    np.save(gen_npy_path, repainted_denorm)

    # 7) 出力CL, 誤差、収束率などの統計
    cl_out_list = []
    for i in range(repainted_denorm.shape[0]):
        cl_val = get_cl(repainted_denorm[i], angle=args.angle)
        cl_out_list.append(cl_val)
    cl_out = np.array(cl_out_list, dtype=float)
    err = cl_out - cl_target  # 誤差
    # 収束率（NaNでない割合）
    valid_mask = ~np.isnan(cl_out)
    convergence_ratio = float(valid_mask.mean())
    # 統計
    cl_mean = float(np.nanmean(cl_out))
    cl_std = float(np.nanstd(cl_out))
    err_mean = float(np.nanmean(err))
    err_std = float(np.nanstd(err))
    # MAPE（相対誤差％）
    mape = float(np.nanmean(np.abs((cl_out - cl_target) / cl_target) * 100.0))

    # 8) 分布図保存
    cl_hist_path = os.path.join(ANALYSIS_DIR, f"hist_cl_{tag}.png")
    err_hist_path = os.path.join(ANALYSIS_DIR, f"hist_err_{tag}.png")
    save_hist(cl_out, f"CL distribution (target={cl_target:.3f})", "CL", cl_hist_path)
    save_hist(err, "Error distribution (CL_out - CL_target)", "Error", err_hist_path)

    # 9) 統計テキスト保存
    stats_path = os.path.join(ANALYSIS_DIR, f"stats_{tag}.txt")
    with open(stats_path, "w") as f:
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
        f.write(f"cl_in (picked): {cl_in:.6f}\n")
        f.write(f"cl_shift: {args.cl_shift}\n")
        f.write(f"cl_target = cl_in + cl_shift: {cl_target:.6f}\n")
        f.write(f"angle(deg): {args.angle}\n")
        f.write("\n")
        f.write(f"convergence_ratio: {convergence_ratio:.4f}\n")
        f.write(f"cl_mean: {cl_mean:.6f}\n")
        f.write(f"cl_std: {cl_std:.6f}\n")
        f.write(f"err_mean: {err_mean:.6f}\n")
        f.write(f"err_std: {err_std:.6f}\n")
        f.write(f"MAPE(%): {mape:.3f}\n")

    # 10) CL–CD 散布図
    # 学習データから散布用にランダムサンプル抽出し .npy で保存
    rng = np.random.default_rng(args.seed if args.seed is not None and args.seed >= 0 else None)
    total_N = dataset.coords_tensor.shape[0]
    take = min(args.dataset_sample_for_scatter, total_N)
    sample_idx = rng.choice(total_N, size=take, replace=False)
    train_coords_denorm = dataset.denormalize_coord(dataset.coords_tensor[sample_idx]).cpu().numpy()
    train_npy_path = os.path.join(ANALYSIS_DIR, f"train_sample_for_scatter_{tag}.npy")
    np.save(train_npy_path, train_coords_denorm)

    clcd_png_path = os.path.join(ANALYSIS_DIR, f"cl_cd_scatter_{tag}.png")
    plot_cl_cd_from_npy_list(
        npy_paths=[train_npy_path, gen_npy_path],
        labels=["train(sample)", "generated"],
        out_path=clcd_png_path,
        angle_deg=args.angle,
    )

    # 11) 追記: JSONメタも保存（将来のPCA/t-SNE用メタ）
    meta = {
        "date": date_str,
        "model_name": MODEL_NAME,
        "weight_path": WEIGHT_PATH,
        "tag": tag,
        "args": vars(args),
        "picked_train_index": int(idx_closest),
        "cl_in": cl_in,
        "cl_target": cl_target,
        "outputs": {
            "generated_npy": gen_npy_path,
            "train_sample_npy": train_npy_path,
            "hist_cl_png": cl_hist_path,
            "hist_err_png": err_hist_path,
            "stats_txt": stats_path,
            "cl_cd_png": clcd_png_path,
        },
    }
    with open(os.path.join(ANALYSIS_DIR, f"meta_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[Analysis Saved]")
    for k, v in meta["outputs"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

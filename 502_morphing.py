#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
500_morphing.py

要件（更新版）:
- 学習データから CL≈target_cl 付近の候補を取り、指定 CD ターゲットに最も近い 7 翼型を選ぶ（欠損前形状）
- 各翼型に対し CL_SHIFT を変えながら RePaint inpainting を実行（全shift×n_per_shift を一括バッチ生成）
- XFoil(CL/CD) が収束しない（NaN）生成が残る場合、"生成自体"をやり直す（XFoilの再計算ではない）
- 収束条件を切替可能:
    - all: 全サンプルが収束するまで（従来）
    - per_shift_min: 各shiftにつき少なくとも min_valid_per_shift 個 収束すればOK（要望）
- 可視化:
  1) 横一列（base + shift別 generated）
     - BASE は全て実線
     - 凡例は known / blend / generated の 3 つのみ
     - 各shiftパネルの代表表示は「最初に収束したサンプル」を優先（無ければ0番）
  2) 全重ね（shift別に色）
     - 代表表示は同上
  3) CL–CD散布図（train + generated、色=shift、figsize=(8,6)）

保存先:
results/{EXECUTION_NAME}/morphing/{tag}/baseXX_idx.../
  ├─ row.png
  ├─ overlay.png
  ├─ cl_cd_scatter.png
  ├─ generated.npy
  ├─ metrics.csv
  └─ meta.json

キャッシュ:
results/{EXECUTION_NAME}/morphing/_cache/train_clcd_cache_angleX.npz
"""

import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

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
    out: List[Optional[float]] = []
    for tok in [x.strip() for x in s.split(",") if x.strip() != ""]:
        if tok.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(float(tok))
    return out


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def fmt(x: float, spec: str) -> str:
    try:
        if np.isnan(x):
            return "nan"
        return format(float(x), spec)
    except Exception:
        return "nan"


def pick_representative_sample_ids(valid_flat: np.ndarray, S: int, N: int) -> np.ndarray:
    """
    各 shift について「最初に valid になった sample_id」を返す。
    valid が無い shift は 0 を返す。
    """
    rep = np.zeros(S, dtype=int)
    for s_i in range(S):
        v = valid_flat[s_i * N : (s_i + 1) * N]
        idx = np.where(v)[0]
        rep[s_i] = int(idx[0]) if idx.size > 0 else 0
    return rep


# ---------------------------
# XFoil: CL/CD
# ---------------------------
def get_cl_cd(coord: np.ndarray, angle: float = 5.0) -> Tuple[float, float]:
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


def _xfoil_worker(args: Tuple[np.ndarray, float]) -> Tuple[float, float]:
    coord, angle = args
    return get_cl_cd(coord, angle=angle)


def compute_clcd_batch(
    coords_flat: np.ndarray, angle: float, workers: int, pool=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    coords_flat: (M,2,L)
    - pool が与えられればそれを使う（multiprocessing Pool）
    - pool が None の場合:
        workers<=1 -> 逐次
        workers>1  -> この関数内で Pool を作って使う
    """
    M = coords_flat.shape[0]
    cls = np.empty(M, dtype=float)
    cds = np.empty(M, dtype=float)

    if pool is None and workers <= 1:
        for i in range(M):
            cls[i], cds[i] = get_cl_cd(coords_flat[i], angle=angle)
        return cls, cds

    if pool is not None:
        it = pool.imap(_xfoil_worker, ((coords_flat[i], angle) for i in range(M)), chunksize=1)
        for i, (cl, cd) in enumerate(it):
            cls[i] = cl
            cds[i] = cd
        return cls, cds

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as p:
        it = p.imap(_xfoil_worker, ((coords_flat[i], angle) for i in range(M)), chunksize=1)
        for i, (cl, cd) in enumerate(it):
            cls[i] = cl
            cds[i] = cd
    return cls, cds


# ---------------------------
# Mask (304 系)
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
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (xs[idxs] - bs) / max((be - bs), 1e-12)
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
            bs, be = x_min, x_min + bw
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (xs[idxs] - bs) / max((be - bs), 1e-12)
            bs, be = x_max - bw, x_max
            idxs = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            mask[idxs] = blend_value if blend_type == "constant" else (be - xs[idxs]) / max((be - bs), 1e-12)

    return mask.astype(np.float32)


# ---------------------------
# Plot helpers（横並びは凡例3つ）
# ---------------------------
blue_rgb = mcolors.to_rgb("tab:blue")
orange_rgb = mcolors.to_rgb("tab:orange")
mid_rgb = tuple((b + o) / 2 for b, o in zip(blue_rgb, orange_rgb))


def plot_inpaint(orig: np.ndarray, rep: np.ndarray, mask: np.ndarray, ax, title: str = "") -> None:
    m = mask
    known = m == 1.0
    missing = m == 0.0
    blend = (m > 0.0) & (m < 1.0)

    ax.plot(
        np.ma.array(orig[0], mask=~known), np.ma.array(orig[1], mask=~known), label="known", color="tab:blue", lw=2.0
    )
    if blend.any():
        ax.plot(
            np.ma.array(rep[0], mask=~blend), np.ma.array(rep[1], mask=~blend), label="blend", color=mid_rgb, lw=2.0
        )
    ax.plot(
        np.ma.array(rep[0], mask=~missing),
        np.ma.array(rep[1], mask=~missing),
        label="generated",
        color="tab:orange",
        lw=2.0,
    )

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.25, 0.4)
    ax.grid(True)
    ax.tick_params(labelsize=8)


def plot_base_panel(base: np.ndarray, mask: np.ndarray, ax, title: str = "") -> None:
    """
    BASEは全て実線。横並び用のBASEパネルは missing を強調しない。
    """
    m = mask
    known = m == 1.0
    blend = (m > 0.0) & (m < 1.0)

    ax.plot(base[0], base[1], color="k", lw=1.5, ls="-", label="_nolegend_")

    ax.plot(
        np.ma.array(base[0], mask=~known),
        np.ma.array(base[1], mask=~known),
        color="tab:blue",
        lw=2.5,
        ls="-",
        label="known",
    )
    if blend.any():
        ax.plot(
            np.ma.array(base[0], mask=~blend),
            np.ma.array(base[1], mask=~blend),
            color=mid_rgb,
            lw=2.5,
            ls="-",
            label="blend",
        )

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.25, 0.4)
    ax.grid(True)
    ax.tick_params(labelsize=8)


def add_row_legend_known_blend_generated(fig, axes) -> None:
    allowed: Set[str] = {"known", "blend", "generated"}
    handles, labels = [], []
    for ax in np.ravel(axes):
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll in allowed:
                handles.append(hh)
                labels.append(ll)

    order = ["known", "blend", "generated"]
    uniq = {}
    for hh, ll in zip(handles, labels):
        if ll not in uniq:
            uniq[ll] = hh

    final_h, final_l = [], []
    for k in order:
        if k in uniq:
            final_h.append(uniq[k])
            final_l.append(k)

    if final_h:
        fig.legend(final_h, final_l, loc="lower center", ncol=3, fontsize=10)
        plt.subplots_adjust(bottom=0.20)


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


def dump_selected_bases(
    out_csv_path: str,
    base_indices: List[int],
    cls_labels: np.ndarray,  # denorm label CL
    train_cls_xfoil: np.ndarray,  # xfoil CL cache
    train_cds_xfoil: np.ndarray,  # xfoil CD cache
    cd_targets: Optional[List[float]] = None,
    select_mode: str = "targets",
) -> List[Dict]:
    rows = []
    for k, idx in enumerate(base_indices, start=1):
        cl_label = float(cls_labels[idx])
        cl_xfoil = float(train_cls_xfoil[idx]) if not np.isnan(train_cls_xfoil[idx]) else np.nan
        cd_xfoil = float(train_cds_xfoil[idx]) if not np.isnan(train_cds_xfoil[idx]) else np.nan
        cd_tgt = (
            float(cd_targets[k - 1])
            if (select_mode == "targets" and cd_targets is not None and k - 1 < len(cd_targets))
            else np.nan
        )

        rows.append(
            {
                "order": k,
                "index": int(idx),
                "cl_label": cl_label,
                "cl_xfoil": cl_xfoil,
                "cd_xfoil": cd_xfoil,
                "cd_target": cd_tgt,
            }
        )

    # print to stdout (table-like)
    print("\n[Selected base indices and CL/CD]")
    header = f"{'order':>5}  {'index':>6}  {'cl_label':>10}  {'cl_xfoil':>10}  {'cd_xfoil':>10}  {'cd_target':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['order']:>5d}  {r['index']:>6d}  "
            f"{r['cl_label']:>10.6f}  {r['cl_xfoil']:>10.6f}  {r['cd_xfoil']:>10.6f}  "
            f"{(r['cd_target'] if not np.isnan(r['cd_target']) else float('nan')):>10.6f}"
        )

    # save CSV
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[Saved] {out_csv_path}\n")
    return rows


# ---------------------------
# Base selection
# ---------------------------
def select_bases_by_cl_cd(
    cls_labels: np.ndarray,
    cds_train: np.ndarray,
    target_cl: float,
    cl_window_init: float,
    cd_targets: List[float],
) -> List[int]:
    windows = [cl_window_init, cl_window_init * 2, cl_window_init * 4, max(0.2, cl_window_init * 8)]
    cd_targets = list(cd_targets)

    for w in windows:
        cand = np.where(np.abs(cls_labels - target_cl) <= w)[0]
        cand = cand[~np.isnan(cds_train[cand])]
        if len(cand) < len(cd_targets):
            continue

        print(f"[DEBUG] candidates in CL window: {len(cand)}")
        print(f"[DEBUG] CD range in window: min={np.min(cand):.6f}, max={np.max(cand):.6f}")
        for q in [0, 5, 25, 50, 75, 95, 100]:
            print(f"[DEBUG] CD p{q:02d}: {np.percentile(cand, q):.6f}")

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
# Generation with regeneration until XFoil converges
# ---------------------------
def generate_until_xfoil_converges(
    *,
    diffuser: RePaintDiffuser,
    model,
    dataset: AirfoilDataset,
    coord_std_1: torch.Tensor,  # (1,2,L)
    cls_std_1: torch.Tensor,  # (1,1)
    mask_1: torch.Tensor,  # (1,L)
    shifts: List[Optional[float]],
    n_per_shift: int,
    angle: float,
    xfoil_workers: int,
    max_regen_attempts: int,
    seed_base: int,
    convergence_mode: str,  # "all" or "per_shift_min"
    min_valid_per_shift: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    生成をやり直して XFoil が収束するまで追い込む。
    mode:
      - all: 全サンプルが valid になるまで再生成
      - per_shift_min: 各 shift で valid が min_valid_per_shift 個以上になれば停止
        （満たした shift の残り invalid は再生成しない）
    戻り:
      rep_denorm_flat: (total,2,L) float32
      cl_out: (total,)
      cd_out: (total,)
      valid: (total,) bool
      attempts_used: int
      per_shift_valid_counts: (S,) int
    """
    device = coord_std_1.device
    S = len(shifts)
    N = n_per_shift
    total = S * N

    if convergence_mode not in {"all", "per_shift_min"}:
        raise ValueError(f"Unknown convergence_mode: {convergence_mode}")
    if convergence_mode == "per_shift_min" and (min_valid_per_shift < 1):
        raise ValueError("min_valid_per_shift must be >= 1")
    if convergence_mode == "per_shift_min" and (N < min_valid_per_shift):
        raise ValueError(
            f"n_per_shift ({N}) must be >= min_valid_per_shift ({min_valid_per_shift}) for per_shift_min mode."
        )

    shift_vals = np.array([0.0 if s is None else float(s) for s in shifts], dtype=np.float32)  # (S,)
    shift_idx_full = np.repeat(np.arange(S, dtype=int), N)  # (total,)
    shift_grid_full = shift_vals[shift_idx_full]  # (total,)

    L = coord_std_1.shape[-1]
    rep_denorm_flat = np.zeros((total, 2, L), dtype=np.float32)
    cl_out = np.full((total,), np.nan, dtype=float)
    cd_out = np.full((total,), np.nan, dtype=float)
    valid = np.zeros((total,), dtype=bool)

    pool = None
    if xfoil_workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=xfoil_workers)

    def per_shift_counts(v: np.ndarray) -> np.ndarray:
        return np.array([int(np.sum(v[s * N : (s + 1) * N])) for s in range(S)], dtype=int)

    def is_satisfied(v: np.ndarray) -> Tuple[bool, np.ndarray]:
        cnt = per_shift_counts(v)
        if convergence_mode == "all":
            return bool(np.all(v)), cnt
        # per_shift_min
        return bool(np.all(cnt >= min_valid_per_shift)), cnt

    try:
        for attempt in range(max_regen_attempts + 1):
            ok, cnt = is_satisfied(valid)
            if ok:
                return rep_denorm_flat, cl_out, cd_out, valid, attempt, cnt

            if convergence_mode == "all":
                idx_to_gen = np.where(~valid)[0]
            else:
                # per_shift_min: 未達の shift だけ、かつ invalid な要素のみ再生成
                need_shifts = np.where(cnt < min_valid_per_shift)[0]
                mask_need = np.isin(shift_idx_full, need_shifts)
                idx_to_gen = np.where(mask_need & (~valid))[0]

            if idx_to_gen.size == 0:
                # 理論上起きにくいが安全のため
                return rep_denorm_flat, cl_out, cd_out, valid, attempt, cnt

            bsz = int(idx_to_gen.size)

            shift_sel = shift_grid_full[idx_to_gen].astype(np.float32)  # (bsz,)
            cls_batch = cls_std_1.repeat(bsz, 1)
            cls_batch = cls_batch + (torch.from_numpy(shift_sel).to(device).view(-1, 1) / dataset.cl_std)

            coord_batch = coord_std_1.repeat(bsz, 1, 1)
            mask_batch = mask_1.repeat(bsz, 1)

            # attempt ごとにノイズを変える（=生成を変える）
            set_seed(seed_base + attempt * 10000 + bsz)

            with torch.no_grad():
                repainted_std = diffuser.repaint(model, coord_batch, mask_batch, cls_batch)

            rep_den = dataset.denormalize_coord(repainted_std).cpu().numpy().astype(np.float32)  # (bsz,2,L)

            cls_sub, cds_sub = compute_clcd_batch(rep_den, angle=angle, workers=xfoil_workers, pool=pool)

            rep_denorm_flat[idx_to_gen] = rep_den
            cl_out[idx_to_gen] = cls_sub
            cd_out[idx_to_gen] = cds_sub
            valid[idx_to_gen] = (~np.isnan(cls_sub)) & (~np.isnan(cds_sub))

        cnt_final = per_shift_counts(valid)
        return rep_denorm_flat, cl_out, cd_out, valid, max_regen_attempts, cnt_final
    finally:
        if pool is not None:
            pool.close()
            pool.join()


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

    # repaint common settings
    parser.add_argument("--mask_type", type=str, default="head", choices=["head", "tail", "middle", "top", "bottom"])
    parser.add_argument("--missing_points_ratio", type=float, default=0.6)
    parser.add_argument("--num_resampling", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=10)
    parser.add_argument("--blend_ratio", type=float, default=0.1)
    parser.add_argument("--blend_type", type=str, default="linear", choices=["linear", "constant"])
    parser.add_argument("--blend_value", type=float, default=0.5)

    # shifts
    parser.add_argument(
        "--cl_shifts",
        type=str,
        default="None,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="comma-separated; use None for no-shift",
    )
    parser.add_argument("--n_per_shift", type=int, default=3, help="samples per CL_SHIFT (>=1)")

    # xfoil
    parser.add_argument("--angle", type=float, default=5.0)
    parser.add_argument("--xfoil_workers", type=int, default=1, help=">1 to parallelize CL/CD computation")
    parser.add_argument("--max_regen_attempts", type=int, default=15, help="max regeneration rounds per base")

    # convergence relaxed mode
    parser.add_argument(
        "--convergence_mode",
        type=str,
        default="per_shift_min",
        choices=["all", "per_shift_min"],
        help="all: all samples must converge; per_shift_min: each shift needs >=min_valid_per_shift valid samples",
    )
    parser.add_argument("--min_valid_per_shift", type=int, default=1, help="used when convergence_mode=per_shift_min")

    # scatter
    parser.add_argument("--train_max_points", type=int, default=8000, help="max train points to draw on scatter")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

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
        f"_CM{args.convergence_mode}_MIN{args.min_valid_per_shift}"
    )
    run_dir = os.path.join(base_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

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
        model_name=MODEL_NAME,
    )

    train_cache_path = get_or_build_train_clcd_cache(dataset, cache_dir, args.angle)
    train_cache = np.load(train_cache_path)
    train_cls_xfoil = train_cache["cls"]
    train_cds_xfoil = train_cache["cds"]

    cls_labels = dataset.denormalize_cl(dataset.cls_tensor).numpy().flatten()

    cd_targets = parse_float_list(args.cd_targets)
    base_indices = select_bases_by_cl_cd(
        cls_labels=cls_labels,
        cds_train=train_cds_xfoil,
        target_cl=args.target_cl,
        cl_window_init=args.cl_window,
        cd_targets=cd_targets,
    )

    picked_csv = os.path.join(run_dir, "selected_bases_clcd.csv")
    selected_rows = dump_selected_bases(
        out_csv_path=picked_csv,
        base_indices=base_indices,
        cls_labels=cls_labels,
        train_cls_xfoil=train_cls_xfoil,
        train_cds_xfoil=train_cds_xfoil,
        cd_targets=cd_targets,
        select_mode=getattr(args, "base_select_mode", "targets"),
    )

    shifts = parse_shifts(args.cl_shifts)

    rng = np.random.default_rng(args.seed)
    valid_train = (~np.isnan(train_cls_xfoil)) & (~np.isnan(train_cds_xfoil))
    valid_idx = np.where(valid_train)[0]
    if args.train_max_points > 0 and len(valid_idx) > args.train_max_points:
        plot_train_idx = rng.choice(valid_idx, size=args.train_max_points, replace=False)
    else:
        plot_train_idx = valid_idx

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

    for b_i, idx in enumerate(base_indices, start=1):
        base_cl_label = float(cls_labels[idx])
        base_cd = float(train_cds_xfoil[idx]) if not np.isnan(train_cds_xfoil[idx]) else np.nan

        base_subdir = os.path.join(run_dir, f"base{b_i:02d}_idx{idx}_CL{base_cl_label:.3f}_CD{base_cd:.4f}")
        os.makedirs(base_subdir, exist_ok=True)

        coord_std_1 = dataset.coords_tensor[idx : idx + 1].to(device)  # (1,2,L)
        cls_std_1 = dataset.cls_tensor[idx : idx + 1].to(device)  # (1,1)

        coord_denorm = dataset.denormalize_coord(coord_std_1).cpu().numpy()[0]  # (2,L)
        mask_np = mask_manual(
            pattern=args.mask_type,
            coords=coord_denorm,
            missing_ratio=args.missing_points_ratio,
            blend_ratio=args.blend_ratio,
            blend_type=args.blend_type,
            blend_value=args.blend_value,
        )
        mask_1 = torch.from_numpy(mask_np).unsqueeze(0).to(device).float()  # (1,L)

        seed_base = args.seed + b_i * 1000
        rep_denorm_flat, cl_out_flat, cd_out_flat, valid_flat, attempts_used, per_shift_cnt = (
            generate_until_xfoil_converges(
                diffuser=diffuser,
                model=model,
                dataset=dataset,
                coord_std_1=coord_std_1,
                cls_std_1=cls_std_1,
                mask_1=mask_1,
                shifts=shifts,
                n_per_shift=args.n_per_shift,
                angle=args.angle,
                xfoil_workers=args.xfoil_workers,
                max_regen_attempts=args.max_regen_attempts,
                seed_base=seed_base,
                convergence_mode=args.convergence_mode,
                min_valid_per_shift=args.min_valid_per_shift,
            )
        )

        S = len(shifts)
        N = args.n_per_shift
        L = coord_denorm.shape[1]
        total = S * N
        gen_all = rep_denorm_flat.reshape(S, N, 2, L)

        gen_path = os.path.join(base_subdir, "generated.npy")
        np.save(gen_path, gen_all)

        shift_vals = np.array([0.0 if s is None else float(s) for s in shifts], dtype=np.float32)

        metrics_rows: List[Dict] = []
        for i in range(total):
            s_i = i // N
            n_i = i % N
            shift_obj = shifts[s_i]
            shift_val = float(shift_vals[s_i])
            cl_target = base_cl_label + shift_val

            cl_out = float(cl_out_flat[i])
            cd_out = float(cd_out_flat[i])
            is_valid = bool(valid_flat[i])

            metrics_rows.append(
                {
                    "base_order": b_i,
                    "base_index": int(idx),
                    "base_cl_label": base_cl_label,
                    "base_cd_train": base_cd,
                    "shift": None if shift_obj is None else float(shift_obj),
                    "shift_numeric": shift_val,
                    "cl_target": float(cl_target),
                    "sample_id": int(n_i),
                    "cl_out": cl_out,
                    "cd_out": cd_out,
                    "valid": int(is_valid),
                    "cl_err": (cl_out - cl_target) if is_valid else np.nan,
                    "cl_ape_pct": (
                        abs((cl_out - cl_target) / max(abs(cl_target), 1e-12)) * 100.0 if is_valid else np.nan
                    ),
                }
            )

        csv_path = os.path.join(base_subdir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            for r in metrics_rows:
                writer.writerow(r)

        rep_ids = pick_representative_sample_ids(valid_flat, S=S, N=N)

        conv_ratio_all = float(np.mean(valid_flat))
        per_shift_cnt_list = [int(x) for x in per_shift_cnt.tolist()]

        base_meta = {
            "base_order": b_i,
            "base_index": int(idx),
            "base_cl_label": base_cl_label,
            "base_cd_train": base_cd,
            "convergence_mode": args.convergence_mode,
            "min_valid_per_shift": int(args.min_valid_per_shift),
            "xfoil_convergence_ratio_all_samples": conv_ratio_all,
            "xfoil_valid_count_per_shift": per_shift_cnt_list,
            "regen_attempts_used": int(attempts_used),
            "representative_sample_id_per_shift": [int(x) for x in rep_ids.tolist()],
            "outputs": {
                "generated_npy": gen_path,
                "metrics_csv": csv_path,
            },
        }

        # ---------------------------
        # Plot 1) row plot
        # ---------------------------
        cols = 1 + S
        fig_w = max(16, 3.2 * cols)
        fig, axes = plt.subplots(1, cols, figsize=(fig_w, 4.5))

        base_title = f"BASE\nCL(label): {base_cl_label:.3f}\nCD(train): {base_cd:.4f}"
        plot_base_panel(coord_denorm, mask_np, axes[0], title=base_title)

        # representative sample per shift
        rep_metrics = {}
        for r in metrics_rows:
            if r["sample_id"] == rep_ids[int(np.where(shift_vals == r["shift_numeric"])[0][0])] if False else False:
                pass  # (unused)
        # (簡潔化: タイトル用には代表sampleの行を都度探す)
        for s_i, shift in enumerate(shifts):
            n_rep = int(rep_ids[s_i])
            rep = gen_all[s_i, n_rep]

            # find matching metric row
            r0 = None
            for r in metrics_rows:
                if (int(r["sample_id"]) == n_rep) and (
                    abs(float(r["shift_numeric"]) - float(shift_vals[s_i])) < 1e-12
                ):
                    r0 = r
                    break

            if r0 is None:
                title = f"SHIFT {shift}\n(rep={n_rep}, no-metrics)"
            else:
                title = (
                    f"SHIFT {shift}\n"
                    f"rep sample_id: {n_rep}\n"
                    f"CL_tgt: {fmt(r0['cl_target'], '.3f')}\n"
                    f"CL_out: {fmt(r0['cl_out'], '.3f')}\n"
                    f"CD_out: {fmt(r0['cd_out'], '.4f')}\n"
                    f"valid: {int(r0['valid'])}"
                )

            plot_inpaint(coord_denorm, rep, mask_np, axes[1 + s_i], title=title)

        plt.tight_layout()
        add_row_legend_known_blend_generated(fig, axes)

        row_png = os.path.join(base_subdir, "row.png")
        fig.savefig(row_png, dpi=250)
        plt.close(fig)

        # ---------------------------
        # Plot 2) overlay plot (representative per shift)
        # ---------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(coord_denorm[0], coord_denorm[1], color="k", linewidth=2.0, label="base (original)")

        cmap = plt.get_cmap("viridis")
        for s_i, shift in enumerate(shifts):
            rep = gen_all[s_i, int(rep_ids[s_i])]
            c = cmap(s_i / max(S - 1, 1))
            lab = "generated (shift=None)" if shifts[s_i] is None else f"generated (shift={float(shifts[s_i]):.1f})"
            ax.plot(rep[0], rep[1], linewidth=1.6, color=c, label=lab)

        ax.set_aspect("equal", "box")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.25, 0.4)
        ax.grid(True)
        ax.set_title("Overlay (base + generated for each CL_SHIFT)")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        overlay_png = os.path.join(base_subdir, "overlay.png")
        fig.savefig(overlay_png, dpi=250)
        plt.close(fig)

        # ---------------------------
        # Plot 3) CL–CD scatter (figsize=(8,6))
        # ---------------------------
        cl_gen = np.array([r["cl_out"] for r in metrics_rows], dtype=float)
        cd_gen = np.array([r["cd_out"] for r in metrics_rows], dtype=float)
        shift_gen = np.array([r["shift_numeric"] for r in metrics_rows], dtype=float)
        valid_gen = (~np.isnan(cl_gen)) & (~np.isnan(cd_gen))

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(
            train_cls_xfoil[plot_train_idx],
            train_cds_xfoil[plot_train_idx],
            s=10,
            alpha=0.25,
            label="train",
        )

        sc = ax.scatter(
            cl_gen[valid_gen],
            cd_gen[valid_gen],
            c=shift_gen[valid_gen],
            s=35,
            label="generated (colored by shift)",
        )
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("CL_SHIFT (None treated as 0.0)")

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

        base_meta["outputs"].update(
            {
                "row_png": row_png,
                "overlay_png": overlay_png,
                "cl_cd_scatter_png": scatter_png,
            }
        )
        with open(os.path.join(base_subdir, "meta.json"), "w") as f:
            json.dump(base_meta, f, indent=2, ensure_ascii=False)

        print(
            f"[DONE base {b_i:02d}] {base_subdir} | mode={args.convergence_mode} "
            f"attempts={attempts_used} per_shift_valid={per_shift_cnt_list} all_valid_ratio={conv_ratio_all:.3f}"
        )

    print("\n[ALL DONE]")
    print(f"Outputs in: {run_dir}")


if __name__ == "__main__":
    main()

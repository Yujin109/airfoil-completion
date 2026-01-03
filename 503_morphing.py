#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
500_morphing.py

更新要件（今回の最終仕様）:
- 各 base 翼型ごとに、全 shift（× n_per_shift）を 1 回の RePaint でまとめて生成する（バッチ化）
- XFoil(CL/CD) が収束しない（NaN）場合は「生成自体」をやり直す（同一形状に対してXFoil再計算はしない）
- 終了条件:
    「全ての shift で、少なくとも 1 つ以上 XFoil が収束した生成が得られるまで」
    その間は “全ての shift を毎回” 再生成し続ける（収束済み shift も継続生成して候補を増やす）
- 採用（プロット/画像に使う）生成:
    収束した候補の中から、各 shift ごとに
        |CL_out - CL_target| が最小（= CL 誤差が最小）
    の生成を採用する（attempt をまたいで最良を選ぶ）
- 画像:
  1) 横並び row.png（BASE + shift別 generated）
     - BASE は全て実線
     - 凡例は known / blend / generated の 3 つだけ
  2) overlay.png（shift別に色、各shiftは上の「最良」生成を使用）
  3) cl_cd_scatter.png（train + best generated、figsize=(8,6)、色=shift）

保存先:
results/{EXECUTION_NAME}/morphing/{tag}/baseXX_idx.../
  ├─ row.png
  ├─ overlay.png
  ├─ cl_cd_scatter.png
  ├─ generated_best.npy          # (S,2,L) 各shiftの採用形状
  ├─ generated_last_attempt.npy  # (S,N,2,L) 最終試行の全生成（デバッグ用）
  ├─ metrics_all.csv             # 全試行×全shift×全サンプル
  ├─ metrics_best.csv            # 各shiftの採用結果（S行）
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
    BASEは全て実線。missing を強調しない（凡例は known/blend/generated の3つに限定するため）。
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
# Core: regenerate until each shift has >=1 converged, while tracking best by CL error
# ---------------------------
def generate_until_all_shifts_have_valid_best_by_cl(
    *,
    diffuser: RePaintDiffuser,
    model,
    dataset: AirfoilDataset,
    coord_std_1: torch.Tensor,  # (1,2,L)
    cls_std_1: torch.Tensor,  # (1,1)
    mask_1: torch.Tensor,  # (1,L)
    base_cl_label: float,
    shifts: List[Optional[float]],
    n_per_shift: int,
    angle: float,
    xfoil_workers: int,
    max_attempts: int,
    seed_base: int,
) -> Tuple[
    np.ndarray,  # best_coords (S,2,L)
    np.ndarray,  # best_cl (S,)
    np.ndarray,  # best_cd (S,)
    np.ndarray,  # best_abs_cl_err (S,)
    np.ndarray,  # best_attempt (S,)
    np.ndarray,  # best_sample_id (S,)
    np.ndarray,  # has_valid (S,)
    int,  # attempts_used
    np.ndarray,  # last_attempt_coords (S,N,2,L)
    np.ndarray,  # last_attempt_cl (S,N)
    np.ndarray,  # last_attempt_cd (S,N)
    List[Dict],  # records_all
]:
    device = coord_std_1.device
    S = len(shifts)
    N = int(n_per_shift)
    if N < 1:
        raise ValueError("n_per_shift must be >= 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    shift_vals = np.array([0.0 if s is None else float(s) for s in shifts], dtype=np.float32)  # (S,)
    cl_targets = base_cl_label + shift_vals  # (S,)
    L = coord_std_1.shape[-1]

    # best trackers (valid only)
    has_valid = np.zeros(S, dtype=bool)
    best_abs_err = np.full(S, np.inf, dtype=float)
    best_coords = np.zeros((S, 2, L), dtype=np.float32)
    best_cl = np.full(S, np.nan, dtype=float)
    best_cd = np.full(S, np.nan, dtype=float)
    best_attempt = np.full(S, -1, dtype=int)
    best_sample_id = np.full(S, -1, dtype=int)

    # last attempt snapshots (always updated)
    last_attempt_coords = np.zeros((S, N, 2, L), dtype=np.float32)
    last_attempt_cl = np.full((S, N), np.nan, dtype=float)
    last_attempt_cd = np.full((S, N), np.nan, dtype=float)

    pool = None
    if xfoil_workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=xfoil_workers)

    records_all: List[Dict] = []

    try:
        for attempt in range(1, max_attempts + 1):
            total = S * N
            shift_grid = np.repeat(shift_vals, N).astype(np.float32)  # (total,)

            cls_batch = cls_std_1.repeat(total, 1)
            cls_batch = cls_batch + (torch.from_numpy(shift_grid).to(device).view(-1, 1) / dataset.cl_std)

            coord_batch = coord_std_1.repeat(total, 1, 1)
            mask_batch = mask_1.repeat(total, 1)

            set_seed(seed_base + attempt * 10000)

            with torch.no_grad():
                repainted_std = diffuser.repaint(model, coord_batch, mask_batch, cls_batch)

            rep_den_flat = dataset.denormalize_coord(repainted_std).cpu().numpy().astype(np.float32)  # (total,2,L)
            cls_out_flat, cds_out_flat = compute_clcd_batch(
                rep_den_flat, angle=angle, workers=xfoil_workers, pool=pool
            )

            rep_den = rep_den_flat.reshape(S, N, 2, L)
            cls_out = cls_out_flat.reshape(S, N)
            cds_out = cds_out_flat.reshape(S, N)

            # update last snapshots
            last_attempt_coords = rep_den.copy()
            last_attempt_cl = cls_out.copy()
            last_attempt_cd = cds_out.copy()

            # update records + best(valid) per shift
            for s_i in range(S):
                tgt = float(cl_targets[s_i])
                for n_i in range(N):
                    clv = float(cls_out[s_i, n_i])
                    cdv = float(cds_out[s_i, n_i])
                    is_valid = (not np.isnan(clv)) and (not np.isnan(cdv))
                    abs_err = abs(clv - tgt) if is_valid else np.nan

                    records_all.append(
                        {
                            "attempt": attempt,
                            "shift": None if shifts[s_i] is None else float(shifts[s_i]),
                            "shift_numeric": float(shift_vals[s_i]),
                            "cl_target": tgt,
                            "sample_id": int(n_i),
                            "cl_out": clv,
                            "cd_out": cdv,
                            "valid": int(is_valid),
                            "abs_cl_err": abs_err,
                        }
                    )

                    if is_valid:
                        has_valid[s_i] = True
                        if abs_err < best_abs_err[s_i]:
                            best_abs_err[s_i] = abs_err
                            best_coords[s_i] = rep_den[s_i, n_i]
                            best_cl[s_i] = clv
                            best_cd[s_i] = cdv
                            best_attempt[s_i] = attempt
                            best_sample_id[s_i] = int(n_i)

            if bool(np.all(has_valid)):
                # 全shiftで1個以上 valid を得たので終了（bestは既に保持済み）
                return (
                    best_coords,
                    best_cl,
                    best_cd,
                    best_abs_err,
                    best_attempt,
                    best_sample_id,
                    has_valid,
                    attempt,
                    last_attempt_coords,
                    last_attempt_cl,
                    last_attempt_cd,
                    records_all,
                )

        # max_attempts reached:
        # 未収束 shift は「最終試行の形状」を採用して埋める（NaNでも形状は描画できる）
        for s_i in range(S):
            if not has_valid[s_i]:
                # ここは「最後にNaNになった形状」を使うので last_attempt から選ぶ
                # N>1でも、全部NaNならどれでも同じなので 0 を採用
                n_sel = 0
                best_coords[s_i] = last_attempt_coords[s_i, n_sel]
                best_cl[s_i] = float(last_attempt_cl[s_i, n_sel])  # nan のまま
                best_cd[s_i] = float(last_attempt_cd[s_i, n_sel])  # nan のまま
                best_abs_err[s_i] = np.nan
                best_attempt[s_i] = max_attempts
                best_sample_id[s_i] = n_sel

        return (
            best_coords,
            best_cl,
            best_cd,
            best_abs_err,
            best_attempt,
            best_sample_id,
            has_valid,
            max_attempts,
            last_attempt_coords,
            last_attempt_cl,
            last_attempt_cd,
            records_all,
        )
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
    parser.add_argument("--n_per_shift", type=int, default=3, help="samples per shift per attempt (>=1)")

    # xfoil + regeneration
    parser.add_argument("--angle", type=float, default=5.0)
    parser.add_argument("--xfoil_workers", type=int, default=1, help=">1 to parallelize CL/CD computation")
    parser.add_argument("--max_attempts", type=int, default=15, help="max regeneration attempts per base")

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
        f"_N{args.n_per_shift}_ATT{args.max_attempts}"
        f"_BESTbyCL"
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
    shift_vals = np.array([0.0 if s is None else float(s) for s in shifts], dtype=np.float32)

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

        (
            best_coords,
            best_cl,
            best_cd,
            best_abs_err,
            best_attempt,
            best_sample_id,
            has_valid,
            attempts_used,
            last_attempt_coords,
            last_attempt_cl,
            last_attempt_cd,
            records_all,
        ) = generate_until_all_shifts_have_valid_best_by_cl(
            diffuser=diffuser,
            model=model,
            dataset=dataset,
            coord_std_1=coord_std_1,
            cls_std_1=cls_std_1,
            mask_1=mask_1,
            base_cl_label=base_cl_label,
            shifts=shifts,
            n_per_shift=args.n_per_shift,
            angle=args.angle,
            xfoil_workers=args.xfoil_workers,
            max_attempts=args.max_attempts,
            seed_base=seed_base,
        )

        S = len(shifts)
        N = int(args.n_per_shift)
        L = coord_denorm.shape[1]

        # 保存（採用形状 + 最終試行の全生成）
        best_npy = os.path.join(base_subdir, "generated_best.npy")
        last_npy = os.path.join(base_subdir, "generated_last_attempt.npy")
        np.save(best_npy, best_coords.astype(np.float32))  # (S,2,L)
        np.save(last_npy, last_attempt_coords.astype(np.float32))  # (S,N,2,L)

        # metrics_all.csv
        metrics_all_path = os.path.join(base_subdir, "metrics_all.csv")
        with open(metrics_all_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records_all[0].keys()))
            writer.writeheader()
            for r in records_all:
                writer.writerow(r)

        # metrics_best.csv (S行)
        metrics_best_rows: List[Dict] = []
        for s_i in range(S):
            cl_tgt = float(base_cl_label + shift_vals[s_i])
            metrics_best_rows.append(
                {
                    "shift": None if shifts[s_i] is None else float(shifts[s_i]),
                    "shift_numeric": float(shift_vals[s_i]),
                    "cl_target": cl_tgt,
                    "best_cl_out": float(best_cl[s_i]),
                    "best_cd_out": float(best_cd[s_i]),
                    "best_abs_cl_err": float(best_abs_err[s_i]) if np.isfinite(best_abs_err[s_i]) else np.nan,
                    "best_attempt": int(best_attempt[s_i]),
                    "best_sample_id": int(best_sample_id[s_i]),
                    "has_valid": int(bool(has_valid[s_i])),
                }
            )
        metrics_best_path = os.path.join(base_subdir, "metrics_best.csv")
        with open(metrics_best_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_best_rows[0].keys()))
            writer.writeheader()
            for r in metrics_best_rows:
                writer.writerow(r)

        # meta
        base_meta = {
            "base_order": b_i,
            "base_index": int(idx),
            "base_cl_label": base_cl_label,
            "base_cd_train": base_cd,
            "attempts_used": int(attempts_used),
            "stop_condition": "all_shifts_have_at_least_one_converged_sample",
            "best_selection": "minimize |CL_out - CL_target| among converged candidates across all attempts",
            "per_shift_has_valid": [int(x) for x in has_valid.tolist()],
            "per_shift_best_attempt": [int(x) for x in best_attempt.tolist()],
            "per_shift_best_sample_id": [int(x) for x in best_sample_id.tolist()],
            "per_shift_best_abs_cl_err": [float(x) if np.isfinite(x) else None for x in best_abs_err.tolist()],
            "outputs": {
                "generated_best_npy": best_npy,
                "generated_last_attempt_npy": last_npy,
                "metrics_all_csv": metrics_all_path,
                "metrics_best_csv": metrics_best_path,
            },
        }

        # ---------------------------
        # Plot 1) row plot (base + best per shift)
        # ---------------------------
        cols = 1 + S
        fig_w = max(16, 3.2 * cols)
        fig, axes = plt.subplots(1, cols, figsize=(fig_w, 4.5))

        base_title = f"BASE\nCL(label): {base_cl_label:.3f}\nCD(train): {base_cd:.4f}"
        plot_base_panel(coord_denorm, mask_np, axes[0], title=base_title)

        for s_i, shift in enumerate(shifts):
            rep = best_coords[s_i]
            converged = bool(has_valid[s_i])
            note = "" if converged else "NOT CONVERGED (using last attempt)"
            title = (
                f"SHIFT {shift}\n"
                f"{note}\n"
                f"best(attempt={int(best_attempt[s_i])}, id={int(best_sample_id[s_i])})\n"
                f"CL_tgt: {fmt(base_cl_label + float(shift_vals[s_i]), '.3f')}\n"
                f"CL_out: {fmt(best_cl[s_i], '.3f')}\n"
                f"CD_out: {fmt(best_cd[s_i], '.4f')}\n"
                f"|err|: {fmt(best_abs_err[s_i], '.4f')}"
            )
            plot_inpaint(coord_denorm, rep, mask_np, axes[1 + s_i], title=title)

        plt.tight_layout()
        add_row_legend_known_blend_generated(fig, axes)

        row_png = os.path.join(base_subdir, "row.png")
        fig.savefig(row_png, dpi=250)
        plt.close(fig)

        # ---------------------------
        # Plot 2) overlay plot (best per shift)
        # ---------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(coord_denorm[0], coord_denorm[1], color="k", linewidth=2.0, label="base (original)")

        cmap = plt.get_cmap("viridis")
        for s_i, shift in enumerate(shifts):
            rep = best_coords[s_i]
            c = cmap(s_i / max(S - 1, 1))
            lab = "generated (shift=None)" if shifts[s_i] is None else f"generated (shift={float(shifts[s_i]):.1f})"
            ax.plot(rep[0], rep[1], linewidth=1.6, color=c, label=lab)

        ax.set_aspect("equal", "box")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.25, 0.4)
        ax.grid(True)
        ax.set_title("Overlay (base + best generated for each CL_SHIFT)")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        overlay_png = os.path.join(base_subdir, "overlay.png")
        fig.savefig(overlay_png, dpi=250)
        plt.close(fig)

        # ---------------------------
        # Plot 3) CL–CD scatter (best only, figsize=(8,6))
        # ---------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(
            train_cls_xfoil[plot_train_idx],
            train_cds_xfoil[plot_train_idx],
            s=10,
            alpha=0.25,
            label="train",
        )

        valid_best = (~np.isnan(best_cl)) & (~np.isnan(best_cd))
        sc = ax.scatter(
            best_cl[valid_best],
            best_cd[valid_best],
            c=shift_vals[valid_best],
            s=60,
            label="generated (best per shift, colored by shift)",
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
        ax.set_title("CL–CD Scatter (train + best generated)")
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
            f"[DONE base {b_i:02d}] {base_subdir} | attempts_used={attempts_used} "
            f"all_shifts_valid={bool(np.all(has_valid))}"
        )

    print("\n[ALL DONE]")
    print(f"Outputs in: {run_dir}")


if __name__ == "__main__":
    main()

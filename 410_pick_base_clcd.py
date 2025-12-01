#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
410_pick_base_clcd.py

学習データから「target_cl に最も近い」翼型を 400 と同じアルゴリズムで選び、
その1枚について XFoil で CL/CD を計算して出力します。

出力は標準出力に加え、results/{EXECUTION_NAME}/analysis/ に
timestamp 付きの .txt と .json を保存します。
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch  # dataset の .to() 呼び出しで必要なため
from xfoil.model import Airfoil

import config
from config import EXECUTION_NAME
from src.data import AirfoilDataset
from src.xfoil_utils import get_xf_instance


def compute_cl_cd(coord_denorm: np.ndarray, angle: float = 5.0):
    """coord_denorm: shape (2, L)（標準化前）→ (cl, cd)"""
    xf = get_xf_instance()
    xf.Re = 3e6
    xf.max_iter = 100
    x, y = coord_denorm.reshape(2, -1)
    xf.airfoil = Airfoil(x=x, y=y)
    try:
        c = xf.a(angle)
        if c is None or len(c) < 2:
            return float("nan"), float("nan")
        cl = float(np.round(c[0], 10))
        cd = float(np.round(c[1], 10))
        return cl, cd
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_cl", type=float, required=True, help="例: 0.682")
    ap.add_argument("--angle", type=float, default=5.0, help="XFoil 迎角[deg]")
    args = ap.parse_args()

    # 出力先
    analysis_dir = os.path.join("results", EXECUTION_NAME, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")

    # 学習データ読み込み
    dataset = AirfoilDataset()

    # 400 と同じ選択アルゴリズム：denorm CL が target_cl に最も近い index
    cls_denorm_all = dataset.denormalize_cl(dataset.cls_tensor).numpy().flatten()
    idx_closest = int(np.argmin(np.abs(cls_denorm_all - args.target_cl)))
    cl_in = float(cls_denorm_all[idx_closest])

    # 座標(標準化→標準化前)を取得
    coord_std = dataset.coords_tensor[idx_closest : idx_closest + 1]
    coord_denorm = dataset.denormalize_coord(coord_std).cpu().numpy()[0]  # (2, L)

    # XFoil で CL/CD 計算
    base_cl, base_cd = compute_cl_cd(coord_denorm, angle=args.angle)

    # 標準出力
    print("=== Picked Base Airfoil (same algorithm as 400) ===")
    print(f"EXECUTION_NAME   : {EXECUTION_NAME}")
    print(f"picked_index     : {idx_closest}")
    print(f"target_cl (input): {args.target_cl:.6f}")
    print(f"cl_in (dataset)  : {cl_in:.6f}")
    print(f"angle [deg]      : {args.angle:.3f}")
    print(f"base_cl (XFoil)  : {base_cl}")
    print(f"base_cd (XFoil)  : {base_cd}")

    # 保存（txt / json）
    txt_path = os.path.join(
        analysis_dir,
        f"{ts}_base_clcd_idx{idx_closest}_CLIN{args.target_cl:.3f}_angle{args.angle:.3f}.txt",
    )
    with open(txt_path, "w") as f:
        f.write(f"EXECUTION_NAME: {EXECUTION_NAME}\n")
        f.write(f"picked_train_index: {idx_closest}\n")
        f.write(f"target_cl: {args.target_cl:.6f}\n")
        f.write(f"cl_in (dataset): {cl_in:.10f}\n")
        f.write(f"angle(deg): {args.angle:.3f}\n")
        f.write(f"base_cl: {base_cl:.10f}\n")
        f.write(f"base_cd: {base_cd:.10f}\n")

    meta = {
        "timestamp": ts,
        "execution_name": EXECUTION_NAME,
        "picked_train_index": idx_closest,
        "target_cl": args.target_cl,
        "cl_in_dataset": cl_in,
        "angle_deg": args.angle,
        "base_cl": base_cl,
        "base_cd": base_cd,
        "paths": {"txt": txt_path},
    }
    json_path = os.path.join(
        analysis_dir,
        f"{ts}_base_clcd_idx{idx_closest}_CLIN{args.target_cl:.3f}_angle{args.angle:.3f}.json",
    )
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print("  TXT :", txt_path)
    print("  JSON:", json_path)


if __name__ == "__main__":
    main()

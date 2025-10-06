#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
401_sweep_repaint_config_analysis.py

- 400_repeat_config_analysis.py をパラメータ直積で連続実行
- 各実行の meta_json パスを拾い、stats を読み取って summary CSV を作成
- 進捗バー / 失敗ログ / ドライラン対応

使い方例:
python 401_sweep_repaint_config_analysis.py \
  --target_cl 0.682 \
  --n 100 \
  --mask_type top,bottom \
  --missing_points_ratio 0.5 \
  --blend_ratio 0.1 \
  --blend_type linear \
  --blend_value 1.0 \
  --num_resampling 10 \
  --jump_length 10 \
  --cl_shift 0.1 \
  --angle 5.0 \
  --train_fraction 1.0 \
  --scatter_labels "train(sample),generated"

※ 単一値もリストも OK（カンマ区切り）
"""

import argparse
import csv
import datetime
import json
import os
import shlex
import subprocess
from itertools import product

from tqdm import tqdm

import config
from config import EXECUTION_NAME

HERE = os.path.abspath(os.path.dirname(__file__))
PY = shlex.quote(os.environ.get("PYTHON", "python"))
SCRIPT_400 = os.path.join(HERE, "400_repeat_config_analysis.py")


def parse_list(s, typ=float):
    # None→[default], "a,b"→[a,b], "1"→[1]
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return [typ(s)]
    if isinstance(s, str):
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        if typ == int:
            return [int(p) for p in parts]
        if typ == float:
            return [float(p) for p in parts]
        if typ == str:
            return parts
    return [s]


def build_cmd(params: dict):
    # 400 の CLI を作成
    args = [
        f"--target_cl {params['target_cl']}",
        f"--n {params['n']}",
        f"--mask_type {params['mask_type']}",
        f"--missing_points_ratio {params['missing_points_ratio']}",
        f"--blend_ratio {params['blend_ratio']}",
        f"--blend_type {params['blend_type']}",
        f"--blend_value {params['blend_value']}",
        f"--num_resampling {params['num_resampling']}",
        f"--jump_length {params['jump_length']}",
        f"--cl_shift {params['cl_shift']}",
        f"--angle {params['angle']}",
        f"--train_fraction {params['train_fraction']}",
        f"--scatter_labels {shlex.quote(params['scatter_labels'])}",
        f"--seed {params['seed']}",
    ]
    return f'{PY} "{SCRIPT_400}" ' + " ".join(args)


def parse_stats(stats_path: str):
    """
    400 が出力する stats_*.txt をパースし、キー→値の dict を返す。
    """
    out = {}
    if not os.path.exists(stats_path):
        return out
    with open(stats_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_cl", type=str, required=True, help="例: 0.62,0.682")
    ap.add_argument("--n", type=str, default="100")
    ap.add_argument("--mask_type", type=str, default="top")
    ap.add_argument("--missing_points_ratio", type=str, default="0.5")
    ap.add_argument("--blend_ratio", type=str, default="0.1")
    ap.add_argument("--blend_type", type=str, default="linear")
    ap.add_argument("--blend_value", type=str, default="1.0")
    ap.add_argument("--num_resampling", type=str, default="10")
    ap.add_argument("--jump_length", type=str, default="10")
    ap.add_argument("--cl_shift", type=str, default="0.1")
    ap.add_argument("--angle", type=str, default="5.0")
    ap.add_argument("--train_fraction", type=str, default="1.0")
    ap.add_argument("--scatter_labels", type=str, default="train(sample),generated")
    ap.add_argument("--seed", type=str, default="42")
    ap.add_argument("--dry_run", action="store_true", help="実行せずコマンド一覧だけ表示")
    args = ap.parse_args()

    # 直積用のリスト化
    grid = {
        "target_cl": parse_list(args.target_cl, float),
        "n": parse_list(args.n, int),
        "mask_type": parse_list(args.mask_type, str),
        "missing_points_ratio": parse_list(args.missing_points_ratio, float),
        "blend_ratio": parse_list(args.blend_ratio, float),
        "blend_type": parse_list(args.blend_type, str),
        "blend_value": parse_list(args.blend_value, float),
        "num_resampling": parse_list(args.num_resampling, int),
        "jump_length": parse_list(args.jump_length, int),
        "cl_shift": parse_list(args.cl_shift, float),
        "angle": parse_list(args.angle, float),
        "train_fraction": parse_list(args.train_fraction, float),
        "scatter_labels": parse_list(args.scatter_labels, str),
        "seed": parse_list(args.seed, int),
    }

    keys = list(grid.keys())
    combos = [dict(zip(keys, values)) for values in product(*[grid[k] for k in keys])]

    analysis_dir = os.path.join("results", EXECUTION_NAME, "analysis")
    sweeps_dir = os.path.join(analysis_dir, "sweeps")
    os.makedirs(sweeps_dir, exist_ok=True)
    batch_ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    csv_path = os.path.join(sweeps_dir, f"{batch_ts}_summary.csv")
    fail_log = os.path.join(sweeps_dir, f"{batch_ts}_failed.txt")

    if args.dry_run:
        print(f"[DRY RUN] total {len(combos)} jobs")
        for p in combos:
            print(build_cmd(p))
        return

    rows = []
    with open(fail_log, "w") as flog:
        pbar = tqdm(total=len(combos), desc="Sweep")
        for params in combos:
            cmd = build_cmd(params)
            print(f"\n>>> RUN: {cmd}")
            try:
                # 実行して stdout を取得
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=dict(os.environ, PYTHONUNBUFFERED="1"),
                )
                stdout = proc.stdout
                stderr = proc.stderr
                if proc.returncode != 0:
                    flog.write(f"RET={proc.returncode}\nCMD={cmd}\nSTDERR={stderr}\n\n")
                    pbar.update(1)
                    continue

                # meta_json の行からパスを取得
                meta_path = None
                for line in stdout.splitlines():
                    if line.strip().startswith("meta_json:"):
                        meta_path = line.split(":", 1)[1].strip()
                        break
                if not meta_path or not os.path.exists(meta_path):
                    flog.write(f"NO_META\nCMD={cmd}\nSTDOUT={stdout}\nSTDERR={stderr}\n\n")
                    pbar.update(1)
                    continue

                with open(meta_path, "r") as f:
                    meta = json.load(f)
                stats_path = meta["outputs"]["stats_txt"]
                stats = parse_stats(stats_path)

                # サマリ行（パラメータ + 主要指標）
                row = {
                    "meta": meta_path,
                    "generated_npy": meta["outputs"]["generated_npy"],
                    "hist_cl_png": meta["outputs"]["hist_cl_png"],
                    "hist_err_png": meta["outputs"]["hist_err_png"],
                    "cl_cd_png": meta["outputs"]["cl_cd_png"],
                    # from params
                    **{k: params[k] for k in keys},
                    # from stats
                    "convergence_ratio": stats.get("convergence_ratio", ""),
                    "cl_mean": stats.get("cl_mean", ""),
                    "cl_std": stats.get("cl_std", ""),
                    "err_mean": stats.get("err_mean", ""),
                    "err_std": stats.get("err_std", ""),
                    "MAPE(%)": stats.get("MAPE(%)", ""),
                }
                rows.append(row)

            except Exception as e:
                flog.write(f"EXC={repr(e)}\nCMD={cmd}\n\n")
            finally:
                pbar.update(1)
        pbar.close()

    if rows:
        # CSV 書き出し
        fieldnames = [
            "meta",
            "generated_npy",
            "hist_cl_png",
            "hist_err_png",
            "cl_cd_png",
            "target_cl",
            "n",
            "mask_type",
            "missing_points_ratio",
            "blend_ratio",
            "blend_type",
            "blend_value",
            "num_resampling",
            "jump_length",
            "cl_shift",
            "angle",
            "train_fraction",
            "scatter_labels",
            "seed",
            "convergence_ratio",
            "cl_mean",
            "cl_std",
            "err_mean",
            "err_std",
            "MAPE(%)",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\n[SWEEP SUMMARY] {csv_path}")
    else:
        print("\n[SWEEP SUMMARY] No successful runs. See fail log:", fail_log)


if __name__ == "__main__":
    main()

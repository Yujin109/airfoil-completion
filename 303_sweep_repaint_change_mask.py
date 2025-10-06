#!/usr/bin/env python
"""
Sweep script for 300_repaint.py

・config.py の値を環境変数で設定してから 300_repaint.py を subprocess で実行
・パラメータ組み合わせは itertools.product で列挙
・失敗した組み合わせは returncode をチェックしてログに残す
"""

import itertools
import os
import subprocess
import sys

# ── スイープ設定 ─────────────────────────────────────────────────────────

# SWEEP_SETTINGS = [
#     {
#         "mask_types": ["head", "tail", "middle"],
#         "missing_points_ratio_list": [0.1, 0.3, 0.5, 0.7, 0.9],
#         "cl_shifts": [None],
#         "num_resampling_list": [1],
#         "jump_length_list": [1],
#     },
#     {
#         "mask_types": ["head", "tail", "middle"],
#         "missing_points_ratio_list": [0.7],
#         "cl_shifts": [None],
#         "num_resampling_list": [10, 20, 30, 40, 50],
#         "jump_length_list": [10],
#     },
#     {
#         "mask_types": ["head", "tail", "middle"],
#         "missing_points_ratio_list": [0.1, 0.3, 0.5, 0.7, 0.9],
#         "cl_shifts": [None, 0.1],
#         "num_resampling_list": [10],
#         "jump_length_list": [10],
#     },
# ]

SWEEP_SETTINGS = [
    {
        "mask_types": ["top", "bottom"],
        "missing_points_ratio_list": [0.5],
        "cl_shifts": [None],
        "num_resampling_list": [1],
        "jump_length_list": [1],
    },
    {
        "mask_types": ["top", "bottom"],
        "missing_points_ratio_list": [0.5],
        "cl_shifts": [None],
        "num_resampling_list": [10, 20, 30, 40, 50],
        "jump_length_list": [10],
    },
    {
        "mask_types": ["top", "bottom"],
        "missing_points_ratio_list": [0.5],
        "cl_shifts": [None, 0.1],
        "num_resampling_list": [10],
        "jump_length_list": [10],
    },
]


def run_one(mask_type, missing_points_ratio, cl_shift, num_resampling, jump_length):
    """
    1回分のスイープを環境変数を使って外部プロセス実行する
    """
    # 環境変数をコピーして必要なものを上書き
    env = os.environ.copy()
    env["MASK_TYPE"] = mask_type
    env["MISSING_POINTS_RATIO"] = str(missing_points_ratio)
    env["CL_SHIFT"] = "" if cl_shift is None else str(cl_shift)
    env["NUM_RESAMPLING"] = str(num_resampling)
    env["JUMP_LENGTH"] = str(jump_length)

    # サブプロセスで300_repaint.pyを実行
    cmd = [sys.executable, "302_repaint_change_mask.py"]
    print(
        f"\n>>> Running: {cmd} with \
"
        f"MASK_TYPE={mask_type}, MISSING_POINTS_RATIO={missing_points_ratio}, CL_SHIFT={cl_shift}, \
"
        f"NUM_RESAMPLING={num_resampling}, JUMP_LENGTH={jump_length}"
    )
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(
            f"[ERROR] combination skipped: mask={mask_type}, "
            f"missing_points_ratio={missing_points_ratio}, cl_shift={cl_shift}, "
            f"num_resampling={num_resampling}, jump_length={jump_length}"
        )


if __name__ == "__main__":
    # 全組み合わせを試行
    for sweep_setting in SWEEP_SETTINGS:
        for mask, miss_ratio, shift, r, j in itertools.product(
            sweep_setting["mask_types"],
            sweep_setting["missing_points_ratio_list"],
            sweep_setting["cl_shifts"],
            sweep_setting["num_resampling_list"],
            sweep_setting["jump_length_list"],
        ):
            run_one(mask, miss_ratio, shift, r, j)

    print("\nAll sweeps finished.")

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

SWEEP_SETTINGS = [
    {
        "mask_types": ["head", "middle", "tail"],
        "missing_points_list": [24, 72, 124, 176],
        "cl_shifts": [None],
        "num_resampling_list": [20],
        "jump_length_list": [10],
    },
    {
        "mask_types": ["head", "middle", "tail"],
        "missing_points_list": [24, 72, 176, 224],
        "cl_shifts": [0.1],
        "num_resampling_list": [10],
        "jump_length_list": [10],
    },
]


def run_one(mask_type, missing_points, cl_shift, num_resampling, jump_length):
    """
    1回分のスイープを環境変数を使って外部プロセス実行する
    """
    # 環境変数をコピーして必要なものを上書き
    env = os.environ.copy()
    env["MASK_TYPE"] = mask_type
    env["MISSING_POINTS"] = str(missing_points)
    env["CL_SHIFT"] = "" if cl_shift is None else str(cl_shift)
    env["NUM_RESAMPLING"] = str(num_resampling)
    env["JUMP_LENGTH"] = str(jump_length)

    # サブプロセスで300_repaint.pyを実行
    cmd = [sys.executable, "300_repaint.py"]
    print(
        f"\n>>> Running: {cmd} with \
"
        f"MASK_TYPE={mask_type}, MISSING_POINTS={missing_points}, CL_SHIFT={cl_shift}, \
"
        f"NUM_RESAMPLING={num_resampling}, JUMP_LENGTH={jump_length}"
    )
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(
            f"[ERROR] combination skipped: mask={mask_type}, "
            f"missing_points={missing_points}, cl_shift={cl_shift}, "
            f"num_resampling={num_resampling}, jump_length={jump_length}"
        )


if __name__ == "__main__":
    # 全組み合わせを試行
    for sweep_setting in SWEEP_SETTINGS:
        for mask, miss, shift, r, j in itertools.product(
            sweep_setting["mask_types"],
            sweep_setting["missing_points_list"],
            sweep_setting["cl_shifts"],
            sweep_setting["num_resampling_list"],
            sweep_setting["jump_length_list"],
        ):
            run_one(mask, miss, shift, r, j)

    print("\nAll sweeps finished.")

#!/usr/bin/env python
"""
Sweep script for 302_repai306_repaint_change_mask_smoothingnt_change_mask.py (with smoothing)

・config.py の値を環境変数で設定してから 306_repaint_change_mask_smoothing.py を subprocess で実行
・RePaint + smoothing のパラメータスイープに対応
・Poetry を用いて仮想環境内で実行
"""

import itertools
import os
import subprocess
import sys

# ── スイープ設定 ─────────────────────────────────────────────────────────

SWEEP_SETTINGS = [
    # ===== お試し =====
    # {
    #     "mask_types": ["head"],
    #     "missing_points_ratio_list": [0.5],
    #     "cl_shifts": [0.1],
    #     "num_resampling_list": [10],
    #     "jump_length_list": [10],
    #     # smoothing
    #     "smooth_methods": ["moving_average"],  # "laplacian" / "moving_average" / "none"
    #     "smooth_known_ratios": [0.1],
    #     "smooth_alpha": [0.2, 0.4, 0.6],
    #     "smooth_ma_window": [5, 7, 9, 11],
    # },
    # ===== 欠損パターン比較 =====
    {
        "mask_types": ["head", "tail", "middle"],
        "missing_points_ratio_list": [0.1, 0.3, 0.5, 0.7, 0.9],
        "cl_shifts": [0.1, None],
        "num_resampling_list": [10],
        "jump_length_list": [10],
        # smoothing
        "smooth_methods": ["laplacian"],  # "laplacian" / "moving_average" / "savgol_threshold" / "none"
        "smooth_known_ratios": [0.1],
        "smooth_alpha": [0.2],
        "smooth_ma_window": [5],  # 関係なし
        "smooth_sg_window": [9],  # 関係なし
        "smooth_sg_degree": [2],  # 関係なし
        "smooth_sg_tau": [0.002],  # 関係なし
    },
    {
        "mask_types": ["head", "tail", "middle"],
        "missing_points_ratio_list": [0.1, 0.3, 0.5, 0.7, 0.9],
        "cl_shifts": [0.1, None],
        "num_resampling_list": [10],
        "jump_length_list": [10],
        # smoothing
        "smooth_methods": ["moving_average"],  # "laplacian" / "moving_average" / "savgol_threshold" / "none"
        "smooth_known_ratios": [0.1],
        "smooth_alpha": [0.4],  # 関係なし
        "smooth_ma_window": [7],
        "smooth_sg_window": [9],  # 関係なし
        "smooth_sg_degree": [2],  # 関係なし
        "smooth_sg_tau": [0.002],  # 関係なし
    },
    {
        "mask_types": ["head", "tail", "middle"],
        "missing_points_ratio_list": [0.1, 0.3, 0.5, 0.7, 0.9],
        "cl_shifts": [0.1, None],
        "num_resampling_list": [10],
        "jump_length_list": [10],
        # smoothing
        "smooth_methods": ["savgol_threshold"],  # "laplacian" / "moving_average" / "savgol_threshold" / "none"
        "smooth_known_ratios": [0.1],
        "smooth_alpha": [0.4],  # 関係なし
        "smooth_ma_window": [7],  # 関係なし
        "smooth_sg_window": [9],
        "smooth_sg_degree": [2],
        "smooth_sg_tau": [0.002],
    },
    # ===== top / bottom 比較 =====
    {
        "mask_types": ["top", "bottom"],
        "missing_points_ratio_list": [0.5],
        "cl_shifts": [0.1, None],
        "num_resampling_list": [10],
        "jump_length_list": [10],
        # smoothing
        "smooth_methods": ["laplacian"],  # "laplacian" / "moving_average" / "savgol_threshold" / "none"
        "smooth_known_ratios": [0.1],
        "smooth_alpha": [0.4],
        "smooth_ma_window": [5],  # 関係なし
        "smooth_sg_window": [9],  # 関係なし
        "smooth_sg_degree": [2],  # 関係なし
        "smooth_sg_tau": [0.002],  # 関係なし
    },
    {
        "mask_types": ["top", "bottom"],
        "missing_points_ratio_list": [0.5],
        "cl_shifts": [0.1, None],
        "num_resampling_list": [10],
        "jump_length_list": [10],
        # smoothing
        "smooth_methods": ["moving_average"],  # "laplacian" / "moving_average" / "savgol_threshold" / "none"
        "smooth_known_ratios": [0.1],
        "smooth_alpha": [0.4],  # 関係なし
        "smooth_ma_window": [7],
        "smooth_sg_window": [9],  # 関係なし
        "smooth_sg_degree": [2],  # 関係なし
        "smooth_sg_tau": [0.002],  # 関係なし
    },
    {
        "mask_types": ["top", "bottom"],
        "missing_points_ratio_list": [0.5],
        "cl_shifts": [0.1, None],
        "num_resampling_list": [10],
        "jump_length_list": [10],
        # smoothing
        "smooth_methods": ["savgol_threshold"],  # "laplacian" / "moving_average" / "savgol_threshold" / "none"
        "smooth_known_ratios": [0.1],
        "smooth_alpha": [0.4],  # 関係なし
        "smooth_ma_window": [7],  # 関係なし
        "smooth_sg_window": [9],
        "smooth_sg_degree": [2],
        "smooth_sg_tau": [0.002],
    },
]


def run_one(
    *,
    mask_type,
    missing_points_ratio,
    cl_shift,
    num_resampling,
    jump_length,
    smooth_method,
    smooth_known_ratio,
    smooth_alpha,
    smooth_ma_window,
    smooth_sg_window,
    smooth_sg_degree,
    smooth_sg_tau,
):
    """
    1 回分の実行を Poetry 経由で行う
    """
    env = os.environ.copy()

    # ---- RePaint 関連 ----
    env["MASK_TYPE"] = mask_type
    env["MISSING_POINTS_RATIO"] = str(missing_points_ratio)
    env["CL_SHIFT"] = "" if cl_shift is None else str(cl_shift)
    env["NUM_RESAMPLING"] = str(num_resampling)
    env["JUMP_LENGTH"] = str(jump_length)

    # ---- smoothing 関連 ----
    env["SMOOTH_METHOD"] = smooth_method
    env["SMOOTH_KNOWN_RATIO"] = str(smooth_known_ratio)
    env["SMOOTH_ALPHA"] = str(smooth_alpha)
    env["SMOOTH_MA_WINDOW"] = str(smooth_ma_window)

    env["SMOOTH_SG_WINDOW"] = str(smooth_sg_window)
    env["SMOOTH_SG_DEGREE"] = str(smooth_sg_degree)
    env["SMOOTH_SG_TAU"] = str(smooth_sg_tau)

    cmd = [
        "poetry",
        "run",
        "python",
        "306_repaint_change_mask_smoothing.py",
    ]

    print(
        "\n>>> Running with parameters:\n"
        f"  MASK_TYPE={mask_type}\n"
        f"  MISSING_POINTS_RATIO={missing_points_ratio}\n"
        f"  CL_SHIFT={cl_shift}\n"
        f"  NUM_RESAMPLING={num_resampling}\n"
        f"  JUMP_LENGTH={jump_length}\n"
        f"  SMOOTH_METHOD={smooth_method}\n"
        f"  SMOOTH_KNOWN_RATIO={smooth_known_ratio}\n"
        f"  SMOOTH_ALPHA={smooth_alpha}\n"
        f"  SMOOTH_MA_WINDOW={smooth_ma_window}\n"
        f"  SMOOTH_SG_WINDOW={smooth_sg_window}\n"
        f"  SMOOTH_SG_DEGREE={smooth_sg_degree}\n"
        f"  SMOOTH_SG_TAU={smooth_sg_tau}\n"
    )

    res = subprocess.run(cmd, env=env)

    if res.returncode != 0:
        print(
            f"[ERROR] skipped: mask={mask_type}, "
            f"mpr={missing_points_ratio}, cl_shift={cl_shift}, "
            f"R={num_resampling}, J={jump_length}, "
            f"smooth={smooth_method}, skr={smooth_known_ratio}"
        )


if __name__ == "__main__":
    for sweep in SWEEP_SETTINGS:
        for (
            mask,
            miss_ratio,
            shift,
            r,
            j,
            sm,
            skr,
            sa,
            smw,
            ssw,
            ssd,
            sst,
        ) in itertools.product(
            sweep["mask_types"],
            sweep["missing_points_ratio_list"],
            sweep["cl_shifts"],
            sweep["num_resampling_list"],
            sweep["jump_length_list"],
            sweep["smooth_methods"],
            sweep["smooth_known_ratios"],
            sweep["smooth_alpha"],
            sweep["smooth_ma_window"],
            sweep["smooth_sg_window"],
            sweep["smooth_sg_degree"],
            sweep["smooth_sg_tau"],
        ):
            run_one(
                mask_type=mask,
                missing_points_ratio=miss_ratio,
                cl_shift=shift,
                num_resampling=r,
                jump_length=j,
                smooth_method=sm,
                smooth_known_ratio=skr,
                smooth_alpha=sa,
                smooth_ma_window=smw,
                smooth_sg_window=ssw,
                smooth_sg_degree=ssd,
                smooth_sg_tau=sst,
            )

    print("\n✅ All sweeps finished.")

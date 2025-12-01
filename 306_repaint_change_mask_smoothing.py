import datetime
import os

# config.py をインポートする前に環境変数を反映
import config

config.MASK_TYPE = os.getenv("MASK_TYPE", config.MASK_TYPE)
config.MISSING_POINTS_RATIO = float(os.getenv("MISSING_POINTS_RATIO", config.MISSING_POINTS_RATIO))
cl_shift_env = os.getenv("CL_SHIFT", "")
config.CL_SHIFT = None if cl_shift_env == "" else float(cl_shift_env)
config.NUM_RESAMPLING = int(os.getenv("NUM_RESAMPLING", config.NUM_RESAMPLING))
config.JUMP_LENGTH = int(os.getenv("JUMP_LENGTH", config.JUMP_LENGTH))

# 平滑化関連のパラメータ（環境変数から上書き可能）
# x 軸全長に対して、既知領域のうちどれだけを「動かしてよい領域」にするか（blend の BLEND_RATIO に相当）
SMOOTH_KNOWN_RATIO = float(os.getenv("SMOOTH_KNOWN_RATIO", "0.1"))  # 例: 0.0〜1.0
# ラプラシアン平滑化のステップサイズ
SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA", "0.25"))
# ラプラシアン平滑化の反復回数
SMOOTH_NUM_ITERS = int(os.getenv("SMOOTH_NUM_ITERS", "20"))
# 移動平均フィルタの窓サイズ（奇数推奨、デフォルト 7）
SMOOTH_MA_WINDOW = int(os.getenv("SMOOTH_MA_WINDOW", "7"))
# Savitzky–Golay 風の局所平滑化用パラメータ
SMOOTH_SG_WINDOW = int(os.getenv("SMOOTH_SG_WINDOW", "9"))  # 奇数推奨
SMOOTH_SG_DEGREE = int(os.getenv("SMOOTH_SG_DEGREE", "2"))  # 多項式次数（2 or 3 が現実的）
SMOOTH_SG_TAU = float(os.getenv("SMOOTH_SG_TAU", "0.002"))  # 2階差分のしきい値
# 平滑化手法の選択
SMOOTH_METHOD = os.getenv("SMOOTH_METHOD", "laplacian")  # "laplacian" / "moving_average" / "savgol_threshold" / "none"

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    CL_SHIFT,
    CLUSTERS,
    DIFFUSION_PARAMS,
    FIG_SIZE,
    GUIDANCE_SCALE,
    JUMP_LENGTH,
    MASK_TYPE,
    MISSING_POINTS,
    MISSING_POINTS_RATIO,
    MODEL_NAME,
    NUM_RESAMPLING,
    REPAINT_DIR,
    SAMPLES_PER_CLUSTER,
    SEED,
    SHOW_KNOWN_MISSING,
    WEIGHT_PATH,
)
from src.data import AirfoilDataset
from src.diffusion import RePaintDiffuser
from src.metrics import smoothness_phi
from src.models.model_registry import MODEL_REGISTRY
from src.xfoil_utils import get_cl


def mask_manual(pattern="head", coords=None, missing_ratio=0.7):
    """
    x 軸の範囲に対して欠損領域を指定するマスク生成関数。
    pattern: "head", "tail", "middle", "top", "bottom"
    coords: numpy array shape (2, L) または 1D array of x (length L)
    missing_ratio: ドメイン全体に対する欠損割合 (0.0–1.0)
    戻り値: 長さ L の bool 配列 (True=known, False=missing)
    """
    if coords is None:
        raise ValueError("coords must be provided")
    if pattern not in {"head", "tail", "middle", "top", "bottom"}:
        raise ValueError(f"Unknown pattern: {pattern}")
    if pattern not in {"top", "bottom"}:
        if not (0.0 <= missing_ratio <= 1.0):
            raise ValueError("missing_ratio must be in [0,1]")

    # x 座標のみ取り出し
    if coords.ndim == 2:
        xs = coords[0]  # shape (L,)
    else:
        xs = coords  # shape (L,)

    x_min, x_max = xs.min(), xs.max()
    range_x = x_max - x_min
    n = xs.shape[0]
    half_n = n // 2

    if pattern == "head":
        # ドメインの先端側を missing_ratio 分だけ欠損
        mask = np.ones_like(xs, dtype=bool)
        end = x_min + missing_ratio * range_x
        mask[(xs >= x_min) & (xs <= end)] = False

    elif pattern == "tail":
        # ドメインの末端側を missing_ratio 分だけ欠損
        mask = np.ones_like(xs, dtype=bool)
        start = x_max - missing_ratio * range_x
        mask[(xs >= start) & (xs <= x_max)] = False

    elif pattern == "middle":
        # ドメイン両端に 1-missing_ratio を equally split した既知領域を配置
        mask = np.zeros_like(xs, dtype=bool)
        half = (1 - missing_ratio) * range_x / 2
        # 先端側 known
        mask[(xs >= x_min) & (xs <= x_min + half)] = True
        # 末端側 known
        mask[(xs >= x_max - half) & (xs <= x_max)] = True

    elif pattern == "top":
        # 上面側（インデックス half_n～n-1）を欠損
        mask = np.ones_like(xs, dtype=bool)
        mask[half_n:] = False

    elif pattern == "bottom":
        # 下面側（インデックス 0～half_n-1）を欠損
        mask = np.ones_like(xs, dtype=bool)
        mask[:half_n] = False

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return mask


# ====== 「変えてよい既知点」の決定：blend 実験と同じ幾何ルール ======


def changeable_known_mask_manual(
    pattern: str,
    coords: np.ndarray,
    missing_ratio: float,
    smooth_ratio: float,
) -> np.ndarray:
    """
    blend 実験の mask_manual() に倣って、
    - missing_ratio: 欠損領域の x 軸割合
    - smooth_ratio:  変えてよい既知領域（blend 領域に相当）の x 軸割合
    を用いて、「変化させてよい既知点」のインデックスを決定する。

    戻り値:
        changeable_known: shape (L,) bool, True=変化させてよい既知点
    """
    if coords is None:
        raise ValueError("coords must be provided")
    if pattern not in {"head", "tail", "middle", "top", "bottom"}:
        raise ValueError(f"Unknown pattern: {pattern}")
    if pattern not in {"top", "bottom"}:
        if not (0.0 <= missing_ratio <= 1.0 and 0.0 <= smooth_ratio <= 1.0 and missing_ratio + smooth_ratio <= 1.0):
            raise ValueError("missing_ratio and smooth_ratio must be in [0,1] and sum <= 1")

    # x 座標のみ取り出し
    xs = coords[0] if coords.ndim == 2 else coords
    x_min, x_max = xs.min(), xs.max()
    range_x = x_max - x_min

    n = xs.shape[0]
    half_n = n // 2

    changeable = np.zeros(n, dtype=bool)

    if pattern == "head":
        # missing: [x_min, x_min + missing_ratio * range_x]
        miss_start = x_min
        miss_end = x_min + missing_ratio * range_x
        # blend/smooth: [miss_end, miss_end + smooth_ratio * range_x]
        blend_start = miss_end
        blend_end = miss_end + smooth_ratio * range_x

        blend_idx = (xs > blend_start) & (xs <= blend_end)
        changeable[blend_idx] = True

    elif pattern == "tail":
        # missing: [x_max - missing_ratio * range_x, x_max]
        miss_end = x_max
        miss_start = x_max - missing_ratio * range_x
        # blend/smooth: [miss_start - smooth_ratio * range_x, miss_start]
        blend_end = miss_start
        blend_start = miss_start - smooth_ratio * range_x

        blend_idx = (xs >= blend_start) & (xs < blend_end)
        changeable[blend_idx] = True

    elif pattern == "middle":
        # 中央欠損: [miss_start, miss_end]
        half_known = (1.0 - missing_ratio) * range_x / 2
        miss_start = x_min + half_known
        miss_end = x_max - half_known

        # 左右に smooth_ratio * range_x を 1/2 ずつ分配
        blend_half = smooth_ratio * range_x / 2
        blend_left_start = miss_start - blend_half
        blend_left_end = miss_start
        blend_right_start = miss_end
        blend_right_end = miss_end + blend_half

        left_idx = (xs > blend_left_start) & (xs <= blend_left_end)
        right_idx = (xs >= blend_right_start) & (xs < blend_right_end)

        changeable[left_idx] = True
        changeable[right_idx] = True

    elif pattern == "top":
        # 上半面 (idx half_n～n-1) を欠損、下半面が既知
        miss_idx = np.arange(half_n, n)
        known_idx = np.arange(0, half_n)

        # ブレンド幅（x軸割合の左右それぞれ smooth_ratio/2）
        bw = smooth_ratio * range_x / 2

        if bw > 0:
            # 左側ブレンド (x_min -> x_min + bw) on known_idx
            bs, be = x_min, x_min + bw
            idxs_left = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            changeable[idxs_left] = True

            # 右側ブレンド (x_max - bw -> x_max) on known_idx
            bs, be = x_max - bw, x_max
            idxs_right = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            changeable[idxs_right] = True

    elif pattern == "bottom":
        # 下半面 (idx 0～half_n-1) を欠損、上半面が既知
        miss_idx = np.arange(0, half_n)
        known_idx = np.arange(half_n, n)

        bw = smooth_ratio * range_x / 2

        if bw > 0:
            # 左側ブレンド (x_min -> x_min + bw) on known_idx
            bs, be = x_min, x_min + bw
            idxs_left = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            changeable[idxs_left] = True

            # 右側ブレンド (x_max - bw -> x_max) on known_idx
            bs, be = x_max - bw, x_max
            idxs_right = known_idx[(xs[known_idx] >= bs) & (xs[known_idx] <= be)]
            changeable[idxs_right] = True

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return changeable


# ====== 平滑化関連のヘルパー関数群 ======


def _laplacian_smooth_single(
    orig_coords: np.ndarray,
    rep_coords: np.ndarray,
    known_mask: np.ndarray,
    changeable_mask: np.ndarray,
    alpha: float,
    n_iter: int,
) -> np.ndarray:
    """
    1 サンプル分のラプラシアン平滑化を実行する（非 wrap, 端点固定）。

    orig_coords:    shape (2,L) 元の（denorm 済み）座標（既知領域の参照用）
    rep_coords:     shape (2,L) RePaint 後（denorm 済み）の座標（初期値）
    known_mask:     shape (L,)  bool, True=既知 (missing=False)
    changeable_mask:shape (L,)  bool, True=変化させてよい既知点（blend 領域）
    alpha:          ラプラシアンのステップサイズ
    n_iter:         反復回数

    戻り値:
        coords: shape (2,L), 平滑化後座標
    """
    coords = rep_coords.copy()
    L = coords.shape[1]

    # 完全に守る既知点
    protected_known = known_mask & ~changeable_mask

    # 端点は常に元の orig を優先（後縁をいじらない）
    protected_known[0] = True
    protected_known[L - 1] = True

    interior = np.arange(1, L - 1)

    for _ in range(n_iter):
        prev = coords.copy()
        updated = prev.copy()

        # 内点のみラプラシアンを適用（wrap しない）
        for i in interior:
            lap = prev[:, i - 1] + prev[:, i + 1] - 2.0 * prev[:, i]
            updated[:, i] = prev[:, i] + alpha * lap

        # 完全に守る既知点は元の orig 座標に固定
        coords[:, protected_known] = orig_coords[:, protected_known]
        # それ以外（未知 + 変化可既知）は更新結果を採用
        free = ~protected_known
        coords[:, free] = updated[:, free]

    return coords


def _moving_average_smooth_single(
    orig_coords: np.ndarray,
    rep_coords: np.ndarray,
    known_mask: np.ndarray,
    changeable_mask: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    1 サンプル分の移動平均平滑化を実行する（非 wrap, 端点固定）。

    orig_coords:    shape (2,L) 元の（denorm 済み）座標
    rep_coords:     shape (2,L) RePaint 後（denorm 済み）の座標
    known_mask:     shape (L,)  bool, True=既知
    changeable_mask:shape (L,)  bool, True=変化させてよい既知点
    window_size:    移動平均窓サイズ（奇数推奨）

    戻り値:
        coords: shape (2,L), 平滑化後座標
    """
    coords = rep_coords.copy()
    L = coords.shape[1]

    protected_known = known_mask & ~changeable_mask
    # 端点は常に元の orig を優先
    protected_known[0] = True
    protected_known[L - 1] = True

    if window_size < 1:
        return coords
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2

    # ベース信号：守る既知点はあらかじめ orig にしておく
    base = coords.copy()
    base[:, protected_known] = orig_coords[:, protected_known]

    smoothed = base.copy()
    interior = np.arange(1, L - 1)

    for c in range(2):  # x, y
        for i in interior:
            left = max(0, i - half)
            right = min(L, i + half + 1)
            smoothed[c, i] = base[c, left:right].mean()

    # 完全に守る既知点は orig に固定し、それ以外は平滑化結果を採用
    coords[:, protected_known] = orig_coords[:, protected_known]
    free = ~protected_known
    coords[:, free] = smoothed[:, free]

    return coords


def _poly_fit_at_index(
    values: np.ndarray,
    index: int,
    window_size: int,
    degree: int,
) -> float:
    """
    1 次元信号 values に対して、index を中心とする局所ウィンドウで
    多項式フィットを行い、その index 位置での近似値を返す。

    - wrap はせず、[0, L-1] の範囲で切り詰める
    """
    L = values.shape[0]
    if window_size < 1:
        return float(values[index])
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2

    left = max(0, index - half)
    right = min(L, index + half + 1)  # [left, right)

    xs = np.arange(left, right, dtype=float)
    ys = values[left:right]

    # 局所座標系に平行移動（数値安定性のため）
    xs_local = xs - xs.mean()
    # 次数がサンプル数より大きくならないように制限
    deg = min(degree, len(xs) - 1)

    # Vandermonde 行列を作成し、最小二乗解を求める
    A = np.vander(xs_local, deg + 1, increasing=True)  # [N, deg+1]
    coeffs, *_ = np.linalg.lstsq(A, ys, rcond=None)

    x0_local = float(index - xs.mean())
    powers = np.array([x0_local**k for k in range(deg + 1)], dtype=float)
    y_hat = float(coeffs @ powers)
    return y_hat


def _savgol_threshold_smooth_single(
    orig_coords: np.ndarray,
    rep_coords: np.ndarray,
    known_mask: np.ndarray,
    changeable_mask: np.ndarray,
    window_size: int,
    degree: int,
    tau: float,
) -> np.ndarray:
    """
    1 サンプル分の「しきい値付き・局所 Savitzky–Golay 風」平滑化。

    流れ:
      1. y 座標について 2階差分を計算
      2. |Δ^2 y| > tau となるインデックスを「凸凹候補」とする
      3. その近傍のみ局所多項式フィットで更新
      4. 端点と protected_known は常に orig を優先

    orig_coords:    shape (2,L) 元の（denorm 済み）座標
    rep_coords:     shape (2,L) RePaint 後（denorm 済み）の座標
    known_mask:     shape (L,)  bool, True=既知
    changeable_mask:shape (L,)  bool, True=変化させてよい既知点
    window_size:    局所ウィンドウサイズ（奇数推奨）
    degree:         多項式次数
    tau:            2階差分のしきい値

    戻り値:
        coords: shape (2,L), 平滑化後座標
    """
    coords = rep_coords.copy()
    L = coords.shape[1]

    protected_known = known_mask & ~changeable_mask
    # 端点は常に元の orig を優先
    protected_known[0] = True
    protected_known[L - 1] = True

    base = coords.copy()
    base[:, protected_known] = orig_coords[:, protected_known]

    # y 座標の 2階差分で凸凹を検出（端点除く）
    y = base[1]
    second_diff = np.zeros(L, dtype=float)
    second_diff[1 : L - 1] = y[2:] - 2.0 * y[1:-1] + y[:-2]

    rough = np.abs(second_diff) > tau

    # 端点は対象外
    rough[0] = False
    rough[L - 1] = False

    # 「動かしてよい」かつ「凸凹」と判定された点だけを更新対象にする
    target = rough & (~protected_known)

    if not np.any(target):
        # 凸凹がなければ何もしない
        coords[:, protected_known] = orig_coords[:, protected_known]
        return coords

    # x, y それぞれ同じ target に対して局所多項式フィット
    smoothed = base.copy()
    for i in np.where(target)[0]:
        for c in range(2):  # x, y
            smoothed[c, i] = _poly_fit_at_index(
                values=base[c],
                index=int(i),
                window_size=window_size,
                degree=degree,
            )

    # 完全保護点は orig、それ以外は smoothed
    coords[:, protected_known] = orig_coords[:, protected_known]
    free = ~protected_known
    coords[:, free] = smoothed[:, free]

    return coords


def smooth_airfoil_batch(
    orig_batch: np.ndarray,
    rep_batch: np.ndarray,
    masks_np: list,
    pattern: str,
    missing_ratio: float,
    method: str = "laplacian",
    known_change_ratio: float = 0.2,
    alpha: float = 0.25,
    n_iter: int = 20,
    ma_window: int = 7,
):
    """
    バッチ単位で airfoil 曲線の平滑化を行う。

    orig_batch:          shape (B,2,L) 元の（denorm 済み）座標
    rep_batch:           shape (B,2,L) RePaint 後（denorm 済み）座標
    masks_np:            list or array, each shape (L,) bool, True=known
    pattern:             "head", "tail", "middle", "top", "bottom"
    missing_ratio:       欠損割合 (x 軸全長に対する)
    method:              "laplacian" / "moving_average" / "none"
    known_change_ratio:  x 軸全長に対する「変化させてよい既知領域」割合
    alpha, n_iter:       ラプラシアン用パラメータ
    ma_window:           移動平均窓サイズ

    返り値:
        rep_smoothed:            shape (B,2,L)
        changeable_known_list:   list of shape (L,) bool
    """
    B, C, L = rep_batch.shape
    rep_smoothed = rep_batch.copy()
    changeable_known_list = []

    if method is None or method.lower() == "none":
        for b in range(B):
            changeable_known_list.append(np.zeros(L, dtype=bool))
        return rep_smoothed, changeable_known_list

    m = method.lower()
    for b in range(B):
        orig = orig_batch[b]
        rep = rep_batch[b]
        known_mask = masks_np[b].astype(bool)

        # blend 実験と同じ幾何ルールで「変化させてよい既知点」を決定
        changeable_mask = changeable_known_mask_manual(
            pattern=pattern,
            coords=orig,
            missing_ratio=missing_ratio,
            smooth_ratio=known_change_ratio,
        )
        # 念のため、既知点以外は False にしておく
        changeable_mask = changeable_mask & known_mask

        if m == "laplacian":
            smoothed = _laplacian_smooth_single(
                orig_coords=orig,
                rep_coords=rep,
                known_mask=known_mask,
                changeable_mask=changeable_mask,
                alpha=alpha,
                n_iter=n_iter,
            )
        elif m == "moving_average":
            smoothed = _moving_average_smooth_single(
                orig_coords=orig,
                rep_coords=rep,
                known_mask=known_mask,
                changeable_mask=changeable_mask,
                window_size=ma_window,
            )
        elif m == "savgol_threshold":
            smoothed = _savgol_threshold_smooth_single(
                orig_coords=orig,
                rep_coords=rep,
                known_mask=known_mask,
                changeable_mask=changeable_mask,
                window_size=SMOOTH_SG_WINDOW,
                degree=SMOOTH_SG_DEGREE,
                tau=SMOOTH_SG_TAU,
            )
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        rep_smoothed[b] = smoothed
        changeable_known_list.append(changeable_mask)

    return rep_smoothed, changeable_known_list


# ====== 可視化 ======


def plot_inpaint(
    orig,
    rep_before,
    rep_after,
    mask_known,
    changeable_known,
    ax,
    title: str = "",
    show_known_missing: bool = False,
):
    """
    可視化用プロット関数。

    - orig:            shape (2,L)  元の座標（denorm 済み）
    - rep_before:      shape (2,L)  平滑化前の RePaint 結果
    - rep_after:       shape (2,L)  平滑化後の RePaint 結果
    - mask_known:      shape (L,)   bool, True=known
    - changeable_known:shape (L,)   bool, True=平滑化で変更させてよい既知点
    """
    m_known = mask_known.astype(bool)
    changeable_known = changeable_known.astype(bool)

    # 既知領域のうち、平滑化していない既知点
    unchanged_known = m_known & ~changeable_known
    # 平滑化の対象になった既知点
    changed_known = m_known & changeable_known
    # 欠損領域（inpaint 部分）
    missing = ~m_known

    # 1) 変更されていない既知領域：orig を青の実線
    if np.any(unchanged_known):
        ax.plot(
            np.ma.array(orig[0], mask=~unchanged_known),
            np.ma.array(orig[1], mask=~unchanged_known),
            label="known_unchanged",
            linestyle="-",
            color="tab:blue",
        )

    # 2) 変更された既知領域：
    #    平滑化前（rep_before）を青の破線
    #    平滑化後（rep_after）を青の実線
    if np.any(changed_known):
        ax.plot(
            np.ma.array(rep_before[0], mask=~changed_known),
            np.ma.array(rep_before[1], mask=~changed_known),
            label="known_changed_before",
            linestyle="--",
            color="tab:blue",
        )
        ax.plot(
            np.ma.array(rep_after[0], mask=~changed_known),
            np.ma.array(rep_after[1], mask=~changed_known),
            label="known_changed_after",
            linestyle="-",
            color="tab:blue",
        )

    # 3) 欠損領域（inpaint 部分）：
    #    平滑化前（rep_before）をオレンジ破線
    #    平滑化後（rep_after）をオレンジ実線
    if np.any(missing):
        ax.plot(
            np.ma.array(rep_before[0], mask=~missing),
            np.ma.array(rep_before[1], mask=~missing),
            label="inpaint_before",
            linestyle="--",
            color="tab:orange",
        )
        ax.plot(
            np.ma.array(rep_after[0], mask=~missing),
            np.ma.array(rep_after[1], mask=~missing),
            label="inpaint_after",
            linestyle="-",
            color="tab:orange",
        )

    if show_known_missing:
        # 本来既知だったが mask 上は欠損として扱われている領域などを確認したい場合用
        ax.plot(
            np.ma.array(orig[0], mask=~missing),
            np.ma.array(orig[1], mask=~missing),
            label="known_missing",
            linestyle=":",
            color="tab:green",
        )

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 0.35)
    ax.tick_params(labelsize=9)
    ax.grid(True)
    # 必要に応じて legend を表示
    # ax.legend(fontsize=6)


def main():
    # prepare directories
    os.makedirs(REPAINT_DIR, exist_ok=True)

    # load dataset and model
    dataset = AirfoilDataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_REGISTRY[MODEL_NAME]
    model = model_cls().to(device)
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # initialize RePaint diffuser
    diffuser = RePaintDiffuser(
        **DIFFUSION_PARAMS,
        device=device,
        guidance_scale=GUIDANCE_SCALE,
        num_resampling=NUM_RESAMPLING,
        jump_length=JUMP_LENGTH,
    )

    # select sample indices per cluster
    np.random.seed(SEED)
    original_cls = dataset.denormalize_cl(dataset.cls_tensor).numpy().flatten()
    selected = []
    for lo, hi in CLUSTERS:
        idxs = np.where((original_cls >= lo) & (original_cls < hi))[0]
        if len(idxs) < SAMPLES_PER_CLUSTER:
            raise ValueError(f"Not enough samples in cluster [{lo},{hi}): {len(idxs)} found")
        sel = np.random.choice(idxs, SAMPLES_PER_CLUSTER, replace=False)
        selected.extend(sel)

    # prepare inputs
    coords_batch = dataset.coords_tensor[selected].to(device)  # [B,2,L] (normalized)
    cls_batch = dataset.cls_tensor[selected].to(device)

    # create masks (denormalized coords を使って mask_manual)
    coords_batch_denorm = dataset.denormalize_coord(coords_batch).cpu().numpy()  # [B,2,L]
    masks_np = []
    for coords in coords_batch_denorm:
        mask = mask_manual(pattern=MASK_TYPE, coords=coords, missing_ratio=MISSING_POINTS_RATIO)
        masks_np.append(mask)
    masks = torch.stack([torch.from_numpy(m) for m in masks_np], dim=0).to(device)

    if CL_SHIFT is not None:
        cls_batch += CL_SHIFT / dataset.cl_std
        for idx in selected:
            original_cls[idx] = original_cls[idx] + CL_SHIFT

    # run RePaint inpainting
    repainted = diffuser.repaint(model, coords_batch, masks, cls_batch)  # [B,2,L] (normalized)

    # denormalize for plotting and XFOIL
    orig_denorm = coords_batch_denorm  # [B,2,L], すでに denorm 済み
    rep_denorm_before = dataset.denormalize_coord(repainted).cpu().numpy()  # [B,2,L]

    # smoothing（平滑化）
    rep_denorm_after, changeable_known_list = smooth_airfoil_batch(
        orig_batch=orig_denorm,
        rep_batch=rep_denorm_before,
        masks_np=masks_np,
        pattern=MASK_TYPE,
        missing_ratio=MISSING_POINTS_RATIO,
        method=SMOOTH_METHOD,
        known_change_ratio=SMOOTH_KNOWN_RATIO,
        alpha=SMOOTH_ALPHA,
        n_iter=SMOOTH_NUM_ITERS,
        ma_window=SMOOTH_MA_WINDOW,
    )

    # compute CL on inpainted (平滑化後で評価)
    cl_out_list = []
    for sample in rep_denorm_after:
        cl_out_list.append(get_cl(sample))

    # plot results
    fig, axes = plt.subplots(SAMPLES_PER_CLUSTER, len(CLUSTERS), figsize=FIG_SIZE)
    for i, idx in enumerate(selected):
        row = i // len(CLUSTERS)
        col = i % len(CLUSTERS)
        ax = axes[row, col]

        if np.isnan(cl_out_list[i]):
            cl_ape = np.nan
        else:
            cl_ape = abs((cl_out_list[i] - original_cls[idx]) / original_cls[idx] * 100)

        title = f"CL_in: {original_cls[idx]:.3f}\n" f"CL_out: {cl_out_list[i]:.3f}\n" f"APE: {cl_ape:.1f}"

        plot_inpaint(
            orig=orig_denorm[i],
            rep_before=rep_denorm_before[i],
            rep_after=rep_denorm_after[i],
            mask_known=masks_np[i],
            changeable_known=changeable_known_list[i],
            ax=ax,
            title=title,
            show_known_missing=SHOW_KNOWN_MISSING,
        )

    plt.tight_layout()

    # save figure
    date_str = datetime.datetime.now().strftime("%y%m%d%H%M")
    # smoothing パラメータを method ごとに付加
    smooth_param_str = ""
    sm_lower = SMOOTH_METHOD.lower()
    if sm_lower == "laplacian":
        smooth_param_str = f"_SA:{SMOOTH_ALPHA}_SI:{SMOOTH_NUM_ITERS}"
    elif sm_lower == "moving_average":
        smooth_param_str = f"_MW:{SMOOTH_MA_WINDOW}"
    elif sm_lower == "savgol_threshold":
        smooth_param_str = f"_SGW:{SMOOTH_SG_WINDOW}_SGD:{SMOOTH_SG_DEGREE}_SGT:{SMOOTH_SG_TAU}"

    fname = (
        f"{date_str}_R:{NUM_RESAMPLING}_J:{JUMP_LENGTH}_M:{MASK_TYPE}_"
        f"MPR:{MISSING_POINTS_RATIO}_SHIFT:{CL_SHIFT}_"
        f"SM:{SMOOTH_METHOD}_SKR:{SMOOTH_KNOWN_RATIO}{smooth_param_str}.png"
    )
    fpath = os.path.join(REPAINT_DIR, fname)
    fig.savefig(fpath, dpi=300)
    print(f"[Saved sample images] {fpath}")

    # compute metrics

    metrics = dict()
    # MSE on cls
    cl_se_list = []
    cl_ape_list = []
    convergence_count = 0
    smoothness_phi_list = []
    for i, idx in enumerate(selected):
        cl_out = cl_out_list[i]
        cl_in = original_cls[idx]
        coord = dataset.coords_tensor[idx].cpu().numpy()
        if np.isnan(cl_out):
            cl_se_list.append(np.nan)
            cl_ape_list.append(np.nan)
        else:
            cl_se_list.append((cl_out - cl_in) ** 2)
            cl_ape_list.append(abs((cl_out - cl_in) / cl_in * 100))
            convergence_count += 1
        smoothness_phi_list.append(smoothness_phi(coord))
    metrics["cl_mse_mean"] = np.nanmean(cl_se_list)
    metrics["cl_mape_mean"] = np.nanmean(cl_ape_list)
    metrics["cl_convergence_ratio"] = convergence_count / len(selected)
    metrics["cl_smoothness_phi_mean"] = np.nanmean(smoothness_phi_list)

    # 追加で、APE の単純平均も計算（既存ロジックを維持）
    for i, idx in enumerate(selected):
        cl_out = cl_out_list[i]
        cl_in = original_cls[idx]
        cl_ape_list.append(abs((cl_out - cl_in) / cl_in * 100))
    metrics["cl_ape_mean"] = np.mean(cl_ape_list)

    # 既知領域のうち、「変化させてよい」としたポイント数の割合（point ベース）
    known_ratios_points = []
    for i in range(len(selected)):
        known_mask = masks_np[i].astype(bool)
        if known_mask.sum() == 0:
            known_ratios_points.append(0.0)
        else:
            ratio_points = changeable_known_list[i].sum() / known_mask.sum()
            known_ratios_points.append(ratio_points)
    metrics["known_change_ratio_points_mean"] = float(np.mean(known_ratios_points))
    metrics["known_change_ratio_x"] = SMOOTH_KNOWN_RATIO
    metrics["smooth_method"] = SMOOTH_METHOD
    metrics["smooth_alpha"] = SMOOTH_ALPHA
    metrics["smooth_num_iters"] = SMOOTH_NUM_ITERS
    metrics["smooth_ma_window"] = SMOOTH_MA_WINDOW

    # .txtファイルとして保存
    smooth_param_str = ""
    sm_lower = SMOOTH_METHOD.lower()
    if sm_lower == "laplacian":
        smooth_param_str = f"_SA:{SMOOTH_ALPHA}_SI:{SMOOTH_NUM_ITERS}"
    elif sm_lower == "moving_average":
        smooth_param_str = f"_MW:{SMOOTH_MA_WINDOW}"

    metrics_fname = (
        f"{date_str}_R:{NUM_RESAMPLING}_J:{JUMP_LENGTH}_M:{MASK_TYPE}_"
        f"MPR:{MISSING_POINTS_RATIO}_SHIFT:{CL_SHIFT}_"
        f"SM:{SMOOTH_METHOD}_SKR:{SMOOTH_KNOWN_RATIO}{smooth_param_str}.txt"
    )
    with open(os.path.join(REPAINT_DIR, metrics_fname), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()

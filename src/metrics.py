import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.data import AirfoilDataset
from src.diffusion import Diffuser
from src.xfoil_utils import get_cl


def running_average_filter(coord, kernel_size=9):
    x, y = coord
    x_filtered = np.convolve(x, np.ones(kernel_size) / kernel_size, mode="same")
    y_filtered = np.convolve(y, np.ones(kernel_size) / kernel_size, mode="same")
    return np.array([x_filtered, y_filtered])


def convexity_loss(coord):
    x, y = coord
    dx = np.diff(x)
    dy = np.diff(y)
    vectors = np.stack([dx, dy], axis=1)
    angles = []
    for k in range(len(vectors) - 1):
        v1 = vectors[k]
        v2 = vectors[k + 1]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            angle = 0
        else:
            dot = np.dot(v1, v2)
            cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
        angles.append(angle)
    return np.sum(angles) / len(angles) if len(angles) > 0 else 0


# def total_turning_angle(coord, unit="pi"):
#     """
#     閉じたループ全体の全外角和を返す。
#     理想的な円形なら ≈ 2π rad になる。

#     Parameters
#     ----------
#     coord : array-like
#         (2, N) か (N, 2) または (x_array, y_array) のタプル。
#     unit : str, optional, default="rad"
#         - "rad" : そのままラジアンで返す。
#         - "pi"  : πrad を単位にした値 (例: 円なら ≈2.0) を返す。

#     Returns
#     -------
#     phi : float
#         全外角和。unit="rad" のときは [rad]、unit="pi" のときは [πrad] の数値。
#     """
#     coord = np.asarray(coord)
#     # x,y を取り出す
#     if coord.ndim == 2 and coord.shape[0] == 2:
#         x, y = coord
#     elif coord.ndim == 2 and coord.shape[1] == 2:
#         x, y = coord[:, 0], coord[:, 1]
#     elif isinstance(coord, (tuple, list)) and len(coord) == 2:
#         x, y = map(np.asarray, coord)
#     else:
#         raise ValueError("coord は (2,N), (N,2), または (x_array, y_array) のいずれかで与えてください。")

#     N = x.size
#     # 辺ベクトル（最後→最初 も含む）
#     dx = np.diff(x, append=x[0])
#     dy = np.diff(y, append=y[0])
#     vecs = np.stack([dx, dy], axis=1)  # (N,2)

#     # 隣接ベクトルの内積・ノルム
#     v1 = vecs
#     v2 = np.roll(vecs, -1, axis=0)
#     dot = np.einsum("ij,ij->i", v1, v2)
#     norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)

#     # cosθ を計算・クリップ（ゼロ長ベクトルは角度0扱い）
#     cos_theta = np.zeros_like(dot)
#     mask = norm > 0
#     cos_theta[mask] = dot[mask] / norm[mask]
#     # zero-length のところは cos=1 にして θ=0, φ=π-0 → 0 に扱うなど
#     cos_theta[~mask] = 1.0
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)

#     # 内角 θ_i → 外角 φ_i = π - θ_i
#     theta = np.arccos(cos_theta)
#     phi = np.pi - theta

#     phi_sum = float(phi.sum())  # rad 単位での合計

#     print(f"phi_sum:{phi_sum}")

#     if unit == "rad":
#         return phi_sum
#     elif unit == "pi":
#         return phi_sum / np.pi
#     else:
#         raise ValueError("unit は 'rad' または 'pi' のいずれかで指定してください。")


def total_turning_angle(coord, unit="pi", signed=False):
    """
    閉じたループの全外角和を返します。
    signed=False（デフォルト）のときは絶対値を返します。

    coord : array-like
        (2, N) or (N, 2) または (x_array, y_array)
    unit : "rad" or "pi"
    signed : bool
        True にすると符号付きの値、False にすると絶対値を返す
    """
    coord = np.asarray(coord)
    # x,y の取り出し（省略可）
    if coord.ndim == 2 and coord.shape[0] == 2:
        x, y = coord
    elif coord.ndim == 2 and coord.shape[1] == 2:
        x, y = coord[:, 0], coord[:, 1]
    elif isinstance(coord, (tuple, list)) and len(coord) == 2:
        x, y = map(np.asarray, coord)
    else:
        raise ValueError("coord は (2,N), (N,2), (x_array,y_array) のいずれかで指定してください。")

    # 辺ベクトル
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    vecs = np.stack([dx, dy], axis=1)  # (N,2)

    # 隣り合う辺同士の外積・内積
    v1 = vecs
    v2 = np.roll(vecs, -1, axis=0)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # 2D 外積（スカラー）
    dot = np.einsum("ij,ij->i", v1, v2)  # 内積

    # 符号付きの回転角を計算
    angles = np.arctan2(cross, dot)  # 各頂点での turning-angle
    phi_sum = float(angles.sum())  # [rad]

    if not signed:
        phi_sum = abs(phi_sum)

    if unit == "rad":
        return phi_sum
    elif unit == "pi":
        return phi_sum / np.pi
    else:
        raise ValueError("unit は 'rad' または 'pi' を指定してください。")


def smoothness_phi(coord, unit: str = "pi") -> float:
    """
    論文の式(13) による滑らかさ指標 phi を計算する。

    Parameters
    ----------
    coord : array-like
        (2, N) または (N, 2) の座標配列，または (x_array, y_array) のタプル／リスト
    unit : {"rad", "pi"}, default="pi"
        - "rad": 結果をラジアン単位で返す
        - "pi" : 結果を π rad 単位で返す（例: 完全円なら 2.0）

    Returns
    -------
    phi : float
        滑らかさ指標 φ
    """
    coord = np.asarray(coord)
    # x, y の取り出し
    if coord.ndim == 2 and coord.shape[0] == 2:
        x, y = coord
    elif coord.ndim == 2 and coord.shape[1] == 2:
        x, y = coord[:, 0], coord[:, 1]
    elif isinstance(coord, (tuple, list)) and len(coord) == 2:
        x, y = map(np.asarray, coord)
    else:
        raise ValueError("coord は (2,N), (N,2), または (x_array, y_array) のタプルで指定してください。")

    # 辺ベクトル v_k = (x_{k+1}-x_k, y_{k+1}-y_k）
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    vecs = np.stack([dx, dy], axis=1)  # shape=(N,2)

    # 隣接ベクトル間の内積・ノルム
    v1 = vecs
    v2 = np.roll(vecs, -1, axis=0)
    dot = np.einsum("ij,ij->i", v1, v2)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)

    # cosθ の計算（ゼロノルムは角度0扱い）
    cosθ = np.where(norm > 0, dot / norm, 1.0)
    cosθ = np.clip(cosθ, -1.0, 1.0)

    # φ = Σ arccos(cosθ)
    φ = np.arccos(cosθ).sum()

    if unit == "pi":
        return φ / np.pi
    elif unit == "rad":
        return φ
    else:
        raise ValueError("unit は 'rad' または 'pi' のいずれかを指定してください。")


def smoothness_loss(coord):
    filtered = running_average_filter(coord, kernel_size=9)
    return np.mean((coord - filtered) ** 2)


def cl_squared_error(cl_conditioned, cl_evaluated):
    return np.mean((cl_conditioned - cl_evaluated) ** 2)


def cl_absolute_percentage_error(cl_conditioned: np.ndarray, cl_evaluated: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE)

    MAPE = mean( |(true - pred) / true| ) * 100

    cl_conditioned: 真値 (np.ndarray)
    cl_evaluated: 予測値 (np.ndarray)
    """
    # ゼロ除算を避けるため、真値が 0 の箇所には小さな値 ε を代入
    eps = np.finfo(float).eps
    denom = np.where(cl_conditioned == 0, eps, cl_conditioned)

    mape = np.mean(np.abs((cl_conditioned - cl_evaluated) / denom)) * 100
    return mape


def compute_generation_diversity(generated_batch):
    """
    Args:
        generated_batch: torch.Tensor of shape [N, 2, 248]
    Returns:
        diversity: float, 分散値 (L2ノルムベース)
    """
    N = generated_batch.size(0)
    flattened = generated_batch.view(N, -1)  # [N, 496]
    mean_vector = torch.mean(flattened, dim=0, keepdim=True)  # [1, 496]
    diffs = flattened - mean_vector  # [N, 496]
    diversity = torch.mean(torch.norm(diffs, dim=1) ** 2).item()
    return diversity


def compute_mean_deviation(batch: torch.Tensor) -> float:
    """
    Args:
        batch: torch.Tensor of shape [N, 2, 248]
    Returns:
        mu: float, 各形状ベクトルと平均形状ベクトルの L2 距離の平均
              μ = 1/N * Σ_i || g_i - g_mean ||
    """
    # バッチサイズ取得
    N = batch.size(0)
    # [N, 2, 248] → [N, 496]
    flattened = batch.view(N, -1)
    # 平均形状ベクトル [1, 496]
    mean_vector = flattened.mean(dim=0, keepdim=True)
    # 各サンプルとの差分 [N, 496]
    diffs = flattened - mean_vector
    # 各サンプルの L2 ノルム [N]
    norms = torch.norm(diffs, p=2, dim=1)
    # μ を計算して Python float で返す
    mu = norms.mean().item()
    return mu


def compute_euclid_dist(batch):
    N = batch.size(0)
    flattened = batch.view(N, -1)
    mean = flattened.mean(dim=0, keepdim=True)
    mu_d = (torch.norm(flattened - mean)).item() / N
    return mu_d


def evaluate_generated_samples(samples, conditioned_cls, dataset):
    convexity_losses, smoothness_losses, total_turning_angles, smoothness_phis, cl_ses, cl_apes = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    convergence_raw = convergence_strict = 0
    total = samples.shape[0]
    for i in range(total):
        sample = samples[i].detach().cpu()
        denorm = dataset.denormalize_coord(sample).cpu().numpy()
        try:
            cl_eval = get_cl(denorm)
        except Exception:
            cl_eval = np.nan

        if cl_eval is not None and not np.isnan(cl_eval):
            # conditioned_cls[i] が Tensor の場合は float に変換
            if isinstance(conditioned_cls, torch.Tensor):
                cl_cond_val = conditioned_cls[i].detach().cpu().item()
            else:
                cl_cond_val = conditioned_cls[i]
            cl_se = cl_squared_error(cl_cond_val, cl_eval)
            cl_ape = cl_absolute_percentage_error(cl_cond_val, cl_eval)
            converged = True
            convergence_raw += 1
        else:
            cl_se = np.nan
            cl_ape = np.nan
            converged = False
        convexity_losses.append(convexity_loss(denorm))
        smoothness_losses.append(smoothness_loss(denorm))
        total_turning_angles.append(total_turning_angle(denorm))
        smoothness_phis.append(smoothness_phi(denorm))
        cl_ses.append(cl_se)
        cl_apes.append(cl_ape)
        if (convexity_losses[-1] < 0.1 and smoothness_losses[-1] < 0.1 and cl_se < 0.1) and converged:
            convergence_strict += 1
    return (
        np.nanmean(convexity_losses),  # convexity_loss
        np.nanmean(smoothness_losses),  # smoothness loss
        np.nanmean(total_turning_angles),  # total_turning_angle
        np.nanmean(smoothness_phis),  # smoothness_phi
        np.nanmean([v for v in cl_ses if not np.isnan(v)]),  # MSE
        np.nanmean([v for v in cl_apes if not np.isnan(v)]),  # MAPE
        convergence_raw / total,  # convergence ratio
        convergence_strict / total,  # convergence ratio strict
        compute_generation_diversity(samples),  # L2 Norm ** 2
        compute_mean_deviation(samples),  # μ(mu) = L2 Norm
        compute_euclid_dist(samples),  # euclid dist?
    )


def evaluate_generated_samples_from_random_noise(
    model: torch.nn.Module,
    diffuser: Diffuser,
    dataset: AirfoilDataset,
    num_samples_for_each_cl: int = 10,
    device: str = "cuda",
):
    cl_eval_values = np.linspace(0.5, 1.2, 71)
    convexity_list = []
    smoothness_list = []
    total_turning_angle_list = []
    smoothness_phi_list = []
    cl_mse_list = []
    cl_mape_list = []
    convergence_ratios_raw = []
    convergence_ratios_strict = []
    diversity_list = []
    mu_list = []
    euclid_dist_list = []

    cl_conditioned_list = []
    coord_generated_list = []

    all_generated = []

    for cl_val in tqdm(cl_eval_values, total=len(cl_eval_values)):
        cond = torch.tensor([[cl_val, 0.0]] * num_samples_for_each_cl, dtype=torch.float32, device=device)
        cond_norm = dataset.normalize_cl(cond)
        generated = diffuser.generate_from_labels(model, cond_norm, coord_shape=(2, 248))

        cl_conditioned = np.array([cl_val] * num_samples_for_each_cl)
        (
            conv_l,
            smooth_l,
            angle_l,
            smoothness_phi,
            cl_mse,
            cl_mape,
            conv_ratio_raw,
            conv_ratio_strict,
            diversity,
            mu,
            euclid_dist,
        ) = evaluate_generated_samples(generated, cl_conditioned, dataset)
        convexity_list.append(conv_l)
        smoothness_list.append(smooth_l)
        total_turning_angle_list.append(angle_l)
        smoothness_phi_list.append(smoothness_phi)
        cl_mse_list.append(cl_mse)
        cl_mape_list.append(cl_mape)
        convergence_ratios_raw.append(conv_ratio_raw)
        convergence_ratios_strict.append(conv_ratio_strict)
        diversity_list.append(diversity)
        mu_list.append(mu)
        euclid_dist_list.append(euclid_dist)

        cl_conditioned_list.append(cl_conditioned)
        coord_generated_list.append(generated.cpu().numpy())

        # 全体多様度計算用にCPUテンソルを蓄積
        generated_denorm = dataset.denormalize_coord(generated)
        all_generated.append(generated_denorm.detach().cpu())

    # ループ後に全CLをまたいだバッチを結合
    all_batch = torch.cat(all_generated, dim=0)  # shape: [71*num_samples_for_each_cl, 2, 248]

    # 全体バッチでの多様度指標を一度だけ計算
    generation_diversity_all = compute_generation_diversity(all_batch)
    mu_all = compute_mean_deviation(all_batch)
    euclid_dist_all = compute_euclid_dist(all_batch)

    return (
        {
            "convexity_loss_mean": np.mean(convexity_list),
            "smoothness_loss_mean": np.mean(smoothness_list),
            "total_turning_angle": np.mean(total_turning_angle_list),
            "smoothness_phi": np.mean(smoothness_phi_list),
            "cl_mse_mean": np.mean(cl_mse_list),
            "cl_mape_mean": np.mean(cl_mape_list),
            "cl_convergence_ratio_raw": np.mean(convergence_ratios_raw),
            "cl_convergence_ratio_strict": np.mean(convergence_ratios_strict),
            "generation_diversity_mean": np.mean(diversity_list),
            "mu": np.mean(mu_list),
            "euclid_dist_mean": np.mean(euclid_dist_list),
            # 全CLをまたいだバッチ指標
            "generation_diversity_all": generation_diversity_all,
            "mu_all": mu_all,
            "euclid_dist_all": euclid_dist_all,
        },
        np.array(cl_conditioned_list),
        np.array(coord_generated_list),
    )


def plot_generated_samples_from_random_noise(
    model: torch.nn.Module,
    diffuser: Diffuser,
    dataset: AirfoilDataset,
    num_samples_for_each_cl: int = 5,
    output_dirs: str = "",
    epoch: int = 0,
    device: str = "cuda",
    suffix: str = "",
):
    assert output_dirs != ""
    if epoch > 0:
        sample_plot_path = os.path.join(output_dirs, f"samples_epoch_{epoch}{suffix}.png")
    else:
        sample_plot_path = os.path.join(output_dirs, f"samples{suffix}.png")

    cl_conditioned_list = []
    coord_generated_list = []

    cl_plot_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    fig, axs = plt.subplots(num_samples_for_each_cl, len(cl_plot_values), figsize=(30, 14))
    for col, cl_val in tqdm(enumerate(cl_plot_values), total=len(cl_plot_values)):
        cond = torch.tensor([[cl_val]] * num_samples_for_each_cl, dtype=torch.float32, device=device)
        cond_norm = dataset.normalize_cl(cond)
        generated = diffuser.generate_from_labels(model, cond_norm, coord_shape=(2, 248))
        cl_cond = cond[0][0].detach().cpu().numpy()

        for row in range(num_samples_for_each_cl):
            sample = generated[row].detach().cpu()
            sample_denorm = dataset.denormalize_coord(sample).cpu().numpy()
            x_coord = sample_denorm[0, :]
            y_coord = sample_denorm[1, :]
            try:
                cl_eval = get_cl(sample_denorm, angle=5)
            except Exception:
                cl_eval = np.nan
            if np.isnan(cl_eval):
                cl_ape = np.nan
                color = "tab:red"
            else:
                cl_ape = np.abs((cl_eval - cl_cond) / cl_cond * 100)
                color = "tab:blue"

            axs[row, col].plot(x_coord, y_coord, color=color)
            axs[row, col].set_title(
                f"CL_in:  {cl_cond:.3f}\nCL_out: {cl_eval:.3f}\nAPE:    {cl_ape:.1f}%", fontsize=12
            )
            axs[row, col].set_aspect("equal", "box")
            # 軸の表示範囲を固定
            axs[row, col].set_xlim(-0.1, 1.1)
            axs[row, col].set_ylim(-0.2, 0.35)
            axs[row, col].tick_params(labelsize=9)
            axs[row, col].grid(True)

        cl_conditioned_list.append(np.array([cl_val] * num_samples_for_each_cl))
        coord_generated_list.append(generated.cpu().numpy())
    plt.tight_layout()
    plt.savefig(sample_plot_path)
    plt.close()
    return sample_plot_path, np.array(cl_conditioned_list), np.array(coord_generated_list)


# def evaluate_generated_samples_from_random_noise_fast(
#     model: torch.nn.Module,
#     diffuser: Diffuser,
#     dataset: AirfoilDataset,
#     num_samples_for_each_cl: int = 10,
#     device: str = "cuda",
#     max_batch: int = 100,
# ):
#     """
#     高速版：全 CL 値 × サンプル数 をまとめてチャンク推論し、
#     各 CL ごとのメトリクス + 全体バッチ指標 + cls_conditioned_array + coords_generated_array を返す。
#     """
#     # 1) CL 値リスト
#     cl_eval_values = np.linspace(0.5, 1.2, 71)
#     K = len(cl_eval_values)

#     # 2) (K×num_samples)×2 のラベル行列を作成
#     cl_list = [np.tile([cl, 0.0], (num_samples_for_each_cl, 1)) for cl in cl_eval_values]
#     cl_array = np.vstack(cl_list)  # shape [K*num_samples, 2]

#     # 3) Tensor 化＆正規化
#     cond = torch.from_numpy(cl_array).float().to(device)  # [N,2]
#     cond_norm = dataset.normalize_cl(cond)  # [N, label_dim]

#     # 4) チャンク単位で生成
#     generated_chunks = []
#     total_N = cond_norm.size(0)
#     for i in tqdm(range(0, total_N, max_batch)):
#         sub = cond_norm[i : i + max_batch]
#         gen = diffuser.generate_from_labels(model, sub, coord_shape=(2, 248))
#         generated_chunks.append(gen.cpu())
#     all_generated = torch.cat(generated_chunks, dim=0)  # [N,2,248]

#     # ─── 全体バッチ指標を一度だけ計算 ──────────────────────────────────
#     all_generated_denorm = dataset.denormalize_coord(all_generated)
#     all_generated_denorm = all_generated_denorm.detach().cpu()
#     generation_diversity_all = compute_generation_diversity(all_generated_denorm)
#     mu_all = compute_mean_deviation(all_generated_denorm)
#     euclid_dist_all = compute_euclid_dist(all_generated_denorm)

#     # 5) CL × サンプル数 の配列に reshape
#     all_generated = all_generated.view(K, num_samples_for_each_cl, 2, 248)  # torch.Tensor

#     # 6) 各 CL ごとのメトリクス計算
#     accum = {
#         "convexity_loss": [],
#         "smoothness_loss": [],
#         "total_turning_angle": [],
#         "smoothness_phi": [],
#         "cl_mse": [],
#         "cl_mape": [],
#         "convergence_raw": [],
#         "convergence_strict": [],
#         "generation_diversity": [],
#         "mu": [],
#         "euclid_dist": [],
#     }

#     for idx, cl_val in tqdm(enumerate(cl_eval_values)):
#         batch = all_generated[idx]  # [num_samples,2,248]
#         cl_cond = np.full(num_samples_for_each_cl, cl_val)  # [num_samples]
#         metrics = evaluate_generated_samples(batch, cl_cond, dataset)
#         (conv_l, smooth_l, angle_l, phi_l, mse, mape, raw_ratio, strict_ratio, diversity, mu, euc) = metrics

#         accum["convexity_loss"].append(conv_l)
#         accum["smoothness_loss"].append(smooth_l)
#         accum["total_turning_angle"].append(angle_l)
#         accum["smoothness_phi"].append(phi_l)
#         accum["cl_mse"].append(mse)
#         accum["cl_mape"].append(mape)
#         accum["convergence_raw"].append(raw_ratio)
#         accum["convergence_strict"].append(strict_ratio)
#         accum["generation_diversity"].append(diversity)
#         accum["mu"].append(mu)
#         accum["euclid_dist"].append(euc)

#     # 7) 結果まとめ
#     summary = {k: float(np.mean(v)) for k, v in accum.items()}
#     summary.update(
#         {
#             "generation_diversity_all": generation_diversity_all,
#             "mu_all": mu_all,
#             "euclid_dist_all": euclid_dist_all,
#         }
#     )

#     # 8) cls_conditioned_array, coords_generated_array の準備
#     cls_conditioned_array = np.repeat(cl_eval_values[:, None], num_samples_for_each_cl, axis=1)
#     coords_generated_array = all_generated.numpy()  # shape [K, num_samples, 2, 248]

#     return summary, cls_conditioned_array, coords_generated_array


# def plot_generated_samples_from_random_noise_fast(
#     model: torch.nn.Module,
#     diffuser: Diffuser,
#     dataset: AirfoilDataset,
#     num_samples_for_each_cl: int = 5,
#     output_dirs: str = "",
#     epoch: int = 0,
#     device: str = "cuda",
#     suffix: str = "",
#     max_batch: int = 100,
# ):
#     """
#     高速版プロット関数：
#     - 全 CL 値 × サンプル数 をチャンク推論して一括生成
#     - 元の plot_generated_samples_from_random_noise と同じ返り値：
#       (sample_plot_path, cls_conditioned_array, coords_generated_array)
#     """
#     assert output_dirs != ""
#     # 保存ファイル名
#     fname = f"samples{suffix}.png" if epoch == 0 else f"samples_epoch_{epoch}{suffix}.png"
#     sample_plot_path = os.path.join(output_dirs, fname)

#     # プロットする CL 値リスト
#     cl_plot_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
#     K = len(cl_plot_values)

#     # 1) 全ラベル行列を作成して正規化
#     #    shape: [K * num_samples_for_each_cl, 1]
#     cl_array = np.repeat(np.array(cl_plot_values)[:, None], num_samples_for_each_cl, axis=1).flatten()
#     cond = torch.from_numpy(cl_array).float().unsqueeze(1).to(device)
#     cond_norm = dataset.normalize_cl(cond)

#     # 2) チャンク単位で一括生成
#     chunks = []
#     N = cond_norm.size(0)
#     for i in tqdm(range(0, N, max_batch)):
#         sub = cond_norm[i : i + max_batch]
#         gen = diffuser.generate_from_labels(model, sub, coord_shape=(2, 248))
#         chunks.append(gen.cpu())
#     all_gen = torch.cat(chunks, dim=0)  # [K*num, 2, 248]

#     # 3) reshape → [K, num_samples, 2, 248]
#     all_gen = all_gen.view(K, num_samples_for_each_cl, 2, 248).numpy()

#     # 4) プロット
#     fig, axs = plt.subplots(num_samples_for_each_cl, K, figsize=(30, 14))
#     for col, cl_val in tqdm(enumerate(cl_plot_values)):
#         for row in range(num_samples_for_each_cl):
#             coord = all_gen[col, row]  # [2,248]
#             sample_denorm = dataset.denormalize_coord(torch.from_numpy(coord)).numpy()
#             x_coord, y_coord = sample_denorm
#             try:
#                 cl_eval = get_cl(sample_denorm, angle=5)
#             except Exception:
#                 cl_eval = np.nan

#             if np.isnan(cl_eval):
#                 color = "tab:red"
#                 cl_ape = np.nan
#             else:
#                 cl_ape = abs((cl_eval - cl_val) / cl_val * 100)
#                 color = "tab:blue"

#             ax = axs[row, col]
#             ax.plot(x_coord, y_coord, color=color)
#             ax.set_title(f"CL_in: {cl_val:.3f}\nCL_out: {cl_eval:.3f}\nAPE: {cl_ape:.1f}%", fontsize=12)
#             ax.set_aspect("equal", "box")
#             ax.set_xlim(-0.1, 1.1)
#             ax.set_ylim(-0.2, 0.35)
#             ax.tick_params(labelsize=9)
#             ax.grid(True)

#     plt.tight_layout()
#     plt.savefig(sample_plot_path)
#     plt.close()

#     # 5) 元関数と同様の戻り値を準備
#     cls_conditioned_array = np.repeat(np.array(cl_plot_values)[:, None], num_samples_for_each_cl, axis=1)
#     coords_generated_array = all_gen  # shape [K, num_samples_for_each_cl, 2, 248]

#     return sample_plot_path, cls_conditioned_array, coords_generated_array


def evaluate_generated_samples_from_random_noise_fast(
    model: torch.nn.Module,
    diffuser: Diffuser,
    dataset: AirfoilDataset,
    num_samples_for_each_cl: int = 10,
    device: str = "cuda",
    max_batch: int = 100,
):
    """
    すべての CL 値 × サンプル数 をまとめてチャンク推論し、
    各 CL ごとのメトリクス ＋ 全体バッチ指標 を計算する高速版。
    """
    # 1) CL 値リスト
    cl_eval_values = np.linspace(0.5, 1.2, 71)
    K = len(cl_eval_values)

    # 2) (K×num_samples)×2 のラベル行列を作成
    cl_list = [np.tile([cl, 0.0], (num_samples_for_each_cl, 1)) for cl in cl_eval_values]
    cl_array = np.vstack(cl_list)  # shape [K*num_samples, 2]

    # 3) Tensor 化＆正規化
    cond = torch.from_numpy(cl_array).float().to(device)  # [N,2]
    cond_norm = dataset.normalize_cl(cond)  # [N, label_dim]

    # 4) チャンク単位で生成
    generated_chunks = []
    total_N = cond_norm.size(0)
    for i in tqdm(range(0, total_N, max_batch)):
        sub = cond_norm[i : i + max_batch]  # [B, ...]
        gen = diffuser.generate_from_labels(model, sub, coord_shape=(2, 248))
        generated_chunks.append(gen.cpu())
    all_generated = torch.cat(generated_chunks, dim=0)  # [N,2,248]

    # ─── 全体バッチ指標を一度だけ計算 ──────────────────────────────────
    all_generated_denorm = dataset.denormalize_coord(all_generated)
    all_generated_denorm = all_generated_denorm.detach().cpu()
    generation_diversity_all = compute_generation_diversity(all_generated_denorm)
    mu_all = compute_mean_deviation(all_generated_denorm)
    euclid_dist_all = compute_euclid_dist(all_generated_denorm)

    # 5) 各 CL ごとに reshape してメトリクス計算
    all_generated = all_generated.view(K, num_samples_for_each_cl, 2, 248)

    accum = {
        "convexity_loss": [],
        "smoothness_loss": [],
        "total_turning_angle": [],
        "smoothness_phi": [],
        "cl_mse": [],
        "cl_mape": [],
        "convergence_raw": [],
        "convergence_strict": [],
        "generation_diversity": [],
        "mu": [],
        "euclid_dist": [],
    }

    for idx, cl_val in tqdm(enumerate(cl_eval_values)):
        batch = all_generated[idx]  # [num_samples,2,248]
        cl_cond = np.full(num_samples_for_each_cl, cl_val)  # 真値リスト
        metrics = evaluate_generated_samples(batch, cl_cond, dataset)
        (conv_l, smooth_l, angle_l, phi_l, mse, mape, raw_ratio, strict_ratio, diversity, mu, euc) = metrics

        accum["convexity_loss"].append(conv_l)
        accum["smoothness_loss"].append(smooth_l)
        accum["total_turning_angle"].append(angle_l)
        accum["smoothness_phi"].append(phi_l)
        accum["cl_mse"].append(mse)
        accum["cl_mape"].append(mape)
        accum["convergence_raw"].append(raw_ratio)
        accum["convergence_strict"].append(strict_ratio)
        accum["generation_diversity"].append(diversity)
        accum["mu"].append(mu)
        accum["euclid_dist"].append(euc)

    # 6) 各 CL ごとの平均 ＋ 全体バッチ指標 をまとめて返却
    summary = {k: float(np.mean(v)) for k, v in accum.items()}
    summary.update(
        {
            "generation_diversity_all": generation_diversity_all,
            "mu_all": mu_all,
            "euclid_dist_all": euclid_dist_all,
        }
    )
    return summary


def plot_generated_samples_from_random_noise_fast(
    model: torch.nn.Module,
    diffuser: Diffuser,
    dataset: AirfoilDataset,
    num_samples_for_each_cl: int = 5,
    output_dirs: str = "",
    epoch: int = 0,
    device: str = "cuda",
    suffix: str = "",
    max_batch: int = 100,
):
    """
    すべての CL 値 × サンプル数 をまとめて生成し、一括でプロットする高速版。
    """
    assert output_dirs != ""
    fname = f"samples{suffix}.png" if epoch == 0 else f"samples_epoch_{epoch}{suffix}.png"
    sample_plot_path = os.path.join(output_dirs, fname)

    # 1) プロットする CL 値
    cl_plot_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    K = len(cl_plot_values)

    # 2) ラベル行列作成 → 正規化
    cl_array = np.repeat(np.array(cl_plot_values)[:, None], num_samples_for_each_cl, axis=1).flatten()
    cond = torch.from_numpy(cl_array).float().unsqueeze(1).to(device)  # [K*num,1]
    cond_norm = dataset.normalize_cl(cond)

    # 3) チャンク推論
    chunks = []
    N = cond_norm.size(0)
    for i in range(0, N, max_batch):
        sub = cond_norm[i : i + max_batch]
        gen = diffuser.generate_from_labels(model, sub, coord_shape=(2, 248))
        chunks.append(gen.cpu())
    all_gen = torch.cat(chunks, dim=0)  # [K*num,2,248]
    all_gen = all_gen.view(K, num_samples_for_each_cl, 2, 248).numpy()

    # 4) プロット
    fig, axs = plt.subplots(num_samples_for_each_cl, K, figsize=(30, 14))
    for col, cl_val in tqdm(enumerate(cl_plot_values)):
        for row in range(num_samples_for_each_cl):
            coord = all_gen[col, row]  # [2,248]
            sample_denorm = dataset.denormalize_coord(torch.from_numpy(coord)).numpy()
            x_coord, y_coord = sample_denorm
            try:
                cl_eval = get_cl(sample_denorm, angle=5)
            except Exception:
                cl_eval = np.nan

            if np.isnan(cl_eval):
                color = "tab:red"
                cl_ape = np.nan
            else:
                cl_ape = abs((cl_eval - cl_val) / cl_val * 100)
                color = "tab:blue"

            ax = axs[row, col]
            ax.plot(x_coord, y_coord, color=color)
            ax.set_title(f"CL_in: {cl_val:.3f}\nCL_out: {cl_eval:.3f}\nAPE: {cl_ape:.1f}%", fontsize=12)
            ax.set_aspect("equal", "box")
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.2, 0.35)
            ax.tick_params(labelsize=9)
            ax.grid(True)

    plt.tight_layout()
    plt.savefig(sample_plot_path)
    plt.close()
    return sample_plot_path

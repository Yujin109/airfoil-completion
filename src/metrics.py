import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from data import AirfoilDataset
from diffusion import Diffuser
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


def smoothness_loss(coord):
    filtered = running_average_filter(coord, kernel_size=9)
    return np.mean((coord - filtered) ** 2)


def cl_loss_function(cl_conditioned, cl_evaluated):
    return np.mean((cl_conditioned - cl_evaluated) ** 2)


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


def evaluate_generated_samples(samples, conditioned_cls, dataset):
    convexity_losses, smoothness_losses, cl_losses = [], [], []
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
            cl_val = cl_loss_function(conditioned_cls[i], cl_eval)
            converged = True
            convergence_raw += 1
        else:
            cl_val = np.nan
            converged = False
        convexity_losses.append(convexity_loss(denorm))
        smoothness_losses.append(smoothness_loss(denorm))
        cl_losses.append(cl_val)
        convergence_raw += converged
        if (convexity_losses[-1] < 0.1 and smoothness_losses[-1] < 0.1 and cl_val < 0.1) and converged:
            convergence_strict += 1
    return (
        np.nanmean(convexity_losses),
        np.nanmean(smoothness_losses),
        np.nanmean([v for v in cl_losses if not np.isnan(v)]),
        convergence_raw / total,
        convergence_strict / total,
        compute_generation_diversity(samples),
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
    cl_loss_list = []
    convergence_ratios_raw = []
    convergence_ratios_strict = []
    diversity_list = []

    for cl_val in tqdm(cl_eval_values, total=len(cl_eval_values)):
        cond = torch.tensor([[cl_val, 0.0]] * num_samples_for_each_cl, dtype=torch.float32, device=device)
        cond_norm = dataset.normalize_cl(cond)
        generated = diffuser.generate_from_labels(model, cond_norm, coord_shape=(2, 248))

        cl_conditioned = np.array([cl_val] * num_samples_for_each_cl)
        conv_l, smooth_l, cl_l, conv_ratio_raw, conv_ratio_strict, diversity = evaluate_generated_samples(
            generated, cl_conditioned, dataset
        )
        convexity_list.append(conv_l)
        smoothness_list.append(smooth_l)
        cl_loss_list.append(cl_l)
        convergence_ratios_raw.append(conv_ratio_raw)
        convergence_ratios_strict.append(conv_ratio_strict)
        diversity_list.append(diversity)

    return {
        "convexity_loss_mean": np.mean(convexity_list),
        "smoothness_loss_mean": np.mean(smoothness_list),
        "cl_loss_mean": np.mean(cl_loss_list),
        "cl_convergence_ratio_raw": np.mean(convergence_ratios_raw),
        "cl_convergence_ratio_strict": np.mean(convergence_ratios_strict),
        "generation_diversity_mean": np.mean(diversity_list),
    }


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

    cl_plot_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    fig, axs = plt.subplots(num_samples_for_each_cl, len(cl_plot_values), figsize=(20, 12))
    for col, cl_val in tqdm(enumerate(cl_plot_values), total=len(cl_plot_values)):
        cond = torch.tensor([[cl_val, 0.0]] * num_samples_for_each_cl, dtype=torch.float32, device=device)
        cond_norm = dataset.normalize_cl(cond)
        generated = diffuser.generate_from_labels(model, cond_norm, coord_shape=(2, 248))
        for row in range(num_samples_for_each_cl):
            sample = generated[row].detach().cpu()
            sample_denorm = dataset.denormalize_coord(sample).cpu().numpy()
            x_coord = sample_denorm[0, :]
            y_coord = sample_denorm[1, :]
            try:
                cl_eval = get_cl(sample_denorm, angle=5)
            except Exception:
                cl_eval = np.nan
            conv_loss = convexity_loss(sample_denorm)
            axs[row, col].plot(x_coord, y_coord)
            axs[row, col].set_title(f"CL: {cl_eval:.2f}\nConv: {conv_loss:.3f}", fontsize=8)
            axs[row, col].tick_params(labelsize=6)
            axs[row, col].grid(True)
    plt.tight_layout()
    plt.savefig(sample_plot_path)
    plt.close()
    return sample_plot_path

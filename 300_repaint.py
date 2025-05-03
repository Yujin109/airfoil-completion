import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    CLUSTERS,
    DIFFUSION_PARAMS,
    FIG_SIZE,
    GUIDANCE_SCALE,
    JUMP_LENGTH,
    MASK_TYPE,
    MODEL_NAME,
    NUM_RESAMPLING,
    REPAINT_DIR,
    SAMPLES_PER_CLUSTER,
    SEED,
    WEIGHT_PATH,
)
from src.data import AirfoilDataset
from src.diffusion import RePaintDiffuser
from src.models.model_registry import MODEL_REGISTRY
from src.xfoil_utils import get_cl


def mask_manual(pattern="upper_center", length=248, missing=20):
    if pattern == "m_upcenter":
        m = np.ones(length, dtype=bool)
        start = (length - missing) // 2
        m[start : start + missing] = False
    elif pattern == "m_head":
        m = np.ones(length, dtype=bool)
        m[:missing] = False
    elif pattern == "m_tail":
        m = np.ones(length, dtype=bool)
        m[-missing:] = False
    return m


def plot_inpaint(orig, rep, mask, ax, title=""):
    m = mask
    ax.plot(np.ma.array(orig[0], mask=~m), np.ma.array(orig[1], mask=~m), label="known")
    ax.plot(np.ma.array(rep[0], mask=m), np.ma.array(rep[1], mask=m), label="inpaint")
    ax.set_title(title, fontsize=8)
    ax.grid(True)


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
    coords_batch = dataset.coords_tensor[selected].to(device)
    cls_batch = dataset.cls_tensor[selected].to(device)
    masks = torch.stack([torch.from_numpy(mask_manual(pattern=MASK_TYPE)) for _ in selected], dim=0).to(device)

    # run RePaint inpainting
    repainted = diffuser.repaint(model, coords_batch, masks, cls_batch)

    # denormalize for plotting
    orig_denorm = dataset.denormalize_coord(coords_batch).cpu().numpy()
    rep_denorm = dataset.denormalize_coord(repainted).cpu().numpy()

    # compute CL on inpainted
    cl_out_list = []
    for sample in rep_denorm:
        cl_out_list.append(get_cl(sample))

    # plot results
    fig, axes = plt.subplots(SAMPLES_PER_CLUSTER, len(CLUSTERS), figsize=FIG_SIZE)
    for i, idx in enumerate(selected):
        row = i // len(CLUSTERS)
        col = i % len(CLUSTERS)
        ax = axes[row, col]
        title = f"Cl_in={original_cls[idx]:.3f}\nCl_out={cl_out_list[i]:.3f}"
        plot_inpaint(orig_denorm[i], rep_denorm[i], masks[i].cpu().numpy(), ax, title)
    plt.tight_layout()

    # save figure
    date_str = datetime.datetime.now().strftime("%y%m%d")
    fname = f"{date_str}_R{NUM_RESAMPLING}_J{JUMP_LENGTH}_M{MASK_TYPE}.png"
    fpath = os.path.join(REPAINT_DIR, fname)
    fig.savefig(fpath, dpi=300)
    print(f"[Saved sample images] {fpath}")

    # compute metrics

    metrics = dict()
    # MSE on cls
    cl_se_list = []
    for i, idx in enumerate(selected):
        cl_out = cl_out_list[i]
        cl_in = original_cls[idx]
        cl_se_list.append((cl_out - cl_in) ** 2)
    metrics["cl_loss_mean"] = np.mean(cl_se_list)

    # .txtファイルとして保存
    with open(os.path.join(REPAINT_DIR, f"{date_str}_R{NUM_RESAMPLING}_J{JUMP_LENGTH}_M{MASK_TYPE}.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()

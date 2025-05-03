import gc
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import wandb
from config import (
    B1,
    B2,
    BATCH_SIZE,
    DIFFUSION_PARAMS,
    EVAL_NUM_SAMPLES_FOR_EACH_CL,
    EVALUATION_INTERVAL,
    EVALUATION_METRICS_DIR,
    EXECUTION_NAME,
    GUIDANCE_SCALE,
    INITIAL_LR,
    MODEL_INFO_DIR,
    MODEL_NAME,
    NUM_EPOCHS,
    OUTPUT_MODE,
    P_UNCOND,
    PLOT_NUM_SAMPLES_FOR_EACH_CL,
    PROJECT_NAME,
    SAMPLES_DIR,
    TRAINING_METRICS_DIR,
    WEIGHTS_DIR,
)
from src.data import get_dataloader
from src.diffusion import Diffuser
from src.metrics import (
    evaluate_generated_samples_from_random_noise,
    plot_generated_samples_from_random_noise,
)
from src.models.model_registry import MODEL_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    wandb.init(
        project=PROJECT_NAME,
        name=EXECUTION_NAME,
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "initial_lr": INITIAL_LR,
            "diffusion": DIFFUSION_PARAMS,
            "output_mode": OUTPUT_MODE,
            "guidance_scale": GUIDANCE_SCALE,
            "p_uncond": P_UNCOND,
            "evaluation_interval": EVALUATION_INTERVAL,
        },
    )
    wandb.config.update({"device": str(DEVICE)})

    for d in [MODEL_INFO_DIR, TRAINING_METRICS_DIR, EVALUATION_METRICS_DIR, SAMPLES_DIR, WEIGHTS_DIR]:
        os.makedirs(d, exist_ok=True)

    loader, dataset = get_dataloader(BATCH_SIZE)
    model_cls = MODEL_REGISTRY[MODEL_NAME]
    model = model_cls().to(DEVICE)
    diffuser = Diffuser(
        num_timesteps=DIFFUSION_PARAMS["num_timesteps"],
        beta_start=DIFFUSION_PARAMS["beta_start"],
        beta_end=DIFFUSION_PARAMS["beta_end"],
        beta_schedule=DIFFUSION_PARAMS["beta_schedule"],
        cosine_s=DIFFUSION_PARAMS["cosine_s"],
        device=DEVICE,
        guidance_scale=GUIDANCE_SCALE,
        model_name=MODEL_NAME,
    )
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, betas=(B1, B2))

    # Save model stats
    param_count = sum(p.numel() for p in model.parameters())
    with open(os.path.join(MODEL_INFO_DIR, "param_count.txt"), "w") as f:
        f.write(str(param_count))

    train_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        losses = []
        for x, cl in tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False):
            x, cl = x.to(DEVICE), cl.to(DEVICE)
            t = torch.randint(low=1, high=diffuser.num_timesteps + 1, size=(x.size(0),), device=DEVICE)
            x_t, noise = diffuser.add_noise(x, t)
            mask = torch.rand(x.size(0), device=DEVICE) < P_UNCOND
            noise_pred = model(x_t, cl, t.unsqueeze(1), uncond_mask=mask)
            loss = torch.nn.MSELoss()(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        wandb.log({"train_loss": avg_loss, "epoch": epoch})

        # Learning rate schedule
        lr = INITIAL_LR * ((1 - epoch / NUM_EPOCHS) ** 0.4)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation block
        if epoch % EVALUATION_INTERVAL == 0:
            print(f"--- Evaluation at epoch {epoch} ---")
            model.eval()
            with torch.no_grad():
                # Save training metrics
                culcurated_metrics = evaluate_generated_samples_from_random_noise(
                    model, diffuser, dataset, num_samples_for_each_cl=EVAL_NUM_SAMPLES_FOR_EACH_CL, device=DEVICE
                )
                wandb.log({**culcurated_metrics, "epoch": epoch})
                # .txtファイルとして保存
                with open(os.path.join(EVALUATION_METRICS_DIR, f"epoch_{epoch}.txt"), "w") as f:
                    for key, value in culcurated_metrics.items():
                        f.write(f"{key}: {value}\n")

                # Save generated samples
                sample_plot_path = plot_generated_samples_from_random_noise(
                    model,
                    diffuser,
                    dataset,
                    num_samples_for_each_cl=PLOT_NUM_SAMPLES_FOR_EACH_CL,
                    device=DEVICE,
                    output_dirs=SAMPLES_DIR,
                    epoch=epoch,
                )
                print(f"生成サンプルプロット保存: {sample_plot_path}")
                wandb.log({"generated_samples": wandb.Image(sample_plot_path), "epoch": epoch})

                # Save intermediate model weights
                intermediate_model_path = os.path.join(WEIGHTS_DIR, f"model_weights_epoch_{epoch}.pth")
                torch.save(model.state_dict(), intermediate_model_path)
                print(f"Epoch {epoch}: 中間モデルの重み保存: {intermediate_model_path}")
                wandb.save(intermediate_model_path)

            # 評価ブロック終了後、ガベージコレクションと GPU キャッシュの解放
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save final model
    final_model_path = os.path.join(WEIGHTS_DIR, "final_model_weights.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"学習終了．最終モデルの重み保存: {final_model_path}")
    wandb.save(final_model_path)
    wandb.finish()


if __name__ == "__main__":
    main()

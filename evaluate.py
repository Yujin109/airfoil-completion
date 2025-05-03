import datetime
import os

import numpy as np
import torch
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    DIFFUSION_PARAMS,
    EVAL_NUM_SAMPLES_FOR_EACH_CL,
    EVALUATION_METRICS_DIR,
    GUIDANCE_SCALE,
    MODEL_NAME,
    PLOT_NUM_SAMPLES_FOR_EACH_CL,
    SAMPLES_DIR,
    WEIGHTS_DIR,
)
from src.data import get_dataloader
from src.diffusion import Diffuser
from src.metrics import (
    evaluate_generated_samples,
    evaluate_generated_samples_from_random_noise,
    plot_generated_samples_from_random_noise,
)
from src.models.model_registry import MODEL_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    for d in [EVALUATION_METRICS_DIR, SAMPLES_DIR, WEIGHTS_DIR]:
        os.makedirs(d, exist_ok=True)

    _, dataset = get_dataloader(BATCH_SIZE)
    model_cls = MODEL_REGISTRY[MODEL_NAME]
    model = model_cls().to(DEVICE)
    ckpt = os.path.join(WEIGHTS_DIR, "final_model_weights.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    diffuser = Diffuser(**DIFFUSION_PARAMS, device=DEVICE, guidance_scale=GUIDANCE_SCALE, model_name=MODEL_NAME)

    with torch.no_grad():

        # datetimeを使い、今日の日付を取得し、250501の形式に変換
        today = datetime.datetime.now().strftime("%y%m%d")

        # Save training metrics
        culcurated_metrics = evaluate_generated_samples_from_random_noise(
            model, diffuser, dataset, num_samples_for_each_cl=EVAL_NUM_SAMPLES_FOR_EACH_CL, device=DEVICE
        )
        # .txtファイルとして保存
        with open(os.path.join(EVALUATION_METRICS_DIR, f"final_{today}.txt"), "w") as f:
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
            epoch=0,
            suffix=f"_final_{today}",
        )
        print(f"生成サンプルプロット保存: {sample_plot_path}")


if __name__ == "__main__":
    main()

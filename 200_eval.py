import datetime
import os

import numpy as np
import torch

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
    evaluate_generated_samples_from_random_noise,
    evaluate_generated_samples_from_random_noise_fast,
    plot_generated_samples_from_random_noise,
    plot_generated_samples_from_random_noise_fast,
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

        # datetimeを使い、今日の時刻を取得し、2505010157Sの形式に変換
        # 例: 2023年10月5日15時57分 -> 2310051557
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")

        # Save training metrics
        culcurated_metrics = evaluate_generated_samples_from_random_noise_fast(
            model, diffuser, dataset, num_samples_for_each_cl=EVAL_NUM_SAMPLES_FOR_EACH_CL, device=DEVICE
        )
        # .txtファイルとして保存
        with open(os.path.join(EVALUATION_METRICS_DIR, f"final_{timestamp}.txt"), "w") as f:
            for key, value in culcurated_metrics.items():
                f.write(f"{key}: {value}\n")
        # # .npzファイルとして保存
        # np.savez(
        #     os.path.join(EVALUATION_METRICS_DIR, f"final_{timestamp}_generated_data_for_metrics.npz"),
        #     cls_conditioned=cls_conditioned_array,
        #     coords_generated=coords_generated_array,
        # )

        # Save generated samples
        sample_plot_path = plot_generated_samples_from_random_noise_fast(
            model,
            diffuser,
            dataset,
            num_samples_for_each_cl=PLOT_NUM_SAMPLES_FOR_EACH_CL,
            device=DEVICE,
            output_dirs=SAMPLES_DIR,
            epoch=0,
            suffix=f"_final_{timestamp}",
        )
        # np.savez(
        #     os.path.join(SAMPLES_DIR, f"final_{timestamp}_generated_data_for_plot.npz"),
        #     cls_conditioned=cls_conditioned_array,
        #     coords_generated=coords_generated_array,
        # )
        print(f"生成サンプルプロット保存: {sample_plot_path}")


if __name__ == "__main__":
    main()

import os

from dotenv import load_dotenv

load_dotenv()

# Training settings -------------------------------------

# Project settings
PROJECT_NAME = "airfoil_diffusion"
# Model selection (refer to src/models/model_registry.py for available models)

# EXECUTION_NAME = "250426-001:CFG(all-zero)+pos-enc+beta_end0.05"  # OK+phi
# MODEL_NAME = "NewBaseConditionalUNet_CFG_zero"

# EXECUTION_NAME = "250424-002:CFG(all-zero-false)+pos-enc+cos-schedule"  # OK+phi
# MODEL_NAME = "NewBaseConditionalUNet"

# EXECUTION_NAME = "250416-001:Baseline"  # OK+phi
# MODEL_NAME = "BaselineConditionalUNet"

# EXECUTION_NAME = "250424-001:CFG(all-zero-false)+pos-enc+beta_end0.05"  # OK+phi
# MODEL_NAME = "NewBaseConditionalUNet"

# EXECUTION_NAME = "250418-001:CFG(all-zero)"  # OK+phi
# MODEL_NAME = "BaselineConditionalUNet_CFG_zero"

# EXECUTION_NAME = "250418-002:CFG(none)"  # OK+phi
# MODEL_NAME = "BaselineConditionalUNet_CFG_none"

# EXECUTION_NAME = "250422-001:ResidualUnet+CFG(none)"  # OK+phi
# MODEL_NAME = "BaselineResUNet_2layer_CFG_none"

# EXECUTION_NAME = "250422-002:CFG(all-zero-false)+pos-enc"  # OK+phi
# MODEL_NAME = "NewBaseConditionalUNet"

# EXECUTION_NAME = "250423-001:ResUnet(3-layer)+CFG(none)"  # OK+phi
# MODEL_NAME = "BaselineResUNet_3layer_CFG_none"

EXECUTION_NAME = "250425-001:ResUnet(3-layer)+CFG(all-zero-false)+pos-enc+beta_end0.05"  # OK+phi 悪すぎる???
MODEL_NAME = "NewBaseResUNet_3layer"

# EXECUTION_NAME = "250430-001:CFG(none)+pos-enc+beta_end0.05"  # OK+phi
# MODEL_NAME = "NewBaseConditionalUNet_CFG_none"

# EXECUTION_NAME = "250419-001:CFG(none)+timestep1000+epoch5000"  # OK+phi
# MODEL_NAME = "BaselineConditionalUNet_CFG_none"

# EXECUTION_NAME = "250418-003:CFG(none)+timestep1000"  # OK+phi
# MODEL_NAME = "BaselineConditionalUNet_CFG_none"


WEIGHT_PATH = f"./results/{EXECUTION_NAME}/weights/final_model_weights.pt"

# Training hyperparameters
NUM_EPOCHS = 2000
BATCH_SIZE = 32
INITIAL_LR = 2e-4
B1 = 0.0
B2 = 0.9

# Diffusion process parameters
DIFFUSION_PARAMS = {
    "num_timesteps": 500,
    "beta_start": 1e-4,  # 1e-4
    "beta_end": 5e-2,  # 5e-2
    "beta_schedule": "linear",  # ["linear", "cosine"]
    "cosine_s": 0.008,  # 0.008 for cosine schedule
}

# Output settings
OUTPUT_MODE = "conv3x3"  # ["co nv3x3", "conv1x1", "fc", "fc_nn"]
GUIDANCE_SCALE = 2.0  # 2.0 (CFG) or 1.0 (normal conditional) or 0.0 (unconditional)
P_UNCOND = 0.1
EVALUATION_INTERVAL = 500
EVAL_NUM_SAMPLES_FOR_EACH_CL = 10
PLOT_NUM_SAMPLES_FOR_EACH_CL = 5

# Dataset prefix
DATASET_PREFIX = "NACA&Joukowski"

# Directories for results
RESULTS_DIR = os.path.join("results", EXECUTION_NAME)
MODEL_INFO_DIR = os.path.join(RESULTS_DIR, "model_info")
TRAINING_METRICS_DIR = os.path.join(RESULTS_DIR, "training_metrics")
EVALUATION_METRICS_DIR = os.path.join(RESULTS_DIR, "evaluation_metrics")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")
WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")


# RePaint settings ----------------------------------------

# RePaint settings
NUM_RESAMPLING = 1
JUMP_LENGTH = 1

# Dataset and inpainting settings
CLUSTERS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.2)]
SAMPLES_PER_CLUSTER = 5
SEED = 42

# Output paths
REPAINT_DIR = os.path.join(RESULTS_DIR, "repaint")
MASK_TYPE = "m_upcenter"  # ["m_upcenter", "m_head", "m_tail"]

# plot settings
FIG_SIZE = (20, 12)

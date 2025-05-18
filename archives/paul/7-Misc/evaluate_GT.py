import sys
import numpy as np
import torch

sys.path.append("./03 CODE/2 DDPM")
from DDPM import setup_DDPM

sys.path.append("./03 CODE/6 Utils")
from Util_Lib import get_device

sys.path.append("./03 CODE/8 X-Foil/Python_DIR/")
from xfoil_runner import eval_CL_XFOIL

n_samples = 160

HP = {
    "Comment": "1x1, septic approx., all kernelsize 15,7,3", 

    # Model
    "Model_Architecture": "FC_FF",  # Model architecture to be used
    "n_T": 1250,
    "betas": torch.tensor((1E-4, 2E-2)),
    "input_features": 496,                  # number of input features
    "flt_half_support": 4,                  # filter support size

    # Don't change these parameters, they are evaluated after model creation.
    "N-Parameters": 0,                      # Number of model parameters, evaluate after model creation
    "Model-Size": 0,                        # Model size in MB, evaluate after model creation
    "Summary": "",                          # Summary of the model architecture, evaluate after model creation
    
    # Training
    "n_epoch": 10000,
    "n_restart": 1,                                         # Number of restarts if the model is trained from prior state
    "initial_lr": 1E-4,                                     # Initial learning rate 5E-4, UNets 2E-4
    "final_lr": 0.0,                                        # Final learning rate 
    "alpha": 0.1,                                           # Powerlaw exponent of learning rate decay
    "LR_Eval_Interval": 25,                           # Interval for evaluation of the learning rate

    "diff_loss_W": 1.0,                     # weight of the diffusion loss in the total loss
    "cl_loss_W": 0.0,                       # weight of the CL loss in the total loss
    "cnvxty_loss_W": 1E-4,                   # weight of the convexity loss in the total loss
    "smthnss_loss_W": 0.0,                  # weight of the smoothness loss in the total loss   
    
    # Data Loader
    "batch_size": 32,                       # number of samples per batch
    "shuffle": True,                        # shuffle data before loading
    "num_workers": 0,                       # number of workers for data loading, multi-process data loading
    "drop_last": True,                      # drop last batch if size is less than batch_size
    "pin_memory": True,                     # faster data transfer to GPU
    "geometry_data": "./03 CODE/4 Data/normalized_NandJ_coords.npz", # Geometry regularized, Feature space normalized, i.e., ~ N(0, 1)
    # "geometry_data": "./03 CODE/4 Data/regularized_NandJ_coords.npz", # Geometry regularized, i.e., zero centered and unit cord length
    # "geometry_data": "./03 CODE/4 Data/Yonekura/standardized_NandJ_coords.npz",
    "performance_data": "./03 CODE/4 Data/Yonekura/standardized_NandJ_perfs.npz",
    
    # General
    "EMA_Factor": 0.95,                      # Exponential moving average factor for data tracking
    "Eval_Gradient_Norm": False,             # Evaluate gradient norm, comes roughly with 20% overhead
    "Eval_Epoch_Interval": 200,              # Interval for evaluation of the model
    "Track_Gradients": False,                # Track gradients in tensorboard
    "Track_Parameters": False,               # Track parameters in tensorboard
    "save_model": True,                      # Save model state after training
    "Identifier": ""                        # Identifier for the model, automatically generated
}

coords = np.load("./03 CODE/4 Data/NandJ_coords_V2.npz")
CLs = np.load("./03 CODE/4 Data/NandJ_perfs_V2.npz")

# Check if GPU is available
device = get_device()
print(f"Using {device} device.")
HP["device"] = device

# Setup and dummy DDPM model
ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs)

cnvxty_loss = []
smthns_loss = []
cl_loss = []
convergence_ratio = []

# create random index permutation for the samples 
idx = np.random.permutation(coords['arr_0'].shape[0])

for u in range(n_samples):
    i = idx[u]
    print(f"Sample {u+1}/{n_samples}, Index: {i}", end="")

    geom = ddpm.renormalize(torch.tensor(coords['arr_0'][i]).float().unsqueeze(0).to(device))
    cl_trgt = (CLs['arr_0'][i] * CLs['arr_2']) + CLs['arr_1']

    cnvxty_loss.append(ddpm.convexity_loss(geom))
    smthns_loss.append(ddpm.smoothness_loss(geom))
    
    cl, cd = eval_CL_XFOIL(geom.detach().cpu().numpy(), viscous=True)
             
    # if cl is nan
    if np.isnan(cl).any():
        convergence_ratio.append(0)
        #cl_loss.append(float("NAN"))
    else:
        convergence_ratio.append(1)
        cl_loss.append((cl - cl_trgt)**2)

    print(" CL: ", cl, " Target: ", cl_trgt, " Loss: ", cl_loss[-1])
        

# Calculate mean values
# Ensure tensors are moved to CPU before converting to NumPy arrays for mean calculation
mean_cnvxty_loss = np.mean([loss.cpu().numpy() for loss in cnvxty_loss])
mean_smthns_loss = np.mean([loss.cpu().numpy() for loss in smthns_loss])
mean_cl_loss = np.mean(cl_loss)
mean_convergence_ratio = np.mean(convergence_ratio) 

# Print to console
print(f"Convexity Loss: {mean_cnvxty_loss:.4f}")
print(f"Smoothness Loss: {mean_smthns_loss:.8f}")
print(f"CL Loss: {mean_cl_loss:.6f}")
print(f"Convergence Ratio: {mean_convergence_ratio:.4f}")


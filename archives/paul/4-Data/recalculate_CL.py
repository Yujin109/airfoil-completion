import sys
import torch
import numpy as np

sys.path.append("./03 CODE/2 DDPM")
from DDPM import setup_DDPM

sys.path.append("./03 CODE/6 Utils")
from Util_Lib import get_device

sys.path.append("./03 CODE/8 X-Foil/Python_DIR/")
from xfoil_runner import eval_CL_XFOIL

HP = {
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


ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs)

cl = []
cl_loss = []
cd = []
P1 = []
P2 = []
P3 = []
convergence_ratio = []
indx_list = []

n_samples = coords["arr_0"].shape[0]

for i in range(n_samples):
    print(f"Sample {i+1}/{n_samples}", end="")
    
    geom = ((coords["arr_0"][i] * coords["arr_2"]) + coords["arr_1"])
    geom = np.reshape(geom, (2, HP["input_features"]//2))
    # add batch dimension
    geom = np.expand_dims(geom, axis=0)

    cl_temp, cd_temp = eval_CL_XFOIL(geom, viscous=True, timeout=5)

    P1_temp, P2_temp, P3_temp = ddpm.roughness_loss(geom)

    # convert to numpy array
    P1_temp = P1_temp.detach().cpu().numpy()
    P2_temp = P2_temp.detach().cpu().numpy()
    P3_temp = P3_temp.detach().cpu().numpy()

    P1.append(P1_temp)
    P2.append(P2_temp)
    P3.append(P3_temp)

    # convert to scalar
    cl_temp = cl_temp[0]
    cd_temp = cd_temp[0]
    
    # if cl is nan
    if np.isnan(cl_temp) or cd_temp > 2 or cd_temp < 0.00001:
        cl.append(0)
        cd.append(0)
        convergence_ratio.append(0)
        indx_list.append(i)
    elif cd_temp > 2 or cd_temp < 0.00001: #arifoil stalled
        cl.append(cl_temp)
        cd.append(cd_temp)
        convergence_ratio.append(0)
        indx_list.append(i)
    else:
        cl.append(cl_temp)
        cl_loss.append((cl_temp - (CLs["arr_0"][i] * CLs["arr_2"] + CLs["arr_1"]))**2)
        cd.append(cd_temp)  
        convergence_ratio.append(1)

    print(f" CL: {cl_temp:.4f}, CD: {cd_temp:.4f}")

    if i == 3 and False:
        break

# calculate the mean convergence ratio
mse_cl = np.mean(cl_loss)
mean_convergence_ratio = np.mean(convergence_ratio)
mean_P1 = np.mean(P1)
mean_P2 = np.mean(P2)
mean_P3 = np.mean(P3)

print(f"Mean Convergence Ratio: {mean_convergence_ratio:.4f}")
print(f"MSE CL: {mse_cl:.9f}")
print(f"Mean P1: {mean_P1:.3f}")
print(f"Mean P2: {mean_P2:.7f}")
print(f"Mean P3: {mean_P3:.9f}")

# Ensure its a numpy array
indx_list = np.array(indx_list).flatten()

# remove the samples that did not converge
if False:
    cl = np.delete(cl, indx_list, axis=0)
    cd = np.delete(cd, indx_list, axis=0)
    print(f"Removed {len(indx_list)} samples that did not converge or stall.")

# calculate mean and standard deviation of the cl values
cl_mean = np.mean(cl)
cl_std = np.std(cl)

print(f"Mean CL: {cl_mean:.4f}")
print(f"Std Dev CL: {cl_std:.4f}")

# normalize the cl values
if True:
    cl = (cl - cl_mean) / cl_std

# save the cl values and the cd values
if False:
    np.savez("./03 CODE/4 Data/recalculated_standardized_NandJ_perfs.npz", cl, cl_mean, cl_std)
    np.savez("./03 CODE/4 Data/recalculated_NandJ_CDs.npz", cd)
    np.savez("./03 CODE/4 Data/index_list.npz", indx_list)

# load airfoil data and adjust to index list
# ...
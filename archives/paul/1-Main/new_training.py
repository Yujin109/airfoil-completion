import sys
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
# Shell command to start tensorboard: tensorboard --logdir "./04 RUNS/1 TB Logs/" --reload_multifile=true
#                                     tensorboard --logdir "./06 REMOTE/TMP/1 TB Logs/" --reload_multifile=true

# import class from different file
sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/3 Models")
sys.path.append("./03 CODE/4 Data")
sys.path.append("./03 CODE/5 Visualization/Visualize Airfoil Profile")
sys.path.append("./03 CODE/6 Utils")

from DDPM import setup_DDPM
from Data_Lib import Custom_Dataset, get_data_loader
from visualize_profiles_tiled import visualize_airfoils_tiled
from Util_Lib import get_device, get_datetime_str
from Training_Loop import train_DDPM

""" 
TODO:
set target cls as hyperparameters and skip the n_Variation variable in all the sampling functions instead determine the numver of samples from the length of the target cl list
exponential learning rate decay, better? maybe just two linear functions connected by knee point
Resample airfoils to 256 points, implement embedding differently
save sample (evolution) to folder not temporary

check if modifications to prolonged sampling are correct, modifiy such that prolonge samlpling is done, i.e., prolonge_sample uses as input geom from sample function
check implementations of smoothness and convexity loss, check if they are implemented correctly, and maybe redo cleanly

log training time explicitly
rename: if output is not renormalized x, else geom
check if eval and train mode are set correctly

Write first letter of function arguments in capital letters, pass lower case arguments to functions

-- Check implementation completely through
-- check for tensor use
-- Store Coords and cl mu and std such that no reshaping is necessary -- renormalize data function, DDPM class

fix reference HP in DDPM class
WHAT ABout batch normalization?
"""


# Create dictionary holding hyperparameters and settings
HP = {
    "Comment": "1x1, 2xRes conv. 64 CHN; KRNL 7,5; DLT 1,1; Data V2, EXP LR after 1000; Cosine Schedule, no add. loss", 

    # Model
    "Model_Architecture": "UNet_Res",  # Model architecture to be used
    "n_T": 1250,
    "betas": torch.tensor((1E-4, 2E-2)),
    "input_features": 496,                  # number of input features
    "flt_half_support": 4,                  # filter support size
    "schedule_type": "cos",         # Learning rate schedule type, options: linear, cos

    # Don't change these parameters, they are evaluated after model creation.
    "N-Parameters": 0,                      # Number of model parameters, evaluate after model creation
    "Model-Size": 0,                        # Model size in MB, evaluate after model creation
    "Summary": "",                          # Summary of the model architecture, evaluate after model creation
    
    # Training
    "n_epoch": 16000,
    "n_restart": 1,                                         # Number of restarts if the model is trained from prior state
    "initial_lr": 1E-4,                                     # Initial learning rate 5E-4, UNets 2E-4
    "final_lr": 0.0,                                        # Final learning rate 
    "alpha": 0.1,                                           # Powerlaw exponent of learning rate decay
    "LR_Eval_Interval": 25,                           # Interval for evaluation of the learning rate

    "diff_loss_W": 1.0,                     # weight of the diffusion loss in the total loss
    "cl_loss_W": 0.0,                       # weight of the CL loss in the total loss
    "cnvxty_loss_W": 0.0,                   # weight of the convexity loss in the total loss
    "smthnss_loss_W": 0.0,                  # weight of the smoothness loss in the total loss
    "rougnss_loss_W": 0.0,                   # weight of the roughness loss in the total loss

    "probabilistic_geom_loss": False,       # Use probabilistic geometry loss
    
    # Data Loader
    "batch_size": 32,                       # number of samples per batch
    "shuffle": True,                        # shuffle data before loading
    "num_workers": 0,                       # number of workers for data loading, multi-process data loading
    "drop_last": True,                      # drop last batch if size is less than batch_size
    "pin_memory": True,                     # faster data transfer to GPU
    "geometry_data": "./03 CODE/4 Data/NandJ_coords_V2.npz", # Geometry regularized, Feature space normalized, only converging airfoils
    # "geometry_data": "./03 CODE/4 Data/normalized_NandJ_coords.npz", # Geometry regularized, Feature space normalized, i.e., ~ N(0, 1)
    # "geometry_data": "./03 CODE/4 Data/regularized_NandJ_coords.npz", # Geometry regularized, i.e., zero centered and unit cord length
    # "geometry_data": "./03 CODE/4 Data/Yonekura/standardized_NandJ_coords.npz",
    
    #"performance_data": "./03 CODE/4 Data/Yonekura/standardized_NandJ_perfs.npz",
    "performance_data": "./03 CODE/4 Data/NandJ_perfs_V2.npz",

    # General
    "EMA_Factor": 0.95,                      # Exponential moving average factor for data tracking
    "Eval_Gradient_Norm": False,             # Evaluate gradient norm, comes roughly with 20% overhead
    "Eval_Epoch_Interval": 200,              # Interval for evaluation of the model
    "Checkpoint_Interval": 2000,             # Interval for saving model checkpoints
    "Track_Gradients": False,                # Track gradients in tensorboard
    "Track_Parameters": False,               # Track parameters in tensorboard
    "save_model": True,                      # Save model state after training
    "Date-Time": get_datetime_str(),     # Current date and time
    "Identifier": ""                        # Identifier for the model, automatically generated
}

# Create identifier for the model / run
HP["Identifier"] = "DDPM_" + HP["Date-Time"] + "_" + HP["Model_Architecture"] + "_EP" + str(HP["n_epoch"])

# Create dictionary holding the model's metrics
model_metrics = {
    "Diffusion-Loss": 0,
    "CL-Loss": 0,
    "CL-Convergence-Ratio": 0,
    "Convexity-Loss": 0,
    "Smoothness-Loss": 0,
    "Avg-Sampling-Time": 0
}
    

# Protect main function for multiprocessing
if __name__ == '__main__':
    # Load geometry and performance data
    # X['arr_0'] ~ Data, X['arr_1'] ~ Mean, X['arr_2'] ~ Std Dev
    coords = np.load(HP["geometry_data"])
    CLs = np.load(HP["performance_data"])

    # Check if GPU is available
    device = get_device()
    print(f"Using {device} device.")
    HP["device"] = device
    
    # Create Custom Dataset and DataLoader
    airfoil_dataset = Custom_Dataset(coords['arr_0'], CLs['arr_0'], HP)
    data_loader = get_data_loader(airfoil_dataset, HP)

    # Delete all .txt files in the temporary folder
    temp_folder = "./04 RUNS/4 Output/2 Temporary/"
    for filename in os.listdir(temp_folder):
        if filename.endswith(".txt") and filename.startswith("sample"):
            os.remove(os.path.join(temp_folder, filename))
    
    # Setup and train DDPM model
    ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs)

    # Create optimizer
    optim = torch.optim.Adam(ddpm.parameters(), lr=HP["initial_lr"])

    train_DDPM(ddpm, optim, HP, data_loader, model_metrics)
    
    if HP["save_model"]:
        torch.save({
            'model_state': ddpm.state_dict(),
            'optimizer_state': optim.state_dict(),
            'model_architecture': ddpm.nn_model,
            'HP': HP,
            }, "./04 RUNS/2 Checkpoints/" + HP["Identifier"] + '.pt')

    # Visualize airfoil profiles
    visualize_airfoils_tiled(plt_label = HP["Identifier"] + "_", show_plot = False, save_plot = True, eval_CL = False)
    
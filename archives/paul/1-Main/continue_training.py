import os
import sys
import numpy as np
import torch

sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/3 Models")
sys.path.append("./03 CODE/4 Data")
sys.path.append("./03 CODE/5 Visualization/Visualize Airfoil Profile")
sys.path.append("./03 CODE/6 Utils")

from DDPM import setup_DDPM
from Data_Lib import Custom_Dataset, get_data_loader
from Util_Lib import get_device
from visualize_profiles_tiled import visualize_airfoils_tiled
from Training_Loop import train_DDPM


# Load prior checkpoint
identifier_str = "DDPM_2024-06-30_22-27-18_UNet_EP30000"
checkpoint = torch.load("./04 RUNS/2 Checkpoints/" + identifier_str + '.pt')
add_epoch = 10000

# Protect main function for multiprocessing
if __name__ == '__main__':
    # Load Hyperparameters
    # Todo Print HP in nice format
    # HP = np.load(HP_path, allow_pickle=True)['arr_0'].item()
    HP = checkpoint['HP']
    print(HP, "\n")

    # Update Hyperparameters
    HP["n_restart"] = HP["n_epoch"] + 1
    HP["n_epoch"] = HP["n_epoch"] + add_epoch
    HP["initial_lr"] = 1.5E-6
    HP["final_lr"] = 0.0
    HP["alpha"] = 1.0

    HP["cnvxty_loss_W"] = 1E-4

    # Create identifier for the model / run
    # Use old identifier and add the new epoch count
    identifier_str = identifier_str.split("_EP")[0]
    HP["Identifier"] = identifier_str + "_EP" + str(HP["n_epoch"])

    # Create dictionary holding the model's metrics
    model_metrics = {
        "Diffusion-Loss": 0,
        "CL-Loss": 0,
        "CL-Convergence-Ratio": 0,
        "Convexity-Loss": 0,
        "Smoothness-Loss": 0,
        "Avg-Sampling-Time": 0
    }

    # Delete all .txt files in the temporary folder
    temp_folder = "./04 RUNS/4 Output/2 Temporary/"
    for filename in os.listdir(temp_folder):
        if filename.endswith(".txt") and filename.startswith("sample"):
            os.remove(os.path.join(temp_folder, filename))

    # Load geometry and performance data
    # X['arr_0'] ~ Data, X['arr_1'] ~ Mean, X['arr_2'] ~ Std Dev
    coords = np.load(HP["geometry_data"])
    CLs = np.load(HP["performance_data"])

    model = checkpoint['model_architecture']

    # Check if GPU is available
    device = get_device()
    print(f"Using {device} device.")
    HP["device"] = device
    
    # Create Custom Dataset and DataLoader
    airfoil_dataset = Custom_Dataset(coords['arr_0'], CLs['arr_0'], HP)
    data_loader = get_data_loader(airfoil_dataset, HP)
    
    # Setup and train DDPM model
    ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs, Model=model)
    # Load model state
    ddpm.load_state_dict(checkpoint['model_state'])

    # Create optimizer and load optimizer state
    optim = torch.optim.Adam(ddpm.parameters(), lr=HP["initial_lr"])
    optim.load_state_dict(checkpoint['optimizer_state'])

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
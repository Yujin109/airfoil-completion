import os
import sys
import numpy as np
import torch

sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/3 Models")
sys.path.append("./03 CODE/6 Utils")

from DDPM import setup_DDPM
from Util_Lib import get_device, get_datetime_str
from CL_Training_Loop import train_CL_DDPM


# Path to the hyperparameters and model
HP_path = "./03 CODE/9 Logs/HP/" +        "DDPM_HP_2024-05-02_17-18-26_ep_5000" + ".npz"
Model_path = "./03 CODE/9 Logs/Models/" + "DDPM_2024-05-02_17-18-26_ep_5000" + ".pth"
add_epoch = 2
epoch_size = 1
CL_max = 1.2
CL_min = 0.5

# Protect main function for multiprocessing
if __name__ == '__main__':
    # Load Hyperparameters
    HP = np.load(HP_path, allow_pickle=True)['arr_0'].item()
    print(HP, "\n")

    # Update Hyperparameters
    HP["n_restart"] = HP["n_epoch"] + 1
    HP["n_epoch"] = HP["n_epoch"] + add_epoch
    HP["initial_lr"] = 5E-5
    HP["final_lr"] = 0.0
    HP["alpha"] = 1.0
    HP["CL_loss_weight"] = 0.1
    HP["filter_loss_weight"] = 0.1
    HP["Batch_Size"] = 10
    HP["Save_Model"] = False
    #"Date-Time": get_datetime_str()

    # Create dictionary holding the model's metrics
    model_metrics = {
        "EMA-Loss": 0
    } 

    # Load geometry and performance data
    # X['arr_0'] ~ Data, X['arr_1'] ~ Mean, X['arr_2'] ~ Std Dev
    coords = np.load(HP["geometry_data"])
    CLs = np.load(HP["performance_data"])

    # Check if GPU is available
    device = get_device()
    print(f"Using {device} device.")
    HP["device"] = device
    
    # Setup and train DDPM model
    ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs)
    # Load model state
    ddpm.load_state_dict(torch.load(Model_path))

    train_CL_DDPM(ddpm, HP, model_metrics, epoch_size=epoch_size, CL_max=CL_max, CL_min=CL_min)
    
    if HP["save_model"]:
        # Save Model state:
        torch.save(ddpm.state_dict(), os.path.join("03 CODE\9 Logs\Models\DDPM_" + HP["Date-Time"] + "_ep_" + str(HP["n_epoch"]) + '.pth'))
        # Save Hyperparameters:
        np.savez("03 CODE\9 Logs\HP\DDPM_HP_" + HP["Date-Time"] + "_ep_" + str(HP["n_epoch"]) + '.npz', HP)
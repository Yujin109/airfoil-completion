import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/6 Utils")

from DDPM import setup_DDPM
from Util_Lib import get_device, estimate_performance

n_variation = 8
n_resample = 1
XFoil_viscous = True

# Path to the hyperparameters and model
Str_Identifier = "DDPM_2024-07-06_09-31-34_UNet_Res_EP16000"

# Protect main function for multiprocessing
if __name__ == '__main__':
    # Load prior checkpoint
    checkpoint = torch.load("./04 RUNS/2 Checkpoints/" + Str_Identifier + '.pt')
    #checkpoint = torch.load("./06 REMOTE/TMP/2 Checkpoints/" + Str_Identifier + '.pt', map_location=torch.device('cpu'))
    # Load Hyperparameters
    HP = checkpoint['HP']
    print(HP, "\n")

    # Load geometry and performance data
    # X['arr_0'] ~ Data, X['arr_1'] ~ Mean, X['arr_2'] ~ Std Dev
    coords = np.load(HP["geometry_data"])
    CLs = np.load(HP["performance_data"])
    
    if True:
        model = checkpoint['model_architecture']
    else:
        model = None

    # Check if GPU is available
    device = get_device()
    print(f"Using {device} device.")
    HP["device"] = device
      
    # Setup DDPM model
    ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs, Model=model, print_summary=True)
    # Load model state
    ddpm.load_state_dict(checkpoint['model_state'])

    # Estimate the performance of the model
    estimate_performance(ddpm, HP, n_variation, n_resample=n_resample, XFoil_viscous=XFoil_viscous)
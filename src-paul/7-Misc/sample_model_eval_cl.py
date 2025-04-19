import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/6 Utils")
sys.path.append("./03 CODE/8 X-Foil/Python_DIR/")

from DDPM import setup_DDPM
from Util_Lib import get_device, get_datetime_str
from xfoil_runner import eval_CL_XFOIL

n_variation = 2
XFoil_viscous = True

# Path to the hyperparameters and model
Str_Identifier = "DDPM_2024-05-28_14-48-52_UNet_EP2000"
HP_path = "./04 RUNS/2 Hyperparameters/" + Str_Identifier + ".npz"
Model_path = "./04 RUNS/3 Model States/" + Str_Identifier + ".pth"


# Protect main function for multiprocessing
if __name__ == '__main__':
    # Load Hyperparameters
    HP = np.load(HP_path, allow_pickle=True)['arr_0'].item()
    print(HP, "\n")

    # Load geometry and performance data
    # X['arr_0'] ~ Data, X['arr_1'] ~ Mean, X['arr_2'] ~ Std Dev
    coords = np.load(HP["geometry_data"])
    CLs = np.load(HP["performance_data"])

    # Check if GPU is available
    device = get_device()
    print(f"Using {device} device.")
    HP["device"] = device
      
    # Setup DDPM model
    ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs)
    # Load model state
    ddpm.load_state_dict(torch.load(Model_path))

    # Set to evaluation mode
    ddpm.eval()

    cl_list = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    #cl_list = [0.6,0.8,1.0,1.2]

    n_sample = n_variation * cl_list.__len__()

    # Sample from the model, skip the history and do not compute gradients
    with torch.no_grad():
        geom = ddpm.sample(cl_list, n_variation)[0]
    #    geom = ddpm.prolonged_sample(cl_list, n_variation, gamma=5E-6, max_iter=200)[0]
        
        # Compute convexity loss
        geom = geom.detach().cpu().numpy()
        cnvx_loss = ddpm.convexity_loss(geom, reduction="None")

    # plot all the samples in tiled form
    fig, axs = plt.subplots(n_variation, cl_list.__len__(), figsize=(20, 5))
    for i in range(n_variation):
        for j in range(cl_list.__len__()):
            axs[i,j].scatter(geom[i*cl_list.__len__()+j,0,:], geom[i*cl_list.__len__()+j,1,:], c = range(len(geom[i*cl_list.__len__()+j,0,:])), cmap = 'viridis', s=1)
            # Title indicates the lift coefficient and convexity loss
            axs[i,j].set_title(f"CL = {cl_list[j]}\nConvexity Loss: {cnvx_loss[i*cl_list.__len__()+j]:.1f}")
            axs[i,j].set_aspect('equal')
            axs[i,j].axis('off')
            axs[i,j].grid(True)

    # Add convexity loss as text to the plot
    fig.text(0.5, 0.04, f"Mean Convexity Loss: {torch.mean(cnvx_loss):.4f}", ha='center')
    plt.show()


    n_class = cl_list.__len__()
    str_label = ""
    for i in range(n_sample):            
        np.savetxt("./04 RUNS/4 Output/2 Temporary/sample_" + Str_Identifier + \
                    str_label + '_cl_' + str(cl_list[i%n_class]) + \
                    '_' + str(i//n_class) + '.txt', geom[i,:,:].squeeze().T)

    exit()
    
    for u in range(0):
        geom = ddpm.filter_op(geom)
  
    geom = geom.detach().cpu().numpy()

    CL, CD = eval_CL_XFOIL(geom, viscous = XFoil_viscous)

    print("\nLift Coefficients:")
    print(CL)


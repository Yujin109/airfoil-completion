import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/6 Utils")

from DDPM import setup_DDPM
from Util_Lib import get_device

index = 2700
n_vars = 2
n_repaint = 1
trgt_cls = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
# Where the mask is 1, geometry is inpainted
num_features = 496//2
mask = np.zeros(num_features)

#mask first and last 10%
mask[:int(0.05*num_features)] = 1
mask[-int(0.05*num_features):] = 1
mask[int(num_features/2)-int(0.05*num_features):int(num_features/2)+int(0.05*num_features)] = 1

mask = 1 - mask

#mask[30:94] = 1
#mask[154:194] = 1

#mask[0:123] = 1

#mask[0:40] = 1
#mask[120:160] = 1

#mask[220:248] = 1

#mask[40:200] = 1

#mask[:] = 1
mask = mask.astype(bool)

# Path to the hyperparameters and model
Str_Identifier = "DDPM_2024-07-02_04-15-24_UNet_Res_EP10000"

# Protect main function for multiprocessing
if __name__ == '__main__':
    # Load prior checkpoint
    checkpoint = torch.load("./04 RUNS/2 Checkpoints/" + Str_Identifier + '.pt')
    # Load Hyperparameters
    HP = checkpoint['HP']
    print(HP, "\n")

    # Load geometry and performance data
    # X['arr_0'] ~ Data, X['arr_1'] ~ Mean, X['arr_2'] ~ Std Dev
    coords = np.load(HP["geometry_data"])
    CLs = np.load(HP["performance_data"])

    # Check if GPU is available
    device = get_device()
    print(f"Using {device} device.")
    HP["device"] = device

    # Load Architecture
    #nn_model = checkpoint['model_architecture']
      
    # Setup DDPM model
    ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs)#, Model=nn_model)
    # Load model state
    ddpm.load_state_dict(checkpoint['model_state'])

    ddpm.eval()

    x_prior = torch.zeros(2, HP["input_features"]//2)
    x_prior[0,:] = torch.tensor(coords['arr_0'][index][:HP["input_features"]//2])
    x_prior[1,:] = torch.tensor(coords['arr_0'][index][HP["input_features"]//2:])

    geom_prior = ddpm.renormalize(x_prior.unsqueeze(0).float().to(HP["device"]))
    geom_prior = geom_prior.detach().cpu().numpy()[0]

    geom_prior[:,mask] = np.nan
    x_prior[:,mask] = torch.tensor(np.nan)

    with torch.no_grad():
        x_prior = x_prior.float().to(HP["device"])
        geom_post = ddpm.inpaint(target_cls=trgt_cls, x_prior=x_prior, n_resample = n_repaint, n_variation=n_vars)[0]

    geom_post = geom_post.detach().cpu().numpy()
    
    # Plot all geometries in a tiled fashion
    fig, axs = plt.subplots(trgt_cls.__len__(), n_vars, figsize=(10, 10))
    for i, cl in enumerate(trgt_cls):
        for j in range(n_vars):
            idx = i + j * trgt_cls.__len__()
            axs[i, j].plot(geom_post[idx, 0, :], geom_post[idx, 1, :], label="Posterior")
            axs[i, j].plot(geom_prior[0, :], geom_prior[1, :], label="Prior")
            axs[i, j].set_title(f"CL: {cl}, Vers.: {j}")
            axs[i, j].set_aspect('equal')
            #axs[i, j].legend()

    plt.tight_layout()
    plt.show()






import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/4 Data")
sys.path.append("./03 CODE/6 Utils")
sys.path.append("./03 CODE/8 X-Foil/Python_DIR/")

from DDPM import setup_DDPM
from Data_Lib import Custom_Dataset, get_data_loader
from Util_Lib import get_device


n_variation = 2

# Path to the hyperparameters and model
Str_Identifier = "DDPM_2024-05-28_00-56-38_Lin_Conv_Network_2_EP2000"
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

    # Create Custom Dataset and DataLoader
    airfoil_dataset = Custom_Dataset(coords['arr_0'], CLs['arr_0'], HP)
    data_loader = get_data_loader(airfoil_dataset, HP)
      
    # Setup and train DDPM model
    ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs)
    # Load model state
    ddpm.load_state_dict(torch.load(Model_path))

    # Forward or Diffusion Process
    ddpm.train()

    # get first batch from data loader
    x, c = next(iter(data_loader))

    x = x.to(HP["device"])
    c = c.to(HP["device"])

    # Forward pass
    ts_eval = (1, ddpm.n_T * 0.1, ddpm.n_T * 0.25, ddpm.n_T * 0.5, ddpm.n_T * 0.75, ddpm.n_T)
    ts_eval = [int(ts) for ts in ts_eval]

    noise = torch.randn_like(x)  # eps ~ N(0, 1)
    for ts in ts_eval:
        ts = torch.tensor(ts).to(HP["device"])
        #ts = torch.randint(1, ddpm.n_T, (x.shape[0],)).to(ddpm.device)  # t ~ Uniform(0, n_T)
        #noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (ddpm.sqrtab[ts, None, None] * x + ddpm.sqrtmab[ts, None, None] * noise)

        geom = ddpm.renormalize(x_t)  
    
        #eps = ddpm.nn_model(x_t, c, (ts / ddpm.n_T).unsqueeze(1))

        geom = geom.detach().cpu().numpy()

        # plot the geometry
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.scatter(geom[0, 0, :], geom[0, 1, :], c=range(len(geom[0, 0, :])), cmap='viridis', s=1)
        #axs[0].set_title(f"t = {ts}")
        axs.set_aspect('equal')
        axs.axis('off')
        axs.grid(True)

        plt.show()
    

    exit()






    # Set to evaluation mode
    ddpm.eval()

    cl_list = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]

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
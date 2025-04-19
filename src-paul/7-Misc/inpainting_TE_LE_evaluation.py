import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/4 Data")
sys.path.append("./03 CODE/6 Utils")

from DDPM import setup_DDPM
from Util_Lib import get_device
from Data_Lib import Custom_Dataset, get_data_loader

batch_size = 24
batches = 10
resample_steps = 10
Model_Identifier = "DDPM_2024-07-02_04-15-24_UNet_Res_EP10000"

# --------------------------------------------------------------------------------

# Load prior checkpoint
checkpoint = torch.load("./04 RUNS/2 Checkpoints/" + Model_Identifier + '.pt')
# Load Hyperparameters
HP = checkpoint['HP']
print(HP, "\n")

# Load geometry and performance data
# X['arr_0'] ~ Data, X['arr_1'] ~ Mean, X['arr_2'] ~ Std Dev
coords = np.load(HP["geometry_data"])
CLs = np.load(HP["performance_data"])

#model = checkpoint['model_architecture']

# Check if GPU is available
device = get_device()
print(f"Using {device} device.")
HP["device"] = device

HP["batch_size"] = batch_size

# Create Custom Dataset and DataLoader
airfoil_dataset = Custom_Dataset(coords['arr_0'], CLs['arr_0'], HP)
data_loader = get_data_loader(airfoil_dataset, HP)

# Setup and train DDPM model
ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs)#, Model=model)
# Load model state
ddpm.load_state_dict(checkpoint['model_state'])

ddpm.eval()

# Todo use a dictionary to store the results 
cl_gt_log = []
cl_trgt_log = []
cl_log = []
cd_log = []
cnvxty_log = []

with torch.no_grad():

    # Pick a random batch of airfoils
    for i, data in enumerate(data_loader):
        print(f"Batch {i+1}/{len(data_loader)}")
        X, cl_gt = data

        X = X.to(HP["device"])

        Geom = ddpm.renormalize(X)

        # Renormalize the ground truth cl
        cl_gt = cl_gt * CLs["arr_2"] + CLs["arr_1"]

        # pick a list of random coefficient of lift between 0.5 and 1.2
        cl_trgt = [np.random.uniform(0.5, 1.2) for _ in range(X.shape[0])]

        # keep first and last 10 percent of the airfoils coordiantes in x direction and set the rest to nan
        for u in range(X.shape[0]):
            xmin, xmax = torch.min(Geom[u, 0, :]), torch.max(Geom[u, 0, :])
            xlen = xmax - xmin
            mask = (Geom[u, 0, :] >= (xmin + 0.1 * xlen)) & (Geom[u, 0, :] <= (xmin + 0.9 * xlen))

            X[u, :, mask] = torch.tensor(np.nan).to(HP["device"])

            #Geom[u, :, mask] = torch.tensor(np.nan).to(HP["device"])
            #plt.plot(Geom[u, 0, :].cpu().numpy(), Geom[u, 1, :].cpu().numpy())
            #plt.show()

        # inpaint the airfoils with the given cl
        Geom_inpainted = ddpm.inpaint(cl_trgt, X, n_resample=resample_steps)[0]

        # evaluate the inpainted airfoils
        cl_loss, convergence_ratio, cl, cd = ddpm.cl_loss(Geom_inpainted, cl_trgt, return_CL=True, return_CD=True)

        # evaluate the convexity of the inpainted airfoils
        convexity = ddpm.convexity_loss(Geom_inpainted, reduction="")

        # convert to numpy
        cl_gt = cl_gt.cpu().numpy()
        cl_trgt = np.array(cl_trgt)
        cl = cl.cpu().numpy()
        #cd = cd.cpu().numpy()
        convexity = convexity.cpu().numpy()

        # store
        cl_gt_log.append(cl_gt)
        cl_trgt_log.append(cl_trgt)
        cl_log.append(cl)
        cd_log.append(cd)
        cnvxty_log.append(convexity)

        if i == batches-1:
            break

# --------------------------------------------------------------------------------

# Save to numpy file
np.savez("inpainting_log.npz", cl_gt_log, cl_trgt_log, cl_log, cd_log, cnvxty_log)
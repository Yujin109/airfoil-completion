import sys
import numpy as np
import torch
import time
import csv
import matplotlib.pyplot as plt

sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/6 Utils")

from DDPM import setup_DDPM
from Util_Lib import get_device, estimate_performance

n_variation = 8
resample_steps = [1, 2, 3, 4, 6, 8, 12, 16, 24]
XFoil_viscous = True

# Path to the hyperparameters and model
Str_Identifier = "DDPM_2024-06-21_09-42-42_UNet_EP10000"

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

#model = checkpoint['model_architecture']

# Check if GPU is available
device = get_device()
print(f"Using {device} device.")
HP["device"] = device
    
# Setup DDPM model
# ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs, Model=model, print_summary=True)
ddpm = setup_DDPM(HP=HP, coords=coords, CLs=CLs, print_summary=True)
# Load model state
ddpm.load_state_dict(checkpoint['model_state'])

ddpm.eval()

L_cnvxty = []
L_smthns = []
L_cl = []
L_convrg = []
Runtime = []

with torch.no_grad():
    for u, n_resample in enumerate(resample_steps):
        # Measure the time for resampling
        start_time = time.perf_counter()

        # Estimate the performance of the model
        cnvxty_loss_tmp, smthns_loss_tmp, cl_loss_tmp, convergence_ratio_tmp = estimate_performance(ddpm, HP, n_variation, n_resample=n_resample, XFoil_viscous=XFoil_viscous)

        end_time = time.perf_counter()

        Runtime.append(end_time - start_time)

        L_cnvxty.append(cnvxty_loss_tmp)
        L_smthns.append(smthns_loss_tmp)
        L_cl.append(cl_loss_tmp)
        L_convrg.append(convergence_ratio_tmp)

# save the results to a csv file
with open('./04 RUNS/4 Output/Resampling_Results.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Resample Steps', 'Convexity Loss', 'Smoothness Loss', 'CL Loss', 'CL Convergence Ratio', 'Runtime'])

    for i in range(len(resample_steps)):
        writer.writerow([resample_steps[i], L_cnvxty[i], L_smthns[i], L_cl[i], L_convrg[i], Runtime[i]])

# Plot losses in subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Performance Metrics')

axs[0, 0].plot(resample_steps, L_cnvxty, 'tab:blue')
axs[0, 0].set_title('Convexity Loss')
axs[0, 0].set(xlabel='Resampling Steps', ylabel='Loss')

axs[0, 1].plot(resample_steps, L_smthns, 'tab:orange')
axs[0, 1].set_title('Smoothness Loss')
axs[0, 1].set(xlabel='Resampling Steps', ylabel='Loss')

axs[1, 0].plot(resample_steps, L_cl, 'tab:green')
axs[1, 0].set_title('CL Loss')
axs[1, 0].set(xlabel='Resampling Steps', ylabel='Loss')

axs[1, 1].plot(resample_steps, L_convrg, 'tab:red')
axs[1, 1].set_title('CL Convergence Ratio')
axs[1, 1].set(xlabel='Resampling Steps', ylabel='Ratio')

plt.tight_layout()

plt.show()

# Plot runtime
plt.figure(figsize=(8, 6))
plt.plot(resample_steps, Runtime, 'tab:purple')
plt.title('Runtime')
plt.xlabel('Resampling Steps')
plt.ylabel('Time [s]')
plt.show()

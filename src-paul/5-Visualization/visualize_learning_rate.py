import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./03 CODE/6 Utils")
from Util_Lib import powerlaw_decay

HP = {
    "n_epoch": 10000,
    "n_restart": 1,                                         # Number of restarts if the model is trained from prior state
    "initial_lr": 1E-4, #5E-6, #1E-4,                                     # Initial learning rate 5E-4, UNets 2E-4
    "final_lr": 0.0,                                        # Final learning rate 
    "alpha": 0.1 #0.125                                         # Powerlaw exponent of learning rate decay
}

if False:
    add_epoch = 5000
    HP["n_restart"] = HP["n_epoch"] + 1
    HP["n_epoch"] = HP["n_epoch"] + add_epoch

step_dist = 1E-3

lr = np.zeros(HP["n_epoch"] + 1 - HP["n_restart"])
lr_2 = np.zeros(HP["n_epoch"] + 1 - HP["n_restart"])
lr_3 = np.zeros(HP["n_epoch"] + 1 - HP["n_restart"])
lr_dist = np.zeros(HP["n_epoch"] + 1 - HP["n_restart"])
lr_dist_2 = np.zeros(HP["n_epoch"] + 1 - HP["n_restart"])
lr_dist_3 = np.zeros(HP["n_epoch"] + 1 - HP["n_restart"])

for idx, ep in enumerate(range(HP["n_restart"], HP["n_epoch"] + 1)):

    if False: # For Retraining
        c_ep = ep - HP["n_restart"]
        if c_ep < 500:
            lr[idx] = powerlaw_decay(0, HP["initial_lr"], HP["alpha"], c_ep / 500)
        else:
            lr[idx] = powerlaw_decay(HP["initial_lr"], 0, HP["alpha"], (c_ep - 500) / (HP["n_epoch"] - HP["n_restart"] - 500))
    else:
        lr[idx] = powerlaw_decay(HP["initial_lr"], HP["final_lr"], HP["alpha"], (ep - HP["n_restart"]) / (HP["n_epoch"] - HP["n_restart"]))
        lr_2[idx] = powerlaw_decay(HP["initial_lr"], HP["final_lr"], HP["alpha"], (ep - HP["n_restart"]) / (HP["n_epoch"] - HP["n_restart"])) if idx < 1000 else lr_2[idx - 1] * 0.99954

    lr_dist[idx] = lr_dist[idx - 1] + step_dist * lr[idx] if idx > 0 else step_dist * lr[idx]
    lr_dist_2[idx] = lr_dist_2[idx - 1] + step_dist * lr_2[idx] if idx > 0 else step_dist * lr_2[idx]
    

# Plot learning rate  and cumulative learning distance in two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(range(HP["n_restart"], HP["n_epoch"] + 1), lr, label="Learning Rate")
axs[0].plot(range(HP["n_restart"], HP["n_epoch"] + 1), lr_2, label="Learning Rate 2")
axs[0].set_title("Learning Rate")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Learning Rate")
axs[0].set_yscale("log")
axs[0].legend()

axs[1].plot(range(HP["n_restart"], HP["n_epoch"] + 1), lr_dist, label="Cumulative Learned Distance")
axs[1].plot(range(HP["n_restart"], HP["n_epoch"] + 1), lr_dist_2, label="Cumulative Learned Distance 2")
axs[1].plot(range(HP["n_restart"], HP["n_epoch"] + 1), lr_dist_3, label="Cumulative Learned Distance 3")
axs[1].set_title("Cumulative Learned Distance")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Cumulative Learned Distance")
axs[1].legend()

plt.tight_layout()
plt.show()
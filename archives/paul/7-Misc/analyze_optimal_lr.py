import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./03 CODE/6 Utils")
from Util_Lib import evaluate_optimal_lr

EMA_factor_2 = 0.5
LR_Monotonic = True

crit_lrs = []
lr_mins = []
opt_lrs = []

eps = [*range(25, 7850, 25)]

for ep in eps:
    file_path = "loss_eval_DDPM_2024-06-16_00-00-18_UNet_EP10000_EP" + str(int(ep)) + ".npy"

    # load lr and loss data
    lr_data = np.load("./04 RUNS/4 Output/2 Temporary/" + file_path, allow_pickle=True)

    # get lr and loss data
    lr = lr_data[0, :]
    loss = lr_data[1, :]

    crit_lr, lr_min, opt_lr = evaluate_optimal_lr(lr, loss)

    crit_lrs.append(crit_lr)
    lr_mins.append(lr_min)
    opt_lrs.append(opt_lr)

    EMA_opt_lr = [opt_lrs[0]]

    for i in range(1, len(opt_lrs)):
        # Guarantee that learning rate decreases
        if opt_lrs[i] > EMA_opt_lr[-1] and LR_Monotonic:
            opt_lrs[i] = EMA_opt_lr[-1]

        EMA_opt_lr.append(EMA_factor_2 * opt_lrs[i] + (1 - EMA_factor_2) * EMA_opt_lr[-1])


plt.plot(eps, crit_lrs, color='r')
plt.plot(eps, lr_mins, color='g')
plt.plot(eps, opt_lrs, color='b')
plt.plot(eps, EMA_opt_lr, color='k')
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs Epochs")
plt.legend(["Critical Learning Rate", "Minimum Learning Rate", "Optimal Learning Rate", "EMA Optimal Learning Rate"])
plt.grid(which='both', axis='both')

plt.show()
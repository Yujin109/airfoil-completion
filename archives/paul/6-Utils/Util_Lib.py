import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_datetime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Powerlaw mapping, specially for learning rate decay. x is in [0, 1]
def powerlaw_decay(y_init, y_end, alpha, x):
    return (y_init - y_end) * (1 - x**alpha) + y_end


def average_runtime(func, args=(), kwargs=None, runs=10):
    """
    Times a function over multiple runs and returns the average runtime.

    Parameters:
    - func (callable): The function to be timed.
    - args (tuple): The positional arguments to be passed to the function.
    - kwargs (dict): The keyword arguments to be passed to the function.
    - runs (int): The number of times the function should be run.

    Returns:
    - float: The average runtime of the function over the specified number of runs.

    # Example usage:
    def sample_function(x, y):
        time.sleep(0.1)
        return x + y

    avg_time = average_runtime(sample_function, args=(1, 2), runs=5)
    print(f"Average runtime: {avg_time:.5f} seconds")
    """
    if kwargs is None:
        kwargs = {}

    total_time = 0.0

    for _ in range(runs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / runs
    return average_time


def estimate_performance(ddpm, HP, n_variation=8, n_resample=1, XFoil_viscous=True):
    # Set to evaluation mode
    ddpm.eval()

    cl_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    n_sample = n_variation * cl_list.__len__()

    with torch.no_grad():
        geom = ddpm.sample(cl_list, n_variation, n_resample=n_resample)[0]
        #    geom = ddpm.prolonged_sample(cl_list, n_variation, gamma=5E-6, max_iter=200)[0]

        # geom = geom.detach().cpu().numpy()

        # Todo fix target input --> see eval_DDPM
        cnvxty_loss = ddpm.convexity_loss(geom)
        smthns_loss = ddpm.smoothness_loss(geom)
        roughnss_loss = ddpm.roughness_loss(geom)

        # Todo find bounds and move threshold to HP
        if cnvxty_loss < 50:
            cl_trgt = torch.tensor(cl_list).float().to(HP["device"])
            # Repeat cl_trgt for n_variation times
            cl_trgt = cl_trgt.repeat(n_variation)
            cl_loss, convergence_ratio = ddpm.cl_loss(geom, cl_trgt, viscous=XFoil_viscous)
        else:
            cl_loss = float("NAN")
            convergence_ratio = float("NAN")

    # Print to console
    print(f"Convexity Loss: {cnvxty_loss:.4f}")
    print(f"Smoothness Loss: {smthns_loss:.8f}")
    print(f"Roughness Loss: {roughnss_loss[0]:.8f}, {roughnss_loss[1]:.8f}, {roughnss_loss[2]:.8f}")
    print(f"CL Loss: {cl_loss:.6f}")
    print(f"Convergence Ratio: {convergence_ratio:.4f}")

    # save to an csv file
    cnvxty_loss = cnvxty_loss.detach().cpu().numpy()
    smthns_loss = smthns_loss.detach().cpu().numpy()
    roughnss_loss = roughnss_loss[1].detach().cpu().numpy()
    if isinstance(cl_loss, torch.Tensor):
        cl_loss = cl_loss.detach().cpu().numpy()
        convergence_ratio = convergence_ratio.detach().cpu().numpy()

    cnvxty_loss, smthns_loss, cl_loss, convergence_ratio, roughnss_loss = (
        np.array([cnvxty_loss]),
        np.array([smthns_loss]),
        np.array([cl_loss]),
        np.array([convergence_ratio]),
        np.array([roughnss_loss]),
    )
    np.savetxt(
        "./04 RUNS/4 Output/2 Temporary/airfoil_losses_" + HP["Identifier"] + ".csv",
        np.column_stack((cl_loss, convergence_ratio, cnvxty_loss, smthns_loss, roughnss_loss)),
        delimiter=",",
    )

    return cnvxty_loss, smthns_loss, cl_loss, convergence_ratio, roughnss_loss


def evaluate_optimal_lr(lr, loss):

    # Settings
    max_loss_factor = 10
    crit_lr_factor_1 = 0.5
    std_dev_iterations = 2
    std_dev_interval = 2
    EMA_factor = 0.1
    crit_lr_factor_2 = 0.5
    alpha = 2 / 3

    # find critical learning rate by finding the last index where the loss (change) is less than 1
    # Todo, do this on the filtered loss
    # Todo, edge case: last index = 0
    last_idx = np.where(abs(loss) < max_loss_factor * abs(loss[0]))[0][-1]

    crit_lr = crit_lr_factor_1 * lr[last_idx]

    # Find index where the learning rate is the critical learning rate
    last_idx = np.where(lr < crit_lr)[0][-1]

    if last_idx == 0:
        print("Warning: Critical learning rate cant be found, using last learning rate")
        last_idx = len(loss)

    # remove data after this index
    loss = loss[:last_idx]
    lr = lr[:last_idx]

    # calculate mean and std of loss
    mean_loss = np.mean(loss)
    std_loss = np.std(loss)

    # two times remove outliers
    for u in range(std_dev_iterations):
        # low pass filter the loss with exponential moving average
        flt_loss = np.zeros_like(loss)
        flt_loss[0] = loss[0]

        for i in range(1, len(loss)):
            flt_loss[i] = EMA_factor * loss[i] + (1 - EMA_factor) * flt_loss[i - 1]

        # remove all data that is outside of two standard deviations
        try:
            last_idx = np.where(abs(flt_loss - mean_loss) > std_dev_interval * std_loss)[0][0]
        except:
            last_idx = len(loss)

        if last_idx == 0:
            print("Warning: No data left after filtering, using last learning rate")
            last_idx = len(loss)

        loss = loss[:last_idx]
        lr = lr[:last_idx]

        # recompute mean and std of loss
        mean_loss = np.mean(loss)
        std_loss = np.std(loss)

    # fit robust approximator to the filtered loss with least absolute error
    # perform the fit in the log space
    lr_hat = np.log10(lr)

    approximator = lambda x, theta: theta[0] + theta[1] * x + theta[2] * x**2
    objective = lambda theta, x, target: np.sum(abs(approximator(x, theta) - target))
    constraint1 = lambda theta: theta[2]

    theta0 = [0, 1, 1]
    cons = {"type": "ineq", "fun": constraint1}

    solution = minimize(objective, theta0, constraints=cons, args=(lr_hat, loss))
    print(solution.message)

    flt_loss_fit = approximator(lr_hat, solution.x)
    lr_hat_min = -solution.x[1] / (2 * solution.x[2])
    lr_min = 10**lr_hat_min

    crit_lr_2 = crit_lr_factor_2 * max(lr)

    opt_lr = 10 ** (alpha * lr_hat_min + (1 - alpha) * np.log10(crit_lr_2))

    opt_lr = min(opt_lr, crit_lr_2)

    if False:
        plt.plot(lr, loss)
        plt.plot(lr, flt_loss)
        plt.plot(lr, flt_loss_fit)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate vs Loss")
        plt.xscale("log")

        plt.axhline(y=loss[0], color="r", linestyle="--")

        plt.axhline(y=mean_loss, color="k", linestyle="--")
        plt.axhline(y=mean_loss + 2 * std_loss, color="k", linestyle="--")
        plt.axhline(y=mean_loss - 2 * std_loss, color="k", linestyle="--")
        plt.axvline(x=lr_min, color="g", linestyle="--")
        plt.axvline(x=crit_lr_2, color="r", linestyle="--")
        plt.axvline(x=opt_lr, color="b", linestyle="--")

        plt.show()

    return crit_lr, lr_min, opt_lr

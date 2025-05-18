import sys

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary as model_summary
from tqdm import tqdm

sys.path.append("./03 CODE/3 Models")
from Network_Definitions import *

sys.path.append("./03 CODE/6 Utils")
from Util_Lib import evaluate_optimal_lr

sys.path.append("./03 CODE/8 X-Foil/Python_DIR/")
from xfoil_runner import eval_CL_XFOIL


def ddpm_schedules(beta1, beta2, T, schedule_type="linear"):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """

    # function to compute cosine schedule
    def f(t, T, s=1e-3):
        arg = np.pi * 0.5 * (t / T + s) / (1 + s)
        return np.cos(arg) ** 2

    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    if schedule_type == "linear":
        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    elif schedule_type == "cos":
        f0 = f(0, T)
        alphabar_t = f(torch.arange(0, T + 1, dtype=torch.float32), T) / f0
        alpha_bar_m = f(torch.arange(-1, T, dtype=torch.float32), T) / f0
        beta_t = 1 - (alphabar_t / alpha_bar_m)
        beta_t[beta_t > 0.999] = 0.999

    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t

    if schedule_type == "linear":
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "beta_t": beta_t,  # \beta_t
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    # Todo fix arguments, pass only hp not betas, model, device, and n_T
    def __init__(self, nn_model, filter_op, HP, x_mu, x_std, cl_mu, cl_std):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(HP["device"])

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(HP["betas"][0], HP["betas"][1], HP["n_T"], HP["schedule_type"]).items():
            self.register_buffer(k, v)

        self.filter_op = filter_op
        self.HP = HP

        self.n_T = HP["n_T"]
        self.device = HP["device"]
        self.loss_mse = nn.MSELoss()

        # Normalize the data
        self.x_mu = torch.tensor(x_mu).float().to(self.device)
        self.x_std = torch.tensor(x_std).float().to(self.device)

        self.cl_mu = torch.tensor(cl_mu).float().to(self.device)
        self.cl_std = torch.tensor(cl_std).float().to(self.device)

        self.stats = ""  # model summary

        self.to(HP["device"])

    def forward(self, x, c, eval_x0_pred=False):
        """
        this method is used in training, so samples t and noise randomly
        returns diff_loss
        """
        # Todo ts is sampled from a uniform(0,1) effectively. This is not zero mean!
        ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # Todo i think sqrt(1 - alpha bar) is not correct!
        # We should predict the "error term" from this x_t. Loss is what we return.
        x_t = self.sqrtab[ts, None, None] * x + self.sqrtmab[ts, None, None] * noise

        # print(x_t.shape, c.shape, ts.shape)
        # Predict the input noise, normalized time
        pred_noise = self.nn_model(x_t, c, (ts / self.n_T).unsqueeze(1))

        # return MSE between added noise, and our predicted noise, i.e., the diffusion loss
        diff_loss = self.loss_mse(noise, pred_noise)

        if eval_x0_pred:
            x_0_pred = (x_t - self.sqrtmab[ts, None, None] * pred_noise) / self.sqrtab[ts, None, None]
            return diff_loss, x_0_pred, ts

        return diff_loss

    # Todo GPU memory seams to grow with each iteration, check for memory leaks
    def sample(self, target_cls: list, n_variation=1, n_resample=1, history=False, renormalize=True):
        n_sample = len(target_cls) * n_variation

        x_i = torch.randn(n_sample, 2, self.HP["input_features"] // 2).to(
            self.device
        )  # x_T ~ N(0, 1), sample initial noise

        # Normalize the target CLs and repeat them n_variation times
        c_i = (
            torch.tensor([[(cl - self.cl_mu) / self.cl_std] for cl in target_cls] * n_variation)
            .float()
            .to(self.device)
        )

        x_i_store = []  # keep track of generated steps in case want to plot something

        for i in tqdm(range(self.n_T, 0, -1), desc="Sampling DDPM: "):
            # Normalize the time step and repeat it n_sample times
            t_is = torch.tensor([i / self.n_T]).repeat(n_sample, 1).to(self.device)

            # Resampling as presented in Lugmayrs RePaint paper:
            for u in range(n_resample):

                z = torch.randn(n_sample, 2, self.HP["input_features"] // 2).to(self.device) if i > 1 else 0

                eps = self.nn_model(x_i, c_i, t_is)

                x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

                if u < n_resample - 1 and i > 1:
                    # Todo i think the beta_t index is wrong here
                    x_i = torch.sqrt(1 - self.beta_t[i]).to(self.device) * x_i + self.sqrt_beta_t[
                        i
                    ] * torch.randn_like(x_i).to(self.device)

            if (i % 50 == 0 or i == self.n_T or i < 5) and history:
                x_i_store.append(x_i.detach().cpu().numpy())

        if history:
            x_i_store = np.array(x_i_store)

        # Revert normalization
        # x_i = (x_i * torch.tensor(coords['arr_2']).to(self.device)) + torch.tensor(coords['arr_1']).to(self.device)
        if renormalize:
            x_i = self.renormalize(x_i)

        return x_i, x_i_store

    # Todo use loss function as threshold here
    # Todo use sampling function as input
    def prolonged_sample(self, target_cls: list, n_variation=1, gamma=1e-3, max_iter=1e3, history=False):

        x_i, x_i_store = self.ddpm.sample(target_cls, n_variation, history=history, renormalize=False)

        # Todo Check if this can be deleted
        # Todo if history is true then x_i_store is already converted to numpy!
        """         
        x_i = torch.randn(n_sample, 2, self.HP["input_features"]//2).to(self.device)  # x_T ~ N(0, 1), sample initial noise
        
        # Normalize the target CLs and repeat them n_variation times
        c_i = torch.tensor([[(cl-self.cl_mu) / self.cl_std] for cl in target_cls]*n_variation).float().to(self.device)

        x_i_store = [] # keep track of generated steps in case want to plot something
                
        for i in tqdm(range(self.n_T, 0, -1), desc = "Sampling DDPM: "):
            # Normalize the time step and repeat it n_sample times
            t_is = torch.tensor([i / self.n_T]).repeat(n_sample, 1).to(self.device)

            z = torch.randn(n_sample, 2, self.HP["input_features"]//2).to(self.device) if i > 1 else 0
            
            eps = self.nn_model(x_i, c_i, t_is)
            
            x_i = (self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)

            if (i%50==0 or i==self.n_T or i<5) and history:
                x_i_store.append(x_i.detach().cpu().numpy()) """

        # Initialize the additional sampling process fixed at time step 1
        n_sample = len(target_cls) * n_variation
        sample_loss = nn.MSELoss(reduction="none")
        i = 1
        t_is = torch.tensor([1 / self.n_T]).repeat(n_sample, 1).to(self.device)
        # Normalize the target CLs and repeat them n_variation times
        c_i = (
            torch.tensor([[(cl - self.cl_mu) / self.cl_std] for cl in target_cls] * n_variation)
            .float()
            .to(self.device)
        )
        x_pred = self.renormalize(x_i)
        x_flt = self.filter_op(x_pred)
        # create a flag for the samples that are not yet converged
        flag = sample_loss(x_flt, x_pred)
        flag = torch.mean(flag, dim=(1, 2)) > gamma

        # Prolong the sampling process
        while flag.sum() > 0 and i < max_iter:
            z = torch.randn(n_sample, 2, self.HP["input_features"] // 2).to(self.device)

            eps = self.nn_model(x_i, c_i, t_is)

            x_i[flag, :, :] = (
                self.oneover_sqrta[1] * (x_i - eps * self.mab_over_sqrtmab[1]) + self.sqrt_beta_t[1] * z
            )[flag, :, :]

            if i % 50 == 0 and history:
                x_i_store.append(x_i.detach().cpu().numpy())

            i += 1
            x_pred = self.renormalize(x_i)
            x_flt = self.filter_op(x_pred)
            flag = sample_loss(x_flt, x_pred)
            flag = torch.mean(flag, dim=(1, 2))
            print(flag)
            flag = flag > gamma

        if history:
            x_i_store = np.array(x_i_store)

        # Revert normalization can be skipped since x_pred is already renormalized

        return x_pred, x_i_store

    # Inpaints the prior with the target CLs where the prior is nan
    # Todo check if this is correct
    def inpaint(self, target_cls: list, x_prior, n_resample=1, n_variation=1, history=False, renormalize=True):
        mask = torch.isnan(x_prior)
        x_prior[mask] = 0

        # move to device
        x_prior = x_prior.to(self.device)
        mask = mask.to(self.device)

        n_sample = len(target_cls) * n_variation

        x_i = torch.randn(n_sample, 2, self.HP["input_features"] // 2).to(
            self.device
        )  # x_T ~ N(0, 1), sample initial noise

        # Normalize the target CLs and repeat them n_variation times
        c_i = (
            torch.tensor([[(cl - self.cl_mu) / self.cl_std] for cl in target_cls] * n_variation)
            .float()
            .to(self.device)
        )

        x_i_store = []  # keep track of generated steps in case want to plot something

        for i in tqdm(range(self.n_T, 0, -1), desc="Sampling DDPM: "):
            # Normalize the time step and repeat it n_sample times
            t_is = torch.tensor([i / self.n_T]).repeat(n_sample, 1).to(self.device)

            for u in range(n_resample):
                w = torch.randn(n_sample, 2, self.HP["input_features"] // 2).to(self.device) if i > 1 else 0

                x_prior_noised = self.sqrtab[i] * x_prior + (1 - self.alphabar_t[i]) * w

                z = torch.randn(n_sample, 2, self.HP["input_features"] // 2).to(self.device) if i > 1 else 0

                # print(x_i.shape, c_i.shape, t_is.shape)
                eps = self.nn_model(x_i, c_i, t_is)

                # Todo i think this should x_i + eps! mab_over_sqrtmab is negative
                x_i_unknown = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

                x_i = x_i_unknown * mask + x_prior_noised * ~mask

                if u < n_resample - 1 and i > 1:
                    # Todo i think the beta_t index is wrong here
                    x_i = torch.sqrt(1 - self.beta_t[i]).to(self.device) * x_i + self.sqrt_beta_t[
                        i
                    ] * torch.randn_like(x_i).to(self.device)

            if (i % 50 == 0 or i == self.n_T or i < 5) and history:
                x_i_store.append(x_i.detach().cpu().numpy())

        if history:
            x_i_store = np.array(x_i_store)

        # Revert normalization
        if renormalize:
            x_i = self.renormalize(x_i)

        return x_i, x_i_store

    # Todo rename output to geom
    def renormalize(self, x):
        """
        Renormalize the generated samples to the original coordinates.
        """
        x = ((x.flatten(1) * self.x_std) + self.x_mu).unflatten(1, (2, self.HP["input_features"] // 2))

        return x

    # Todo check if this is correct
    def cl_loss(self, geom, cl_trgt, viscous=True, return_CL=False, return_CD=False):
        """
        Loss function to enforce the generated airfoil profiles to have the target CLs.
        """
        cl, cd = eval_CL_XFOIL(geom.detach().cpu().numpy(), viscous=viscous)
        cl = torch.tensor(cl).float().to(self.device)

        # Create a boolean mask for non-NaN values in cl
        mask = ~torch.isnan(cl)

        # Calculate the ratio of the converged CLs
        convergence_ratio = torch.mean(mask.to(dtype=torch.float))

        # Todo this is terrible, fix this
        # convert mask to tensor
        cl_trgt_tn = torch.tensor(cl_trgt).float().to(self.device)

        # Calculate the loss without the nan values
        cl_loss = self.loss_mse(cl[mask], cl_trgt_tn[mask])

        # Todo fix this output handling
        if return_CD:
            return cl_loss, convergence_ratio, cl, cd
        elif return_CL:
            return cl_loss, convergence_ratio, cl
        else:
            return cl_loss, convergence_ratio

    # Todo check if this is correct
    def convexity_loss(self, geom, reduction="mean"):
        """
        Loss function to enforce convexity of the generated airfoil profiles.
        (Minimize the angle between two consecutive points.)
        geom: Batch size x 2 x n_features // 2 --> cnvx_loss: Batch size --> mean
        """

        # Todo: remove Old implementation, check if equivalent
        """         
        n_samples, _, n_features = geom.shape
        cnvx_loss = torch.zeros(n_samples).to(self.device)

        for u in range(n_samples):
            for v in range(n_features):
                vp = (v + 1) % (n_features - 1)
                vpp = (v + 2) % (n_features - 1)

                dxv = geom[u, 0, vp] - geom[u, 0, v]
                dxvp = geom[u, 0, vpp] - geom[u, 0, vp]
                dyv = geom[u, 1, vp] - geom[u, 1, v]
                dyvp = geom[u, 1, vpp] - geom[u, 1, vp]

                # Todo fix this / remoce
                #dxv, dyv, dxvp, dyvp = map(lambda x: torch.from_numpy(np.array(x)), [dxv, dyv, dxvp, dyvp])
                
                arg = (dxv*dxvp + dyv*dyvp) / torch.sqrt((dxv**2 + dyv**2) * (dxvp**2 + dyvp**2))
                arg = arg.clamp(-1.0, 1.0)
                cnvx_loss[u] += torch.arccos(arg) """

        # Efficient implementation using torch functions
        # Todo this can be done more efficently, first calculate all the differences then reorder
        # Calculate differences
        dx1 = geom[:, 0, 1:] - geom[:, 0, :-1]
        dx2 = geom[:, 0, 2:] - geom[:, 0, 1:-1]

        dy1 = geom[:, 1, 1:] - geom[:, 1, :-1]
        dy2 = geom[:, 1, 2:] - geom[:, 1, 1:-1]

        # Add wrap-around differences
        dx1 = torch.cat((dx1, (geom[:, 0, 0] - geom[:, 0, -1]).unsqueeze(1)), dim=1)
        dx2 = torch.cat(
            (dx2, (geom[:, 0, 0] - geom[:, 0, -1]).unsqueeze(1), (geom[:, 0, 1] - geom[:, 0, 0]).unsqueeze(1)), dim=1
        )

        dy1 = torch.cat((dy1, (geom[:, 1, 0] - geom[:, 1, -1]).unsqueeze(1)), dim=1)
        dy2 = torch.cat(
            (dy2, (geom[:, 1, 0] - geom[:, 1, -1]).unsqueeze(1), (geom[:, 1, 1] - geom[:, 1, 0]).unsqueeze(1)), dim=1
        )

        dot_product = dx1 * dx2 + dy1 * dy2
        magnitude = torch.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2))
        # magnitude = torch.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2)).clamp(min=1E-6)

        # Todo Remove Check for nan values in the differences
        """         if torch.isnan(dx1).any() or torch.isnan(dx2).any() or torch.isnan(dy1).any() or torch.isnan(dy2).any():
            raise ValueError("Nan values in differences!")
        # check for nan values in the dot product
        if torch.isnan(dot_product).any():
            raise ValueError("Nan values in dot product!")
        # check for nan values in the magnitude
        if torch.isnan(magnitude).any():
            raise ValueError("Nan values in magnitude!") """

        # angles = torch.acos((dot_product / (magnitude)).clamp(-1.0, 1.0))
        angles = dot_product / magnitude
        # angles = (1 - angles) * np.pi / 2 # Linear Approximator
        # Todo check if i can rearrange that for better performance
        angles = (np.pi / 2) - 0.7547 * angles - 0.8161 * angles**3  # Cubic Approximator
        # angles = (np.pi / 2) - 1.01377 * angles - 0.557027 * angles**7 # Septic Approximator
        cnvx_loss = angles.sum(dim=1)

        return torch.mean(cnvx_loss) if reduction == "mean" else cnvx_loss

    # Todo check if this is correct
    def smoothness_loss(self, geom, reduction="mean"):
        """
        Loss function to enforce smoothness of the generated airfoil profiles.
        (Minimize the difference between the original and filtered airfoil profiles.)
        """
        # geom = torch.from_numpy(geom).float().to(self.device)
        geom_flt = self.filter_op(geom)

        smthnss_loss = self.loss_mse(geom, geom_flt)  # Todo reduction=reduction

        return smthnss_loss

    # Todo select which derivative to evaluate
    def roughness_loss(self, geom, reduction="mean"):
        """
        Loss function to enforce roughness of the generated airfoil profiles.
        (Minimize the integral magnitude of higher derivatives.)
        """
        # If geom is a numpy array convert it to a torch tensor
        if isinstance(geom, np.ndarray):
            geom = torch.from_numpy(geom).float().to(self.device)

        # Assuming geom is a PyTorch tensor
        # derv_1 = torch.zeros_like(geom)

        # Shift the last element of the last dimension to the first position
        geom_p = torch.cat((geom[..., -1:], geom[..., :-1]), dim=-1)
        geom_p2 = torch.cat((geom_p[..., -1:], geom_p[..., :-1]), dim=-1)

        # Shift the first element of the last dimension to the last position
        geom_n = torch.cat((geom[..., 1:], geom[..., :1]), dim=-1)
        geom_n2 = torch.cat((geom_n[..., 1:], geom_n[..., :1]), dim=-1)

        # Finite Difference Approximation
        derv_1 = geom_p - geom_n
        derv_2 = geom_p + geom_n - 2 * geom
        derv_3 = 0.5 * geom_p2 - geom_p + geom_n - 0.5 * geom_n2

        # Square the derivatives, output dimensions: Batch x n_features
        mag_derv_1 = derv_1[:, 0, :] ** 2 + derv_1[:, 1, :] ** 2
        mag_derv_2 = derv_2[:, 0, :] ** 2 + derv_2[:, 1, :] ** 2
        mag_derv_3 = derv_3[:, 0, :] ** 2 + derv_3[:, 1, :] ** 2

        # Pseudo integrate i.e. sum the magnitudes
        P1 = torch.sum(mag_derv_1, dim=1)
        P2 = torch.sum(mag_derv_2, dim=1)
        P3 = torch.sum(mag_derv_3, dim=1)

        if reduction == "mean":
            P1 = torch.mean(P1)
            P2 = torch.mean(P2)
            P3 = torch.mean(P3)

        return P1, P2, P3


# Todo do not pass coords
def setup_DDPM(HP, coords, CLs, Model=None, print_summary=True):
    print("Setting up DDPM model...")

    # Create new model architecture instance from HP string
    if Model is None:
        nn_model = globals()[HP["Model_Architecture"]](n_feat=HP["input_features"])
    else:
        nn_model = Model

    # Create smoothing filter function for smoothing loss
    # Create a smoothing kernel
    kernel_size = 2 * HP["flt_half_support"] + 1
    kernel = (torch.ones(2, 1, kernel_size) / kernel_size).float().to(HP["device"])

    # Apply the smoothing kernel to each channel separately
    conv_op = torch.nn.Conv1d(
        2, 2, kernel_size, padding=HP["flt_half_support"], groups=2, padding_mode="circular", bias=False
    ).to(HP["device"])
    conv_op.weight = nn.Parameter(kernel)

    # Prevent the weights from being updated during training
    for param in conv_op.parameters():
        param.requires_grad = False

    ddpm = DDPM(
        nn_model=nn_model,
        filter_op=conv_op,
        HP=HP,
        x_mu=coords["arr_1"],
        x_std=coords["arr_2"],
        cl_mu=CLs["arr_1"],
        cl_std=CLs["arr_2"],
    )

    # Print model summary: number of parameters, layers, etc.
    if print_summary:
        print("Model Summary:")
        model_stats, model_params, model_size = model_summary(
            ddpm.nn_model,
            [(2, HP["input_features"] // 2), (1,), (1,)],
            batch_size=HP["batch_size"],
            device=HP["device"],
        )
        HP["N-Parameters"] = int(model_params)
        HP["Model-Size"] = float(model_size)
        # Todo Convert model to String representation
        # model_stats = str(model_stats).replace("\n", "<br>")
        # HP["Summary"] = model_stats
    return ddpm


# Todo make object function of ddpm class
def eval_DDPM(ddpm, n_variation=1, save=True, str_label=""):
    # print("Sampling from DDPM model...")
    ddpm.eval()

    with torch.no_grad():
        cl_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

        n_class = cl_list.__len__()
        n_sample = n_variation * n_class

        geom, x_gen_store = ddpm.sample(cl_list, n_variation)
        geom_cpu = (
            geom.detach().cpu().numpy()
        )  # Convert to numpy array, detach from gradient computation, and move to CPU

        if save:
            for i in range(n_sample):
                np.savetxt(
                    "./04 RUNS/4 Output/2 Temporary/sample"
                    + "_cl_"
                    + str(cl_list[i % n_class])
                    + "_"
                    + str(i // n_class)
                    + str_label
                    + ".txt",
                    geom_cpu[i, :, :].squeeze().T,
                )

    return geom


def eval_opt_LR(HP, ddpm, optim, pbar, ep, lr, decades=1.5, EMA_factor=0.5, LR_Monotonic=False):
    """
    Evaluate the optimal learning rate for the optimizer.

    Save checkpoint
    Save current losses
    Set lr_low, lr_high around current learning rate, e.g., +- one decade

    For sample in epoch
    Set lr linear between lr_low and lr_high
    Evaluate and save current losses, backpropagate, and update weights

    Calculate loss change: losses_i -1 – losses_i
    If necessary, smooth result
    Choose optimal learning rate

    Load checkpoint
    """

    loss_history = []
    lr_history = []

    curr_lr = optim.param_groups[0]["lr"]
    lr_low = curr_lr / (10**decades)
    lr_high = curr_lr * (10**decades)
    num_batches = len(pbar)

    lr_set = np.geomspace(lr_low, lr_high, num_batches)

    i = 0

    for x, c in pbar:

        # Set learning rate
        lr_0 = lr_set[i]

        for param_group in optim.param_groups:
            param_group["lr"] = lr_0

        optim.zero_grad()
        x = x.to(HP["device"])
        c = c.to(HP["device"])

        if HP["cnvxty_loss_W"] == 0:
            diff_loss = ddpm(x, c)
            trng_loss = (
                HP["diff_loss_W"] * diff_loss
            )  # + HP["cl_loss_W"] * cl_loss + HP["cnvxty_loss_W"] * cnvxty_loss + HP["smthnss_loss_W"] * smthnss_loss
        else:
            diff_loss, x0_pred = ddpm(x, c, eval_x0_pred=True)

            geom = ddpm.renormalize(x0_pred)
            cnvxty_loss = ddpm.convexity_loss(geom)

            trng_loss = (
                HP["diff_loss_W"] * diff_loss + HP["cnvxty_loss_W"] * cnvxty_loss
            )  # + HP["cl_loss_W"] * cl_loss + HP["smthnss_loss_W"] * smthnss_loss

        trng_loss.backward()

        pbar.set_description(f"Evaluating optimal learning rate: {lr_0:.4f}")

        optim.step()

        lr_history.append(lr_0)
        loss_history.append(trng_loss.item())

        i += 1

    # Save loss and learning rate history to file
    np.save(
        "./04 RUNS/4 Output/2 Temporary/loss_eval_" + HP["Identifier"] + "_EP" + str(ep), [lr_history, loss_history]
    )

    # convert to numpy
    lr_history = np.array(lr_history)
    loss_history = np.array(loss_history)

    crit_lr, lr_min, opt_lr = evaluate_optimal_lr(lr_history, loss_history)

    if opt_lr > lr and LR_Monotonic:
        opt_lr = lr

    opt_lr = EMA_factor * opt_lr + (1 - EMA_factor) * lr

    return opt_lr

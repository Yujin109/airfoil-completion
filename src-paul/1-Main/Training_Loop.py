import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("./03 CODE/6 Utils")
sys.path.append("./03 CODE/2 DDPM")
from Util_Lib import powerlaw_decay, average_runtime, estimate_performance
from DDPM import eval_DDPM, eval_opt_LR


def train_DDPM(ddpm, optim, HP, data_loader, model_metrics):
    print("Training DDPM model...")

    #Debugging
    #torch.autograd.set_detect_anomaly(True)
    
    # Tensorboard TB_writer
    TB_writer = SummaryWriter(log_dir="./04 RUNS/1 TB Logs/" + HP["Identifier"], filename_suffix=".TB_LOG")
    
    # Log hyperparameters and metrics
    TB_writer.add_hparams(hparam_dict=HP, metric_dict=model_metrics, run_name="HP-Metrics")
    
    # Fix graph visualization
    # TB_writer.add_graph(ddpm.nn_model, 
                        # (torch.randn(1, HP["input_features"]).to(device), 
                        #  torch.randn(1, 1).to(device), 
                        #  torch.randn(1, 1).to(device)))
    
    # Estimate statistics of the diffusion process
    latent_mean = 1.0
    latent_stddev = 0.0    
    for ts in range(ddpm.n_T):
        latent_mean = ddpm.sqrtab[ts]
        latent_stddev = ddpm.sqrtmab[ts]
        if ts % (ddpm.n_T//100) == 0 or ts == ddpm.n_T - 1:
            TB_writer.add_scalar("Latent Mean", latent_mean, ts)
            TB_writer.add_scalar("Latent Std. Dev.", latent_stddev, ts)
            TB_writer.add_scalar("Diffusion Schedule", ddpm.beta_t[ts], ts)

    trng_loss_ema = None
    diff_loss_ema = None
    cnvxty_loss_ema = None
    #roughnss_loss_ema = None
    
    ddpm.train()
    for ep in range(HP["n_restart"], HP["n_epoch"] + 1):    
        
        if False:
            if ep < HP["LR_Eval_Interval"]:
                lr = powerlaw_decay(HP["initial_lr"], HP["final_lr"], HP["alpha"], (ep - HP["n_restart"]) / (HP["n_epoch"] - HP["n_restart"]))
            elif ep % HP["LR_Eval_Interval"] == 0:
                # Evaluate optimal learning rate
                torch.save({
                    'model_state': ddpm.state_dict(),
                    'optimizer_state': optim.state_dict(),
                    'model_architecture': ddpm.nn_model,
                    'HP': HP,
                    }, "./04 RUNS/2 Checkpoints/" + HP["Identifier"] + '_tmp.pt')
                print("Evaluating optimal learning rate...")
                lr = eval_opt_LR(HP, ddpm, optim, pbar, ep, lr)
                print(f"Optimal learning rate: {lr}")
                checkpoint = torch.load("./04 RUNS/2 Checkpoints/" + HP["Identifier"] + '_tmp.pt')
                ddpm.load_state_dict(checkpoint['model_state'])
                optim.load_state_dict(checkpoint['optimizer_state'])

        # Retraining
        if ep == HP["n_restart"] and True:
            lr = optim.param_groups[0]['lr']

        if False: # For Retraining
            c_ep = ep - HP["n_restart"]
            if c_ep < 500:
                lr = powerlaw_decay(0, HP["initial_lr"], HP["alpha"], c_ep / 500)
            else:
                lr = powerlaw_decay(HP["initial_lr"], 0, HP["alpha"], (c_ep - 500) / (HP["n_epoch"] - HP["n_restart"] - 500))
        else:
            # Powerlaw / Linearly reduce learning rate to zero
            # lr = powerlaw_decay(HP["initial_lr"], HP["final_lr"], HP["alpha"], (ep - HP["n_restart"]) / (HP["n_epoch"] - HP["n_restart"]))
            
            # Initial powerlaw, switches to exponential decay after 1000 epochs
            lr = powerlaw_decay(HP["initial_lr"], HP["final_lr"], HP["alpha"], (ep - HP["n_restart"]) / (10000 - HP["n_restart"])) if ep < 1000 else lr * 0.99954
        
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        pbar = tqdm(data_loader)
        trng_loss_ema = None
        diff_loss_ema = None
        cnvxty_loss_ema = None
        #roughnss_loss_ema = None

        for x, c in pbar:
            optim.zero_grad()
            x = x.to(HP["device"])
            c = c.to(HP["device"])

            diff_loss, x0_pred, ts = ddpm(x, c, eval_x0_pred=True)

            if HP["cnvxty_loss_W"] != 0 or HP["rougnss_loss_W"] != 0:
                geom = ddpm.renormalize(x0_pred)

            # Set Geometry predition probabilistic to zero, if the time step is very early. Then the loss and its gradient will be zero too.
            if HP["probabilistic_geom_loss"]:
                ts_bar = torch.randint(1, ddpm.n_T, (x.shape[0],)).to(ddpm.device)  # t ~ Uniform(0, n_T)
                mask = ts >= ts_bar
                geom = geom[mask,:,:]

            if HP["cnvxty_loss_W"] != 0:
                cnvxty_loss = ddpm.convexity_loss(geom)
            else:
                cnvxty_loss = 0
                
            if HP["rougnss_loss_W"] != 0:
                roughnss_loss = ddpm.roughness_loss(geom)
            else:
                roughnss_loss = [0,0,0]

            # Trains on P2 roughness
            trng_loss = HP["diff_loss_W"] * diff_loss + HP["cnvxty_loss_W"] * cnvxty_loss + HP["rougnss_loss_W"] * roughnss_loss[1] # + HP["cl_loss_W"] * cl_loss + HP["smthnss_loss_W"] * smthnss_loss
            
            trng_loss.backward()

            # Log gradient norm, comes roughly with 20% overhead
            if HP["Eval_Gradient_Norm"]:
                grads = [param.grad.detach().flatten() for param in ddpm.parameters() if param.grad is not None]
                grad_norm = torch.cat(grads).norm()
                TB_writer.add_scalar("Gradient-Norm", grad_norm.item(), ep)
            
            # Exponential moving average of the losses for tensorboard
            if trng_loss_ema is None:
                trng_loss_ema = trng_loss.item()
                diff_loss_ema = diff_loss.item()
                if HP["cnvxty_loss_W"] != 0:
                    cnvxty_loss_ema = cnvxty_loss.item()
            else:
                trng_loss_ema = HP["EMA_Factor"] * trng_loss_ema + (1 - HP["EMA_Factor"]) * trng_loss.item()
                diff_loss_ema = HP["EMA_Factor"] * diff_loss_ema + (1 - HP["EMA_Factor"]) * diff_loss.item()
                if HP["cnvxty_loss_W"] != 0:
                    cnvxty_loss_ema = HP["EMA_Factor"] * cnvxty_loss_ema + (1 - HP["EMA_Factor"]) * cnvxty_loss.item()
            pbar.set_description(f"Epoch {ep}, loss: {trng_loss_ema:.5f}")
            
            # Todo Clip gradients
            #torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1E2)
            optim.step()


        # Log metrics
        TB_writer.add_scalar("Training-Loss", trng_loss_ema, ep)
        TB_writer.add_scalar("Diffusion-Loss", diff_loss_ema, ep)
        if HP["cnvxty_loss_W"] != 0:
            TB_writer.add_scalar("Convexity-Loss/Training", cnvxty_loss_ema, ep)
        if HP["rougnss_loss_W"] != 0:
            TB_writer.add_scalar("Roughness-Loss-P1/Training", roughnss_loss[0], ep)
            TB_writer.add_scalar("Roughness-Loss-P2/Training", roughnss_loss[1], ep)
            TB_writer.add_scalar("Roughness-Loss-P3/Training", roughnss_loss[2], ep)
        TB_writer.add_scalar("Learning-Rate", lr, ep)
        TB_writer.add_scalar("Convexity-Loss-Weight", HP["cnvxty_loss_W"], ep)
        TB_writer.add_scalar("Roughness-Loss-Weight", HP["rougnss_loss_W"], ep)
        # TB_writer.add_scalar("Diffusion-Loss-Weight", HP["diff_loss_W"], ep)
        # TB_writer.add_scalar("CL-Loss-Weight", HP["cl_loss_W"], ep)
        # TB_writer.add_scalar("Smoothness-Loss-Weight", HP["smthnss_loss_W"], ep)
        
        # Evaluate Metrics, plot Gradients, sample from the model and save the generated samples
        if ep % HP["Eval_Epoch_Interval"] == 0 or ep in [1, 5, 10, 25, 50, 100]:
            # Plot gradients
            if HP["Track_Gradients"]:
                for name, param in ddpm.named_parameters():
                    if param.grad is not None:
                        TB_writer.add_histogram(f'{name}.grad', param.grad, ep)
            # Plot parameters
            if HP["Track_Parameters"]:
                for name, param in ddpm.named_parameters():
                    TB_writer.add_histogram(f'{name}', param, ep)

            #Generate samples, save to temporary folder
            # Todo plot on Tensorboard
            # add_figure(tag, figure, global_step=None, close=True, walltime=None)
            geom = eval_DDPM(ddpm, n_variation=2, str_label="_ep"+str(ep).zfill(5))

            # Evaluate further losses
            with torch.no_grad():
                # Todo fix target input --> see eval_DDPM
                cnvxty_loss = ddpm.convexity_loss(geom)
                smthns_loss = ddpm.smoothness_loss(geom)
                roughnss_loss = ddpm.roughness_loss(geom)

                # Todo find bounds and move threshold to HP
                if cnvxty_loss < 50:
                    cl_trgt = torch.tensor([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]).float().to(HP["device"]).repeat(2)
                    cl_loss, convergence_ratio = ddpm.cl_loss(geom, cl_trgt, viscous=True)
                else:
                    cl_loss = float("NAN")
                    convergence_ratio = float("NAN")

            # Log metrics
            TB_writer.add_scalar("Convexity-Loss/Sampling", cnvxty_loss, ep)
            TB_writer.add_scalar("Smoothness-Loss/Sampling", smthns_loss, ep)
            TB_writer.add_scalar("Roughness-Loss-P1/Sampling", roughnss_loss[0], ep)
            TB_writer.add_scalar("Roughness-Loss-P2/Sampling", roughnss_loss[1], ep)
            TB_writer.add_scalar("Roughness-Loss-P3/Sampling", roughnss_loss[2], ep)
            TB_writer.add_scalar("CL-Loss/Sampling", cl_loss, ep)
            TB_writer.add_scalar("CL-Convergence-Ratio/Sampling", convergence_ratio, ep)


            ddpm.train()

        # Save model checkpoint
        if ep % HP["Checkpoint_Interval"] == 0 and HP["save_model"]:
            ident_Str = HP["Identifier"]
            ident_Str = ident_Str.split("_EP")[0]
            ident_Str = ident_Str + "_EP" + str(ep)
            torch.save({
                'model_state': ddpm.state_dict(),
                'optimizer_state': optim.state_dict(),
                'model_architecture': ddpm.nn_model,
                'HP': HP,
            }, "./04 RUNS/2 Checkpoints/" + ident_Str + '.pt')
    
    # Evaluate wall time needed for sampling
    print("Evaluating average sampling time:")
    avg_time = average_runtime(eval_DDPM, args=(ddpm, 1, False), runs=6)
    # Todo, do this The training time is logged in tesorbord and can be seen when hovering over the training curve

    # Evaluate final performance
    # Todo roughness loss final metric
    print("Evaluating final performance:")
    cnvxty_loss, smthns_loss, cl_loss, convergence_ratio, roughnss_loss = estimate_performance(ddpm, HP, n_variation=8, XFoil_viscous=True)

    # Log final metrics
    model_metrics["Diffusion-Loss"] = diff_loss_ema
    model_metrics["CL-Loss"] = cl_loss
    model_metrics["CL-Convergence-Ratio"] = convergence_ratio
    model_metrics["Convexity-Loss"] = cnvxty_loss
    model_metrics["Smoothness-Loss"] = smthns_loss
    model_metrics["Roughness-Loss"] = roughnss_loss
    model_metrics["Avg-Sampling-Time"] = avg_time

    # Log metrics
    TB_writer.add_hparams(hparam_dict=HP, metric_dict=model_metrics, run_name="HP-Metrics")
    
    # Finish tensorboard TB_writer    
    TB_writer.flush()
    TB_writer.close()
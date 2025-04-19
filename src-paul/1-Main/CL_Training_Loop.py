import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("./03 CODE/2 DDPM")
sys.path.append("./03 CODE/6 Utils")
sys.path.append("./03 CODE/8 X-Foil/Python_DIR")
from Util_Lib import powerlaw_decay
from DDPM import eval_DDPM
from xfoil_runner import eval_CL_XFOIL

#Todo:
# - If CL NAN or out of bounds then cl loss = ...


def train_CL_DDPM(ddpm, HP, model_metrics, epoch_size=100, CL_max=1.2, CL_min=0.5):
    print("Training DDPM model...")
    
    # If no Tensorboard path is given, use current date and time for directory
    if HP["TB_Path"] == "":
        HP["TB_Path"] = "03 CODE\9 Logs\TB_logs\DDPM_" + HP["Date-Time"]
    
    # Tensorboard TB_writer
    TB_writer = SummaryWriter(log_dir=HP["TB_Path"], filename_suffix=".TB_LOG")

    optim = torch.optim.Adam(ddpm.parameters(), lr=HP["initial_lr"])
    
    # Todo only add hparams once with metrics
    TB_writer.add_hparams(hparam_dict=HP, metric_dict={}, run_name=HP["Date-Time"])

    loss_ema = None
    ddpm.train()
    for ep in range(HP["n_restart"], HP["n_epoch"] + 1):    

        # Powerlaw / Linearly reduce learning rate to zero
        lr = powerlaw_decay(HP["initial_lr"], HP["final_lr"], HP["alpha"], (ep - HP["n_restart"]) / (HP["n_epoch"] - HP["n_restart"]))
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        pbar = tqdm(range(epoch_size))
        loss_ema = None
        for idx in pbar:

            CLs = torch.rand(HP["batch_size"], 1).to(HP["device"])
            CLs = CL_min + (CL_max - CL_min) * CLs

            optim.zero_grad()

            geom = ddpm.sample(CLs)[0]
            
            #Calculate smoothing / filter loss
            if HP["flt_loss_weight"] == 0:
                filter_loss = 0
            else:
                # Moving average over coordinates to smooth out remaining noise
                geom_flt = ddpm.apply_filter(geom)

                filter_loss = ddpm.loss_mse(geom, geom_flt)
            
            # Calculate target cl loss
            if HP["CL_loss_weight"] == 0:
                cl_loss = 0
            else:
                geom = geom.detach().cpu().numpy()
                ddpm_cl, cd = eval_CL_XFOIL(geom)
                ddpm_cl = torch.tensor(ddpm_cl).float().to(HP["device"])
                cl_loss = ddpm.loss_mse(CLs, ddpm_cl)


            loss = cl_loss * HP["CL_loss_weight"] + filter_loss * HP["flt_loss_weight"]
            #loss = cl_loss * HP["CL_loss_weight"]
            #loss = filter_loss * HP["flt_loss_weight"]
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"Epoch {ep}, loss: {loss_ema:.4f}")
            optim.step()

        
        TB_writer.add_scalar("CL-Training-Loss", loss, ep)
        TB_writer.add_scalar("CL-Loss", cl_loss, ep)
        TB_writer.add_scalar("Filter-Loss", filter_loss, ep)
        TB_writer.add_scalar("Learning-Rate", lr, ep)
        TB_writer.add_scalar("Filter-Loss-Weight", HP["flt_loss_weight"], ep)
        TB_writer.add_scalar("CL-Loss-Weight", HP["CL_loss_weight"], ep)
        
        # Plot Gradients, Sample from the model and save the generated samples
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
            x_gen = eval_DDPM(ddpm, n_variation=1, str_label="_ep"+str(ep).zfill(5))
            ddpm.train()

        # Todo save model
        """  if (ep+1)%n_T == 0:
            # for eval, save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            ddpm.eval()
            with torch.no_grad():
                #wandb_log_dict = dict()
                n_sample = 800 #4*n_classes
                for w_i, w in enumerate(ws_test):
                    x_gen, x_gen_store = ddpm.sample(n_sample, (1, x_dim), device, guide_w=w)
                    x_gen = x_gen.detach().cpu().numpy()
                    for i in range(5):
                        plt.scatter(x_gen[i,0,:248],x_gen[i,0,248:])
                        plt.savefig('../../tmp/cdiffusion_ep'+str(ep)+'_'+str(i)+'.png')
                        plt.clf()
                    index = 'sample_img_guidance' + str(w)
                    #wandb_log_dict[index] = [wandb.Image(Image.open('../../tmp/cdiffusion'+str(i)+'.png')) for i in range(5)]
            #torch.save(ddpm.state_dict(), os.path.join(wandb.run.dir, 'ddpm_'+str(ep+1)+'.pth'))
            torch.save(ddpm.state_dict(), os.path.join('../../tmp/ddpm_'+str(ep+1)+'.pth'))
            #wandb.log(wandb_log_dict) """
    
    model_metrics["EMA-Loss"] = loss_ema
    TB_writer.add_hparams(hparam_dict=HP, metric_dict=model_metrics, run_name=HP["Date-Time"])
    
    # Finish tensorboard TB_writer    
    TB_writer.flush()
    TB_writer.close()
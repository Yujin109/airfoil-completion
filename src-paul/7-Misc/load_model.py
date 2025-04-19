import numpy as np


# Load the hyperparameters
#hp_path = "03 CODE/9 Logs/TB_logs/DDPM_HP_2024-04-27_15-33-361500.npz"
hp_path = "03 CODE/9 Logs/TB_logs/DDPM_HP_2024-04-27_23-23-43_ep_2000.npz"

hp = np.load(hp_path, allow_pickle=True)

# print each item in the dictionary
for key, value in hp.items():
    print(key, value)

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

# Load the model
# ddpm.load_state_dict(torch.load('../../tmp/ddpm_'+str(9499+1)+'.pth',map_location=torch.device('cpu') ))
# ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))
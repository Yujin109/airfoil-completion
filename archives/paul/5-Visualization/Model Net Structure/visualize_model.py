from torchviz import make_dot
import torch
import sys

# Visualize the model graph using another library
# https://github.com/mert-kurttutan/torchview/blob/main/README.md
import torchvision
from torchview import draw_graph

sys.path.append("./03 CODE/3 Models")
from Network_Definitions import *

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


#U-Net
""" n_T = 500
n_feat = 256

# un_model = ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).to(device)
un_model = ContextUnet(in_channels=1, n_feat=n_feat).to(device)

# Create a random input sample
x = torch.load("./03 CODE/5 Visualization/Model Net Structure/x.pt").to(device)
c = torch.load("./03 CODE/5 Visualization/Model Net Structure/c.pt").to(device)
t = torch.load("./03 CODE/5 Visualization/Model Net Structure/t.pt").to(device)
context_mask = torch.load("./03 CODE/5 Visualization/Model Net Structure/context_mask.pt").to(device)

print(x.shape)
print(c.shape)
print(t.shape)
print(context_mask.shape)

yhat = un_model(x, c, t, context_mask).to(device) # Give dummy batch to forward().

make_dot(yhat, params=dict(list(un_model.named_parameters()))).render("rnn_torchviz", format="png")

model_graph = draw_graph(un_model, input_size=[(128, 1, 496), (128, 1), (128, 1), (128, 1)], expand_nested=True)
model_graph.visual_graph
model_graph.visual_graph.render(format='svg') """

# Legacy Linear-Convolutional Network (Batch x Features) Format
"""
model = Lin_Conv_Network_4(496).to(device)

x = torch.rand(128, 496).to(device)
c = torch.rand(128, 1).to(device)
t = torch.rand(128, 1).to(device)

out = model(x, c, t).to(device) # Give dummy batch to forward().
make_dot(out, params=dict(list(model.named_parameters()))).render("Linear-Conv-NN_4", format="png")

model_graph = draw_graph(model, input_size=[(128, 496), (128, 1), (128, 1)], expand_nested=True)
model_graph.visual_graph
model_graph.visual_graph.render(filename="Linear-Conv-NN_4", format='svg')
"""

model_name = "UNet"
model = UNet(n_feat=496).to(device)

x = torch.rand(128, 2, 496//2).to(device)
c = torch.rand(128, 1).to(device)
t = torch.rand(128, 1).to(device)

# I dont like this visualization
# out = model(x, c, t).to(device) # Give dummy batch to forward().
# make_dot(out, params=dict(list(model.named_parameters()))).render(model_name, format="png")

model_graph = draw_graph(model, input_size=[(128, 2, 496//2), (128, 1), (128, 1)], expand_nested=True, graph_dir="TB")
model_graph.visual_graph
model_graph.visual_graph.render(filename=model_name, format='svg')
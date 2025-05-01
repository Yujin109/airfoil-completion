import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def smoothing_filter(x):
    """
    Smoothing filter to remove remaining noise from the generated samples.
    x has dimensions [batch_size, features]
    """
    
    assert x.shape[1] % 2 == 0, "Number of input features must be even."
    n_coords = x.shape[1]//2
    
    flt_half_support = 4
    
    # Split x into x and y coordinates
    # create empty torch tensor
    x_split = torch.zeros((x.shape[0], 2, n_coords)).float()
    x_split[:, 0, :] = x[:, :n_coords]
    x_split[:, 1, :] = x[:, n_coords:]
    
    # # Create a smoothing kernel
    kernel_size = 2 * flt_half_support + 1
    kernel = torch.ones(2, 1, kernel_size) / kernel_size
    
    # Apply the smoothing kernel to each channel separately
    # x_smoothed = F.conv1d(x_split, kernel, stride=1, padding=flt_half_support, padding_mode='circular')
    conv_op = torch.nn.Conv1d(2, 2, kernel_size, padding=flt_half_support, groups=2, padding_mode='circular', bias=False)
    conv_op.weight = nn.Parameter(kernel)
    
    # Prevent the weights from being updated during training
    for param in conv_op.parameters():
        param.requires_grad = False
    
    x_split = conv_op(x_split)
    
    x_split = torch.cat((x_split[:, 0, :], x_split[:, 1, :]), -1)

    # for u in range(n_coords):
    #     x_smoothed[:, u, :] = torch.sum(x_split[:, u-flt_half_support:u+flt_half_support, :], 1) / ((2 * flt_half_support) + 1)
                
    # x_ret = torch.cat((x_smoothed[:, :, 0], x_smoothed[:, :, 1]), 1)
    
    return x_split


# load airfoil shape from file
file_path = "./03 CODE/temporary/sample_cl_0.6_0_ep00150.txt"

gcoords = []
with open(file_path, "r") as file:
    for line in file:
        x, y = line.strip().split()
        gcoords.append([float(x), float(y)])

#coords = torch.zeros((1, 2*len(gcoords))).float()
#coords[0, :] = torch.tensor(gcoords).T.float().flatten()

coords = torch.rand((10, 2*len(gcoords))).float()

n_coords = coords.shape[1]//2 

print(coords.shape)

# Smoothing filter
coords_smoothed = smoothing_filter(coords)

print(coords_smoothed.shape)

filter_loss = nn.MSELoss()(coords, coords_smoothed)
print(filter_loss.item())

# Plot both airfoil shapes
plt.figure(figsize=(10, 5))

# Plot original and smoothed shape
plt.subplot(1, 2, 1)
plt.scatter(coords[0, :n_coords], coords[0, n_coords:], c=range(n_coords), label="Original")
plt.axis("equal")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Original Airfoil Shape")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(coords_smoothed[0, :n_coords], coords_smoothed[0, n_coords:], c=range(n_coords), label="Smoothed")
plt.axis("equal")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Smoothed Airfoil Shape")
plt.legend()

plt.show()
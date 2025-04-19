import numpy as np
import torch

import matplotlib.pyplot as plt

x_scale = 0.10

# create list of theta interpolation angles
theta_interp = np.linspace(-1, 1, 100) * np.pi

coords = np.load("./03 CODE/1 Yonekura/2nd/wing_data/standardized_NandJ_coords.npz")
coords = torch.tensor(coords['arr_0']).float().squeeze()

# single airfoil
coord = coords[0,:]

# Extract x and y coordinates from the array
coord = coord.numpy()
coord = coord.reshape(2, -1)

# normalize the x and y coordinates
coord[0,:] = (coord[0,:] - np.mean(coord[0,:])) * x_scale
coord[1,:] = coord[1,:] - np.mean(coord[1,:])

# transform to polar coordinates
r = np.sqrt(coord[0,:]**2 + coord[1,:]**2)
theta = np.arctan2(coord[1,:], coord[0,:])

# interpolate the polar coordinates to the new theta values
r_interp = np.interp(theta_interp, theta, r, period=2*np.pi)

# transform back to cartesian coordinates
coord_interp = np.zeros((2, len(theta_interp)))
coord_interp[0,:] = r_interp * np.cos(theta_interp) / x_scale
coord_interp[1,:] = r_interp * np.sin(theta_interp)


# Plot the cartesian coordinates and the polar coordinates as subplots
fig, axs = plt.subplots(2)
axs[0].scatter(coord[0,:], coord[1,:])
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('Coordinate Plot: ')
axs[0].grid(True)
axs[0].set_aspect('equal')

axs[0].scatter(coord_interp[0,:], coord_interp[1,:], color='red')

axs[1].scatter(theta, r)
axs[1].set_xlabel('Theta')
axs[1].set_ylabel('R')
axs[1].set_title('Coordinate Plot: ')
#axs[1].set_xlim(-np.pi, 0)
axs[1].grid(True)

axs[1].scatter(theta_interp, r_interp, color='red')
plt.show()


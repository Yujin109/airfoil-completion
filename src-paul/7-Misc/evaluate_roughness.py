import numpy as np
import matplotlib.pyplot as plt

def drvtv(geom):
    del1 = np.zeros_like(geom)

    # Shift the last element of the last dimension to the first position
    geom_p = np.concatenate((geom[..., -1:], geom[..., :-1]), axis=-1)

    # Shift the first element of the last dimension to the last position
    geom_n = np.concatenate((geom[..., 1:], geom[..., :1]), axis=-1)

    del1 = geom_p - geom_n

    return del1

def sec_drvtv(geom):
    del2 = np.zeros_like(geom)

    # Shift the last element of the last dimension to the first position
    geom_p = np.concatenate((geom[..., -1:], geom[..., :-1]), axis=-1)

    # Shift the first element of the last dimension to the last position
    geom_n = np.concatenate((geom[..., 1:], geom[..., :1]), axis=-1)

    del2 = geom_p + geom_n - 2 * geom

    return del2

def trd_drvtv(geom):
    del3 = np.zeros_like(geom)

    # Shift the last element of the last dimension to the first position
    geom_p = np.concatenate((geom[..., -1:], geom[..., :-1]), axis=-1)
    geom_p2 = np.concatenate((geom_p[..., -1:], geom_p[..., :-1]), axis=-1)

    # Shift the first element of the last dimension to the last position
    geom_n = np.concatenate((geom[..., 1:], geom[..., :1]), axis=-1)
    geom_n2 = np.concatenate((geom_n[..., 1:], geom_n[..., :1]), axis=-1)

    del3 = 0.5*geom_p2 - geom_p + geom_n - 0.5*geom_n2

    return del3

geometry_data = "./03 CODE/4 Data/normalized_NandJ_coords.npz"

# Load geometry and performance data
# X['arr_0'] ~ Data, X['arr_1'] ~ Mean, X['arr_2'] ~ Std Dev
coords = np.load(geometry_data)

x = coords['arr_0'][500]

#renormalize
geom = x * coords['arr_2'] + coords['arr_1']

# Sample circle as test geometry
if False:
    geom = np.zeros((2, 248))
    for i in range(248):
        geom[0, i] = np.cos(2 * np.pi * i / 248)
        geom[1, i] = np.sin(2 * np.pi * i / 248)

geom = geom.reshape(-1, 248)

# Calculate the first derivative
del1 = drvtv(geom)

# Calculate the second derivative
del2 = sec_drvtv(geom)

# Calculate the third derivative
del3 = trd_drvtv(geom)

# sqaure the derivatives
mag_del1 = del1[0,:]**2 + del1[1,:]**2
mag_del2 = del2[0,:]**2 + del2[1,:]**2
mag_del3 = del3[0,:]**2 + del3[1,:]**2

# pseudo integrate i.e. sum the magnitudes
P1 = np.sum(mag_del1)
P2 = np.sum(mag_del2)
P3 = np.sum(mag_del3)

print(P1)
print(P2)
print(P3)

# Plot the magnitude of the second derivative and third derivative
plt.figure(figsize=(10, 5))
plt.plot(mag_del1, label="First Derivative")
plt.plot(mag_del2, label="Second Derivative")
plt.plot(mag_del3, label="Third Derivative")
plt.yscale("log")
plt.xlabel("Index")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(geom[0,:], geom[1,:], c=mag_del2, cmap='viridis')
for i in range(1, 248, 4):
    plt.arrow(geom[0,i], geom[1,i], del2[0,i], del2[1,i], head_width=0.005, head_length=0.02, fc='red', ec='red')
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')

plt.show()

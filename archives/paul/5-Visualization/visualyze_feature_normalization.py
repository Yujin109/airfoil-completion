import numpy as np
import matplotlib.pyplot as plt

coords = np.load("./03 CODE/4 Data/NandJ_coords_V2.npz")
n_feature = coords['arr_0'].shape[1]//2

# plot mean and std versus index
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(coords['arr_1'])
axs[0].set_title('Mean')
axs[1].plot(coords['arr_2'])
axs[1].set_title('Std Dev')

plt.show()

# plot mean as x and y coordinates
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(coords['arr_1'][:n_feature], coords['arr_1'][n_feature:], s=1, label='Mean')
ax.scatter(coords['arr_1'][:n_feature] + coords['arr_2'][:n_feature], coords['arr_1'][n_feature:] + coords['arr_2'][n_feature:], s=1, label='Mean + Std Dev')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Mean')
ax.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

coords = np.load("./03 CODE/4 Data/NandJ_coords_V2.npz")
CLs = np.load("./03 CODE/4 Data/NandJ_perfs_V2.npz")
CDs = np.load("./03 CODE/4 Data/recalculated_NandJ_CDs.npz")
index_list = np.load("./03 CODE/4 Data/index_list.npz")['arr_0']
index_list = index_list.astype(int)

old_CLs = np.load("./03 CODE/4 Data/Yonekura/standardized_NandJ_perfs.npz")

# Remove index list from the data
print("Removing", len(index_list), "samples.")
CDs = np.delete(CDs['arr_0'], index_list, axis=0)

# Plot the histograms of the cl and cd values in two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].hist(CLs['arr_0'], bins=50, color='blue', alpha=0.7, label="normalized")
axs[0].hist(CLs['arr_0']*CLs["arr_2"] + CLs["arr_1"], bins=50, color='red', alpha=0.7, label="true")
axs[0].set_title('CL Distribution')
axs[0].legend()

axs[1].hist(CDs, bins=50, color='red', alpha=0.7)
axs[1].set_title('CD Distribution')

plt.show()

# Scatter plot of the cl and cd values
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(CLs['arr_0']*CLs["arr_2"] + CLs["arr_1"], CDs, s=1)
ax.set_xlabel('CL')
ax.set_ylabel('CD')
ax.set_title('CL vs CD')
plt.show()

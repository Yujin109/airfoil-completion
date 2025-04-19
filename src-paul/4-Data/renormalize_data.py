import numpy as np
import matplotlib.pyplot as plt

normalize = True

features = 496
features_2 = features//2

# Load the data
coords_npz = np.load("./03 CODE/4 Data/Yonekura/standardized_NandJ_coords.npz")
cls_npz = np.load("./03 CODE/4 Data/recalculated_standardized_NandJ_perfs.npz")
index_list = np.load("./03 CODE/4 Data/index_list.npz")['arr_0']
index_list = index_list.astype(int)

# Remove index list from the data
print("Removing", len(index_list), "samples.")
coords_array = np.delete(coords_npz["arr_0"], index_list, axis=0)
cls_array = np.delete(cls_npz["arr_0"], index_list, axis=0)

# Access the arrays from the original NPZ files
geometry = coords_array
print(geometry.shape)

norm_geometry = np.zeros(geometry.shape)

# Standardize individual samples w.r.t. x and y coordinates respectively, s.t. each airfoil is zero centered. Scaling for unit standard deviation however is not possible. Alternatively, the airfoils are uniformly scaled for a cord length of 1.
for u in range(geometry.shape[0]):
    norm_geometry[u, :features_2] = geometry[u, :features_2] - np.mean(geometry[u, :features_2])
    norm_geometry[u, features_2:] = geometry[u, features_2:] - np.mean(geometry[u, features_2:])
    cord_length = np.max(norm_geometry[u, :features_2]) - np.min(norm_geometry[u, :features_2])
    norm_geometry[u, :] = norm_geometry[u, :] / cord_length


if normalize:
    # Calculate mean and standard deviation of the standardized data
    norm_geometry_mean = np.mean(norm_geometry, axis=0)
    norm_geometry_std = np.std(norm_geometry, axis=0)
    
    # Normalize features
    norm_geometry = (norm_geometry - norm_geometry_mean) / norm_geometry_std
    
    #print(norm_geometry.shape, norm_geometry_mean.shape, norm_geometry_std.shape)

    if True: # Renormalize
        cls_array = (cls_array * cls_npz["arr_2"]) + cls_npz["arr_1"]
    
    cl_mean = np.mean(cls_array)
    cl_std = np.std(cls_array)

    cls_array = (cls_array - cl_mean) / cl_std
    
else:
    norm_geometry_mean = 0
    norm_geometry_std = 1

    cl_mean = cls_npz["arr_1"]
    cl_std = cls_npz["arr_2"]
    
    
# Check if the normalization worked
print(np.mean(norm_geometry, axis=0), np.std(norm_geometry, axis=0))

# add singleton dimension to cls_array
cls_array = np.expand_dims(cls_array, axis=1)

# Save the normalized data
if True:
    np.savez("./03 CODE/4 Data/NandJ_coords_V2.npz",
            norm_geometry, norm_geometry_mean, norm_geometry_std)
    np.savez("./03 CODE/4 Data/NandJ_perfs_V2.npz", cls_array, cl_mean, cl_std)
    
    print("Data saved.")


# Plot some initial, normalized, renormalized, and rescaled samples in one plot
if False:
    for u in range(10):
        # Initial samples
        plt.scatter(geometry[u, :features_2], geometry[u, features_2:], s=1, c = "black")
        # Normalized samples
        plt.scatter(norm_geometry[u, :features_2], norm_geometry[u, features_2:], s=1, c = "blue")
        # Renormalized samples
        if normalize:
            plt.scatter(norm_geometry[u, :features_2] * norm_geometry_std[:features_2] + norm_geometry_mean[:features_2],
                    norm_geometry[u, features_2:] * norm_geometry_std[features_2:] + norm_geometry_mean[features_2:], s=1, c = "red")
    
    plt.show()
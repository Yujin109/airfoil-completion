import sys
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("./03 CODE/8 X-Foil/Python_DIR/")
from xfoil_runner import eval_CL_XFOIL

# --------------------------------------------------------------------------- #
# Plot all airfoil profiles in a tiled layout
# --------------------------------------------------------------------------- #
def visualize_airfoils_tiled(plt_label = "", show_plot = False, save_plot = True, eval_CL = False, XFoil_viscous = True):
    # Define the path to the folder containing the .txt files
    # folder_path = "./03 CODE/1 Yonekura/2nd/models/tmp/"
    folder_path = "./04 RUNS/4 Output/2 Temporary/"

    file_list = os.listdir(folder_path)
    file_list.sort()

    pattern = 'sample_cl_*.*_0_ep*.txt'
    file_list = fnmatch.filter(file_list, pattern)

    # sort file list by epoch number in ascending order "ep*.txt"
    file_list.sort(key=lambda x: int(x.split('ep')[-1].split('.txt')[0]))

    # Count the number of text files in the path
    num_files = len(file_list)

    num_cols = 25# num_files // 8

    # Calculate the number of rows for the subplots
    num_rows = 8 #(num_files + num_cols - 1) // num_cols

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8*num_cols, 6*num_rows))

    itr = 0
    file_itr = 0

    # Iterate through all files in the folder again
    for file_name in tqdm(file_list, desc = "Plotting airfoil profiles: "):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            file_name_wo = os.path.splitext(os.path.basename(file_path))[0]

            coordinates = []
            with open(file_path, "r") as file:
                for line in file:
                    x, y = line.strip().split()
                    coordinates.append([float(x), float(y)])

            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]

            # Evaluate CL using XFOIL
            if eval_CL:
                # Convert the coordinates to a numpy array and expand the dimensions to 1x2x248
                geom_data = np.array(coordinates)
                geom_data = np.expand_dims(geom_data.T, axis = 0)

                # Evaluate the lift coefficient using XFOIL
                CL, CD = eval_CL_XFOIL(geom_data, viscous = XFoil_viscous)

            # Calculate the row and column indices for the subplot
            row_idx = itr % num_rows
            col_idx = (itr // num_rows) % num_cols

            # Plot the coordinates on the corresponding subplot, colorcode the coordinate index to check the order
            axes[row_idx, col_idx].scatter(x_coords, y_coords, c = range(len(x_coords)), cmap = 'viridis')
            axes[row_idx, col_idx].set_xlabel('X')
            axes[row_idx, col_idx].set_ylabel('Y')
            axes[row_idx, col_idx].set_title('Coordinate Plot: ' + file_name_wo)
            axes[row_idx, col_idx].grid(True)
            axes[row_idx, col_idx].set_aspect('equal')

            # Add the lift coefficient to the title
            if eval_CL:
                axes[row_idx, col_idx].set_title('Coordinate Plot: ' + file_name_wo + ' - CL: ' + str(CL))

            itr += 1

            if itr%(num_cols*num_rows) == 0:
                # Adjust the spacing between subplots
                plt.tight_layout()

                # set plot background color to white
                #fig.patch.set_facecolor('white')

                # Save the figure
                if save_plot:
                    plt.savefig(folder_path + plt_label + 'Tiled_Samples_' + str(file_itr) + '.png', dpi = 150)

                # Close the figure
                plt.close(fig)

                file_itr += 1

                 # Create a grid of subplots
                fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8*num_cols, 6*num_rows))


    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure
    if save_plot:
        plt.savefig(folder_path + plt_label + 'Tiled_Samples_' + str(file_itr) + '.png', dpi = 150)

    # Close the figure
    plt.close(fig)

    print("Done!")

# --------------------------------------------------------------------------- #
# Main function
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Visualize airfoil profiles
    visualize_airfoils_tiled(plt_label = "DDPM_2024-06-18_11-16-58_UNet_EP10000_", show_plot = False, save_plot = True, eval_CL = False)
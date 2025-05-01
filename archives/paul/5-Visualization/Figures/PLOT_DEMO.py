import numpy as np
import matplotlib.pyplot as plt

# Print all available rcParams keys
# print(plt.rcParams.keys())

#Load style
plt.style.use("./03 CODE/5 Visualization/Figures/LaTex_Style_Plot.mplstyle")

# Custom Color Cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#000000', '#0000f1', '#009cff', '#00ff00', '#ffff00', '#ffa500', '#ff0000', '#800000'])

# Double figsize in x direction
tmp_figsize = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = [2*tmp_figsize[0], tmp_figsize[1]]

plt_name = "plot_test.pdf"
#plt_path = "./03 CODE/5 Visualization/Figures/"
plt_path = "./02 DOCUMENTATION/MA Thesis/images/"

plt_show = True
plt_save = False

################################################

fig, ax = plt.subplots()
    
################################################
var = 3

# Define Gaussian function
def gaussian(x, mu, var):
    sig = np.sqrt(var)
    return (1 / np.sqrt(2*np.pi*sig**2)) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

x = np.linspace(-5, 5, 1000)

################################################
plt.plot(x, gaussian(x, -1, var), color="C0", label="")
plt.plot(x, (0.4 - gaussian(x, 1, var)), color="C0", label="")
plt.plot(x, gaussian(x, -1, var) * (0.4 - gaussian(x, 1, var)), color="C1", label="")

#plt.xlabel("$t$")
#plt.ylabel("$mean$")

#plt.yscale("log")

plt.grid(True)
plt.grid(which="minor", linestyle="--")

plt.legend()

################################################
# print final values
print("Final values:")

################################################

plt.tight_layout(pad=0.5)

if plt_save:
    plt.savefig(plt_path + plt_name, dpi=400, transparent=True)
if plt_show:
    plt.show()
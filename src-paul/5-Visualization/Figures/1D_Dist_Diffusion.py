import numpy as np
import matplotlib.pyplot as plt

# Print all available rcParams keys
# print(plt.rcParams.keys())

#Load style
plt.style.use("./03 CODE/5 Visualization/Figures/LaTex_Style_Visual.mplstyle")

# Custom Color Cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#000000', '#0000f1', '#009cff', '#00ff00', '#ffff00', '#ffa500', '#ff0000', '#800000'])

plt_ticks = False

plt_name = "plot_test.pdf"
#plt_path = "./03 CODE/5 Visualization/Figures/"
plt_path = "./02 DOCUMENTATION/MA Thesis/images/"

plt_show = True
plt_save = False

################################################

fig, ax = plt.subplots()
# Move the left and bottom spines to x = 0 and y = 0, respectively.
ax.spines[["left", "bottom"]].set_position(("data", 0))
# Hide the top and right spines.
ax.spines[["top", "right"]].set_visible(False)

# Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
# case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
# respectively) and the other one (1) is an axes coordinate (i.e., at the very
# right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
# actually spills out of the axes.
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=4)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=4)

# Remove axis ticks
if not plt_ticks:
    plt.xticks([]), plt.yticks([])
    
################################################

# Define Gaussian function
def gaussian(x, mu, var):
    sig = np.sqrt(var)
    return (1 / np.sqrt(2*np.pi*sig**2)) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Define forward kernel
def forward_kernel(xt, xtt, beta):
    return gaussian(xt, np.sqrt(1-beta) * xtt, beta)

# Define forward step
def forward_step(qtt, x, beta):
    qt = np.zeros_like(qtt)
    
    for i in range(len(x)):
        qt[i] = np.sum(forward_kernel(x[i], x, beta) * qtt)
    qt = qt / np.sum(qt)
    
    return qt

# Generate data
x = np.linspace(-5, 5, 1000)

qx = 2 * gaussian(x, -3, 0.25) + 0.8 * gaussian(x, -0.5, 0.75) + 0.1 * gaussian(x, 1, 3) + gaussian(x, 2, 0.1)

# Normalize
qx = qx / np.sum(qx)

steps = 100
beta = 0.05

# Plot initial distribution
plt.plot(x, qx)

for u in range(steps):
    qt = forward_step(qx, x, beta) if u == 0 else forward_step(qt, x, beta)
    
    # Normalize
    qt = qt / np.sum(qt)
    
    #if u in [0, 9, 29, 99]:
    #if u in [9, 29, 99]:
    if u in [9, 19, 29, 39, 49, 59, 99]:
        plt.plot(x, qt)

plt.xlabel("$\mathbf{x}_t$")
plt.ylabel("$q(\mathbf{x}_t)$", rotation=0)

################################################

# Place the labels at the end of the axes
ax.xaxis.set_label_coords(1.02, 0.025)
ax.yaxis.set_label_coords(0.58, 0.95)

plt.tight_layout()

if plt_save:
    plt.savefig(plt_path + plt_name, dpi=400, transparent=True)
if plt_show:
    plt.show()



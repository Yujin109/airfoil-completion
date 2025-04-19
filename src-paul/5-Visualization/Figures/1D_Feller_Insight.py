import numpy as np
import matplotlib.pyplot as plt

# Print all available rcParams keys
# print(plt.rcParams.keys())

#Load style
plt.style.use("./03 CODE/5 Visualization/Figures/LaTex_Style_Visual.mplstyle")

# Custom Color Cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#000000', '#0000f1', '#009cff', '#00ff00', '#ffff00', '#ffa500', '#ff0000', '#800000'])

plt_ticks = False

plt_name = "plot2.pdf"
#plt_path = "./03 CODE/5 Visualization/Figures/"
plt_path = "./02 DOCUMENTATION/MA Thesis/images/"

plt_show = True
plt_save = False

plt_1 = True

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

# Generate data
beta1 = 0.1
beta2 = 0.3
beta3 = 0.75

xt = -1.5

x = np.linspace(-4, 3, 1000)

qtm = 2 * gaussian(x, -3, 0.25) + 0.8 * gaussian(x, -0.5, 0.75) + 0.1 * gaussian(x, 1, 3) + gaussian(x, 2, 0.1)

k1 = forward_kernel(x, xt, beta1)
k2 = forward_kernel(x, xt, beta2)
k3 = forward_kernel(x, xt, beta3)

# Normalize
qtm = qtm / np.sum(qtm)
k1 = k1 / np.sum(k1)
k2 = k2 / np.sum(k2)
k3 = k3 / np.sum(k3)

p1 = qtm * k1
p2 = qtm * k2
p3 = qtm * k3

# Normalize
p1 = p1 / np.sum(p1)
p2 = p2 / np.sum(p2)
p3 = p3 / np.sum(p3)

# Plot initial distribution
################################################
if plt_1:
    plt.plot(x, qtm, label="$q(x_{t-1})$")

    plt.plot(x, p1, color="C1", label="$q(x_{t-1}|x_t, \\beta^1)$")
    plt.plot(x, p2, color="C3", label="$q(x_{t-1}|x_t, \\beta^2)$")
    plt.plot(x, p3, color="C6", label="$q(x_{t-1}|x_t, \\beta^3)$")

    plt.xlabel("$x_{t-1}$")
    #plt.ylabel("$q(x_t)$", rotation=0)

    # line at xt
    plt.axvline(x=xt, color="black", linestyle=":", alpha=0.5)
    # text at xt
    plt.text(xt, -0.001, "$x_{t}$", ha="center")
    
    plt.legend(loc=(0.6, 0.4))

    
    # Place the labels at the end of the axes
    ax.xaxis.set_label_coords(1.02, 0.025)
    ax.yaxis.set_label_coords(0.58, 0.95)

# Plot kernel
################################################
else:
    plt.plot(x, k1, color="C1", label="$\\beta^1 = " + str(beta1) + "$")
    plt.plot(x, k2, color="C3", label="$\\beta^2 = " + str(beta2) + "$")
    plt.plot(x, k3, color ="C6", label="$\\beta^3 = " + str(beta3) + "$")
    
    # line at xt
    plt.axvline(x=xt, color="black", linestyle=":", alpha=0.5)
    # text at xt
    plt.text(xt, -0.001, "$x_{t-1}$", ha="center")

    plt.xlabel("$x_{t}$")
    plt.ylabel("$q(x_t|x_{t-1})$", rotation=0)
    
    # legend in the middle
    plt.legend(loc="right")
    
    # Place the labels at the end of the axes
    ax.xaxis.set_label_coords(1.02, 0.025)
    ax.yaxis.set_label_coords(0.7, 0.95)

################################################

plt.tight_layout()

if plt_save:
    plt.savefig(plt_path + plt_name, dpi=400, transparent=True)
if plt_show:
    plt.show()



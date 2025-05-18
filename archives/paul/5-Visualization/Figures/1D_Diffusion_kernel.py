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

# Generate data
x = np.linspace(-0.1, 2, 1000)
xtt = 1.25
beta = 0.2

Kern = forward_kernel(x, xtt, beta)

# Normalize
Kern = Kern / np.sum(Kern)

# Plot the kernel distribution
plt.plot(x, Kern)

# plot line at x = xtt
plt.plot([xtt, xtt], [-0.0001, 0.0001], color="black")
# put text at x = xtt
plt.text(xtt, 0.00035, "$\mathbf{x}_{t-1}$", verticalalignment="top", horizontalalignment="center")

# plot a dotted line indicating the mean
plt.plot([np.sqrt(1-beta) * xtt, np.sqrt(1-beta) * xtt], [0, np.max(Kern)], color="gray", linestyle=":")

# label the mean with text
plt.text(np.sqrt(1-beta) * xtt, 1.175 * np.max(Kern), "$\sqrt{1-\\beta_t}\mathbf{x}_{t-1}$", verticalalignment="top", horizontalalignment="center")

# plot dotted lines indicating the standard deviation
plt.plot([np.sqrt(1-beta) * xtt - np.sqrt(beta), np.sqrt(1-beta) * xtt - np.sqrt(beta)], [0, 0.8 * np.max(Kern)], color="gray", linestyle=":")
plt.plot([np.sqrt(1-beta) * xtt + np.sqrt(beta), np.sqrt(1-beta) * xtt + np.sqrt(beta)], [0, 0.8 * np.max(Kern)], color="gray", linestyle=":")

# draw an arrow between the 1st and 2nd dotted line
plt.annotate("", xy=(np.sqrt(1-beta) * xtt + np.sqrt(beta), 0.5*np.max(Kern)), xytext=(np.sqrt(1-beta) * xtt - np.sqrt(beta), 0.5*np.max(Kern)),
             arrowprops=dict(arrowstyle="<->", color="gray"))

# mark standard deviation
plt.text(np.sqrt(1-beta) * xtt - 0.18, 0.0013, "$2\sqrt{\\beta_t}$", verticalalignment="top", horizontalalignment="center")

plt.xlabel("$\mathbf{x}_t$")
plt.ylabel("$K(\mathbf{x}_t|\mathbf{x}_{t-1},\,\\beta_t)$", rotation=0)

# set y axis limit to 1.1 * max value
plt.ylim(-0.1 * np.max(Kern), 1.3 * np.max(Kern))

################################################

# Place the labels at the end of the axes
ax.xaxis.set_label_coords(1.02, 0.05)
ax.yaxis.set_label_coords(0.275, 0.95)

plt.tight_layout()

if plt_save:
    plt.savefig(plt_path + plt_name, dpi=400, transparent=True)
if plt_show:
    plt.show()



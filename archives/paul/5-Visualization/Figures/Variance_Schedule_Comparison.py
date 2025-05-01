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

steps = 1000

beta_1 = 0.01

beta_2 = 1E-4
beta_3 = 0.02

def beta_const(t):
    return beta_1

def beta_lin(t):
    steps = 1000
    return beta_2 + (beta_3 - beta_2) * t / steps

def alpha_bar(t, beta_func):
    alpha_bar = 1
    for u in range(1,t+1):
        alpha_bar *= 1 - beta_func(u)
    
    return alpha_bar

mean_t_const = np.zeros(steps)
var_t_const = np.zeros(steps)

mean_t_lin = np.zeros(steps)
var_t_lin = np.zeros(steps)

beta_const_t = np.zeros(steps)
beta_lin_t = np.zeros(steps)

for t in range(steps):
    beta_const_t[t] = beta_const(t)
    beta_lin_t[t] = beta_lin(t)
    
    alpha_bar_t_const = alpha_bar(t, beta_const)
    alpha_bar_t_lin = alpha_bar(t, beta_lin)
    
    mean_t_const[t] = np.sqrt(alpha_bar_t_const)
    var_t_const[t] = 1 - alpha_bar_t_const
    
    mean_t_lin[t] = np.sqrt(alpha_bar_t_lin)
    var_t_lin[t] = 1 - alpha_bar_t_lin
    
# Second Plot
################################################
plt.plot(range(1, steps+1), mean_t_const, color="C{}".format(0), label="${\\rm E}[x_t] / x_0$, $\\beta_t$ const.")
plt.plot(range(1, steps+1), mean_t_lin, color="C{}".format(1), label="${\\rm E}[x_t] / x_0$, $\\beta_t$ linear")

plt.plot(range(1, steps+1), var_t_const, color="C{}".format(0), linestyle="--", label="${\\rm Var}[x_t]$, $\\beta_t$ const.")
plt.plot(range(1, steps+1), var_t_lin, color="C{}".format(1), linestyle="--", label="${\\rm Var}[x_t]$, $\\beta_t$ linear")

plt.xlabel("$t$")
#plt.ylabel("$mean$")

#plt.yscale("log")

plt.grid(True)
plt.grid(which="minor", linestyle="--")

plt.legend()

################################################
# print final values
print("Final values:")
print("Const. beta:")
print("Mean: ", mean_t_const[-1])
print("Var: ", var_t_const[-1])

print("Lin. beta:")
print("Mean: ", mean_t_lin[-1])
print("Var: ", var_t_lin[-1])


################################################

plt.tight_layout(pad=0.5)

if plt_save:
    plt.savefig(plt_path + plt_name, dpi=400, transparent=True)
if plt_show:
    plt.show()
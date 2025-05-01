import numpy as np
import matplotlib.pyplot as plt

Nt = 1000
b1 = 1E-4
b2 = 0.02

# function to compute cosine schedule
def f(t, T, s=8E-3):
    arg = np.pi * 0.5 * (t/T + s) / (1 + s)
    return np.cos(arg)**2

# Linear variance schedule --> b(t) in time from b1 to b2
def get_beta(t, Nt, b1, b2):
    return t * (b2 - b1) / Nt + b1

# cosine schedule
def get_beta_cos(t, Nt):
    f0 = f(0, Nt)
    alpha_bar = f(t, Nt) / f0
    alpha_bar_m = f(t-1, Nt) / f0
    beta = 1 - (alpha_bar / alpha_bar_m)
    if beta > 0.999:
        beta = 0.999
    return beta

def get_alpha(t, Nt, b1, b2):
    return 1 - get_beta(t, Nt, b1, b2)

def get_alpha_bar(t, Nt, b1, b2):
    alpha_bar = get_alpha(1, Nt, b1, b2)
    
    for i in range(2, t + 1):
        alpha_bar *= get_alpha(i, Nt, b1, b2)
    
    return alpha_bar

def get_alpha_bar_cos(t, Nt):
    f0 = f(0, Nt)
    alpha_bar = f(t, Nt) / f0
    
    return alpha_bar

# Forward / Diffusion process
beta = [get_beta(t, Nt, b1, b2) for t in range(Nt)]
beta_cos = [get_beta_cos(t, Nt) for t in range(Nt)]

mu_scale = []
std_scale = []

mu_scale_cos = []
std_scale_cos = []

for i in range(Nt):
    alpha_bar = get_alpha_bar(i, Nt, b1, b2)
    mu_scale.append(np.sqrt(alpha_bar))
    std_scale.append(np.sqrt(1 - alpha_bar))

    alpha_bar_cos = get_alpha_bar_cos(i, Nt)
    mu_scale_cos.append(np.sqrt(alpha_bar_cos))
    std_scale_cos.append(np.sqrt(1 - alpha_bar_cos))

# Plot the variance schedule, mu_scale and std_scale
plt.plot(range(Nt), beta, label='Variance schedule')
plt.plot(range(Nt), mu_scale, label='Latent Mean / x_0')
plt.plot(range(Nt), std_scale, label='Latend Std. Dev.')

plt.plot(range(Nt), beta_cos, label='Variance schedule Cos')
plt.plot(range(Nt), mu_scale_cos, label='Latent Mean / x_0 Cos')
plt.plot(range(Nt), std_scale_cos, label='Latend Std. Dev. Cos')

plt.xlabel('Diffusion Step')
plt.yscale('log')
plt.minorticks_on()
plt.grid(which='both')
plt.legend()
plt.show()

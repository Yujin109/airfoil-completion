import numpy as np
from scipy.optimize import minimize
from scipy.misc import derivative
import matplotlib.pyplot as plt

# Define the function to be approximated
target_function = lambda x: np.arccos(x)

# Define the function approximator
approximator = lambda x, theta: theta[0] + theta[1]*x + theta[2]*x**7
#approximator = lambda x, theta: theta[0] + theta[1]*(x-theta[3]) + theta[2]*(x-theta[4])**3

f_evals = int(1E3)

global weight, x
x = np.linspace(-1, 1, f_evals)

weight = np.linspace(1, 1, f_evals)

def objective(theta):
    global weight, x

    approx = approximator(x, theta)
    target = target_function(x)

    # Return the sum of weighted squared differences
    return np.sum(weight * (approx - target)**2)

def constraint1(theta):
    x_exact = np.array([1, -1])
    approx = approximator(x_exact, theta)
    target = target_function(x_exact)

    return approx - target


theta0 = [1, 1, 1]
cons = {'type': 'eq', 'fun': constraint1}

solution = minimize(objective, theta0, constraints=cons)
#solution = minimize(objective, theta0)
print(solution)

# Evaluate the optimal approximator and target
deg_x = np.arccos(x) * 180 / np.pi
opt_approx = approximator(x, solution.x)
target = target_function(x)

# Evaluate the first derivatives
dx_opt_approx = derivative(approximator, x, args=(solution.x,), dx=1e-6)
dx_target = derivative(target_function, x, dx=1e-6)


# Plot the target function, the approximator in one subplot and the first derivatives in another
fig, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].plot(x, opt_approx, label="Optimal Approximation")
ax[0].plot(x, target, label="Target Function")
#ax[0].plot(deg_x, weight, label="Weight")
# plot a second x-axis with the different scale
ax[0].twinx().plot(x, np.abs(opt_approx - target), color='red', label="Abs. Error")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].legend()
ax[0].set_title("Optimal Function Approximation")
ax[0].grid()

#ax[0].invert_xaxis()

# set axis equal
ax[0].axis('equal')


ax[1].plot(x, dx_opt_approx, label="Optimal Approximation Derivative")
ax[1].plot(x, dx_target, label="Target Function Derivative")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y'")
ax[1].legend()
ax[1].set_title("First Derivatives")
ax[1].grid()
ax[1].invert_xaxis()

plt.show()

print("Optimal Parameters: ", solution.x)
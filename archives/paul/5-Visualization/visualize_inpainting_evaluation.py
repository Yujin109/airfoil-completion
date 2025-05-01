import numpy as np
import matplotlib.pyplot as plt

# load inpainting_log
inpainting_log = np.load("inpainting_log_resmpl10.npz")

cl_gt, cl_trgt, cl, cd, cnvxty = inpainting_log["arr_0"], inpainting_log["arr_1"], inpainting_log["arr_2"], inpainting_log["arr_3"], inpainting_log["arr_4"]

# remove last dimension from cl_gt
cl_gt = cl_gt[:, :, 0]

# find not converged airfoils from nan values in cl
converged = ~np.isnan(cl)

# evaluate mse cl vs cl_trgt
mse_cl_cl_trgt = np.mean((cl[converged] - cl_trgt[converged])**2)
# evaluate mean cnvxty
mean_cnvxty = np.mean(cnvxty[converged])

# evaluate convergence ratio from nan values in cl
convergence_ratio = 100*(np.mean(converged))

print(f"MSE CL vs CL Target: {mse_cl_cl_trgt}")
print(f"Mean Convexity: {mean_cnvxty}")
print(f"Convergence Ratio: {convergence_ratio}%")

# calculate linear regression for cl vs cl_trgt
m, b = np.polyfit(cl_trgt[converged], cl[converged], 1)

# evaluate linear regression
cl_space = np.linspace(np.min(cl_trgt), np.max(cl_trgt), 40)
cl_regression = m*cl_space + b

cl_min = np.min([np.min(cl[converged]), np.min(cl_trgt[converged]), np.min(cl_gt[converged])])
cl_max = np.max([np.max(cl[converged]), np.max(cl_trgt[converged]), np.max(cl_gt[converged])])

# plot cl_gt vs cl_trgt
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(cl_gt[converged], cl_trgt[converged], s=1)
ax.scatter(cl_gt[~converged], cl_trgt[~converged], s=1, color='red')
ax.set_xlabel('CL Ground Truth')
ax.set_ylabel('CL Target')
ax.set_title('CL Ground Truth vs CL Target')
ax.set_aspect('equal')
ax.set_xlim(cl_min, cl_max)
ax.set_ylim(cl_min, cl_max)
ax.grid()
plt.show()

# remove not converged airfoils
cl_gt = cl_gt[converged]
cl_trgt = cl_trgt[converged]
cl = cl[converged]
cd = cd[converged]
cnvxty = cnvxty[converged]

# plot cl vs cl_trgt
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(cl, cl_trgt, s=1)
ax.plot(cl_regression, cl_space, color='red')
ax.text(0.5, 0.3, f"y = {m:.2f}x + {b:.2f}", transform=ax.transAxes)
ax.set_xlabel('CL')
ax.set_ylabel('CL Target')
ax.set_title('CL vs CL Target')
# axis have same aspect ratio
ax.set_aspect('equal')
ax.set_xlim(cl_min, cl_max)
ax.set_ylim(cl_min, cl_max)
ax.grid()
plt.show()

cl_prjct = cl - cl_trgt
# plot histogram of cl - cl_trgt ? cl projection
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(cl_prjct, bins=15)
ax.set_xlabel('CL - CL Target')
ax.set_ylabel('Frequency')
ax.set_title('CL - CL Target Histogram')
ax.grid()
plt.show()

# plot kernel density estimation of cl - cl_trgt
from scipy import stats
kde = stats.gaussian_kde(cl_prjct)
x = np.linspace(np.min(cl_prjct), np.max(cl_prjct), 100)
y = kde(x)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y)
ax.set_xlabel('CL - CL Target')
ax.set_ylabel('Density')
ax.set_title('CL - CL Target KDE')
ax.grid()
plt.show()

#plot cl-cl_trgt vs. cl_trgt-cl_gt
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(cl-cl_trgt, cl_trgt-cl_gt, s=1)
ax.set_xlabel('CL - CL Target')
ax.set_ylabel('CL Target - CL Ground Truth')
ax.set_title('CL - CL Target vs CL Target - CL Ground Truth')
ax.set_aspect('equal')
ax.grid()
plt.show()

# plot cd vs cl - cl_trgt
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(cd, cl - cl_trgt, s=1)
ax.set_xlabel('CD')
ax.set_ylabel('CL - CL Target')
ax.set_title('CD vs CL - CL Target')
ax.grid()
plt.show()


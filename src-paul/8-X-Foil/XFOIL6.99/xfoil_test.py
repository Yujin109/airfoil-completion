
"""Runs an XFOIL analysis for a given airfoil and flow conditions"""
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

"""Todo
- Change paneling distribution at leading and trailing edge
- perform viscous analysis
- perform polar analysis
"""

# Inputs
airfoil_name = "cdiffusion_cl_0.7_g_0.0_w_15_0"
# Iterate over the angle of attack from alpha_i to alpha_f with a step of
alpha_i = -10
alpha_f = 25
alpha_step = 1.0
Re = 1000000    # Reynolds number
n_iter = 50    # Number of iterations for convergence

# Change the working directory to the XFOIL folder
os.chdir("D:/PAUL_BERTHOLD/master-thesis-generative-models/03 CODE/4 X-Foil/XFOIL6.99")

# XFOIL input file writer 
if os.path.exists("polar_file.txt"):
    os.remove("polar_file.txt")

input_file = open("input_file.in", 'w')
input_file.write("LOAD {0}.txt\n".format(airfoil_name))
input_file.write(airfoil_name + '\n')
input_file.write("PANE\n")
input_file.write("OPER\n")
input_file.write("Visc {0}\n".format(Re))
input_file.write("PACC\n")
input_file.write("polar_file.txt\n\n")
input_file.write("ITER {0}\n".format(n_iter))
input_file.write("ASeq {0} {1} {2}\n".format(alpha_i, alpha_f,
                                             alpha_step))
input_file.write("\n\n")
input_file.write("quit\n")
input_file.close()

subprocess.call("xfoil.exe < input_file.in", shell=True)

polar_data = np.loadtxt("polar_file.txt", skiprows=12)

AOA = polar_data[:, 0]
CL = polar_data[:, 1]
CD = polar_data[:, 2]

# Plot the results
plt.figure()
plt.plot(AOA, CL, label="CL")
plt.plot(AOA, CD, label="CD")
plt.xlabel("Angle of Attack (deg)")
# plt.ylabel("Coefficient")
plt.legend()
plt.grid()
plt.show()

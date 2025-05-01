
"""Runs an XFOIL analysis for a given airfoil and flow conditions"""
import os
import subprocess
import numpy as np

# TODO:
# Do multiple analysis within one run w.o. restarting XFOIL / cmd
# Paths hard coded
# Check if correctly ordered
# Parallelize for loop

# geom_data is a Tensor with shape [batch_size, 2, 248]
def eval_CL_XFOIL(geom_data, viscous=True, AOA=5.0, timeout=6, verbose=False, linux=False):

    n_evals = geom_data.shape[0]
    
    # Path to temporary airfoil file
    if linux:
        xfoil_input = "/home/MA/03 CODE/8 X-Foil/Python_DIR/tmp_airfoil.txt"
    else:
        xfoil_input = "D:/PAUL_BERTHOLD/master-thesis-generative-models/03 CODE/8 X-Foil/Python_DIR/tmp_airfoil.txt"
    
    # Path to X-foil analysis configuration
    if viscous:
        #xfoil_config = "Config_file_visc.in" # 248 Panels, 100 Iterations
        #xfoil_config = "Config_file_visc_2.in" # Default 160 Panels, 100 Iterations
        xfoil_config = "Config_file_visc_3.in" # Default 160 Panels, 600 Iterations -- 100 per AOA --> goes to AOA 6
    else:
        xfoil_config = "Config_file_invisc.in"
    
    # Path to X-foil executable
    if linux:
        wrk_dir = "/home/MA/03 CODE/8 X-Foil/Python_DIR"
    else:
        wrk_dir = r'D:\PAUL_BERTHOLD\master-thesis-generative-models\03 CODE\8 X-Foil\Python_DIR'
    
    # Path to X-Foil output file
    if linux:
        xfoil_output = "/home/MA/03 CODE/8 X-Foil/Python_DIR/tmp_CL.txt"
    else:
        xfoil_output = "D:/PAUL_BERTHOLD/master-thesis-generative-models/03 CODE/8 X-Foil/Python_DIR/tmp_CL.txt"

    # Initialize the lift coefficient array
    CL = np.zeros(n_evals)
    CD = np.zeros(n_evals)

    # Loop over each airfoil in the batch
    for u in range(n_evals):
        # Delete the output file if it exists
        if os.path.exists(xfoil_output):
            os.remove(xfoil_output)
            # Todo check whats faster
            #open(xfoil_output, 'w').close()

        # Write the coordinates to a .txt file
        np.savetxt(xfoil_input, geom_data[u,:,:].squeeze().T)

        if ~linux:
            # Path to the PowerShell script that acts as a watchdog
            ps_script_path = 'xfoil_watchdog.ps1'

            # Construct the PowerShell command properly
            command = f'powershell -ExecutionPolicy Bypass -File "{ps_script_path}" -inputFile "{xfoil_config}" -workingDir "{wrk_dir}" -timeout {timeout}'
        
        # kill subprocess if it takes too long, in case of timeout, continue with next airfoil
        try:  
            if ~linux:         
                proc = subprocess.run(command, cwd=wrk_dir, capture_output=True, text=True)
            else:
                proc = subprocess.run("xfoil.exe < " + xfoil_config, cwd=wrk_dir, timeout=timeout, shell=True)

            # Output the results
            if verbose:
                print(proc.stdout)
                print(proc.stderr)

        except:
            if ~linux:
                proc.kill()
            print(f"{u+1}/{n_evals} Process Error.")
            CL[u] = float("NAN")
            CD[u] = float("NAN")
            continue


        # Read the results from the Output file
        try:
            # AOA CL CD CDp CM Top_Xtr Bot_Xtr
            xfoil_data = np.loadtxt(xfoil_output, skiprows=12)

            # Extract the lift coefficient for AOA = 5
            if viscous:
                # Flag line with AOA
                flag = xfoil_data[:,0] == AOA
                if flag.any():
                    CL[u] = xfoil_data[flag, 1]
                    CD[u] = xfoil_data[flag, 3]
                    print(f"{u+1}/{n_evals} CL: {CL[u]} CD: {CD[u]}")
                else: #No solution found
                    print(f"{u+1}/{n_evals} Error AOA not found in XFOIL output.")
                    CL[u] = float("NAN")
                    CD[u] = float("NAN")

            else: # Inviscid
                CL[u] = xfoil_data[1]
                CD[u] = xfoil_data[3]
                print(f"{u+1}/{n_evals} CL: {CL[u]} CD: {CD[u]}")

        except:
            print(f"{u+1}/{n_evals} Error XFOIL output not found.")
            CL[u] = float("NAN")
            CD[u] = float("NAN")
    
    return CL, CD
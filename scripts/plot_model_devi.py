import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Root directory containing iter_* folders
    root_dir = "."

    # Dictionary to store data for each model file
    data_dict = {}

    # Loop over iter_* directories
    for iter_dir in sorted(glob.glob(os.path.join(root_dir, "iter_*"))):
        iter_num = int(iter_dir.split("_")[-1])  # Extract iteration number
        dpmd_dir = os.path.join(iter_dir, "02.dpmd")
        if not os.path.isdir(dpmd_dir):
            continue

        # Find all model_dev_*.out files
        model_files = sorted(glob.glob(os.path.join(dpmd_dir, "model_dev_*.out")))

        for model_file in model_files:
            model_name = os.path.basename(model_file)

            steps = []
            max_devi_f = []

            # Read and parse the file
            with open(model_file, "r") as f:
                lines = f.readlines()

            # Extract data from lines (skip header lines)
            for line in lines[2:]:  # Skip first two header lines
                cols = line.split()
                if len(cols) >= 5:  
                    try:
                        steps.append(int(cols[0])) 
                        max_devi_f.append(float(cols[4]))  # read: max_devi_f
                    except ValueError:
                        continue

            # Store in dictionary
            if model_name not in data_dict:
                data_dict[model_name] = []
            data_dict[model_name].append((iter_num, steps, max_devi_f))

    # Plotting
    plt.figure(figsize=(8, 6))
    for model, data in data_dict.items():
        for iter_num, steps, max_devi_f in sorted(data):
            plt.plot(steps, max_devi_f, linestyle="-", lw=2, label=f"{model} (iter: {iter_num})", marker='o', ms=5, alpha=1.0)

    plt.xlim(0, None)
    plt.xlabel("Points", fontsize=18)
    plt.ylabel(r"Max. Force Deviation ($\rm{eV/\AA}$)", fontsize=18)
    plt.title("Force deviation from each iteration/s", fontsize=16, color='blue')

    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=12)

    plt.legend()
    plt.grid(ls='-.')
    plt.savefig("model_devi.png", dpi=300)
    # plt.show()

if __name__ == "__main__":
    main()

#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#  
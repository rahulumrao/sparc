import os
import glob
import numpy as np

import matplotlib.pyplot as plt
from ase.io import read
########################################################################################################
# Parity plots for energy and forces
########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import dpdata
import os
########################################################################################################
# Parity plots for energy and forces
########################################################################################################
def compute_rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))

def compute_mae(true, pred):
    return np.mean(np.abs(true - pred))

def ParityPlot(data_dir, model_path, per_atom=False, type="all", save_fig=None):
    """
    Generate parity plots for energy and/or force components with RMSE + MAE annotations.

    Parameters:
    -----------
    data_dir : str
        Path to test dataset in DeepMD .npy format.
    model_path : str
        Path to frozen model.
    per_atom : bool
        Whether to plot energy per atom instead of total.
    plot_type : str
        'all' (default), 'energy', or 'forces'
    save_fig : str or None
        Path to save the output figure.
        
    Example:
        >>> ParityPlot("data_dir", "frozen_model.pb", per_atom=True, type="energy", save_fig='lcurve.png')
        >>> ParityPlot("data_dir", "frozen_model.pb", per_atom=True, type="forces")
    """

    if not os.path.exists(data_dir):
        print(f"[ANALYSIS][ERROR] Test data not found: {data_dir}")
        return
    if not os.path.isfile(model_path):
        print(f"[ANALYSIS][ERROR] Model file not found: {model_path}")
        return

    system = dpdata.LabeledSystem(data_dir, fmt="deepmd/npy")
    prediction = system.predict(dp=model_path)

    # Extract data
    e_true = np.array(system["energies"])
    e_pred = np.array(prediction["energies"])
    natoms = system.get_natoms()

    if per_atom:
        e_true /= natoms
        e_pred /= natoms
        e_unit = "eV/Atom"
    else:
        e_unit = "eV"

    f_true = np.vstack(system["forces"])
    f_pred = np.vstack(prediction["forces"])

    # Setup plot layout
    if type == "energy":
        fig, ax_energy = plt.subplots(1, 1, figsize=(6, 5), dpi=250)
    elif type == "forces":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    else:  # 'all'
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=250)
        ax_energy = axes[0, 0]

    # === Energy Parity Plot ===
    if type in ("all", "energy"):
        ax_energy.scatter(e_true, e_pred, c='blue', alpha=0.7, s=40, edgecolors='k')
        ax_energy.plot([e_true.min(), e_true.max()], [e_true.min(), e_true.max()], 'r--', lw=1.2)
        rmse_e = compute_rmse(e_true, e_pred)
        mae_e = compute_mae(e_true, e_pred)
        ax_energy.text(0.05, 0.90,
                       f"RMSE = {rmse_e:.4f} {e_unit}\nMAE = {mae_e:.4f} {e_unit}",
                       transform=ax_energy.transAxes,
                       fontsize=12, verticalalignment='top', color='blue')
        ax_energy.set_xlabel(f"Observed (DFT) [{e_unit}]", fontsize=14)
        ax_energy.set_ylabel(f"Predicted (MLP) [{e_unit}]", fontsize=14)
        ax_energy.set_title("(A) Energy", fontsize=16)
        ax_energy.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_energy.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_energy.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_energy.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_energy.tick_params(labelsize=12)
        ax_energy.grid(ls='--', alpha=0.7)

    # === Force Parity Plots ===
    if type in ("all", "forces"):
        components = ['fx', 'fy', 'fz']
        for i, comp in enumerate(components):
            if type == "forces":
                ax = axes[i]
            else:
                row, col = divmod(i + 1, 2)
                ax = axes[row, col]

            f_t = f_true[:, i]
            f_p = f_pred[:, i]
            rmse_f = compute_rmse(f_t, f_p)
            mae_f = compute_mae(f_t, f_p)

            ax.scatter(f_t, f_p, c='blue', alpha=0.6, s=30, edgecolors='k')
            ax.plot([f_t.min(), f_t.max()], [f_t.min(), f_t.max()], 'r--', lw=1.2)
            ax.text(0.05, 0.90,
                    f"RMSE = {rmse_f:.4f} eV/Å\nMAE = {mae_f:.4f} eV/Å",
                    transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', color='blue')
            ax.set_xlabel(r"Observed (DFT) [eV/$\rm{\AA}$]", fontsize=14)
            ax.set_ylabel(r"Predicted (MLP) [eV/$\rm{\AA}$]", fontsize=14)
            ax.set_title(f"({chr(66+i)}) {comp}", fontsize=16)
            ax.tick_params(labelsize=12)
            ax.grid(ls='--', alpha=0.7)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
        print(f"[ANALYSIS][INFO] Saved parity plot to: {save_fig}")
    else:
        plt.show()
########################################################################################################
# Plot Learning Curve
########################################################################################################        
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PlotLcurve(lcurve_file, save_fig=None):
    """
    Plots the learning curve (L-curve) from DeepMD training log.

    Parameters:
    -----------
    lcurve_file : str
        Path to lcurve.out file.
    save_fig : str or None
        If provided, path to save the figure (e.g., "lcurve.png").
        
    Example:
        >>> PlotLcurve(lcurve_file="iter_000000/01.train/training_1/lcurve.out")
    """
    if not os.path.isfile(lcurve_file):
        print(f"[ANALYSIS][ERROR] File not found: {lcurve_file}")
        return

    with open(lcurve_file) as f:
        headers = f.readline().split()[1:]

    data = pd.DataFrame(np.loadtxt(lcurve_file), columns=headers)

    legends = {
        "rmse_e_val": "RMSE Energy (val)",
        "rmse_e_trn": "RMSE Energy (train)",
        "rmse_f_val": "RMSE Force (val)",
        "rmse_f_trn": "RMSE Force (train)"
    }

    plt.figure(figsize=(8, 6), dpi=250)
    for key, label in legends.items():
        if key in data.columns:
            plt.loglog(data["step"], data[key], label=label, lw=2.2, alpha=0.9)

    plt.xlabel("Training steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("DeepMD Learning Curve", fontsize=16, pad=10)
    plt.legend(fontsize=12)
    plt.grid(which='both', ls='--', alpha=0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
        print(f"[ANALYSIS][INFO] Saved learning curve plot to: {save_fig}")
    else:
        plt.show()
        
########################################################################################################
# Plot error in Forces from various DeepMD models
########################################################################################################
def PlotForceDeviation(root_dir=".", iteration_window="all", target_iteration=None, dmin=0.05, dmax=0.5):
    """
    Parses model_dev_*.out files from iter_* directories (or a specific range/iteration/all),
    extracts max force deviation, and plots the results.
    
    Parameters:
        root_dir (str): Root directory containing iter_* folders.
        iteration_window (tuple or str): A tuple (start, end) to specify a range of iterations, or "all" to process all iterations.
        target_iteration (int): A specific iteration number to analyze.
        dmin (float): Lower threshold for candidate force deviation (default: 0.05).
        dmax (float): Upper threshold for candidate force deviation (default: 0.5).
        
    Example:
        >>> PlotForceDeviation("/path/to/root", iteration_window=(2, 5))
        >>> PlotForceDeviation("/path/to/root", target_iteration=3)
        >>> PlotForceDeviation("/path/to/root", iteration_window="all")
    """
    data_dict = {}
    
    # Collect relevant iteration directories
    iter_dirs = sorted(glob.glob(os.path.join(root_dir, "iter_*")))
    selected_dirs = []
    
    if target_iteration is not None:
        selected_dirs = [d for d in iter_dirs if int(d.split("_")[-1]) == target_iteration]
    elif iteration_window == "all":
        selected_dirs = iter_dirs
    elif isinstance(iteration_window, tuple):
        selected_dirs = [d for d in iter_dirs if iteration_window[0] <= int(d.split("_")[-1]) <= iteration_window[1]]
    
    # Loop over selected iter_* directories
    for iter_dir in selected_dirs:
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
                        max_devi_f.append(float(cols[4]))  # Read: max_devi_f
                    except ValueError:
                        continue

            # Store in dictionary
            if model_name not in data_dict:
                data_dict[model_name] = []
            data_dict[model_name].append((iter_num, steps, max_devi_f))

    # Plotting
    plt.figure(figsize=(12, 7), dpi=300)
    num_iterations = sum(len(data) for data in data_dict.values())
    ncol = min(num_iterations, 5)  # Adjust number of legend columns dynamically
    
    for model, data in data_dict.items():
        for iter_num, steps, max_devi_f in sorted(data):
            plt.plot(steps, max_devi_f, linestyle="-", lw=2.2, label=f"Iter: {iter_num}", marker='o', ms=5, alpha=0.7)

    plt.axhline(y=dmin, color='black', lw=2, ls='--', alpha=0.4)
    plt.axhline(y=dmax, color='black', lw=2, ls='--', alpha=0.4)
    plt.fill_between([min(steps), max(steps)], dmin, dmax, color='grey', alpha=0.3)
    plt.xlim(0, None)   
    plt.xlabel("Candidates", fontsize=22)
    plt.ylabel(r"Max. Force Deviation ($\rm{eV/\AA}$)", fontsize=22)
    # plt.title("Force deviation from each iteration/s", fontsize=16, color='blue')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Move the legend to the top and remove model_devi
    plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=ncol)
    plt.grid(ls='-.')
    plt.show()

########################################################################################################
# Plot Potential energy from Labelles candidates
########################################################################################################

def PlotPotentialEnergy(root_dir=".", iteration_window="all", target_iteration=None, traj_filename="AseMD.traj"):
    """
    Parses trajectory files from iter_* directories (or a specific range/iteration/all),
    extracts potential energy, and plots the results.
    
    Parameters:
        root_dir (str): Root directory containing iter_* folders.
        iteration_window (tuple or str): A tuple (start, end) to specify a range of iterations, or "all" to process all iterations.
        target_iteration (int): A specific iteration number to analyze.
        traj_filename (str): Name of the trajectory file to read in each iteration folder (default: "dpmd.traj").
        
    Example:
        >>> plot_potential_energy("/path/to/root", iteration_window=(2, 5), traj_filename="AseMD.traj")
        >>> plot_potential_energy("/path/to/root", target_iteration=3, traj_filename="AseMD.traj")
        >>> plot_potential_energy("/path/to/root", iteration_window="all", traj_filename="AseMD.traj")
    """
    if traj_filename is None:
        traj_filename = input("Enter the trajectory filename (e.g., AseMD.traj): ")
    
    energy_dict = {}
    
    # Collect relevant iteration directories
    iter_dirs = sorted(glob.glob(os.path.join(root_dir, "iter_*")))
    selected_dirs = []
    
    if target_iteration is not None:
        selected_dirs = [d for d in iter_dirs if int(d.split("_")[-1]) == target_iteration]
    elif iteration_window == "all":
        selected_dirs = iter_dirs
    elif isinstance(iteration_window, tuple):
        selected_dirs = [d for d in iter_dirs if iteration_window[0] <= int(d.split("_")[-1]) <= iteration_window[1]]
    
    # Loop over selected iter_* directories
    for iter_dir in selected_dirs:
        iter_num = int(iter_dir.split("_")[-1])  # Extract iteration number
        dpmd_dir = os.path.join(iter_dir, "00.dft")
        if not os.path.isdir(dpmd_dir):
            continue

        # Find the specified trajectory file
        traj_path = os.path.join(dpmd_dir, traj_filename)
        if not os.path.isfile(traj_path):
            print(f"Warning: {traj_filename} not found in {dpmd_dir}")
            continue

        # Read trajectory file
        traj = read(traj_path, index=":")
        
        # Extract potential energies
        energies = [item.get_potential_energy() for item in traj]
        
        # Store in dictionary
        if iter_num not in energy_dict:
            energy_dict[iter_num] = []
        energy_dict[iter_num].extend(energies)
    
    # Plotting
    plt.figure(figsize=(12, 7))
    num_iterations = len(energy_dict)
    ncol = min(num_iterations, 6)  # Auto-adjust legend columns
    
    for iter_num, energies in sorted(energy_dict.items()):
        plt.plot(range(len(energies)), energies, marker='o', linestyle='-', lw=2, label=f"Iter {iter_num}", ms=5, alpha=0.9)
    
    # Labels and formatting
    plt.xlabel("Labelled Candidates", fontsize=20)
    plt.ylabel("Potential Energy (eV)", fontsize=20)
    # plt.title("Potential Energy vs Labelles Candidates Across Iterations", fontsize=18, color='b')
    
    plt.xlim(0, None)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    
    # Legend
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=ncol)
    plt.grid(ls='-.')
    plt.show()

########################################################################################################
# Plot Distribution of Properties
########################################################################################################
import os
import glob
import matplotlib.pyplot as plt
from ase.io import read
import seaborn as sns

def PlotDistribution(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    traj_filename="AseMD.traj",
    type="line",         # Options: "line", "kde", "hist"
    get="energy",        # "energy" or "distance:i,j"
    **kwargs
):
    """
    Plot potential energy or bond distance from ASE trajectories across iter_* folders.

    This function reads trajectory files (e.g., AseMD.traj) inside `00.dft` folders located within iter_* 
    directories, extracts either potential energy or bond distance over time, and visualizes them using 
    line plots, KDE plots, or histograms.

    Parameters:
        root_dir (str): Root directory containing iter_* folders. Default is current directory.
        iteration_window (tuple or str): (start, end) tuple to specify a range of iterations, or "all".
        target_iteration (int): A specific iteration to analyze. Overrides iteration_window if given.
        traj_filename (str): Name of the ASE-readable trajectory file (default: "AseMD.traj").
        type (str): Type of plot to generate:
                    - 'line': property vs. frame index
                    - 'kde' : kernel density estimate
                    - 'hist': histogram of property values
        get (str): What property to extract:
                   - "energy" : uses get_potential_energy()
                   - "distance:i,j" : uses get_distance(i, j) between atoms i and j

        **kwargs: Additional keyword arguments forwarded to the underlying matplotlib/seaborn plotting calls.
                  Common kwargs include:
                      color, linestyle, linewidth, marker, alpha, bins, fill, edgecolor, etc.

    Example usage:
        # Default plot: potential energy across all iterations as a line plot
        PlotDistribution()

        # Explicit energy line plot
        PlotDistribution(get="energy", type="line")

        # Bond distance between atoms 0 and 7 as KDE
        PlotDistribution(get="distance:0,7", type="kde")

        # Histogram of energies from iteration 3
        PlotDistribution(target_iteration=3, get="energy", type="hist", bins=40, alpha=0.5)

        # Customized markers for energy plot
        PlotDistribution(get="energy", type="line", color="red", marker="x", linestyle="--")
    """

    property_dict = {}
    is_energy = get.lower() == "energy"
    is_distance = get.lower().startswith("distance:")
    symbol_pair = None

    if is_energy:
        prop_label = "Potential Energy (eV)"
    elif is_distance:
        try:
            i, j = map(int, get.split(":")[1].split(","))
        except:
            raise ValueError("For distance, use format: 'distance:i,j'")
    else:
        raise ValueError("get must be 'energy' or 'distance:i,j'")

    iter_dirs = sorted(glob.glob(os.path.join(root_dir, "iter_*")))
    if target_iteration is not None:
        selected_dirs = [d for d in iter_dirs if int(d.split("_")[-1]) == target_iteration]
    elif iteration_window == "all":
        selected_dirs = iter_dirs
    elif isinstance(iteration_window, tuple):
        selected_dirs = [d for d in iter_dirs if iteration_window[0] <= int(d.split("_")[-1]) <= iteration_window[1]]
    else:
        raise ValueError("iteration_window must be 'all' or (start, end)")

    for iter_dir in selected_dirs:
        iter_num = int(iter_dir.split("_")[-1])
        dpmd_dir = os.path.join(iter_dir, "00.dft")
        traj_path = os.path.join(dpmd_dir, traj_filename)

        if not os.path.isfile(traj_path):
            print(f"Warning: {traj_filename} not found in {dpmd_dir}")
            continue

        traj = read(traj_path, index=":")

        if is_distance and symbol_pair is None and len(traj) > max(i, j):
            try:
                symbols = traj[0].get_chemical_symbols()
                symbol_pair = (symbols[i], symbols[j])
            except Exception as ex:
                print(f"  Could not extract atom symbols: {ex}")
                symbol_pair = None

        if is_energy:
            values = [atoms.get_potential_energy() for atoms in traj]
        elif is_distance:
            values = [atoms.get_distance(i, j) for atoms in traj]

        property_dict[iter_num] = values

    if is_energy:
        if type=='line':
            xlabel = "Candidates"
            ylabel = "Potential Energy (eV)"
        elif type=='kde':
            xlabel = "Potential Energy (eV)"
            ylabel = "Density"
        elif type=='hist':
            xlabel = "Potential Energy (eV)"
            ylabel = "P(Energy)"
            
    elif is_distance:
        if symbol_pair:
            sym_i, sym_j = symbol_pair
            xlabel = rf"Distance [$\mathrm{{{sym_i}}}_{{{i}}}$-$\mathrm{{{sym_j}}}_{{{j}}}$] ($\rm{{\AA}}$)"
        else:
            xlabel = rf"Bond Distance ($\rm{{\AA}}$) between atoms {i} and {j}"
        ylabel = "Bond Distance Distribution" if type == "hist" else "Density"

    plt.figure(figsize=(12, 7), dpi=250)
    ncol = min(len(property_dict), 6)

    for iter_num, values in sorted(property_dict.items()):
        label = f"Iter {iter_num}"
        if type == "line":
            plt.plot(
                range(len(values)), values,
                label=label,
                marker=kwargs.get("marker", 'o'),
                linestyle=kwargs.get("linestyle", '-'),
                linewidth=kwargs.get("linewidth", 2),
                markersize=kwargs.get("ms", 5),
                alpha=kwargs.get("alpha", 0.9),
                color=kwargs.get("color", None)
            )
        elif type == "kde":
            sns.kdeplot(
                values, label=label,
                linewidth=kwargs.get("linewidth", 2),
                fill=kwargs.get("fill", True),
                alpha=kwargs.get("alpha", 0.4),
                linestyle=kwargs.get("linestyle", '-'),
                color=kwargs.get("color", None)
            )
        elif type == "hist":
            plt.hist(
                values, bins=kwargs.get("bins", 25),
                edgecolor=kwargs.get("edgecolor", "black"),
                alpha=kwargs.get("alpha", 0.7),
                label=label,
                color=kwargs.get("color", None)
            )
        else:
            raise ValueError("type must be 'line', 'kde', or 'hist'.")

    plt.xlabel(xlabel, fontsize=23)
    plt.ylabel(ylabel if type != "line" else prop_label, fontsize=23)

    if type == "line":
        plt.xlim(0, None)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=ncol)
    plt.grid(ls='-.')
    plt.tight_layout()
    plt.show()
########################################################################################################
# Plot Potential Energy Surface
########################################################################################################
def PlotPES(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    traj_filename="AseMD.traj",
    distance_pair=(0, 7),
    type="kde",  # "kde", "heatmap", or "hexbin"
    bins=(50, 50),
    **kwargs
):
    """
    Plot energy vs. bond distance across trajectories as a 2D KDE, heatmap, or hexbin.

    Parameters:
        root_dir (str): Root directory containing iter_* folders.
        iteration_window (tuple or str): (start, end) or "all".
        target_iteration (int): A specific iteration to analyze.
        traj_filename (str): ASE-readable trajectory file inside 00.dft folders.
        distance_pair (tuple): Atom indices (i, j) for bond distance calculation.
        type (str): "kde", "heatmap", or "hexbin".
        bins (tuple): Number of bins for heatmap/hexbin (x, y or just x).
        **kwargs: Extra kwargs forwarded to seaborn/matplotlib functions.

    Example usage:
        # Default KDE plot of energy vs. bond distance between atoms 0 and 7
        PlotPES()

        # Hexbin plot with custom bin count
        PlotPES(type="hexbin", bins=(60,))

        # Heatmap from a specific iteration
        PlotPES(target_iteration=3, type="heatmap", bins=(40, 40))

        # KDE plot with a custom colormap
        PlotPES(type="kde", cmap="mako")
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.io import read
    import glob
    import os

    i, j = distance_pair
    x_vals, y_vals = [], []
    total_frames = 0
    symbol_pair = None  # To store (symbol_i, symbol_j)

    # Get folders
    iter_dirs = sorted(glob.glob(os.path.join(root_dir, "iter_*")))
    if target_iteration is not None:
        selected_dirs = [d for d in iter_dirs if int(d.split("_")[-1]) == target_iteration]
    elif iteration_window == "all":
        selected_dirs = iter_dirs
    elif isinstance(iteration_window, tuple):
        selected_dirs = [d for d in iter_dirs if iteration_window[0] <= int(d.split("_")[-1]) <= iteration_window[1]]
    else:
        raise ValueError("iteration_window must be 'all' or (start, end)")

    print("Parsing iterations:")
    for iter_dir in selected_dirs:
        iter_num = int(iter_dir.split("_")[-1])
        traj_path = os.path.join(iter_dir, "00.dft", traj_filename)
        if not os.path.isfile(traj_path):
            print(f"  Iter {iter_num:>2}: MISSING ({traj_filename})")
            continue

        traj = read(traj_path, index=":")
        num_frames = len(traj)
        print(f"  Iter {iter_num:>2}: {num_frames} frames")
        total_frames += num_frames

        # Get atomic symbols from the first valid trajectory only
        if symbol_pair is None and num_frames > max(i, j):
            try:
                symbols = traj[0].get_chemical_symbols()
                symbol_pair = (symbols[i], symbols[j])
            except Exception as ex:
                print(f"  Could not extract atom symbols: {ex}")
                symbol_pair = None

        for atoms in traj:
            try:
                d = atoms.get_distance(i, j)
                e = atoms.get_potential_energy()
                x_vals.append(d)
                y_vals.append(e)
            except Exception as ex:
                print(f"    Error in {traj_path}: {ex}")
                continue

    print(f"\nTotal frames: {total_frames}")

    x = np.array(x_vals)
    y = np.array(y_vals)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 7), dpi=250)

    if type == "kde":
        sns.kdeplot(
            x=x, y=y, ax=ax,
            fill=True,
            cmap=kwargs.pop("cmap", "viridis"),
            levels=100,
            thresh=0.05,
            **kwargs
        )
    elif type == "heatmap":
        h = ax.hist2d(
            x, y,
            bins=bins,
            cmap=kwargs.pop("cmap", "plasma"),
            cmin=1,
            **kwargs
        )
        cb = fig.colorbar(h[3], ax=ax, pad=0.01)
        cb.set_label("Counts", fontsize=16)
        cb.ax.tick_params(labelsize=16)
    elif type == "hexbin":
        cmap = kwargs.pop("cmap", "inferno")
        gridsize = bins[0] if isinstance(bins, (tuple, list)) else bins
        hb = ax.hexbin(
            x, y,
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            linewidths=0.3,
            **kwargs
        )
        cb = fig.colorbar(hb, ax=ax, pad=0.01)
        cb.set_label("Counts", fontsize=20)
        cb.ax.tick_params(labelsize=16)
    else:
        raise ValueError("plot_type must be 'kde', 'heatmap', or 'hexbin'.")

    # Axis labels
    if symbol_pair:
        sym_i, sym_j = symbol_pair
        bond_label = rf"Distance [$\mathrm{{{sym_i}}}_{{{i}}}$-$\mathrm{{{sym_j}}}_{{{j}}}$] ($\rm{{\AA}}$)"
    else:
        bond_label = rf"Bond Distance ($\rm{{\AA}}$) between atoms {i} and {j}"

    ax.set_xlabel(bond_label, fontsize=20)
    ax.set_ylabel("Potential Energy (eV)", fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(ls='--', alpha=0.3)

    fig.subplots_adjust(right=0.98, left=0.12, top=0.95, bottom=0.12)
    plt.show()
########################################################################################################
# Plot Temperature from Deep Potential Dynamics
########################################################################################################

def PlotTemp(root_dir=".", iteration_window="all", target_iteration=None, traj_filename="dpmd.traj"):
    """
    Parses trajectory files from iter_* directories (or a specific range/iteration/all),
    extracts temperature, and plots the results.
    
    Parameters:
        root_dir (str): Root directory containing iter_* folders.
        iteration_window (tuple or str): A tuple (start, end) to specify a range of iterations, or "all" to process all iterations.
        target_iteration (int): A specific iteration number to analyze.
        traj_filename (str): Name of the trajectory file to read in each iteration folder.
        
    Example:
        >>> plot_temperature("/path/to/root", iteration_window=(2, 5), traj_filename="dpmd.traj")
        >>> plot_temperature("/path/to/root", target_iteration=3, traj_filename="dpmd.traj")
        >>> plot_temperature("/path/to/root", iteration_window="all", traj_filename="dpmd.traj")
    """
    if traj_filename is None:
        traj_filename = input("Enter the trajectory filename (default: dpmd.traj): ") or "dpmd.traj"
    
    temp_dict = {}
    
    # Collect relevant iteration directories
    iter_dirs = sorted(glob.glob(os.path.join(root_dir, "iter_*")))
    selected_dirs = []
    
    if target_iteration is not None:
        selected_dirs = [d for d in iter_dirs if int(d.split("_")[-1]) == target_iteration]
    elif iteration_window == "all":
        selected_dirs = iter_dirs
    elif isinstance(iteration_window, tuple):
        selected_dirs = [d for d in iter_dirs if iteration_window[0] <= int(d.split("_")[-1]) <= iteration_window[1]]
    
    # Loop over selected iter_* directories
    for iter_dir in selected_dirs:
        iter_num = int(iter_dir.split("_")[-1])  # Extract iteration number
        dpmd_dir = os.path.join(iter_dir, "02.dpmd")
        if not os.path.isdir(dpmd_dir):
            continue

        # Find the specified trajectory file
        traj_path = os.path.join(dpmd_dir, traj_filename)
        if not os.path.isfile(traj_path):
            print(f"Warning: {traj_filename} not found in {dpmd_dir}")
            continue

        # Read trajectory file
        traj = read(traj_path, index=":")
        
        # Extract temperatures
        temperatures = [item.get_temperature() for item in traj]
        
        # Store in dictionary
        if iter_num not in temp_dict:
            temp_dict[iter_num] = []
        temp_dict[iter_num].extend(temperatures)
    
    # Plotting
    plt.figure(figsize=(12, 7))
    num_iterations = len(temp_dict)
    ncol = min(num_iterations, 6)  # Auto-adjust legend columns
    
    for iter_num, temperatures in sorted(temp_dict.items()):
        plt.plot(range(len(temperatures)), temperatures, marker='o', linestyle='-', lw=2.2, label=f"Iter {iter_num}", ms=5, alpha=0.8)
        mean_temp = np.mean(temperatures)
        print(f"Mean Temperature (K) from Iteration {iter_num} := {mean_temp:.2f}")
    
    # Labels and formatting
    plt.xlabel("MD Steps", fontsize=20)
    plt.ylabel("Temperature (K)", fontsize=20)
    # plt.title("Temperature vs Frame Index Across Iterations", fontsize=18)
    
    plt.xlim(0, None)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    
    # Legend
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=ncol)
    plt.grid(ls='-.')
    plt.show()
########################################################################################################
# Read COLVAR file fro PLUMED
########################################################################################################
import pandas as pd

def ReadColvar(file_path="COLVAR"):
    """
    Reads a COLVAR file, automatically detecting column names from the first line if present.

    Parameters:
        file_path (str): Path to the COLVAR file.

    Returns:
        pd.DataFrame: DataFrame containing the COLVAR data.
        
    Example:
        >>> df = read_colvar("COLVAR")
        >>> print(df.head())
    """
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        column_names = first_line.split()[2:] if first_line.startswith("#! FIELDS") else None

    return pd.read_csv(file_path, sep='\s+', comment="#", names=column_names)

########################################################################################################
# Compute Histogram and Create Surface from a ASE Trajectory
########################################################################################################
import numpy as np
import pandas as pd
import ase.io
from scipy.stats import gaussian_kde

def get_2dSurface(traj, bonds, T=300):
    """
        Compute the free energy surface for the given bond distances from an ASE trajectory.

        Parameters:
            traj  : ASE trajectory object (list of Atoms objects)
                    The trajectory object containing the atomic configurations.
            bonds : List of tuples specifying the atom indices for two bonds [(a1, b1), (a2, b2)]
                    Each tuple represents a bond, where each element is an atom index.
            T     : Temperature in Kelvin (default = 300 K)

        Returns:
            R1 : 2D array of bond distances for bond 1, corresponding to the first bond in the input.
            R2 : 2D array of bond distances for bond 2, corresponding to the second bond in the input.
            F  : 2D array of free energies for the given bond distances, calculated from the probability distribution.

        Notes:
            The free energy surface is computed using Kernel Density Estimation (KDE) to approximate the joint
            probability distribution of the two bond distances. The result is normalized and converted to free energy
            values using the Boltzmann constant. The surface is then adjusted by subtracting its minimum value, 
            ensuring that the lowest energy is set to zero. 
            
            This free energy estimate is intended for visualization purposes only and should not be used for publication.
        Example:
            >>> R1, R2, F = get_2dSurface(traj, [(0,1), (1,2)])
    """
    # Extract bond distances for each frame
    b_1 = np.array([atoms.get_distance(*bonds[0]) for atoms in traj])
    b_2 = np.array([atoms.get_distance(*bonds[1]) for atoms in traj])

    # Stack bond lengths into a 2D array for joint distribution
    bond_data = np.vstack([b_1, b_2])

    # Use Kernel Density Estimation (KDE) to approximate P(r1, r2)
    kde = gaussian_kde(bond_data)

    # Define a grid of bond distances
    r1_range = np.linspace(min(b_1), max(b_1), 150)
    r2_range = np.linspace(min(b_2), max(b_2), 150)
    R1, R2 = np.meshgrid(r1_range, r2_range)

    # Evaluate KDE on the grid
    P = kde(np.vstack([R1.ravel(), R2.ravel()])).reshape(R1.shape)

    # Normalize P to sum to 1 (convert KDE estimate into probability)
    P /= np.sum(P)

    # Define physical constants
    kB = 0.010364 #0.001987  #8.617333262e-5  # Boltzmann constant in eV/K

    # Compute free energy surface
    F = -kB * T * np.log(P)
    F -= np.min(F)  # Normalize by setting the minimum to 0

    # Avoid numerical issues (replace very low probabilities with NaN)
    F[P < 1e-10] = np.nan

    return R1, R2, F

########################################################################################################
# Compute Histogram and Create 1D Surface along some Distance from a ASE Trajectory
########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
def get_1dSurface(traj, bond):
    """
    Returns the energy profile along a bond distance from an ASE trajectory
    using linear interpolation of potential energies.

    Parameters:
        traj  : ASE trajectory object (list of Atoms objects)
        bond  : Tuple specifying the atom indices for the bond (a, b)

    Returns:
        r_range     : Bond distances 
        F           : Free energy profile estimated from potential energy (units: eV)
        bond_lengths : Original bond distances (for segment scatter plot)
        energies : Corresponding potential energies (for segment scatter plot) (units: eV)
        
    Example:
        >>> r, F, r_raw, e_raw = get_1dSurface(traj, (0, 1))        
    """

    # Extract bond distances and potential energy
    bond_lengths = np.array([atoms.get_distance(*bond) for atoms in traj])
    energies = np.array([atoms.get_potential_energy() for atoms in traj])

    # Sort bond lengths for proper interpolation
    sorted_indices = np.argsort(bond_lengths)
    bond_lengths = bond_lengths[sorted_indices]
    energies = energies[sorted_indices]

    # Create interpolation function using Scipy
    interp_func = interp1d(bond_lengths, energies, kind='linear', fill_value="extrapolate")

    # Smooth bond distance in a grid for interpolation
    r_range = np.linspace(min(bond_lengths), max(bond_lengths), 200)
    F = interp_func(r_range)

    # Normalize free energy (shift to zero)
    F -= np.nanmin(F)
    
    return r_range, F, bond_lengths, energies
########################################################################################################
# Visualize Trajectory with nglview
########################################################################################################
from ase.io import read
import nglview as nv

def ViewTraj(traj, style="ball_and_stick", background="white", size=400):
    """
    Returns an interactive nglview widget with custom styling for a given ASE trajectory.

    Parameters:
        traj (list of ase.Atoms or str): A trajectory (list of ASE Atoms) or a file path.
        style (str): Representation style (e.g., 'ball_and_stick', 'spacefill', 'licorice').
        background (str): Viewer background color.

    Returns:
        nglview.NGLWidget: Configured NGL viewer widget.
    """
    # If a string is passed, assume it's a file path
    if isinstance(traj, str):
        traj = read(traj, index=':')

    # Create the viewer
    view = nv.NGLWidget(nv.ASETrajectory(traj))
    view.clear_representations()

    # Add the desired representation
    if style == "ball_and_stick":
        view.add_ball_and_stick()
    elif style == "spacefill":
        view.add_spacefill()
    elif style == "licorice":
        view.add_licorice()
    else:
        view.add_ball_and_stick()  # fallback

    # Add atom index labels
    view.add_label(
        selection='all',
        label_type='atomindex',
        color='black',
        zOffset=1.0,      # Push labels above atoms
        attachment='middle-center'
    )

    # Viewer settings
    view.background = background
    view.camera = "orthographic"
    view.center()
    view._set_size(f"{size}px", f"{size}px")
    view.parameters = {
        "clipNear": 0,
        "clipFar": 100,
        "clipDist": -5,
        "impostor": True,
        "fog": False,
        "antialias": True,
        "autoRotate": False
    }

    return view
########################################################################################################
#                                      End of File
########################################################################################################
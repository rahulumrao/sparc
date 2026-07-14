"""
Plotting utilities for SPARC workflow analysis using matplotlib.

All plotting functions preserve original logic from plot_utils.py.
Shared utilities are imported from main.py.
"""

import glob
import os

import numpy as np

from .main import (
    compute_mae,
    compute_rmse,
    extract_iteration_number,
    get_iteration_dirs,
)

try:
    import dpdata
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    _MATPLOT_AVAILABLE = True
except ImportError as _matplot_import_err:
    _MATPLOT_AVAILABLE = False
    _matplot_import_err_msg = str(_matplot_import_err)


def _require_matplot_deps():
    if not _MATPLOT_AVAILABLE:
        raise ImportError(
            "Matplotlib plotting requires: matplotlib, seaborn, pandas, dpdata.\n"
            f"Install with: pip install matplotlib seaborn pandas dpdata\n"
            f"(original error: {_matplot_import_err_msg})"
        )


########################################################################################################
# Parity plots for energy and forces
########################################################################################################


def ParityPlot(
    data_dir,
    model_path,
    per_atom=False,
    type="all",
    force_mode="components",
    heatmap=False,
    cmap="coolwarm",
    save_fig=None,
):
    """
    Generate parity plots for energy and/or force components with RMSE + MAE annotations.
    Points are colored by absolute error using the specified colormap.

    Parameters:
    -----------
    data_dir : str
        Path to test dataset in DeepMD .npy format.
    model_path : str
        Path to frozen model.
    per_atom : bool
        Whether to plot energy per atom instead of total.
    type : str
        'all' (default), 'energy', or 'forces'
    force_mode : str
        'components' (default) — separate fx/fy/fz plots.
        'flatten' — all force components in one plot.
    heatmap : bool
        If True, color points by absolute error using cmap (default: False).
    cmap : str
        Matplotlib colormap used when heatmap=True (default: 'coolwarm').
    save_fig : str or None
        Path to save the output figure.

    Example:
    >>> ParityPlot("data_dir", "frozen_model.pb", per_atom=True, type="energy", cmap="viridis")
    >>> ParityPlot("data_dir", "frozen_model.pb", per_atom=True, type="forces", force_mode="flatten")
    >>> ParityPlot("data_dir", "frozen_model.pb", per_atom=True, type="forces", force_mode="components")
    """
    _require_matplot_deps()
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
        e_unit_ann = "meV/Atom"
    else:
        e_unit = "eV"
        e_unit_ann = "meV"

    f_true = np.vstack(system["forces"])
    f_pred = np.vstack(prediction["forces"])

    # Setup plot layout
    forces_flatten = (type in ("all", "forces")) and force_mode in ("flatten", "flat")
    forces_components = (type in ("all", "forces")) and force_mode in (
        "components",
        "comp",
    )

    if type == "energy":
        fig, ax_energy = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
    elif type == "forces":
        if force_mode in ("flatten", "flat"):
            fig, ax_force = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    else:  # 'all'
        if force_mode in ("flatten", "flat"):
            fig, (ax_energy, ax_force) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
            ax_energy = axes[0, 0]

    # === Energy Parity Plot ===
    if type in ("all", "energy"):
        if heatmap:
            abs_err_e = np.abs(e_pred - e_true)
            ax_energy.scatter(
                e_true,
                e_pred,
                c=abs_err_e,
                cmap=cmap,
                alpha=0.85,
                s=40,
                edgecolors="none",
            )
        else:
            ax_energy.scatter(e_true, e_pred, c="blue", alpha=0.7, s=40, edgecolors="k")
        ax_energy.plot(
            [e_true.min(), e_true.max()], [e_true.min(), e_true.max()], "r--", lw=1.2
        )

        rmse_e = compute_rmse(e_true, e_pred)
        mae_e = compute_mae(e_true, e_pred)
        ax_energy.text(
            0.05,
            0.90,
            f"RMSE = {rmse_e * 1000:.2f} {e_unit_ann}\nMAE = {mae_e * 1000:.2f} {e_unit_ann}",
            transform=ax_energy.transAxes,
            fontsize=12,
            verticalalignment="top",
            color="blue",
        )
        ax_energy.set_xlabel(f"Observed (DFT) [{e_unit}]", fontsize=18)
        ax_energy.set_ylabel(f"Predicted (MLP) [{e_unit}]", fontsize=18)
        ax_energy.set_title("(A) Energy", fontsize=16)
        ax_energy.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax_energy.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax_energy.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_energy.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_energy.tick_params(labelsize=14)
        ax_energy.grid(ls="--", alpha=0.7)

    # === Force Parity — flatten mode ===
    if forces_flatten:
        f_t_all = f_true.ravel()
        f_p_all = f_pred.ravel()
        rmse_f = compute_rmse(f_t_all, f_p_all)
        mae_f = compute_mae(f_t_all, f_p_all)
        if heatmap:
            abs_err_f = np.abs(f_p_all - f_t_all)
            ax_force.scatter(
                f_t_all,
                f_p_all,
                c=abs_err_f,
                cmap=cmap,
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        else:
            ax_force.scatter(
                f_t_all, f_p_all, c="blue", alpha=0.6, s=15, edgecolors="k"
            )
        ax_force.plot(
            [f_t_all.min(), f_t_all.max()],
            [f_t_all.min(), f_t_all.max()],
            "r--",
            lw=1.2,
        )
        ax_force.text(
            0.05,
            0.90,
            f"RMSE = {rmse_f * 1000:.2f} meV/Å\nMAE = {mae_f * 1000:.2f} meV/Å",
            transform=ax_force.transAxes,
            fontsize=12,
            verticalalignment="top",
            color="blue",
        )
        ax_force.set_xlabel(r"Observed (DFT) [eV/$\rm{\AA}$]", fontsize=18)
        ax_force.set_ylabel(r"Predicted (MLP) [eV/$\rm{\AA}$]", fontsize=18)
        title_label = "(B) Forces" if type == "all" else "(A) Forces"
        ax_force.set_title(title_label, fontsize=16)
        ax_force.tick_params(labelsize=14)
        ax_force.grid(ls="--", alpha=0.7)

    # === Force Parity — components mode ===
    if forces_components:
        components = ["fx", "fy", "fz"]
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
            if heatmap:
                abs_err_f = np.abs(f_p - f_t)
                ax.scatter(
                    f_t, f_p, c=abs_err_f, cmap=cmap, alpha=0.8, s=30, edgecolors="none"
                )
            else:
                ax.scatter(f_t, f_p, c="blue", alpha=0.6, s=30, edgecolors="k")
            ax.plot([f_t.min(), f_t.max()], [f_t.min(), f_t.max()], "r--", lw=1.2)
            ax.text(
                0.05,
                0.90,
                f"RMSE = {rmse_f * 1000:.2f} meV/Å\nMAE = {mae_f * 1000:.2f} meV/Å",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                color="blue",
            )
            ax.set_xlabel(r"Observed (DFT) [eV/$\rm{\AA}$]", fontsize=16)
            ax.set_ylabel(r"Predicted (MLP) [eV/$\rm{\AA}$]", fontsize=16)
            ax.set_title(f"({chr(66 + i)}) {comp}", fontsize=16)
            ax.tick_params(labelsize=14)
            ax.grid(ls="--", alpha=0.7)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
        print(f"[ANALYSIS][INFO] Saved parity plot to: {save_fig}")
    else:
        plt.show()


########################################################################################################
# Plot Learning Curve
########################################################################################################


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
    _require_matplot_deps()
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
        "rmse_f_trn": "RMSE Force (train)",
    }

    plt.figure(figsize=(8, 6), dpi=250)

    for key, label in legends.items():
        if key in data.columns:
            plt.loglog(data["step"], data[key], label=label, lw=2.2, alpha=0.9)

    plt.xlabel("Training steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("DeepMD Learning Curve", fontsize=16, pad=10)
    plt.legend(fontsize=12)
    plt.grid(which="both", ls="--", alpha=0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
        print(f"[ANALYSIS][INFO] Saved learning curve plot to: {save_fig}")
    else:
        plt.show()


########################################################################################################
# Plot error in Forces from various DeepMD models
########################################################################################################


def PlotForceDeviation(
    root_dir=".", iteration_window="all", target_iteration=None, dmin=0.05, dmax=0.5
):
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
    _require_matplot_deps()
    data_dict = {}
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    # Loop over selected iter_* directories
    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)
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
    ncol = min(num_iterations, 5)

    for model, data in data_dict.items():
        for iter_num, steps, max_devi_f in sorted(data):
            plt.plot(
                steps,
                max_devi_f,
                linestyle="-",
                lw=2.2,
                label=f"Iter: {iter_num}",
                marker="o",
                ms=5,
                alpha=0.7,
            )

    plt.axhline(y=dmin, color="black", lw=2, ls="--", alpha=0.4)
    plt.axhline(y=dmax, color="black", lw=2, ls="--", alpha=0.4)
    plt.fill_between(
        [0, max(steps) if steps else 1], dmin, dmax, color="grey", alpha=0.3
    )
    plt.xlim(0, None)
    plt.xlabel("Candidates", fontsize=22)
    plt.ylabel(r"Max. Force Deviation ($\rm{eV/\AA}$)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=ncol)
    plt.grid(ls="-.")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


########################################################################################################
# Violin plot of force deviation error distribution across AL iterations
########################################################################################################


def PlotForceError(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    dmin=None,
    dmax=0.5,
    bw_adjust=1.4,
    gridsize=200,
    connect_means=True,
    connect_medians=False,
    log_scale=False,
    palette="viridis",
    violin_alpha=0.85,
    save_fig=None,
):
    """
    Violin plot of max force deviation distribution across AL iterations.

    Parameters:
    root_dir (str): Root directory containing iter_* folders.
    iteration_window (tuple or str): (start, end) or "all".
    target_iteration (int): Specific iteration to analyze.
    dmin (float): Threshold shown as dashed horizontal line (default: 0.05).
    dmax (float): Upper y-axis limit for linear scale (default: 0.5).
    bw_adjust (float): KDE bandwidth scaling (default: 1.4).
    gridsize (int): KDE grid resolution (default: 200).
    connect_means (bool): Draw mean trajectory line (default: True).
    connect_medians (bool): Draw median trajectory line (default: False).
    log_scale (bool): Use log10 y-axis with 10^x tick labels (default: False).
    palette (str): Seaborn color palette (default: 'viridis').
    violin_alpha (float): Opacity of violin bodies (default: 0.85).
    save_fig (str or None): Path to save figure.

    Example:
    >>> PlotForceError(iteration_window="all", palette="hot", log_scale=True)
    >>> PlotForceError(iteration_window=(0, 5), connect_means=True, dmin=0.05)
    """
    _require_matplot_deps()
    data_dict = {}
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)
        dpmd_dir = os.path.join(iter_dir, "02.dpmd")
        if not os.path.isdir(dpmd_dir):
            continue

        model_files = sorted(glob.glob(os.path.join(dpmd_dir, "model_dev_*.out")))
        for model_file in model_files:
            model_name = os.path.basename(model_file)
            steps = []
            max_devi_f = []
            with open(model_file, "r") as f:
                lines = f.readlines()
            for line in lines[2:]:
                cols = line.split()
                if len(cols) >= 5:
                    try:
                        steps.append(int(cols[0]))
                        max_devi_f.append(float(cols[4]))
                    except ValueError:
                        continue
            if model_name not in data_dict:
                data_dict[model_name] = []
            data_dict[model_name].append((iter_num, steps, max_devi_f))

    rows = []
    for model, data in data_dict.items():
        for iter_num, _steps, max_devi_f_list in sorted(data):
            for v in max_devi_f_list:
                if v > 0:
                    rows.append(
                        {
                            "iter": str(iter_num),
                            "iter_i": iter_num,
                            "max_devi_f": float(v),
                            "log10_max_devi_f": np.log10(float(v)),
                        }
                    )
    df_violin = pd.DataFrame(rows)
    n_iter = sorted(df_violin["iter_i"].unique())
    order = [str(x) for x in n_iter]
    ycol = "log10_max_devi_f" if log_scale else "max_devi_f"

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    sns.violinplot(
        data=df_violin,
        x="iter",
        y=ycol,
        order=order,
        hue="iter",
        palette=palette,
        legend=False,
        inner="box",
        cut=0,
        linewidth=1.2,
        dodge=False,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        ax=ax,
    )
    for coll in ax.collections:
        try:
            coll.set_alpha(violin_alpha)
        except Exception:
            pass

    if dmin is not None:
        thresh_y = np.log10(dmin) if log_scale else dmin
        ax.axhline(
            thresh_y, color="royalblue", linestyle="--", linewidth=0.7, zorder=10
        )
    if not log_scale:
        ax.set_ylim(0, dmax)

    ax.grid(True, linestyle="-.", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("AL Iterations", fontsize=16)
    ax.set_ylabel(r"Maximum Force Deviation (eV/$\mathrm{\AA}$)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=15)

    if log_scale:
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [
                rf"$10^{{{t:.0f}}}$"
                if abs(t - round(t)) < 1e-6
                else rf"$10^{{{t:.1f}}}$"
                for t in yticks
            ]
        )

    xpos = np.arange(len(n_iter))

    if connect_means:
        means = (
            df_violin.groupby("iter_i", observed=True)["max_devi_f"]
            .mean()
            .reindex(n_iter)
            .values
        )
        means = np.asarray(means, dtype=float)
        y_mean = np.log10(np.where(means > 0, means, np.nan)) if log_scale else means
        ax.plot(xpos, y_mean, color="black", linewidth=2.2, zorder=20, label="Mean")
        ax.scatter(
            xpos,
            y_mean,
            facecolor="white",
            edgecolor="black",
            s=55,
            linewidth=1.5,
            zorder=21,
        )

    if connect_medians:
        meds = (
            df_violin.groupby("iter_i", observed=True)["max_devi_f"]
            .median()
            .reindex(n_iter)
            .values
        )
        meds = np.asarray(meds, dtype=float)
        y_med = np.log10(np.where(meds > 0, meds, np.nan)) if log_scale else meds
        ax.plot(
            xpos,
            y_med,
            color="gray",
            linestyle="--",
            linewidth=1.8,
            zorder=19,
            label="Median",
        )
        ax.scatter(
            xpos,
            y_med,
            facecolor="white",
            edgecolor="gray",
            s=35,
            linewidth=1.2,
            zorder=19,
        )

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        plt.show()


########################################################################################################
# Plot Potential energy from Labelled candidates
########################################################################################################


def PlotPotentialEnergy(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    traj_filename="AseMD.traj",
):
    """
    Parses trajectory files from iter_* directories (or a specific range/iteration/all),
    extracts potential energy, and plots the results.

    Parameters:
    root_dir (str): Root directory containing iter_* folders.
    iteration_window (tuple or str): A tuple (start, end) to specify a range of iterations, or "all" to process all iterations.
    target_iteration (int): A specific iteration number to analyze.
    traj_filename (str): Name of the trajectory file to read in each iteration folder (default: "dpmd.traj").

    Example:
    >>> PlotPotentialEnergy("/path/to/root", iteration_window=(2, 5), traj_filename="AseMD.traj")
    >>> PlotPotentialEnergy("/path/to/root", target_iteration=3, traj_filename="AseMD.traj")
    >>> PlotPotentialEnergy("/path/to/root", iteration_window="all", traj_filename="AseMD.traj")
    """
    _require_matplot_deps()
    if traj_filename is None:
        traj_filename = input("Enter the trajectory filename (e.g., AseMD.traj): ")

    energy_dict = {}
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    # Loop over selected iter_* directories
    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)
        dpmd_dir = os.path.join(iter_dir, "00.dft")
        if not os.path.isdir(dpmd_dir):
            continue

        # Find the specified trajectory file
        traj_path = os.path.join(dpmd_dir, traj_filename)
        if not os.path.isfile(traj_path):
            print(f"Warning: {traj_filename} not found in {dpmd_dir}")
            continue

        # Read trajectory file
        from ase.io import read

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
    ncol = min(num_iterations, 6)

    for iter_num, energies in sorted(energy_dict.items()):
        plt.plot(
            range(len(energies)),
            energies,
            marker="o",
            linestyle="-",
            lw=2,
            label=f"Iter {iter_num}",
            ms=5,
            alpha=0.9,
        )

    # Labels and formatting
    plt.xlabel("Labelled Candidates", fontsize=20)
    plt.ylabel("Potential Energy (eV)", fontsize=20)
    plt.xlim(0, None)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=ncol)
    plt.grid(ls="-.")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


########################################################################################################
# Plot Distribution of Properties
########################################################################################################


def PlotDistribution(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    traj_filename="AseMD.traj",
    type="line",  # Options: "line", "kde", "hist"
    get="energy",  # "energy" or "distance:i,j"
    **kwargs,
):
    """
    Plot potential energy or bond distance from ASE trajectories across iter_* folders.

    [Full docstring preserved from original]
    """
    _require_matplot_deps()
    property_dict = {}
    is_energy = get.lower() == "energy"
    is_distance = get.lower().startswith("distance:")
    symbol_pair = None

    if is_energy:
        prop_label = "Potential Energy (eV)"
    elif is_distance:
        try:
            i, j = map(int, get.split(":")[1].split(","))
        except Exception:
            raise ValueError("For distance, use format: 'distance:i,j'")
    else:
        raise ValueError("get must be 'energy' or 'distance:i,j'")

    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)
        dpmd_dir = os.path.join(iter_dir, "00.dft")
        traj_path = os.path.join(dpmd_dir, traj_filename)

        if not os.path.isfile(traj_path):
            print(f"Warning: {traj_filename} not found in {dpmd_dir}")
            continue

        from ase.io import read

        traj = read(traj_path, index=":")

        if is_distance and symbol_pair is None and len(traj) > max(i, j):
            try:
                symbols = traj[0].get_chemical_symbols()
                symbol_pair = (symbols[i], symbols[j])
            except Exception as ex:
                print(f" Could not extract atom symbols: {ex}")
                symbol_pair = None

        if is_energy:
            values = [
                float(np.asarray(atoms.get_potential_energy()).flat[0])
                for atoms in traj
            ]
        elif is_distance:
            values = [atoms.get_distance(i, j) for atoms in traj]

        property_dict[iter_num] = values

    # Determine labels
    if is_energy:
        if type == "line":
            xlabel = "Candidates"
            ylabel = "Potential Energy (eV)"
        elif type == "kde":
            xlabel = "Potential Energy (eV)"
            ylabel = "Density"
        elif type == "hist":
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
                range(len(values)),
                values,
                label=label,
                marker=kwargs.get("marker", "o"),
                linestyle=kwargs.get("linestyle", "-"),
                linewidth=kwargs.get("linewidth", 2),
                markersize=kwargs.get("ms", 5),
                alpha=kwargs.get("alpha", 0.9),
                color=kwargs.get("color", None),
            )
        elif type == "kde":
            sns.kdeplot(
                values,
                label=label,
                linewidth=kwargs.get("linewidth", 2),
                fill=kwargs.get("fill", True),
                alpha=kwargs.get("alpha", 0.4),
                linestyle=kwargs.get("linestyle", "-"),
                color=kwargs.get("color", None),
            )
        elif type == "hist":
            plt.hist(
                values,
                bins=kwargs.get("bins", 25),
                edgecolor=kwargs.get("edgecolor", "black"),
                alpha=kwargs.get("alpha", 0.7),
                label=label,
                color=kwargs.get("color", None),
            )
        else:
            raise ValueError("type must be 'line', 'kde', or 'hist'.")

    plt.xlabel(xlabel, fontsize=23)
    plt.ylabel(ylabel if type != "line" else prop_label, fontsize=23)
    if type == "line":
        plt.xlim(0, None)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=ncol)
    plt.grid(ls="-.")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
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
    **kwargs,
):
    """
    Plot energy vs. bond distance across trajectories as a 2D KDE, heatmap, or hexbin.

    [Full docstring preserved from original]
    """
    _require_matplot_deps()
    i, j = distance_pair
    x_vals, y_vals = [], []
    total_frames = 0
    symbol_pair = None

    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    print("Parsing iterations:")
    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)
        traj_path = os.path.join(iter_dir, "00.dft", traj_filename)

        if not os.path.isfile(traj_path):
            print(f" Iter {iter_num:>2}: MISSING ({traj_filename})")
            continue

        from ase.io import read

        traj = read(traj_path, index=":")
        num_frames = len(traj)
        print(f" Iter {iter_num:>2}: {num_frames} frames")
        total_frames += num_frames

        # Get atomic symbols from the first valid trajectory only
        if symbol_pair is None and num_frames > max(i, j):
            try:
                symbols = traj[0].get_chemical_symbols()
                symbol_pair = (symbols[i], symbols[j])
            except Exception as ex:
                print(f" Could not extract atom symbols: {ex}")
                symbol_pair = None

        for atoms in traj:
            try:
                d = atoms.get_distance(i, j)
                e = float(np.asarray(atoms.get_potential_energy()).flat[0])
                x_vals.append(d)
                y_vals.append(e)
            except Exception as ex:
                print(f" Error in {traj_path}: {ex}")
                continue

    print(f"\nTotal frames: {total_frames}")

    x = np.array(x_vals)
    y = np.array(y_vals)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 7), dpi=250)

    if type == "kde":
        sns.kdeplot(
            x=x,
            y=y,
            ax=ax,
            fill=True,
            cmap=kwargs.pop("cmap", "viridis"),
            levels=100,
            thresh=0.05,
            **kwargs,
        )
    elif type == "heatmap":
        h = ax.hist2d(
            x, y, bins=bins, cmap=kwargs.pop("cmap", "plasma"), cmin=1, **kwargs
        )
        cb = fig.colorbar(h[3], ax=ax, pad=0.01)
        cb.set_label("Counts", fontsize=16)
        cb.ax.tick_params(labelsize=16)
    elif type == "hexbin":
        cmap = kwargs.pop("cmap", "inferno")
        gridsize = bins[0] if isinstance(bins, (tuple, list)) else bins
        hb = ax.hexbin(
            x, y, gridsize=gridsize, cmap=cmap, mincnt=1, linewidths=0.3, **kwargs
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
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(ls="--", alpha=0.3)
    fig.subplots_adjust(right=0.98, left=0.12, top=0.95, bottom=0.12)
    plt.show()


########################################################################################################
# Plot Temperature from Deep Potential Dynamics
########################################################################################################


def PlotTemp(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    traj_filename="dpmd.traj",
):
    """
    Parses trajectory files from iter_* directories (or a specific range/iteration/all),
    extracts temperature, and plots the results.

    [Full docstring preserved from original]
    """
    _require_matplot_deps()
    if traj_filename is None:
        traj_filename = (
            input("Enter the trajectory filename (default: dpmd.traj): ") or "dpmd.traj"
        )

    temp_dict = {}
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    # Loop over selected iter_* directories
    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)
        dpmd_dir = os.path.join(iter_dir, "02.dpmd")
        if not os.path.isdir(dpmd_dir):
            continue

        # Find the specified trajectory file
        traj_path = os.path.join(dpmd_dir, traj_filename)
        if not os.path.isfile(traj_path):
            print(f"Warning: {traj_filename} not found in {dpmd_dir}")
            continue

        # Read trajectory file
        from ase.io import read

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
    ncol = min(num_iterations, 6)

    for iter_num, temperatures in sorted(temp_dict.items()):
        plt.plot(
            range(len(temperatures)),
            temperatures,
            marker="o",
            linestyle="-",
            lw=2.2,
            label=f"Iter {iter_num}",
            ms=5,
            alpha=0.8,
        )
        mean_temp = np.mean(temperatures)
        print(f"Mean Temperature (K) from Iteration {iter_num} := {mean_temp:.2f}")

    # Labels and formatting
    plt.xlabel("MD Steps", fontsize=20)
    plt.ylabel("Temperature (K)", fontsize=20)
    plt.xlim(0, None)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=ncol)
    plt.grid(ls="-.")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


########################################################################################################
# Plot AL workflow step timings
########################################################################################################


def PlotWorkflowTiming(
    timing_file="timings.csv",
    plot_type="grouped",
    unit="h",
    save_fig=None,
    Ycmin=None,
    Ycmax=None,
):
    """
    Plot per-step wall-clock times from ``timings.csv``.

    Parameters
    ----------
    timing_file : str
        Path to ``timings.csv``.
    plot_type : str
        ``grouped`` (default) or ``line``.
    unit : str
        ``h`` (default) or ``m`` for the y-axis scale.
    save_fig : str or None
        If provided, save the figure to this path.
    Ycmin : (float, float), optional
        Explicit (lo, hi) for the bottom panel of broken axis, e.g. (0, 4).
    Ycmax : (float, float), optional
        Explicit (lo, hi) for the top panel of broken axis, e.g. (6, 16).
    """
    _require_matplot_deps()
    from sparc.src.utils.plotting.timing_plot import PlotStepTimingLine, PlotStepTimings
    from sparc.src.utils.timing import load_workflow_timing

    try:
        df = load_workflow_timing(timing_file)
    except FileNotFoundError:
        print(f"[ANALYSIS][ERROR] Timing file not found: {timing_file}")
        return

    if df.empty:
        print(f"[ANALYSIS][ERROR] No timing records in: {timing_file}")
        return

    if plot_type == "line":
        PlotStepTimingLine(df, unit=unit, save_fig=save_fig, show=save_fig is None)
    else:
        PlotStepTimings(
            df,
            unit=unit,
            save_fig=save_fig,
            show=save_fig is None,
            Ycmin=Ycmin,
            Ycmax=Ycmax,
        )


########################################################################################################
# END OF FILE
########################################################################################################

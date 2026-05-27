"""
Plotting utilities for SPARC workflow analysis using Plotly.

Interactive, web-based plots with zoom, pan, and hover capabilities.
All plotting functions leverage shared utilities from main.py.
"""

import os

import dpdata
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import shared utilities
from .main import (
    compute_mae,
    compute_rmse,
    extract_iteration_number,
    get_iteration_dirs,
    load_trajectory,
)

########################################################################################################
# Parity plots for energy and forces
########################################################################################################


def ParityPlot(data_dir, model_path, per_atom=False, type="all", save_fig=None):
    """
    Generate interactive parity plots for energy and/or force components.

    Parameters
    ----------
    data_dir : str
        Path to test dataset in DeepMD .npy format
    model_path : str
        Path to frozen model
    per_atom : bool
        Whether to plot energy per atom
    type : str
        'all' (default), 'energy', or 'forces'
    save_fig : str or None
        Path to save HTML file

    Example
    -------
    >>> ParityPlot("data_dir", "model.pb", save_fig='parity.html')
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

    # Create plots
    if type == "energy":
        fig = _plot_energy_parity(e_true, e_pred, e_unit)
    elif type == "forces":
        fig = _plot_forces_parity(f_true, f_pred)
    else:  # 'all'
        fig = _plot_all_parity(e_true, e_pred, e_unit, f_true, f_pred)

    if save_fig:
        fig.write_html(save_fig)
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        fig.show()


def _plot_energy_parity(e_true, e_pred, e_unit):
    """Create energy parity plot."""
    rmse = compute_rmse(e_true, e_pred)
    mae = compute_mae(e_true, e_pred)

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=e_true,
            y=e_pred,
            mode="markers",
            marker=dict(
                size=8, color="blue", opacity=0.7, line=dict(color="black", width=1)
            ),
            name="Data",
            hovertemplate="DFT: %{x:.3f}<br>MLP: %{y:.3f}<extra></extra>",
        )
    )

    # Ideal line
    fig.add_trace(
        go.Scatter(
            x=[e_true.min(), e_true.max()],
            y=[e_true.min(), e_true.max()],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Ideal",
            hoverinfo="skip",
        )
    )

    # Annotation
    fig.add_annotation(
        text=f"RMSE = {rmse:.4f} {e_unit}<br>MAE = {mae:.4f} {e_unit}",
        xref="paper",
        yref="paper",
        x=0.05,
        y=0.95,
        showarrow=False,
        font=dict(size=14, color="blue"),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="blue",
        borderwidth=1,
    )

    fig.update_layout(
        title="Energy Parity Plot",
        xaxis_title=f"Observed (DFT) [{e_unit}]",
        yaxis_title=f"Predicted (MLP) [{e_unit}]",
        width=600,
        height=500,
        template="plotly_white",
        font=dict(size=14),
        hovermode="closest",
    )

    return fig


def _plot_forces_parity(f_true, f_pred):
    """Create forces parity plot."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=["fx", "fy", "fz"])

    components = ["fx", "fy", "fz"]
    for i, comp in enumerate(components):
        f_t = f_true[:, i]
        f_p = f_pred[:, i]
        rmse_f = compute_rmse(f_t, f_p)
        mae_f = compute_mae(f_t, f_p)

        fig.add_trace(
            go.Scatter(
                x=f_t,
                y=f_p,
                mode="markers",
                marker=dict(
                    size=6,
                    color="blue",
                    opacity=0.6,
                    line=dict(color="black", width=0.5),
                ),
                showlegend=False,
                hovertemplate=f"{comp} DFT: %{{x:.3f}}<br>{comp} MLP: %{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=i + 1,
        )

        # Ideal line
        fig.add_trace(
            go.Scatter(
                x=[f_t.min(), f_t.max()],
                y=[f_t.min(), f_t.max()],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=i + 1,
        )

        # Annotation
        fig.add_annotation(
            text=f"RMSE={rmse_f:.4f}<br>MAE={mae_f:.4f}",
            xref=f"x{i + 1}",
            yref=f"y{i + 1}",
            x=f_t.min() + 0.05 * (f_t.max() - f_t.min()),
            y=f_t.max() - 0.1 * (f_t.max() - f_t.min()),
            showarrow=False,
            font=dict(size=12, color="blue"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue",
            borderwidth=1,
        )

        fig.update_xaxes(title_text="Observed (DFT) [eV/Å]", row=1, col=i + 1)
        fig.update_yaxes(title_text="Predicted (MLP) [eV/Å]", row=1, col=i + 1)

    fig.update_layout(
        width=1400, height=400, template="plotly_white", hovermode="closest"
    )
    return fig


def _plot_all_parity(e_true, e_pred, e_unit, f_true, f_pred):
    """Create combined energy + forces parity plot."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["(A) Energy", "(B) fx", "(C) fy", "(D) fz"],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # Energy plot
    rmse_e = compute_rmse(e_true, e_pred)
    mae_e = compute_mae(e_true, e_pred)

    fig.add_trace(
        go.Scatter(
            x=e_true,
            y=e_pred,
            mode="markers",
            marker=dict(
                size=8, color="blue", opacity=0.7, line=dict(color="black", width=1)
            ),
            showlegend=False,
            hovertemplate="DFT: %{x:.3f}<br>MLP: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[e_true.min(), e_true.max()],
            y=[e_true.min(), e_true.max()],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_annotation(
        text=f"RMSE={rmse_e:.4f}<br>MAE={mae_e:.4f}",
        xref="x",
        yref="y",
        x=e_true.min() + 0.05 * (e_true.max() - e_true.min()),
        y=e_true.max() - 0.1 * (e_true.max() - e_true.min()),
        showarrow=False,
        font=dict(size=12, color="blue"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="blue",
        borderwidth=1,
    )

    fig.update_xaxes(title_text=f"Observed [{e_unit}]", row=1, col=1)
    fig.update_yaxes(title_text=f"Predicted [{e_unit}]", row=1, col=1)

    # Force plots
    components = ["fx", "fy", "fz"]
    positions = [(1, 2), (2, 1), (2, 2)]

    for i, (comp, pos) in enumerate(zip(components, positions)):
        f_t = f_true[:, i]
        f_p = f_pred[:, i]
        rmse_f = compute_rmse(f_t, f_p)
        mae_f = compute_mae(f_t, f_p)

        fig.add_trace(
            go.Scatter(
                x=f_t,
                y=f_p,
                mode="markers",
                marker=dict(
                    size=6,
                    color="blue",
                    opacity=0.6,
                    line=dict(color="black", width=0.5),
                ),
                showlegend=False,
                hovertemplate=f"{comp}: %{{x:.3f}} → %{{y:.3f}}<extra></extra>",
            ),
            row=pos[0],
            col=pos[1],
        )

        fig.add_trace(
            go.Scatter(
                x=[f_t.min(), f_t.max()],
                y=[f_t.min(), f_t.max()],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=pos[0],
            col=pos[1],
        )

        fig.add_annotation(
            text=f"RMSE={rmse_f:.4f}<br>MAE={mae_f:.4f}",
            xref=f"x{i + 2}",
            yref=f"y{i + 2}",
            x=f_t.min() + 0.05 * (f_t.max() - f_t.min()),
            y=f_t.max() - 0.1 * (f_t.max() - f_t.min()),
            showarrow=False,
            font=dict(size=12, color="blue"),
            bgcolor="rgba(255,255,255,0.8)",
        )

        fig.update_xaxes(title_text="Observed [eV/Å]", row=pos[0], col=pos[1])
        fig.update_yaxes(title_text="Predicted [eV/Å]", row=pos[0], col=pos[1])

    fig.update_layout(
        width=1000, height=800, template="plotly_white", hovermode="closest"
    )
    return fig


########################################################################################################
# Plot Learning Curve
########################################################################################################


def PlotLcurve(lcurve_file, save_fig=None):
    """
    Plot interactive learning curve from DeepMD training log.

    Parameters
    ----------
    lcurve_file : str
        Path to lcurve.out file
    save_fig : str or None
        Path to save HTML file
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
        "rmse_f_trn": "RMSE Force (train)",
    }

    fig = go.Figure()

    for key, label in legends.items():
        if key in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data["step"],
                    y=data[key],
                    mode="lines",
                    name=label,
                    line=dict(width=3),
                    hovertemplate="Step: %{x}<br>Loss: %{y:.4e}<extra></extra>",
                )
            )

    fig.update_layout(
        # UPDATED: Center the title
        title={
            "text": "DeepMD Learning Curve",
            "x": 0.5,  # Center position
            "xanchor": "center",  # Anchor at center
            "font": {"size": 22},  # Larger title font
        },
        xaxis_title="Training steps",
        yaxis_title="Loss",
        xaxis_type="log",
        yaxis_type="log",
        width=900,
        height=600,
        template="plotly_white",
        font=dict(size=16),  # Increased base font size from 14 to 16
        legend=dict(x=0.7, y=0.95, font=dict(size=16)),  # Larger legend font
        hovermode="x unified",
    )

    # UPDATED: Increase axis label and tick font sizes
    fig.update_xaxes(
        title_font=dict(size=20),  # Larger x-axis label
        tickfont=dict(size=16),  # Larger x-axis tick labels
    )

    fig.update_yaxes(
        title_font=dict(size=20),  # Larger y-axis label
        tickfont=dict(size=16),  # Larger y-axis tick labels
    )

    if save_fig:
        fig.write_html(save_fig)
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        fig.show()


########################################################################################################
# Plot Force Deviation
########################################################################################################


def PlotForceDeviation(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    dmin=0.05,
    dmax=0.5,
    save_fig=None,
):
    """
    Plot max force deviation from model_dev_*.out files.

    Parameters
    ----------
    root_dir : str
        Root directory containing iter_* folders
    iteration_window : tuple or str
        (start, end) or "all"
    target_iteration : int
        Specific iteration
    dmin, dmax : float
        Lower and upper thresholds
    save_fig : str
        Path to save HTML file
    """
    data_dict = {}

    # Use main.py utility
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)
        dpmd_dir = os.path.join(iter_dir, "02.dpmd")

        if not os.path.isdir(dpmd_dir):
            continue

        import glob

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

    # Plotting
    fig = go.Figure()

    for model, data in data_dict.items():
        for iter_num, steps, max_devi_f in sorted(data):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=max_devi_f,
                    mode="lines+markers",
                    name=f"Iter {iter_num}",
                    line=dict(width=3),
                    marker=dict(size=6),
                    hovertemplate="Candidate: %{x}<br>Max Deviation: %{y:.4f} eV/Å<extra></extra>",
                )
            )

    # Add threshold lines
    max_steps = max(
        [max(steps) for _, data in data_dict.items() for _, steps, _ in data]
    )

    fig.add_hline(
        y=dmin,
        line_dash="dash",
        line_color="black",
        line_width=2,
        opacity=0.6,
        annotation_text=f"dmin={dmin}",
    )
    fig.add_hline(
        y=dmax,
        line_dash="dash",
        line_color="black",
        line_width=2,
        opacity=0.6,
        annotation_text=f"dmax={dmax}",
    )

    # Shaded region
    fig.add_shape(
        type="rect",
        x0=0,
        x1=max_steps,
        y0=dmin,
        y1=dmax,
        fillcolor="grey",
        opacity=0.2,
        line_width=0,
        layer="below",
    )

    fig.update_layout(
        title="Force Deviation from Iterations",
        xaxis_title="Candidates",
        yaxis_title="Max. Force Deviation (eV/Å)",
        width=1200,
        height=700,
        template="plotly_white",
        font=dict(size=18),
        hovermode="closest",
    )

    if save_fig:
        fig.write_html(save_fig)
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        fig.show()


########################################################################################################
# Plot Potential Energy
########################################################################################################


def PlotPotentialEnergy(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    traj_filename="AseMD.traj",
    save_fig=None,
):
    """
    Plot potential energy from labelled candidates.

    Parameters
    ----------
    root_dir : str
        Root directory
    iteration_window : tuple or str
        (start, end) or "all"
    target_iteration : int
        Specific iteration
    traj_filename : str
        Trajectory filename
    save_fig : str
        Path to save HTML file
    """
    energy_dict = {}

    # Use main.py utility
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)

        # Use main.py loader
        traj = load_trajectory(iter_dir, subdir="00.dft", traj_filename=traj_filename)

        if traj is None:
            continue

        energies = [atoms.get_potential_energy() for atoms in traj]

        if iter_num not in energy_dict:
            energy_dict[iter_num] = []
        energy_dict[iter_num].extend(energies)

    # Plotting
    fig = go.Figure()

    for iter_num, energies in sorted(energy_dict.items()):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(energies))),
                y=energies,
                mode="lines+markers",
                name=f"Iter {iter_num}",
                line=dict(width=2),
                marker=dict(size=5),
                hovertemplate="Candidate: %{x}<br>Energy: %{y:.3f} eV<extra></extra>",
            )
        )

    fig.update_layout(
        title="Potential Energy vs Labelled Candidates",
        xaxis_title="Labelled Candidates",
        yaxis_title="Potential Energy (eV)",
        width=1200,
        height=700,
        template="plotly_white",
        font=dict(size=18),
        hovermode="x unified",
    )

    if save_fig:
        fig.write_html(save_fig)
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        fig.show()


########################################################################################################
# Plot Distribution
########################################################################################################


def PlotDistribution(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    traj_filename="AseMD.traj",
    bins=50,
    save_fig=None,
):
    """
    Plot energy and force distributions as interactive histograms.

    Parameters
    ----------
    root_dir : str
        Root directory
    iteration_window : tuple or str
        (start, end) or "all"
    target_iteration : int
        Specific iteration
    traj_filename : str
        Trajectory filename
    bins : int
        Number of histogram bins
    save_fig : str
        Path to save HTML file
    """
    energy_data = []
    force_data = []

    # Use main.py utility
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)

        # Use main.py loader
        traj = load_trajectory(iter_dir, subdir="00.dft", traj_filename=traj_filename)

        if traj is None:
            continue

        for atoms in traj:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))

            energy_data.append({"Iteration": f"Iter {iter_num}", "Energy": energy})
            force_data.append({"Iteration": f"Iter {iter_num}", "Max Force": max_force})

    df_energy = pd.DataFrame(energy_data)
    df_force = pd.DataFrame(force_data)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Energy Distribution", "Max Force Distribution")
    )

    # Energy histogram
    for iteration in sorted(df_energy["Iteration"].unique()):
        data = df_energy[df_energy["Iteration"] == iteration]["Energy"]
        fig.add_trace(
            go.Histogram(
                x=data,
                name=iteration,
                opacity=0.7,
                nbinsx=bins,
                hovertemplate="Energy: %{x:.2f} eV<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Force histogram
    for iteration in sorted(df_force["Iteration"].unique()):
        data = df_force[df_force["Iteration"] == iteration]["Max Force"]
        fig.add_trace(
            go.Histogram(
                x=data,
                name=iteration,
                opacity=0.7,
                nbinsx=bins,
                showlegend=False,
                hovertemplate="Max Force: %{x:.2f} eV/Å<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Energy (eV)", row=1, col=1)
    fig.update_xaxes(title_text="Max Force (eV/Å)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)

    fig.update_layout(
        width=1400,
        height=600,
        template="plotly_white",
        font=dict(size=14),
        barmode="overlay",
        hovermode="closest",
    )

    if save_fig:
        fig.write_html(save_fig)
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        fig.show()


########################################################################################################
# Plot PES (Potential Energy Surface)
########################################################################################################


def PlotPES(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    traj_filename="AseMD.traj",
    coord_type="distance",
    atom_indices=None,
    save_fig=None,
):
    """
    Plot 2D potential energy surface using KDE or contour.

    Parameters
    ----------
    root_dir : str
        Root directory
    iteration_window : tuple or str
        (start, end) or "all"
    target_iteration : int
        Specific iteration
    traj_filename : str
        Trajectory filename
    coord_type : str
        "distance", "angle", or "dihedral"
    atom_indices : list of tuples
        [(i1,j1), (i2,j2)] for distances
    save_fig : str
        Path to save HTML file
    """
    if atom_indices is None or len(atom_indices) != 2:
        print("[ANALYSIS][ERROR] Provide exactly 2 coordinate specifications")
        return

    coord1_list = []
    coord2_list = []
    energy_list = []

    # Use main.py utility
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    for iter_dir in selected_dirs:
        # Use main.py loader
        traj = load_trajectory(iter_dir, subdir="00.dft", traj_filename=traj_filename)

        if traj is None:
            continue

        for atoms in traj:
            energy = atoms.get_potential_energy()

            if coord_type == "distance":
                coord1 = atoms.get_distance(*atom_indices[0])
                coord2 = atoms.get_distance(*atom_indices[1])
            elif coord_type == "angle":
                coord1 = atoms.get_angle(*atom_indices[0])
                coord2 = atoms.get_angle(*atom_indices[1])
            elif coord_type == "dihedral":
                coord1 = atoms.get_dihedral(*atom_indices[0])
                coord2 = atoms.get_dihedral(*atom_indices[1])
            else:
                print(f"[ANALYSIS][ERROR] Unknown coord_type: {coord_type}")
                return

            coord1_list.append(coord1)
            coord2_list.append(coord2)
            energy_list.append(energy)

    if not coord1_list:
        print("[ANALYSIS][ERROR] No data found")
        return

    # Create grid
    coord1_array = np.array(coord1_list)
    coord2_array = np.array(coord2_list)
    energy_array = np.array(energy_list)

    coord1_grid = np.linspace(coord1_array.min(), coord1_array.max(), 100)
    coord2_grid = np.linspace(coord2_array.min(), coord2_array.max(), 100)
    coord1_mesh, coord2_mesh = np.meshgrid(coord1_grid, coord2_grid)

    # Interpolate
    from scipy.interpolate import griddata

    energy_mesh = griddata(
        (coord1_array, coord2_array),
        energy_array,
        (coord1_mesh, coord2_mesh),
        method="cubic",
    )

    # Create contour plot
    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=coord1_grid,
            y=coord2_grid,
            z=energy_mesh,
            colorscale="Viridis",
            colorbar=dict(title="Energy (eV)"),
            contours=dict(
                start=energy_array.min(),
                end=energy_array.max(),
                size=(energy_array.max() - energy_array.min()) / 20,
            ),
            hovertemplate="Coord1: %{x:.2f}<br>Coord2: %{y:.2f}<br>Energy: %{z:.2f} eV<extra></extra>",
        )
    )

    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=coord1_array,
            y=coord2_array,
            mode="markers",
            marker=dict(
                size=4, color="white", opacity=0.5, line=dict(color="black", width=0.5)
            ),
            name="Sampled points",
            hovertemplate="Coord1: %{x:.2f}<br>Coord2: %{y:.2f}<extra></extra>",
        )
    )

    # Labels
    if coord_type == "distance":
        xlabel = f"Distance {atom_indices[0]} (Å)"
        ylabel = f"Distance {atom_indices[1]} (Å)"
    elif coord_type == "angle":
        xlabel = f"Angle {atom_indices[0]} (°)"
        ylabel = f"Angle {atom_indices[1]} (°)"
    else:
        xlabel = f"Dihedral {atom_indices[0]} (°)"
        ylabel = f"Dihedral {atom_indices[1]} (°)"

    fig.update_layout(
        title="Potential Energy Surface",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=900,
        height=700,
        template="plotly_white",
        font=dict(size=14),
    )

    if save_fig:
        fig.write_html(save_fig)
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        fig.show()


########################################################################################################
# Plot Temperature
########################################################################################################


def PlotTemp(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    traj_filename="AseMD.traj",
    save_fig=None,
):
    """
    Plot temperature evolution from MD trajectories.

    Parameters
    ----------
    root_dir : str
        Root directory
    iteration_window : tuple or str
        (start, end) or "all"
    target_iteration : int
        Specific iteration
    traj_filename : str
        Trajectory filename
    save_fig : str
        Path to save HTML file
    """
    temp_dict = {}

    # Use main.py utility
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)

        # Use main.py loader
        traj = load_trajectory(iter_dir, subdir="02.dpmd", traj_filename=traj_filename)

        if traj is None:
            continue

        temperatures = []
        for atoms in traj:
            if atoms.get_kinetic_energy() > 0:
                temp = atoms.get_temperature()
                temperatures.append(temp)

        if temperatures:
            temp_dict[iter_num] = temperatures

    # Plotting
    fig = go.Figure()

    for iter_num, temps in sorted(temp_dict.items()):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(temps))),
                y=temps,
                mode="lines",
                name=f"Iter {iter_num}",
                line=dict(width=2),
                hovertemplate="Step: %{x}<br>Temp: %{y:.1f} K<extra></extra>",
            )
        )

    fig.update_layout(
        title="Temperature Evolution",
        xaxis_title="MD Steps",
        yaxis_title="Temperature (K)",
        width=1200,
        height=700,
        template="plotly_white",
        font=dict(size=18),
        hovermode="x unified",
    )

    if save_fig:
        fig.write_html(save_fig)
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        fig.show()


########################################################################################################
# END OF FILE
########################################################################################################

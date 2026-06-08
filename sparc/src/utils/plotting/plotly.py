"""
Plotting utilities for SPARC workflow analysis using Plotly.

Interactive, web-based plots with zoom, pan, and hover capabilities.
All plotting functions leverage shared utilities from main.py.
"""

import os

import dpdata
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .main import (
    compute_mae,
    compute_rmse,
    extract_iteration_number,
    get_iteration_dirs,
    load_trajectory,
)


def _white_bg_colorscale(cmap):
    """Return colorscale with white at position 0 so empty bins appear white."""
    cs = pc.get_colorscale(cmap)
    return [[0.0, "white"], [1e-9, cs[0][1]]] + cs[1:]


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
    force_mode : str
        'components' (default) — separate fx/fy/fz panels.
        'flatten' — all force components in one panel.
    heatmap : bool
        Color points by absolute error using cmap (default False)
    cmap : str
        Colormap for heatmap mode (default 'coolwarm')
    save_fig : str or None
        Path to save HTML file

    Example
    -------
    >>> ParityPlot("data_dir", "model.pth", type="all", force_mode="flatten", heatmap=True, cmap="plasma")
    >>> ParityPlot("data_dir", "model.pth", type="forces", force_mode="components")
    """
    if not os.path.exists(data_dir):
        print(f"[ANALYSIS][ERROR] Test data not found: {data_dir}")
        return
    if not os.path.isfile(model_path):
        print(f"[ANALYSIS][ERROR] Model file not found: {model_path}")
        return

    system = dpdata.LabeledSystem(data_dir, fmt="deepmd/npy")
    prediction = system.predict(dp=model_path)

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

    flatten = force_mode in ("flatten", "flat")

    # --- Determine subplot layout ---
    if type == "energy":
        fig = go.Figure()
        _add_parity_panel(
            fig,
            e_true,
            e_pred,
            heatmap,
            cmap,
            xlabel=f"Observed (DFT) [{e_unit}]",
            ylabel=f"Predicted (MLP) [{e_unit}]",
            title="(A) Energy",
            rmse_label=e_unit_ann,
            marker_size=8,
            opacity=0.7,
        )
        fig.update_layout(
            width=600,
            height=500,
            template="plotly_white",
            font=dict(size=14),
            hovermode="closest",
        )

    elif type == "forces":
        if flatten:
            fig = go.Figure()
            f_t_all, f_p_all = f_true.ravel(), f_pred.ravel()
            _add_parity_panel(
                fig,
                f_t_all,
                f_p_all,
                heatmap,
                cmap,
                xlabel="Observed (DFT) [eV/Å]",
                ylabel="Predicted (MLP) [eV/Å]",
                title="(A) Forces",
                rmse_label="meV/Å",
                marker_size=5,
                opacity=0.6,
            )
            fig.update_layout(
                width=600,
                height=500,
                template="plotly_white",
                font=dict(size=14),
                hovermode="closest",
            )
        else:
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=["(A) fx", "(B) fy", "(C) fz"],
                horizontal_spacing=0.08,
            )
            for i, comp in enumerate(["fx", "fy", "fz"]):
                _add_parity_panel(
                    fig,
                    f_true[:, i],
                    f_pred[:, i],
                    heatmap,
                    cmap,
                    xlabel="Observed (DFT) [eV/Å]",
                    ylabel="Predicted (MLP) [eV/Å]",
                    title=None,
                    rmse_label="meV/Å",
                    marker_size=5,
                    opacity=0.6,
                    row=1,
                    col=i + 1,
                    xref=f"x{i + 1 if i else ''}",
                    yref=f"y{i + 1 if i else ''}",
                )
            fig.update_layout(
                width=1300,
                height=450,
                template="plotly_white",
                font=dict(size=13),
                hovermode="closest",
            )

    else:  # 'all'
        if flatten:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["(A) Energy", "(B) Forces"],
                horizontal_spacing=0.12,
            )
            _add_parity_panel(
                fig,
                e_true,
                e_pred,
                heatmap,
                cmap,
                xlabel=f"Observed (DFT) [{e_unit}]",
                ylabel=f"Predicted (MLP) [{e_unit}]",
                title=None,
                rmse_label=e_unit_ann,
                marker_size=8,
                opacity=0.7,
                row=1,
                col=1,
                xref="x",
                yref="y",
            )
            f_t_all, f_p_all = f_true.ravel(), f_pred.ravel()
            _add_parity_panel(
                fig,
                f_t_all,
                f_p_all,
                heatmap,
                cmap,
                xlabel="Observed (DFT) [eV/Å]",
                ylabel="Predicted (MLP) [eV/Å]",
                title=None,
                rmse_label="meV/Å",
                marker_size=5,
                opacity=0.6,
                row=1,
                col=2,
                xref="x2",
                yref="y2",
            )
            fig.update_layout(
                width=1100,
                height=500,
                template="plotly_white",
                font=dict(size=14),
                hovermode="closest",
            )
        else:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=["(A) Energy", "(B) fx", "(C) fy", "(D) fz"],
                vertical_spacing=0.12,
                horizontal_spacing=0.10,
            )
            _add_parity_panel(
                fig,
                e_true,
                e_pred,
                heatmap,
                cmap,
                xlabel=f"Observed [{e_unit}]",
                ylabel=f"Predicted [{e_unit}]",
                title=None,
                rmse_label=e_unit_ann,
                marker_size=8,
                opacity=0.7,
                row=1,
                col=1,
                xref="x",
                yref="y",
            )
            positions = [(1, 2, "x2", "y2"), (2, 1, "x3", "y3"), (2, 2, "x4", "y4")]
            for i, (r, c, xr, yr) in enumerate(positions):
                _add_parity_panel(
                    fig,
                    f_true[:, i],
                    f_pred[:, i],
                    heatmap,
                    cmap,
                    xlabel="Observed [eV/Å]",
                    ylabel="Predicted [eV/Å]",
                    title=None,
                    rmse_label="meV/Å",
                    marker_size=5,
                    opacity=0.6,
                    row=r,
                    col=c,
                    xref=xr,
                    yref=yr,
                )
            fig.update_layout(
                width=1000,
                height=800,
                template="plotly_white",
                font=dict(size=13),
                hovermode="closest",
            )

    if save_fig:
        fig.write_html(save_fig)
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        fig.show()


def _add_parity_panel(
    fig,
    true_vals,
    pred_vals,
    heatmap,
    cmap,
    xlabel,
    ylabel,
    title,
    rmse_label,
    marker_size=6,
    opacity=0.7,
    row=None,
    col=None,
    xref="paper",
    yref="paper",
):
    """Add a single parity scatter + ideal line + RMSE annotation to fig."""
    rmse = compute_rmse(true_vals, pred_vals)
    mae = compute_mae(true_vals, pred_vals)
    ann_text = (
        f"RMSE = {rmse * 1000:.2f} {rmse_label}<br>MAE = {mae * 1000:.2f} {rmse_label}"
    )

    subplot_kw = dict(row=row, col=col) if row is not None else {}

    # Scatter
    if heatmap:
        abs_err = np.abs(pred_vals - true_vals)
        marker = dict(
            size=marker_size,
            color=abs_err,
            colorscale=cmap,
            opacity=opacity,
            showscale=True,
            colorbar=dict(title=f"|err| ({rmse_label})"),
            line=dict(width=0),
        )
    else:
        marker = dict(
            size=marker_size,
            color="blue",
            opacity=opacity,
            line=dict(color="black", width=0.5),
        )

    fig.add_trace(
        go.Scatter(
            x=true_vals,
            y=pred_vals,
            mode="markers",
            marker=marker,
            showlegend=False,
            hovertemplate="DFT: %{x:.4f}<br>MLP: %{y:.4f}<extra></extra>",
        ),
        **subplot_kw,
    )

    # Ideal line
    vmin, vmax = true_vals.min(), true_vals.max()
    fig.add_trace(
        go.Scatter(
            x=[vmin, vmax],
            y=[vmin, vmax],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            showlegend=False,
            hoverinfo="skip",
        ),
        **subplot_kw,
    )

    # Annotation — use paper coords for standalone, data coords for subplots
    if row is None:
        fig.add_annotation(
            text=ann_text,
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.95,
            showarrow=False,
            font=dict(size=13, color="blue"),
            align="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="blue",
            borderwidth=1,
        )
    else:
        fig.add_annotation(
            text=ann_text,
            xref=xref,
            yref=yref,
            x=vmin + 0.05 * (vmax - vmin),
            y=vmax - 0.08 * (vmax - vmin),
            showarrow=False,
            font=dict(size=11, color="blue"),
            align="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="blue",
            borderwidth=1,
        )

    if row is not None:
        fig.update_xaxes(title_text=xlabel, row=row, col=col)
        fig.update_yaxes(title_text=ylabel, row=row, col=col)
    else:
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title or "")


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
# Violin plot of force deviation error distribution across AL iterations
########################################################################################################


def PlotForceError(
    root_dir=".",
    iteration_window="all",
    target_iteration=None,
    dmin=None,
    dmax=0.5,
    connect_means=True,
    connect_medians=False,
    log_scale=False,
    palette="viridis",
    violin_alpha=0.85,
    save_fig=None,
):
    """
    Interactive violin plot of max force deviation distribution across AL iterations.

    Parameters
    ----------
    root_dir : str
        Root directory containing iter_* folders.
    iteration_window : tuple or str
        (start, end) or "all".
    target_iteration : int
        Specific iteration to analyze.
    dmin : float
        Threshold shown as dashed horizontal line (default: 0.05).
    dmax : float
        Upper y-axis limit for linear scale (default: 0.5).
    connect_means : bool
        Draw mean trajectory line across violins (default: True).
    connect_medians : bool
        Draw median trajectory line across violins (default: False).
    log_scale : bool
        Use log10 y-axis (default: False).
    palette : str
        Plotly colorscale name for violin colors (default: 'viridis').
    violin_alpha : float
        Opacity of violin bodies (default: 0.85).
    save_fig : str or None
        Path to save HTML file.

    Example
    -------
    >>> PlotForceError(iteration_window="all", palette="hot", log_scale=True)
    >>> PlotForceError(iteration_window=(0, 5), connect_means=True, dmin=0.05)
    """
    import glob

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
                    rows.append({"iter_i": iter_num, "max_devi_f": float(v)})
    df_violin = pd.DataFrame(rows)
    n_iter = sorted(df_violin["iter_i"].unique())

    colors = pc.sample_colorscale(
        palette, [i / max(len(n_iter) - 1, 1) for i in range(len(n_iter))]
    )

    fig = go.Figure()

    for idx, iter_num in enumerate(n_iter):
        vals = df_violin[df_violin["iter_i"] == iter_num]["max_devi_f"].values
        color = colors[idx]
        fig.add_trace(
            go.Violin(
                x=[f"Iter {iter_num}"] * len(vals),
                y=vals,
                name=f"Iter {iter_num}",
                fillcolor=color,
                opacity=violin_alpha,
                box_visible=True,
                meanline_visible=False,
                points=False,
                line_color=color,
            )
        )

    if connect_means:
        means = [
            df_violin[df_violin["iter_i"] == i]["max_devi_f"].mean() for i in n_iter
        ]
        fig.add_trace(
            go.Scatter(
                x=[f"Iter {i}" for i in n_iter],
                y=means,
                mode="lines+markers",
                name="Mean",
                line=dict(color="black", width=2.5),
                marker=dict(size=10, color="white", line=dict(color="black", width=2)),
            )
        )

    if connect_medians:
        meds = [
            df_violin[df_violin["iter_i"] == i]["max_devi_f"].median() for i in n_iter
        ]
        fig.add_trace(
            go.Scatter(
                x=[f"Iter {i}" for i in n_iter],
                y=meds,
                mode="lines+markers",
                name="Median",
                line=dict(color="gray", width=2, dash="dash"),
                marker=dict(size=8, color="white", line=dict(color="gray", width=1.5)),
            )
        )

    if dmin is not None:
        fig.add_hline(
            y=dmin,
            line_dash="dash",
            line_color="royalblue",
            line_width=1.5,
            annotation_text=f"dmin={dmin}",
        )

    yaxis_cfg = dict(type="log") if log_scale else dict(range=[0, dmax])

    fig.update_layout(
        title="Force Deviation Distribution",
        xaxis_title="AL Iterations",
        yaxis_title="Max. Force Deviation (eV/Å)",
        yaxis=yaxis_cfg,
        width=1000,
        height=500,
        template="plotly_white",
        font=dict(size=16),
        violinmode="overlay",
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

        energies = [
            float(np.asarray(atoms.get_potential_energy()).flat[0]) for atoms in traj
        ]

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
    get="energy",
    type="line",
    bins=25,
    save_fig=None,
):
    """
    Plot potential energy or bond distance from ASE trajectories across iter_* folders.

    Parameters
    ----------
    root_dir : str
        Root directory containing iter_* folders
    iteration_window : tuple or str
        (start, end) or "all"
    target_iteration : int
        Specific iteration number
    traj_filename : str
        Trajectory filename
    get : str
        "energy" or "distance:i,j" (e.g. "distance:0,7")
    type : str
        "line" (default), "hist", or "kde"
    bins : int
        Histogram bins for type="hist" (default 25)
    save_fig : str or None
        Path to save HTML file

    Example
    -------
    >>> PlotDistribution(get="energy", type="line")
    >>> PlotDistribution(get="energy", type="hist", bins=20)
    >>> PlotDistribution(get="distance:0,7", type="line")
    """
    is_energy = get.lower() == "energy"
    is_distance = get.lower().startswith("distance:")
    symbol_pair = None

    if is_distance:
        try:
            i, j = map(int, get.split(":")[1].split(","))
        except Exception:
            raise ValueError("For distance use format: 'distance:i,j'")
    elif not is_energy:
        raise ValueError("get must be 'energy' or 'distance:i,j'")

    property_dict = {}
    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)
        traj = load_trajectory(iter_dir, subdir="00.dft", traj_filename=traj_filename)

        if traj is None:
            continue

        if is_distance and symbol_pair is None and len(traj) > max(i, j):
            try:
                symbols = traj[0].get_chemical_symbols()
                symbol_pair = (symbols[i], symbols[j])
            except Exception:
                pass

        if is_energy:
            values = [
                float(np.asarray(atoms.get_potential_energy()).flat[0])
                for atoms in traj
            ]
        else:
            values = [float(atoms.get_distance(i, j)) for atoms in traj]

        property_dict[iter_num] = values

    if not property_dict:
        print("[PlotDistribution] No data found.")
        return

    # Labels
    if is_energy:
        data_label = "Potential Energy (eV)"
        hover_x, hover_unit = "Energy", "eV"
    else:
        if symbol_pair:
            sym_i, sym_j = symbol_pair
            data_label = f"Distance {sym_i}{i}-{sym_j}{j} (Å)"
        else:
            data_label = f"Bond Distance atoms {i}-{j} (Å)"
        hover_x, hover_unit = "Distance", "Å"

    all_vals = [v for vals in property_dict.values() for v in vals]

    fig = go.Figure()

    if type == "line":
        for iter_num, values in sorted(property_dict.items()):
            mode = "lines+markers" if len(values) > 1 else "markers"
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode=mode,
                    name=f"Iter {iter_num}",
                    marker=dict(size=8),
                    line=dict(width=2),
                    hovertemplate=f"Candidate: %{{x}}<br>{hover_x}: %{{y:.4f}} {hover_unit}<extra></extra>",
                )
            )
        xlabel, ylabel = "Candidates", data_label

    elif type == "hist":
        # Shared bin edges across all iterations so they're comparable
        vmin, vmax = min(all_vals), max(all_vals)
        bin_size = (vmax - vmin) / bins if vmax > vmin else 1.0
        for iter_num, values in sorted(property_dict.items()):
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=f"Iter {iter_num}",
                    opacity=0.7,
                    xbins=dict(start=vmin, end=vmax, size=bin_size),
                    marker=dict(line=dict(color="black", width=1)),
                    hovertemplate=f"{hover_x}: %{{x:.3f}} {hover_unit}<br>Count: %{{y}}<extra></extra>",
                )
            )
        xlabel, ylabel = data_label, "Count"

    elif type == "kde":
        # KDE via plotly density contour (1D: use violin instead)
        for iter_num, values in sorted(property_dict.items()):
            fig.add_trace(
                go.Violin(
                    x=values,
                    name=f"Iter {iter_num}",
                    box_visible=True,
                    meanline_visible=True,
                    opacity=0.7,
                    hovertemplate=f"{hover_x}: %{{x:.4f}} {hover_unit}<extra></extra>",
                )
            )
        xlabel, ylabel = data_label, "Density"

    else:
        raise ValueError("type must be 'line', 'hist', or 'kde'")

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=1000,
        height=600,
        template="plotly_white",
        font=dict(size=16),
        barmode="overlay",
        hovermode="x unified" if type == "line" else "closest",
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
    distance_pair=(0, 7),
    type="kde",
    bins=(50, 50),
    save_fig=None,
    atom_indices=None,
    coord_type="distance",
    **kwargs,
):
    """
    Plot energy vs bond distance as interactive 2D density (kde/heatmap/hexbin).

    Parameters
    ----------
    root_dir : str
        Root directory containing iter_* folders
    iteration_window : tuple or str
        (start, end) or "all"
    target_iteration : int
        Specific iteration number
    traj_filename : str
        Trajectory filename
    distance_pair : tuple
        (i, j) atom indices for bond distance (x-axis)
    type : str
        "kde" (density contour) or "heatmap" (2D histogram)
    bins : tuple
        (nbins_x, nbins_y) for heatmap
    save_fig : str or None
        Path to save HTML file
    atom_indices : list of tuples, optional
        Advanced: [(i1,j1), (i2,j2)] — use 2-coordinate contour mode instead
    coord_type : str
        Used only with atom_indices: "distance", "angle", or "dihedral"

    Example
    -------
    >>> PlotPES(distance_pair=(0, 7), type="heatmap")
    >>> PlotPES(distance_pair=(0, 7), type="kde", iteration_window=(0, 3))
    """
    # --- Advanced 2-coordinate contour mode ---
    if atom_indices is not None:
        if len(atom_indices) != 2:
            print("[ANALYSIS][ERROR] atom_indices must have exactly 2 entries")
            return
        return _PlotPES_2coord(
            root_dir,
            iteration_window,
            target_iteration,
            traj_filename,
            coord_type,
            atom_indices,
            save_fig,
        )

    # --- Primary mode: energy vs 1 bond distance ---
    i, j = distance_pair
    x_vals, y_vals = [], []
    total_frames = 0
    symbol_pair = None

    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    print("Parsing iterations:")
    for iter_dir in selected_dirs:
        iter_num = extract_iteration_number(iter_dir)
        traj = load_trajectory(iter_dir, subdir="00.dft", traj_filename=traj_filename)

        if traj is None:
            print(f" Iter {iter_num:>2}: MISSING ({traj_filename})")
            continue

        num_frames = len(traj)
        print(f" Iter {iter_num:>2}: {num_frames} frames")
        total_frames += num_frames

        if symbol_pair is None and num_frames > max(i, j):
            try:
                symbols = traj[0].get_chemical_symbols()
                symbol_pair = (symbols[i], symbols[j])
            except Exception:
                pass

        for atoms in traj:
            try:
                x_vals.append(float(atoms.get_distance(i, j)))
                y_vals.append(float(np.asarray(atoms.get_potential_energy()).flat[0]))
            except Exception as ex:
                print(f" Error: {ex}")

    print(f"\nTotal frames: {total_frames}")

    x = np.array(x_vals)
    y = np.array(y_vals)

    if symbol_pair:
        sym_i, sym_j = symbol_pair
        xlabel = f"Distance {sym_i}{i}-{sym_j}{j} (Å)"
    else:
        xlabel = f"Bond Distance atoms {i}-{j} (Å)"

    cmap = kwargs.pop("cmap", "Viridis")
    fig = go.Figure()

    if type == "kde":
        fig.add_trace(
            go.Histogram2dContour(
                x=x,
                y=y,
                colorscale=_white_bg_colorscale(cmap),
                reversescale=False,
                showscale=True,
                contours=dict(showlabels=False),
                hovertemplate="Distance: %{x:.3f} Å<br>Energy: %{y:.3f} eV<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=3, color="black", opacity=0.2),
                showlegend=False,
                hovertemplate="Distance: %{x:.3f} Å<br>Energy: %{y:.3f} eV<extra></extra>",
            )
        )
    elif type == "heatmap":
        nbx, nby = bins if isinstance(bins, (tuple, list)) else (bins, bins)
        fig.add_trace(
            go.Histogram2d(
                x=x,
                y=y,
                nbinsx=nbx,
                nbinsy=nby,
                colorscale=_white_bg_colorscale(cmap),
                zmin=1,
                colorbar=dict(title="Counts"),
                hovertemplate="Distance: %{x:.3f} Å<br>Energy: %{y:.3f} eV<br>Count: %{z}<extra></extra>",
            )
        )
    else:
        raise ValueError("type must be 'kde' or 'heatmap'")

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title="Potential Energy (eV)",
        width=900,
        height=700,
        template="plotly_white",
        plot_bgcolor="white",
        font=dict(size=16),
    )

    if save_fig:
        fig.write_html(save_fig)
        print(f"[ANALYSIS][INFO] Saved to: {save_fig}")
    else:
        fig.show()


def _PlotPES_2coord(
    root_dir,
    iteration_window,
    target_iteration,
    traj_filename,
    coord_type,
    atom_indices,
    save_fig,
):
    """2-coordinate contour PES (energy vs 2 structural coords)."""
    coord1_list, coord2_list, energy_list = [], [], []

    selected_dirs = get_iteration_dirs(root_dir, iteration_window, target_iteration)

    for iter_dir in selected_dirs:
        traj = load_trajectory(iter_dir, subdir="00.dft", traj_filename=traj_filename)
        if traj is None:
            continue
        for atoms in traj:
            energy = float(np.asarray(atoms.get_potential_energy()).flat[0])
            if coord_type == "distance":
                c1 = atoms.get_distance(*atom_indices[0])
                c2 = atoms.get_distance(*atom_indices[1])
            elif coord_type == "angle":
                c1 = atoms.get_angle(*atom_indices[0])
                c2 = atoms.get_angle(*atom_indices[1])
            elif coord_type == "dihedral":
                c1 = atoms.get_dihedral(*atom_indices[0])
                c2 = atoms.get_dihedral(*atom_indices[1])
            else:
                print(f"[ANALYSIS][ERROR] Unknown coord_type: {coord_type}")
                return
            coord1_list.append(c1)
            coord2_list.append(c2)
            energy_list.append(energy)

    if not coord1_list:
        print("[ANALYSIS][ERROR] No data found")
        return

    from scipy.interpolate import griddata

    c1 = np.array(coord1_list)
    c2 = np.array(coord2_list)
    e = np.array(energy_list)
    g1 = np.linspace(c1.min(), c1.max(), 100)
    g2 = np.linspace(c2.min(), c2.max(), 100)
    m1, m2 = np.meshgrid(g1, g2)
    em = griddata((c1, c2), e, (m1, m2), method="cubic")

    unit = "Å" if coord_type == "distance" else "°"
    xlabel = f"{coord_type.capitalize()} {atom_indices[0]} ({unit})"
    ylabel = f"{coord_type.capitalize()} {atom_indices[1]} ({unit})"

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=g1,
            y=g2,
            z=em,
            colorscale="Viridis",
            colorbar=dict(title="Energy (eV)"),
            hovertemplate="Coord1: %{x:.2f}<br>Coord2: %{y:.2f}<br>Energy: %{z:.2f} eV<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=c1,
            y=c2,
            mode="markers",
            marker=dict(
                size=4, color="white", opacity=0.5, line=dict(color="black", width=0.5)
            ),
            name="Sampled points",
        )
    )
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

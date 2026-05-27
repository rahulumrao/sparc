"""
Common utilities for plotting functions.
Shared logic for iteration selection, data loading, and helper functions.
"""

import glob
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ase.io import read
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde


def get_iteration_dirs(
    root_dir: str,
    iteration_window: Union[str, Tuple[int, int]] = "all",
    target_iteration: Optional[int] = None,
) -> List[str]:
    """
    Get list of iteration directories based on selection criteria.

    Parameters
    ----------
    root_dir : str
        Root directory containing iter_* folders
    iteration_window : str or tuple
        "all" for all iterations, or (start, end) tuple
    target_iteration : int, optional
        Specific iteration number to select

    Returns
    -------
    list of str
        Sorted list of selected iteration directory paths
    """
    iter_dirs = sorted(glob.glob(os.path.join(root_dir, "iter_*")))

    if target_iteration is not None:
        return [d for d in iter_dirs if int(d.split("_")[-1]) == target_iteration]

    if iteration_window == "all":
        return iter_dirs

    if isinstance(iteration_window, tuple):
        start, end = iteration_window
        return [d for d in iter_dirs if start <= int(d.split("_")[-1]) <= end]

    return []


def load_trajectory(
    iter_dir: str, subdir: str = "00.dft", traj_filename: str = "AseMD.traj"
) -> Optional[List]:
    """
    Load trajectory from iteration directory.

    Parameters
    ----------
    iter_dir : str
        Iteration directory path
    subdir : str
        Subdirectory name (e.g., "00.dft", "02.dpmd")
    traj_filename : str
        Trajectory filename

    Returns
    -------
    list of Atoms or None
        Trajectory frames, or None if file not found
    """
    traj_path = os.path.join(iter_dir, subdir, traj_filename)
    if not os.path.isfile(traj_path):
        return None
    try:
        return read(traj_path, index=":")
    except Exception as e:
        print(f" [ERROR] Failed to load {traj_path}: {e}")
        return None


def extract_iteration_number(iter_dir: str) -> int:
    """Extract iteration number from directory name."""
    return int(iter_dir.split("_")[-1])


def compute_rmse(true, pred) -> float:
    """Compute root mean square error."""
    return np.sqrt(np.mean((true - pred) ** 2))


def compute_mae(true, pred) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(true - pred))


########################################################################################################
# Helper Functions (Moved from plotting backends)
########################################################################################################


def ReadColvar(file_path="COLVAR"):
    """
    Read PLUMED COLVAR file.

    Parameters
    ----------
    file_path : str
        Path to COLVAR file

    Returns
    -------
    pd.DataFrame
        DataFrame containing COLVAR data
    """
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        column_names = (
            first_line.split()[2:] if first_line.startswith("#! FIELDS") else None
        )
    return pd.read_csv(file_path, sep="\\s+", comment="#", names=column_names)


def get_2dSurface(traj, bonds, T=300):
    """
    Compute 2D free energy surface from trajectory.

    Parameters
    ----------
    traj : ASE trajectory
        Trajectory object
    bonds : list of tuples
        [(a1, b1), (a2, b2)] atom indices for two bonds
    T : float
        Temperature in Kelvin

    Returns
    -------
    R1, R2, F : 2D arrays
        Bond distances and free energy surface
    """
    # Extract bond distances
    b_1 = np.array([atoms.get_distance(*bonds[0]) for atoms in traj])
    b_2 = np.array([atoms.get_distance(*bonds[1]) for atoms in traj])
    bond_data = np.vstack([b_1, b_2])
    kde = gaussian_kde(bond_data)

    # Define grid
    r1_range = np.linspace(min(b_1), max(b_1), 150)
    r2_range = np.linspace(min(b_2), max(b_2), 150)
    R1, R2 = np.meshgrid(r1_range, r2_range)

    # Evaluate KDE
    P = kde(np.vstack([R1.ravel(), R2.ravel()])).reshape(R1.shape)
    P /= np.sum(P)

    # Compute free energy
    kB = 0.010364  # Boltzmann constant in eV/K
    F = -kB * T * np.log(P)
    F -= np.min(F)
    F[P < 1e-10] = np.nan

    return R1, R2, F


def get_1dSurface(traj, bond):
    """
    Get 1D energy profile along a bond distance.

    Parameters
    ----------
    traj : ASE trajectory
        Trajectory object
    bond : tuple
        (a, b) atom indices

    Returns
    -------
    r_range, F, bond_lengths, energies
        Interpolated and raw data
    """
    bond_lengths = np.array([atoms.get_distance(*bond) for atoms in traj])
    energies = np.array([atoms.get_potential_energy() for atoms in traj])

    # Sort for interpolation
    sorted_indices = np.argsort(bond_lengths)
    bond_lengths = bond_lengths[sorted_indices]
    energies = energies[sorted_indices]

    # Interpolate
    interp_func = interp1d(
        bond_lengths, energies, kind="linear", fill_value="extrapolate"
    )
    r_range = np.linspace(min(bond_lengths), max(bond_lengths), 200)
    F = interp_func(r_range)
    F -= np.nanmin(F)

    return r_range, F, bond_lengths, energies


def ViewTraj(traj, style="ball_and_stick", background="white", size=400):
    """
    Interactive trajectory viewer with nglview.

    Parameters
    ----------
    traj : ASE trajectory or str
        Trajectory object or file path
    style : str
        Representation style
    background : str
        Background color
    size : int
        Viewer size in pixels

    Returns
    -------
    nglview.NGLWidget
        Configured viewer widget
    """
    import nglview as nv

    if isinstance(traj, str):
        traj = read(traj, index=":")

    view = nv.NGLWidget(nv.ASETrajectory(traj))
    view.clear_representations()

    if style == "ball_and_stick":
        view.add_ball_and_stick()
    elif style == "spacefill":
        view.add_spacefill()
    elif style == "licorice":
        view.add_licorice()
    else:
        view.add_ball_and_stick()

    view.add_label(
        selection="all",
        label_type="atomindex",
        color="black",
        zOffset=1.0,
        attachment="middle-center",
    )
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
        "autoRotate": False,
    }

    return view


########################################################################################################
# END OF FILE
########################################################################################################

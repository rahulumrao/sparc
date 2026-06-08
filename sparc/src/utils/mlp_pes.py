#!/usr/bin/python
# mlp_pes.py
"""
Module for comparing DFT and ML-predicted potential energies and forces along
a bond distance in a trajectory.

Outputs a compressed NPZ file with distances, DFT and ML energies, and forces
keyed by iteration directory name.

Example usage (from within SPARC CLI):
  >>> sparc --analysis get_energies --dft_file OUTCAR --bond 0 7 --out energy_forces.npz
"""

import os

import numpy as np
from ase import Atoms
from ase.io import read

try:
    from deepmd.tf.calculator import DP
except ImportError:
    from deepmd.calculator import DP

from sparc.src.utils.logger import SparcLog


# --------------------------------------------------------------------------------------
# Extract DFT energy, forces, and bond distance for one frame
# --------------------------------------------------------------------------------------
def dft_energy_forces(frame, bond):
    """
    Compute DFT bond distance, total potential energy, and forces for a frame.

    Parameters:
    ------------
    frame : ase.Atoms
    bond : list of int
        Pair of atom indices for bond distance.

    Returns:
    --------
    float, float, np.ndarray : distance (Å), energy (eV), forces (N×3, eV/Å)
    """
    d = frame.get_distance(bond[0], bond[1])
    e = float(np.asarray(frame.get_potential_energy()).flat[0])
    f = frame.get_forces()
    return d, e, f


# --------------------------------------------------------------------------------------
# Extract ML energy and forces for one frame using a pre-built calculator
# --------------------------------------------------------------------------------------
def ml_energy_forces(frame, calc):
    """
    Compute ML-predicted energy and forces for a frame.

    Parameters:
    ------------
    frame : ase.Atoms
    calc : DeepMD DP calculator (pre-initialized)

    Returns:
    --------
    float, np.ndarray : energy (eV), forces (N×3, eV/Å)
    """
    atoms = Atoms(frame, calculator=calc)
    e = float(np.asarray(atoms.get_potential_energy()).flat[0])
    f = atoms.get_forces()
    return e, f


# --------------------------------------------------------------------------------------
# Prompt for iteration numbers or use all
# --------------------------------------------------------------------------------------
def get_selected_iters(all_iter_dirs):
    """
    Prompt user to select specific iteration folders from available ones.

    Parameters:
    ------------
    all_iter_dirs : list
        List of all iteration directory names.

    Returns:
    --------
    list : Selected iteration directory names
    """
    SparcLog("\nAvailable iteration folders:", level="INFO")
    for i, name in enumerate(all_iter_dirs):
        SparcLog(f"[{i}] {name}", level="INFO")

    inp = input(
        "\nEnter space-separated iteration numbers (press Enter for all): "
    ).strip()

    if inp:
        try:
            selected_indices = [int(i) for i in inp.split()]
            return [all_iter_dirs[i] for i in selected_indices]
        except Exception as e:
            SparcLog(f"Invalid input: {e}", level="ERROR")
            exit(1)

    return all_iter_dirs


# --------------------------------------------------------------------------------------
# Main Function: Extract DFT and ML Energies + Forces → NPZ
# --------------------------------------------------------------------------------------
def get_energies(
    dft_file,
    ifmt="vasp-out",
    skip=1,
    model="training_1/frozen_model_1.pth",
    bond=None,
    out="energy_forces.npz",
):
    """
    Extract DFT and ML energies and forces for each frame, save to NPZ.

    Parameters:
    ------------
    dft_file : str
        Path to DFT output trajectory (e.g., OUTCAR).
    ifmt : str
        ASE-compatible format string (default: 'vasp-out').
    skip : int
        Report progress every n frames (default: 1).
    model : str
        Model path relative to each iter_*/01.train/ folder.
        Use ``{i}`` as placeholder for the integer in the iter_* name
        (default: 'training_1/frozen_model_1.pth').
        Example: iter_000003 → iter_000003/01.train/training_3/model_3.pth
    bond : list of two ints
        Atom indices for bond distance (default: [0, 1]).
    out : str
        Output NPZ filename (default: 'energy_forces.npz').

    NPZ keys
    --------
    dist        : (N,)     bond distances
    E_dft       : (N,)     DFT energies
    F_dft       : (N,M,3)  DFT forces
    bond        : (2,)     bond atom indices
    symbols     : (M,)     chemical symbols
    iter_dirs   : (K,)     selected iteration directory names
    E_<iter>    : (N,)     ML energy for each iteration
    F_<iter>    : (N,M,3)  ML forces for each iteration

    Example:
    --------
    >>> get_energies("OUTCAR", ifmt="vasp-out", bond=[0, 7], out="energy_forces.npz")
    """
    if bond is None:
        bond = [0, 1]

    # Step 1: Read trajectory
    traj = read(dft_file, index=":", format=ifmt)
    SparcLog("*" * 70)
    SparcLog(f"Total Frames  : {len(traj)}")
    SparcLog(f"Bond indices  : {bond[0]} — {bond[1]}")
    SparcLog(f"Model pattern : 01.train/{model}")
    SparcLog(f"Output file   : {out}")
    SparcLog("*" * 70)

    # Step 2: Discover and select iteration directories
    all_iter_dirs = sorted(
        [d for d in os.listdir() if d.startswith("iter_") and os.path.isdir(d)]
    )
    if not all_iter_dirs:
        SparcLog("No iter_* directories found.", level="ERROR")
        return
    iter_dirs = get_selected_iters(all_iter_dirs)

    # Step 3: Build one DP calculator per iteration (model loaded once, not per frame)
    calcs = {}
    for it in iter_dirs:
        # extract integer from iter_* name (e.g. iter_3 → 3)
        try:
            it_num = int(it.rsplit("_", 1)[-1])
        except ValueError:
            it_num = iter_dirs.index(it)
        mp = os.path.join(it, "01.train", model.format(i=it_num))
        if not os.path.isfile(mp):
            SparcLog(f"Missing model: {mp}", level="ERROR")
            ans = input(f"  Skip '{it}' and continue? [y/N]: ").strip().lower()
            if ans != "y":
                SparcLog("Aborted.", level="ERROR")
                return
            continue
        calcs[it] = DP(mp)
        SparcLog(f"Loaded: {mp}", level="INFO")

    # only process iters that have a loaded calculator
    active_iters = list(calcs.keys())
    if not active_iters:
        SparcLog("No models loaded. Exiting.", level="ERROR")
        return

    # Step 4: Stream through trajectory
    dists, E_dft, F_dft = [], [], []
    E_ml = {it: [] for it in active_iters}
    F_ml = {it: [] for it in active_iters}

    for i, frame in enumerate(traj):
        try:
            d, e_dft, f_dft = dft_energy_forces(frame, bond)
        except Exception as e:
            SparcLog(f"Skipping frame {i} (DFT failed): {e}", level="WARNING")
            continue

        dists.append(d)
        E_dft.append(e_dft)
        F_dft.append(f_dft)

        for it in active_iters:
            try:
                e_ml, f_ml = ml_energy_forces(frame, calcs[it])
            except Exception as e:
                SparcLog(f"Frame {i}, {it} ML failed: {e}", level="WARNING")
                e_ml = np.nan
                f_ml = np.full((len(frame), 3), np.nan)
            E_ml[it].append(e_ml)
            F_ml[it].append(f_ml)

        if (i + 1) % max(skip, 1) == 0:
            SparcLog(f"Processed {i + 1}/{len(traj)} frames...", level="INFO")

    # Step 5: Pack and save
    out_dict = {
        "dist": np.asarray(dists),
        "E_dft": np.asarray(E_dft),
        "F_dft": np.asarray(F_dft),
        "bond": np.asarray(bond, dtype=int),
        "symbols": np.asarray(traj[0].get_chemical_symbols()),
        "iter_dirs": np.asarray(active_iters),
    }
    for it in active_iters:
        out_dict[f"E_{it}"] = np.asarray(E_ml[it])
        out_dict[f"F_{it}"] = np.asarray(F_ml[it])

    np.savez_compressed(out, **out_dict)
    SparcLog(f"Saved: {out}", level="INFO")


# --------------------------------------------------------------------------------------
# End of File
# --------------------------------------------------------------------------------------

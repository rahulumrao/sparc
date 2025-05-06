#!/usr/bin/python
# mlp_pes.py
"""
Module for comparing DFT and ML-predicted potential energies along a bond distance in a trajectory.
This script supports selection of DeepMD model directories and outputs a CSV file of bond distances
along with corresponding DFT and ML energies.

Example usage (from within SPARC CLI):
  >>> sparc --analysis compare_energy --dft_file OUTCAR --bond 0 7 --out energy.csv
"""

import os
import sys
import argparse
import time
import numpy as np

# Third party import
from ase.io import read
from ase import Atoms
from deepmd.calculator import DP

# Local import
from sparc.src.utils.logger import SparcLog
#--------------------------------------------------------------------------------------
# Extract DFT Energy for one frame
#--------------------------------------------------------------------------------------
def dft_energy_single(frame, bond):
    """
    Compute DFT bond distance and total potential energy for a given frame.

    Parameters:
    ------------
    frame : ase.Atoms
        Frame from ASE trajectory.
    bond : tuple of int
        Pair of atom indices to compute bond distance.

    Returns:
    --------
    float, float : bond distance (Å), total energy (eV)
    """
    d = frame.get_distance(bond[0], bond[1])
    e = frame.get_potential_energy()
    return d, e

#--------------------------------------------------------------------------------------
# Setup ML Calculator
#--------------------------------------------------------------------------------------
def dpmd_calculator(atom, model):
    """
    Attach DeepMD calculator to ASE Atoms object.

    Parameters:
    ------------
    atom : ase.Atoms
    model : str
        Path to frozen_model.pb

    Returns:
    --------
    ase.Atoms with calculator attached
    """
    return Atoms(atom, calculator=DP(model))

#--------------------------------------------------------------------------------------
# Extract ML Energy for one frame and one model
#--------------------------------------------------------------------------------------
def ml_energy_single(frame, model_path):
    """
    Compute ML-predicted potential energy for a frame using a DeepMD model.

    Parameters:
    ------------
    frame : ase.Atoms
    model_path : str
        Path to frozen_model.pb

    Returns:
    --------
    float : predicted energy (eV)
    """
    atoms = dpmd_calculator(atom=frame, model=model_path)
    e = atoms.get_potential_energy()
    return e

#--------------------------------------------------------------------------------------
# Prompt for iteration numbers or use all
#--------------------------------------------------------------------------------------
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
    SparcLog("\nAvailable iteration folders:", origin='ANALYSIS', level='INFO')
    for i, name in enumerate(all_iter_dirs):
        SparcLog(f"[{i}] {name}", origin='ANALYSIS', level='INFO')

    inp = input("\nEnter space-separated iteration numbers to compute (press Enter for all): ").strip()

    if inp:
        try:
            selected_indices = [int(i) for i in inp.split()]
            return [all_iter_dirs[i] for i in selected_indices]
        except Exception as e:
            SparcLog(f"Invalid input. Error: {e}", origin='ANALYSIS', level='ERROR')
            exit(1)
    else:
        return all_iter_dirs
# ------------------------------------------------------------------------------
# Main Function: Compare DFT and ML Energies
# ------------------------------------------------------------------------------
def get_energies(dft_file, ifmt, skip, bond, out):
    """
    Extract DFT and ML energies for each frame and write to CSV.

    Parameters:
    ------------
    dft_file : str
        Path to DFT output trajectory (e.g., OUTCAR or traj.xyz)
    ifmt : str
        Format string readable by ASE (e.g., 'vasp-out')
    skip : int
        Skip every n-th frame
    bond : list of two ints
        Atom indices for bond distance
    out : str
        Output filename (CSV)

    Example usage:
    --------------
    >>> get_energies("OUTCAR", "vasp-out", 1, [0, 7], "energy.csv")
    """
    traj = read(dft_file, index=f'::{skip}', format=ifmt)
    SparcLog('*' * 70, origin='ANALYSIS')
    SparcLog(f"Total Frames: {len(traj)}", origin='ANALYSIS')
    SparcLog(f"Calculating distance for bond between atom: {bond[0]}, and atom: {bond[1]}", origin='ANALYSIS')
    SparcLog('*' * 70, origin='ANALYSIS')
    time.sleep(2)

    all_iter_dirs = sorted([d for d in os.listdir() if d.startswith("iter_")])
    iter_dirs = get_selected_iters(all_iter_dirs)

    col_width = 15

    with open(out, "w") as f:
        header = f"{'Dist.':<{col_width}} {'E(DFT)':<{col_width}}" + "".join(
            f"{f'E({iter_dir})':<{col_width}}" for iter_dir in iter_dirs
        )
        f.write(header + "\n")

        for i, frame in enumerate(traj):
            d, e_dft = dft_energy_single(frame, bond)
            energies = []

            for iter_dir in iter_dirs:
                model_path = os.path.join(iter_dir, "01.train/training_1/frozen_model_1.pb")
                e_ml = ml_energy_single(frame, model_path)
                energies.append(e_ml)

            line = f"{d:<{col_width}.6f}{e_dft:<{col_width}.6f}" + "".join(
                f"{e:<{col_width}.6f}" for e in energies
            )
            f.write(line + "\n")
            f.flush()

    SparcLog(f"Energy data saved to {out}", origin='ANALYSIS')
#--------------------------------------------------------------------------------------
# End of File
#--------------------------------------------------------------------------------------

#!/usr/bin/python3

# utils.py

################################################################

import glob
import json
import os
import pickle
from pathlib import Path
from typing import List, Union

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

################################################################
# Third party import
from ase.io import read, write
from ase.io.trajectory import TrajectoryWriter

################################################################
# Local Import
from sparc.src.utils.logger import SparcLog

# ===================================================================================================
# Backend Mismatch Check
# ===================================================================================================


def check_backend_mismatch(model_path: str, backend: str):
    """
    Validate that the model file format matches the installed DeePMD-kit backend.

    Parameters
    ----------
    model_path : str
        Path to the frozen model file (.pth or .pb)
    backend : str
        Installed backend ('pytorch' or 'tensorflow')

    Raises
    ------
    RuntimeError
        If the model format does not match the installed backend
    """
    model_ext = Path(model_path).suffix.lower()

    if model_ext == ".pth" and backend != "pytorch":
        SparcLog("=" * 80)
        SparcLog("ERROR: Backend mismatch detected!", level="ERROR")
        SparcLog(f"  Model file : {model_path} (PyTorch format)", level="ERROR")
        SparcLog(f"  Backend    : {backend.upper()} (installed)", level="ERROR")
        SparcLog("", level="ERROR")
        SparcLog(
            "  The model was trained with PyTorch but the current environment",
            level="ERROR",
        )
        SparcLog("  has TensorFlow backend. To fix this either:", level="ERROR")
        SparcLog(
            "    1. Install SPARC with DeePMD-GNN (PyTorch):  pip install deepmd-kit[torch]",
            level="ERROR",
        )
        SparcLog(
            "    2. Retrain models in the current TensorFlow environment", level="ERROR"
        )
        SparcLog("=" * 80)
        raise RuntimeError(
            f"Backend mismatch: model '{Path(model_path).name}' is PyTorch (.pth) "
            f"but installed backend is {backend.upper()}. "
            f"Install PyTorch backend: pip install deepmd-kit[torch]"
        )
    elif model_ext == ".pb" and backend == "pytorch":
        SparcLog("=" * 80)
        SparcLog("ERROR: Backend mismatch detected!", level="ERROR")
        SparcLog(f"  Model file : {model_path} (TensorFlow format)", level="ERROR")
        SparcLog(f"  Backend    : {backend.upper()} (installed)", level="ERROR")
        SparcLog("", level="ERROR")
        SparcLog(
            "  The model was trained with TensorFlow but the current environment",
            level="ERROR",
        )
        SparcLog("  has PyTorch backend. To fix this either:", level="ERROR")
        SparcLog(
            "    1. Install SPARC with TensorFlow:  pip install deepmd-kit[tf]",
            level="ERROR",
        )
        SparcLog(
            "    2. Retrain models in the current PyTorch environment", level="ERROR"
        )
        SparcLog("=" * 80)
        raise RuntimeError(
            f"Backend mismatch: model '{Path(model_path).name}' is TensorFlow (.pb) "
            f"but installed backend is {backend.upper()}. "
            f"Install TensorFlow backend: pip install deepmd-kit[tf]"
        )


# ===================================================================================================
"""
Function is called to log the dynamics. It prints the potential energy (Epot), kinetic energy (Ekin),
total energy (Epot + Ekin), and temperature (Temp) of the system to a file.
"""


# ---------------------------------------------------------------------------------------------------#
def create_iteration_dirs(iter_num):
    """Create iteration directory structure."""
    iter_name = f"iter_{iter_num:06d}"
    iter_dir = Path(iter_name)
    iter_dir.mkdir(exist_ok=True)

    # Create subdirectories with new naming
    dft_dir = iter_dir / "00.dft"  # DFT calculations (VASP)
    train_dir = iter_dir / "01.train"  # DeepMD training
    dpmd_dir = iter_dir / "02.dpmd"  # DeepMD runs and model deviation

    # Print iteration information
    # SparcLog("="*72)
    SparcLog(f"Creating Directories for Iteration: {iter_num:06d}")
    SparcLog("=" * 80)
    SparcLog(f"├── {iter_name}")
    SparcLog(f"│   ├── {dft_dir.name}")
    SparcLog(f"│   ├── {train_dir.name}")
    SparcLog(f"│   └── {dpmd_dir.name}")
    # SparcLog("="*72 + "\n")

    for folder in [dft_dir, train_dir, dpmd_dir]:
        folder.mkdir(exist_ok=True)

    return {
        "iter_num": iter_num,
        "iter_dir": iter_dir,
        "dft_dir": dft_dir,
        "train_dir": train_dir,
        "dpmd_dir": dpmd_dir,
    }


# ---------------------------------------------------------------------------------------------------#
def log_md_setup(dyn, atoms, dir_name, write_dist=False):
    """
    Log molecular dynamics simulation details including energies, temperature, and NPT properties.

    Automatically detects NPT ensemble by checking if stress tensor is available.

    Args:
        dyn: ASE dynamics object
            The molecular dynamics simulation object
        atoms: ase.Atoms
            The atomic system being simulated
        dir_name: str
            Directory path for log files
        write_dist: bool, optional
            Whether to log distance between atoms 0 and 4 (default: False)
    """
    # Get energies and ensure they're scalar values
    epot = atoms.get_potential_energy()
    if isinstance(epot, (list, np.ndarray)):
        epot = epot[0]

    ekin = atoms.get_kinetic_energy()
    total = epot + ekin

    # Get other system properties
    step = dyn.get_number_of_steps()
    temp = float(atoms.get_temperature())

    # Auto-detect NPT by checking if stress is available
    is_npt = False
    pressure_gpa = None
    volume = None

    try:
        # Try to get stress tensor (only available in NPT)
        stress = atoms.get_stress(voigt=False)  # 3x3 stress tensor in eV/A^3
        # Pressure is negative trace of stress tensor / 3
        # Convert from eV/A^3 to GPa: 1 eV/A^3 = 160.21766208 GPa
        pressure_gpa = -np.trace(stress) / 3.0 * 160.21766208
        volume = atoms.get_volume()
        is_npt = True
    except (RuntimeError, NotImplementedError, AttributeError):
        # Stress not available - standard NVT/NVE ensemble
        pass

    # Console output (formatted as table row)
    if is_npt:
        SparcLog(
            f"{step:<8} {epot:>11.4f} {ekin:>11.4f} {temp:>9.2f} {pressure_gpa:>9.4f} {volume:>9.2f}"
        )
    else:
        SparcLog(f"{step:<8} {epot:>11.4f} {ekin:>11.4f} {temp:>9.2f}")

    # File logging (always write to file)
    if is_npt:
        with MDLogger(f"{dir_name}/AseMolDyn.log") as log:
            if step == 0:
                log.file.write(
                    f"# {'Step':<6} {'Epot':<10} {'Ekin':<10} {'Total':<10} "
                    f"{'Temp':<6} {'Pressure(GPa)':<14} {'Volume(Å³)':<12}\n"
                )
            log.file.write(
                f"{float(step):<8} {epot:<10.6f} {ekin:<10.6f} {total:<10.6f} "
                f"{temp:<6.2f} {pressure_gpa:<14.6f} {volume:<12.4f}\n"
            )
    else:
        with MDLogger(f"{dir_name}/AseMolDyn.log") as log:
            if step == 0:
                log.file.write(
                    f"# {'Step':<6} {'Epot':<10} {'Ekin':<10} {'Total':<10} {'Temp':<6}\n"
                )
            log.file.write(
                f"{float(step):<8} {epot:<10.6f} {ekin:<10.6f} {total:<10.6f} {temp:<6.2f}\n"
            )

    # Optionally log atomic distances
    if write_dist:
        distance = atoms.get_distance(0, 4, mic=True)
        with MDLogger(f"{dir_name}/dist.dat") as dist_log:
            dist_log.file.write(f"Step: {step}, Distance: {distance:.6f}\n")


# ---------------------------------------------------------------------------------------------------#
def save_xyz(atoms, trajfile, write_mode, dir_name):
    """
    Save atomic configuration to ASE trajectory and XYZ files.

    Works on a shallow copy of atoms so the original object (and its
    calculator cache) is never modified.  This prevents PLUMED (or any
    wrapped calculator) from being re-invoked: center()/wrap() change
    atom positions, which would otherwise invalidate ASE's force cache
    and cause the live calculator to recompute — advancing PLUMED's
    internal step counter and writing duplicate rows to COLVAR/cv_force.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic system to save. Must have a calculator with computed
        energy and forces (i.e. called after a dynamics step).
    trajfile : str
        Filename for the ASE binary trajectory (written inside dir_name).
    write_mode : str
        Trajectory write mode — 'w' to overwrite, 'a' to append.
    dir_name : str or Path
        Directory where trajectory and XYZ files are written.
        The XYZ file is always written as AseTraj.xyz in this directory.
    """
    properties = ["energy", "forces", "coordinates", "cell", "pbc"]

    # Read from the live calculator's cache — positions are unchanged at
    # this point so no re-computation (and no PLUMED call) occurs.
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # Operate on a copy so the original atoms object is never mutated.
    # center()/wrap() on the copy cannot invalidate the original's cache.
    write_atoms = atoms.copy()
    # pretty_translation=True wraps whole connected molecules as one unit
    # before centering, preventing split-molecule PBC artifacts where bonded
    # atoms land on opposite sides of the cell in raw XYZ/traj coordinates.
    # Must come before center() so CoM is computed on a whole molecule.
    write_atoms.wrap(pretty_translation=True)
    write_atoms.center()
    write_atoms.calc = SinglePointCalculator(write_atoms, energy=energy, forces=forces)

    traj_file = f"{dir_name}/{trajfile}"
    xyz_file = f"{dir_name}/AseTraj.xyz"

    trr = TrajectoryWriter(
        filename=traj_file, mode=write_mode, atoms=write_atoms, properties=properties
    )
    trr.write(write_atoms)

    # Strip momenta only for the xyz text file — binary traj retains them
    # so temperature is recoverable from dpmd.traj / AseMD.traj.
    write_atoms.set_momenta(None)
    write(xyz_file, write_atoms, append=True)


# ---------------------------------------------------------------------------------------------------#
# Add context managers for file handling
class MDLogger:
    """
    Context manager for handling MD log files.

    Args:
        filename: str
            Path to the log file
    """

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename, "a")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


# ---------------------------------------------------------------------------------------------------#
def save_checkpoint(dyn, atoms, filename="md_checkpoint.pkl"):
    """
    Save molecular dynamics checkpoint to resume later.

    Args:
        dyn: ASE dynamics object
            The dynamics object containing simulation state
        atoms: ASE atoms object
            The atoms object containing atomic positions, velocities etc.
        filename: str
            Checkpoint filename to save state (default: 'md_checkpoint.pkl')
    """
    positions = atoms.get_positions()
    velocities = atoms.get_velocities()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    numbers = atoms.get_atomic_numbers()
    momenta = dyn.atoms.get_momenta()
    step = dyn.get_number_of_steps()

    # Only for NPT - try to get stress if available
    stress = None
    try:
        stress = atoms.get_stress(voigt=False)
    except (RuntimeError, NotImplementedError, AttributeError):
        pass

    state = {
        "positions": positions,
        "velocities": velocities,
        "cell": cell,
        "pbc": pbc,
        "numbers": numbers,
        "step": step,
        "momenta": momenta,
        "stress": stress,
    }

    with open(filename, "wb") as f:
        pickle.dump(state, f)


def load_checkpoint(atoms, filename="md_checkpoint.pkl"):
    """
    Load molecular dynamics checkpoint.

    Args:
        atoms: ASE atoms object
        filename: str, checkpoint filename (default: 'md_checkpoint.pkl')

    Returns:
        tuple: (atoms, mdstep) where atoms is the updated ASE atoms object and
                mdstep is the MD step number from the checkpoint
    """
    if os.path.exists(filename):
        SparcLog(f"Restarting simulation from checkpoint: {filename}")
        with open(filename, "rb") as f:
            state = pickle.load(f)

        atoms.set_positions(state["positions"])
        atoms.set_velocities(state["velocities"])
        atoms.set_cell(state["cell"])
        atoms.set_pbc(state["pbc"])
        atoms.set_atomic_numbers(state["numbers"])
        atoms.set_momenta(state["momenta"])

        # Only restore stress if it was saved (for NPT)
        if state.get("stress") is not None:
            try:
                atoms.set_stress(state["stress"])
            except (RuntimeError, NotImplementedError, AttributeError):
                pass

        mdstep = state["step"]
    else:
        raise FileNotFoundError(f"\nCheckpoint file {filename} not found.")

    return atoms, mdstep


# ---------------------------------------------------------------------------------------------------#
def combine_trajectories(trajfilename, current_iter):
    """
    Combine trajectory files from all iterations up to current_iter.

    Uses incremental append: reads the existing combined trajectory (if present)
    and only adds frames from the latest iteration, avoiding O(N^2) re-reads.

    Args:
        trajfilename (str): Name of the trajectory file to combine
        current_iter (int): Current iteration number

    Returns:
        str: Path to combined trajectory file
    """
    combined_traj = Path("TrajCombined.traj")

    # If combined file exists from a previous iteration, only append the new one
    if combined_traj.exists() and current_iter > 0:
        all_frames = read(str(combined_traj), index=":")
        SparcLog(f"  Loaded {len(all_frames)} existing frames from {combined_traj}")

        # Only read the current iteration's trajectory
        iter_name = f"iter_{current_iter:06d}"
        dft_traj = Path(iter_name) / "00.dft" / trajfilename

        SparcLog(f"  Iteration [{current_iter}]")
        SparcLog(f"   → Checking File : {dft_traj}")

        if dft_traj.exists():
            new_frames = read(dft_traj, index=":")
            all_frames.extend(new_frames)
            SparcLog(f"   → Added Frames  : {len(new_frames)}\n")
    else:
        # First iteration or no existing combined file - read everything
        all_frames = []
        for i in range(current_iter + 1):
            iter_name = f"iter_{i:06d}"
            dft_traj = Path(iter_name) / "00.dft" / trajfilename

            SparcLog(f"  Iteration [{i}]")
            SparcLog(f"   → Checking File : {dft_traj}")

            if dft_traj.exists():
                frames = read(dft_traj, index=":")
                all_frames.extend(frames)
                SparcLog(f"   → Added Frames  : {len(frames)}\n")

    if not all_frames:
        raise ValueError("No trajectory data found from any iteration")

    write(str(combined_traj), all_frames)

    SparcLog("=" * 80)
    SparcLog(f" Total Frames in Combined Trajectory: {len(all_frames)}".center(72))
    SparcLog("=" * 80)

    return str(combined_traj)


# ===================================================================================================
# Save current state of Active Learning iteration in a JSON file
# ===================================================================================================


def save_progress(state, progress_file="progress.json"):
    """
    Save the current iteration state to the progress file.
    This can be used to resume the iteration from the last saved state
    The state includes the current iteration number and current step
    """
    with open(progress_file, "w") as f:
        json.dump(state, f, indent=4)


# ------------------------------------------------------------------------------------------
def load_progress(progress_file="progress.json"):
    """Load the last saved iteration state from the progress file."""
    try:
        with open(progress_file, "r") as f:
            progress_data = json.load(f)

        n_candidate = progress_data.get("candidate", None)
        i_candidate = progress_data.get("idx", None)

        if "state" in progress_data and "iteration" in progress_data:
            split_path = progress_data["state"].split("/")
            directory = split_path[1] if len(split_path) > 1 else None

            # If idx is missing (workflow was killed during ML/MD),
            # check the previous iteration's dft_candidates
            if i_candidate is None:
                current_iter = progress_data["iteration"]
                prev_iter = current_iter - 1
                prev_candidate_dir = (
                    Path(f"iter_{prev_iter:06d}") / "02.dpmd" / "dft_candidates"
                )

                # Default values
                i_candidate = 0
                n_candidate = 0

                if prev_candidate_dir.exists():
                    # Check for candidates trajectory file
                    candidates_traj = prev_candidate_dir / "candidates.extxyz"
                    if candidates_traj.exists():
                        from ase.io import read as ase_read

                        num_candidates = len(ase_read(str(candidates_traj), index=":"))
                        i_candidate = num_candidates
                        n_candidate = num_candidates

                        SparcLog("-" * 80)
                        SparcLog(" Detected restart from ML/MD stage (02.dpmd)")
                        SparcLog(
                            f" Found {num_candidates} candidates from iteration {prev_iter}"
                        )
                        SparcLog(
                            f" Setting idx = {i_candidate}, candidate = {n_candidate}"
                        )
                        SparcLog("-" * 80)

            json_data = {
                "iteration": progress_data["iteration"],
                "directory": directory,
                "candidate": n_candidate,
                "idx": i_candidate,
            }
            return json_data
        else:
            return 0

    except (FileNotFoundError, json.JSONDecodeError):
        return 0


# ------------------------------------------------------------------------------------------
def restart_progress(start_iteration):
    """
    Read labelled candidates in case of restart

    Args:
        start_iteration (dict): dictionary containing the current state which includes:
            - 'iteration': last AL iteration.
            - 'idx': index of last processed candidates.
            - 'candidate': total number of candidates.

    Returns:
        tuple: (iter, iddx, candidate_idx, candidate_found_is, candidates_file)
            - iter (int): Current iteration
            - iddx (int): Index of last processed candidate
            - candidate_idx (int): Total number of candidates
            - candidate_found_is (bool): True/False
            - candidates_file (str): Path to candidates.extxyz trajectory
    """

    # Retrieve iteration
    iter = start_iteration.get("iteration")
    if iter is None:
        raise ValueError(
            "Error: 'iteration' key is missing or None in the progress file."
        )

    # Retrieve candidate
    iddx = start_iteration.get("idx")
    candidate_idx = start_iteration.get("candidate")

    # Check if candidate is found
    candidate_found_is = True if candidate_idx else False

    iter_folder = Path(f"iter_{iter - 1:06d}")
    candidate_dir = iter_folder / "02.dpmd" / "dft_candidates"
    candidates_file = str(candidate_dir / "candidates.extxyz")

    SparcLog("-" * 80)
    SparcLog(f" Resuming Active Learning from Iteration: {iter} ")
    SparcLog("-" * 80)
    SparcLog(f" Candidates file    | {candidates_file:<35}")
    SparcLog(f" Starting Candidate | {iddx:<35}")
    SparcLog(f" Total Candidates   | {candidate_idx:<35}")
    SparcLog("-" * 80)

    return iter, iddx, candidate_idx, candidate_found_is, candidates_file


# ===================================================================================================
def remove_backup_files(file_ext="bck.*"):
    backup_files = glob.glob(file_ext)
    for file in backup_files:
        os.remove(file)


def check_physical_limits(atoms, distance_metrics):
    """
    Check if any distances exceed physical limits.

    Args:
        atoms: ase.Atoms
            The atomic system to check.
        distance_metrics: list of DistanceMetric objects or None
            List of DistanceMetric dataclass objects containing pairs of atom indices and their limits.
            Can be None or empty list if no distance checks are configured.

    Returns:
        bool: True if limits are exceeded, False otherwise.
    """
    # Handle optional distance_metrics (can be None or empty list)
    if not distance_metrics:
        return False

    for check in distance_metrics:
        # Access dataclass attributes (not dict keys)
        atom1, atom2 = check.pair
        min_distance = check.min_distance
        max_distance = check.max_distance
        distance = atoms.get_distance(atom1, atom2, mic=True)

        if distance < min_distance or distance > max_distance:
            # Get chemical symbols for the atoms
            symbol1 = atoms.get_chemical_symbols()[atom1]
            symbol2 = atoms.get_chemical_symbols()[atom2]

            SparcLog("=" * 50)
            SparcLog(" WARNING: DISTANCE CHANGED BEYOND PHYSICAL LIMIT ")
            SparcLog("-" * 50)
            SparcLog(f" ATOMS:    {symbol1} ({atom1}) -- {symbol2} ({atom2}) ")
            SparcLog(f" MEASURED: {distance:.2f} Å (MIN. LIMIT: {min_distance:.2f} Å) ")
            SparcLog(f" MEASURED: {distance:.2f} Å (MAX. LIMIT: {max_distance:.2f} Å) ")
            SparcLog("=" * 50 + "\n")
            return True

    return False


# ---------------------------------------------------------------------------------------
# Helper Function to Restart ML/MD exploration
# ---------------------------------------------------------------------------------------
def get_initial_structure(
    iter: int,
    sample_idx: int,
    config,  # SparcConfig
    structure_file: Union[str, Path, List[str]],
    parent_dir: Path,
    struct_idx: int = None,
):
    """
    Get initial structure for ML-MD run with optional restart exploration.

    If restart_exploration is enabled and iter > 0, loads structure from previous
    iteration based on restart_mode strategy. Otherwise, loads from original structure.

    Parameters
    ----------
    iter : int
        Current active learning iteration number
    sample_idx : int
        Sample index for multiple parallel runs (0, 1, 2, ...)
    config : SparcConfig
        Full configuration object
    structure_file : str, Path, or list of str
        Path to original structure file(s)
    parent_dir : Path
        Parent directory containing iteration folders

    Returns
    -------
    atoms : ase.Atoms
        Initial structure for MD simulation

    Notes
    -----
    Restart strategies:
    - "last": All runs start from last frame (best for single run)
    - "random": Each run starts from different random frame from trajectory
    - "candidates": Each run starts from different random DFT candidate (safest)
    """
    import random

    # Check if restart exploration is enabled
    if config.mlip_setup.restart_exploration and iter > 0:
        restart_mode = config.mlip_setup.restart_frame  # restart_frame
        prev_iter = iter - 1

        # Path to previous iteration's trajectory
        prev_traj_file = (
            parent_dir / f"iter_{prev_iter:06d}" / "02.dpmd" / config.output.dptraj_file
        )

        if os.path.exists(prev_traj_file):
            try:
                if restart_mode == "last":
                    # All runs use last frame (most equilibrated)
                    atoms = read(prev_traj_file, index=-1)
                    SparcLog(
                        f" Restart exploration: Using last frame from iteration {prev_iter}"
                    )

                elif restart_mode == "random":
                    # Each run gets a DIFFERENT random frame from trajectory
                    prev_traj = read(prev_traj_file, index=":")

                    # Use sample_idx as seed to get different frames for each run
                    random.seed(iter * 1000 + sample_idx)
                    random_idx = random.randint(0, len(prev_traj) - 1)

                    atoms = prev_traj[random_idx]
                    SparcLog(
                        f" Restart exploration: Run {sample_idx + 1} using frame {random_idx}/{len(prev_traj)} from iteration {prev_iter}"
                    )

                elif restart_mode == "candidates":
                    # Each run starts from a DIFFERENT random DFT candidate (safest)
                    candidate_file = (
                        parent_dir
                        / f"iter_{prev_iter:06d}"
                        / "00.dft"
                        / config.output.aimdtraj_file
                    )

                    if os.path.exists(candidate_file):
                        candidates = read(candidate_file, index=":")

                        # Random selection with different seed for each run
                        random.seed(iter * 1000 + sample_idx)
                        random_candidates = random.sample(
                            range(len(candidates)), min(len(candidates), 100)
                        )
                        cand_idx = random_candidates[
                            sample_idx % len(random_candidates)
                        ]

                        atoms = candidates[cand_idx]
                        SparcLog(
                            f" Restart exploration: Run {sample_idx + 1} using DFT candidate {cand_idx + 1}/{len(candidates)} from iteration {prev_iter}"
                        )
                    else:
                        SparcLog(
                            " Candidates file not found, using last frame instead",
                            level="WARNING",
                        )
                        atoms = read(prev_traj_file, index=-1)

                else:
                    raise ValueError(
                        f"Invalid restart_mode: {restart_mode}. Options: 'last', 'random', 'candidates'"
                    )

                return atoms

            except Exception as e:
                SparcLog(
                    f" Warning: Could not load from previous iteration: {e}",
                    level="WARNING",
                )
                SparcLog(" Falling back to original structure")

        else:
            SparcLog(
                f" Warning: Previous trajectory not found at {prev_traj_file}",
                level="WARNING",
            )
            SparcLog(" Falling back to original structure")

    # Default: Use original structure
    if isinstance(structure_file, list):
        # struct_idx (cross-product index) takes priority over sample_idx % n
        idx = struct_idx if struct_idx is not None else sample_idx % len(structure_file)
        return read(structure_file[idx])
    return read(structure_file)


# ===================================================================================================
# END OF FILE
# ===================================================================================================

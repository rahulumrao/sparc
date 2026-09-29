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
            - 'idx': index of the last candidate that completed labelling.
            - 'candidate': total number of candidates.

    Returns:
        tuple: (iter, iddx, candidate_idx, candidate_found_is, candidates_file)
            - iter (int): Current iteration
            - iddx (int): Index of the first candidate still to be labelled
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

    # save_progress writes 'idx' after that candidate is labelled and written, so
    # resume starts at idx + 1. Resuming at idx repeats one DFT calculation and
    # duplicates its frame in the training set. Missing/null idx means none done.
    last_completed = start_iteration.get("idx") or 0
    iddx = last_completed + 1
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
    SparcLog(f" Last Completed     | {last_completed:<35}")
    SparcLog(f" Resuming at        | {iddx:<35}")
    SparcLog(f" Total Candidates   | {candidate_idx:<35}")
    SparcLog("-" * 80)

    return iter, iddx, candidate_idx, candidate_found_is, candidates_file


# ===================================================================================================
def remove_backup_files(file_ext="bck.*"):
    backup_files = glob.glob(file_ext)
    for file in backup_files:
        os.remove(file)


def flush_memory(backend: str = "pytorch"):
    """
    Best-effort cleanup of GPU memory after a calculator or model is released.
    """
    import gc

    gc.collect()

    if backend == "pytorch":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    elif backend == "tensorflow":
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except ImportError:
            pass


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
# Label validation — acceptance criteria for a newly labelled candidate frame
# ---------------------------------------------------------------------------------------
def labelled_reference_energy(
    trajfilename: str, current_iter: int, parent_dir: Union[str, Path] = "."
) -> Union[float, None]:
    """
    Return the minimum potential energy over frames already in the training set.

    Serves as the anchor E_ref for the acceptance window applied to newly
    labelled candidates. A separate anchor is required because the ML-MD gate in
    ``ExecuteMlpDynamics`` tests E_ML(t) against E_ML(t=0) of the same run: both
    terms carry the model's error, so the test is meaningful only while that
    error is small. Outside the training manifold the error is unbounded — for a
    dissociating molecule the committee under-predicts E by several eV, the frame
    appears more stable than it is, and the window is never crossed. Referencing
    the labelled energy against a fixed point of the reference energy scale
    removes the model from the test.

    The minimum is used rather than a mean or a first frame: it is invariant to
    high-energy contamination already present in the set, and it is monotonic
    under dataset growth, decreasing only when a more stable configuration is
    labelled.

    Parameters
    ----------
    trajfilename : str
        Labelled-trajectory filename inside each ``iter_XXXXXX/00.dft/``.
    current_iter : int
        Iteration being labelled; only preceding iterations contribute.
    parent_dir : str or Path
        Root directory holding ``iter_XXXXXX/`` and ``TrajCombined.traj``.

    Returns
    -------
    float or None
        Reference energy in eV, or None when no labelled data exists yet.
    """
    parent_dir = Path(parent_dir)

    sources = []
    combined = parent_dir / "TrajCombined.traj"
    if combined.exists():
        # Rewritten after each labelling step; already spans iterations 0..N-1.
        sources.append(combined)
    else:
        for i in range(current_iter):
            traj = parent_dir / f"iter_{i:06d}" / "00.dft" / trajfilename
            if traj.exists():
                sources.append(traj)

    energies = []
    for src in sources:
        try:
            for atoms in read(str(src), index=":"):
                energy = float(np.ravel(atoms.get_potential_energy())[0])
                if np.isfinite(energy):
                    energies.append(energy)
        except Exception as exc:
            SparcLog(
                f" Warning: could not read reference energies from {src}: {exc}",
                level="WARNING",
            )

    return min(energies) if energies else None


def validate_labelled_frame(system, energy, forces, epot_ref=None, epot_threshold=None):
    """
    Test whether a labelled frame is admissible as a training label.

    Two independent criteria are applied.

    Finiteness of E, F and R. A non-finite value is not a large value but an
    absent one: the SCF diverged, the run was killed, or the parser returned a
    placeholder. Such a frame carries no information and propagates NaN through
    the loss on the first training step.

    Energy window E <= E_ref + epot_threshold. The bound is one-sided. An excess
    above E_ref indicates either fragmentation of the system or an unconverged
    SCF; in both cases the label is physically unwanted or numerically wrong. A
    deficit below E_ref indicates a more stable configuration, which is the
    intended product of biased exploration and must not be rejected.

    No bound is placed on |F|. Over the labelled frames of this workflow |F|max
    for fragmented configurations falls inside the range spanned by intact ones,
    so force magnitude carries no discriminating power for this failure mode and
    any cutoff would be arbitrary.

    Parameters
    ----------
    system : ase.Atoms
        The labelled structure.
    energy : float or None
        Potential energy [eV].
    forces : numpy.ndarray or None
        Forces [eV/Å]; None skips the force check.
    epot_ref : float or None
        Reference energy from :func:`labelled_reference_energy` [eV]. None
        disables the energy window.
    epot_threshold : float or None
        Width of the accepted window above ``epot_ref`` [eV]. None disables the
        energy window.

    Returns
    -------
    tuple
        ``(is_valid, reason)``; *reason* is an empty string when valid.
    """
    if energy is None:
        return False, "calculator returned no energy"

    energy = float(np.ravel(energy)[0])
    if not np.isfinite(energy):
        return False, f"non-finite energy ({energy})"

    if forces is not None and not np.isfinite(np.asarray(forces, dtype=float)).all():
        return False, "non-finite forces (NaN/Inf)"

    if not np.isfinite(system.get_positions()).all():
        return False, "non-finite atomic positions"

    if epot_ref is not None and epot_threshold is not None:
        epot_ref = float(epot_ref)
        excess = energy - (epot_ref + float(epot_threshold))
        if excess > 0.0:
            return False, (
                f"E = {energy:.3f} eV exceeds E_ref + threshold "
                f"({epot_ref:.3f} + {float(epot_threshold):.2f} eV) by {excess:.3f} eV"
            )

    return True, ""


def save_rejected_frame(idx, system, energy, forces, reason, dir_name):
    """
    Record a candidate rejected by :func:`validate_labelled_frame`.

    The structure is appended to ``rejected.extxyz`` and the reason to
    ``rejected_candidates.dat``, both inside *dir_name*.

    Energy and forces are reattached through a SinglePointCalculator when both
    are finite, so a frame rejected on the energy window alone remains a
    complete labelled result and can be readmitted without recomputing it if the
    window is later widened. Frames rejected for non-finite values are written
    as geometry only: the numbers they carry are absent rather than large, and
    would propagate to any reader that treats them as data.

    Parameters
    ----------
    idx : int
        Candidate index within ``candidates.extxyz``.
    system : ase.Atoms
        The rejected structure.
    energy : float or None
        Potential energy [eV], or None if the calculator returned nothing.
    forces : numpy.ndarray or None
        Forces [eV/Å], or None if the calculator returned nothing.
    reason : str
        Rejection reason from :func:`validate_labelled_frame`.
    dir_name : str or Path
        Directory holding the iteration's labelling output.
    """
    frame = system.copy()
    frame.info["rejection_reason"] = reason
    frame.info["candidate_index"] = idx

    energy_finite = energy is not None and np.isfinite(energy)
    forces_finite = forces is not None and np.isfinite(np.asarray(forces)).all()
    if energy_finite and forces_finite:
        frame.calc = SinglePointCalculator(
            frame, energy=float(energy), forces=np.asarray(forces)
        )

    rejected_traj = os.path.join(str(dir_name), "rejected.extxyz")
    rejected_log = os.path.join(str(dir_name), "rejected_candidates.dat")

    try:
        write(rejected_traj, frame, format="extxyz", append=True)
    except Exception as exc:
        SparcLog(
            f"Warning: could not write rejected frame {idx} to {rejected_traj}: {exc}",
            level="WARNING",
        )

    write_header = not os.path.exists(rejected_log)
    with open(rejected_log, "a") as f:
        if write_header:
            f.write("# Candidates rejected before entering the training set\n")
            f.write(f"# {'Candidate':>10} {'Epot[eV]':>14}  Reason\n")
            f.write(f"#{'=' * 78}\n")
        epot_col = f"{float(energy):14.6f}" if energy_finite else f"{'n/a':>14}"
        f.write(f"{idx:>12} {epot_col}  {reason}\n")


# ---------------------------------------------------------------------------------------
# Restart exploration — seed ML-MD from the validated candidate ensemble
# ---------------------------------------------------------------------------------------
def _validated_pool(parent_dir: Path, iter: int, trajfilename: str) -> Path:
    """
    Frames of iteration *iter* that passed validation during labelling.

    This is the QbC pool minus everything validate_labelled_frame rejected,
    written by the labelling step of the same iteration, which runs before ML-MD.
    Seeding from the raw QbC pool instead lets a rejected geometry start the next
    trajectory: exploration continues inside the region the energy window
    excludes, every candidate it yields is rejected in turn, and the run cannot
    recover without intervention.
    """
    return parent_dir / f"iter_{iter:06d}" / "00.dft" / trajfilename


def get_initial_structure(
    iter: int,
    sample_idx: int,
    config,  # SparcConfig
    structure_file: Union[str, Path, List[str]],
    parent_dir: Path,
    struct_idx: int = None,
):
    """
    Return the initial atomic configuration for an ML-MD exploration run.

    When ``restart_exploration`` is active (iteration > 0), the starting geometry
    is drawn from the frames this iteration's labelling step accepted
    (``00.dft/<aimdtraj_file>``).  These lie within the committee force-deviation
    window, passed geometric deduplication, and carry a validated DFT label — so
    they are locally distinct, model-uncertain configurations that are also
    physically admissible.

    Parameters
    ----------
    iter : int
        Current active-learning iteration index.
    sample_idx : int
        Index of the parallel MD shoot (0, 1, …).
    config : SparcConfig
        Workflow configuration.
    structure_file : str, Path, or list of str
        Original input structure path(s).
    parent_dir : Path
        Root directory containing ``iter_XXXXXX/`` folders.

    Returns
    -------
    ase.Atoms
        Initial structure for the MD integrator.

    Notes
    -----
    ``restart_frame`` strategies (source: accepted ``00.dft/<aimdtraj_file>``):

    * ``"last"`` — final candidate in the pool; continues from the most
      recently flagged uncertain configuration.
    * ``"candidates"`` — deterministic round-robin over the pool; parallel
      shoots seed from distinct committee-selected frames (no replacement
      until the pool is exhausted).
    * ``"random"`` — independent uniform draw from the pool; stochastic
      resampling that may revisit the same configuration across shoots.
    """
    import random

    if config.mlip_setup.restart_exploration and iter > 0:
        restart_mode = config.mlip_setup.restart_frame
        candidate_file = _validated_pool(parent_dir, iter, config.output.aimdtraj_file)

        if candidate_file.exists():
            try:
                candidates = read(str(candidate_file), index=":")
                n_candidates = len(candidates)

                if n_candidates == 0:
                    raise ValueError(f"Empty validated pool: {candidate_file}")

                if restart_mode == "last":
                    # Most recent frame in the committee-selected ensemble.
                    atoms = candidates[-1]
                    SparcLog(
                        f" Restart exploration [{restart_mode}]: "
                        f"candidate {n_candidates}/{n_candidates}, iter {iter}"
                    )

                elif restart_mode == "candidates":
                    # Partition the QbC ensemble across parallel shoots.
                    cand_idx = sample_idx % n_candidates
                    atoms = candidates[cand_idx]
                    SparcLog(
                        f" Restart exploration [{restart_mode}]: "
                        f"candidate {cand_idx + 1}/{n_candidates}, "
                        f"shoot {sample_idx + 1}, iter {iter}"
                    )

                elif restart_mode == "random":
                    # Independent uniform resampling from the QbC pool.
                    random.seed(iter * 1000 + sample_idx)
                    cand_idx = random.randint(0, n_candidates - 1)
                    atoms = candidates[cand_idx]
                    SparcLog(
                        f" Restart exploration [{restart_mode}]: "
                        f"candidate {cand_idx + 1}/{n_candidates}, "
                        f"shoot {sample_idx + 1}, iter {iter}"
                    )

                else:
                    raise ValueError(
                        f"Invalid restart_frame: {restart_mode}. "
                        "Options: 'last', 'candidates', 'random'"
                    )

                return atoms

            except Exception as e:
                SparcLog(
                    f" Warning: Could not load validated pool {candidate_file}: {e}",
                    level="WARNING",
                )
                SparcLog(" Falling back to input structure")

        else:
            SparcLog(
                f" Warning: Validated pool not found: {candidate_file}",
                level="WARNING",
            )
            SparcLog(" Falling back to input structure")

    # Default: Use original structure
    if isinstance(structure_file, list):
        # struct_idx (cross-product index) takes priority over sample_idx % n
        idx = struct_idx if struct_idx is not None else sample_idx % len(structure_file)
        return read(structure_file[idx])
    return read(structure_file)


# ===================================================================================================
# END OF FILE
# ===================================================================================================

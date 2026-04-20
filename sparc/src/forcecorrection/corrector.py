"""
PLUMED file parsers and bias force correction pipeline.

Reads biased forces from an ASE trajectory, applies the PLUMED bias
correction using CV forces and derivatives, and rewrites the trajectory
with physical (unbiased) forces.

Assumption: PLUMED DUMPFORCES/DUMPDERIVATIVES STRIDE must equal
aimd_setup.log_frequency so that row i in the PLUMED files aligns
with frame i in the ASE trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.io.trajectory import Trajectory

from sparc.src.forcecorrection.compute_bias import (
    compute_bias_forces,
    remove_bias,
    correction_summary,
)
from sparc.src.utils.logger import SparcLog


# ──────────────────────────────────────────────────────────────────────
# PLUMED file parsers
# ──────────────────────────────────────────────────────────────────────

def _parse_cv_forces(filepath: Path) -> np.ndarray:
    """
    Parse a PLUMED DUMPFORCES file.

    Expected format::

        #! FIELDS time cv1 cv2 ...
         0.500000   0.419...   0.005...

    Returns
    -------
    forces : ndarray, shape (nsteps, ncvs)
        Scalar force on each CV at each step (columns after 'time').
    """
    rows: list[list[float]] = []
    with open(filepath) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('@'):
                continue
            cols = s.split()
            rows.append([float(c) for c in cols[1:]])  # skip time column

    if not rows:
        raise ValueError(f"No data found in {filepath}")

    return np.array(rows)  # (nsteps, ncvs)


def _parse_cv_derivs(
    filepath: Path,
    cv_atoms: List[List[int]],
) -> List[np.ndarray]:
    """
    Parse a PLUMED DUMPDERIVATIVES file.

    Expected format::

        #! FIELDS time parameter cv1 cv2 ...
         0.500000 0   dx_cv1  dx_cv2
         0.500000 1   dy_cv1  dy_cv2
         ...

    Parameter indices are LOCAL to each CV:
        param 3*j+0 = atom j, x-component
        param 3*j+1 = atom j, y-component
        param 3*j+2 = atom j, z-component

    Parameters beyond 3 * max(natoms_per_cv) are cell/virial — ignored.

    Parameters
    ----------
    cv_atoms : list of lists of 1-based atom indices, one per CV.

    Returns
    -------
    derivs_per_cv : list of ndarray, each shape (nframes, natoms_in_cv_k, 3)
    """
    ncvs = len(cv_atoms)
    cv_natoms = [len(a) for a in cv_atoms]
    max_needed = max(3 * n for n in cv_natoms)

    # Group rows by frame (contiguous block with same time)
    frame_groups: list[dict[int, list[float]]] = []
    current_time: float | None = None
    current_block: dict[int, list[float]] = {}

    with open(filepath) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('@'):
                continue
            cols = s.split()
            t   = float(cols[0])
            param = int(cols[1])
            vals  = [float(c) for c in cols[2:]]  # one value per CV

            if len(vals) != ncvs:
                raise ValueError(
                    f"{filepath}: expected {ncvs} CV columns at param {param}, "
                    f"got {len(vals)}"
                )

            if current_time is None:
                current_time = t

            if t != current_time:
                frame_groups.append(current_block)
                current_block = {}
                current_time  = t

            current_block[param] = vals

    if current_block:
        frame_groups.append(current_block)

    if not frame_groups:
        raise ValueError(f"No derivative frames found in {filepath}")

    nframes = len(frame_groups)

    # Build per-CV derivative arrays
    derivs_per_cv: list[np.ndarray] = []
    for k in range(ncvs):
        needed = 3 * cv_natoms[k]
        arr = np.zeros((nframes, cv_natoms[k], 3))

        for fi, block in enumerate(frame_groups):
            for p in range(needed):
                if p not in block:
                    raise ValueError(
                        f"{filepath}: frame {fi}, CV {k}: "
                        f"missing parameter {p} (expected 0..{needed-1})"
                    )
                local_atom = p // 3
                xyz        = p % 3
                arr[fi, local_atom, xyz] = block[p][k]

        derivs_per_cv.append(arr)

    SparcLog(
        f"  Parsed {nframes} derivative frames x {ncvs} CVs from "
        f"{filepath.name} "
        f"({max_needed} atomic params + "
        f"{max(0, len(next(iter(frame_groups))) - max_needed)} cell params ignored)"
    )
    return derivs_per_cv


# ──────────────────────────────────────────────────────────────────────
# Validation checks
# ──────────────────────────────────────────────────────────────────────

def force_validation(
    f_total: np.ndarray,
    f_physical: np.ndarray,
) -> None:
    """
    Validate corrected forces via Newton's 3rd law (net force check).

    For a closed periodic system the sum of all physical forces must be
    zero. PLUMED bias forces violate this; after correction the net force
    should be negligible.

    Parameters
    ----------
    f_total   : ndarray (nframes, natoms, 3) — original biased forces
    f_physical: ndarray (nframes, natoms, 3) — corrected forces
    """
    SparcLog("  --- Validation ---")

    net_biased    = np.linalg.norm(f_total.sum(axis=1),    axis=1)  # (nframes,)
    net_corrected = np.linalg.norm(f_physical.sum(axis=1), axis=1)  # (nframes,)

    SparcLog("  Net force |ΣF| (should be ~0 after correction):")
    SparcLog(f"    Biased    — mean: {net_biased.mean():.4f}  max: {net_biased.max():.4f} eV/Å")
    SparcLog(f"    Corrected — mean: {net_corrected.mean():.4f}  max: {net_corrected.max():.4f} eV/Å")

    if net_corrected.mean() > net_biased.mean() + 1e-6:
        SparcLog(
            "    WARNING: corrected net force is larger than biased — "
            "check that cv_atoms order matches ARG= in plumed.dat"
        )
    elif net_corrected.mean() < 1e-3:
        SparcLog("    PASS: net force is negligible after correction")
    else:
        SparcLog("    OK: net force reduced (small residual may come from cell/virial coupling)")

    SparcLog("  ------------------")


# ──────────────────────────────────────────────────────────────────────
# Main correction function
# ──────────────────────────────────────────────────────────────────────

def correct_aimd_forces(
    traj_path: Path,
    cv_atoms: List[List[int]],
    dft_dir: Optional[Path] = None,
    cv_force_file: str = "cv_force",
    cv_derivs_file: str = "cv_derivs",
) -> None:
    """
    Remove PLUMED bias forces from an ASE AIMD trajectory.

    Reads biased forces from `traj_path`, loads the PLUMED DUMPFORCES
    and DUMPDERIVATIVES files, computes the bias force for each frame
    via the chain rule, subtracts it, and overwrites `traj_path` with
    the corrected (physical) forces.

    Supported trajectory formats: ASE binary (.traj) and extxyz (.xyz).

    Parameters
    ----------
    traj_path : Path
        Trajectory file to correct (overwritten in-place).
    cv_atoms : list of lists of 1-based atom indices
        One entry per CV, in the same order as ARG= in plumed.dat.
        e.g. [[1, 4], [1, 2]] for two distance CVs.
    dft_dir : Path, optional
        Directory containing the PLUMED output files. Defaults to the
        parent directory of traj_path (or cwd if traj_path has no parent).
    cv_force_file : str
        Filename of the PLUMED DUMPFORCES output inside dft_dir.
    cv_derivs_file : str
        Filename of the PLUMED DUMPDERIVATIVES output inside dft_dir.
    """
    traj_path   = Path(traj_path)
    dft_dir     = Path(dft_dir) if dft_dir is not None else traj_path.parent or Path(".")
    force_path  = dft_dir / cv_force_file
    derivs_path = dft_dir / cv_derivs_file

    SparcLog("")
    SparcLog("=" * 60)
    SparcLog("PLUMED Bias Force Correction")
    SparcLog(f"  Trajectory   : {traj_path}")
    SparcLog(f"  CV forces    : {force_path}")
    SparcLog(f"  CV derivs    : {derivs_path}")
    SparcLog(f"  CVs ({len(cv_atoms)}): {cv_atoms}")

    for p in (force_path, derivs_path):
        if not p.exists():
            raise FileNotFoundError(
                f"PLUMED output file not found: {p}\n"
                f"Make sure DUMPFORCES/DUMPDERIVATIVES are configured in "
                f"your plumed.dat with FILE pointing inside the run directory."
            )

    # ── Read biased trajectory ──
    fmt = traj_path.suffix.lower()
    if fmt == ".traj":
        with Trajectory(str(traj_path), 'r') as traj:
            frames = list(traj)
    else:
        frames = read(str(traj_path), index=':')

    natoms  = len(frames[0])
    nframes = len(frames)
    SparcLog(f"  Trajectory   : {nframes} frames, {natoms} atoms")

    # ── Parse PLUMED files ──
    f_cv_all   = _parse_cv_forces(force_path)    # (nsteps, ncvs)
    derivs_all = _parse_cv_derivs(derivs_path, cv_atoms)  # list of (nframes, natoms_k, 3)

    ncvs = len(cv_atoms)
    if f_cv_all.shape[1] != ncvs:
        raise ValueError(
            f"cv_force has {f_cv_all.shape[1]} CV columns but "
            f"cv_atoms specifies {ncvs} CVs"
        )

    # Align: take min of available frames
    n = min(nframes, f_cv_all.shape[0])
    if n < nframes:
        SparcLog(
            f"  WARNING: PLUMED has {f_cv_all.shape[0]} rows but trajectory "
            f"has {nframes} frames — using first {n} frames",
        )
    frames     = frames[:n]
    f_cv_all   = f_cv_all[:n]
    derivs_all = [d[:n] for d in derivs_all]

    # ── Stack biased forces from trajectory ──
    f_total = np.array([atoms.get_forces() for atoms in frames])  # (n, natoms, 3)

    # ── Convert cv_atoms to 0-based global indices ──
    atom_indices_0based = [[a - 1 for a in cv] for cv in cv_atoms]

    # ── Compute and subtract bias ──
    f_bias     = compute_bias_forces(f_cv_all, derivs_all, atom_indices_0based, natoms)
    f_physical = remove_bias(f_total, f_bias)

    # ── Validate correction ──
    force_validation(f_total, f_physical)

    # ── Log diagnostics ──
    stats = correction_summary(f_total, f_bias)
    SparcLog(f"  |F_bias|  mean={stats['bias_norm_mean']:.4f}  max={stats['bias_norm_max']:.4f} eV/Å")
    SparcLog(f"  |F_total| mean={stats['total_norm_mean']:.4f}  max={stats['total_norm_max']:.4f} eV/Å")
    SparcLog(f"  Bias/Total ratio : {stats['bias_to_total_ratio']:.4f}")
    if stats['top_atoms']:
        SparcLog("  Most affected atoms (1-based):")
        for idx, val in stats['top_atoms']:
            SparcLog(f"    atom {idx+1:4d} : {val:.4f} eV/Å")

    # ── Overwrite trajectory with corrected forces ──
    corrected = []
    for atoms, f_corr in zip(frames, f_physical):
        energy = atoms.get_potential_energy()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=f_corr)
        corrected.append(atoms)

    if fmt == ".traj":
        with Trajectory(str(traj_path), 'w') as out_traj:
            for atoms in corrected:
                out_traj.write(atoms)
    else:
        write(str(traj_path), corrected)

    SparcLog(f"  Corrected trajectory written -> {traj_path}")
    SparcLog("=" * 60)
    SparcLog("")

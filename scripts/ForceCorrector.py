#!/usr/bin/env python3
"""
forceCorr.py — Remove PLUMED bias forces from CP2K trajectory forces
               and produce ML-ready extended XYZ training data.

Supports one or multiple collective variables (CVs). PLUMED writes all
CV data as columns in shared files:

    cv_force:   #! FIELDS time d1 d2 ...
    cv_derivs:  #! FIELDS time parameter d1 d2 ...

Each CV contributes a bias force via the chain rule:

    F_bias = sum_k  f_cv_k * (ds_k / dR_k)

The corrected (unbiased) forces are:

    F_unbiased = F_total - F_bias

When a CP2K position file (--pos) is provided, the script also reads
coordinates and energies, converts energy from Hartree to eV, and
writes a combined extended XYZ file ready for ML training.

Usage examples
--------------
Produce extxyz for ML training (two CVs):
    python forceCorr.py \\
        --frc btd-frc-1.xyz \\
        --pos btd-pos-1.xyz \\
        --cv-force cv_force --cv-derivs cv_derivs \\
        --cv-atoms 1,4 1,2 \\
        --lattice 17.0 0 0  0 17.0 0  0 0 17.0 \\
        --out-extxyz train.xyz

With stress tensor and cell file:
    python forceCorr.py \\
        --frc btd-frc-1.xyz \\
        --pos btd-pos-1.xyz \\
        --cv-force cv_force --cv-derivs cv_derivs \\
        --cv-atoms 1,4 1,2 \\
        --cell-file btd-1.cell \\
        --stress btd-stress-1.stress_tensor \\
        --stress-unit au \\
        --out-extxyz train.xyz

Via YAML config file:
    python forceCorr.py --config config.yaml

Example config.yaml:
    frc: btd-frc-1.xyz
    pos: btd-pos-1.xyz
    cv_force: cv_force
    cv_derivs: cv_derivs
    cvs:
      - name: d1
        atoms: [1, 4]
      - name: d2
        atoms: [1, 2]
    lattice: [17.0, 0, 0, 0, 17.0, 0, 0, 0, 17.0]
    # OR: cell_file: btd-1.cell
    stress:
      file: btd-stress-1.stress_tensor
      unit: au       # au | eV/A^3 | GPa
    output:
      extxyz: train.xyz
      xyz: forces_unbiased.xyz
      flat: forces_unbiased.dat
      bias_xyz: bias_forces.xyz
    pbc: [true, true, true]
    frames:
      start: 0
      end: null
      stride: 1
    time_tolerance: 1.0e-8
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ComputeBias import (
    compute_bias_forces,
    remove_bias,
    correction_summary,
)

logger = logging.getLogger("forceCorr")


# ──────────────────────────────────────────────────────────────────────
# Unit conversions
# ──────────────────────────────────────────────────────────────────────

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903

# Stress conversions TO eV/A^3
_STRESS_CONVERSIONS = {
    "ev/a^3": 1.0,
    "ev/ang^3": 1.0,
    "au": HARTREE_TO_EV / (BOHR_TO_ANG ** 3),  # Hartree/Bohr^3
    "gpa": 1.0 / 160.21766208,                  # 1 eV/A^3 = 160.2 GPa
}


def stress_conversion_factor(unit: str) -> float:
    """Return factor to convert from `unit` to eV/A^3."""
    key = unit.lower().strip()
    if key not in _STRESS_CONVERSIONS:
        valid = ", ".join(_STRESS_CONVERSIONS.keys())
        raise ValueError(
            f"Unknown stress unit '{unit}'. Valid options: {valid}"
        )
    return _STRESS_CONVERSIONS[key]


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CVSpec:
    """Specification for a single collective variable."""
    name: str
    atom_indices: list[int]  # 1-based

    @property
    def natoms(self) -> int:
        return len(self.atom_indices)

    def __str__(self) -> str:
        atoms_str = ",".join(str(a) for a in self.atom_indices)
        return f"{self.name}(atoms=[{atoms_str}])"


@dataclass
class Config:
    """Full run configuration."""
    frc_file: str
    cv_force_file: str
    cv_derivs_file: str
    cvs: list[CVSpec]
    pos_file: Optional[str] = None
    stress_file: Optional[str] = None
    stress_unit: str = "au"
    cell_file: Optional[str] = None
    lattice: Optional[list[float]] = None  # 9 floats (3x3 row-major)
    pbc: list[bool] = field(default_factory=lambda: [True, True, True])
    out_extxyz: Optional[str] = None
    out_xyz: str = "forces_unbiased.xyz"
    out_flat: str = "forces_unbiased.dat"
    out_bias_xyz: Optional[str] = None
    frame_start: int = 0
    frame_end: Optional[int] = None
    frame_stride: int = 1
    time_tol: float = 1e-8
    validate: bool = True

    @property
    def ncvs(self) -> int:
        return len(self.cvs)

    def summary(self) -> str:
        lines = [
            "=== Configuration ===",
            f"  Force file     : {self.frc_file}",
        ]
        if self.pos_file:
            lines.append(f"  Position file  : {self.pos_file}")
        lines += [
            f"  CV force file  : {self.cv_force_file}",
            f"  CV derivs file : {self.cv_derivs_file}",
            f"  CVs ({self.ncvs}):",
        ]
        for i, cv in enumerate(self.cvs):
            lines.append(f"    [{i + 1}] {cv}")
        if self.stress_file:
            lines.append(
                f"  Stress file    : {self.stress_file} "
                f"(unit: {self.stress_unit})"
            )
        if self.cell_file:
            lines.append(f"  Cell file      : {self.cell_file}")
        elif self.lattice:
            L = self.lattice
            lines.append(
                f"  Lattice        : "
                f"[{L[0]:.4f} {L[1]:.4f} {L[2]:.4f}] "
                f"[{L[3]:.4f} {L[4]:.4f} {L[5]:.4f}] "
                f"[{L[6]:.4f} {L[7]:.4f} {L[8]:.4f}]"
            )
        pbc_str = " ".join("T" if p else "F" for p in self.pbc)
        lines.append(f"  PBC            : {pbc_str}")
        lines += [
            f"  Frame range    : start={self.frame_start}, "
            f"end={self.frame_end}, stride={self.frame_stride}",
        ]
        if self.out_extxyz:
            lines.append(f"  Output extxyz  : {self.out_extxyz}")
        lines += [
            f"  Output XYZ     : {self.out_xyz}",
            f"  Output flat    : {self.out_flat}",
        ]
        if self.out_bias_xyz:
            lines.append(f"  Bias XYZ       : {self.out_bias_xyz}")
        lines.append(f"  Time tol       : {self.time_tol}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# I/O — XYZ trajectory reader (forces or positions + energy)
# ──────────────────────────────────────────────────────────────────────

def read_xyz_trajectory(
    filename: str,
    extract_energy: bool = False,
) -> tuple[np.ndarray, list[str], np.ndarray, Optional[np.ndarray]]:
    """
    Read an XYZ-format trajectory (positions or forces).

    The comment line is expected to contain ``time = <value>``.
    If `extract_energy` is True, also parses ``E = <value>``.

    Parameters
    ----------
    filename : str
    extract_energy : bool
        If True, extract energy from comment line (Hartree for CP2K pos).

    Returns
    -------
    times : ndarray, shape (nframes,)
    symbols : list of str
    data : ndarray, shape (nframes, natoms, 3)
    energies : ndarray, shape (nframes,) or None
        Raw energies as read (no unit conversion applied here).
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"XYZ file not found: {filename}")

    times: list[float] = []
    frames: list[list[list[float]]] = []
    energies: list[float] = []
    symbols_ref: Optional[list[str]] = None

    with open(filename) as f:
        lineno = 0
        while True:
            line = f.readline()
            lineno += 1
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                natoms = int(line)
            except ValueError:
                raise ValueError(
                    f"{filename}:{lineno}: expected atom count, got '{line}'"
                )

            comment = f.readline().rstrip("\n")
            lineno += 1

            m_time = re.search(r"time\s*=\s*([-+0-9.eE]+)", comment)
            if not m_time:
                raise ValueError(
                    f"{filename}:{lineno}: could not parse 'time = ...' "
                    f"from comment: {comment}"
                )
            time = float(m_time.group(1))

            if extract_energy:
                m_energy = re.search(r"E\s*=\s*([-+0-9.eE]+)", comment)
                if not m_energy:
                    raise ValueError(
                        f"{filename}:{lineno}: could not parse 'E = ...' "
                        f"from comment: {comment}"
                    )
                energies.append(float(m_energy.group(1)))

            symbols: list[str] = []
            coords: list[list[float]] = []
            for _ in range(natoms):
                parts = f.readline().split()
                lineno += 1
                if len(parts) < 4:
                    raise ValueError(
                        f"{filename}:{lineno}: malformed atom line "
                        f"(need >=4 columns): {parts}"
                    )
                symbols.append(parts[0])
                coords.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )

            if symbols_ref is None:
                symbols_ref = symbols
            elif symbols != symbols_ref:
                raise ValueError(
                    f"{filename}: atom symbols changed at time={time} "
                    f"(expected {symbols_ref[:3]}..., got {symbols[:3]}...)"
                )

            times.append(time)
            frames.append(coords)

    nframes = len(times)
    if nframes == 0:
        raise ValueError(f"{filename}: no frames found")

    logger.info(
        f"Read {nframes} frames ({len(symbols_ref)} atoms: "
        f"{symbols_ref[0]}...{symbols_ref[-1]}) from {filename}"
    )

    energy_arr = np.array(energies) if energies else None
    return np.array(times), symbols_ref, np.array(frames), energy_arr


def write_xyz_vectors(
    filename: str,
    symbols: list[str],
    times: np.ndarray,
    data: np.ndarray,
    energy_placeholder: float = 0.0,
) -> None:
    """Write forces/positions in XYZ trajectory format."""
    with open(filename, "w") as f:
        for i, (t, frame) in enumerate(zip(times, data)):
            f.write(f"{len(symbols):6d}\n")
            f.write(
                f" i = {i:8d}, time = {t:12.3f}, "
                f"E = {energy_placeholder:18.10f}\n"
            )
            for sym, vec in zip(symbols, frame):
                f.write(
                    f"{sym:>3s} {vec[0]:20.10f} "
                    f"{vec[1]:20.10f} {vec[2]:20.10f}\n"
                )


# ──────────────────────────────────────────────────────────────────────
# I/O — CP2K cell file
# ──────────────────────────────────────────────────────────────────────

def parse_cp2k_cell_file(
    filename: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a CP2K cell file.

    Format::

        # Step  Time [fs]  Ax Ay Az  Bx By Bz  Cx Cy Cz  Volume
        0  0.000  17.0 0.0 0.0  0.0 17.0 0.0  0.0 0.0 17.0  4913.0

    Returns
    -------
    times : ndarray, shape (nsteps,)
    cells : ndarray, shape (nsteps, 3, 3)
        Row-major: cells[i] = [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Cell file not found: {filename}")

    times: list[float] = []
    cells: list[list[float]] = []

    with open(filename) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = s.split()
            # Expect: step time ax ay az bx by bz cx cy cz [volume]
            if len(cols) < 11:
                continue
            try:
                t = float(cols[1])
                cell_vals = [float(cols[i]) for i in range(2, 11)]
            except (ValueError, IndexError):
                continue
            times.append(t)
            cells.append(cell_vals)

    if not times:
        raise ValueError(f"No cell data found in {filename}")

    cells_arr = np.array(cells).reshape(-1, 3, 3)
    logger.info(f"Read {len(times)} cell frames from {filename}")

    return np.array(times), cells_arr


# ──────────────────────────────────────────────────────────────────────
# I/O — CP2K stress tensor file
# ──────────────────────────────────────────────────────────────────────

def parse_cp2k_stress_file(
    filename: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a CP2K stress tensor file (XYZ-like format).

    Format (3 "atoms" per frame = rows of 3x3 tensor)::

           3
         i =        1, time =        0.500, E = ...
          X   sxx   sxy   sxz
          Y   syx   syy   syz
          Z   szx   szy   szz

    Returns
    -------
    times : ndarray, shape (nframes,)
    stress : ndarray, shape (nframes, 3, 3)
    """
    # Re-use the generic XYZ reader — stress is stored as 3x3
    times, _, data, _ = read_xyz_trajectory(filename, extract_energy=False)

    if data.shape[1] != 3:
        raise ValueError(
            f"{filename}: expected 3 rows per frame (3x3 stress tensor), "
            f"got {data.shape[1]}"
        )

    logger.info(f"Read {len(times)} stress frames from {filename}")
    return times, data  # shape (nframes, 3, 3)


def stress_3x3_to_voigt(stress_3x3: np.ndarray) -> np.ndarray:
    """
    Convert (nframes, 3, 3) stress tensor to Voigt notation
    (nframes, 6) as [xx, yy, zz, yz, xz, xy].

    This is the convention used by ASE.
    """
    return np.stack([
        stress_3x3[:, 0, 0],  # xx
        stress_3x3[:, 1, 1],  # yy
        stress_3x3[:, 2, 2],  # zz
        stress_3x3[:, 1, 2],  # yz
        stress_3x3[:, 0, 2],  # xz
        stress_3x3[:, 0, 1],  # xy
    ], axis=1)


# ──────────────────────────────────────────────────────────────────────
# I/O — Extended XYZ writer
# ──────────────────────────────────────────────────────────────────────

def write_extxyz(
    filename: str,
    symbols: list[str],
    positions: np.ndarray,
    forces: np.ndarray,
    energies: np.ndarray,
    lattice: Optional[np.ndarray] = None,
    stress: Optional[np.ndarray] = None,
    pbc: list[bool] = None,
) -> None:
    """
    Write an extended XYZ file for ML training.

    Parameters
    ----------
    filename : str
    symbols : list of str
    positions : ndarray, shape (nframes, natoms, 3) — Angstrom
    forces : ndarray, shape (nframes, natoms, 3) — eV/Angstrom
    energies : ndarray, shape (nframes,) — eV
    lattice : ndarray, shape (nframes, 3, 3) or (3, 3) — Angstrom
        If 2D, same lattice is used for all frames.
    stress : ndarray, shape (nframes, 6) — eV/Angstrom^3, Voigt order
        Optional stress tensor in Voigt notation [xx,yy,zz,yz,xz,xy].
    pbc : list of 3 bools
    """
    if pbc is None:
        pbc = [True, True, True]

    nframes = len(energies)
    natoms = len(symbols)
    pbc_str = " ".join("T" if p else "F" for p in pbc)

    # Handle static vs dynamic lattice
    if lattice is not None:
        if lattice.ndim == 2:
            # Static lattice, broadcast to all frames
            lattice = np.broadcast_to(
                lattice[None, :, :], (nframes, 3, 3)
            ).copy()

    with open(filename, "w") as f:
        for i in range(nframes):
            f.write(f"{natoms}\n")

            # Build comment line
            parts = []

            if lattice is not None:
                L = lattice[i].flatten()
                lat_str = " ".join(f"{v:.10f}" for v in L)
                parts.append(f'Lattice="{lat_str}"')

            parts.append(
                'Properties=species:S:1:pos:R:3:forces:R:3'
            )
            parts.append(f"energy={energies[i]:.10f}")

            if stress is not None:
                s = stress[i]
                s_str = " ".join(f"{v:.10f}" for v in s)
                parts.append(f'stress="{s_str}"')

            parts.append(f'pbc="{pbc_str}"')

            f.write(" ".join(parts) + "\n")

            # Atom lines: symbol  px py pz  fx fy fz
            for j in range(natoms):
                sym = symbols[j]
                px, py, pz = positions[i, j]
                fx, fy, fz = forces[i, j]
                f.write(
                    f"{sym:>3s} {px:20.10f} {py:20.10f} {pz:20.10f}"
                    f" {fx:20.10f} {fy:20.10f} {fz:20.10f}\n"
                )


# ──────────────────────────────────────────────────────────────────────
# I/O — PLUMED header parsing
# ──────────────────────────────────────────────────────────────────────

def parse_fields_header(filename: str) -> Optional[list[str]]:
    """
    Extract field names from PLUMED's ``#! FIELDS`` header line.
    """
    with open(filename) as f:
        for line in f:
            s = line.strip()
            if s.startswith("#! FIELDS"):
                return s.split()[2:]
            if not s.startswith("#") and not s.startswith("@") and s:
                break
    return None


# ──────────────────────────────────────────────────────────────────────
# I/O — PLUMED DUMPFORCES (multi-column)
# ──────────────────────────────────────────────────────────────────────

def parse_cv_forces(
    filename: str,
    ncvs_expected: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Parse PLUMED DUMPFORCES output (multi-column).

    Returns
    -------
    times : ndarray, shape (nsteps,)
    forces : ndarray, shape (nsteps, ncvs)
    cv_names : list of str
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"PLUMED force file not found: {filename}")

    fields = parse_fields_header(filename)
    cv_names = fields[1:] if fields is not None else None

    times: list[float] = []
    rows: list[list[float]] = []

    with open(filename) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("@"):
                continue
            cols = s.split()
            times.append(float(cols[0]))
            rows.append([float(c) for c in cols[1:]])

    if not times:
        raise ValueError(f"No data rows found in {filename}")

    forces = np.array(rows)
    ncvs = forces.shape[1]

    if cv_names is None:
        cv_names = [f"cv{i + 1}" for i in range(ncvs)]

    if len(cv_names) != ncvs:
        raise ValueError(
            f"{filename}: FIELDS header lists {len(cv_names)} CVs "
            f"but data has {ncvs} value columns"
        )

    if ncvs_expected is not None and ncvs != ncvs_expected:
        raise ValueError(
            f"{filename}: expected {ncvs_expected} CV columns, found {ncvs}. "
            f"Number of --cv-atoms groups must match columns in {filename}."
        )

    logger.info(
        f"Read {len(times)} steps x {ncvs} CVs "
        f"({', '.join(cv_names)}) from {filename}"
    )
    return np.array(times), forces, cv_names


# ──────────────────────────────────────────────────────────────────────
# I/O — PLUMED DUMPDERIVATIVES (multi-column)
# ──────────────────────────────────────────────────────────────────────

def parse_cv_derivs(
    filename: str,
    cv_specs: list[CVSpec],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Parse PLUMED DUMPDERIVATIVES output (multi-column).

    Returns
    -------
    times : ndarray, shape (nframes,)
    derivs_per_cv : list of ndarray, each (nframes, natoms_in_cv_k, 3)
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f"PLUMED derivatives file not found: {filename}"
        )

    ncvs = len(cv_specs)

    # Read all rows, grouped by time
    frame_groups: list[dict[int, list[float]]] = []
    frame_times: list[float] = []
    current_time: Optional[float] = None
    current_vals: dict[int, list[float]] = {}
    total_params_per_frame = 0

    with open(filename) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("@"):
                continue

            cols = s.split()
            time = float(cols[0])
            param = int(cols[1])
            values = [float(c) for c in cols[2:]]

            if len(values) != ncvs:
                raise ValueError(
                    f"{filename}: at time={time}, parameter={param}: "
                    f"expected {ncvs} CV columns but got {len(values)}. "
                    f"Check that --cv-atoms has the right number of entries."
                )

            if current_time is None:
                current_time = time

            if time != current_time:
                total_params_per_frame = len(current_vals)
                frame_times.append(current_time)
                frame_groups.append(current_vals)
                current_vals = {}
                current_time = time

            current_vals[param] = values

    if current_vals:
        total_params_per_frame = len(current_vals)
        frame_times.append(current_time)
        frame_groups.append(current_vals)

    if not frame_times:
        raise ValueError(f"No derivative frames found in {filename}")

    nframes = len(frame_times)

    # Extract atomic derivatives per CV
    derivs_per_cv: list[np.ndarray] = []

    for k, cv in enumerate(cv_specs):
        needed = 3 * cv.natoms
        cv_derivs = np.zeros((nframes, cv.natoms, 3))

        for fi, group in enumerate(frame_groups):
            for p in range(needed):
                if p not in group:
                    raise ValueError(
                        f"{filename}: at time={frame_times[fi]}, "
                        f"missing parameter {p} for CV '{cv.name}'. "
                        f"Expected parameters 0..{needed - 1} for "
                        f"{cv.natoms} atoms."
                    )
                atom_local = p // 3
                xyz_idx = p % 3
                cv_derivs[fi, atom_local, xyz_idx] = group[p][k]

        derivs_per_cv.append(cv_derivs)

    max_needed = max(3 * cv.natoms for cv in cv_specs)
    n_cell = total_params_per_frame - max_needed

    logger.info(
        f"Read {nframes} derivative frames x {ncvs} CVs from {filename} "
        f"({total_params_per_frame} params/frame: {max_needed} atomic + "
        f"{max(0, n_cell)} cell ignored)"
    )

    return np.array(frame_times), derivs_per_cv


# ──────────────────────────────────────────────────────────────────────
# Time alignment
# ──────────────────────────────────────────────────────────────────────

def align_by_time(
    xyz_times: np.ndarray,
    xyz_data: np.ndarray,
    plumed_force_times: np.ndarray,
    plumed_deriv_times: np.ndarray,
    forces_all: np.ndarray,
    derivs_all: list[np.ndarray],
    tol: float = 1e-8,
    # Optional extras to co-align
    pos_times: Optional[np.ndarray] = None,
    pos_data: Optional[np.ndarray] = None,
    pos_energies: Optional[np.ndarray] = None,
    cell_times: Optional[np.ndarray] = None,
    cell_data: Optional[np.ndarray] = None,
    stress_times: Optional[np.ndarray] = None,
    stress_data: Optional[np.ndarray] = None,
) -> dict:
    """
    Align all data sources by time, returning only matched frames.

    Returns a dict with keys:
        times, forces, cv_forces, cv_derivs, matched_idx,
        positions (optional), energies (optional),
        cells (optional), stress (optional)
    """
    # Check force/deriv time consistency
    n_check = min(len(plumed_force_times), len(plumed_deriv_times))
    if not np.allclose(
        plumed_force_times[:n_check],
        plumed_deriv_times[:n_check],
        atol=tol, rtol=0,
    ):
        diffs = np.abs(
            plumed_force_times[:n_check] - plumed_deriv_times[:n_check]
        )
        idx = int(np.argmax(diffs > tol))
        raise ValueError(
            f"cv_force and cv_derivs times diverge near index {idx}: "
            f"force_t={plumed_force_times[idx]}, "
            f"deriv_t={plumed_deriv_times[idx]}"
        )

    if len(plumed_force_times) != len(plumed_deriv_times):
        raise ValueError(
            f"cv_force has {len(plumed_force_times)} steps but "
            f"cv_derivs has {len(plumed_deriv_times)} steps"
        )

    # Build lookup maps
    plumed_map = {
        round(t, 8): i for i, t in enumerate(plumed_force_times)
    }
    pos_map = (
        {round(t, 8): i for i, t in enumerate(pos_times)}
        if pos_times is not None
        else None
    )
    cell_map = (
        {round(t, 8): i for i, t in enumerate(cell_times)}
        if cell_times is not None
        else None
    )
    stress_map = (
        {round(t, 8): i for i, t in enumerate(stress_times)}
        if stress_times is not None
        else None
    )

    # Collect matches
    out: dict[str, list] = {
        "times": [], "forces": [], "cv_forces": [],
        "matched_idx": [],
    }
    out_derivs = [[] for _ in derivs_all]

    if pos_map is not None:
        out["positions"] = []
        out["energies"] = []
    if cell_map is not None:
        out["cells"] = []
    if stress_map is not None:
        out["stress"] = []

    for i, t in enumerate(xyz_times):
        key = round(t, 8)

        # Must match PLUMED
        if key not in plumed_map:
            continue
        j_pl = plumed_map[key]

        # If pos file given, must also match there
        if pos_map is not None and key not in pos_map:
            continue
        # cell and stress are optional per-frame
        if cell_map is not None and key not in cell_map:
            continue
        if stress_map is not None and key not in stress_map:
            continue

        out["times"].append(t)
        out["forces"].append(xyz_data[i])
        out["cv_forces"].append(forces_all[j_pl])
        out["matched_idx"].append(i)
        for k, dv in enumerate(derivs_all):
            out_derivs[k].append(dv[j_pl])

        if pos_map is not None:
            j_pos = pos_map[key]
            out["positions"].append(pos_data[j_pos])
            if pos_energies is not None:
                out["energies"].append(pos_energies[j_pos])

        if cell_map is not None:
            out["cells"].append(cell_data[cell_map[key]])

        if stress_map is not None:
            out["stress"].append(stress_data[stress_map[key]])

    if not out["times"]:
        xyz_t0 = xyz_times[0]
        xyz_t1 = xyz_times[1] if len(xyz_times) > 1 else xyz_t0
        pl_t0 = plumed_force_times[0]
        pl_t1 = (
            plumed_force_times[1]
            if len(plumed_force_times) > 1
            else pl_t0
        )
        raise ValueError(
            f"No matching times found between data sources.\n"
            f"  XYZ force times start: {xyz_t0}, {xyz_t1}, ...\n"
            f"  PLUMED times start:    {pl_t0}, {pl_t1}, ...\n"
            f"Common cause: CP2K writes frame 0 at t=0.0 but PLUMED "
            f"starts at the first MD step (e.g. t=0.5)."
        )

    # Convert to arrays
    result = {
        "times": np.array(out["times"]),
        "forces": np.array(out["forces"]),
        "cv_forces": np.array(out["cv_forces"]),
        "cv_derivs": [np.array(d) for d in out_derivs],
        "matched_idx": np.array(out["matched_idx"], dtype=int),
    }
    if "positions" in out:
        result["positions"] = np.array(out["positions"])
    if "energies" in out and out["energies"]:
        result["energies"] = np.array(out["energies"])
    if "cells" in out:
        result["cells"] = np.array(out["cells"])
    if "stress" in out:
        result["stress"] = np.array(out["stress"])

    n_skip_xyz = len(xyz_times) - len(result["times"])
    n_skip_pl = len(plumed_force_times) - len(result["times"])
    logger.info(
        f"Time alignment: {len(result['times'])} matched frames "
        f"(skipped {n_skip_xyz} XYZ, {n_skip_pl} PLUMED)"
    )

    return result


# ──────────────────────────────────────────────────────────────────────
# Bias force construction (delegates to bias_correction)
# ──────────────────────────────────────────────────────────────────────

def compute_total_bias(
    cv_specs: list[CVSpec],
    fcv_all: np.ndarray,
    dsdr_list: list[np.ndarray],
    natoms_total: int,
    nframes: int,
) -> np.ndarray:
    """Convert 1-based CVSpec atom indices to 0-based and call bias_correction."""
    atom_indices_list = [
        [a - 1 for a in cv.atom_indices] for cv in cv_specs
    ]
    return compute_bias_forces(
        f_cv=fcv_all,
        ds_dR_list=dsdr_list,
        atom_indices_list=atom_indices_list,
        natoms_total=natoms_total,
    )


# ──────────────────────────────────────────────────────────────────────
# Diagnostics (uses bias_correction, adds logging)
# ──────────────────────────────────────────────────────────────────────

def run_diagnostics(
    cv_specs: list[CVSpec],
    dsdr_list: list[np.ndarray],
    f_bias: np.ndarray,
    f_unbiased: np.ndarray,
    f_total: np.ndarray,
) -> None:
    """Run sanity checks and log results."""
    logger.info("=== Diagnostics ===")

    stats = correction_summary(f_total, f_bias)
    logger.info(
        f"  |F_bias|  : mean={stats['bias_norm_mean']:.6f}, "
        f"max={stats['bias_norm_max']:.6f}"
    )
    logger.info(
        f"  |F_total| : mean={stats['total_norm_mean']:.6f}, "
        f"max={stats['total_norm_max']:.6f}"
    )
    logger.info(f"  Bias/Total ratio : {stats['bias_to_total_ratio']:.4f}")

    logger.info("  Top 5 atoms by max |bias force| (1-based):")
    for atom_idx, max_val in stats["top_atoms"]:
        logger.info(f"    atom {atom_idx + 1:4d}: {max_val:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Frame selection
# ──────────────────────────────────────────────────────────────────────

def apply_frame_selection_dict(
    aligned: dict,
    start: int = 0,
    end: Optional[int] = None,
    stride: int = 1,
) -> dict:
    """Apply start/end/stride to all arrays in the alignment dict."""
    n = len(aligned["times"])
    end_idx = min(end, n) if end is not None else n
    sel = slice(start, end_idx, stride)

    result = {}
    for key, val in aligned.items():
        if key == "cv_derivs":
            result[key] = [arr[sel] for arr in val]
        elif isinstance(val, np.ndarray):
            result[key] = val[sel]
        else:
            result[key] = val

    return result


# ──────────────────────────────────────────────────────────────────────
# Config from YAML
# ──────────────────────────────────────────────────────────────────────

def load_config_yaml(path: str) -> Config:
    """Load configuration from a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for --config. "
            "Install with: pip install pyyaml"
        )

    with open(path) as f:
        d = yaml.safe_load(f)

    cvs = []
    for cv_entry in d.get("cvs", []):
        atoms = cv_entry["atoms"]
        if isinstance(atoms, str):
            atoms = [int(x) for x in atoms.split(",")]
        name = cv_entry.get("name", f"cv{len(cvs) + 1}")
        cvs.append(CVSpec(name=name, atom_indices=atoms))

    out = d.get("output", {})
    frames = d.get("frames", {})
    stress_cfg = d.get("stress", {})
    lattice_raw = d.get("lattice", None)
    lattice = [float(x) for x in lattice_raw] if lattice_raw else None
    pbc_raw = d.get("pbc", [True, True, True])
    pbc = [bool(x) for x in pbc_raw]

    return Config(
        frc_file=d["frc"],
        cv_force_file=d["cv_force"],
        cv_derivs_file=d["cv_derivs"],
        cvs=cvs,
        pos_file=d.get("pos", None),
        stress_file=stress_cfg.get("file", None) if isinstance(stress_cfg, dict) else None,
        stress_unit=stress_cfg.get("unit", "au") if isinstance(stress_cfg, dict) else "au",
        cell_file=d.get("cell_file", None),
        lattice=lattice,
        pbc=pbc,
        out_extxyz=out.get("extxyz", None),
        out_xyz=out.get("xyz", "forces_unbiased.xyz"),
        out_flat=out.get("flat", "forces_unbiased.dat"),
        out_bias_xyz=out.get("bias_xyz", None),
        frame_start=frames.get("start", 0),
        frame_end=frames.get("end", None),
        frame_stride=frames.get("stride", 1),
        time_tol=d.get("time_tolerance", 1e-8),
        validate=d.get("validate", True),
    )


# ──────────────────────────────────────────────────────────────────────
# Config from CLI
# ──────────────────────────────────────────────────────────────────────

def build_config_from_args(args: argparse.Namespace) -> Config:
    """Build Config from parsed CLI arguments."""
    if not args.cv_atoms:
        raise ValueError(
            "No CV atoms specified. Use --cv-atoms to define which atoms "
            "each CV acts on, e.g. --cv-atoms 1,4 1,2"
        )

    cv_names = None
    fields = parse_fields_header(args.cv_force)
    if fields is not None:
        cv_names = fields[1:]

    cvs = []
    for i, atoms_str in enumerate(args.cv_atoms):
        atoms = [int(x) for x in atoms_str.split(",")]
        name = (
            cv_names[i]
            if cv_names and i < len(cv_names)
            else f"cv{i + 1}"
        )
        cvs.append(CVSpec(name=name, atom_indices=atoms))

    if cv_names and len(cvs) != len(cv_names):
        logger.warning(
            f"--cv-atoms provides {len(cvs)} CVs but force file header "
            f"lists {len(cv_names)}: {cv_names}. "
            f"Make sure the ordering matches."
        )

    lattice = None
    if args.lattice:
        if len(args.lattice) == 9:
            lattice = args.lattice
        elif len(args.lattice) == 3:
            # Shorthand: a b c for orthorhombic cell
            a, b, c = args.lattice
            lattice = [a, 0, 0, 0, b, 0, 0, 0, c]
        else:
            raise ValueError(
                f"--lattice expects 3 (orthorhombic) or 9 values, "
                f"got {len(args.lattice)}"
            )

    pbc = [True, True, True]
    if args.no_pbc:
        pbc = [False, False, False]

    return Config(
        frc_file=args.frc,
        cv_force_file=args.cv_force,
        cv_derivs_file=args.cv_derivs,
        cvs=cvs,
        pos_file=args.pos,
        stress_file=args.stress,
        stress_unit=args.stress_unit,
        cell_file=args.cell_file,
        lattice=lattice,
        pbc=pbc,
        out_extxyz=args.out_extxyz,
        out_xyz=args.out_xyz,
        out_flat=args.out_flat,
        out_bias_xyz=args.out_bias_xyz,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        time_tol=args.time_tol,
        validate=not args.no_validate,
    )


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove PLUMED bias forces from CP2K trajectory forces\n"
            "and produce ML-ready extended XYZ training data.\n\n"
            "PLUMED writes all CV data as columns in shared files\n"
            "(DUMPFORCES, DUMPDERIVATIVES). Each --cv-atoms entry\n"
            "maps to one column, in the same order as PLUMED's ARG=."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # Produce extxyz for ML training (2 CVs):\n"
            "  %(prog)s --frc btd-frc-1.xyz --pos btd-pos-1.xyz \\\n"
            "      --cv-force cv_force --cv-derivs cv_derivs \\\n"
            "      --cv-atoms 1,4 1,2 \\\n"
            "      --lattice 17.0 0 0  0 17.0 0  0 0 17.0 \\\n"
            "      --out-extxyz train.xyz\n\n"
            "  # With stress and cell file:\n"
            "  %(prog)s --frc btd-frc-1.xyz --pos btd-pos-1.xyz \\\n"
            "      --cv-force cv_force --cv-derivs cv_derivs \\\n"
            "      --cv-atoms 1,4 1,2 \\\n"
            "      --cell-file btd-1.cell \\\n"
            "      --stress btd-stress-1.stress_tensor --stress-unit au \\\n"
            "      --out-extxyz train.xyz\n\n"
            "  # Orthorhombic cell shorthand:\n"
            "  %(prog)s ... --lattice 17.0 17.0 17.0\n\n"
            "  # From config file:\n"
            "  %(prog)s --config run.yaml\n\n"
            "unit conversions:\n"
            "  Energy : Hartree -> eV  (1 Ha = 27.2114 eV)\n"
            "  Forces : assumed eV/A (PLUMED UNITS LENGTH=A ENERGY=eV)\n"
            "  Stress : --stress-unit au|eV/A^3|GPa -> eV/A^3\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--config", metavar="YAML",
        help="YAML configuration file (alternative to CLI flags)",
    )
    mode.add_argument(
        "--frc", metavar="FILE",
        help="Biased force XYZ trajectory from CP2K",
    )

    cp2k = parser.add_argument_group("CP2K inputs")
    cp2k.add_argument(
        "--pos", metavar="FILE",
        help=(
            "CP2K position XYZ trajectory (has energy in Hartree). "
            "Required for --out-extxyz."
        ),
    )
    cp2k.add_argument(
        "--stress", metavar="FILE",
        help=(
            "CP2K stress tensor file (XYZ-like, 3x3 per frame). "
            "Optional."
        ),
    )
    cp2k.add_argument(
        "--stress-unit",
        default="au",
        choices=["au", "eV/A^3", "GPa"],
        help="Unit of stress in the input file (default: au)",
    )

    plumed = parser.add_argument_group("PLUMED inputs")
    plumed.add_argument(
        "--cv-force", metavar="FILE",
        help="PLUMED DUMPFORCES file (columns: time cv1 cv2 ...)",
    )
    plumed.add_argument(
        "--cv-derivs", metavar="FILE",
        help="PLUMED DUMPDERIVATIVES file",
    )
    plumed.add_argument(
        "--cv-atoms", nargs="+", metavar="ATOMS",
        help=(
            "Comma-separated 1-based atom indices for each CV, "
            "one entry per CV in ARG= order. "
            "Example: --cv-atoms 1,4 1,2"
        ),
    )

    cell = parser.add_argument_group("cell / lattice")
    cell_ex = cell.add_mutually_exclusive_group()
    cell_ex.add_argument(
        "--lattice", type=float, nargs="+", metavar="V",
        help=(
            "Lattice vectors: 9 values (ax ay az bx by bz cx cy cz) "
            "or 3 values (a b c) for orthorhombic. In Angstrom."
        ),
    )
    cell_ex.add_argument(
        "--cell-file", metavar="FILE",
        help="CP2K cell file (time-dependent lattice)",
    )
    cell.add_argument(
        "--no-pbc", action="store_true",
        help="Set pbc to F F F (default: T T T)",
    )

    out = parser.add_argument_group("output options")
    out.add_argument(
        "--out-extxyz", metavar="FILE",
        help=(
            "Output extended XYZ with positions, corrected forces, "
            "energy (eV), and optionally stress — ready for ML training"
        ),
    )
    out.add_argument(
        "--out-xyz", default="forces_unbiased.xyz",
        help="Output XYZ with corrected forces (default: forces_unbiased.xyz)",
    )
    out.add_argument(
        "--out-flat", default="forces_unbiased.dat",
        help="Output flat array, nframes x 3N (default: forces_unbiased.dat)",
    )
    out.add_argument(
        "--out-bias-xyz", default=None,
        help="Write bias forces to XYZ for inspection",
    )

    sel = parser.add_argument_group("frame selection")
    sel.add_argument(
        "--frame-start", type=int, default=0,
        help="First matched frame to include (default: 0)",
    )
    sel.add_argument(
        "--frame-end", type=int, default=None,
        help="Last frame, exclusive (default: all)",
    )
    sel.add_argument(
        "--frame-stride", type=int, default=1,
        help="Frame stride (default: 1)",
    )

    misc = parser.add_argument_group("miscellaneous")
    misc.add_argument(
        "--time-tol", type=float, default=1e-8,
        help="Tolerance for time matching (default: 1e-8)",
    )
    misc.add_argument(
        "--no-validate", action="store_true",
        help="Skip diagnostic checks",
    )
    misc.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v info, -vv debug)",
    )
    misc.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress all output except errors",
    )

    return parser


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Logging ──
    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose >= 1:
        level = logging.INFO
    else:
        level = logging.INFO

    logging.basicConfig(format="%(message)s", level=level, stream=sys.stderr)

    # ── Load config ──
    if args.config:
        cfg = load_config_yaml(args.config)
    else:
        if not args.cv_force or not args.cv_derivs:
            parser.error(
                "--cv-force and --cv-derivs are required when not "
                "using --config"
            )
        cfg = build_config_from_args(args)

    # Validate extxyz requirements
    if cfg.out_extxyz and not cfg.pos_file:
        parser.error(
            "--pos is required when using --out-extxyz "
            "(need coordinates and energy for extxyz output)"
        )

    logger.info(cfg.summary())
    logger.info("")

    # ── Read XYZ forces ──
    frc_times, symbols, f_total, _ = read_xyz_trajectory(
        cfg.frc_file, extract_energy=False
    )

    # ── Read position file (optional, for extxyz) ──
    pos_times = pos_data = pos_energies = None
    if cfg.pos_file:
        pos_times, pos_symbols, pos_data, pos_energies = (
            read_xyz_trajectory(cfg.pos_file, extract_energy=True)
        )
        if pos_symbols != symbols:
            raise ValueError(
                f"Atom symbols differ between force file and position file: "
                f"{symbols[:3]}... vs {pos_symbols[:3]}..."
            )
        if pos_energies is not None:
            logger.info(
                f"  Energy range (Hartree): "
                f"{np.min(pos_energies):.6f} to {np.max(pos_energies):.6f}"
            )

    # ── Read PLUMED forces ──
    force_times, forces_all, cv_names = parse_cv_forces(
        cfg.cv_force_file, ncvs_expected=cfg.ncvs,
    )
    for k, cv in enumerate(cfg.cvs):
        if cv.name.startswith("cv") and k < len(cv_names):
            cv.name = cv_names[k]

    # ── Read PLUMED derivatives ──
    deriv_times, derivs_per_cv = parse_cv_derivs(
        cfg.cv_derivs_file, cfg.cvs,
    )

    # ── Read cell file (optional) ──
    cell_times = cell_data = None
    if cfg.cell_file:
        cell_times, cell_data = parse_cp2k_cell_file(cfg.cell_file)
    elif cfg.lattice:
        # Static lattice — will be handled after alignment
        pass

    # ── Read stress file (optional) ──
    stress_times = stress_data = None
    if cfg.stress_file:
        stress_times, stress_data = parse_cp2k_stress_file(cfg.stress_file)

    # ── Align all data by time ──
    aligned = align_by_time(
        frc_times, f_total,
        force_times, deriv_times, forces_all, derivs_per_cv,
        tol=cfg.time_tol,
        pos_times=pos_times, pos_data=pos_data,
        pos_energies=pos_energies,
        cell_times=cell_times, cell_data=cell_data,
        stress_times=stress_times, stress_data=stress_data,
    )

    # ── Frame selection ──
    aligned = apply_frame_selection_dict(
        aligned,
        start=cfg.frame_start,
        end=cfg.frame_end,
        stride=cfg.frame_stride,
    )

    times = aligned["times"]
    f_tot_al = aligned["forces"]
    fcv_al = aligned["cv_forces"]
    dsdr_al = aligned["cv_derivs"]
    nframes = len(times)
    natoms = f_tot_al.shape[1]

    logger.info(f"After frame selection: {nframes} frames, {natoms} atoms")

    # ── Compute bias and correct ──
    f_bias = compute_total_bias(
        cfg.cvs, fcv_al, dsdr_al, natoms, nframes,
    )
    f_unbiased = remove_bias(f_tot_al, f_bias)

    logger.info("")
    logger.info("=== Shapes ===")
    logger.info(f"  F_total    : {f_tot_al.shape}")
    logger.info(f"  F_bias     : {f_bias.shape}")
    logger.info(f"  F_unbiased : {f_unbiased.shape}")
    for k, cv in enumerate(cfg.cvs):
        logger.info(
            f"  d({cv.name})/dR : {dsdr_al[k].shape}  "
            f"(atoms: {cv.atom_indices})"
        )

    # ── Diagnostics ──
    if cfg.validate:
        logger.info("")
        run_diagnostics(
            cfg.cvs, dsdr_al, f_bias, f_unbiased, f_tot_al,
        )

    # ── Write outputs ──
    logger.info("")

    # Force-only XYZ
    write_xyz_vectors(cfg.out_xyz, symbols, times, f_unbiased)
    logger.info(f"Wrote corrected forces -> {cfg.out_xyz}")

    # Flat array
    np.savetxt(
        cfg.out_flat,
        f_unbiased.reshape(nframes, -1),
        fmt="%.10f",
    )
    logger.info(f"Wrote flat array       -> {cfg.out_flat}")

    # Bias forces (optional)
    if cfg.out_bias_xyz:
        write_xyz_vectors(cfg.out_bias_xyz, symbols, times, f_bias)
        logger.info(f"Wrote bias forces      -> {cfg.out_bias_xyz}")

    # ── Extended XYZ for ML ──
    if cfg.out_extxyz:
        # Convert energy: Hartree -> eV
        energies_ev = aligned["energies"] * HARTREE_TO_EV
        logger.info(
            f"  Energy converted: Hartree -> eV "
            f"(1 Ha = {HARTREE_TO_EV:.6f} eV)"
        )
        logger.info(
            f"  Energy range (eV): "
            f"{np.min(energies_ev):.6f} to {np.max(energies_ev):.6f}"
        )

        # Resolve lattice
        lattice_for_extxyz = None
        if "cells" in aligned:
            lattice_for_extxyz = aligned["cells"]  # (nframes, 3, 3)
        elif cfg.lattice:
            lattice_for_extxyz = np.array(cfg.lattice).reshape(3, 3)

        # Convert stress if present
        stress_voigt = None
        if "stress" in aligned:
            conv = stress_conversion_factor(cfg.stress_unit)
            stress_ev_a3 = aligned["stress"] * conv
            stress_voigt = stress_3x3_to_voigt(stress_ev_a3)
            logger.info(
                f"  Stress converted: {cfg.stress_unit} -> eV/A^3 "
                f"(factor = {conv:.6f})"
            )

        write_extxyz(
            cfg.out_extxyz,
            symbols=symbols,
            positions=aligned["positions"],
            forces=f_unbiased,
            energies=energies_ev,
            lattice=lattice_for_extxyz,
            stress=stress_voigt,
            pbc=cfg.pbc,
        )
        logger.info(f"Wrote extxyz           -> {cfg.out_extxyz}")

    logger.info("")
    logger.info(
        f"Done. {nframes} frames corrected across "
        f"{cfg.ncvs} CV(s): "
        + ", ".join(cv.name for cv in cfg.cvs)
        + "."
    )


if __name__ == "__main__":
    main()
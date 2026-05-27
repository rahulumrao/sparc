"""
ComputeBias.py — Recover physical forces from biased MD simulations.

This module contains the core physics for removing bias forces introduced
by enhanced sampling methods (metadynamics, umbrella sampling, etc.)
acting through collective variables (CVs).


The problem
-----------
In a biased simulation the force on each atom is not the true physical
force. The MD engine (e.g. CP2K) reports:

    F_total(t) = F_physical(t) + F_bias(t)

We want F_physical for training machine-learning interatomic potentials.


How the bias acts
-----------------
The bias potential V_bias depends on atomic positions only through
collective variables s_k(R):

    V_bias = V_bias(s_1, s_2, ..., s_K)

The bias force on atom i from all K collective variables follows from
the chain rule:

                    K
    F_bias_i(t) = SUM   f_k(t)  *  (ds_k/dR_i)(t)
                   k=1

where:

    f_k(t) = -dV_bias/ds_k      scalar force on CV k (from DUMPFORCES)
    ds_k/dR_i                    how CV k changes when atom i moves
                                 (from DUMPDERIVATIVES)

This is general — it works for any CV type (distance, angle, dihedral,
coordination number, RMSD, path CV, etc.). The specific geometry of
the CV is encoded entirely in the derivatives ds_k/dR_i that PLUMED
provides. No CV-specific logic is needed here.

Most atoms are not involved in any CV, so their bias force is zero.
If an atom appears in multiple CVs, contributions are summed.


Recovering the physical forces
------------------------------
    F_physical(t) = F_total(t) - F_bias(t)


Unit conventions
----------------
This module is unit-agnostic — it performs only multiplication and
subtraction. In the typical CP2K + PLUMED workflow with
UNITS LENGTH=A ENERGY=eV:

    Forces (CP2K, after conversion) : eV/Angstrom
    PLUMED cv_force                 : eV/Angstrom
    PLUMED cv_derivs                : dimensionless
    Bias force = cv_force * cv_derivs : eV/Angstrom


Functions
---------
    compute_bias_forces  — F_bias from CV forces and derivatives
    remove_bias          — F_physical = F_total - F_bias
    correction_summary   — statistics on the bias correction
"""

from __future__ import annotations

import numpy as np


def compute_bias_forces(
    f_cv: np.ndarray,
    ds_dR_list: list[np.ndarray],
    atom_indices_list: list[list[int]],
    natoms_total: int,
) -> np.ndarray:
    """
    Compute bias forces on all atoms from one or more collective variables.

    Applies the chain rule:

                        K
        F_bias_i(t) = SUM   f_k(t)  *  (ds_k / dR_i)(t)
                       k=1

    Parameters
    ----------
    f_cv : ndarray, shape (nframes, n_cvs)
        Scalar force on each CV at each timestep.
        Column k is f_k(t) = -dV_bias/ds_k.
        For a single CV, shape (nframes, 1).

    ds_dR_list : list of ndarray
        One array per CV. Entry k has shape (nframes, natoms_in_cv_k, 3).
        These are the derivatives of CV k with respect to each of its
        constituent atoms' positions.

    atom_indices_list : list of list of int
        One list per CV. Entry k gives 0-based global atom indices
        for the atoms involved in CV k.
        Length of entry k must match natoms_in_cv_k.

    natoms_total : int
        Total number of atoms in the system.

    Returns
    -------
    f_bias : ndarray, shape (nframes, natoms_total, 3)
        Bias force on every atom. Atoms not in any CV get zeros.

    Example
    -------
    >>> # Two CVs: d1 between atoms 0,3  d2 between atoms 0,1
    >>> nframes = 100
    >>> f_cv = np.random.randn(nframes, 2)        # 2 CVs
    >>> ds_dR_d1 = np.random.randn(nframes, 2, 3)  # 2 atoms in d1
    >>> ds_dR_d2 = np.random.randn(nframes, 2, 3)  # 2 atoms in d2
    >>> f_bias = compute_bias_forces(
    ...     f_cv,
    ...     [ds_dR_d1, ds_dR_d2],
    ...     [[0, 3], [0, 1]],
    ...     natoms_total=10
    ... )
    >>> f_bias.shape
    (100, 10, 3)
    """
    nframes, n_cvs = f_cv.shape

    if len(ds_dR_list) != n_cvs:
        raise ValueError(
            f"f_cv has {n_cvs} columns but ds_dR_list has {len(ds_dR_list)} entries"
        )
    if len(atom_indices_list) != n_cvs:
        raise ValueError(
            f"f_cv has {n_cvs} columns but "
            f"atom_indices_list has {len(atom_indices_list)} entries"
        )

    f_bias = np.zeros((nframes, natoms_total, 3))

    for k in range(n_cvs):
        ds_dR_k = ds_dR_list[k]
        indices_k = atom_indices_list[k]
        f_k = f_cv[:, k]  # (nframes,)

        if len(indices_k) != ds_dR_k.shape[1]:
            raise ValueError(
                f"CV {k}: atom_indices has {len(indices_k)} entries but "
                f"ds_dR has {ds_dR_k.shape[1]} atoms"
            )

        for local_idx, global_idx in enumerate(indices_k):
            #
            # f_k[:, None]              shape: (nframes, 1)
            # ds_dR_k[:, local_idx, :]  shape: (nframes, 3)
            #
            # Product:  f_k(t) * [ds_k/dx_i, ds_k/dy_i, ds_k/dz_i]
            # Result:   (nframes, 3) — bias force on atom i from CV k
            #
            f_bias[:, global_idx, :] += f_k[:, None] * ds_dR_k[:, local_idx, :]

    return f_bias


def remove_bias(
    f_total: np.ndarray,
    f_bias: np.ndarray,
) -> np.ndarray:
    """
    Recover physical forces by subtracting the bias.

        F_physical = F_total - F_bias

    Parameters
    ----------
    f_total : ndarray, shape (nframes, natoms, 3)
        Total forces from the biased simulation.

    f_bias : ndarray, shape (nframes, natoms, 3)
        Reconstructed bias forces (from ``compute_bias_forces``).

    Returns
    -------
    f_physical : ndarray, shape (nframes, natoms, 3)
        The true physical forces, suitable for ML training.
    """
    if f_total.shape != f_bias.shape:
        raise ValueError(
            f"Shape mismatch: f_total {f_total.shape} vs f_bias {f_bias.shape}"
        )
    return f_total - f_bias


def correction_summary(
    f_total: np.ndarray,
    f_bias: np.ndarray,
) -> dict:
    """
    Compute summary statistics about the bias correction.

    Parameters
    ----------
    f_total : ndarray, shape (nframes, natoms, 3)
    f_bias : ndarray, shape (nframes, natoms, 3)

    Returns
    -------
    stats : dict
        bias_norm_mean      : float — mean |F_bias| over frames
        bias_norm_max       : float — max |F_bias| over frames
        total_norm_mean     : float — mean |F_total| over frames
        total_norm_max      : float — max |F_total| over frames
        bias_to_total_ratio : float — mean( |F_bias| / |F_total| )
        top_atoms           : list of (atom_index_0based, max_component)
                              for the 5 most-affected atoms
    """
    nframes = f_total.shape[0]

    bias_mag = np.linalg.norm(f_bias.reshape(nframes, -1), axis=1)
    total_mag = np.linalg.norm(f_total.reshape(nframes, -1), axis=1)

    per_atom_max = np.max(np.abs(f_bias), axis=(0, 2))
    top_idx = np.argsort(per_atom_max)[-5:][::-1]
    top_atoms = [
        (int(a), float(per_atom_max[a])) for a in top_idx if per_atom_max[a] > 0
    ]

    return {
        "bias_norm_mean": float(np.mean(bias_mag)),
        "bias_norm_max": float(np.max(bias_mag)),
        "total_norm_mean": float(np.mean(total_mag)),
        "total_norm_max": float(np.max(total_mag)),
        "bias_to_total_ratio": float(np.mean(bias_mag / (total_mag + 1e-30))),
        "top_atoms": top_atoms,
    }

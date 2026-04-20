"""
Pure math for removing PLUMED bias forces from MD trajectories.

The bias force on atom i from K collective variables:

    F_bias_i = SUM_k  f_k * (ds_k / dR_i)

where f_k = -dV_bias/ds_k  (scalar CV force, from DUMPFORCES)
and ds_k/dR_i               (CV derivative, from DUMPDERIVATIVES)

Physical forces are recovered as:

    F_physical = F_total - F_bias
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

    Parameters
    ----------
    f_cv : ndarray, shape (nframes, n_cvs)
        Scalar force on each CV at each frame (column k = f_k = -dV/ds_k).
    ds_dR_list : list of ndarray, each shape (nframes, natoms_in_cv_k, 3)
        Derivatives ds_k/dR for each CV. Parameter indices are local to
        the CV — entry j corresponds to the j-th atom in atom_indices_list[k].
    atom_indices_list : list of list of int
        0-based global atom indices for each CV, in the same order as the
        local parameter indices in ds_dR_list.
    natoms_total : int
        Total number of atoms in the system.

    Returns
    -------
    f_bias : ndarray, shape (nframes, natoms_total, 3)
        Bias force on every atom. Atoms not in any CV are zero.
    """
    nframes, n_cvs = f_cv.shape

    if len(ds_dR_list) != n_cvs:
        raise ValueError(
            f"f_cv has {n_cvs} columns but ds_dR_list has {len(ds_dR_list)} entries"
        )
    if len(atom_indices_list) != n_cvs:
        raise ValueError(
            f"f_cv has {n_cvs} columns but atom_indices_list has "
            f"{len(atom_indices_list)} entries"
        )

    f_bias = np.zeros((nframes, natoms_total, 3))

    for k in range(n_cvs):
        ds_dR_k = ds_dR_list[k]          # (nframes, natoms_in_cv_k, 3)
        indices_k = atom_indices_list[k]  # 0-based global indices
        f_k = f_cv[:, k]                  # (nframes,)

        if len(indices_k) != ds_dR_k.shape[1]:
            raise ValueError(
                f"CV {k}: atom_indices has {len(indices_k)} entries but "
                f"ds_dR has {ds_dR_k.shape[1]} atoms"
            )

        for local_j, global_j in enumerate(indices_k):
            # f_k[:, None]              shape (nframes, 1)
            # ds_dR_k[:, local_j, :]   shape (nframes, 3)
            # product → (nframes, 3) bias force contribution on atom global_j
            f_bias[:, global_j, :] += f_k[:, None] * ds_dR_k[:, local_j, :]

    return f_bias


def remove_bias(
    f_total: np.ndarray,
    f_bias: np.ndarray,
) -> np.ndarray:
    """
    Recover physical forces: F_physical = F_total - F_bias.

    Parameters
    ----------
    f_total : ndarray, shape (nframes, natoms, 3)
    f_bias  : ndarray, shape (nframes, natoms, 3)

    Returns
    -------
    f_physical : ndarray, shape (nframes, natoms, 3)
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
    Summary statistics for the bias correction.

    Returns
    -------
    dict with keys:
        bias_norm_mean, bias_norm_max,
        total_norm_mean, total_norm_max,
        bias_to_total_ratio,
        top_atoms : list of (0-based atom index, max |bias| component)
    """
    nframes = f_total.shape[0]
    bias_mag  = np.linalg.norm(f_bias.reshape(nframes, -1),  axis=1)
    total_mag = np.linalg.norm(f_total.reshape(nframes, -1), axis=1)

    per_atom_max = np.max(np.abs(f_bias), axis=(0, 2))
    top_idx  = np.argsort(per_atom_max)[-5:][::-1]
    top_atoms = [
        (int(a), float(per_atom_max[a]))
        for a in top_idx
        if per_atom_max[a] > 0
    ]

    return {
        "bias_norm_mean":      float(np.mean(bias_mag)),
        "bias_norm_max":       float(np.max(bias_mag)),
        "total_norm_mean":     float(np.mean(total_mag)),
        "total_norm_max":      float(np.max(total_mag)),
        "bias_to_total_ratio": float(np.mean(bias_mag / (total_mag + 1e-30))),
        "top_atoms":           top_atoms,
    }

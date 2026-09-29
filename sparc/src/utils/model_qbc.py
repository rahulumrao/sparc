#!/usr/bin/python3
# model_qbc.py
################################################################
import numpy as np
from ase.io import read
from sparc.src.utils.utils import flush_memory

################################################################
"""
    Query-by-Committee model deviation for MLIP active learning.

    Reimplements what `dp model-devi` computes, callable in-process instead
    of shelling out to the `dp` CLI. Two evaluation paths, picked by
    ``framework``:

    * ``deepmd`` — batched: one ``DeepPot.eval`` call per model for the
      whole trajectory. Verified against deepmd-kit's own
      ``calc_model_devi_{f,v,e}`` (same formulas, same per-atom
      normalization of energy/virial) and against `dp model-devi` output
      directly (agrees to float32 noise floor, ~1e-6).
    * ``nequip`` / ``allegro`` / ``mace`` — no batched Python eval API
      exists for these, so this loops frame-by-frame through the model's
      ASE calculator. Same deviation formulas, same output format.

    Functions
    ---------
    load_calculator(framework, model_path, device) -> ase.calculators.Calculator
        Return an ASE calculator for one committee member (ASE-loop path).

    devi_f(forces) -> deviation array, one dim fewer than ``forces``
        Force deviation across models. Works for a single frame
        (forces: (n_models, n_atoms, 3) -> (n_atoms,)) or a batch
        (forces: (n_models, n_frames, n_atoms, 3) -> (n_frames, n_atoms)).

    devi_v(virials) -> (max, min, avg)
        Virial deviation across models (virials already divided by natoms).
        Same shape-agnostic convention as devi_f.

    devi_e(energies) -> deviation array
        Per-atom energy deviation across models (energies already /natoms).
        Same shape-agnostic convention as devi_f.

    get_virial(atoms) -> (3, 3) array
        Virial from an ASE Atoms' stress; zeros if the calculator has no
        stress head or the system isn't periodic (ASE-loop path).

    compute_model_devi(traj_file, model_paths, framework, device, out_file)
        Public entry point. Dispatches to the batched deepmd path or the
        ASE-loop path, writes `out_file` in the same 8-column layout as
        `dp model-devi`, and returns the (n_frames, 8) array.
"""
################################################################


def load_calculator(framework: str, model_path: str, device: str = "cpu"):
    """Return an ASE calculator for one ensemble member."""
    framework = framework.lower()
    if framework in ("nequip", "allegro"):
        from nequip.ase import NequIPCalculator

        return NequIPCalculator.from_deployed_model(model_path, device=device)
    if framework == "mace":
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=model_path, device=device)
    if framework == "deepmd":
        from deepmd.calculator import DP

        return DP(model=model_path)
    raise ValueError(
        f"Unknown framework '{framework}': use nequip, allegro, mace, or deepmd"
    )


def devi_f(forces: np.ndarray) -> np.ndarray:
    """forces: (n_models, ..., n_atoms, 3) -> per-atom force deviation, (..., n_atoms)."""
    return np.linalg.norm(np.std(forces, axis=0), axis=-1)


def devi_v(virials_per_atom: np.ndarray):
    """virials_per_atom: (n_models, ..., 9), each already divided by natoms.
    Returns (max, min, avg) exactly as DeePMD reports them -- avg is
    norm(std)/3, not the mean of max/min."""
    v = np.std(virials_per_atom, axis=0)
    return v.max(axis=-1), v.min(axis=-1), np.linalg.norm(v, axis=-1) / 3


def devi_e(energies_per_atom: np.ndarray) -> np.ndarray:
    """energies_per_atom: (n_models, ...) total energy / natoms."""
    return np.std(energies_per_atom, axis=0)


def get_virial(atoms) -> np.ndarray:
    try:
        stress = atoms.get_stress(voigt=False)
        return -stress * atoms.get_volume()
    except Exception:
        # no stress head / non-periodic system: virial deviation is meaningless here
        return np.zeros((3, 3))


def _frames_to_arrays(frames, type_map):
    """Convert fixed-composition ASE frames into DeepPot.eval's coord/box/atype."""
    n_frames = len(frames)
    natoms = len(frames[0])
    coord = np.empty((n_frames, natoms * 3), dtype=np.float64)
    box = np.empty((n_frames, 9), dtype=np.float64)
    has_pbc = bool(np.any(frames[0].pbc))
    symbols0 = frames[0].get_chemical_symbols()
    atype = np.array([type_map.index(s) for s in symbols0], dtype=np.int32)

    for i, atoms in enumerate(frames):
        if len(atoms) != natoms:
            raise ValueError(
                f"Frame {i} has {len(atoms)} atoms, expected {natoms}. "
                "Variable-atom trajectories are not supported by the batched path."
            )
        if atoms.get_chemical_symbols() != symbols0:
            raise ValueError(f"Frame {i} type order differs from frame 0.")
        coord[i] = atoms.get_positions().reshape(-1)
        box[i] = atoms.cell.array.reshape(-1) if has_pbc else 0.0

    return coord, (box if has_pbc else None), atype, natoms


def compute_model_devi(traj_file, model_paths, framework, device, out_file):
    """
    Query-by-Committee model deviation, in-process (no `dp` CLI subprocess).

    Parameters
    ----------
    traj_file : str
        ASE-readable trajectory.
    model_paths : list of str
        Paths to >=2 independently trained models.
    framework : str
        "deepmd" -> batched DeepPot.eval path (one call per model for the
        whole trajectory). "nequip" / "allegro" / "mace" -> frame-by-frame
        ASE-calculator path (no batched Python eval API exists for these).
    device : str
        Passed to the ASE calculator for nequip/allegro/mace; ignored for
        deepmd (DeepPot picks CUDA/CPU on its own).
    out_file : str
        Where to write the model-devi table (same 8-column layout as
        `dp model-devi`).

    Returns
    -------
    np.ndarray, shape (n_frames, 8)
        step, max_devi_v, min_devi_v, avg_devi_v,
        max_devi_f, min_devi_f, avg_devi_f, devi_e.
    """
    if len(model_paths) < 2:
        raise ValueError(
            "Query-by-Committee needs at least 2 independently trained models"
        )

    frames = read(traj_file, index=":")

    if framework.lower() == "deepmd":
        from deepmd.infer.deep_pot import DeepPot

        models = [DeepPot(m, auto_batch_size=True) for m in model_paths]
        type_map = models[0].get_type_map()
        for i, m in enumerate(models[1:], start=2):
            if m.get_type_map() != type_map:
                raise RuntimeError(
                    f"Model {i} type_map mismatch: {m.get_type_map()} vs {type_map}"
                )

        coord, box, atype, natoms = _frames_to_arrays(frames, type_map)
        n_models = len(models)
        n_frames = len(frames)

        forces = np.empty((n_models, n_frames, natoms, 3), dtype=np.float64)
        virials = np.empty((n_models, n_frames, 9), dtype=np.float64)
        energies = np.empty((n_models, n_frames), dtype=np.float64)

        for i, dp in enumerate(models):
            e, f, v = dp.eval(coord, box, atype)[:3]
            energies[i] = np.asarray(e).reshape(n_frames) / natoms
            forces[i] = np.asarray(f).reshape(n_frames, natoms, 3)
            virials[i] = np.asarray(v).reshape(n_frames, 9) / natoms

        f_atom = devi_f(forces)
        v_max, v_min, v_avg = devi_v(virials)
        e_devi = devi_e(energies)

        rows = np.empty((n_frames, 8), dtype=np.float64)
        rows[:, 0] = np.arange(n_frames)
        rows[:, 1] = v_max
        rows[:, 2] = v_min
        rows[:, 3] = v_avg
        rows[:, 4] = f_atom.max(axis=-1)
        rows[:, 5] = f_atom.min(axis=-1)
        rows[:, 6] = f_atom.mean(axis=-1)
        rows[:, 7] = e_devi

        del models
        flush_memory("pytorch")

    else:
        calcs = [load_calculator(framework, m, device) for m in model_paths]

        rows = []
        for step, atoms in enumerate(frames):
            natoms = len(atoms)
            forces, virials, energies = [], [], []
            for calc in calcs:
                a = atoms.copy()
                a.calc = calc
                forces.append(a.get_forces())
                virials.append(get_virial(a).reshape(-1) / natoms)
                energies.append(a.get_potential_energy() / natoms)
            forces = np.array(forces)
            virials = np.array(virials)
            energies = np.array(energies)

            f = devi_f(forces)
            v_max, v_min, v_avg = devi_v(virials)
            e = devi_e(energies)
            rows.append((step, v_max, v_min, v_avg, f.max(), f.min(), f.mean(), e))

        rows = np.array(rows)

        del calcs
        flush_memory("pytorch")

    header = "step  max_devi_v  min_devi_v  avg_devi_v  max_devi_f  min_devi_f  avg_devi_f  devi_e"
    np.savetxt(out_file, rows, fmt=["%12d"] + ["%12.6e"] * 7, header=header)
    return rows


################################################################
# END OF FILE
################################################################

"""
Tests for DeepPotential model loading and ML/MD (sparc/src/deepmd.py, sparc/src/ase_md.py).

Updated for v0.2: filesystem-first model detection, .pth support.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import numpy as np

from ase.io import read
from sparc.src.deepmd import setup_DeepPotential
from sparc.src.ase_md import NoseNVT, LangevinNVT


def test_mlmd_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):

    root_dir = Path(__file__).resolve().parents[1]

    data_dir = root_dir / "tests" / "data" / "mlp"
    model_dir = root_dir / "tests" / "data" / "mlmd"
    traj_file = "AseMD.traj"

    if not (data_dir / traj_file).exists():
        pytest.skip("MLP test data not found")
    if not model_dir.exists():
        pytest.skip("MLMD test data not found")

    # v0.2: detect model extension from filesystem
    model_name = None
    for ext in (".pth", ".pb"):
        candidate = f"frozen_model_1{ext}"
        if (model_dir / candidate).exists():
            model_name = candidate
            break

    if model_name is None:
        pytest.skip("No frozen model found in test data")

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    atoms = read(data_dir / traj_file, index=0)
    system, calc = setup_DeepPotential(
        atoms=atoms,
        model_path=model_dir,
        model_name=model_name,
    )

    assert calc is not None, "returned None from DeepPotential calculator setup"

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"\nDeepPotential regression check:")
    print(f"   Energy (eV): {energy:.6f}")
    print(f"   Forces shape: {forces.shape}")

    assert energy is not None and np.isfinite(energy), "energy is not computed"
    assert forces is not None and np.all(np.isfinite(forces)), "forces are not computed"

    # Check ML/MD modules
    dyn_nvt = NoseNVT(atoms=system, temperature=330)
    results = dyn_nvt.run(1)
    assert results is not None, "Error in ML/MD NVT module"

    dyn_lag = LangevinNVT(atoms=system, temperature=330, friction=0.01)
    results = dyn_lag.run(1)
    assert results is not None, "Error in ML/MD Langevin module"

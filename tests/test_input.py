"""
Tests for input parsing utilities.

Updated for v0.2: uses SparcConfig dataclass-based parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.io import read
from sparc.src.utils.read_incar import parse_incar
from sparc.src.utils.utils import load_checkpoint


def test_read_incar(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "data" / "vasp_tmp"
    incar_file = "INCAR"

    if not (data_dir / incar_file).exists():
        pytest.skip("VASP test data not found")

    incar = parse_incar(str(data_dir / incar_file))
    assert isinstance(incar, dict), "Invalid INCAR file"


def test_load_checkpoint(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = repo_root / "tests" / "data" / "water_checkpoint.pkl"
    traj_path = repo_root / "tests" / "data" / "water.traj"

    if not checkpoint_path.exists() or not traj_path.exists():
        pytest.skip("Checkpoint test data not found")

    atoms = read(traj_path)
    updated_atoms, mdstep = load_checkpoint(atoms, checkpoint_path)
    assert isinstance(float(mdstep), float), "Invalid step, check your file is correct."

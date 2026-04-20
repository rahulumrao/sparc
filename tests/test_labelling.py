"""
Tests for v0.2 consolidated candidate trajectory (sparc/src/labelling.py).

Covers:
- labelling() return signature: (candidate_found, candidates_file, n_candidates)
- Candidates written to single candidates.extxyz file
- No candidates case
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.io import write as ase_write


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def model_devi_data(tmp_path):
    """
    Create a fake trajectory and model deviation file for testing.
    Returns (traj_path, devi_path, tmp_path).
    """
    # Create a small trajectory
    frames = []
    for i in range(10):
        atoms = molecule("H2O")
        atoms.set_cell([8, 8, 8])
        atoms.set_pbc(True)
        atoms.calc = EMT()
        atoms.get_potential_energy()
        atoms.positions += np.random.RandomState(i).randn(3, 3) * 0.05
        frames.append(atoms)

    traj_path = tmp_path / "traj.xyz"
    ase_write(str(traj_path), frames, format="extxyz")

    # Create fake model deviation file (dp model-devi format)
    # Columns: step max_devi_v min_devi_v avg_devi_v max_devi_f min_devi_f avg_devi_f
    devi_path = tmp_path / "model_devi.out"
    lines = ["#  step  max_devi_v  min_devi_v  avg_devi_v  max_devi_f  min_devi_f  avg_devi_f\n"]
    np.random.seed(42)
    for i in range(10):
        f_dev = np.random.uniform(0.0, 0.5)
        lines.append(f"{i}  0.001  0.0005  0.0008  {f_dev:.6f}  {f_dev*0.5:.6f}  {f_dev*0.75:.6f}\n")
    devi_path.write_text("".join(lines))

    return str(traj_path), str(devi_path), tmp_path


# ============================================================
# Return signature
# ============================================================

class TestLabellingReturnSignature:
    """Test that labelling() returns (bool, str, int) tuple."""

    def test_return_types(self, model_devi_data, monkeypatch):
        traj_path, devi_path, tmp_path = model_devi_data
        monkeypatch.chdir(tmp_path)

        from sparc.src.labelling import labelling
        result = labelling(
            trajfile=traj_path,
            outfile=devi_path,
            min_lim=0.1,
            max_lim=0.3,
            output_dir=str(tmp_path / "candidates"),
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        candidate_found = result[0]
        assert isinstance(candidate_found, bool)

    def test_candidates_single_file(self, model_devi_data, monkeypatch):
        """All candidates should be in a single extxyz file."""
        traj_path, devi_path, tmp_path = model_devi_data
        monkeypatch.chdir(tmp_path)

        from sparc.src.labelling import labelling
        candidate_found, candidates_file, n_candidates = labelling(
            trajfile=traj_path,
            outfile=devi_path,
            min_lim=0.1,
            max_lim=0.3,
            output_dir=str(tmp_path / "candidates"),
        )

        if candidate_found:
            assert Path(candidates_file).exists()
            assert candidates_file.endswith(".extxyz") or candidates_file.endswith(".xyz")
            assert n_candidates > 0

    def test_no_candidates(self, model_devi_data, monkeypatch):
        """When thresholds exclude all frames, should return (False, ...)."""
        traj_path, devi_path, tmp_path = model_devi_data
        monkeypatch.chdir(tmp_path)

        from sparc.src.labelling import labelling
        # Use thresholds that exclude everything
        result = labelling(
            trajfile=traj_path,
            outfile=devi_path,
            min_lim=10.0,   # impossibly high
            max_lim=20.0,
            output_dir=str(tmp_path / "candidates"),
        )

        assert result[0] is False

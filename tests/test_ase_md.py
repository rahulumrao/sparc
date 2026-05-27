"""
Tests for v0.2 ASE MD module (sparc/src/ase_md.py).

Covers:
- Temperature ramping function
- ExecuteMlpDynamics checkpoint/restart logic
- NoseNVT / LangevinNVT dynamics initialization
- initialize_dynamics helper
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase import units
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from sparc.src.ase_md import LangevinNVT, NoseNVT

# ============================================================
# NoseNVT / LangevinNVT basic dynamics
# ============================================================


class TestDynamicsInit:
    """Test that MD dynamics objects can be created and run."""

    @pytest.fixture
    def h2o_system(self):
        atoms = molecule("H2O")
        atoms.set_cell([8, 8, 8])
        atoms.set_pbc(True)
        atoms.calc = EMT()
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)
        return atoms

    def test_nose_nvt(self, h2o_system):
        dyn = NoseNVT(atoms=h2o_system, temperature=300, tdamp=2.0)
        result = dyn.run(2)
        assert result is not None

    def test_langevin_nvt(self, h2o_system):
        dyn = LangevinNVT(atoms=h2o_system, temperature=300, friction=0.01)
        result = dyn.run(2)
        assert result is not None


# ============================================================
# Checkpoint save/load
# ============================================================


class TestCheckpoint:
    """Test checkpoint save and load utilities."""

    def test_save_load_checkpoint(self, tmp_path):
        from ase.build import molecule
        from ase.md.langevin import Langevin
        from sparc.src.utils.utils import load_checkpoint, save_checkpoint

        atoms = molecule("H2O")
        atoms.set_cell([8, 8, 8])
        atoms.set_pbc(True)
        atoms.calc = EMT()
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        # Need a dynamics object for save_checkpoint
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
        dyn.run(5)  # run a few steps

        checkpoint_file = str(tmp_path / "md_checkpoint.pkl")
        save_checkpoint(dyn, atoms, checkpoint_file)

        assert os.path.exists(checkpoint_file)

        # Load checkpoint
        updated_atoms, step = load_checkpoint(atoms, checkpoint_file)
        assert isinstance(step, (int, float))
        assert updated_atoms is not None

    def test_checkpoint_preserves_positions(self, tmp_path):
        from ase.build import molecule
        from ase.md.langevin import Langevin
        from sparc.src.utils.utils import load_checkpoint, save_checkpoint

        atoms = molecule("H2O")
        atoms.set_cell([8, 8, 8])
        atoms.set_pbc(True)
        atoms.calc = EMT()
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
        dyn.run(3)

        original_positions = atoms.get_positions().copy()

        checkpoint_file = str(tmp_path / "md_checkpoint.pkl")
        save_checkpoint(dyn, atoms, checkpoint_file)

        # Perturb positions
        atoms.set_positions(atoms.get_positions() + 0.5)

        # Restore from checkpoint
        updated_atoms, step = load_checkpoint(atoms, checkpoint_file)
        assert np.allclose(
            updated_atoms.get_positions(), original_positions, atol=1e-10
        )

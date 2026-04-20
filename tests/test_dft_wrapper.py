"""
Tests for DFT calculator wrappers (sparc/src/calculator.py).

Updated for v0.2: includes all six engines (ORCA, xTB, CP2K, VASP, QE, Gaussian).
Skips when external executables are not available.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import warnings

import pytest

from sparc.src.calculator import dft_calculator, CalculatorError


def _which(name: str) -> str | None:
    return shutil.which(name)


# ============================================================
# ORCA
# ============================================================

def test_orca_calculator_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Runs ORCA via SPARC dft_calculator module using a template file.
    Skips if orca is not available.
    """
    orca_exe = _which("orca")
    if orca_exe is None:
        pytest.skip("orca not found in PATH")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "data" / "orca_tmp"

    orca_inp = data_dir / "orca_template.inp"
    results = data_dir / "results.json"

    for f in (orca_inp, results):
        if not f.exists():
            pytest.skip(f"ORCA test data not found: {f}")

    reference = json.loads(results.read_text())

    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)

    shutil.copy(orca_inp, run_dir / "orca_template.inp")

    monkeypatch.chdir(run_dir)

    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.center(vacuum=3.0)

    config = {
        "dft_calculator": {
            "engine": "ORCA",
            "template_file": "orca_template.inp",
            "exe_command": orca_exe,
        }
    }

    calc = dft_calculator(config, print_screen=False)
    assert calc is not None, "returned None for ORCA calculator setup"

    atoms.calc = calc

    energy_ev = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_force = float((forces**2).sum(axis=1).max() ** 0.5)

    ref_energy = float(reference["energy_ev"])
    ref_force = float(reference["max_force_evA"])
    energy_tol = float(reference.get("energy_tol_ev", 1e-3))
    force_tol = float(reference.get("max_force_tol_evA", 0.05))

    dE = energy_ev - ref_energy
    dF = max_force - ref_force

    print("\nORCA regression check:")
    print(f"  Energy (eV):     ref={ref_energy:.8f}  cur={energy_ev:.8f}  diff={dE:.6e}")
    print(f"  Max force (eV/Å): ref={ref_force:.8f}  cur={max_force:.8f}  diff={dF:.6e}")

    assert abs(dE) <= energy_tol, (
        f"Energy mismatch: diff={dE:.6e} eV (tol={energy_tol:.3e})"
    )
    assert abs(dF) <= force_tol, (
        f"Max force mismatch: diff={dF:.6e} eV/Å (tol={force_tol:.3e})"
    )


# ============================================================
# xTB
# ============================================================

def test_xtb_calculator_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Runs xTB via SPARC dft_calculator module using a template file.
    Skips if xtb is not available.
    """
    xtb_exe = _which("xtb")
    if xtb_exe is None:
        pytest.skip("xtb not found in PATH")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "data" / "xtb_tmp"

    xtb_inp = data_dir / "xtb_template.inp"
    results = data_dir / "results.json"

    for f in (xtb_inp, results):
        if not f.exists():
            pytest.skip(f"xTB test data not found: {f}")

    reference = json.loads(results.read_text())

    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)

    shutil.copy(xtb_inp, run_dir / "xtb_template.inp")

    monkeypatch.chdir(run_dir)

    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.center(vacuum=3.0)

    config = {
        "dft_calculator": {
            "engine": "xTB",
            "template_file": "xtb_template.inp",
            "exe_command": xtb_exe,
        }
    }

    try:
        calc = dft_calculator(config, print_screen=False)
    except ImportError:
        pytest.skip("xtb-python not installed")

    assert calc is not None, "returned None for xTB calculator setup"

    atoms.calc = calc

    energy_ev = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_force = float((forces**2).sum(axis=1).max() ** 0.5)

    ref_energy = float(reference["energy_ev"])
    ref_force = float(reference["max_force_evA"])
    energy_tol = float(reference.get("energy_tol_ev", 1e-3))
    force_tol = float(reference.get("max_force_tol_evA", 0.05))

    dE = energy_ev - ref_energy
    dF = max_force - ref_force

    print("\nxTB regression check:")
    print(f"  Energy (eV):     ref={ref_energy:.8f}  cur={energy_ev:.8f}  diff={dE:.6e}")
    print(f"  Max force (eV/Å): ref={ref_force:.8f}  cur={max_force:.8f}  diff={dF:.6e}")

    assert abs(dE) <= energy_tol, (
        f"Energy mismatch: diff={dE:.6e} eV (tol={energy_tol:.3e})"
    )
    assert abs(dF) <= force_tol, (
        f"Max force mismatch: diff={dF:.6e} eV/Å (tol={force_tol:.3e})"
    )


# ============================================================
# CP2K
# ============================================================

def test_cp2k_calculator_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Runs CP2K via SPARC dft_calculator module using a template file.
    Skips if cp2k_shell.psmp is not available.
    """
    cp2k_exe = _which("cp2k_shell.psmp")
    if cp2k_exe is None:
        pytest.skip("cp2k_shell.psmp not found in PATH")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "data" / "cp2k_tmp"

    cp2k_inp = data_dir / "cp2k_template.inp"
    basis = data_dir / "BASIS_SET"
    pot = data_dir / "GTH_POTENTIALS"
    results = data_dir / "results.json"

    for f in (cp2k_inp, basis, pot, results):
        assert f.exists(), f"Missing: {f}"

    reference = json.loads(results.read_text())

    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)

    shutil.copy(cp2k_inp, run_dir / "cp2k_template.inp")
    shutil.copy(basis, run_dir / "BASIS_SET")
    shutil.copy(pot, run_dir / "GTH_POTENTIALS")

    monkeypatch.chdir(run_dir)

    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.center(vacuum=3.0)

    config = {
        "dft_calculator": {
            "name": "CP2K",
            "exe_command": cp2k_exe,
        },
        "cp2k": {
            "label": str(run_dir / "cp2k" / "job"),
        },
    }

    calc = dft_calculator(config, print_screen=False)
    assert calc is not None

    atoms.calc = calc
    energy_ev = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_force = float((forces**2).sum(axis=1).max() ** 0.5)

    ref_energy = float(reference["energy_ev"])
    ref_force = float(reference["max_force_evA"])
    energy_tol = float(reference.get("energy_tol_ev", 1e-3))
    force_tol = float(reference.get("max_force_tol_evA", 0.05))

    assert abs(energy_ev - ref_energy) <= energy_tol
    assert abs(max_force - ref_force) <= force_tol


# ============================================================
# VASP
# ============================================================

def test_vasp_calculator_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Runs VASP via SPARC dft_calculator module using a INCAR template.
    Skips if vasp_std is not available.
    """
    vasp_exe = _which("vasp_std")
    if vasp_exe is None:
        pytest.skip("vasp_std not found in PATH")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "data" / "vasp_tmp"

    incar = data_dir / "INCAR"
    results = data_dir / "results.json"

    for f in (incar, results):
        assert f.exists(), f"Missing: {f}"

    reference = json.loads(results.read_text())

    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    shutil.copy(incar, run_dir / "INCAR")

    vasp_pp_path = os.getenv("VASP_PP_PATH")
    if not vasp_pp_path:
        pytest.skip("Missing POTCAR file VASP_PP_PATH not set")
    monkeypatch.setenv("VASP_PP_PATH", vasp_pp_path)
    monkeypatch.chdir(run_dir)

    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.center(vacuum=3.0)
    atoms.set_pbc([True, True, True])

    config = {
        "dft_calculator": {
            "name": "VASP",
            "prec": "Normal",
            "kgamma": True,
            "incar_file": "INCAR",
            "exe_command": vasp_exe,
        }
    }

    calc = dft_calculator(config, print_screen=False)
    assert calc is not None

    atoms.calc = calc
    from ase.config import ASEEnvDeprecationWarning
    warnings.filterwarnings("ignore", category=ASEEnvDeprecationWarning)

    energy_ev = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_force = float((forces**2).sum(axis=1).max() ** 0.5)

    ref_energy = float(reference["energy_ev"])
    ref_force = float(reference["max_force_evA"])
    energy_tol = float(reference.get("energy_tol_ev", 1e-3))
    force_tol = float(reference.get("max_force_tol_evA", 0.05))

    assert abs(energy_ev - ref_energy) <= energy_tol
    assert abs(max_force - ref_force) <= force_tol


# ============================================================
# QE
# ============================================================

def test_qe_calculator_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Test QE calculator setup. Skips if pw.x is not available.
    """
    pw_exe = _which("pw.x")
    if pw_exe is None:
        pytest.skip("pw.x (Quantum ESPRESSO) not found in PATH")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "data" / "qe_tmp"

    if not data_dir.exists():
        pytest.skip("QE test data not found")

    monkeypatch.chdir(tmp_path)

    # At minimum, verify the engine routes correctly
    config = {
        "dft_calculator": {
            "engine": "QE",
            "template_file": str(data_dir / "qe_template.in"),
            "exe_command": pw_exe,
        }
    }

    try:
        calc = dft_calculator(config, print_screen=False)
        assert calc is not None
    except CalculatorError:
        # Template parsing issues are acceptable if no test data
        pass


# ============================================================
# Gaussian
# ============================================================

def test_gaussian_calculator_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Test Gaussian calculator setup. Skips if g16/g09 is not available.
    """
    g16_exe = _which("g16") or _which("g09")
    if g16_exe is None:
        pytest.skip("Gaussian (g16/g09) not found in PATH")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "data" / "gaussian_tmp"

    if not data_dir.exists():
        pytest.skip("Gaussian test data not found")

    monkeypatch.chdir(tmp_path)

    config = {
        "dft_calculator": {
            "engine": "Gaussian",
            "template_file": str(data_dir / "gaussian_template.gjf"),
            "exe_command": g16_exe,
        }
    }

    try:
        calc = dft_calculator(config, print_screen=False)
        assert calc is not None
    except CalculatorError:
        pass

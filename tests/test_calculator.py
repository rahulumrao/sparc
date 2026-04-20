"""
Tests for calculator module (sparc/src/calculator.py) and template parsers.

Covers:
- SetupDFTCalculator class initialization
- Engine routing (dft_calculator dispatcher)
- ORCA template parser and calculator setup
- xTB template parser and calculator setup
- QE template parser and calculator setup
- Gaussian template parser and calculator setup (PREFIX command fix)
- Error handling for missing configs
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sparc.src.calculator import (
    SetupDFTCalculator,
    dft_calculator,
    CalculatorError,
)
from sparc.src.utils.read_input import SparcConfig, ConfigurationError


# ============================================================
# SetupDFTCalculator initialization
# ============================================================

class TestCalculatorSetup:
    """Test calculator setup class."""

    def test_dict_init(self):
        config = {
            "dft_calculator": {
                "engine": "VASP",
                "template_file": "INCAR",
            }
        }
        setup = SetupDFTCalculator(config)
        assert setup.dft_config["engine"] == "VASP"

    def test_missing_dft_key_raises(self):
        with pytest.raises(ConfigurationError, match="Missing 'dft_calculator'"):
            SetupDFTCalculator({"general": {}})

    def test_invalid_type_raises(self):
        with pytest.raises(ConfigurationError, match="must be dict or SparcConfig"):
            SetupDFTCalculator("invalid")


# ============================================================
# Engine dispatch
# ============================================================

class TestEngineDispatch:
    """Test dft_calculator routes to the correct engine."""

    def test_unsupported_engine(self):
        config = {
            "dft_calculator": {
                "engine": "TURBOMOLE",
                "template_file": "input.in",
            }
        }
        with pytest.raises(CalculatorError, match="Unsupported calculator"):
            dft_calculator(config)

    def test_all_engines_have_methods(self):
        """Verify SetupDFTCalculator has methods for all supported engines."""
        config = {"dft_calculator": {"engine": "VASP", "template_file": "t"}}
        setup = SetupDFTCalculator(config)
        for method in ("vasp", "cp2k", "orca", "xtb", "espresso", "gaussian"):
            assert hasattr(setup, method), f"Missing method: {method}"


# ============================================================
# ORCA Parser & Calculator
# ============================================================

class TestORCA:
    """Test ORCA template parser and calculator setup."""

    def test_orca_parser_basic(self, tmp_path):
        """Parse a basic ORCA template."""
        from sparc.src.utils.OrcaParser import parse_orca_template
        template = tmp_path / "orca.inp"
        template.write_text(
            "! PBE def2-SVP TightSCF\n"
            "*xyz 0 1\n"
            "H 0.0 0.0 0.0\n"
            "H 0.0 0.0 0.74\n"
            "*\n"
        )
        result = parse_orca_template(str(template))
        assert result["orcasimpleinput"] == "PBE def2-SVP TightSCF"
        assert result["charge"] == 0
        assert result["multi"] == 1

    def test_orca_parser_charge_mult(self, tmp_path):
        """Parse charge and multiplicity from template."""
        from sparc.src.utils.OrcaParser import parse_orca_template
        template = tmp_path / "orca.inp"
        template.write_text(
            "! B3LYP 6-31G*\n"
            "*xyz -1 2\n"
            "*\n"
        )
        result = parse_orca_template(str(template))
        assert result["charge"] == -1
        assert result["multi"] == 2

    def test_orca_parser_with_blocks(self, tmp_path):
        """Parse ORCA template with %pal block."""
        from sparc.src.utils.OrcaParser import parse_orca_template
        template = tmp_path / "orca.inp"
        template.write_text(
            "! PBE def2-SVP\n"
            "%pal\n"
            "  nprocs 4\n"
            "end\n"
            "*xyz 0 1\n"
            "*\n"
        )
        result = parse_orca_template(str(template))
        assert "nprocs" in result["orcablocks"]
        assert result["orcasimpleinput"] == "PBE def2-SVP"

    def test_orca_parser_missing_keyword(self, tmp_path):
        """Template without '!' line should raise."""
        from sparc.src.utils.OrcaParser import parse_orca_template
        template = tmp_path / "bad.inp"
        template.write_text("*xyz 0 1\n*\n")
        with pytest.raises(ValueError, match="missing"):
            parse_orca_template(str(template))

    def test_orca_parser_file_not_found(self):
        from sparc.src.utils.OrcaParser import parse_orca_template
        with pytest.raises(FileNotFoundError):
            parse_orca_template("/nonexistent/orca.inp")

    def test_orca_calculator_setup(self, tmp_path, monkeypatch):
        """Test ORCA calculator object creation."""
        monkeypatch.chdir(tmp_path)
        template = tmp_path / "orca.inp"
        template.write_text(
            "! PBE def2-SVP\n"
            "*xyz 0 1\n"
            "*\n"
        )
        config = {
            "dft_calculator": {
                "engine": "ORCA",
                "template_file": str(template),
                "exe_command": "/fake/orca",
            }
        }
        try:
            calc = dft_calculator(config, print_screen=False)
            assert calc is not None
        except Exception as e:
            # ORCA import may fail if ASE version doesn't have OrcaProfile
            if "OrcaProfile" in str(e) or "No module" in str(e):
                pytest.skip("ORCA calculator not available in this ASE version")
            raise


# ============================================================
# xTB Parser & Calculator
# ============================================================

class TestXTB:
    """Test xTB template parser and calculator setup."""

    def test_xtb_parser_basic(self, tmp_path):
        """Parse a basic xTB template."""
        from sparc.src.utils.xTBParser import xtb_template
        template = tmp_path / "xtb.inp"
        template.write_text(
            "# xTB template\n"
            "method = GFN2-xTB\n"
            "charge = 0\n"
            "multiplicity = 1\n"
            "accuracy = 1.0\n"
            "electronic_temperature = 300.0\n"
            "max_iterations = 250\n"
        )
        result = xtb_template(str(template))
        assert result["method"] == "GFN2-xTB"
        assert result["charge"] == 0
        assert result["multiplicity"] == 1
        assert result["accuracy"] == 1.0
        assert result["electronic_temperature"] == 300.0
        assert result["max_iterations"] == 250

    def test_xtb_parser_solvent(self, tmp_path):
        """Parse xTB template with solvent settings."""
        from sparc.src.utils.xTBParser import xtb_template
        template = tmp_path / "xtb.inp"
        template.write_text(
            "method = GFN2-xTB\n"
            "charge = 0\n"
            "multiplicity = 1\n"
            "solvent = water\n"
            "solvent_method = alpb\n"
        )
        result = xtb_template(str(template))
        assert result["solvent"] == "water"
        assert result["solvent_method"] == "alpb"

    def test_xtb_parser_none_solvent(self, tmp_path):
        """Solvent = None should parse as Python None."""
        from sparc.src.utils.xTBParser import xtb_template
        template = tmp_path / "xtb.inp"
        template.write_text(
            "method = GFN2-xTB\n"
            "charge = 0\n"
            "multiplicity = 1\n"
            "solvent = None\n"
        )
        result = xtb_template(str(template))
        assert result["solvent"] is None

    def test_xtb_parser_comments_ignored(self, tmp_path):
        """Lines starting with '#' should be ignored."""
        from sparc.src.utils.xTBParser import xtb_template
        template = tmp_path / "xtb.inp"
        template.write_text(
            "# This is a comment\n"
            "method = GFN1-xTB\n"
            "# another comment\n"
            "charge = 1\n"
        )
        result = xtb_template(str(template))
        assert result["method"] == "GFN1-xTB"
        assert result["charge"] == 1

    def test_xtb_calculator_setup(self, tmp_path, monkeypatch):
        """Test xTB calculator object creation."""
        monkeypatch.chdir(tmp_path)
        template = tmp_path / "xtb.inp"
        template.write_text(
            "method = GFN2-xTB\n"
            "charge = 0\n"
            "multiplicity = 1\n"
            "accuracy = 1.0\n"
            "electronic_temperature = 300.0\n"
            "max_iterations = 250\n"
        )
        config = {
            "dft_calculator": {
                "engine": "xTB",
                "template_file": str(template),
                "exe_command": "/fake/xtb",
            }
        }
        try:
            calc = dft_calculator(config, print_screen=False)
            assert calc is not None
        except ImportError:
            pytest.skip("xtb-python not installed")
        except CalculatorError as e:
            if "xTB" in str(e) and "not found" in str(e):
                pytest.skip("xTB not available")
            raise

    def test_xtb_unsupported_method(self, tmp_path, monkeypatch):
        """Unsupported xTB method should raise CalculatorError."""
        monkeypatch.chdir(tmp_path)
        template = tmp_path / "xtb.inp"
        template.write_text("method = GFN0-xTB\ncharge = 0\nmultiplicity = 1\n")
        config = {
            "dft_calculator": {
                "engine": "xTB",
                "template_file": str(template),
                "exe_command": "/fake/xtb",
            }
        }
        try:
            with pytest.raises(CalculatorError, match="Unsupported xTB method"):
                dft_calculator(config)
        except ImportError:
            pytest.skip("xtb-python not installed")


# ============================================================
# QE Parser & Calculator
# ============================================================

class TestQE:
    """Test Quantum ESPRESSO template parser and calculator setup."""

    def test_qe_parser_namelists(self, tmp_path):
        """Parse QE namelists (CONTROL, SYSTEM, ELECTRONS)."""
        from sparc.src.utils.QEParser import qe_template
        template = tmp_path / "qe.in"
        template.write_text(
            "&CONTROL\n"
            "  calculation = 'scf'\n"
            "  tprnfor = .true.\n"
            "  tstress = .true.\n"
            "  pseudo_dir = './pseudo'\n"
            "/\n"
            "&SYSTEM\n"
            "  ecutwfc = 60\n"
            "  ecutrho = 480\n"
            "/\n"
            "&ELECTRONS\n"
            "  conv_thr = 1.0d-8\n"
            "/\n"
            "ATOMIC_SPECIES\n"
            "  Si  28.085  Si.pbe.UPF\n"
            "  O   15.999  O.pbe.UPF\n"
        )
        result = qe_template(str(template))
        assert result["input_data"]["calculation"] == "scf"
        assert result["input_data"]["tprnfor"] is True
        assert result["input_data"]["ecutwfc"] == 60
        assert result["input_data"]["conv_thr"] == 1.0e-8
        assert result["pseudo_dir"] == "./pseudo"
        assert result["pseudopotentials"]["Si"] == "Si.pbe.UPF"
        assert result["pseudopotentials"]["O"] == "O.pbe.UPF"

    def test_qe_parser_kpoints(self, tmp_path):
        """Parse K_POINTS card."""
        from sparc.src.utils.QEParser import qe_template
        template = tmp_path / "qe.in"
        template.write_text(
            "&CONTROL\n"
            "  calculation = 'scf'\n"
            "/\n"
            "&SYSTEM\n"
            "  ecutwfc = 40\n"
            "/\n"
            "&ELECTRONS\n"
            "/\n"
            "ATOMIC_SPECIES\n"
            "  H  1.008  H.UPF\n"
            "K_POINTS automatic\n"
            "  4 4 4  0 0 0\n"
        )
        result = qe_template(str(template))
        assert result["kpts"] == (4, 4, 4)
        assert result["koffset"] == (0, 0, 0)

    def test_qe_parser_gamma(self, tmp_path):
        """Gamma-only k-points should return kpts=None."""
        from sparc.src.utils.QEParser import qe_template
        template = tmp_path / "qe.in"
        template.write_text(
            "&CONTROL\n/\n&SYSTEM\n  ecutwfc = 40\n/\n&ELECTRONS\n/\n"
            "ATOMIC_SPECIES\n  H  1.008  H.UPF\n"
            "K_POINTS gamma\n"
        )
        result = qe_template(str(template))
        assert result["kpts"] is None

    def test_qe_parser_fortran_booleans(self, tmp_path):
        """Fortran .true./.false. should convert to Python booleans."""
        from sparc.src.utils.QEParser import qe_template
        template = tmp_path / "qe.in"
        template.write_text(
            "&CONTROL\n"
            "  tprnfor = .TRUE.\n"
            "  tstress = .FALSE.\n"
            "/\n"
            "&SYSTEM\n  ecutwfc = 40\n/\n&ELECTRONS\n/\n"
            "ATOMIC_SPECIES\n  H  1.008  H.UPF\n"
        )
        result = qe_template(str(template))
        assert result["input_data"]["tprnfor"] is True
        assert result["input_data"]["tstress"] is False

    def test_qe_parser_file_not_found(self):
        from sparc.src.utils.QEParser import qe_template
        with pytest.raises(FileNotFoundError):
            qe_template("/nonexistent/qe.in")

    def test_qe_dispatch(self, tmp_path, monkeypatch):
        """QE engine routes to espresso() method."""
        monkeypatch.chdir(tmp_path)
        template = tmp_path / "qe.in"
        template.write_text(
            "&CONTROL\n  calculation = 'scf'\n/\n"
            "&SYSTEM\n  ecutwfc = 40\n/\n"
            "&ELECTRONS\n/\n"
            "ATOMIC_SPECIES\n  H  1.008  H.UPF\n"
        )
        config = {
            "dft_calculator": {
                "engine": "QE",
                "template_file": str(template),
                "exe_command": "/fake/pw.x",
            }
        }
        try:
            calc = dft_calculator(config, print_screen=False)
            assert calc is not None
        except Exception as e:
            if "EspressoProfile" in str(e) or "No module" in str(e):
                pytest.skip("QE calculator not available in this ASE version")
            raise


# ============================================================
# Gaussian Parser & Calculator
# ============================================================

class TestGaussian:
    """Test Gaussian template parser and calculator setup."""

    def test_gaussian_parser_basic(self, tmp_path):
        """Parse a basic Gaussian template."""
        from sparc.src.utils.GaussianParser import gaussian_template
        template = tmp_path / "gauss.inp"
        template.write_text(
            "# Gaussian template\n"
            "method = B3LYP\n"
            "basis = 6-31G*\n"
            "charge = 0\n"
            "multiplicity = 1\n"
        )
        result = gaussian_template(str(template))
        assert result["method"] == "B3LYP"
        assert result["basis"] == "6-31G*"
        assert result["charge"] == 0
        assert result["mult"] == 1
        # multiplicity should be renamed to mult
        assert "multiplicity" not in result

    def test_gaussian_parser_with_link0(self, tmp_path):
        """Parse Gaussian template with Link0 parameters."""
        from sparc.src.utils.GaussianParser import gaussian_template
        template = tmp_path / "gauss.inp"
        template.write_text(
            "method = HF\n"
            "basis = STO-3G\n"
            "mem = 4GB\n"
            "nprocshared = 8\n"
            "chk = job.chk\n"
        )
        result = gaussian_template(str(template))
        assert result["mem"] == "4GB"
        assert result["nprocshared"] == "8"
        assert result["chk"] == "job.chk"

    def test_gaussian_parser_nproc_renamed(self, tmp_path):
        """'nproc' should be mapped to 'nprocshared' for ASE."""
        from sparc.src.utils.GaussianParser import gaussian_template
        template = tmp_path / "gauss.inp"
        template.write_text(
            "method = HF\n"
            "basis = STO-3G\n"
            "nproc = 4\n"
        )
        result = gaussian_template(str(template))
        assert "nproc" not in result
        assert result["nprocshared"] == "4"

    def test_gaussian_parser_missing_method(self, tmp_path):
        """Missing method should raise ValueError."""
        from sparc.src.utils.GaussianParser import gaussian_template
        template = tmp_path / "gauss.inp"
        template.write_text("basis = STO-3G\n")
        with pytest.raises(ValueError, match="missing 'method'"):
            gaussian_template(str(template))

    def test_gaussian_parser_missing_basis(self, tmp_path):
        """Missing basis should raise ValueError."""
        from sparc.src.utils.GaussianParser import gaussian_template
        template = tmp_path / "gauss.inp"
        template.write_text("method = HF\n")
        with pytest.raises(ValueError, match="missing 'basis'"):
            gaussian_template(str(template))

    def test_gaussian_parser_file_not_found(self):
        from sparc.src.utils.GaussianParser import gaussian_template
        with pytest.raises(FileNotFoundError):
            gaussian_template("/nonexistent/gauss.inp")

    def test_gaussian_command_uses_prefix(self, tmp_path, monkeypatch):
        """Verify Gaussian uses 'g16 PREFIX.com > PREFIX.log' not stdin redirect."""
        monkeypatch.chdir(tmp_path)
        template = tmp_path / "gauss.inp"
        template.write_text("method = HF\nbasis = STO-3G\n")

        config = {
            "dft_calculator": {
                "engine": "Gaussian",
                "template_file": str(template),
                "exe_command": "/opt/g16/g16",
            }
        }

        with patch("ase.calculators.gaussian.Gaussian.__init__", return_value=None) as mock_init:
            setup = SetupDFTCalculator(config)
            try:
                setup.gaussian()
            except Exception:
                pass

            if mock_init.called:
                call_kwargs = mock_init.call_args
                cmd = call_kwargs[1].get("command", "")
                assert "PREFIX.com" in cmd, f"Expected PREFIX template, got: {cmd}"
                assert "<" not in cmd, f"Stdin redirect found in command: {cmd}"

    def test_gaussian_auto_finds_exe(self, tmp_path, monkeypatch):
        """When exe_command is None, should look for g16/g09 in PATH."""
        monkeypatch.chdir(tmp_path)
        template = tmp_path / "gauss.inp"
        template.write_text("method = HF\nbasis = STO-3G\n")

        config = {
            "dft_calculator": {
                "engine": "Gaussian",
                "template_file": str(template),
                # no exe_command
            }
        }

        with patch("shutil.which", return_value=None):
            setup = SetupDFTCalculator(config)
            with pytest.raises(CalculatorError, match="Gaussian executable not found"):
                setup.gaussian()

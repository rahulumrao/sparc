#!/usr/bin/python3
# calculator.py

"""
Unified interface for setting up DFT calculators as ASE calculator objects.

Supports VASP, CP2K, ORCA, xTB, Quantum ESPRESSO, and Gaussian. Each engine
reads its parameters from an engine-specific template file defined in the
``dft_calculator`` section of ``input.yaml``.
"""

import argparse
import os
import shutil
from pathlib import Path

import yaml

################################################################
# Third party imports
# Lazy imports: each backend is imported only when its calculator is requested.
# This prevents ImportError when a user has only one DFT code installed.
from sparc.src.utils.GaussianParser import gaussian_template

################################################################
# Local imports
from sparc.src.utils.logger import SparcLog
from sparc.src.utils.OrcaParser import parse_orca_template
from sparc.src.utils.QEParser import qe_template
from sparc.src.utils.read_incar import parse_incar
from sparc.src.utils.read_input import ConfigurationError, SparcConfig
from sparc.src.utils.xTBParser import xtb_template

################################################################
# Custom Exceptions
################################################################


class CalculatorError(Exception):
    """
    Exception raised when calculator setup or execution fails.

    This exception is raised for issues such as:
    - Missing required configuration parameters
    - Invalid template files
    - Calculator initialization failures
    - Unsupported calculator engines
    """

    pass


################################################################
# Calculator Setup Class
################################################################


class SetupDFTCalculator:
    """
    Class to set up DFT calculators for VASP, CP2K, ORCA, and xTB using ASE.

    This class provides a unified interface for configuring different DFT engines
    based on template files and configuration dictionaries. It handles parsing
    of engine-specific input files and creates appropriate ASE calculator objects.

    Parameters
    ----------
    input_config : dict or SparcConfig
        Configuration dictionary or SparcConfig object containing calculator settings.
        Must include a 'dft_calculator' key with engine-specific parameters.
    print_screen : bool, optional
        Whether to print calculator configuration details to log. Default is False.

    Attributes
    ----------
    input_config : dict
        Processed configuration dictionary
    dft_config : dict
        DFT calculator specific configuration
    print_screen : bool
        Flag for verbose output

    Raises
    ------
    ConfigurationError
        If input_config is invalid or missing required keys
    CalculatorError
        If calculator setup fails

    Examples
    --------
    >>> config = {'dft_calculator': {'engine': 'VASP', 'template_file': 'INCAR', ...}}
    >>> calc_setup = SetupDFTCalculator(config, print_screen=True)
    >>> calculator = calc_setup.vasp()
    """

    def __init__(self, input_config, print_screen=False):
        """
        Initialize the DFT calculator setup.

        Parameters
        ----------
        input_config : dict or SparcConfig
            Configuration containing calculator settings
        print_screen : bool, optional
            Whether to print configuration details
        """
        # Handle both dict and SparcConfig
        if isinstance(input_config, SparcConfig):
            self.input_config = input_config.to_dict()
            self.is_config_object = True
        elif isinstance(input_config, dict):
            if "dft_calculator" not in input_config:
                raise ConfigurationError("Missing 'dft_calculator' key in input_config")
            self.input_config = input_config
            self.is_config_object = False
        else:
            raise ConfigurationError("input_config must be dict or SparcConfig")

        self.print_screen = print_screen
        self.dft_config = self.input_config["dft_calculator"]

    def vasp(self):
        """
        Set up the VASP calculator from INCAR template.
        """
        from ase.calculators.vasp import Vasp

        engine = self.dft_config.get("engine") or self.dft_config.get("name")

        if engine.upper() != "VASP":
            raise CalculatorError(f"Wrong engine: {engine}. Expected VASP")

        # Get required parameters
        exe_command = self.dft_config.get("exe_command")
        template_file = self.dft_config.get("template_file")  # INCAR file

        if not exe_command:
            raise CalculatorError("exe_command required for VASP")
        if not template_file:
            raise CalculatorError("template_file (INCAR) required for VASP")

        # Parse INCAR file
        incar_path = Path(template_file)
        if not incar_path.exists():
            raise CalculatorError(f"VASP INCAR file not found: {incar_path}")

        try:
            incar_params = parse_incar(str(incar_path))
        except Exception as e:
            raise CalculatorError(f"Failed to parse INCAR: {e}")

        # Print INCAR parameters if requested
        if self.print_screen:
            SparcLog("-" * 50)
            SparcLog(" VASP INPUT PARAMETERS ".center(50))
            SparcLog("-" * 50)
            if incar_params:
                max_key_length = max(len(key) for key in incar_params.keys())
                for key, value in incar_params.items():
                    SparcLog(f"| {key.upper():<{max_key_length}} : {value}")
            SparcLog("-" * 50)

        # Default parameters (can be overridden by **incar_params)
        prec = self.dft_config.get("prec", "Normal")
        kgamma = self.dft_config.get("kgamma", True)
        xc = self.dft_config.get("xc", "PBE")
        pp = self.dft_config.get("pp", "PBE")
        directory = self.dft_config.get("directory", "vasp")

        try:
            calc = Vasp(
                prec=prec,
                kgamma=kgamma,
                gamma=not kgamma,
                xc=xc,
                pp=pp,
                directory=directory,
                command=exe_command,
                **incar_params,
            )

            SparcLog("VASP Calculator Setup Successfully!")
            return calc

        except Exception as e:
            raise CalculatorError(f"Failed to create VASP calculator: {e}")

    def cp2k(self):
        """
        Set up the CP2K calculator from template file.
        """
        from ase.calculators.cp2k import CP2K

        engine = self.dft_config.get("engine") or self.dft_config.get("name")

        if engine.upper() != "CP2K":
            raise CalculatorError(f"Wrong engine: {engine}. Expected CP2K")

        # Get template file
        template_file = self.dft_config.get("template_file", "cp2k_template.inp")

        # Read template
        inpp = ""
        try:
            with open(template_file, "r") as f:
                for line in f:
                    if not line.startswith("#") and not line.startswith("!"):
                        inpp += line.strip() + "\n"
        except FileNotFoundError:
            raise CalculatorError(f"CP2K template file not found: {template_file}")
        except Exception as e:
            raise CalculatorError(f"Error reading CP2K template: {e}")

        # Default parameters
        # cutoff=None: read CUTOFF from &MGRID in the template instead of injecting it.
        # basis_set='6-31G*': global fallback applied to all elements. To use different
        #   basis sets per element, set basis_set: null in the cp2k: yaml section and
        #   add a BASIS_SET line to each &KIND block in the template.
        # pseudo_potential=None: suppress ASE's global potential injection; set POTENTIAL
        #   per element in &KIND blocks of the template.
        # basis_set_file / potential_file: filenames passed to BASIS_SET_FILE_NAME and
        #   POTENTIAL_FILE_NAME in &DFT; override in the cp2k: yaml section as needed.
        default_params = {
            "xc": "PBE",
            "cutoff": None,
            "max_scf": 500,
            "basis_set": None,
            "basis_set_file": "BASIS_MOLOPT",
            "potential_file": "GTH_POTENTIALS",
            "pseudo_potential": None,
            "label": os.path.join("cp2k", "job"),
        }

        # Update with user config
        cp2k_config = self.input_config.get("cp2k", {})
        default_params.update(cp2k_config)

        # Get exe command
        exe_command = self.dft_config.get("exe_command", "cp2k.popt")

        try:
            calc = CP2K(command=exe_command, inp=inpp, **default_params)

            if self.print_screen:
                SparcLog("-" * 50)
                SparcLog(" CP2K PARAMETERS ".center(50))
                SparcLog("-" * 50)
                for key, value in default_params.items():
                    if value is not None:
                        SparcLog(f"  {key.capitalize():<28} {value}")
                SparcLog(f"  {'Template_file':<28} {template_file}")
                SparcLog("-" * 50)
                SparcLog(" CP2K TEMPLATE INPUT ".center(50))
                SparcLog("-" * 50)
                for line in inpp.splitlines():
                    if line.strip():
                        SparcLog(f"  {line}")
                SparcLog("-" * 50)

            SparcLog("CP2K calculator setup successfully")
            return calc

        except Exception as e:
            raise CalculatorError(f"Failed to create CP2K calculator: {e}")

    def orca(self):
        """
        Set up the ORCA calculator from template file.
        """
        import shutil

        from ase.calculators.orca import ORCA, OrcaProfile

        engine = self.dft_config.get("engine") or self.dft_config.get("name")

        if engine.upper() != "ORCA":
            raise CalculatorError(f"Wrong engine: {engine}. Expected ORCA")

        # Get configuration
        orca_cfg = self.input_config.get("orca", {})

        # Get template file
        template_file = self.dft_config.get("template_file", "orca_template.inp")
        if not template_file:
            raise CalculatorError("template_file required for ORCA")

        # Parse template
        try:
            tfl = parse_orca_template(template_file)
        except Exception as e:
            raise CalculatorError(f"Failed to parse ORCA template: {e}")

        # Extract parameters from template or config
        orcasimpleinput = orca_cfg.get("orcasimpleinput", tfl["orcasimpleinput"])
        orcablocks = orca_cfg.get("orcablocks", tfl["orcablocks"])
        charge = orca_cfg.get("charge", tfl["charge"])
        multiplicity = orca_cfg.get("multi", tfl["multi"])

        # Resolve executable
        exe_command = self.dft_config.get("exe_command")
        if exe_command:
            cmd = exe_command
        else:
            found = shutil.which("orca")
            if not found:
                raise CalculatorError(
                    "ORCA executable not found. Install ORCA or set exe_command"
                )
            cmd = found

        # Setup profile and directory
        profile = OrcaProfile(command=str(cmd))
        directory = "orca_run"
        Path(directory).mkdir(parents=True, exist_ok=True)

        try:
            calc = ORCA(
                label="orca",
                directory=directory,
                profile=profile,
                charge=charge,
                mult=multiplicity,
                orcasimpleinput=orcasimpleinput,
                orcablocks=orcablocks,
            )

            if self.print_screen:
                SparcLog("-" * 50)
                SparcLog(" ORCA CALCULATOR ".center(50))
                SparcLog("-" * 50)
                SparcLog(f" {'Executable':<20} {cmd}")
                SparcLog(f" {'Charge, Mult':<20} : {charge}, {multiplicity}")
                SparcLog(f" {'Simple Input':<20} {orcasimpleinput}")
                SparcLog(f" {'ORCA Blocks':<22}")
                for line in orcablocks.splitlines():
                    SparcLog(f"   {line}")
                SparcLog("-" * 50)

            SparcLog("ORCA calculator setup successfully")
            return calc

        except Exception as e:
            raise CalculatorError(f"Failed to create ORCA calculator: {e}")

    def xtb(self):
        """
        Set up the xTB (extended tight-binding) calculator from template.
        """
        import shutil

        from xtb.ase.calculator import XTB

        engine = self.dft_config.get("engine") or self.dft_config.get("name")

        if engine.upper() != "XTB":
            raise CalculatorError(f"Wrong engine: {engine}. Expected xTB")

        # Get template file
        template_file = self.dft_config.get("template_file")
        if not template_file or not Path(template_file).exists():
            raise CalculatorError("xTB template_file required in dft_calculator config")

        # Parse template
        try:
            cfg = xtb_template(template_file)
        except Exception as e:
            raise CalculatorError(f"Failed to parse xTB template: {e}")

        # Resolve executable
        exe_command = self.dft_config.get("exe_command")
        if exe_command:
            cmd = exe_command
        else:
            found = shutil.which("xtb")
            if not found:
                raise CalculatorError(
                    "xTB executable not found. Install xTB or set exe_command"
                )
            cmd = found

        # Extract parameters
        method = cfg.get("method", "GFN2-xTB")
        charge = int(cfg.get("charge", 0))
        multiplicity = int(cfg.get("multiplicity", 1))
        uhf = max(0, multiplicity - 1)
        accuracy = float(cfg.get("accuracy", 1.0))
        etemp = float(cfg.get("electronic_temperature", 300.0))
        maxiter = int(cfg.get("max_iterations", 250))
        solvent = cfg.get("solvent", None)
        solvmet = cfg.get("solvent_method", None)
        workdir = cfg.get("directory", "xtb_run")
        label = cfg.get("label", "xtb")

        # Validate method
        if method not in {"GFN1-xTB", "GFN2-xTB"}:
            raise CalculatorError(
                f"Unsupported xTB method: {method}. Use GFN1-xTB or GFN2-xTB"
            )

        # Build calculator kwargs
        kwargs = {
            "charge": charge,
            "uhf": uhf,
            "accuracy": accuracy,
            "electronic_temperature": etemp,
            "max_iterations": maxiter,
            "command": cmd,
            "label": label,
        }

        # Add solvent if specified
        if isinstance(solvent, str) and solvent.strip():
            kwargs["solvent"] = solvent.strip()
            if isinstance(solvmet, str) and solvmet.strip():
                kwargs["solvation"] = solvmet.strip()

        # Create working directory
        Path(workdir).mkdir(parents=True, exist_ok=True)

        try:
            calc = XTB(method=method, directory=workdir, **kwargs)

            if self.print_screen:
                SparcLog("-" * 50)
                SparcLog(" xTB CALCULATOR ".center(50))
                SparcLog("-" * 50)
                SparcLog(f" {'Executable':<20} {cmd}")
                SparcLog(f" {'Method':<20} {method}")
                SparcLog(f" {'Charge, Mult':<20} {charge}, {multiplicity} (UHF={uhf})")
                SparcLog(f" {'Accuracy':<20} {accuracy}")
                SparcLog(f" {'Elec. Temp (K)':<20} {etemp}")
                SparcLog(f" {'Max Iter':<20} {maxiter}")
                SparcLog(f" {'Solvent':<20} {solvent or 'None'}")
                SparcLog(f" {'xTB workdir, Label':<20} {workdir}, {label}")
                SparcLog("-" * 50)

            SparcLog("xTB calculator setup successfully")
            return calc

        except Exception as e:
            raise CalculatorError(f"Failed to create xTB calculator: {e}")

    def espresso(self):
        """
        Set up the Quantum ESPRESSO calculator from a pw.x template file.

        The template file should be a standard pw.x input containing
        namelists (&CONTROL, &SYSTEM, &ELECTRONS, ...), ATOMIC_SPECIES,
        and K_POINTS cards. Coordinates are provided by ASE, so
        ATOMIC_POSITIONS and CELL_PARAMETERS in the template are ignored.
        """
        import shutil

        from ase.calculators.espresso import Espresso, EspressoProfile

        engine = self.dft_config.get("engine") or self.dft_config.get("name")
        if engine.upper() != "QE":
            raise CalculatorError(f"Wrong engine: {engine}. Expected QE")

        # Get template file
        template_file = self.dft_config.get("template_file")
        if not template_file:
            raise CalculatorError("template_file required for Quantum ESPRESSO")

        # Parse template
        try:
            cfg = qe_template(str(template_file))
        except Exception as e:
            raise CalculatorError(f"Failed to parse QE template: {e}")

        input_data = cfg["input_data"]
        pseudopotentials = cfg["pseudopotentials"]
        pseudo_dir = cfg["pseudo_dir"]
        kpts = cfg["kpts"]
        koffset = cfg["koffset"]

        if not pseudopotentials:
            raise CalculatorError(
                "No pseudopotentials found in QE template. "
                "Add an ATOMIC_SPECIES card with element, mass, and PP filename."
            )

        # Force single-point calculation for ASE-driven MD
        input_data["calculation"] = "scf"
        input_data["tprnfor"] = True
        input_data["tstress"] = True

        # Resolve executable
        exe_command = self.dft_config.get("exe_command")
        if exe_command:
            cmd = exe_command
        else:
            found = shutil.which("pw.x")
            if not found:
                raise CalculatorError(
                    "pw.x executable not found. "
                    "Install Quantum ESPRESSO or set exe_command in dft_calculator config."
                )
            cmd = found

        # Build profile
        profile_kwargs = {"command": cmd}
        if pseudo_dir:
            profile_kwargs["pseudo_dir"] = pseudo_dir

        profile = EspressoProfile(**profile_kwargs)

        # Working directory
        directory = self.dft_config.get("directory", "espresso")
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Build calculator kwargs
        calc_kwargs = {
            "profile": profile,
            "pseudopotentials": pseudopotentials,
            "input_data": input_data,
            "directory": directory,
        }
        if kpts is not None:
            calc_kwargs["kpts"] = kpts
        if koffset is not None:
            calc_kwargs["koffset"] = koffset

        try:
            calc = Espresso(**calc_kwargs)

            if self.print_screen:
                SparcLog("-" * 50)
                SparcLog(" QUANTUM ESPRESSO CALCULATOR ".center(50))
                SparcLog("-" * 50)
                SparcLog(f"  {'Executable':<24} {cmd}")
                SparcLog(f"  {'Pseudo Dir':<24} {pseudo_dir or 'default'}")
                SparcLog(f"  {'K-points':<24} {kpts or 'Gamma'}")
                SparcLog(f"  {'K-offset':<24} {koffset or 'None'}")
                SparcLog(f"  {'Directory':<24} {directory}")
                SparcLog("")
                SparcLog("  Pseudopotentials:")
                for elem, pp in pseudopotentials.items():
                    SparcLog(f"    {elem:<6} {pp}")
                SparcLog("")
                SparcLog("  Input Parameters:")
                for key, value in input_data.items():
                    SparcLog(f"    {key:<24} {value}")
                SparcLog("-" * 50)

            SparcLog("Quantum ESPRESSO calculator setup successfully")
            return calc

        except Exception as e:
            raise CalculatorError(f"Failed to create QE calculator: {e}")

    def gaussian(self):
        """
        Set up the Gaussian calculator from a template file.

        All keywords from the template are passed directly to ASE's
        Gaussian(). ASE handles Link0 vs route section sorting internally.
        force=None is always added so ASE drives single-point force calculations.
        """
        from ase.calculators.gaussian import Gaussian

        engine = self.dft_config.get("engine") or self.dft_config.get("name")
        if engine.upper() != "GAUSSIAN":
            raise CalculatorError(f"Wrong engine: {engine}. Expected Gaussian")

        template_file = self.dft_config.get("template_file")
        if not template_file:
            raise CalculatorError("template_file required for Gaussian")

        try:
            params = gaussian_template(str(template_file))
        except Exception as e:
            raise CalculatorError(f"Failed to parse Gaussian template: {e}")

        # Working directory
        directory = self.dft_config.get("directory", "gaussian")
        Path(directory).mkdir(parents=True, exist_ok=True)

        # force=None ensures ASE requests forces (single-point, no opt)
        params["force"] = None

        # label and command are FileIOCalculator constructor args,
        # NOT Gaussian route/link0 parameters — keep them separate
        label = os.path.join(directory, "gaussian")
        exe_command = self.dft_config.get("exe_command")

        #
        if exe_command:
            g16_bin = exe_command
        else:
            g16_bin = shutil.which("g16") or shutil.which("g09")
            if not g16_bin:
                raise CalculatorError(
                    "Gaussian executable not found. Install Gaussian or set exe_command"
                )
        cmd = f"{g16_bin} PREFIX.com > PREFIX.log"

        try:
            calc = Gaussian(label=label, command=cmd, **params)

            if self.print_screen:
                SparcLog("-" * 50)
                SparcLog(" GAUSSIAN CALCULATOR (non-periodic) ".center(50))
                SparcLog("-" * 50)
                SparcLog(f"  {'Command':<24} {exe_command or 'g16 (default)'}")
                SparcLog(f"  {'Label':<24} {label}")
                for k, v in params.items():
                    if v is None:
                        SparcLog(f"  {k:<24} (enabled)")
                    else:
                        SparcLog(f"  {k:<24} {v}")
                SparcLog("-" * 50)

            SparcLog("Gaussian calculator setup successfully")
            return calc

        except Exception as e:
            raise CalculatorError(f"Failed to create Gaussian calculator: {e}")


################################################################
# Helper Function
################################################################


def dft_calculator(config, print_screen=False):
    """
    Helper function to set up a DFT calculator based on configuration.

    This is the main entry point for creating DFT calculators. It automatically
    determines the calculator engine from the configuration and creates the
    appropriate ASE calculator object.

    Parameters
    ----------
    config : dict or SparcConfig
        Configuration dictionary or SparcConfig object containing calculator
        settings. Must include 'dft_calculator' section with 'engine' key.
    print_screen : bool, optional
        Whether to print calculator configuration details to log. Default is False.

    Returns
    -------
    ASE Calculator
        The configured ASE calculator instance (Vasp, CP2K, ORCA, or XTB)

    Raises
    ------
    CalculatorError
        If engine is unsupported or calculator setup fails

    Examples
    --------
    >>> config = load_config('input.yaml')
    >>> calc = dft_calculator(config, print_screen=True)
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()
    """
    calculator_setup = SetupDFTCalculator(config, print_screen)

    # Determine engine name
    if isinstance(config, SparcConfig):
        dft_config = config.dft_calculator
        engine = dft_config.engine
    else:
        dft_config = config["dft_calculator"]
        engine = dft_config.get("engine") or dft_config.get("name")

    engine_lower = engine.lower()

    SparcLog("")
    SparcLog(f"Setting up {engine} calculator...")

    if engine_lower == "vasp":
        return calculator_setup.vasp()
    elif engine_lower == "cp2k":
        return calculator_setup.cp2k()
    elif engine_lower == "orca":
        return calculator_setup.orca()
    elif engine_lower == "xtb":
        return calculator_setup.xtb()
    elif engine_lower == "qe":
        return calculator_setup.espresso()
    elif engine_lower == "gaussian":
        return calculator_setup.gaussian()
    else:
        raise CalculatorError(
            f"Unsupported calculator: {engine}. "
            f"Supported: VASP, CP2K, ORCA, xTB, QE, Gaussian"
        )


################################################################
# Main
################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up DFT calculator.")
    parser.add_argument(
        "-i", "--input_file", required=True, help="YAML configuration file"
    )
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print calculator details"
    )

    args = parser.parse_args()

    try:
        with open(args.input_file, "r") as f:
            config = yaml.safe_load(f)

        calc = dft_calculator(config, print_screen=args.print)

        engine = config["dft_calculator"].get("engine") or config["dft_calculator"].get(
            "name"
        )
        SparcLog(f"Calculator {engine} is set up successfully")

    except Exception as e:
        SparcLog(f"Error: {e}")
        raise

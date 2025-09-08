#!/usr/bin/python3
# calculator.py
################################################################
import os
from pathlib import Path
import argparse
import yaml
################################################################
# Third part import
from ase.calculators.vasp import Vasp
from ase.calculators.cp2k import CP2K
from ase.units import Rydberg

################################################################
# Local import
from sparc.src.utils.logger import SparcLog
from sparc.src.utils.read_incar import parse_incar
#===================================================================================================#
class SetupDFTCalculator:
    """
    Class to set up DFT calculators for VASP and CP2K using ASE atom objects.

    Parameters
    ----------
    input_config : dict
        Dictionary containing configuration for the DFT calculator.
    print_screen : bool, optional
        Whether to print details of the configuration. Default is False.
    """

    def __init__(self, input_config, print_screen=False):
        """
        Initializes the DFT calculator setup class.

        Parameters
        ----------
        input_config : dict
            Dictionary containing configuration for the DFT calculator.
        print_screen : bool, optional
            Whether to print details. Default is False.
        """
        if not isinstance(input_config, dict):
            raise ValueError("input_config must be a dictionary.")
        if 'dft_calculator' not in input_config:
            raise ValueError("Missing 'dft_calculator' key in input_config.")

        self.input_config = input_config
        self.print_screen = print_screen
        self.dft_config = input_config['dft_calculator']

    def vasp(self):
        """
        Set up the VASP calculator for ASE atom objects.

        Returns
        -------
        Vasp
            The configured VASP calculator object.
        """
        if self.dft_config['name'] != "VASP":
            raise ValueError("Unsupported DFT calculator. Only VASP is supported.")

        required_params = ['exe_command', 'prec', 'kgamma', 'incar_file']
        for param in required_params:
            if param not in self.dft_config:
                raise ValueError(f"{param} must be provided in dft_calculator config")

        exe_run = self.dft_config['exe_command']
        vasp_exe = Path(exe_run.split()[-1])
        if not vasp_exe.is_absolute() or not vasp_exe.exists():
            raise FileNotFoundError(f"VASP executable not found: {vasp_exe}")

        incar_path = Path(self.dft_config['incar_file'])
        if not incar_path.exists():
            raise FileNotFoundError(f"INCAR file not found: {incar_path}")

        incar_params = parse_incar(str(incar_path))

        if self.print_screen:
            SparcLog("="*50)
            SparcLog("              INCAR PARAMETERS                ")
            SparcLog("="*50)
            max_key_length = max(len(key) for key in incar_params.keys())
            for key, value in incar_params.items():
                SparcLog(f"  {key.upper():<{max_key_length}} : {value}")
            SparcLog("="*50 + "\n")

        gamma_point = self.dft_config.get('kgamma', False)

        calc = Vasp(
            prec=self.dft_config['prec'],
            kgamma=self.dft_config['kgamma'],
            gamma=not gamma_point,
            xc=self.dft_config.get('xc', 'PBE'),
            pp=self.dft_config.get('pp', 'PBE'),
            directory=self.dft_config.get('directory', 'vasp'),
            command=exe_run,
            **incar_params
        )

        return calc

    def cp2k(self):
        """
        Set up the CP2K calculator for ASE atom objects.

        Returns
        -------
        CP2K or None
            The configured CP2K calculator object, or None if an error occurs.
        """
        inpp = ''
        try:
            with open('cp2k_template.inp') as f:
                for line in f:
                    if not line.startswith('#') and not line.startswith('!'):
                        inpp += line.strip() + '\n'
        except FileNotFoundError:
            SparcLog("Error: Template file 'cp2k_template.inp' not found.")
            return None
        except Exception as e:
            SparcLog(f"An error occurred while reading the file: {e}")
            return None

        default_params = {
            'xc': 'PBE',
            'cutoff': 400 * Rydberg,
            'max_scf': 500,
            'basis_set_file': 'BASIS_SET',
            'basis_set': '6-31G*',
            'potential_file': 'GTH_POTENTIALS',
            'label': os.path.join('cp2k', 'job')
        }

        cp2k_config = self.input_config.get('cp2k', {})
        default_params.update(cp2k_config)

        calc = CP2K(
            command=self.dft_config.get('exe_command', 'cp2k.popt'),
            **default_params,
            inp=inpp
        )

        if self.print_screen:
            SparcLog("CP2K calculator setup with the following parameters:")
            for key, value in default_params.items():
                SparcLog(f"{key}: {value}")

        return calc

#===================================================================================================#
def dft_calculator(config, print_screen=False):
    """
    Helper function to set up the DFT calculator based on configuration.

    Parameters
    ----------
    config : dict
        Dictionary containing the full DFT configuration.
    print_screen : bool, optional
        Whether to print the calculator details to the screen.

    Returns
    -------
    ASE Calculator
        The configured ASE calculator instance.
    """
    calculator_name = config['dft_calculator']['name'].lower()
    calculator_setup = SetupDFTCalculator(config, print_screen)

    if calculator_name == 'vasp':
        return calculator_setup.vasp()
    elif calculator_name == 'cp2k':
        return calculator_setup.cp2k()
    else:
        raise ValueError(f"Unsupported calculator: {calculator_name}. Supported: VASP, CP2K")

#===================================================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up DFT calculator.")
    parser.add_argument('-i', '--input_file', required=True, help="YAML configuration file.")
    parser.add_argument('-p', '--print', action='store_true', help="Print calculator details.")
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        config = yaml.safe_load(f)

    calc = dft_calculator(config, print_screen=args.print)
    SparcLog(f"Calculator {config['dft_calculator']['name']} is set up successfully.")
#===================================================================================================#
# Enf of File
#===================================================================================================#

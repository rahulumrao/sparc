#!/usr/bin/python3
# plumed_wrapper.py
################################################################
import os
import sys
import numpy as np
################################################################
# Third party imports
import yaml
from ase.io import read, write
import ase.units
from ase.calculators.plumed import Plumed
# Local imports
from sparc.src.deepmd import setup_DeepPotential
from sparc.src.ase_md import NoseNVT, LangevinNVT, ExecuteMlpDynamics
from sparc.src.utils.logger import SparcLog
#===================================================================================================#
# Setup PLUMED
#===================================================================================================#
def modify_forces(calculator, system, timestep, kT, restart, plumed_input, iteration, PlumedLog='PLUMED.log'):
    """
    Set up and return a PLUMED wrapped ASE calculator for a molecular dynamics (MD) run.

    This function reads a PLUMED input file and wraps an existing ASE calculator
    to apply biasing forces during MD. Useful for enhanced sampling simulations.

    For more information, see the ASE documentation:
    https://wiki.fysik.dtu.dk/ase/ase/calculators/plumed.html#module-ase.calculators.plumed

    Parameters:
    ----------
    calculator : ase.calculators.Calculator
        The underlying ASE calculator for the system (e.g., VASP, CP2K)
    system : ase.Atoms
        The atomic configuration to which the calculator is applied
    timestep : float
        Integration timestep used in the MD simulation
    kT : float or None
        Thermal energy (k_B * T) in PLUMED units. If None, defaults will be calculated for 300 K (8.617333e-5 x 300 eV/K)
    restart : bool
        Restart option for PLUMED
    plumed_input : str
        Path to the PLUMED input file (e.g., "plumed.dat")

    Returns:
    -------
    plumed_calc : ase.calculators.plumed.Plumed
        The PLUMED-wrapped ASE calculator, ready for molecular dynamics simulations.
    """
    
    # Open the PLUMED input file and read its contents
    setup = [line for line in open(plumed_input, "r").read().splitlines() if not line.startswith('#')]
    
    modified_setup = []
    for line in setup:
        if "FILE=" in line:
            # Extract the part after FILE= and handle multiple filenames
            parts = line.split("FILE=")
            if len(parts) > 1:
                filenames = parts[1].strip()  # Get the filenames
                filenames = filenames.split(',')  # Handle multiple filenames
                filenames = [f"{iteration}/{filename.strip()}" for filename in filenames]
                new_filenames = ','.join(filenames)
                line = parts[0] + f"FILE={new_filenames}"
        modified_setup.append(line)
    SparcLog(f'PLUMED output files will be written in {iteration}')    
    if kT is None:
        kT = 8.617333e-5 * 300  # Boltzmann in eV/K at 300K
    
    # Initialize PLUMED calculator
    plumed_calc = Plumed(
                    calc=calculator, 
                    input=modified_setup, 
                    timestep=timestep, 
                    atoms=system, 
                    kT=kT, 
                    log=PlumedLog, 
                    restart=restart)
    # atoms.calc = plumed_calc

    # Return the configured PLUMED calculator
    return plumed_calc
#===================================================================================================#
# Umbrella Sampling
#===================================================================================================#
def umbrella(config, us_dir, dp_path, dp_model):   
    """
    This function sets up a PLUMED calculator for umbrella sampling simulations.
    It reads a configuration file defining the umbrella sampling windows and applies
    biasing forces to the system to keep the atoms within these windows.

    Parameters:
    config : dict   
        The configuration dictionary containing the umbrella sampling parameters.
    iter_structure : dict
        The iteration structure dictionary containing the iteration number and directory.
    base_dir : str
        The base directory for the umbrella sampling simulations.

    Returns:    
    None
    """
    # Open the PLUMED input file and read its contents
    # config['deepmd_setup']['umbrella_sampling'].get('config_file', 'umbrella_sampling.yaml')
    with open(config['deepmd_setup']['umbrella_sampling']['config_file']) as f:
        umbrella_config = yaml.safe_load(f)
    
    # Initialize
    # Initialize simulation parameters
    temperature = config.get('md_simulation', {}).get('temperature', 300)
    thermostat = config.get('md_simulation', {}).get('thermostat', 'Nose')
    thermostat_func = {'Nose': NoseNVT, 'Langevin': LangevinNVT}
    MDSteps = config.get('deepmd_setup', {}).get('md_steps', 0)
    # print(f"MD Steps {MDSteps}")
    # sys.exit(1)
    # Set system
    for w, window in enumerate(umbrella_config['umbrella_windows']):
        SparcLog(f"Running Umbrella Sampling for window {w}\n")
        struct_file = window['structure']
        plumed_file = window['plumed_file']
        window_dir = os.path.join(us_dir['dpmd_dir'], f"window_{w:03d}")
        os.makedirs(window_dir, exist_ok=True)
        usmd_log = os.path.join(f"window_{w:03d}", "usmd.log") # f"{us_dir['dpmd_dir']}/usmd.log"
        # print(f"Current Window: {window_dir} || {usmd_log}")
        # sys.exit(1)
        # Load structure and re-initialize system and calculator
        dp_atoms = read(struct_file)
        dp_atoms.write(os.path.join(window_dir, "input.xyz"))
        dp_atoms, dp_calc = setup_DeepPotential(
            atoms=dp_atoms,
            model_path=dp_path,
            model_name=dp_model
        )

        # Create new dynamics object
        if thermostat == 'Nose':
            dyn_dp = thermostat_func[thermostat](
                atoms=dp_atoms,
                timestep=config['deepmd_setup']['timestep_fs'] * ase.units.fs,
                tdamp=config['md_simulation']['tdamp'] * ase.units.fs,
                temperature=temperature
            )
        else:
            dyn_dp = thermostat_func[thermostat](
                atoms=dp_atoms,
                timestep=config['deepmd_setup']['timestep_fs'] * ase.units.fs,
                temperature=temperature,
                friction=config['md_simulation']['friction'] / ase.units.fs
            )

        # Wrap with PLUMED
        SparcLog(f"  → Window {w:03d} | Structure: {struct_file} | PLUMED: {plumed_file}")
        dp_atoms.calc = modify_forces(
            calculator=dp_calc,
            system=dp_atoms,
            timestep=config['deepmd_setup']['timestep_fs'] * ase.units.fs,
            kT=config['plumed']['kT'],
            restart=config['plumed']['restart'],
            plumed_input=plumed_file,
            iteration=window_dir,
            PlumedLog=os.path.join(us_dir['dpmd_dir'], f"window_{w:03d}", "PLUMED.log")
        )

        ExecuteMlpDynamics(
            system=dp_atoms,
            dyn=dyn_dp,
            steps=MDSteps,
            pace=config['deepmd_setup']['log_frequency'],
            log_filename=usmd_log,
            trajfile=config['output']['dptraj_file'],
            dir_name=us_dir['dpmd_dir'],
            distance_metrics=config['distance_metrics'],
            name=thermostat,
            epot_threshold=config['deepmd_setup']['epot_threshold']
        )
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#

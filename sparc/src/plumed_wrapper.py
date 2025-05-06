#!/usr/bin/python3
# plumed_wrapper.py
################################################################
import numpy as np
################################################################
# Third party inport
from ase.io import read, write
from ase.calculators.plumed import Plumed
from sparc.src.utils.logger import SparcLog
#===================================================================================================#
# Setup PLUMED
#===================================================================================================#
def modify_forces(calculator, system, timestep, kT, restart, plumed_input, iteration):
    """
    Set up and return a PLUMED-enhanced ASE calculator for a molecular dynamics (MD) run.

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
        Thermal energy (k_B * T) in PLUMED units. If None, defaults will be calculated for 300 K
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
                    log='PLUMED.log', 
                    restart=restart)
    # atoms.calc = plumed_calc

    # Return the configured PLUMED calculator
    return plumed_calc

#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#

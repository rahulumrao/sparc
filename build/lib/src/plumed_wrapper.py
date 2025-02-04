import numpy as np
from ase.io import read, write
from ase.calculators.plumed import Plumed
#===================================================================================================#
# Setup PLUMED
#===================================================================================================#
def modify_forces(calculator, system, timestep, temperature=300, kT=2.5, plumed_input='plumed.dat'):
    """
        Sets up the PLUMED calculator for AIMD run, modifying forces based on PLUMED input.
        For more information checkout the ASE documentation, see: 
        https://wiki.fysik.dtu.dk/ase/ase/calculators/plumed.html#module-ase.calculators.plumed
    Args:
    -----
        calculator: ASE calculator
            ASE calculator object for the system.
        system: ASE Atoms
            ASE Atoms object for the system.
        timestep: float
            Timestep for MD simulation (PLUMED units).
        temperature: float
            Target Temperature in Kelvin (K) (default: 300K).
        kT: float (optional)
            Thermal energy (PLUMED units).
        plumed_input: str
            Path to the PLUMED input file (default: "plumed.dat").
    """
    
    # Open the PLUMED input file and read its contents
    setup = [line for line in open(plumed_input, "r").read().splitlines() if not line.startswith('#')]
    
    # Initialize PLUMED calculator
    plumed_calc = Plumed(calc=calculator, input=setup, timestep=timestep, atoms=system, kT=2.5)
    # atoms.calc = plumed_calc

    # Return the configured PLUMED calculator
    return plumed_calc

#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#

import os
import sys
import numpy as np
from ase import Atoms
import ase.units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md import MDLogger

#---------------------------------------------------------------------------------------------------#
from src.utils import log_md_setup, save_xyz, wrap_positions  # Import these functions from utils module
#===================================================================================================#
def NoseNVT(atoms, timestep = 1 * ase.units.fs, temperature=300, tdamp=100 * ase.units.fs):
    """
    Nose-Hoover(constant N, V, T) molecular dynamics simulation setup.
      
    Args:
    -----
        atoms: ASE.atoms
        timestep (float): Timestep for MD simulation in fs.
        temperature (float): Target Temperature in Kelvin (K).
        tdamp (float): Damping time for Nose-Hoover thermostat in fs (ase units).
        
    Returns:
    --------
        dyn: ASE NoseHooverChainNVT object configured for simulation.
        
    Example Usage:
    
    ..code-block:: python

        NoseNVT = NoseHoover(
            atoms=atoms, 
            timestep=1 * ase.units.fs,
            temperature=300,
            tdamp=10 * ase.units.fs
            )
    """
    
    # Set initial velocities according to the traget temperature
    MaxwellBoltzmannDistribution(atoms, 
                                 temperature_K=temperature, 
                                 force_temp=True, 
                                 rng=np.random.RandomState(100))
    
    # Create the NoseHooverChainNVT dynamics object
    dyn = NoseHooverChainNVT(
        atoms=atoms, 
        timestep=timestep, 
        temperature_K=temperature, 
        tdamp=tdamp, 
        trajectory=None)

    return dyn

def run_md(system, dyn, steps, pace, log_filename, trajfile='md.traj'):
    """
        Run ab-initio molecular dynamics simulation for VASP, and setup logging save data at specified intervals.

    Args:
    -----
        system (Atoms): ASE.atom object.
        dyn: The dynamics object (e.g., NoseHooverChainNVT).
        steps (int): Number of MD steps to run.
        pace (int): Frequency of logging and data saving.
        log_filename (str): logfile for MD data.
        trajfile (str): trajectory filename (default: 'md.traj').

    """
    
    # Attach various functions to the dynamics object for handeling different tasks
    # dyn.attach(lambda: wrap_positions(system), interval=1)  # Wrap around periodic boundaries
    dyn.attach(lambda: log_md_setup(dyn, system), interval=pace)
    dyn.attach(lambda: save_xyz(system, trajfile), interval=pace)
    
    logger = MDLogger(
        dyn=dyn, 
        atoms=system, 
        logfile=log_filename, 
        header=True,                # Include the header in the log file
        stress=False,       
        peratom=False,              # Write per/atom energies
        mode="w")
    
    # Attach the logger to the dynamics object
    dyn.attach(logger, interval=pace)
    
    # Run the MD simulation for the specified number of steps.
    dyn.run(steps)
    
    
def run_Ase_DPMD(system, dyn, steps, pace, log_filename, trajfile='dpmd.traj'):
    """
        Run DeepPotential molecular dynamics simulation, and setup logging save data at specified intervals.

    Args:
    -----
        system (Atoms): ASE.atom object.
        dyn: The dynamics object (e.g., NoseHooverChainNVT).
        steps (int): Number of MD steps to run.
        pace (int): Frequency of logging and data saving.
        log_filename (str): logfile for MD data.
        trajfile (str): trajectory filename (default: 'md.traj').    
    """

    # Attach various functions to the dynamics object for handeling different tasks
    # dyn.attach(lambda: wrap_positions(apos=system), interval=pace)  # Wrap around periodic boundaries
    dyn.attach(lambda: log_md_setup(dyn, system), interval=pace)
    dyn.attach(lambda: save_xyz(system, trajfile), interval=pace)
    
    logger = MDLogger(
        dyn=dyn, 
        atoms=system, 
        logfile=log_filename, 
        header=True, 
        stress=False, 
        peratom=False, 
        mode="w")
    
    # Attach the logger to the dynamics object
    dyn.attach(logger, interval=pace)
    
    # Run the MD simulation for the specified number of md_steps.
    dyn.run(steps)
    
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#
#! ase_md.py
import os
import subprocess
import numpy as np
import dpdata
from ase import Atoms
import ase.units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md import MDLogger
#---------------------------------------------------------------------------------------------------#
from src.utils import log_md_setup, save_xyz, wrap_positions, load_md_checkpoint, save_md_checkpoint  # Import these functions from utils module
#===================================================================================================#
def NoseNVT(atoms, timestep=1 * ase.units.fs, temperature=300, tdamp=100 * ase.units.fs, 
            restart=False):
    """
    Nose-Hoover(constant N, V, T) molecular dynamics simulation setup.
      
    Args:
    -----
        atoms: ASE.atoms
        timestep (float): Timestep for MD simulation in fs.
        temperature (float): Target Temperature in Kelvin (K).
        tdamp (float): Damping time for Nose-Hoover thermostat in fs (ase units).
        restart (bool): Whether to restart from checkpoint (default: False)
        
    Returns:
    --------
        dyn: ASE NoseHooverChainNVT object configured for simulation.
        
    Example Usage:
    
    ..code-block:: python

        # New simulation
        dyn = NoseNVT(atoms=atoms, temperature=300)
        
        # Restart simulation
        dyn = NoseNVT(atoms=atoms, temperature=300, restart=True)
    """
    
    checkpoint_file = 'md_checkpoint.pkl'  # Default checkpoint filename
    
    if restart and os.path.exists(checkpoint_file):
        print(f"\nRestarting simulation from checkpoint: {checkpoint_file}")
        # Load state from checkpoint
        state = load_md_checkpoint()  # Uses default filename
        atoms.set_positions(state['positions'])
        atoms.set_velocities(state['velocities'])
        atoms.set_cell(state['cell'])
        atoms.set_pbc(state['pbc'])
        atoms.set_atomic_numbers(state['numbers'])
        atoms.set_momenta(state['momenta'])
        
        # Create the NoseHooverChainNVT dynamics object
        dyn = NoseHooverChainNVT(
            atoms=atoms, 
            timestep=timestep, 
            temperature_K=temperature, 
            tdamp=tdamp, 
            trajectory=None)
        
        # Restore the step number
        dyn.nsteps = state['step']  # Set the step counter
        
    else:
        if restart:
            print(f"\nWarning: Checkpoint file {checkpoint_file} not found. Starting new simulation.")
        # Set initial velocities for new simulation
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
        
        # Create new dynamics object
        dyn = NoseHooverChainNVT(
            atoms=atoms, 
            timestep=timestep, 
            temperature_K=temperature, 
            tdamp=tdamp, 
            trajectory=None)
    
    return dyn

def run_aimd(system, dyn, steps, pace, log_filename, trajfile, header=True, mode="a"):
    """
    Run ab-initio molecular dynamics simulation for VASP, and setup logging save data at specified intervals.

    Args:
    -----
        system (Atoms): ASE.atom object.
        dyn: The dynamics object (e.g., NoseHooverChainNVT).
        steps (int): Number of MD steps to run.
        pace (int): Frequency of logging and data saving.
        log_filename (str): logfile for MD data.
        trajfile (str): trajectory filename (default: 'AseMD.traj').
        header (bool): Whether to write header in log file (default: True)
        mode (str): 'w' or 'a' for writing or appending to the trajectory file.
    """
    
    if (steps > 0):
        print("\n========================================================================")
        print("{}".format("Starting Molecular Dynamics Simulation".center(72)))
        print("========================================================================")
    
    # Attach checkpoint saving using default filename
    dyn.attach(lambda: save_md_checkpoint(dyn, system), interval=pace)
    
    # Attach various functions to the dynamics object for handling different tasks
    # dyn.attach(lambda: wrap_positions(system), interval=1)  # Wrap around periodic boundaries
    # dyn.attach(lambda: MolecularDynamics.checkpoint(dyn, 'md.checkpoint', interval=pace))
    dyn.attach(lambda: log_md_setup(dyn, system), interval=pace)
    dyn.attach(lambda: save_xyz(system, trajfile, mode), interval=pace)
    
    # Only write header if it's a new simulation (not a restart) and header is True
    write_header = header and dyn.get_number_of_steps() == 0
    
    logger = MDLogger(
        dyn=dyn, 
        atoms=system, 
        logfile=log_filename, 
        header=header,                # Include the header in the log file
        stress=False,       
        peratom=False,              # Write per/atom energies
        mode=mode)
    
    # Attach the logger to the dynamics object
    dyn.attach(logger, interval=pace)
    
    # Run the MD simulation for the specified number of steps.
    dyn.run(steps)
    
def run_Ase_DPMD(system, dyn, steps, pace, log_filename, trajfile):
    """
        Run DeepPotential molecular dynamics simulation, and setup logging save data at specified intervals.

    Args:
    -----
        system (Atoms): ASE.atom object.
        dyn: The dynamics object (e.g., NoseHooverChainNVT).
        steps (int): Number of MD steps to run.
        pace (int): Frequency of logging and data saving.
        log_filename (str): logfile for MD data.
        trajfile (str): trajectory filename (default: 'dpmd.traj').    
        mode (str): 'w' or 'a' for writing or appending to the trajectory file.
    """
    print("\n========================================================================")
    print("{}".format("Initializing DeepPotential MD Simulation".center(72)))
    print("========================================================================")

    # Attach various functions to the dynamics object for handeling different tasks
    # dyn.attach(lambda: wrap_positions(apos=system), interval=pace)  # Wrap around periodic boundaries
    dyn.attach(lambda: log_md_setup(dyn, system), interval=pace)
    dyn.attach(lambda: save_xyz(system, trajfile, 'a'), interval=pace)
    #
    logger = MDLogger(
        dyn=dyn, 
        atoms=system, 
        logfile=log_filename, 
        header=True, 
        stress=False, 
        peratom=False, 
        mode='a')
    
    # Attach the logger to the dynamics object
    dyn.attach(logger, interval=pace)
    
    # Run the MD simulation for the specified number of md_steps.
    dyn.run(steps)
    
    
def lammps_md(system, model_path, model_name):
    print("\n========================================================================")
    print("{}".format("Starting LAMMPS MD Simulation".center(72)))
    print("========================================================================")
    
    # RUN the MD simulation for the specified number of md_steps
    run_command = (['lmp', '-i', 'in.lammps'])
    try:
        subprocess.run(run_command, check=True)
        print("\n========================================================================")
        print("!{}!".format("LAMMPS MD Simulation Completed Successfully".center(70)))
        print("========================================================================")
    except subprocess.CalledProcessError as e:
        print("\n========================================================================")
        print("!{}!".format("Error in LAMMPS MD Simulation".center(70)))
        print("!{}!".format(str(e).center(70)))
        print("========================================================================")
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#
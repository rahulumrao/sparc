#!/usr/bin/python3
# ase_md.py
# Standard library imports
import os
import subprocess
import numpy as np
################################################################
# Third party imports
import ase.units
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.langevin import Langevin
from ase.md import MDLogger
################################################################
# Local imports
from sparc.src.utils.logger import SparcLog
from sparc.src.utils.utils import (
    log_md_setup,
    save_xyz,
    save_checkpoint,
    load_checkpoint,
    check_physical_limits
)
################################################################
#===================================================================================================#
# Helper Functions
#===================================================================================================#
def initialize_dynamics(atoms, dyn_class, timestep, temperature, restart, **kwargs):
    """
    Initialize MD dynamics, optionally restarting from a checkpoint.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the system.
    dyn_class : type
        The MD dynamics class to use (e.g., NoseHooverChainNVT, Langevin).
    timestep : float
        The simulation time step in femtoseconds.
    temperature : float
        The target temperature in Kelvin.
    restart : bool
        If True, the simulation will be restarted from a checkpoint.
    **kwargs : dict
        Additional keyword arguments. For example, 'checkpoint_file' (default is 'md_checkpoint.pkl').

    Returns
    -------
    dyn : instance of dyn_class
        The initialized dynamics object.
    """
    checkpoint_file = kwargs.get('checkpoint_file', 'md_checkpoint.pkl')

    if restart:
        atoms, mdstep = load_checkpoint(atoms, checkpoint_file)
        dyn = dyn_class(atoms, timestep=timestep, temperature_K=temperature, **kwargs)
        dyn.nsteps = mdstep
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
        dyn = dyn_class(atoms, timestep=timestep, temperature_K=temperature, **kwargs)

    return dyn

#===================================================================================================#
# Thermostat Functions
#===================================================================================================#
def NoseNVT(atoms, timestep=1, temperature=300, tdamp=10, restart=False):
    """
    Set up a Nose-Hoover chain NVT thermostat for MD simulation.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the system.
    timestep : float, optional
        The simulation time step in femtoseconds (default is 1 fs).
    temperature : float, optional
        The target temperature in Kelvin (default is 300 K).
    tdamp : float, optional
        The damping time for the thermostat in femtoseconds (default is 10 fs).
    restart : bool, optional
        If True, the simulation will be restarted from a checkpoint.

    Returns
    -------
    dynamics : NoseHooverChainNVT
        The initialized dynamics object using the Nose-Hoover chain thermostat.
    """
    return initialize_dynamics(
        atoms, NoseHooverChainNVT, timestep * ase.units.fs, temperature, restart, tdamp=tdamp * ase.units.fs
    )


def LangevinNVT(atoms, timestep=1, temperature=300, friction=0.01, restart=False):
    """
    Set up a Langevin thermostat for NVT MD simulation.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the system.
    timestep : float, optional
        The simulation time step in femtoseconds (default is 1 fs).
    temperature : float, optional
        The target temperature in Kelvin (default is 300 K).
    friction : float, optional
        The friction coefficient for the Langevin thermostat (default is 0.01 fs^-1).
    restart : bool, optional
        If True, the simulation will be restarted from a checkpoint.

    Returns
    -------
    dynamics : Langevin
        The initialized dynamics object using the Langevin thermostat.
    """
    return initialize_dynamics(
        atoms, Langevin, timestep * ase.units.fs, temperature, restart, friction=friction / ase.units.fs
    )

#===================================================================================================#
# MD Execution Functions
#===================================================================================================#
def ExecuteAbInitioDynamics(system, dyn, steps, pace, log_filename, trajfile, dir_name, name):
    """
    Run an ab initio MD simulation.

    Parameters
    ----------
    system : ase.Atoms
        The ASE Atoms object representing the system.
    dyn : dynamics object
        The initialized MD dynamics object.
    steps : int
        The number of MD steps to run.
    pace : int
        The interval (in steps) at which to log data and save checkpoints.
    log_filename : str
        The filename for the MD log file.
    trajfile : str
        The filename for the trajectory file.
    dir_name : str
        The directory where log and trajectory files will be saved.
    name : str
        A label for the simulation (e.g., the thermostat type).

    Returns
    -------
    None
    """
    if steps <= 0:
        return

    SparcLog("\n" + "=" * 72 + "\n")
    SparcLog(f"Starting AIMD Simulation [{name}]".center(72) + "\n")
    SparcLog("=" * 72 + "\n")

    dyn.attach(lambda: save_checkpoint(dyn, system), interval=pace)
    dyn.attach(lambda: log_md_setup(dyn, system, dir_name), interval=pace)
    dyn.attach(lambda: save_xyz(system, trajfile, 'a', dir_name), interval=pace)

    logger = MDLogger(
        dyn=dyn, atoms=system, logfile=f"{dir_name}/{log_filename}",
        header=True, stress=False, peratom=False, mode='a'
    )
    dyn.attach(logger, interval=pace)

    dyn.run(steps)


def ExecuteMlpDynamics(system, dyn, steps, pace, log_filename, trajfile, dir_name, distance_metrics, name, epot_threshold):
    """
    Run a Deep Potential MD simulation.

    Parameters
    ----------
    system : ase.Atoms
        The ASE Atoms object representing the system.
    dyn : dynamics object
        The initialized MD dynamics object.
    steps : int
        The number of MD steps to run.
    pace : int
        The interval (in steps) at which to log data.
    log_filename : str
        The filename for the MD log file.
    trajfile : str
        The filename for the trajectory file.
    dir_name : str
        The directory where log and trajectory files will be saved.
    distance_metrics : dict or list
        Metrics used to check physical limits during the simulation.
    name : str
        A label for the simulation (e.g., the thermostat type).

    Returns
    -------
    None
    """
    SparcLog("\n" + "=" * 72 + "\n")
    SparcLog(f"Initializing DeepPotential MD Simulation [{name}]".center(72) + "\n")
    SparcLog("=" * 72 + "\n")

    dyn.attach(lambda: log_md_setup(dyn, system, dir_name), interval=pace)
    dyn.attach(lambda: save_xyz(system, trajfile, 'a', dir_name), interval=pace)

    logger = MDLogger(
        dyn=dyn, atoms=system, logfile=f"{dir_name}/{log_filename}",
        header=True, stress=False, peratom=False, mode='a'
    )
    dyn.attach(logger, interval=pace)
    
    # Store reference energy for comparison
    epot_ref = None
    for step in range(steps):
        dyn.run(1)
        
        # Check if physical limits are exceeded
        if distance_metrics and check_physical_limits(system, distance_metrics):
            SparcLog("Physical limits exceeded. Stopping MLMD simulation!!!", level="WARNING")
            break
        
        # Check if the Potential Energy becomes undefined
        epot = np.array(system.get_potential_energy())
        if np.isnan(epot):
            SparcLog("Potential Energy is Nan! || Stopping MLMD simulation !!!\n", level="ERROR")
            break

        # Store reference energy in the first step
        if epot_ref is None:
            epot_ref = epot
            SparcLog(f"Reference Potential Energy//STEP:0 -> {epot_ref}")

        # Check if the Potential Energy is too high
        Llim = epot_ref - epot_threshold
        Ulim = epot_ref + epot_threshold
        if epot > Ulim or epot < Llim:
            SparcLog("-" * 72, level="ERROR")
            SparcLog("Potential Energy Exceeded Limit:", level="ERROR")
            SparcLog(f"    Reference_eV  : {float(epot_ref): .2f} eV", level="ERROR")
            SparcLog(f"    Threshold_eV  : {float(epot_threshold): .2f} eV", level="ERROR")
            SparcLog(f"    Lower_limit   : {float(Llim): .2f} eV", level="ERROR")
            SparcLog(f"    Upper_limit   : {float(Ulim): .2f} eV", level="ERROR")
            SparcLog(f"    Current_eV    : {float(epot): .2f} eV", level="ERROR")
            SparcLog("Stopping ML/MD Simulation!!!", level="ERROR")
            SparcLog("-" * 72, level="ERROR")
            break


def CalculateDFTEnergy(idx, header, system, timestep, log_filename, dir_name, trajfile, pace=1):
    """
    Calculate the DFT energy and forces for a candidate structure.

    Parameters
    ----------
    idx : int
        An identifier index for the candidate.
    header : bool
        If True, include a header in the log.
    system : ase.Atoms
        The ASE Atoms object representing the candidate structure.
    timestep : float
        The simulation time step in femtoseconds.
    log_filename : str
        The filename for the energy log.
    dir_name : str
        The directory where log and trajectory files will be saved.
    trajfile : str
        The filename for the trajectory file.
    pace : int, optional
        The logging interval (default is 1).

    Returns
    -------
    None
    """
    dyn = VelocityVerlet(system, timestep, trajectory=None)
    dyn.attach(lambda: save_xyz(system, trajfile, 'a', dir_name), interval=pace)

    epot = system.get_potential_energy()
    epot = epot if not isinstance(epot, (list, np.ndarray)) else epot[0]

    SparcLog(f"Candidate: {idx:5d} | Epot: {epot:10.6f} [eV]\n")

    log = MDLogger(
        dyn=dyn, atoms=system, logfile=f"{dir_name}/{log_filename}",
        header=header, stress=False, peratom=False, mode='a'
    )
    dyn.attach(log, interval=pace)
    dyn.run(0)


#===================================================================================================#
# LAMMPS MD Execution
#===================================================================================================#
def lammps_md(system, model_path, model_name):
    """
    Run a LAMMPS MD simulation.

    Parameters
    ----------
    system : ase.Atoms
        The ASE Atoms object representing the system (not used directly in the call).
    model_path : str
        The path to the model files required by LAMMPS.
    model_name : str
        The name of the model to be used in the simulation.

    Returns
    -------
    None
    """
    SparcLog("\n" + "=" * 72)
    SparcLog("Starting LAMMPS MD Simulation".center(72))
    SparcLog("=" * 72)

    run_command = ['lmp', '-i', 'in.lammps']
    try:
        subprocess.run(run_command, check=True)
        SparcLog("\n" + "=" * 72)
        SparcLog("LAMMPS MD Simulation Completed Successfully".center(72))
        SparcLog("=" * 72)
    except subprocess.CalledProcessError as e:
        SparcLog("\n" + "=" * 72)
        SparcLog("Error in LAMMPS MD Simulation".center(72))
        SparcLog(str(e).center(72))
        SparcLog("=" * 72)


#===================================================================================================#
# Standalone Demonstration
#===================================================================================================#
if __name__ == '__main__':
    """
    Standalone demonstration of the ase_md module.

    This example creates a simple H2 molecule, sets up a Nose-Hoover NVT simulation,
    and runs a short MD simulation for demonstration purposes. The MD log and trajectory
    files are saved in the current directory.
    """
    from ase import Atoms

    # Create a simple diatomic molecule (H₂)
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.set_pbc([False, False, False])

    # Simulation parameters
    timestep = 1 * ase.units.fs
    temperature = 300  # Kelvin
    steps = 10
    pace = 1

    # Display a message to the user
    SparcLog("Running MD simulation with Nose-Hoover Thermostat...\n")

    # Initialize dynamics using Nose-Hoover thermostat
    dyn_test = NoseNVT(atoms, timestep=timestep, temperature=temperature, tdamp=10 * ase.units.fs, restart=False)

    # Run the simulation (log file: demo_md.log, trajectory: demo_traj.xyz, saved in current directory)
    ExecuteAbInitioDynamics(
         system=atoms,
         dyn=dyn_test,
         steps=steps,
         pace=pace,
         log_filename='md.log',
         trajfile='traj.xyz',
         dir_name='.',
         name='Nose'
    )

    SparcLog("Simulation completed.\n")
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================# 

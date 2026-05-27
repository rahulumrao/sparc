#!/usr/bin/python3
# ase_md.py

"""
ASE-based Molecular Dynamics module for SPARC package.

This module provides MD simulation capabilities using ASE's MD integrators,
supporting various ensembles (NVE, NVT, NPT) with different thermostats.
"""

import os
import subprocess
from typing import Dict, List, Optional

import numpy as np

################################################################
# Third party imports
from ase import Atoms, units
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

################################################################
# Local imports
from sparc.src.utils.logger import SparcLog
from sparc.src.utils.utils import (
    check_physical_limits,
    load_checkpoint,
    log_md_setup,
    save_checkpoint,
    save_xyz,
)

################################################################
# Helper Functions
################################################################


# ---------------------------------------
# Dynamics Initialization
# ---------------------------------------
def initialize_dynamics(
    atoms: Atoms, dyn_class, timestep: float, restart: bool = False, **kwargs
):
    """
    Initialize MD dynamics, optionally restarting from a checkpoint.

    This is a unified initialization function that handles velocity initialization,
    checkpoint loading, and dynamics object creation for all ensemble types.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the system
    dyn_class : type
        The MD dynamics class (NoseHooverChainNVT, Langevin, NPTBerendsen, VelocityVerlet)
    timestep : float
        The simulation timestep in ASE internal units (already converted)
    restart : bool, optional
        If True, restart from checkpoint (default: False)
    **kwargs : dict
        Additional parameters for specific ensemble types

    Returns
    -------
    dyn
        The initialized dynamics object
    """
    cfg = {
        "temperature": None,
        "friction": None,
        "tdamp": None,
        "taut": None,
        "pressure_au": None,
        "taup": None,
        "compressibility_au": None,
        "checkpoint_file": "md_checkpoint.pkl",
    }

    cfg.update(kwargs)
    checkpoint_file = cfg.pop("checkpoint_file")

    if restart:
        atoms, mdstep = load_checkpoint(atoms, checkpoint_file)
    else:
        # Only initialize velocities if temperature is provided (not for NVE)
        if cfg["temperature"] is not None:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=cfg["temperature"], force_temp=True
            )

    # Filter out None values and temperature (handled separately)
    _kwargs = {k: v for k, v in cfg.items() if v is not None and k != "temperature"}

    # Create dynamics object
    if cfg["temperature"] is not None:
        dyn = dyn_class(
            atoms, timestep=timestep, temperature_K=cfg["temperature"], **_kwargs
        )
    else:
        dyn = dyn_class(atoms, timestep=timestep)

    if restart:
        dyn.nsteps = mdstep

    return dyn


################################################################
# Temperature Ramping (Linear)
################################################################
def TemperatureRamp(
    dyn,
    atoms: Atoms,
    current_step: int,
    total_steps: int,
    temp_start: Optional[float] = None,
    temp_end: Optional[float] = None,
    ensemble: str = "NVT",
):
    """
    Apply linear temperature ramping to MD dynamics object.

    Implements VASP-style temperature ramping:
        T(t) = T_start + (T_end - T_start) * (t / t_total)

    Temperature ramping is only applicable to NVT ensemble.

    Parameters
    ----------
    dyn : dynamics object
        MD dynamics object (NoseHooverChainNVT, Langevin, NPTBerendsen)
    atoms : ase.Atoms
        ASE Atoms object
    current_step : int
        Current MD step
    total_steps : int
        Total MD steps
    temp_start : float, optional
        Starting temperature (K). If None, no ramping applied
    temp_end : float, optional
        Ending temperature (K). If None, no ramping applied
    ensemble : str, optional
        MD ensemble ('NVT', 'NPT', 'NVE'). Default: 'NVT'
    """
    # Early return if no ramping specified
    if temp_start is None or temp_end is None:
        return

    # Check ensemble compatibility
    if ensemble.upper() == "NVE":
        raise ValueError(
            "Temperature ramping not applicable for NVE ensemble (energy conservation). "
            "Remove 'temp_end' from YAML or change to ensemble: NVT"
        )

    if ensemble.upper() == "NPT":
        SparcLog(
            " Temperature ramping not recommended for NPT ensemble (pressure-temperature coupling).",
            level="WARNING",
        )
        SparcLog(
            " Solution: Use NVT for ramping, then switch to NPT at final temperature",
            level="WARNING",
        )
        SparcLog(
            " Alternative: Remove 'temp_end' from YAML configuration", level="WARNING"
        )
        return

    if ensemble.upper() != "NVT":
        SparcLog(
            f" Temperature ramping only supported for NVT ensemble (current: {ensemble})",
            level="WARNING",
        )
        SparcLog(" Solution: Set ensemble: NVT in YAML configuration", level="WARNING")
        return

    # Avoid division by zero
    if total_steps == 0:
        return

    # Calculate target temperature: T(t) = T_start + (T_end - T_start) * (t / t_total)
    progress = current_step / total_steps
    target_temp = temp_start + (temp_end - temp_start) * progress
    scale = target_temp / temp_start
    # if current_step == 100:
    #     print(f"TempRamp: {current_step, atoms.get_temperature(), target_temp}")
    # sys.exit(1)
    # Apply temperature scaling
    if abs(scale - 1.0) > 1e-10:
        current_temp = atoms.get_temperature()

        if current_temp > 0:
            velocity_scale = np.sqrt(target_temp / current_temp)
            velocities = atoms.get_velocities()
            atoms.set_velocities(velocities * velocity_scale)

            # Update thermostat if supported
            # For Langevin thermostat (has set_temperature method)
            if hasattr(dyn, "set_temperature"):
                dyn.set_temperature(temperature_K=target_temp)
            # For Nose-Hoover thermostat (stores temperature in _kT)
            elif hasattr(dyn, "_kT"):
                dyn._kT = units.kB * target_temp
            # For thermostats that store temp attribute directly
            elif hasattr(dyn, "temp"):
                dyn.temp = units.kB * target_temp


################################################################
# Thermostat Functions
################################################################


# ---------------------------------------
# Nose-Hoover Chain Thermostat
# ---------------------------------------
def NoseNVT(
    atoms: Atoms,
    timestep: float = 1,
    temperature: float = 300,
    tdamp: float = 10,
    restart: bool = False,
):
    """
    Set up a Nose-Hoover chain NVT thermostat for MD simulation.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the system
    timestep : float, optional
        The simulation timestep in femtoseconds (default: 1 fs)
    temperature : float, optional
        The target temperature in Kelvin (default: 300 K)
    tdamp : float, optional
        The damping time for the thermostat in femtoseconds (default: 10 fs)
    restart : bool, optional
        If True, restart from checkpoint (default: False)

    Returns
    -------
    dynamics
        The initialized NoseHooverChainNVT dynamics object
    """
    return initialize_dynamics(
        atoms,
        NoseHooverChainNVT,
        timestep * units.fs,
        temperature=temperature,
        tdamp=tdamp * units.fs,
        restart=restart,
    )


# ---------------------------------------
# Langevin Thermostat
# ---------------------------------------
def LangevinNVT(
    atoms: Atoms,
    timestep: float = 1,
    temperature: float = 300,
    friction: float = 0.01,
    restart: bool = False,
):
    """
    Set up a Langevin thermostat for NVT MD simulation.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the system
    timestep : float, optional
        The simulation timestep in femtoseconds (default: 1 fs)
    temperature : float, optional
        The target temperature in Kelvin (default: 300 K)
    friction : float, optional
        The friction coefficient in fs^-1 (default: 0.01)
    restart : bool, optional
        If True, restart from checkpoint (default: False)

    Returns
    -------
    dynamics
        The initialized Langevin dynamics object
    """
    return initialize_dynamics(
        atoms,
        Langevin,
        timestep * units.fs,
        restart=restart,
        temperature=temperature,
        friction=friction / units.fs,
    )


# ---------------------------------------
# NPT Ensemble (Berendsen)
# ---------------------------------------
def NPT(
    system: Atoms,
    timestep: float,
    temperature: float,
    tau_t: float,
    pressure: float,
    tau_p: float,
    compressibility: Optional[float] = None,
    restart: Optional[bool] = False,
    **kwargs,
):
    """
    Set up NPT dynamics using ASE's NPTBerendsen integrator.

    Parameters
    ----------
    system : ase.Atoms
        The ASE Atoms object representing the system
    timestep : float
        MD integration timestep in femtoseconds
    temperature : float
        Target temperature in Kelvin
    tau_t : float
        Thermostat time constant in femtoseconds
    pressure : float
        Target pressure in bar
    tau_p : float
        Barostat time constant in femtoseconds
    compressibility : float, optional
        Isothermal compressibility in 1/bar. If None, uses the default
        for Cu (~7.1e-7 bar⁻¹)
    restart : bool, optional
        If True, restart from checkpoint (default: False)
    **kwargs : dict
        Extra ASE MD options

    Returns
    -------
    dynamics
        The initialized NPTBerendsen dynamics object
    """
    if compressibility is None:
        compressibility = 7.1e-7  # Cu default in 1/bar (= 7.1e-12 Pa⁻¹)

    return initialize_dynamics(
        system,
        NPTBerendsen,
        timestep * units.fs,
        restart=restart,
        temperature=temperature,
        taut=tau_t * units.fs,
        pressure_au=pressure * units.bar,
        taup=tau_p * units.fs,
        compressibility_au=compressibility / units.bar,
        **kwargs,
    )


# ---------------------------------------
# NVE Ensemble
# ---------------------------------------
def NVE(system: Atoms, timestep: float, restart: Optional[bool] = False):
    """
    Set up NVE dynamics using ASE's VelocityVerlet integrator.

    Parameters
    ----------
    system : ase.Atoms
        The ASE Atoms object representing the system
    timestep : float
        MD integration timestep in femtoseconds
    restart : bool, optional
        If True, restart from checkpoint (default: False)

    Returns
    -------
    dynamics
        The initialized VelocityVerlet dynamics object
    """
    return initialize_dynamics(system, VelocityVerlet, timestep * units.fs, restart)


################################################################
# MD Execution Functions
################################################################


# ---------------------------------------
# Ab Initio MD
# ---------------------------------------
def ExecuteAbInitioDynamics(
    system: Atoms,
    dyn,
    steps: int,
    pace: int,
    log_filename: str,
    trajfile: str,
    dir_name: str,
    name: str,
    temp_start: Optional[float] = None,
    temp_end: Optional[float] = None,
):
    """
    Run an ab initio MD simulation with DFT calculator.

    Parameters
    ----------
    system : ase.Atoms
        The ASE Atoms object representing the system
    dyn : dynamics object
        The initialized MD dynamics object
    steps : int
        The number of MD steps to run
    pace : int
        The interval (in steps) for logging and saving checkpoints
    log_filename : str
        The filename for the MD log file
    trajfile : str
        The filename for the trajectory file
    dir_name : str
        The directory where log and trajectory files will be saved
    name : str
        A label for the simulation (e.g., thermostat type)
    """
    if steps <= 0:
        return

    steps_completed = dyn.nsteps
    if steps_completed >= steps:
        SparcLog(f"AIMD already completed ({steps_completed}/{steps} steps). Skipping.")
        return

    remaining_steps = steps - steps_completed

    SparcLog("")
    SparcLog("-" * 80)
    if steps_completed > 0:
        SparcLog(
            f"Resuming AIMD from step {steps_completed}, running {remaining_steps} more steps"
        )
    # Print table header
    SparcLog(
        f"{'Step':<8} {'Epot (eV)':<12} {'Ekin (eV)':<12} {'Temp (K)':<10} {'P (GPa)':<10} {'V (Ang.^3)':<10}"
    )
    SparcLog("-" * 80)

    dyn.attach(lambda: save_checkpoint(dyn, system), interval=pace)
    dyn.attach(lambda: log_md_setup(dyn, system, dir_name), interval=pace)
    dyn.attach(lambda: save_xyz(system, trajfile, "a", dir_name), interval=pace)

    # Extract base ensemble from name (e.g., 'NVT-Langevin' -> 'NVT')
    ensemble = name.split("-")[0] if "-" in name else name

    if temp_end is not None and temp_start is not None:
        for i_md in range(remaining_steps):
            TemperatureRamp(
                dyn,
                system,
                steps_completed + i_md,
                steps,
                temp_start,
                temp_end,
                ensemble,
            )
            dyn.run(1)
    else:
        dyn.run(remaining_steps)


# ---------------------------------------
# Machine Learning Potential MD
# ---------------------------------------
def ExecuteMlpDynamics(
    system: Atoms,
    dyn,
    steps: int,
    pace: int,
    log_filename: str,
    trajfile: str,
    dir_name: str,
    distance_metrics: Optional[List[Dict]],
    name: str,
    epot_threshold: float,
    temp_start: Optional[float] = None,
    temp_end: Optional[float] = None,
    restart: bool = False,
):
    """
    Run a machine learning potential MD simulation with safety checks.

    Supports restart from checkpoint saved inside dir_name.

    Parameters
    ----------
    system : ase.Atoms
        The ASE Atoms object representing the system
    dyn : dynamics object
        The initialized MD dynamics object
    steps : int
        The number of MD steps to run
    pace : int
        The interval (in steps) for logging
    log_filename : str
        The filename for the MD log file
    trajfile : str
        The filename for the trajectory file
    dir_name : str
        The directory where log, trajectory and checkpoint files are saved
    distance_metrics : list of dict or None
        Metrics used to check physical limits during simulation
    name : str
        A label for the simulation (e.g., thermostat type)
    epot_threshold : float
        Maximum allowed energy deviation from reference (eV)
    restart : bool
        If True, resume from checkpoint in dir_name (default: False)
    """
    # Checkpoint file lives inside the simulation directory (e.g., 02.dpmd/)
    checkpoint_file = os.path.join(str(dir_name), "md_checkpoint.pkl")
    steps_completed = 0

    if restart and os.path.exists(checkpoint_file):
        system, steps_completed = load_checkpoint(system, checkpoint_file)
        dyn.nsteps = steps_completed
        SparcLog("=" * 80)
        SparcLog("RESTARTING ML-MD FROM CHECKPOINT")
        SparcLog(f"  Checkpoint : {checkpoint_file}")
        SparcLog(f"  Completed  : {steps_completed}/{steps} steps")
        SparcLog("=" * 80 + "\n")
        if steps_completed >= steps:
            SparcLog("ML-MD already completed. Skipping.")
            return

    remaining_steps = steps - steps_completed

    SparcLog("=" * 80)
    SparcLog(f"MACHINE LEARNING POTENTIAL MD SIMULATION FOR [{name}]".center(80))
    SparcLog(f"Output Logfile: {log_filename}")
    if steps_completed > 0:
        SparcLog(
            f"Resuming from step {steps_completed}, running {remaining_steps} more steps"
        )
    SparcLog("=" * 80)
    # Print table header
    SparcLog(
        f"{'Step':<8} {'Epot (eV)':<12} {'Ekin (eV)':<12} {'Temp (K)':<10} {'P (GPa)':<10} {'V (Ang^3)':<10}"
    )
    SparcLog("-" * 80)
    # Console output every 10*pace steps
    console_pace = 10 * pace
    dyn.attach(lambda: log_md_setup(dyn, system, dir_name), interval=console_pace)
    dyn.attach(lambda: save_checkpoint(dyn, system, checkpoint_file), interval=pace)

    # Per-iteration log file (e.g. Iter4_dpmd_0.log) — mirrors GitHub MDLogger behaviour
    logger = MDLogger(
        dyn=dyn,
        atoms=system,
        logfile=f"{dir_name}/{log_filename}",
        header=True,
        stress=False,
        peratom=False,
        mode="a",
    )
    dyn.attach(logger, interval=pace)

    # Extract base ensemble from name (e.g., 'NVT-Langevin' -> 'NVT')
    ensemble = name.split("-")[0] if "-" in name else name

    # Capture reference energy from step 0 before any MD runs
    _epot0 = system.get_potential_energy()
    if isinstance(_epot0, (list, np.ndarray)):
        _epot0 = float(_epot0.item() if hasattr(_epot0, "item") else _epot0[0])
    else:
        _epot0 = float(_epot0)
    epot_ref = _epot0
    SparcLog("***********************************************************")
    SparcLog(f"Reference Potential Energy (Step 0): {epot_ref:.6f} eV")
    if epot_threshold is not None:
        SparcLog(f"Threshold limit: +/- {float(epot_threshold):.2f} eV", level="INFO")
    SparcLog("***********************************************************")

    def _check_mlmd_safety(
        system, distance_metrics, epot_ref, epot_threshold, dir_name
    ):
        """Check safety conditions during MLMD. Returns (sim_failed, epot_ref)."""
        if distance_metrics and check_physical_limits(system, distance_metrics):
            SparcLog(
                "Physical limits exceeded. Stopping MLMD simulation!!!", level="WARNING"
            )
            return True, epot_ref

        epot = system.get_potential_energy()
        if isinstance(epot, (list, np.ndarray)):
            epot = float(epot.item() if hasattr(epot, "item") else epot[0])
        else:
            epot = float(epot)
        if np.isnan(epot):
            SparcLog(
                "Potential Energy is NaN! Stopping MLMD simulation!", level="ERROR"
            )
            return True, epot_ref

        if epot_threshold is not None:
            Llim = epot_ref - epot_threshold
            Ulim = epot_ref + epot_threshold
            if epot > Ulim or epot < Llim:
                SparcLog(f"{f'Iteration {dir_name}':-^80}", level="ERROR")
                SparcLog("Potential Energy Exceeded Limit", level="ERROR")
                SparcLog(f"Reference Energy: {float(epot_ref):.2f} eV", level="ERROR")
                SparcLog(
                    f"Threshold Energy: {float(epot_threshold):.2f} eV", level="ERROR"
                )
                SparcLog(f"Lower limit: {float(Llim):.2f} eV", level="ERROR")
                SparcLog(f"Upper limit: {float(Ulim):.2f} eV", level="ERROR")
                SparcLog(f"Current Energy: {float(epot):.2f} eV", level="ERROR")
                SparcLog("Stopping MLMD Simulation!!!", level="ERROR")
                SparcLog("-" * 80, level="ERROR")
                return True, epot_ref

        return False, epot_ref

    # save_xyz is called AFTER the safety check so that unphysical frames
    # (energy out of bounds / physical limits exceeded) are never written to
    # dpmd.traj and therefore never appear as QbC candidates.
    if temp_end is not None and temp_start is not None:
        for i_mlmd in range(steps_completed, steps):
            TemperatureRamp(dyn, system, i_mlmd, steps, temp_start, temp_end, ensemble)
            dyn.run(1)
            sim_failed, epot_ref = _check_mlmd_safety(
                system, distance_metrics, epot_ref, epot_threshold, dir_name
            )
            if sim_failed:
                break
            if dyn.nsteps % pace == 0:
                save_xyz(system, trajfile, "a", dir_name)
    else:
        for _ in range(remaining_steps):
            dyn.run(1)
            sim_failed, epot_ref = _check_mlmd_safety(
                system, distance_metrics, epot_ref, epot_threshold, dir_name
            )
            if sim_failed:
                break
            if dyn.nsteps % pace == 0:
                save_xyz(system, trajfile, "a", dir_name)

    # Final checkpoint
    save_checkpoint(dyn, system, checkpoint_file)


# ---------------------------------------
# DFT Energy Calculation
# ---------------------------------------
def CalculateDFTEnergy(
    idx: int,
    header: bool,
    system: Atoms,
    log_filename: str,
    dir_name: str,
    trajfile: str,
):
    """
    Calculate the DFT energy and forces for a candidate structure.

    Parameters
    ----------
    idx : int
        Candidate index (used in log output)
    header : bool
        If True, write column header to log file
    system : ase.Atoms
        Candidate structure with DFT calculator attached
    log_filename : str
        Filename for the per-candidate energy log
    dir_name : str
        Directory where log and trajectory files are written
    trajfile : str
        Filename for the ASE trajectory file
    """
    epot = system.get_potential_energy()
    epot = epot if not isinstance(epot, (list, np.ndarray)) else epot[0]

    SparcLog(f"Candidate: {idx:5d} | Epot: {epot:10.6f} [eV]")

    log_path = f"{dir_name}/{log_filename}"
    with open(log_path, "a") as f:
        if header:
            f.write(f"{'Candidate':>12} {'Epot[eV]':>14}\n")
        f.write(f"{idx:>12} {epot:>14.6f}\n")

    save_xyz(system, trajfile, "a", dir_name)


################################################################
# LAMMPS MD Execution
################################################################


# ---------------------------------------
# LAMMPS Interface
# ---------------------------------------
def lammps_md(system: Atoms, model_path: str, model_name: str):
    """
    Run a LAMMPS MD simulation.

    Parameters
    ----------
    system : ase.Atoms
        The ASE Atoms object representing the system
    model_path : str
        The path to the model files required by LAMMPS
    model_name : str
        The name of the model to be used in the simulation
    """
    SparcLog("\n" + "=" * 80)
    SparcLog("Starting LAMMPS MD Simulation".center(80))
    SparcLog("=" * 80)

    run_command = ["lmp", "-i", "in.lammps"]

    try:
        subprocess.run(run_command, check=True)
        SparcLog("\n" + "=" * 80)
        SparcLog("LAMMPS MD Simulation Completed Successfully".center(80))
        SparcLog("=" * 80)
    except subprocess.CalledProcessError as e:
        SparcLog("\n" + "=" * 80)
        SparcLog("Error in LAMMPS MD Simulation".center(80))
        SparcLog(str(e).center(80))
        SparcLog("=" * 80)


################################################################
# END OF FILE
################################################################

#!/usr/bin/python3
# plumed_wrapper.py

"""
PLUMED wrapper module for enhanced sampling MD simulations.

This module provides integration between ASE calculators and PLUMED
for enhanced sampling techniques like metadynamics and umbrella sampling.
"""

import os
from pathlib import Path
from typing import Optional

import ase.units
import yaml

################################################################
# Third party imports
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.plumed import Plumed
from ase.io import read

from sparc.src.ase_md import NPT, NVE, ExecuteMlpDynamics, LangevinNVT, NoseNVT

################################################################
# Local imports
from sparc.src.deepmd import setup_DeepPotential
from sparc.src.utils.logger import SparcLog

################################################################


def modify_forces(
    calculator: Calculator,
    system: Atoms,
    timestep: float,
    kT: Optional[float],
    restart: bool,
    plumed_input: str,
    iteration: str,
    PlumedLog: str = "PLUMED.log",
):
    """
    Set up and return a PLUMED wrapped ASE calculator for molecular dynamics.

    This function reads a PLUMED input file and wraps an existing ASE calculator
    to apply biasing forces during MD. Useful for enhanced sampling simulations.

    For more information, see the ASE documentation:
    https://wiki.fysik.dtu.dk/ase/ase/calculators/plumed.html

    Parameters
    ----------
    calculator : ase.calculators.Calculator
        The underlying ASE calculator for the system (e.g., DeepMD, VASP, CP2K)
    system : ase.Atoms
        The atomic configuration to which the calculator is applied
    timestep : float
        Integration timestep used in the MD simulation (already in ASE units)
    kT : float, optional
        Thermal energy (k_B * T) in eV. If None, defaults to 300 K (0.02585 eV)
    restart : bool
        Restart option for PLUMED
    plumed_input : str
        Path to the PLUMED input file (e.g., "plumed.dat")
    iteration : str
        Directory name for PLUMED output files
    PlumedLog : str, optional
        PLUMED log filename (default: 'PLUMED.log')

    Returns
    -------
    ase.calculators.plumed.Plumed
        The PLUMED-wrapped ASE calculator
    """
    # Ensure iteration directory exists
    out_dir = Path(iteration)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / Path(PlumedLog).name

    # Open the PLUMED input file and read its contents
    setup = [
        line
        for line in open(plumed_input, "r").read().splitlines()
        if not line.startswith("#")
    ]

    modified_setup = []
    for line in setup:
        if "FILE=" in line:
            # Extract the part after FILE= and handle multiple filenames
            parts = line.split("FILE=")
            if len(parts) > 1:
                filenames = parts[1].strip()  # Get the filenames
                filenames = filenames.split(",")  # Handle multiple filenames
                filenames = [
                    f"{iteration}/{filename.strip()}" for filename in filenames
                ]
                new_filenames = ",".join(filenames)
                line = parts[0] + f"FILE={new_filenames}"
        modified_setup.append(line)

    SparcLog(f"PLUMED output files will be written in {iteration}")

    if kT is None:
        kT = 8.617333e-5 * 300  # Boltzmann in eV/K at 300K

    # Initialize PLUMED calculator
    plumed_calc = Plumed(
        calc=calculator,
        input=modified_setup,
        timestep=timestep,
        atoms=system,
        kT=kT,
        log=str(log_path),
        restart=restart,
    )

    return plumed_calc


################################################################


def umbrella(config, us_dir: dict, dp_path: str, dp_model: str):
    """
    Set up and run umbrella sampling simulations with PLUMED biasing.

    This function reads a configuration file defining umbrella sampling windows
    and applies restraining forces to collective variables for each window.
    Supports NVE, NVT (Nose-Hoover, Langevin), and NPT ensembles.

    Parameters
    ----------
    config : SparcConfig or dict
        Configuration object or dictionary from input.yaml
    us_dir : dict
        Dictionary with 'dpmd_dir' key specifying base directory for windows
    dp_path : str
        Path to DeepMD model directory
    dp_model : str
        Name of the DeepMD model file
    """
    # Handle both SparcConfig dataclass and dict
    from sparc.src.utils.read_input import SparcConfig

    if isinstance(config, SparcConfig):
        us_config_file = config.mlip_setup.plumed.umbrella_sampling.config_file
        ensemble = config.mlip_setup.ensemble
        temperature = config.mlip_setup.temperature
        thermostat_type = config.mlip_setup.thermostat.type
        MDSteps = config.mlip_setup.md_steps
        timestep_fs = config.mlip_setup.timestep_fs
        log_frequency = config.mlip_setup.log_frequency
        epot_threshold = getattr(config.mlip_setup, "epot_threshold", None)
        plumed_kT = config.mlip_setup.plumed.kT
        plumed_restart = config.mlip_setup.plumed.restart
        dptraj_file = config.output.dptraj_file
        distance_metrics = config.distance_metrics
        tdamp = config.mlip_setup.thermostat.tdamp
        friction = config.mlip_setup.thermostat.friction
        temp_start = config.mlip_setup.temperature
        temp_end = config.mlip_setup.temp_end
    else:
        mlip_config = config.get("mlip_setup", {})
        thermostat_config = mlip_config.get("thermostat", {})
        us_config_file = mlip_config["plumed"]["umbrella_sampling"]["config_file"]
        ensemble = mlip_config.get("ensemble", "NVT")
        temperature = thermostat_config.get("temperature", 300)
        thermostat_type = thermostat_config.get("type", "Nose")
        MDSteps = mlip_config.get("md_steps", 0)
        timestep_fs = mlip_config.get("timestep_fs", 1.0)
        log_frequency = mlip_config.get("log_frequency", 5)
        epot_threshold = mlip_config.get("epot_threshold", None)
        plumed_kT = mlip_config.get("plumed", {}).get("kT", 0.02585)
        plumed_restart = mlip_config.get("plumed", {}).get("restart", False)
        dptraj_file = config.get("output", {}).get("dptraj_file", "dpmd.traj")
        distance_metrics = config.get("distance_metrics", None)
        tdamp = thermostat_config.get("tdamp", 10)
        friction = thermostat_config.get("friction", 0.01)
        temp_start = temperature
        temp_end = mlip_config.get("temp_end", None)

    # Load umbrella sampling configuration
    with open(us_config_file) as f:
        umbrella_config = yaml.safe_load(f)

    SparcLog(f"Umbrella Sampling: {len(umbrella_config['umbrella_windows'])} windows")
    SparcLog(
        f"Ensemble: {ensemble} | Thermostat: {thermostat_type} | Temperature: {temperature} K\n"
    )

    # Loop over umbrella windows
    for w, window in enumerate(umbrella_config["umbrella_windows"]):
        SparcLog(f"Running Umbrella Sampling for window {w}\n")

        struct_file = window["structure"]
        plumed_file = window["plumed_file"]
        window_dir = os.path.join(us_dir["dpmd_dir"], f"window_{w:03d}")
        os.makedirs(window_dir, exist_ok=True)

        usmd_log = os.path.join(f"window_{w:03d}", "usmd.log")

        # Load structure and re-initialize system and calculator
        dp_atoms = read(struct_file)
        dp_atoms.write(os.path.join(window_dir, "input.xyz"))

        dp_atoms, dp_calc = setup_DeepPotential(
            atoms=dp_atoms, model_path=dp_path, model_name=dp_model
        )

        # Create dynamics object based on ensemble
        if ensemble == "NVE":
            dyn_dp = NVE(system=dp_atoms, timestep=timestep_fs, restart=False)
        elif ensemble == "NPT":
            dyn_dp = NPT(
                system=dp_atoms,
                timestep=timestep_fs,
                temperature=temperature,
                tau_t=tdamp,
                pressure=1.0,
                tau_p=1000,
                restart=False,
            )
        else:  # NVT
            if thermostat_type == "Nose":
                dyn_dp = NoseNVT(
                    atoms=dp_atoms,
                    timestep=timestep_fs,
                    temperature=temperature,
                    tdamp=tdamp,
                    restart=False,
                )
            else:
                dyn_dp = LangevinNVT(
                    atoms=dp_atoms,
                    timestep=timestep_fs,
                    temperature=temperature,
                    friction=friction,
                    restart=False,
                )

        # Wrap with PLUMED (timestep needs to be in ASE units for PLUMED)
        SparcLog(
            f" → Window {w:03d} | Structure: {struct_file} | PLUMED: {plumed_file}"
        )

        dp_atoms.calc = modify_forces(
            calculator=dp_calc,
            system=dp_atoms,
            timestep=timestep_fs * ase.units.fs,
            kT=plumed_kT,
            restart=plumed_restart,
            plumed_input=plumed_file,
            iteration=window_dir,
            PlumedLog="PLUMED.log",
        )

        # Run MD
        ExecuteMlpDynamics(
            system=dp_atoms,
            dyn=dyn_dp,
            steps=MDSteps,
            pace=log_frequency,
            log_filename=usmd_log,
            trajfile=dptraj_file,
            dir_name=us_dir["dpmd_dir"],
            distance_metrics=distance_metrics,
            name=f"{ensemble}-{thermostat_type}",
            epot_threshold=epot_threshold,
            temp_start=temp_start,
            temp_end=temp_end,
        )


################################################################
# END OF FILE
################################################################

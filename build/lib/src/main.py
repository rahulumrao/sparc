#!/usr/bin/python3 main.py
import ase
import ase.units
from ase.io import read
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

#
from src.vasp_setup import setup_dft_calculator
from src.deepmd_training import deepmd_training
from src.dpmd_setup import setup_DeepPotential
from src.ase_md import NoseNVT, run_md, run_Ase_DPMD
from src.active_learning import QueryByCommittee
from src.plumed_wrapper import modify_forces


def run_aimd(system, timestep, tdamp, temperature, steps, log_frequency, trajfile, plumed_config=None):
    """Run AIMD with optional Plumed integration."""
    dyn_nose = NVTBerendsen(system, timestep, temperature, tdamp)
    
    if plumed_config:
        print("Using Plumed wrapper for AIMD simulation.")
        system.calc = modify_forces(
            calculator=system.calc, 
            system=system,
            timestep=timestep, 
            temperature=temperature,
            kT=plumed_config['kT'], 
            plumed_input=plumed_config['input_file']
        )
    
    MaxwellBoltzmannDistribution(system, temperature * ase.units.kB)
    
    dyn_nose.attach(lambda: print(f"Step: {dyn_nose.get_time():.1f} ps"), interval=log_frequency)
    trajfile = open(trajfile, 'w')
    dyn_nose.attach(lambda: system.write(trajfile, format="traj"), interval=log_frequency)
    dyn_nose.run(steps)


def train_deepmd_models(config, num_models=2):
    """Train DeepMD models and return the paths of trained models."""
    print(f"Training {num_models} DeepMD models...")
    model_paths = []
    for i in range(num_models):
        model_dir = f"{config['deepmd_setup']['train_dir']}/model_{i+1}"
        deepmd_training(
            training_dir=model_dir,
            num_models=1,  # Train models one at a time
            input_file=config['deepmd_setup']['input_file']
        )
        model_paths.append(model_dir)
    return model_paths


def run_dpmd_and_query(system, config, model_paths, dpmd_steps, log_frequency, trajfile):
    """Run DeepMD simulation and query high-uncertainty configurations."""
    dp_calc = setup_DeepPotential(
        atoms=system,
        model_path=model_paths,
        model_name=config['deepmd_setup']['model_name']
    )
    system.calc = dp_calc
    
    dyn_dp = NVTBerendsen(
        system, 
        config['md_simulation']['timestep_fs'] * ase.units.fs,
        config['md_simulation']['temperature'],
        config['md_simulation']['tdamp']
    )
    
    MaxwellBoltzmannDistribution(system, config['md_simulation']['temperature'] * ase.units.kB)
    
    dyn_dp.attach(lambda: print(f"DPMD step: {dyn_dp.get_time():.1f} ps"), interval=log_frequency)
    trajfile = open(trajfile, 'w')
    dyn_dp.attach(lambda: system.write(trajfile, format="traj"), interval=log_frequency)
    dyn_dp.run(dpmd_steps)
    
    # Query by Committee for high-uncertainty candidates
    return QueryByCommittee(
        trajfile=trajfile,
        model_path=config['deepmd_setup']['train_dir'],
        num_models=config['deepmd_setup']['num_models'],
        data_path=config['deepmd_setup']['dpmd_dir']
    )


def main(config):
    """Main function to execute the active learning workflow."""
    timestep = config['md_simulation']['timestep_fs'] * ase.units.fs
    temperature = config['md_simulation']['temperature']
    candidate_found_is = True
    iter = 0
    
    while candidate_found_is:
        print(f"Iteration {iter}: Running active learning loop...")
        
        # DFT Labelling
        labelled_files = []
        for files in config['data']['candidates']:
            poscar = read(files, format="vasp")
            poscar.calc = setup_dft_calculator(config)
            run_aimd(
                system=poscar,
                timestep=timestep,
                tdamp=config['md_simulation']['tdamp'] * ase.units.fs,
                temperature=temperature,
                steps=config['md_simulation']['labeling_steps'],
                log_frequency=config['general']['log_frequency'],
                trajfile=config['output']['aimdtraj_file'],
                plumed_config=config['plumed'] if config['plumed_is'] else None
            )
            labelled_files.append(config['output']['aimdtraj_file'])
        
        # DeepMD Training
        model_paths = train_deepmd_models(config, num_models=2)
        
        # Deep Potential MD and Query by Committee
        candidate_found_is, labelled_files = run_dpmd_and_query(
            system=read(labelled_files[0], format="vasp"),  # Read the system from labelled files
            config=config,
            model_paths=model_paths,
            dpmd_steps=config['deepmd_setup']['md_steps'][0],
            log_frequency=config['general']['log_frequency'],
            trajfile=config['output']['dptraj_file']
        )
        
        iter += 1

    print("Active learning loop completed. No more candidates found.")


if __name__ == "__main__":
    import yaml
    # Load the configuration file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    main(config)

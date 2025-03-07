#!/usr/bin/python3
"""
SPARC (**S**mart **P**otential with **A**tomistic **R**are Events and **C**ontinuous Learning)

A Python package for automated molecular dynamics simulations with active learning.

Key Features:
1. Ab initio Molecular Dynamics (AIMD) with VASP
2. DeepMD model training for machine learning potentials
3. Deep Potential Molecular Dynamics (DPMD) simulations
4. Active Learning cycles for continuous model improvement

Usage:
    python sparc.py -i input.yaml

The workflow is configured through a YAML input file. See examples/input.yaml for a template.
"""

#===================================================================================================#
# Standard library imports
import yaml
import cProfile
import os
import sys
import subprocess
import argparse
import cProfile
import pstats

# Third-party imports
from ase.io import read, write
import ase.units
from ase.md import MDLogger

# 
# from scripts.plot_model_devi import main as plot_model_devi

# Local imports
from src.read_input import load_config
from src.deepmd_training import deepmd_training
from src.vasp_setup import setup_dft_calculator
from src.ase_md import NoseNVT, run_aimd, run_Ase_DPMD
from src.plumed_wrapper import modify_forces
from src.data_processing import get_data
from src.dpmd_setup import setup_DeepPotential
from src.active_learning import QueryByCommittee
from src.utils import create_iteration_dirs, combine_trajectories, load_progress, save_progress, remove_backup_files
#===================================================================================================#
def main():
    """Main function that coordinates the entire workflow."""
    
    #--------------------------------------------------------------------------------------#
    # Parse command line arguments and load configuration
    parser = argparse.ArgumentParser(description='Run AIMD and DeepMD simulations.')
    parser.add_argument("-i", "--input_file", type=str, default="input.yaml", 
                       help="Input YAML file")
    # parser.add_argument("-p", "--plot", action="store_true", help="Plot model deviation from each iteration")
    args = parser.parse_args()
    
    # if args.plot:
    #     plot_model_devi()
    #     sys.exit()
    
    try:
        yaml_config = load_config(args.input_file)
        print("\nYAML configuration loaded successfully!\n")
    except FileNotFoundError:
        print(f"\nError: YAML configuration File not found! (default: input.yaml)\n")
        sys.exit(1)
    config = yaml_config
    
    #--------------------------------------------------------------------------------------#
    # System initialization
    parent_dir = os.getcwd()
    
    # Read and prepare atomic structure
    system = read(config['general']['structure_file']) #poscar 
    atom_types = list(dict.fromkeys(system.get_chemical_symbols()))
    system.set_pbc([True, True, True])
    system.center()
    original_system = system
    
    # Set up DFT calculator
    vasp_calc = setup_dft_calculator(config, True)
    
    # Initialize simulation parameters
    timestep = config['md_simulation']['timestep_fs'] * ase.units.fs
    temperature = config['md_simulation']['temperature']
    DftMDSteps = config['general']['md_steps']
    plumed_is = config['md_simulation']['use_plumed']
    iter_structure = create_iteration_dirs(iter_num=0)        # Create iteration directory structure 
    #--------------------------------------------------------------------------------------#
    # SECTION 1: Ab initio Molecular Dynamics (AIMD)
    #--------------------------------------------------------------------------------------#
    dftmd_is = DftMDSteps > 0
    if (dftmd_is):
        # Only when it is True
        print("\n========================================================================")
        print(f"\n ! ab-initio MD Simulations will be performed at Temp.: {temperature}K !")
        print("\n========================================================================")
        dyn_nose = NoseNVT(
            atoms=system, 
            timestep=timestep,
            tdamp=config['md_simulation']['tdamp'] * ase.units.fs,
            temperature=temperature,
            restart=config['md_simulation']['restart'])
        
        # Configure PLUMED if enabled
        if (plumed_is):
            print("\n========================================================================")
            print("               PLUMED IS CALLED FOR MD SIMULATION. !")
            print("========================================================================")
            #
            system.calc = modify_forces(
            calculator=vasp_calc, 
            system=system, timestep=timestep, 
            temperature=temperature, 
            kT=config['plumed']['kT'],
	        restart=config['plumed']['restart'], 
            plumed_input=config['plumed']['input_file'])
        else:
            system.calc = vasp_calc
        
        # Run AIMD simulation
        run_aimd(
            system=system, 
            dyn=dyn_nose, 
            steps=DftMDSteps, 
            pace=config['general']['log_frequency'], 
            log_filename=config['output']['log_file'],
            trajfile=config['output']['aimdtraj_file'],
            dir_name=iter_structure['dft_dir'])
    #--------------------------------------------------------------------------------------#
    # SECTION 2: DeepMD Training
    #--------------------------------------------------------------------------------------#
    training_is = config['deepmd_setup']['training']
    datadir = os.path.join(parent_dir,config['deepmd_setup']['data_dir'])
    if training_is:
        # Process AIMD trajectory for training
        get_data(
            ase_traj=iter_structure['dft_dir'] / config['output']['aimdtraj_file'], 
            dir_name=datadir, 
            skip_min=0, 
            skip_max=None)
        
        # Train DeepMD models
        deepmd_training(
            active_learning=False,
            training_dir=iter_structure['train_dir'],
            num_models=config['deepmd_setup']['num_models'], 
            input_file=config['deepmd_setup']['input_file'],
            datadir=datadir,
            atom_types=atom_types)
    
    #--------------------------------------------------------------------------------------#
    # SECTION 3: Deep Potential Molecular Dynamics
    #--------------------------------------------------------------------------------------#
    dpmd_run_is = config['deepmd_setup']['MdSimulation']
    if dpmd_run_is:
        n_sample = config['deepmd_setup'].get('multiple_run', 1)
        print("\n========================================================================")
        print("      MULTIPLE MLP-MD SIMULATION STARTING FROM SAME CONFIGURATION!")
        print("========================================================================")
        for i in range(n_sample):
        # Setup DeepMD calculator
            dp_path = iter_structure['train_dir'] #config['deepmd_setup']['train_dir']
            dp_model = "training_1/frozen_model_1.pb" #config['deepmd_setup']['model_name']
            dp_system = original_system
            dp_atoms, dp_calc = setup_DeepPotential(
                                atoms=dp_system, 
                                model_path=dp_path, 
                                model_name=dp_model)

            # Configure and run DPMD
            dyn_dp = NoseNVT(
                atoms=dp_atoms, 
                timestep=config['deepmd_setup']['timestep_fs'] * ase.units.fs,
                tdamp=config['md_simulation']['tdamp'] * ase.units.fs, 
                temperature=temperature
                )
            
            MDsteps = config['deepmd_setup']['md_steps']
            writePace = config['deepmd_setup']['log_frequency']
            
            dp_plumed_is = config['deepmd_setup']['use_plumed']
            # Configure PLUMED if enabled
            if (dp_plumed_is):
                print("\n========================================================================")
                print(f"    Sim:[{i}]       PLUMED IS CALLED FOR DPMD SIMULATION !")
                print("========================================================================")
                
                # Get PLUMED input file - use default if not specified
                plumed_file = config['deepmd_setup'].get('plumed_file', 'plumed.dat')
                print(f"Using PLUMED input file: {plumed_file}")
                remove_backup_files(file_ext="bck.*")
                dp_atoms.calc = modify_forces(
                    calculator=dp_calc, 
                    system=dp_atoms, 
                    timestep=timestep, 
                    temperature=temperature, 
                    kT=config['plumed']['kT'],
                    restart=config['plumed']['restart'], 
                    plumed_input=plumed_file)
            else:
                dp_atoms.calc = dp_calc
                
            # print("DPMD Calculator: ",system.calc,'\n', dp_calc)
            # sys.exit()
            run_Ase_DPMD(
                system=dp_atoms,
                dyn=dyn_dp,
                steps=MDsteps,
                pace=writePace,
                log_filename=config['deepmd_setup']['log_file'],
                trajfile=config['output']['dptraj_file'],
                dir_name=iter_structure['dpmd_dir'],
                distance_metrics=config['distance_metrics']  # Pass the user-provided distance metrics
        )
        
        # Check for structures requiring labeling
        candidate_found_is, labelled_files, latest_models = QueryByCommittee(
            trajfile=iter_structure['dpmd_dir'] / config['output']['dptraj_file'], 
            model_path=iter_structure['train_dir'], 
            num_models=config['deepmd_setup']['num_models'],
            min_lim=config['model_dev']['f_min_dev'],
            max_lim=config['model_dev']['f_max_dev'],
            dpmd_data_path=iter_structure['dpmd_dir'],
            iteration=0)
            
        if not candidate_found_is:
            print("\n========================================================================")
            print("!                No More Candidates Found for Labelling                !")
            print("!                 End of Active Learning Loop                          !")
            print(f"!     Any one of the Latest Models can be Used for MD Simulation       !")
            print("========================================================================")
            return  # Use return instead of break to exit the function
        else:
            print(f"Candidates found for labelling: {candidate_found_is}")
    #=======================================================================================
    # SECTION 4: Active Learning Protocol Strating from here !                             #
    #=======================================================================================
    # Load the last completed iteration and state
    start_iteration = load_progress()
    print(f"Resuming from iteration: {start_iteration}")

    # Active Learning Loop
    learning_is = config['active_learning']
    al_iter = config['iteration']

    if learning_is:
        print("=" * 72)
        print(f"{'Active Learning Protocol is set to: ' + str(learning_is):^{72}}")
        print(f"{'Total AL Iterations will be run: ' + str(al_iter):^{72}}")
        iter = start_iteration  # Start from the last completed iteration
        while candidate_found_is and iter < al_iter:
            print("\n========================================================================")
            print("{}".format(f"Starting Iteration {iter}".center(72)))
            print("========================================================================")
            
            iter_structure = create_iteration_dirs(iter_num=iter)   # Create iteration directory structure 
            
            # Process candidates for labeling
            # for idx, files in enumerate(labelled_files[1:], start=1):
            for idx, files in enumerate(labelled_files, start=1):
                if not os.path.exists(files):
                    print(f"Warning: Candidate file {files} not found, skipping...")
                    continue
                
                # Run DFT calculations
                poscar_ = read(files, format='vasp')
                poscar_.calc = setup_dft_calculator(config, False)
                
                # Run AIMD for each labelled structure
                header = True if idx == 0 else False
                if header:
                    print("\n========================================================================")
                    print("!{}!".format(f"[Iteration {iter}] Computing Energy and Forces for Candidates".center(70)))
                    print("========================================================================")
                run_aimd(
                    system=poscar_, 
                    dyn=NoseNVT(
                            atoms=poscar_, 
                            timestep=timestep,
                            tdamp=config['md_simulation']['tdamp'] * ase.units.fs,
                            temperature=0),
                        steps=0, 
                        pace=1, 
                        log_filename=f"Iter{iter}_{config['output']['log_file']}",
                        trajfile=config['output']['aimdtraj_file'],
                        dir_name=iter_structure['dft_dir'],
                        header=header,
                        mode='a')
            
            # Retrain DeepMD models
            print("\nProcessing data for DeepMD training")
            combined_traj = combine_trajectories(
                trajfilename=config['output']['aimdtraj_file'],
                current_iter=iter_structure['iter_num'])
            get_data(
                ase_traj=combined_traj, 
                dir_name=config['deepmd_setup']['data_dir'], 
                skip_min=0, 
                skip_max=None)
            
            print("\nStarting DeepMD training")
            deepmd_training(active_learning=True,
                training_dir=iter_structure['train_dir'],
                num_models=config['deepmd_setup']['num_models'], 
                input_file=config['deepmd_setup']['input_file'],
                datadir=datadir,
                atom_types=atom_types)

            print("\n{}".format("Setting up DeepPotential Calculator".center(72)))
            #--------------------------------------------------------------------------------------#
            # Set DeepPotential calculator
            #--------------------------------------------------------------------------------------#
            n_sample = config['deepmd_setup'].get('multiple_run', 1)
            print("\n========================================================================")
            print("      MULTIPLE SAPMLE MD-SIMULATION STARTING FROM SAME CONFIGURATION!")
            print("========================================================================")
            for i in range(n_sample):
                dp_system = original_system
                dp_atoms, dp_calc = setup_DeepPotential(
                                    atoms=dp_system, 
                                    model_path=parent_dir, 
                                    model_name=latest_models[0])
                
                # original_system.calc = dp_calc

                # DeepMD Simulation
                print("\n{}".format("Initializing DeepMD Simulation".center(72)))
                #--------------------------------------------------------------------------------------#
                # Set Nose-Hoover Chain NVT Dynamics
                #--------------------------------------------------------------------------------------#
                dyn_dp = NoseNVT(
                    atoms=dp_atoms, 
                    timestep=timestep,
                    tdamp=config['md_simulation']['tdamp'] * ase.units.fs, 
                    temperature=temperature
                    )
                
                MDsteps = config['deepmd_setup']['md_steps']
                writePace = config['deepmd_setup']['log_frequency']
                # dp_traj_file = f"Iter{iter}_{config['output']['dptraj_file']}"
                
                dp_plumed_is = config['deepmd_setup']['use_plumed']
                # Configure PLUMED if enabled
                if (dp_plumed_is):
                    print("\n========================================================================")
                    print("               PLUMED IS CALLED FOR DPMD SIMULATION !")
                    print("========================================================================")
                    
                    # Get PLUMED input file - use default if not specified
                    plumed_file = config['deepmd_setup'].get('plumed_file', 'plumed.dat')
                    print(f"Using PLUMED input file: {plumed_file}")
                    
                    dp_atoms.calc = modify_forces(
                        calculator=dp_calc, 
                        system=dp_atoms, 
                        timestep=timestep, 
                        temperature=temperature, 
                        kT=config['plumed']['kT'],
                        restart=config['plumed']['restart'], 
                        plumed_input=plumed_file)
                else:
                    dp_atoms.calc = dp_calc
                
                # Perform DeepPotential MD siulation with re-trained model

                run_Ase_DPMD(
                    system=dp_atoms,
                    dyn=dyn_dp,
                    steps=MDsteps,
                    pace=writePace,
                    log_filename=f"Iter{iter}_{config['deepmd_setup']['log_file']}",
                    trajfile=config['output']['dptraj_file'], #dp_traj_file,
                    dir_name=iter_structure['dpmd_dir'],
                    distance_metrics=config['distance_metrics']  # Pass the user-provided distance metrics
            )
            
            # Check for new candidates
            candidate_found_is, labelled_files, latest_models = QueryByCommittee(
                    trajfile=iter_structure['dpmd_dir'] / config['output']['dptraj_file'], 
                    model_path=iter_structure['train_dir'], 
                    num_models=config['deepmd_setup']['num_models'], 
                    min_lim=config['model_dev']['f_min_dev'],
                    max_lim=config['model_dev']['f_max_dev'],
                    dpmd_data_path=iter_structure['dpmd_dir'],
                    iteration=iter)
            
            if not candidate_found_is:
                print("\n========================================================================")
                print("                 No More Candidates Found for Labelling.                 ")
                print("                 End of Active Learning Loop.                             ")
                print("========================================================================")
                break
            
            # At the end of each iteration, save progress
            save_progress(iter)  # Save the current state (subfolder)

            iter += 1
    
    # Validate required configuration parameters
    required_params = [
        'general.structure_file',
        'md_simulation.temperature',
        'md_simulation.timestep_fs'
    ]
    
    for param in required_params:
        section, key = param.split('.')
        if section not in config or key not in config[section]:
            raise ValueError(f"Missing required parameter: {param}")

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='sparc.prof')

#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#

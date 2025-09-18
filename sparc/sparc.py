#!/usr/bin/python3
# sparc.py
"""
    SPARC (**S**mart **P**otential with **A**tomistic **R**are Events and **C**ontinuous Learning)

    Main module for SPARC, coordinating the active learning workflow for reactive ML potentials.
    This module sets up logging, reads input files, and sequentially triggers:
        - Data processing
        - MD simulations (both ab initio and ML/MD)
        - Model training using DeepMD-kit
        - Active learning iterations

    Usage:
    -----
        python sparc.py -i input.yaml

The workflow is configured through a YAML input file. See examples/input.yaml for a template.
"""

#===================================================================================================#
# Standard library imports
import yaml
import os
import sys
import copy
import subprocess
import argparse
import cProfile
import pstats
from pathlib import Path
################################################################
# Third-party imports
from ase.io import read, write
import ase.units
from ase.md import MDLogger
################################################################
# from scripts.plot_model_devi import main as plot_model_devi
# Local imports
from sparc.src.utils.read_input import load_config
from sparc.src.deepmd import deepmd_training, setup_DeepPotential
from sparc.src.calculator import dft_calculator
from sparc.src.ase_md import NoseNVT, LangevinNVT, ExecuteAbInitioDynamics, ExecuteMlpDynamics, CalculateDFTEnergy
from sparc.src.plumed_wrapper import modify_forces, umbrella
from sparc.src.data_processing import get_data
from sparc.src.active_learning import QueryByCommittee
from sparc.src.utils.logger import setup_logger, SparcLog
from sparc.src.utils.banner import banner
from sparc.src.utils.utils import create_iteration_dirs, combine_trajectories, load_progress, save_progress, restart_progress, remove_backup_files
#===================================================================================================#
# Analysis Modules
#--------------------------------------------------------------------------------------#
if "--analysis" in sys.argv:
    from sparc.src.utils.analysis import main as compute_energy
    #
    # Strip the --analysis flag and everyrthing before it
    iidx = sys.argv.index("--analysis")
    analysis_args = sys.argv[iidx + 1:]
    sys.argv = [sys.argv[0]] + analysis_args # reset sys.argv for analysis
    compute_energy()
    sys.exit(0)
#--------------------------------------------------------------------------------------#
def main():
    """Main function that coordinates the entire workflow."""
    #--------------------------------------------------------------------------------------#
    banner()
    setup_logger(enable=True)
    #--------------------------------------------------------------------------------------#
    # Parse command line arguments and load configuration
    parser = argparse.ArgumentParser(description='Run main SPARC workflow')
    parser.add_argument("-i", "--input_file", type=str, default="input.yaml", 
                       help="Input YAML file")
    # parser.add_argument("-p", "--plot", action="store_true", help="Plot model deviation from each iteration")
    args = parser.parse_args()
    
    # if args.plot:
    #     plot_model_devi()
    #     sys.exit()
    try:
        yaml_config = load_config(args.input_file)
        SparcLog("\nYAML configuration loaded successfully!\n")
    except FileNotFoundError:
        SparcLog(f"\nError: YAML configuration File not found! (default: input.yaml)\n")
        sys.exit(1)
    config = yaml_config
    
    #--------------------------------------------------------------------------------------#
    # Workflow Initialization
    parent_dir = os.getcwd()
    
    # Read and prepare atomic structure
    system = read(config['general']['structure_file'])  # e.g., POSCAR or xyz file (ASE accepted formats)
    atom_types = list(dict.fromkeys(system.get_chemical_symbols()))
    if system.get_pbc().any():          # True for perodic system (eg: POSCAR)
        system.set_pbc([True, True, True])
        system.center()
    original_system = system
    SparcLog(f'STRUCTURE FILE: {system}')
    # sys.exit(1)
    # Initialize simulation parameters
    temperature = config.get('md_simulation', {}).get('temperature', 300)
    thermostat = config.get('md_simulation', {}).get('thermostat', 'Nose')
    thermostat_func = {'Nose': NoseNVT, 'Langevin': LangevinNVT}
    
    # If AIMD Steps == 0 then dftmd_is = False
    DftMDSteps = config.get('md_simulation', {}).get('steps', 0)
    dftmd_is = DftMDSteps > 0
    training_is = config.get('deepmd_setup', {}).get('training', False)
    dpmd_run_is = config.get('deepmd_setup', {}).get('MdSimulation', False)
    
    if dftmd_is or training_is or dpmd_run_is:
        iter_structure = create_iteration_dirs(iter_num=0)  # Create iteration directory structure 

    #--------------------------------------------------------------------------------------#
    # SECTION 1: Ab initio Molecular Dynamics (AIMD)
    #--------------------------------------------------------------------------------------#
    if dftmd_is:
        # Set up DFT calculator
        dft_calc = dft_calculator(config, True)
        SparcLog("========================================================================")
        SparcLog(f" ! ab-initio MD Simulations will be performed at Temp.: {temperature}K !")
        SparcLog("========================================================================")
        # Configure and run AIMD
        if thermostat == 'Nose':
            dyn_dft = NoseNVT(
                atoms=system, 
                timestep=config['md_simulation']['timestep_fs'] * ase.units.fs,
                tdamp=config['md_simulation']['tdamp'] * ase.units.fs, 
                temperature=temperature,
                restart=config['md_simulation']['restart']
            )
        elif thermostat == 'Langevin':
            dyn_dft = LangevinNVT(
                atoms=system,
                timestep=config['md_simulation']['timestep_fs'] * ase.units.fs,
                temperature=temperature,
                friction=config['md_simulation']['friction'] / ase.units.fs,
                restart=config['md_simulation']['restart']
            )
        # Configure PLUMED if enabled
        plumed_is = config.get('md_simulation', {}).get('use_dft_plumed', False)
        if plumed_is:
            SparcLog("========================================================================")
            SparcLog("               PLUMED IS CALLED FOR MD SIMULATION. !")
            SparcLog("========================================================================")
            
            # Get PLUMED input file - use default if not specified
            plumed_dft = config['dft_plumed'].get('input_file', 'plumed.dat')
            SparcLog(f"Using PLUMED input file: {plumed_dft}")
            remove_backup_files(file_ext="bck.*")   # remove PLUMED generated backup files if present!
            system.calc = modify_forces(
                calculator=dft_calc, 
                system=system, 
                timestep=config['md_simulation']['timestep_fs'] * ase.units.fs, 
                kT=config['plumed']['kT'],
                restart=config['plumed']['restart'], 
                plumed_input=plumed_dft,
                iteration=iter_structure['dft_dir']
            )
        else:
            system.calc = dft_calc
        
        # Run AIMD simulation
        ExecuteAbInitioDynamics(
            system=system, 
            dyn=dyn_dft, 
            steps=DftMDSteps, 
            pace=config['md_simulation']['log_frequency'], 
            log_filename=config['output']['log_file'],
            trajfile=config['output']['aimdtraj_file'],
            dir_name=iter_structure['dft_dir'],
            name=thermostat
        )
    
    #--------------------------------------------------------------------------------------#
    # SECTION 2: DeepMD Training
    #--------------------------------------------------------------------------------------#
    datadir = os.path.join(parent_dir, config['deepmd_setup']['data_dir'])
    if training_is:
        # Process AIMD trajectory for ML model training using `dpdata`
        get_data(
            ase_traj=iter_structure['dft_dir'] / config['output']['aimdtraj_file'], 
            dir_name=datadir, 
            skip_min=config['deepmd_setup']['skip_min'], 
            skip_max=config['deepmd_setup']['skip_max']
        )
        
        # Train DeepMD models
        deepmd_training(
            active_learning=False,
            training_dir=iter_structure['train_dir'],
            num_models=config['deepmd_setup']['num_models'], 
            input_file=config['deepmd_setup']['input_file'],
            datadir=datadir,
            atom_types=atom_types
        )
    
    #--------------------------------------------------------------------------------------#
    # SECTION 3: Deep Potential Molecular Dynamics
    #--------------------------------------------------------------------------------------#
    if dpmd_run_is:
        n_sample = config['deepmd_setup'].get('multiple_run', 1)
        SparcLog("========================================================================")
        SparcLog(f"      MULTIPLE MLP-MD SIMULATION STARTING FROM SAME CONFIGURATION!")
        SparcLog("========================================================================")
        
        dp_path = iter_structure['train_dir'] 
        dp_model = "training_1/frozen_model_1.pb"   # Default model to run ML/MD simulation [optional]
        dp_system = original_system
        # Setup DeepMD calculator and run multiple MD simulations if requested
        for i in range(n_sample):
            dp_atoms, dp_calc = setup_DeepPotential(
                atoms=dp_system, 
                model_path=dp_path, 
                model_name=dp_model
            )            
            if thermostat not in thermostat_func:
                raise ValueError(f"Unknown Thermostat: {thermostat}")

            dyn_dp_class = thermostat_func[thermostat]
            if thermostat == 'Nose':
                dyn_dp = dyn_dp_class(
                    atoms=dp_atoms, 
                    timestep=config['deepmd_setup']['timestep_fs'] * ase.units.fs,
                    tdamp=config['md_simulation']['tdamp'] * ase.units.fs, 
                    temperature=temperature
                )
            else:
                dyn_dp = dyn_dp_class(
                    atoms=dp_atoms, 
                    timestep=config['deepmd_setup']['timestep_fs'] * ase.units.fs,
                    temperature=temperature, 
                    friction=config['md_simulation']['friction'] / ase.units.fs
                )
        
            MDsteps = config['deepmd_setup']['md_steps']
            writePace = config['deepmd_setup']['log_frequency']
            
            dp_plumed_is = config['deepmd_setup']['use_plumed']
            umbrella_enabled = config['deepmd_setup']['umbrella_sampling'].get('enabled', False)
            # print(f"Umbrella Sampling is Enabled {umbrella_enabled}")
            # sys.exit(1)
            # Configure PLUMED if enabled
            if dp_plumed_is and umbrella_enabled:
                SparcLog("========================================================================")
                SparcLog("       Umbrella Sampling Enabled — Running MLMD Windows with PLUMED       ")
                SparcLog("========================================================================")

                umbrella(config=config, 
                         us_dir=iter_structure, 
                         dp_path=dp_path, 
                         dp_model=dp_model
                         )
                break
            if dp_plumed_is and not umbrella_enabled:
                SparcLog("========================================================================")
                SparcLog(f"    Sim:[{i}]       PLUMED IS CALLED FOR DPMD SIMULATION !")
                SparcLog("========================================================================")
                
                # Get PLUMED input file - use default if not specified
                plumed_file = config['deepmd_setup'].get('plumed_file', 'plumed.dat')
                SparcLog(f"Using PLUMED input file: {plumed_file}")
                remove_backup_files(file_ext="bck.*")   # remove PLUMED backup files
                dp_atoms.calc = modify_forces(
                    calculator=dp_calc, 
                    system=dp_atoms, 
                    timestep=config['deepmd_setup']['timestep_fs'] * ase.units.fs, 
                    kT=config['plumed']['kT'],
                    restart=config['plumed']['restart'], 
                    plumed_input=plumed_file,
                    iteration=iter_structure['dpmd_dir']
                )
            else:
                dp_atoms.calc = dp_calc
                
            ExecuteMlpDynamics(
                system=dp_atoms,
                dyn=dyn_dp,
                steps=MDsteps,
                pace=writePace,
                log_filename=f"Iter0_dpmd.log",
                trajfile=config['output']['dptraj_file'],
                dir_name=iter_structure['dpmd_dir'],
                distance_metrics=config['distance_metrics'],
                name=thermostat,
                epot_threshold=config['deepmd_setup']['epot_threshold']
            )
        if config['active_learning']:            
            # Check for structures requiring labeling (Query-by-Committee [QbC])
            candidate_found_is, labelled_files, latest_models = QueryByCommittee(
                trajfile=iter_structure['dpmd_dir'] / config['output']['dptraj_file'], 
                model_path=iter_structure['train_dir'], 
                num_models=config['deepmd_setup']['num_models'],
                min_lim=config['model_dev']['f_min_dev'],
                max_lim=config['model_dev']['f_max_dev'],
                dpmd_data_path=iter_structure['dpmd_dir'],
                iteration=0
            )
            candidates = len(labelled_files)  # store total candidates for restart
            save_progress({'state': str(iter_structure['dft_dir']), 'iteration': 1, 'candidate': candidates, 'idx': 1})
            if not candidate_found_is:
                SparcLog("\n========================================================================")
                SparcLog("!                No Candidates Found for Labelling                !")
                SparcLog("!                 End of Active Learning Loop                          !")
                SparcLog("========================================================================")
                return
            else:
                SparcLog(f"Candidates found for labelling: {candidate_found_is}")
    
    #=======================================================================================#
    # SECTION 4: Active Learning Protocol Starting from here!
    #=======================================================================================#
    dp_tau = config['deepmd_setup']['timestep_fs'] * ase.units.fs
    learning_is = config['active_learning']
    learning_restart = config['learning_restart']
    al_iter = config['iteration']
    # starting from iteration 1
    i_start = 1
    iter = 1
    SparcLog("========================================================================")
    SparcLog(f"{'Active Learning Protocol is set to: ' + str(learning_is):^{72}}")
    SparcLog("========================================================================")
    if learning_is:
        # Load the last completed iteration and state
        if learning_restart:
            iter, i_start, candidates, candidate_found_is, labelled_files = restart_progress(start_iteration=load_progress())
            latest_models = [config['latest_model']]
        SparcLog(f"{'Total AL Iterations will be run: ' + str(al_iter):^{72}}")
        #--------------------------------------------------------------------------------------#
        # Active Learning Loop
        #--------------------------------------------------------------------------------------#
        while candidate_found_is and iter < al_iter:
            SparcLog("========================================================================")
            SparcLog("{}".format(f"Starting Iteration {iter}".center(72)))
            SparcLog("========================================================================")
            
            iter_structure = create_iteration_dirs(iter_num=iter)  # Create iteration directory structure 
            # Process candidates for labeling
            for idx, files in enumerate(labelled_files, start=i_start):
                if not os.path.exists(files):
                    SparcLog(f"Warning: Candidate file {files} not found, skipping...")
                    continue
                
                # Run DFT calculations (for now // relabelling only supports POSCAR format)
                poscar_ = read(files, format='vasp')
                poscar_.calc = dft_calculator(config, False)
                
                header = True if idx == 1 else False
                if header:
                    SparcLog("========================================================================")
                    SparcLog("!{}!".format(f"[Iteration {iter}] Computing Energy and Forces for Candidates".center(70)))
                    SparcLog("========================================================================")
                CalculateDFTEnergy(
                    idx=idx, 
                    header=(idx == 1),
                    system=poscar_,
                    timestep=config['md_simulation']['timestep_fs'] * ase.units.fs,
                    log_filename=f"Iter{iter}_{config['output']['log_file']}",
                    trajfile=config['output']['aimdtraj_file'],
                    dir_name=iter_structure['dft_dir']
                )
                save_progress({'state': str(iter_structure['dft_dir']), 'iteration': iter, 'candidate': candidates, 'idx': idx})
            # Re-train DeepMD models
            SparcLog("========================================================================")
            SparcLog("Processing Data for DeepMD Training\n")
            combined_traj = combine_trajectories(
                trajfilename=config['output']['aimdtraj_file'],
                current_iter=iter_structure['iter_num']
            )
            get_data(
                ase_traj=combined_traj, 
                dir_name=config['deepmd_setup']['data_dir'], 
                skip_min=0, 
                skip_max=None
            )
            SparcLog("!{}!".format("Starting MLIP Training".center(70)))
            # SparcLog("Starting MLIP Training")
            deepmd_training(
                active_learning=True,
                training_dir=iter_structure['train_dir'],
                num_models=config['deepmd_setup']['num_models'], 
                input_file=config['deepmd_setup']['input_file'],
                datadir=datadir,
                atom_types=atom_types
            )

            SparcLog("{}".format("Setting up DeepPotential Calculator".center(72)))
            #--------------------------------------------------------------------------------------#
            # Set DeepPotential calculator
            #--------------------------------------------------------------------------------------#
            n_sample = config['deepmd_setup'].get('multiple_run', 1)           
            SparcLog("========================================================================")
            SparcLog("      MULTIPLE SAMPLE MD-SIMULATION STARTING FROM SAME CONFIGURATION!")
            SparcLog("========================================================================")
            for i in range(n_sample):
                dp_system = original_system
                dp_atoms, dp_calc = setup_DeepPotential(
                    atoms=dp_system, 
                    model_path=parent_dir, 
                    model_name=latest_models[0]
                )
                SparcLog("\n{}".format("Initializing DeepMD Simulation".center(72)))
                if thermostat not in thermostat_func:
                    raise ValueError(f"Unknown thermostat: {thermostat}")

                dyn_dp_class = thermostat_func[thermostat]
                if thermostat == 'Nose':
                    dyn_dp = dyn_dp_class(
                        atoms=dp_atoms, 
                        timestep=dp_tau,
                        tdamp=config['md_simulation']['tdamp'] * ase.units.fs, 
                        temperature=temperature
                    )
                else:
                    dyn_dp = dyn_dp_class(
                        atoms=dp_atoms, 
                        timestep=dp_tau,
                        temperature=temperature, 
                        friction=config['md_simulation']['friction'] / ase.units.fs
                    )
                
                MDsteps = config['deepmd_setup']['md_steps']
                writePace = config['deepmd_setup']['log_frequency']
                
                dp_plumed_is = config['deepmd_setup']['use_plumed']
                umbrella_enabled = config['deepmd_setup']['umbrella_sampling'].get('enabled', False)
                SparcLog(f"Umbrella Sampling : {umbrella_enabled}")
                #
                if dp_plumed_is and umbrella_enabled:
                    SparcLog("========================================================================")
                    SparcLog("       Umbrella Sampling Enabled — Running MLMD Windows with PLUMED       ")
                    SparcLog("========================================================================")

                    umbrella(config=config, 
                            us_dir=iter_structure, 
                            dp_path=parent_dir, 
                            dp_model=latest_models[0]
                            )
                    break
                if dp_plumed_is and not umbrella_enabled:
                    SparcLog("========================================================================")
                    SparcLog("               PLUMED IS CALLED FOR DPMD SIMULATION !")
                    SparcLog("========================================================================")
                    
                    plumed_file = config['deepmd_setup'].get('plumed_file', 'plumed.dat')
                    SparcLog(f"Using PLUMED input file: {plumed_file}")
                    
                    dp_atoms.calc = modify_forces(
                        calculator=dp_calc, 
                        system=dp_atoms, 
                        timestep=dp_tau, 
                        kT=config['plumed']['kT'],
                        restart=config['plumed']['restart'], 
                        plumed_input=plumed_file,
                        iteration=iter_structure['dpmd_dir']
                    )
                else:
                    dp_atoms.calc = dp_calc
                
                ExecuteMlpDynamics(
                    system=dp_atoms,
                    dyn=dyn_dp,
                    steps=MDsteps,
                    pace=writePace,
                    log_filename=f"Iter{iter}_dpmd.log",
                    trajfile=config['output']['dptraj_file'],
                    dir_name=iter_structure['dpmd_dir'],
                    distance_metrics=config['distance_metrics'],
                    name=thermostat,
                    epot_threshold=config['deepmd_setup']['epot_threshold']
                )
            
            # Check for new candidates
            candidate_found_is, labelled_files, latest_models = QueryByCommittee(
                trajfile=iter_structure['dpmd_dir'] / config['output']['dptraj_file'], 
                model_path=iter_structure['train_dir'], 
                num_models=config['deepmd_setup']['num_models'],
                min_lim=config['model_dev']['f_min_dev'],
                max_lim=config['model_dev']['f_max_dev'],
                dpmd_data_path=iter_structure['dpmd_dir'],
                iteration=iter
            )
            candidates = len(labelled_files)
            if not candidate_found_is:
                SparcLog("========================================================================")
                SparcLog("                 No More Candidates Found for Labelling.                 ")
                SparcLog("                 End of Active Learning Loop.                             ")
                SparcLog("========================================================================")
                break
            
            # Increment AL iteration and save progress
            iter += 1
            i_start = 1
            save_progress({'state': str(iter_structure['dft_dir']), 'iteration': iter, 'candidate': candidates})
    
    # Validate required configuration parameters
    required_params = [
        'general.structure_file',
    ]
    
    for param in required_params:
        section, key = param.split('.')
        if section not in config or key not in config[section]:
            raise ValueError(f"Missing required parameter: {param}")

if __name__ == '__main__':
    main()
#----------------------------
# Uncomment for Profiling
#----------------------------
    # pr = cProfile.Profile()  # Create profiler instance
    # pr.enable()  # Start profiling

    # main()  # Run function

    # pr.disable()  # Stop profiling

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats(filename='sparc.prof')  # Save profile data

#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#

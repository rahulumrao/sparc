# main.py
#!==============================================================!
import yaml
import os
import sys
import subprocess
import argparse
from ase.io import read, write
import ase.units
from ase.md import MDLogger

#!==============================================================!
from src.read_input import load_config
from src.deepmd_training import deepmd_training
from src.vasp_setup import setup_dft_calculator
from src.ase_md import NoseNVT, run_aimd, run_Ase_DPMD
from src.plumed_wrapper import modify_forces
from src.data_processing import get_data
from src.utils import log_md_setup, save_xyz, wrap_positions
from src.dpmd_setup import setup_DeepPotential
from src.active_learning import QueryByCommittee
#!==============================================================!

'''
    This is the main function that runs the MD simulation and the DeepMD training.
'''

def main():
    #------------------------------------------------------------------------------------
    # Command line arguments
    parser = argparse.ArgumentParser(description='Run AIMD and DeepMD simulations.')
    parser.add_argument("-i", "--input_file", type=str, default="input.yaml", help="Input YAML file")
    args = parser.parse_args()
    try:
        yaml_config = load_config(args.input_file)
        print("\nYAML configuration loaded successfully!\n")
    
    except FileNotFoundError:
        print(f"\n Error: YAML configuration File not found! (default: input.yaml)\n")
        sys.exit(1)
    config=yaml_config
    
    #------------------------------------------------------------------------------------
    # Read atomic structure
    parent_dir = os.getcwd()
    system = read(config['general']['structure_file'])   #read('POSCAR')
    system.set_pbc([True, True, True])
    system.center()
    original_system = system
    # Set up DFT calculator
    vasp_calc = setup_dft_calculator(config)
    # system.calc = vasp_calc
    #------------------------------------------------------------------------------------
    # Initialize variables
    timestep = config['md_simulation']['timestep_fs'] * ase.units.fs
    temperature = config['md_simulation']['temperature']        # Target temperature in Kelvin
    DftMDSteps = config['general']['md_steps']
    plumed_is = config['md_simulation']['use_plumed']           # Check if Plumed is to be wrapped with AIMD run
    #------------------------------------------------------------------------------------
    #=======================================================================================
    # Setup AIMD simulation
    #=======================================================================================
    dftmd_is = DftMDSteps > 0
    # Set up MD simulations
    if (dftmd_is):
        # Only when it is True
        print("\n========================================================================")
        print(f"\n ! ab-initio MD Simulations will be performed at Temp.: {temperature}K !")
        print("\n========================================================================")
        dyn_nose = NoseNVT(
            atoms=system, 
            timestep=timestep,
            tdamp=config['md_simulation']['tdamp'] * ase.units.fs,
            temperature=temperature)
        
        # Plumed wrapper function for AIMD run
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
            plumed_input=config['plumed']['input_file'])
        else:
            system.calc = vasp_calc
    
        # Run MD simulation
        run_aimd(
            system=system, 
            dyn=dyn_nose, 
            steps=DftMDSteps, 
            pace=config['general']['log_frequency'], 
            log_filename=config['output']['log_file'],
            trajfile=config['output']['aimdtraj_file'])
        
    #=======================================================================================
    # DeepMD Training
    #=======================================================================================        
    training_is = config['deepmd_setup']['training']
    #
    if training_is:
        get_data(ase_traj=config['output']['aimdtraj_file'], 
                dir_name=config['deepmd_setup']['data_dir'], 
                skip_min=0, skip_max=None)
        
        deepmd_training(
            training_dir=config['deepmd_setup']['train_dir'],
            num_models=config['deepmd_setup']['num_models'], 
            input_file=config['deepmd_setup']['input_file'])

    #=======================================================================================
    # DeepPotential Molecular Dynamics Run
    #=======================================================================================     
    dpmd_run_is = config['deepmd_setup']['MdSimulation']
    if dpmd_run_is:
        dp_path = config['deepmd_setup']['train_dir']
        dp_names = config['deepmd_setup']['model_name']
        dp_calc = setup_DeepPotential(atoms=system, model_path=dp_path, model_name=dp_names)
        original_system.calc = dp_calc

        dyn_dp = NoseNVT(
            atoms=dp_calc, 
            timestep=timestep,
            tdamp=config['md_simulation']['tdamp'] * ase.units.fs, 
            temperature=temperature)
        
        MDsteps = config['deepmd_setup']['md_steps']
        writePace = config['deepmd_setup']['log_frequency']
        
        run_Ase_DPMD(
            system=dp_calc,
            dyn=dyn_dp,
            steps=MDsteps,
            pace=writePace,
            log_filename=config['deepmd_setup']['log_file'],
            trajfile=config['output']['dptraj_file'])
        
        # Run Active Learning and check for candidates
        candidate_found_is, labelled_files, latest_models = QueryByCommittee(
                                    trajfile=config['output']['dptraj_file'], 
                                    model_path=config['deepmd_setup']['train_dir'], 
                                    num_models=config['deepmd_setup']['num_models'], 
                                    data_path=config['deepmd_setup']['dpmd_dir'],
                                    iteration=0)
        if not candidate_found_is:
            print("\n========================================================================")
            print("                 No more candidates found for labelling.                 ")
            print("                 End of Active Learning Loop.                             ")
            print("========================================================================")
            return  # Use return instead of break to exit the function
        else:
            print(f"Candidates found for labelling: {candidate_found_is}")
            # print(f"Latest models: {latest_models}")
    #=======================================================================================
    # Run DeepMD training after MD run finished
    #=======================================================================================
    learning_is = config['active_learning']
    training_is = True
    print("\n========================================================================")
    print(f"\n !  Active Learning Protocol for Training is set to : {training_is}    !")

    if learning_is:
        iter = 1
        while candidate_found_is:
            print(f"\n========================================================================")
            print(f"                     Starting iteration {iter}                            ")
            print(f"========================================================================")
            
            # If candidates are found, label them and run AIMD for each
            for idx, files in enumerate(labelled_files):
                poscar_ = read(files, format='vasp')
                poscar_.calc = setup_dft_calculator(config)
                
                # Run AIMD for each labelled structure
                header = True if idx == 0 else False
                if header:
                    print("\n========================================================================")
                    print(f"  [Iteration :{iter}] Computing Energy and Forces for Candidates !")
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
                    header=header,
                    mode='a')
            
            # Process data for DeepMD
            get_data(ase_traj=config['output']['aimdtraj_file'], 
                    dir_name=config['deepmd_setup']['data_dir'], 
                    skip_min=0, skip_max=None)
            
            # Extract data from and appended trajectory
            deepmd_training(
                training_dir=config['deepmd_setup']['train_dir'],
                num_models=config['deepmd_setup']['num_models'], 
                input_file=config['deepmd_setup']['input_file']) 

            # Set DeepPotential calculator
            dp_calc = setup_DeepPotential(atoms=system, model_path=parent_dir, model_name=latest_models[0])
            original_system.calc = dp_calc

            # DeepMD Simulation
            dyn_dp = NoseNVT(
                atoms=dp_calc, 
                timestep=timestep,
                tdamp=config['md_simulation']['tdamp'] * ase.units.fs, 
                temperature=temperature)
            
            MDsteps = config['deepmd_setup']['md_steps']
            writePace = config['deepmd_setup']['log_frequency']
            
            # Perform DeepPotential MD siulation with re-trained model
            run_Ase_DPMD(
                system=dp_calc,
                dyn=dyn_dp,
                steps=MDsteps,
                pace=writePace,
                log_filename=f"Iter{iter}_{config['deepmd_setup']['log_file']}",
                trajfile=config['output']['dptraj_file'])
            
            # Re-check candidates for labelling
            candidate_found_is, labelled_files, latest_models = QueryByCommittee(
                                trajfile=config['output']['dptraj_file'], 
                                model_path=config['deepmd_setup']['train_dir'], 
                                num_models=config['deepmd_setup']['num_models'], 
                                data_path=config['deepmd_setup']['dpmd_dir'],
                                iteration=iter)

            # If no candidate found, break the loop
            if not candidate_found_is:
                print("\n========================================================================")
                print("                 No more candidates found for labelling.                 ")
                print("                 End of Active Learning Loop.                             ")
                print("========================================================================")
                break
            
            # Increment iteration counter before next iteration
            iter += 1
        
        #------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#
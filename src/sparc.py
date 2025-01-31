# main.py
#!==============================================================!
import yaml
import sys
import argparse
from ase.io import read, write
import ase.units
from ase.md import MDLogger

#!==============================================================!
from src.read_input import load_config
from src.deepmd_training import deepmd_training
from src.vasp_setup import setup_dft_calculator
from src.ase_md import NoseNVT, run_md, run_Ase_DPMD
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
    # Command line arguments
    parser = argparse.ArgumentParser(description='Run AIMD and DeepMD simulations.')
    parser.add_argument("-i", "--input_file", type=str, default="input.yaml", help="Input YAML file")
    args = parser.parse_args()
    try:
        yaml_config = load_config(args.input_file)
        print("\nYAML configuration loaded successfully!\n")
    # print(yaml_config)
    except FileNotFoundError:
        print(f"\n Error: YAML configuration File not found! (default: input.yaml)\n")
        sys.exit(1)
    config=yaml_config
    
    # Read atomic structure
    system = read(config['general']['structure_file'])   #read('POSCAR')
    system.set_pbc([True, True, True])
    system.center()
    original_system = system
    # Set up DFT calculator
    vasp_calc = setup_dft_calculator(config)
    # system.calc = vasp_calc
    
    # Initialize variables
    timestep = config['md_simulation']['timestep_fs'] * ase.units.fs
    temperature = config['md_simulation']['temperature']  # Target temperature in Kelvin
    DftMDSteps = config['general']['md_steps']
    
    if DftMDSteps <= 0:
        dftmd_is = False
    else:
        dftmd_is = True
    # Set up MD simulation
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
        
        # Check if Plumed is to be wrapped with AIMD run
        plumed_is = config['md_simulation']['use_plumed']
        # Plumed wrapper function for AIMD run
        if (plumed_is):
            print("\n========================================================================")
            print("               PLUMED IS CALLED FOR MD SIMULATION. !")
            print("========================================================================")
            # plumed_calc = modify_forces(
            # calculator=vasp_calc, 
            # system=system, timestep=timestep, 
            # temperature=temperature, 
            # kT=config['plumed']['kT'], 
            # plumed_input=config['plumed']['input_file'])
            system.calc = modify_forces(
            calculator=vasp_calc, 
            system=system, timestep=timestep, 
            temperature=temperature, 
            kT=config['plumed']['kT'], 
            plumed_input=config['plumed']['input_file'])
        else:
            system.calc = vasp_calc
    
        # Run MD simulation
        run_md(
            system=system, 
            dyn=dyn_nose, 
            steps=DftMDSteps, 
            pace=config['general']['log_frequency'], 
            log_filename=config['output']['log_file'],
            trajfile=config['output']['aimdtraj_file'])
    
    # Run DeepMD training after MD run finished
    training_is = config['deepmd_setup']['training']
    # print('DeePMD Training is:',training_is)
    print("\n========================================================================")
    print(f"\n !    DeePMD Training is set to : {training_is}    !")
    #
    if (training_is):
        deepmd_dir = config['deepmd_setup']['data_dir'] #'DeePMD_training/00.data'
        # get_data(ase_traj='md.traj', dir_name=deepmd_dir, skip_min=0, skip_max=None)
        get_data(ase_traj=config['output']['aimdtraj_file'], 
                 dir_name=deepmd_dir, 
                 skip_min=0, skip_max=None)
        deepmd_training(
            training_dir=config['deepmd_setup']['train_dir'],
            num_models=config['deepmd_setup']['num_models'], 
            input_file=config['deepmd_setup']['input_file']) 
    
    # If MD with Deep potential is True
    dpmd_run_is = config['deepmd_setup']['MdSimulation']
    if (dpmd_run_is):
        dp_path = config['deepmd_setup']['train_dir']
        dp_names = config['deepmd_setup']['model_name'] #'./models'
        dp_calc = setup_DeepPotential(atoms=system, model_path=dp_path, model_name=dp_names)
        original_system.calc = dp_calc
        
        dyn_dp = NoseNVT(
            atoms=dp_calc, 
            timestep=timestep,
            tdamp=config['md_simulation']['tdamp'] * ase.units.fs, 
            temperature=temperature)
        
        # dyn_dp.attach(lambda: wrap_positions(apos=system), interval=1)
        # dyn_nose = NoseNVT(
        # atoms=dp_calc, 
        # timestep=timestep, 
        # temperature=temperature)
    
        # logger = MDLogger(dyn_dp, dp_calc, 'dpmd.log', header=True, stress=False, peratom=False, mode="w")
        # dyn_dp.attach(logger, interval=5)
        # dyn_dp.run(100)
        ########################################################################
        # ASE DeepPotential Molecular Dynamics
        MDsteps=config['deepmd_setup']['md_steps'],
        writePace=config['deepmd_setup']['log_frequency']
        # print(MDsteps[0], writePace)
        ######################################################################## 
        
        run_Ase_DPMD(
            system=dp_calc,
            dyn=dyn_dp,
            steps=MDsteps[0],
            pace=writePace,
            log_filename=config['deepmd_setup']['log_file'],
            trajfile=config['output']['dptraj_file'])
        
        # Run Active Learning
        QueryByCommittee(trajfile=config['output']['dptraj_file'], 
                         model_path=config['deepmd_setup']['train_dir'], 
                         num_models=config['deepmd_setup']['num_models'], 
                         data_path=config['deepmd_setup']['dpmd_dir'])
                        
if __name__ == '__main__':
    main()
    # parser = argparse.ArgumentParser(description='Run AIMD and DeepMD simulations.')
    # parser.add_argument("-i", "--input_file", type=str, default="input.yaml", help="Input YAML file")
    # args = parser.parse_args()
    # try:
    #     yaml_config = load_config(args.input_file)
    #     print("\nYAML configuration loaded successfully!\n")
    #     # print(yaml_config)
    # except FileNotFoundError:
    #     print(f"\n Error: YAML configuration File not found! (default: input.yaml)\n")
    #     sys.exit(1)
    # Run the main function
    # main()
    
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#
#!/usr/bin/python3
# read_input.py
################################################################
import yaml
import os
import sys
################################################################
# Local Import
from sparc.src.utils.logger import SparcLog
################################################################
def load_config(input_file: str = "input.yaml") -> dict:
    """Load and validate SPARC configuration from YAML file.
    
    Args:
        input_file: Path to YAML configuration file
        
    Returns:
        Dictionary containing validated configuration parameters
        
    Raises:
        ValueError: If required parameters are missing
        yaml.YAMLError: If YAML file is invalid
    """
    with open(input_file, "r") as file:
        config = yaml.safe_load(file)
        SparcLog("========================================================================")
        SparcLog("  Input Configurations (- PLEASE CHECK SPARC INPUTS CAREFULLY! -)")
        SparcLog("========================================================================")
        SparcLog(yaml.dump(config, default_flow_style=False))

    config['general']['structure_file'] = config['general'].get('structure_file', 'POSCAR')
    # Set defaults for missing keys
    if 'dft_calculator' in config:
        config['dft_calculator']['prec'] = config['dft_calculator'].get('prec', 'Normal')
        config['dft_calculator']['kgamma'] = config['dft_calculator'].get('kgamma', 'True')
        config['dft_calculator']['name'] = config['dft_calculator'].get('name', 'VASP')
        config['dft_calculator']['incar_file'] = config['dft_calculator'].get('incar_file', 'INCAR') 
        # AIMD Settings
        config['md_simulation'] = config.get('md_simulation', {})
        config['md_simulation']['timestep_fs'] = config['md_simulation'].get('timestep_fs', 1.0)
        config['md_simulation']['temperature'] = config['md_simulation'].get('temperature', 300.0)
        config['md_simulation']['thermostat'] = config['md_simulation'].get('thermostat', 'Nose')
    
        # Only if this condition is True
        if 'md_simulation' in config:
            if config["md_simulation"].get("thermostat") == "Nose":
                tdamp = config["md_simulation"].get("tdamp", None)  # Default to None if tdamp is not found
                if tdamp is None:
                    raise KeyError("WARNING!!!: key[tdamp] is Required When Using Nose-Hoover Thermostat!")
            elif config["md_simulation"].get("thermostat") == "Langevin":
                friction = config["md_simulation"].get("friction", None)  # Default to None if friction is not found
                if friction is None:
                    raise KeyError("WARNING!!!: key[friction] is Required When Using Langevin Thermostat!")
        
        config['md_simulation']['log_frequency'] = config['md_simulation'].get('log_frequency', 1)
        config['md_simulation']['restart'] = config['md_simulation'].get('restart', False)
        config['md_simulation']['use_dft_plumed'] = config['md_simulation'].get('use_dft_plumed', False)
        use_dft_plumed = config['md_simulation']['use_dft_plumed']
        #
        config['plumed'] = config.get('plumed', {})
        config['plumed']['restart'] = config['plumed'].get('restart', False)
        config['plumed']['kT'] = config['plumed'].get('kT', 0.02585)                    # eV --> 2.5 kj/mol
        config['md_simulation']['plumed_file'] = config['md_simulation'].get('plumed_file', 'plumed.dat')
        if use_dft_plumed:
            plumed_file = config['md_simulation']['plumed_file']
            if not os.path.exists(plumed_file):
                SparcLog(f"Warning: PLUMED is enabled, but the input file does not exist.")
                sys.exit(1)

    #
    if 'deepmd_setup' in config:
        config['deepmd_setup'] = config.get('deepmd_setup', {})
        config['deepmd_setup']['data_dir'] = config['deepmd_setup'].get('data_dir', 'DeePMD_training/00.data')
        config['deepmd_setup']['input_file'] = config['deepmd_setup'].get('input_file', 'input.json')
        if not os.path.exists(config['deepmd_setup']['input_file']):
            SparcLog(f"Warning!!!: DeepMD Training Input File Not Found!\n")
            sys.exit(1)
        config['deepmd_setup']['model_name'] = config['deepmd_setup'].get('model_name', 'models')
        config['deepmd_setup']['training'] = config['deepmd_setup'].get('training', False)
        config['deepmd_setup']['MdSimulation'] = config['deepmd_setup'].get('MdSimulation', False)
        config['deepmd_setup']['log_frequency'] = config['deepmd_setup'].get('log_frequency', 1)
        config['deepmd_setup']['epot_threshold'] = config['deepmd_setup'].get('epot_threshold', 5.0)
        config['deepmd_setup']['use_plumed'] = config['deepmd_setup'].get('use_plumed', False)
        config['deepmd_setup']['umbrella_sampling'] = config['deepmd_setup'].get('umbrella_sampling', {})
        config['deepmd_setup']['umbrella_sampling']['enabled'] = config['deepmd_setup']['umbrella_sampling'].get('enabled', False)
        config['deepmd_setup']['umbrella_sampling']['config_file'] = config['deepmd_setup']['umbrella_sampling'].get('config_file', 'umbrella_sampling.yaml')
        
    # Add defaults for active learning and model_dev section
    config['active_learning'] = config.get('active_learning', False)
    config['learning_restart'] = config.get('learning_restart', False)
    config['iteration'] = config.get('iteration', 1)
    config['model_dev'] = config.get('model_dev', {})
    # Only set model_dev defaults if active_learning is True
    if config['active_learning']:
        config['model_dev']['f_min_dev'] = config['model_dev'].get('f_min_dev', 0.05)
        config['model_dev']['f_max_dev'] = config['model_dev'].get('f_max_dev', 0.20)            
    
    # Load distance metrics
    config['distance_metrics'] = config.get('distance_metrics', [])  # Default to empty list if not provided
    
    config['output'] = config.get('output', {})
    config['output']['log_file'] = config['output'].get('log_file', 'aimd.log')
    config['output']['aimdtraj_file'] = config['output'].get('aimdtraj_file', 'AseMD.traj')
    config['output']['dptraj_file'] = config['output'].get('dptraj_file', 'dpmd.traj')

    
    return config

if __name__ == "__main__":
    config = load_config()
    SparcLog(config)  # For debugging: print loaded configuration
    
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#

import yaml
import sys

def load_config(input_file="input.yaml"):
    """
        Reads the input configuration from a YAML file and sets up 
        the simulation parameters for SPARC.
        
        Args:
        -----
        input_file: str
            Path to the YAML file.
            
        Returns:
        --------
            config: dict
            A dictionary containing the input configuration.
    """
    with open(input_file, "r") as file:
        config = yaml.safe_load(file)
        print("\n========================================================================")
        print("  Input Configurations (- PLEASE CHECK SPARC INPUTS CAREFULLY! -)")
        print("========================================================================")
        print(yaml.dump(config, default_flow_style=False))
        
        # Check if the input file is in the correct format
        if config["md_simulation"]["thermostat"] == "Nose":
            tdamp = config["md_simulation"].get("tdamp", None)  # Default to None if tdamp is not found
            if tdamp is None:
                raise ValueError("tdamp is required when using Nose thermostat.")
    
    # Set defaults for missing keys
    config['dft_calculator']['prec'] = config['dft_calculator'].get('prec', 'Normal')
    config['general'] = config.get('general', {})
    config['general']['structure_file'] = config['general'].get('structure_file', 'POSCAR')
    config['output'] = config.get('output', {})
    config['output']['log_file'] = config['output'].get('log_file', 'aimd.log')
    config['output']['aimdtraj_file'] = config['output'].get('aimdtraj_file', 'AseMD.traj')
    config['output']['dptraj_file'] = config['output'].get('dptraj_file', 'dpmd.traj')
    config['deepmd_setup'] = config.get('deepmd_setup', {})
    config['deepmd_setup']['data_dir'] = config['deepmd_setup'].get('data_dir', 'DeePMD_training/00.data')
    config['deepmd_setup']['train_dir'] = config['deepmd_setup'].get('train_dir', 'DeePMD_training')
    config['deepmd_setup']['input_file'] = config['deepmd_setup'].get('input_file', '../input.json')
    config['deepmd_setup']['model_name'] = config['deepmd_setup'].get('model_name', 'models')
    config['md_simulation'] = config.get('md_simulation', {})
    config['md_simulation']['tdamp'] = config['md_simulation'].get('tdamp', 25)
    config['md_simulation']['timestep_fs'] = config['md_simulation'].get('timestep_fs', 1.0)
    config['md_simulation']['temperature'] = config['md_simulation'].get('temperature', 300.0)
    config['md_simulation']['use_plumed'] = config['md_simulation'].get('use_plumed', False)
    config['plumed'] = config.get('plumed', {})
    config['plumed']['kT'] = config['plumed'].get('kT', 2.5)
    config['plumed']['input_file'] = config['plumed'].get('input_file', 'plumed.dat')    
            
    
    return config

if __name__ == "__main__":
    config = load_config()
    print(config)  # For debugging: print loaded configuration
    
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#
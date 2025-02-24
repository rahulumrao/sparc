import os
from ase.calculators.vasp import Vasp
from pathlib import Path
from src.read_incar import parse_incar
#===================================================================================================#
def setup_dft_calculator(input_config, print_screen):
    """
    Setup the DFT calculator for ASE atom object.
    Currently supports the VASP calculator with extensive parameter options.
    
    Args:
        input_config (dict): Dictionary containing the configuration for setting up the DFT calculator.
            This should include paths, parameters, and the calculator name.
            The parameters in input_config['dft_calculator'] are passed directly to ASE's VASP calculator.
            
    Returns:
        vasp_calc: ASE Vasp object configured for DFT calculations.
        
    Raises:
        ValueError: If required parameters are missing or invalid
        FileNotFoundError: If VASP executable or INCAR file is not found
    """
    # Name of the DFT calculator
    dft_calc = input_config['dft_calculator']['name']
    
    # Set up VASP calculator for DFT calculations
    if (dft_calc == "VASP"):
        # Validate required parameters
        required_params = ['exe_command', 'prec', 'kgamma', 'incar_file']
        for param in required_params:
            if param not in input_config['dft_calculator']:
                raise ValueError(f"{param} must be provided in dft_calculator config")
        
        exe_run = input_config['dft_calculator']['exe_command']
        
        # Extract and validate VASP executable path
        vasp_exe = exe_run.split()[-1]
        vasp_path = Path(vasp_exe)
        if not vasp_path.is_absolute():
            raise ValueError(f"VASP executable path must be absolute: {vasp_exe}")
        if not vasp_path.exists():
            raise FileNotFoundError(f"VASP executable not found: {vasp_exe}")
            
        # Validate INCAR file path
        incar_path = Path(input_config['dft_calculator']['incar_file'])
        if not incar_path.exists():
            raise FileNotFoundError(f"INCAR file not found: {incar_path}")
        #---------------------------------------------------------------------------------------------#
        # Parse INCAR parameters
        incar_params = parse_incar(str(incar_path))
        if print_screen:
            # Print the INCAR parameters in a box
            print("\n" + "="*50)
            print("              INCAR PARAMETERS                ")
            print("="*50)
            # Find the length of the longest key for alignment
            max_key_length = max(len(key) for key in incar_params.keys())
            for key, value in incar_params.items():
                # 
                padded_key = f"{key.upper():<{max_key_length}}"
                print(f"  {padded_key} : {value}")
            print("="*50 + "\n")
        #---------------------------------------------------------------------------------------------#
        # Set gamma based on kgamma
        gamma_point = not input_config['dft_calculator'].get('kgamma', False)   
        
        # Get working directory from config or use default
        vasp_dir = input_config['dft_calculator'].get('directory', 'vasp')
        
        # Get exchange-correlation and pseudopotential settings
        xc_functional = input_config['dft_calculator'].get('xc', 'PBE')
        pseudopotential = input_config['dft_calculator'].get('pp', 'PBE')
        
        vasp_calc = Vasp(
            prec=input_config['dft_calculator']['prec'],
            kgamma=input_config['dft_calculator']['kgamma'],
            gamma=gamma_point,
            xc=xc_functional,
            pp=pseudopotential,
            directory=vasp_dir,  # Use configurable directory
            command=exe_run,
            **incar_params)
    else:
        raise ValueError(f"Unsupported DFT calculator: {dft_calc}. Currently only VASP is supported.")
    
    return vasp_calc
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#        
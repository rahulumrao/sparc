import os
from ase.calculators.vasp import Vasp
from pathlib import Path
#===================================================================================================#
def setup_dft_calculator(input_config):
    """
        This function setup the DFT calculator for ASE atom object,
        Currently supports the VASP calculator but will be updated for others as well.
        Checkout the ASE documentation for more details, see:
        https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#module-ase.calculators.
        
    Args:
    -----
        input_config (dict): Dictionary containing the configuration for setting up the DFT calculator. 
        This should include paths, parameters, and the calculator name.
        
    Returns:
    --------
        vasp_calc: ASE Vasp object configured for DFT calculations.                
    
    """
    # Name of the DFT calculator
    dft_calc = input_config['dft_calculator']['name']
    
    # Set up VASP calculator for DFT calculations
    if (dft_calc == "VASP"):
        # Set Environment for VASP
        # example: command='mpiexec -np 2 /home/prg/Softwares/VASP_gcc/VASP-642_CPU/bin/vasp_std',
        # Construct the full path for the VASP executable using the provided path and name
        exe_run = os.path.join(
            input_config['dft_calculator']['exe_path'],
            input_config['dft_calculator']['exe_name'])
        
        # If an executable command is provided, override the default executable path
        if (input_config['dft_calculator']['exe_command']) is not None:
            exe_run = input_config['dft_calculator']['exe_command']
        
        # Validate path
        exe_path = input_config['dft_calculator']['exe_path']
        if not os.path.isabs(exe_path):
            raise ValueError("Executable path must be absolute")
        
        exe_path = Path(exe_path).resolve()
        if not exe_path.exists():
            raise FileNotFoundError(f"Executable not found: {exe_path}")
        
        # Set up the VASP calculator with the necessary parameters                       
        vasp_calc = Vasp(
            prec=input_config['dft_calculator']['prec'],    # Precision of the calculation
            encut=input_config['dft_calculator']['encut'],  # Plane-Wave Energy cutoff
            endiff=input_config['dft_calculator']['ediff'], # Convergence criteria
            xc='PBE',                                       # Exchange-Correlation functional
            pp='PBE',                                       # Pseudopotential functional
            gamma=input_config['dft_calculator']['kgamma'], # Gamma point
            directory='vasp',                               # Directory to write the files
            command=exe_run,                                # VASP executable command
        )
    
    return vasp_calc

#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#        
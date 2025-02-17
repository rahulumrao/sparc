import os
from ase import Atoms
from deepmd.calculator import DP


def setup_DeepPotential(atoms, model_path, model_name="frozen_model.pb"):
    """
    Setup a DeepPotential calculator for an ASE atoms object.

    Args:
        atoms: ase.Atoms
            The atomic structure to assign the DeepPotential model to
        model_path: str
            Path to the directory containing DeepPotential model
        model_name: str, optional
            Name of the DeepPotential model file (default: "frozen_model.pb")

    Returns:
        ase.Atoms: ASE atoms object with DeepPotential calculator attached

    Raises:
        Exception: If an issue occurs while setting up or testing the DeepPotential model
    """
    # Construct full path to model file
    dp_model = os.path.join(model_path, model_name)
    dp_calc = DP(model=dp_model)
    
    # Create atoms object with DeepPotential calculator
    dp_system = Atoms(atoms, calculator=dp_calc)
    
    try:
        # Test calculator by computing energy and forces
        potential_energy = dp_system.get_potential_energy()
        forces = dp_system.get_forces()
        
        if potential_energy is not None and forces is not None:
            print("\n" + "=" * 72)
            print(f"DeepPotential model successfully loaded and tested: {dp_model}")
            print("=" * 72)
        else:
            print("\n" + "=" * 72)
            print("Error: Failed to compute energy and forces with DeepPotential model")
            print("=" * 72)
            
    except Exception as e:
        print("\n" + "=" * 72)
        print("Error: Failed to setup DeepPotential model")
        print(f"Details: {str(e)}")
        print("=" * 72)
    
    return dp_system, dp_calc
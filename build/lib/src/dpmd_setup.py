import os
from ase import Atoms
from deepmd.calculator import DP


def setup_DeepPotential(atoms, model_path, model_name="frozen_model.pb"):
    """
        Setup the DeepPotential calculator for a ASE atoms object.
        
        Parameters:
        ------------
        atoms: ase.Atoms
            The atomic structure to which the the DeepPotential model will be assigned.
        
        model_path: str
            Path to the directory containing DeepPotential model.
        
        model_name: str
            Name of the DeepPotential model file (default: "frozen_model.pb").
            
        Returns:
        --------
        dp_system: ase.Atoms
            ASE atoms object with DeepPotential calculator attached.
            
        Raises:
        -------
        Exception: If an issue occurs while setting up the DeepPotential model.
    """
    dp_model = os.path.join(model_path, model_name)
    
    dp_system = Atoms(atoms, calculator=DP(model=dp_model))
    #
    try:
        # Check if potential energy and forces are computed
        Epot = dp_system.get_potential_energy()
        Forces = dp_system.get_forces()
        
        if Epot is not None and Forces is not None:
            print("\n========================================================================")
            print("    DeepPotential Model Successfully Loaded.!")
            # print("========================================================================")
        else:
            print("\n========================================================================")
            print("    Error: An issue occurred while setting up the DeepPotential model.")
            print("========================================================================")
    except Exception as e:
        print("\n========================================================================")
        print("    Error: An issue occurred while setting up the DeepPotential model.")
        print(f"    Details: {str(e)}")
        print("========================================================================")
    
    
    # print("Potential Computed using DP Model: ",dp_system.get_potential_energy())
    # print("Forces Computed using DP Model:" , dp_system.get_forces())
    
    return dp_system
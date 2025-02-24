import numpy as np
from ase.io import write
from ase.io.trajectory import TrajectoryWriter
import pickle
from pathlib import Path
from ase.io import read, write    
import os
from ase.geometry import get_angles
#===================================================================================================
"""
    Function is called to log the dynamics. It write the potential energy (Epot), kinetic energy (Ekin),
    total energy (Epot + Ekin), and temperature (Temp) of the system..
"""
#---------------------------------------------------------------------------------------------------#   
def create_iteration_dirs(iter_num):
    """Create iteration directory structure."""
    iter_name = f"iter_{iter_num:06d}"
    iter_dir = Path(iter_name)
    iter_dir.mkdir(exist_ok=True)
    
    # Create subdirectories with new naming
    dft_dir = iter_dir/"00.dft"      # DFT calculations (VASP)
    train_dir = iter_dir/"01.train"   # DeepMD training
    dpmd_dir = iter_dir/"02.dpmd"     # DeepMD runs and model deviation
    
    # Print iteration information
    print("\n" + "="*72)
    print(f"Creating directories for Iteration: {iter_num:06d}")
    print("="*72)
    print(f"├── {iter_name}/")
    print(f"│   ├── {dft_dir.name}/")
    print(f"│   ├── {train_dir.name}/")
    print(f"│   └── {dpmd_dir.name}/")
    print("="*72 + "\n")
    
    for folder in [dft_dir, train_dir, dpmd_dir]:
        folder.mkdir(exist_ok=True)
        
    return {
        'iter_num': iter_num,
        'iter_dir': iter_dir,
        'dft_dir': dft_dir,
        'train_dir': train_dir,
        'dpmd_dir': dpmd_dir
    }
#---------------------------------------------------------------------------------------------------#   
def log_md_setup(dyn, atoms, dir_name, write_dist=False):
    """
    Log molecular dynamics simulation details including energies and temperature.

    Args:
        dyn: ASE dynamics object
            The molecular dynamics simulation object
        atoms: ase.Atoms
            The atomic system being simulated
        dir_name: str
            Directory path for log files
        write_dist: bool, optional
            Whether to log distance between atoms 0 and 4 (default: False)
    """
    # Get energies and ensure they're scalar values
    epot = atoms.get_potential_energy()
    if isinstance(epot, (list, np.ndarray)):
        epot = epot[0]
    ekin = atoms.get_kinetic_energy()
    total = epot + ekin
    
    # Get other system properties
    step = dyn.get_number_of_steps()
    temp = float(atoms.get_temperature())
    
    # Print current step info to the console
    print(f'Steps: {step}, Epot: {epot:.6f}, Ekin: {ekin:.6f}, Temp: {temp:.2f}')
    
    # Write to log file using MDLogger context manager
    with MDLogger(f"{dir_name}/AseMolDyn.log") as log:
        if step == 0:
            log.file.write(f"# {'Steps':<6} {'Epot':<10} {'Ekin':<10} {'Total':<10} {'Temp':<6}\n")
        log.file.write(f"{float(step):<8} {epot:<10.6f} {ekin:<10.6f} {total:<10.6f} {temp:<6.2f}\n")
    
    # Optionally log atomic distances using MDLogger
    if write_dist:
        distance = atoms.get_distance(0, 4, mic=True)
        with MDLogger(f"{dir_name}/dist.dat") as dist_log:
            dist_log.file.write(f"Step: {step}, Distance: {distance:.6f}\n")

#---------------------------------------------------------------------------------------------------#                
def save_xyz(atoms, trajfile, write_mode, dir_name):
    """
    Save atomic configuration to trajectory and XYZ files.

    Args:
        atoms: ase.Atoms
            The atomic system to save
        trajfile: str
            Path to the trajectory file
    
    Note:
        The 'stress' property is excluded when PLUMED is active since it's not supported
    """
    # Properties to save in trajectory
    # properties = ['energy', 'forces', 'coordinates', 'velocities', 'cell', 'pbc']
    properties = ['energy', 'forces', 'coordinates', 'cell', 'pbc']
    
    # Write to trajectory file
    # wrapped_atoms = wrap_positions(atoms)
    atoms.center()
    atoms.wrap()    
    
    # Write trajectory and XYZ files to specified directory
    traj_file = f"{dir_name}/{trajfile}"
    xyz_file = f"{dir_name}/AseTraj.xyz"
    
    trr = TrajectoryWriter(
        filename=traj_file,
        mode=write_mode,
        atoms=atoms,
        properties=properties
    )
    trr.write(atoms)
    
    # Save additional XYZ format
    write(xyz_file, atoms, append=True)
#---------------------------------------------------------------------------------------------------#
# Add context managers for file handling
class MDLogger:
    """
    Context manager for handling MD log files.
    
    Args:
        filename: str
            Path to the log file
    """
    def __init__(self, filename):
        self.filename = filename
        
    def __enter__(self):
        self.file = open(self.filename, 'a')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()        
        
def save_md_checkpoint(dyn, atoms, filename='md_checkpoint.pkl'):
    """
    Save molecular dynamics checkpoint to resume later.
    
    Args:
        dyn: ASE dynamics object
        atoms: ASE atoms object
        filename: str, checkpoint filename (default: 'md_checkpoint.pkl')
    """
    state = {
        'positions': atoms.get_positions(),
        'velocities': atoms.get_velocities(),
        'cell': atoms.get_cell(),
        'pbc': atoms.get_pbc(),
        'numbers': atoms.get_atomic_numbers(),
        'step': dyn.get_number_of_steps(),
        'momenta': dyn.atoms.get_momenta()
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
        
def load_md_checkpoint(filename='md_checkpoint.pkl'):
    """
    Load molecular dynamics checkpoint.
    
    Args:
        filename: str, checkpoint filename
        
    Returns:
        dict: Checkpoint state dictionary
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

#---------------------------------------------------------------------------------------------------#
def combine_trajectories(trajfilename, current_iter):
    """
    Combine trajectory files from all previous iterations and current iteration.
    
    Args:
        trajfilename (str): Name of the trajectory file to combine
        iter_dir (Path): Path to the current iteration directory
        current_iter (int): Current iteration number
        
    Returns:
        str: Path to combined trajectory file
    """
    # Create a list to store all trajectories
    all_frames = []
    
    # Read trajectories from previous iterations
    for i in range(current_iter + 1):
        iter_name = f"iter_{i:06d}"
        dft_traj = Path(iter_name) / "00.dft" / trajfilename
        if dft_traj.exists():
            frames = read(dft_traj, index=':')
            all_frames.extend(frames)
            print(f"Added {len(frames)} frames from iteration {i}")
    
    if not all_frames:
        raise ValueError("No trajectory data found from any iteration")
        
    # Write combined trajectory
    # combined_traj = iter_dir / "TrajCombined.traj"
    combined_traj =  "TrajCombined.traj"
    write(str(combined_traj), all_frames)
    print(f"\nCombined trajectory contains {len(all_frames)} frames")
    
    return str(combined_traj)   
#===================================================================================================#
def load_progress(progress_file='progress.txt'):
    """Load the last completed iteration and current state from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            lines = f.readlines()
            iteration = int(lines[0].strip())
            return iteration
    return 1  # Start from iteration 1 if no progress file exists

def save_progress(iteration, progress_file='progress.txt'):
    """Save the current iteration number and state to the progress file."""
    with open(progress_file, 'w') as f:
        f.write(f"{iteration}")    

def check_physical_limits(atoms, distance_metrics):
    """
    Check if any distances exceed physical limits.

    Args:
        atoms: ase.Atoms
            The atomic system to check.
        distance_metrics: list of dicts
            List of dictionaries containing pairs of atom indices and their max distances.

    Returns:
        bool: True if limits are exceeded, False otherwise.
    """
    for check in distance_metrics:
        atom1, atom2 = check['pair']
        max_distance = check['max_distance']
        distance = atoms.get_distance(atom1, atom2)
        if distance > max_distance:
            # Get chemical symbols for the atoms
            symbol1 = atoms.get_chemical_symbols()[atom1]
            symbol2 = atoms.get_chemical_symbols()[atom2]
            
            print("\n" + "=" * 50)
            print("  WARNING: DISTANCE EXCEEDED  ")
            print("-" * 50)
            print(f"  ATOMS: {symbol1} ({atom1}) -- {symbol2} ({atom2})  ")
            print(f"  MEASURED: {distance:.2f} Å   (LIMIT: {max_distance:.2f} Å)  ")
            print("=" * 50 + "\n")

            return True

    return False
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#   

     
        
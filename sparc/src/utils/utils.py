#!/usr/bin/python3
# utils.py
################################################################
import numpy as np
import os
import glob
import json
import pickle
from pathlib import Path
################################################################
# Third patty import 
from ase.io import read, write   
from ase.io.trajectory import TrajectoryWriter
from ase.geometry import get_angles
################################################################
# Local Import
from sparc.src.utils.logger import SparcLog
#===================================================================================================
"""
    Function is called to log the dynamics. It prints the potential energy (Epot), kinetic energy (Ekin),
    total energy (Epot + Ekin), and temperature (Temp) of the system to a file.
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
    SparcLog("="*72)
    SparcLog(f"Creating directories for Iteration: {iter_num:06d}")
    SparcLog("="*72)
    SparcLog(f"├── {iter_name}/")
    SparcLog(f"│   ├── {dft_dir.name}/")
    SparcLog(f"│   ├── {train_dir.name}/")
    SparcLog(f"│   └── {dpmd_dir.name}/")
    SparcLog("="*72 + "\n")
    
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
    SparcLog(f'Steps: {step}, Epot: {epot:.6f}, Ekin: {ekin:.6f}, Temp: {temp:.2f}')
    
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
#---------------------------------------------------------------------------------------------------#
def save_checkpoint(dyn, atoms, filename='md_checkpoint.pkl'):
    """
    Save molecular dynamics checkpoint to resume later.
    
    Args:
        dyn: ASE dynamics object
            The dynamics object containing simulation state
        atoms: ASE atoms object 
            The atoms object containing atomic positions, velocities etc.
        filename: str
            Checkpoint filename to save state (default: 'md_checkpoint.pkl')
    """
    positions = atoms.get_positions()
    velocities = atoms.get_velocities()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    numbers = atoms.get_atomic_numbers()
    momenta = dyn.atoms.get_momenta()
    step = dyn.get_number_of_steps()
    
    state = {
        'positions': positions,
        'velocities': velocities,
        'cell': cell,
        'pbc': pbc,
        'numbers': numbers,
        'step': step,
        'momenta': momenta
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
        
def load_checkpoint(atoms, filename='md_checkpoint.pkl'):
    """
    Load molecular dynamics checkpoint.
    
    Args:
        atoms: ASE atoms object
        filename: str, checkpoint filename (default: 'md_checkpoint.pkl')
        
    Returns:
        tuple: (atoms, mdstep) where atoms is the updated ASE atoms object and 
               mdstep is the MD step number from the checkpoint
    """
    if os.path.exists(filename):
        SparcLog(f"Restarting simulation from checkpoint: {filename}")
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        atoms.set_positions(state['positions'])
        atoms.set_velocities(state['velocities'])
        atoms.set_cell(state['cell']) 
        atoms.set_pbc(state['pbc'])
        atoms.set_atomic_numbers(state['numbers'])
        atoms.set_momenta(state['momenta'])
        mdstep = state['step']
    else:
        raise FileNotFoundError(f"\nCheckpoint file {filename} not found.")
    
    return atoms, mdstep
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
        
        SparcLog(f" Iteration [{i}]".ljust(20))
        SparcLog(f" → Checking File    : {dft_traj}")
        #
        if dft_traj.exists():
            frames = read(dft_traj, index=':')
            all_frames.extend(frames)
            SparcLog(f" → Added Frames     : {len(frames)}\n")
    
    if not all_frames:
        raise ValueError("No trajectory data found from any iteration")
        
    # Write combined trajectory
    # combined_traj = iter_dir / "TrajCombined.traj"
    combined_traj =  "TrajCombined.traj"
    write(str(combined_traj), all_frames)
    SparcLog("========================================================================")
    SparcLog(f" Total Frames in Combined Trajectory: {len(all_frames)}".center(72))
    SparcLog("========================================================================")
    return str(combined_traj)   
#===================================================================================================#
# Save current state of Active Learning iteration in a JSON file
#===================================================================================================#
def save_progress(state, progress_file='progress.json'):
    """
    Save the current iteration state to the progress file.
    This can be used to resume the iteration from the last saved state
    The state includes the current iteration number and current step
    """
    # try:
    #     with open(progress_file, 'r') as f:
    #         progress_data = json.load(f)
    # except (FileNotFoundError, json.JSONDecodeError):
    #     progress_data = {}
    
    # # Update the progress with the new state, which can include the iteration
    # progress_data.update(state)
    
    with open(progress_file, 'w') as f:
        json.dump(state, f, indent=4)

def load_progress(progress_file='progress.json'):
    """Load the last saved iteration state from the progress file."""
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            n_candidate = progress_data.get('candidate', None)
            i_candidate = progress_data.get('idx', None)
            # Assuming progress_data contains a state and iteration key
            if 'state' in progress_data and 'iteration' in progress_data:
                split_path = progress_data['state'].split('/')
                json_data = {
                    "iteration": progress_data['iteration'],
                    "directory": split_path[1],
                    "candidate": n_candidate,
                    "idx": i_candidate
                }
                return json_data
            else:
                return 0  # Return 0 if the expected keys are missing
    except (FileNotFoundError, json.JSONDecodeError):
        return 0
    
def restart_progress(start_iteration):
    '''
        Read labelled candidates in case of 
        
    Args:
        start_iterration (dict): dictionary containig the current state which includes:
            - 'iteration': last AL iteration.
            - 'idx': index of last processed candidates.
            - 'candidate': total number of candidates.
            
    Returns:
        tuple: (iddx, candidates, candidate_found_is, labelled_files)
            - iddx (int): Index of last processed candidate.
            - candidates (int): Total number of candidates
            - candidate_found_is (bool): True/False
            - labelled_files (list): List of POSCAR files for remaining candidates
    '''
    #   
    # Retrieve iteration
    iter = start_iteration.get('iteration')
    if iter is None:
        raise ValueError("Error: 'iteration' key is missing or None in the progress file.")
    
    # print(f"Resuming Active Learning from Iteration: {iter}")
    
    # Retrieve candidate
    iddx = start_iteration.get('idx')
    nid = start_iteration.get('candidate')
    
    # Check if candidate is found
    candidate_found_is = True if start_iteration.get('candidate') else False
    # print(f"Candidate Found: {candidate_found_is}")
    # print(f"Last Candidate: {iddx}, Total Candidates: {nid}")
    
    # candidates = len([f for f in Path(iter_folder, '02.dpmd', 'dft_candidates').iterdir() if f.is_dir()])
    iter_folder = Path(f"iter_{iter-1:06d}")
    candidate_dir = iter_folder / '02.dpmd' / 'dft_candidates'
    candidates = sum(1 for f in candidate_dir.iterdir() if f.is_dir())
     
    labelled_files=[]   # List to store candidate input (VASP: POSCAR)
    for serial in range(iddx, nid + 1):
        # serial_dir = (candidate_dir / f"{serial:04d}")
        # poscar_filename = serial_dir / "POSCAR"
        poscar_filename = (candidate_dir / f"{serial:04d}" / "POSCAR")
        labelled_files.append(str(poscar_filename))
    
    SparcLog("------------------------------------------------------------------------")
    SparcLog(f" Resuming Active Learning from Iteration: {iter}            ")
    SparcLog("------------------------------------------------------------------------")
    SparcLog(f" Candidate Folder      | {str(candidate_dir):<35}")
    SparcLog(f" Starting Candidate    | {iddx:<35}")
    SparcLog(f" Total Candidates      | {nid:<35}")
    SparcLog("------------------------------------------------------------------------")

    return iter, iddx, candidates, candidate_found_is, labelled_files
#===================================================================================================#  

def remove_backup_files(file_ext="bck.*"):
    backup_files = glob.glob(file_ext)
    for file in backup_files:
        os.remove(file)
    # print("REMOVING BACKUP FILES GENERATED FROM PLUMED")

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
        min_distance = check['min_distance']
        max_distance = check['max_distance']
        distance = atoms.get_distance(atom1, atom2)
        if distance < min_distance or distance > max_distance:
            # Get chemical symbols for the atoms
            symbol1 = atoms.get_chemical_symbols()[atom1]
            symbol2 = atoms.get_chemical_symbols()[atom2]
            
            SparcLog("=" * 50)
            SparcLog("  WARNING: DISTANCE CHANGED BEYOND PHYSICAL LIMIT ")
            SparcLog("-" * 50)
            SparcLog(f"  ATOMS: {symbol1} ({atom1}) -- {symbol2} ({atom2})  ")
            SparcLog(f"  MEASURED: {distance:.2f} Å   (MIN. LIMIT: {min_distance:.2f} Å)  ")
            SparcLog(f"  MEASURED: {distance:.2f} Å   (MAX. LIMIT: {max_distance:.2f} Å)  ")
            SparcLog("=" * 50 + "\n")

            return True

    return False

# if __name__ == '__main__':
#     main()
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#   

     
        

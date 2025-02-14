import numpy as np
from ase.io import write
from ase.io.trajectory import TrajectoryWriter
import pickle
#===================================================================================================
"""
    Function is called to log the dynamics. It write the potential energy (Epot), kinetic energy (Ekin),
    total energy (Epot + Ekin), and temperature (Temp) of the system..
"""
def log_md_setup(dyn, atoms, filename='AseMolDyn.log', write_dist=False):
    """
    Log molecular dynamics simulation details including energies and temperature.

    Args:
        dyn: ASE dynamics object
            The molecular dynamics simulation object
        atoms: ase.Atoms
            The atomic system being simulated
        filename: str, optional
            Path to the MD log file (default: 'AseMolDyn.log')
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
    
    # Print current step info to console
    print(f'Steps: {step}, Epot: {epot:.6f}, Ekin: {ekin:.6f}, Temp: {temp:.2f}')
    
    # Write to log file using MDLogger context manager
    with MDLogger(filename) as log:
        if step == 0:
            log.file.write(f"# {'Steps':<6} {'Epot':<10} {'Ekin':<10} {'Total':<10} {'Temp':<6}\n")
        log.file.write(f"{float(step):<8} {epot:<10.6f} {ekin:<10.6f} {total:<10.6f} {temp:<6.2f}\n")
    
    # Optionally log atomic distances using MDLogger
    if write_dist:
        distance = atoms.get_distance(0, 4, mic=True)
        with MDLogger('dist.dat') as dist_log:
            dist_log.file.write(f"Step: {step}, Distance: {distance:.6f}\n")

def wrap_positions(apos):
    """
    Wrap atomic positions within periodic boundary conditions.

    Args:
        apos: ase.Atoms
            The atomic system to wrap

    Returns:
        ase.Atoms: System with wrapped positions
    """
    return apos.wrap()
                
def save_xyz(atoms, trajfile, write_mode):
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
    atoms.wrap()
    trr = TrajectoryWriter(
        filename=trajfile,
        mode=write_mode,
        atoms=atoms,
        properties=properties
    )
    trr.write(atoms)
    
    # Save additional XYZ format
    write('AseTraj.xyz', atoms, append=True)
    
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

#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#        
        
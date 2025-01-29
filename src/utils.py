import numpy as np
from ase.io import write
from ase.io.trajectory import TrajectoryWriter
#===================================================================================================
"""
    Function is called to log the dynamics. It write the potential energy (Epot), kinetic energy (Ekin),
    total energy (Epot + Ekin), and temperature (Temp) of the system..
"""
def log_md_setup(dyn, atoms, filename='AseMolDyn.log', write_dist=False):
    """
        Logs the details of MD simulation.

    Args:
        dyn: dynamics object
        atoms: ASE Atoms object
        filename (str): Name of the file to write the MD log (default: 'AseMolDyn.log').
        write_dist (bool): Flag to indicate whether to log the distance between atoms (default: False).
    """
    # Get potential energy (Epot) from the system; ensure it's a single value if it's an array
    epot = atoms.get_potential_energy()
    if isinstance(epot, (list, np.ndarray)):
        epot = epot[0]  # array on 1 element

    # Get kinetic energy (Ekin)
    ekin = atoms.get_kinetic_energy()
    
    # Get the current simulation step
    step = dyn.get_number_of_steps()
    
    # Example: Get the distance between two atoms (0 and 4)
    distance = atoms.get_distance(0, 4, mic=True)  # Example distance, can be modified
    
    # Get the temperature of the system
    temp = float(atoms.get_temperature())
    
    # Calculate total energy
    total = epot + ekin
    if step == 0:
        print("\n========================================================================")
        print("         Starting... MD Simulation !")
        print("========================================================================")
    print(f'Steps: {step}, Epot: {epot:.6f}, Ekin: {ekin:.6f}, Temp: {temp:.2f}')
    
    # Open the log file in append mode and write 
    with open(filename, 'a') as enr_file:
        if step == 0:
            enr_file.write(f"# {'Steps':<6} {'Epot':<10} {'Ekin':<10} {'Total':<10} {'Temp':<6}\n")
        enr_file.write(f"{float(step):<8} {epot:<10.6f} {ekin:<10.6f} {total:<10.6f} {temp:<6.2f}\n")
        
        if (write_dist):
            with open('dist.dat', 'a') as dist_file:
                dist_file.write(f"Step: {step}, Distance: {distance:.6f}\n")
                
def wrap_positions(apos):
    """
        Wraps the positions of atoms within the periodic boundary conditions (PBC).
        
    Args:
    -----
        apos: The ASE Atoms object.

    Returns:
    --------
        Atoms object, with positions wrapped within the PBC.        
    """
    return apos.wrap()
                
def save_xyz(atoms, trajfile):
    """
        Saves the current atomic configuration (positions, velocities, etc.) to an extened XYZ file.

    Args:
    -----
        atoms: ASE Atoms object.
        trajfile (str): Name of the trajectory file.
    """
    # List of properties to include in the trajectory file
    # names = ['energy', 'forces', 'coordinates', 'velocities', 'stress', 'cell', 'pbc']
    # !<NOTE>! The 'stress' property is not supported in the ASE TrajectoryWriter when PLUMED is called
    
    names = ['energy', 'forces', 'coordinates', 'velocities', 'cell', 'pbc']
    
    # Use the ASE TrajectoryWriter to write the trajectory file
    trr = TrajectoryWriter(filename=f'{trajfile}', mode='a' ,atoms=wrap_positions(atoms), properties=names)
    trr.write(atoms)
    
    # Optionally, save the atomic configuration to a separate XYZ file
    write('AseTraj.xyz', atoms, append=True)
    
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#        
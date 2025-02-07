#!/usr/bin/python3
"""
This code finds the maximum deviation in forces averaged over
multiple models trained on the same dataset with different random weights.
Source: https://doi.org/10.1016/j.cpc.2020.107206i

Configurations along MD trajectories are recorded at a time interval of 100 ps.
Structures with model deviation (max_devi_f) between 0.05 eV/Å and 0.20 eV/Å
are selected for labeling.
"""

import os
import shutil
import pandas as pd
from ase import Atoms
from ase.io import read, write

#===================================================================================================#
def labelling(trajfile, outfile, min_lim, max_lim):
    """
    Select and extract structures for labeling based on force deviations.
    
    Args:
        trajfile: Path to trajectory file
        outfile: Path to model deviation output file
        min_lim: Minimum force deviation threshold (eV/Å)
        max_lim: Maximum force deviation threshold (eV/Å)
    
    Returns:
        tuple: (candidate_found, labelled_files)
            - candidate_found: Boolean indicating if candidates were found
            - labelled_files: List of paths to generated POSCAR files
    """
    # Read trajectory file
    dptraj = read(trajfile, index=':')
    
    # Read model deviation file
    names = ['step', 'max_devi_v', 'min_devi_v', 'avg_devi_v',
             'max_devi_f', 'min_devi_f', 'avg_devi_f', 'dev_e']
    data = pd.read_csv(outfile, sep='\s+', comment='#', names=names)
    
    # Set default deviation limits if not provided
    if (min_lim is None or max_lim is None):
        min_lim = 0.05  # eV/Å
        max_lim = 0.20  # eV/Å

    # Filter structures within deviation range
    candidates = data[(data['max_devi_f'] >= min_lim) & (data['max_devi_f'] <= max_lim)]
    labelled_files = []
    
    if candidates.empty:
        print(f"\nNo candidates found for labelling within range [{min_lim:.2f}, {max_lim:.2f}] eV/Å")
        candidate_found = False
    else:
        print("\n" + "="*90)
        print(f"Found {len(candidates)} candidates for labelling within range [{min_lim:.2f}, {max_lim:.2f}] eV/Å")
        print("=" * 90 + "\n")
        
        # Process each candidate structure
        # [Added new]  
        labelling_dir = 'poscar_files'
        if os.path.exists(labelling_dir):
            shutil.rmtree(labelling_dir)
        os.makedirs(labelling_dir)
        # 
        
        for serial, (_, candidate) in enumerate(candidates.iterrows(), start=1):
            frame_index = int(candidate['step'])
            serial_dir = os.path.join(labelling_dir, f"{serial}")
            os.makedirs(serial_dir, exist_ok=True)
            
            poscar_filename = os.path.join(serial_dir, "POSCAR")
            write(poscar_filename, dptraj[frame_index], format='vasp')
            print(f"Generated POSCAR file for structure {frame_index} in {serial_dir}/")
            labelled_files.append(poscar_filename)
        
        candidate_found = True

    return candidate_found, labelled_files

#===================================================================================================#
#                                     END OF FILE
#===================================================================================================#

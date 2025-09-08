#!/usr/bin/python3
# labelling.py
"""
This code finds the maximum deviation in forces averaged over
multiple models trained on the same dataset with different random weights.
Source: https://doi.org/10.1016/j.cpc.2020.107206i

Configurations along MD trajectories are recorded at a time interval of 100 ps.
Structures with model deviation (max_devi_f) between 0.05 eV/Å and 0.20 eV/Å
are selected for labeling.
"""
################################################################
import os
import pandas as pd
################################################################
# Third party import
from ase import Atoms
from ase.io import read, write
################################################################
# Local import
from sparc.src.utils.logger import SparcLog
#===================================================================================================#
def labelling(trajfile, outfile, min_lim, max_lim, output_dir=None):
    """
    Select and extract structures for labeling based on force deviations.
    
    Args:
        trajfile: Path to trajectory file
        outfile: Path to model deviation output file
        min_lim: Minimum force deviation threshold (eV/Å)
        max_lim: Maximum force deviation threshold (eV/Å)
        output_dir: Path to directory for saving POSCAR files (default: None)
    
    Returns:
        tuple: (candidate_found, labelled_files)
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
        SparcLog(f"\nNo candidates found for labelling within range [{min_lim:.2f}, {max_lim:.2f}] eV/Å")
        candidate_found = False
    else:
        SparcLog("\n" + "="*90)
        SparcLog(f"Found {len(candidates)} candidates for labelling within range [{min_lim:.2f}, {max_lim:.2f}] eV/Å")
        SparcLog("=" * 90 + "\n")
        
        # Use provided output directory or create default
        if output_dir is None:
            output_dir = 'poscar_files'
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each candidate structure
        for serial, (_, candidate) in enumerate(candidates.iterrows(), start=1):
            frame_index = int(candidate['step'])
            serial_dir = os.path.join(output_dir, f"{serial:04d}")
            os.makedirs(serial_dir, exist_ok=True)
            
            poscar_filename = os.path.join(serial_dir, "POSCAR")
            write(poscar_filename, dptraj[frame_index], format='vasp')
            SparcLog(f"Generated POSCAR file for structure {frame_index} in {serial_dir}/")
            labelled_files.append(poscar_filename)
        
        candidate_found = True

    return candidate_found, labelled_files

#===================================================================================================#
#                                     END OF FILE
#===================================================================================================#

#!/usr/bin/python3
"""
This code finds the maximum deviation in forces averaged over
multiple models trained on the same dataset with different random weights.
Source: https://doi.org/10.1016/j.cpc.2020.107206i

Configurations along MD trajectories are recorded at a time interval of 100 ps,
and those with model deviation (max_devi_f) between 0.05 eV/Å and 0.20 eV/Å
are selected and passed to the labeling stage.
"""

import os
import pandas as pd
from ase import Atoms
from ase.io import read, write
import MDAnalysis as mda

#===================================================================================================#
def labelling(trajfile, outfile, min_lim, max_lim, dpmd_dir):
    #
    dptraj = read(trajfile, index=':')
    # Read model deviation file
    names = ['step', 'max_devi_v', 'min_devi_v', 'avg_devi_v', 
                'max_devi_f', 'min_devi_f', 'avg_devi_f', 'dev_e']
    data = pd.read_csv(outfile, sep='\s+', comment='#', names=names)
    #
    # Minimum and maximum limits for model deviation
    if (min_lim is None or max_lim is None):
        min_lim = 0.05  # eV/Å
        max_lim = 0.20  # eV/Å

    # Filter rows where 'max_devi_f' is within the specified range
    candidates = data[(data['max_devi_f'] >= min_lim) & (data['max_devi_f'] <= max_lim)]

    if candidates.empty:
        print('No candidates found for labelling within the specified range.')
    
    else:
        print("\n==============================================================================================")
        print(f'   {len(candidates)}, Candidates found for labelling within the range, [Min: {min_lim}, Max: {max_lim}]!.')
        print("==============================================================================================\n")
        for serial, (_, candidate) in enumerate(candidates.iterrows(), start=1):
            frame_index = int(candidate['step'])
            # print(f'Processing candidate for labelling: frame {frame_index} as serial {serial}')
            # output_dir = 'test_1/poscar_files'
            serial_dir = os.path.join(dpmd_dir, f"{serial}")
            os.makedirs(serial_dir, exist_ok=True)
            poscar_filename = os.path.join(serial_dir, "POSCAR")
            print(f'POSCAR file written for index {frame_index} in directory {serial_dir}.') 
            write(poscar_filename, dptraj[frame_index], format='vasp')
                # labelling(dptraj, frame_index ,serial)           
#===================================================================================================#
#                                     END OF FILE
#===================================================================================================#

#!/usr/bin/python3
# active_learning.py
################################################################
import os
import sys
import subprocess
from pathlib import Path
################################################################
# Third party import
import dpdata
import pandas as pd
from ase.io import read, write
################################################################
# Local Import
from sparc.src.utils.logger import SparcLog
from sparc.src.labelling import labelling
from sparc.src.utils.utils import combine_trajectories
################################################################

def QueryByCommittee(trajfile, model_path, num_models, max_lim, min_lim, dpmd_data_path, iteration=0):
    """
    This code finds the maximum deviation in forces averaged over
    multiple models trained on the same dataset with different random weights.
    
    Args:
        trajfile (str): Path to the ASE trajectory file containing atomic coordinates
        model_path (str): Path to the directory containing DeepMD training folders
        num_models (int): Number of models to consider (minimum 2)
        data_path (str): Path to the directory where DeePMD npy data will be saved
        max_lim (float): Maximum force deviation threshold
        min_lim (float): Minimum force deviation threshold
        iteration (int): Current iteration number
    """
    # Create iteration directory structure
    # iter_paths = create_iter_folders(iteration)
    # print(f"Trajfile name from QbC {trajfile}")
    # For training, combine trajectories from all previous iterations
    # if iteration > 0:
    #     try:
    #         # combined_traj = combine_trajectories(trajfile, Path.cwd(), iteration-1)
    #         combined_traj = combine_trajectories(trajfile, iteration-1)
    #         # Update trajfile to use combined trajectory for model training
    #         trajfile = combined_traj
    #     except ValueError as e:
    #         print(f"Warning: {e}")
    #         print("Proceeding with current trajectory only")
    
    # Load the ASE trajectory and convert it to DeePMD format
    dataset = dpdata.LabeledSystem(trajfile, fmt='ase/traj')
    # dpmd_data_path = iter_paths['dpmd_dir']
    # dataset.to_deepmd_npy(str(dpmd_data_path))
    dataset.to_deepmd_npy(str(dpmd_data_path))
    
    SparcLog("========================================================================")
    SparcLog("{}".format(f"Model Path: {model_path}".center(72)))
    SparcLog("========================================================================")
    
    outfile = f"{str(dpmd_data_path)}/model_dev_{iteration}.out"
    
    # Dynamically find the model files
    model_names = []
    for folder in os.listdir(model_path):
        folder_path = os.path.join(model_path, folder)
        if folder.startswith("training_") and os.path.isdir(folder_path):
            # Extract the model number and construct the model file name
            model_number = folder.split("_")[1]
            model_file = os.path.join(folder_path, f"frozen_model_{model_number}.pb")
            if os.path.exists(model_file):
                model_names.append(model_file)

    # Sort model names to ensure we get the latest ones
    model_names.sort()
    # Take only the last num_models
    model_names = model_names[-num_models:]

    # Validate the number of models
    if len(model_names) < num_models:
        SparcLog("========================================================================")
        SparcLog("!{}!".format(f"Error: Found only {len(model_names)} models, but {num_models} are required".center(70)))
        SparcLog("!{}!".format("Check the model_path!".center(70)))
        SparcLog("========================================================================")
        raise ValueError(
            f"Found only {len(model_names)} models, but {num_models} are required. Check the model_path!"
        )

    SparcLog("========================================================================")
    SparcLog("{}".format("Using the following models:".center(72)))
    SparcLog("========================================================================")
    for model in model_names:
        SparcLog("{}".format(model.center(72)))
    SparcLog("========================================================================")

    # Construct the dp model-devi command
    command = ["dp", "model-devi", "-m"] + model_names + ["-s", str(dpmd_data_path), "-o", str(outfile)]

    # Run the command
    try:
        subprocess.run(command, check=True)
        SparcLog("========================================================================")
        SparcLog("!{}!".format("Model deviation calculation completed successfully!".center(70)))
        SparcLog("{}".format(f"Results saved in: {outfile}".center(72)))
        SparcLog("========================================================================")
    except subprocess.CalledProcessError as e:
        SparcLog("========================================================================")
        SparcLog("!{}!".format("Error in dp model-devi command execution".center(70)))
        SparcLog("!{}!".format(str(e).center(70)))
        SparcLog("========================================================================")

    # Update labelling to use the dft_dir
    candidate_found, labelled_files = labelling(
        trajfile, 
        str(outfile), 
        min_lim, 
        max_lim,
        output_dir=f"{str(dpmd_data_path)}/dft_candidates"  # Pass the dft directory for POSCAR files
    )
      
    # Log iteration info
    with open('learning_state.log', 'a') as f:
        f.write(f"\nIteration {iteration:06d}\n")
        f.write(f"Training data from: {trajfile}\n")
        f.write(f"Model deviation range: [{min_lim:.3f}, {max_lim:.3f}] eV/Å\n")
        f.write(f"Candidates found: {len(labelled_files) if candidate_found else 0}\n")
        f.write("-" * 80 + "\n")
    
    return candidate_found, labelled_files, model_names
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#    

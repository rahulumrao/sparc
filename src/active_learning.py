# active_learning.py
################################################################
import os
import sys
import subprocess
import dpdata
import pandas as pd
import MDAnalysis as mda
from ase import Atoms
from ase.io import read, write
from multiprocessing import Pool
from deepmd_utils.main import main
################################################################
from src.labelling import labelling
################################################################

def QueryByCommittee(trajfile, model_path, num_models, data_path, max_lim, min_lim, iteration=0):
    """
        This code finds the maximum deviation in forces averaged over
        multiple models trained on the same dataset with different random weights.
    
    Args:
    -----
        coord_file (str): Path to the ASE trajectory file containing atomic coordinates.
        model_path (str): Path to the directory containing DeepMD training folders.
        num_models (int): Number of models to consider (minimum 2).
        data_path (str): Path to the directory where DeePMD npy data will be saved.
        outfile (str): Path to the output file where model deviation results will be stored.
    """
    # Load the ASE trajectory and convert it to DeePMD format
    # print('TRAJ FILENAME :', coord_file)
    # sys.exit(0)
    dataset = dpdata.LabeledSystem(trajfile, fmt='ase/traj')
    dataset.to_deepmd_npy(data_path)
    print("\n========================================================================")
    print("{}".format(f"Model Path: {model_path}".center(72)))
    print("========================================================================")
    
    outfile = f'model_dev_{iteration}.out'
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
        print("\n========================================================================")
        print("!{}!".format(f"Error: Found only {len(model_names)} models, but {num_models} are required".center(70)))
        print("!{}!".format("Check the model_path!".center(70)))
        print("========================================================================")
        raise ValueError(
            f"Found only {len(model_names)} models, but {num_models} are required. Check the model_path!"
        )

    print("\n========================================================================")
    print("{}".format("Using the following models:".center(72)))
    print("========================================================================")
    for model in model_names:
        print("{}".format(model.center(72)))
    print("========================================================================")

    # Construct the dp model-devi command
    command = ["dp", "model-devi", "-m"] + model_names + ["-s", data_path, "-o", outfile]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print("\n========================================================================")
        print("!{}!".format("Model deviation calculation completed successfully!".center(70)))
        print("{}".format(f"Results saved in: {outfile}".center(72)))
        print("========================================================================")
    except subprocess.CalledProcessError as e:
        print("\n========================================================================")
        print("!{}!".format("Error in dp model-devi command execution".center(70)))
        print("!{}!".format(str(e).center(70)))
        print("========================================================================")

    # extract the candidates for labelling data
    candidate_found, labelled_files = labelling(trajfile, outfile, min_lim, max_lim)
    
    if candidate_found:
        print("\n========================================================================")
        print("{}".format(f"Processing {len(labelled_files)} Candidate Structures".center(72)))
        print("========================================================================")
        
    from tqdm import tqdm
    labelled_files = list(tqdm(labelled_files, 
                              desc="Processing Candidates", 
                              unit="structure",
                              ncols=80,
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"))
    
    return candidate_found, labelled_files, model_names
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#    
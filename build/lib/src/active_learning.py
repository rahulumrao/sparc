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

def QueryByCommittee(trajfile, model_path, num_models, data_path, iteration=0):
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
    print("Model Path : ",model_path)
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
        raise ValueError(
            f"Found only {len(model_names)} models, but {num_models} are required. Check the model_path!"
        )

    print(f"Using the following models :\n{model_names}")

    # Construct the dp model-devi command
    command = ["dp", "model-devi", "-m"] + model_names + ["-s", data_path, "-o", outfile]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Model deviation calculation completed successfully! Results saved in: {outfile}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the dp model-devi command: {e}")
#----------------------------------------------------------------------------------------------------
    # extract the candidates for labelling data
    candidate_found, labelled_files = labelling(trajfile, outfile, min_lim=None, max_lim=None)
    return candidate_found, labelled_files, model_names
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#    
#! deepmd_training.py
import os
import logging
import subprocess
import random
import datetime
import json
import shutil
import sys
#!==============================================================!
"""
    This module contains the function for DeepMD model training.
    It sets up the training directory, runs the training, freezes the model, 
    and compresses the model using DeepMD tools.
    
    Args:
    -----
        training_file: str
            DeepPotential training file name.
        
        training_dir: str
            Path to the directory where the training will be done.
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace all occurrences of SEED with the generated random number
def update_seed(data):
    # Generate random seed once at the start to use the same value for all 'seed' fields
    random_number = random.randint(100000, 999999)
    
    def _update_recursively(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    _update_recursively(value)
                elif key == 'seed':
                    data[key] = random_number
        elif isinstance(data, list):
            for item in data:
                _update_recursively(item)
    
    _update_recursively(data)
    return data

def deepmd_training(training_dir, num_models, input_file='input.json'):
    
    # Save the current working directory to restor later
    original_dir = os.getcwd()
    print(f"Original directory: {original_dir}")
    # Define the training directory (e.g., '01.train')
    base_dir = training_dir
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
     
    # Check if the number of models is at least 2
    if num_models < 2:
        raise ValueError("The number of models must be at least 2.")
    elif num_models > 4:
        print("*" * 100,
            "\n !! WARNING: More than 4 models will not give additional advantage. Proceed with caution! !!\n",
            "*" * 100)
    print("\n========================================================================")
    print(f"          DEEPMD WILL TRAIN {num_models} MODELS !")
    print("========================================================================")
    # Loop through the required number of models
    for i in range(1, num_models + 1):
        #
        # Find the next available folder number inside 'training_sir'
        training_folders = [f for f in os.listdir(base_dir) if f.startswith('training_')]
        folder_numbers = [int(f.split('_')[1]) for f in training_folders]

        # If no folders exist, start with training_1
        next_folder_number = max(folder_numbers, default=0) + 1
        dir_name = os.path.join(base_dir, f"training_{next_folder_number}")
        print("\n========================================================================")
        print(f"  RUNNING TRAINING IN FOLDER ({dir_name}) !")
        print("========================================================================\n")
        #
        try:
            # Create the training directory if it doesn't already exist
            os.makedirs(dir_name, exist_ok=True)
            
            # Path to the original input file
            input_path = os.path.join(original_dir, input_file)
            # training_input = os.path.join(dir_name, input_file)

            # Check if the input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Copy the input file to the training directory
            # shutil.copy(input_path, training_input)
            os.chdir(dir_name)
            logging.info(f'Training directory path: {os.getcwd()}')

            # Load the JSON configuration file
            with open(input_path, 'r') as f:
                config_data = json.load(f)
                update_seed(config_data)
            
            # Write the updated data back to the file
            with open(input_file, 'w') as f:
                json.dump(config_data, f, indent=4) 
            
            # Set arguments to run the DeepMD training with the provided input file
            # sys.argv = ['dp', 'train', input_path]
            # main()                                                      # Run DeepMD training
            subprocess.run(['dp', 'train', input_file], check=True)
            logging.info("Training completed successfully !")
            
            # Freeze the trained model (default: "frozen_model.pb")
            # sys.argv = ['dp', 'freeze']
            # main()                                                      # Freeze model
            frozen_model_name = f"frozen_model_{next_folder_number}.pb"
            subprocess.run(['dp', 'freeze', '-o', frozen_model_name], check=True)
            logging.info("Model frozen successfully !")
            
            # Compress the frozen model
            # sys.argv = ['dp', 'compress', '-t', {input_path}]
            # main()                                                      # Compress model
            compressed_model = f"frozen_model_compressed_{next_folder_number}.pb"
            subprocess.run(['dp', 'compress', '-t', input_file, '-i', frozen_model_name, '-o', compressed_model], check=True)
            logging.info("Model compressed successfully !")

        except Exception as e:
            # Log any errors that occur during training, freezing, or compression
            logging.error(f"An error occurred during training: {str(e)}")
        finally:
            # Change back to the original working directory
            os.chdir(original_dir)
        
# #===================================================================================================#
# #                                     END OF FILE 
# #===================================================================================================#        
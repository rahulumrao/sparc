#! deepmd_training.py
import os
import logging
import subprocess
import random
import json
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

# Replace all occurreny somces of SEED with the generated random number
def update_json(data, datadir, atom_types):
    # Generate random seed once at the start to use the same value for all 'seed' fields
    random_number = random.randint(100000, 999999)
    
    def _update_recursively(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'seed':
                    data[key] = random_number
                elif key == 'type_map':
                    data[key] = atom_types
                elif key == 'training_data' and isinstance(value, dict):
                    value['systems'] = [os.path.join(datadir, 'training_data')]
                elif key == 'validation_data' and isinstance(value, dict):
                    value['systems'] = [os.path.join(datadir, 'validation_data')]
                elif isinstance(value, (dict, list)):
                    _update_recursively(value)
        elif isinstance(data, list):
            for item in data:
                _update_recursively(item)
    
    _update_recursively(data)
    return data
#---------------------------------------------------------------------------------------------------#
# DeePMD Training Function 
#---------------------------------------------------------------------------------------------------#
def deepmd_training(active_learning: bool, datadir: str, atom_types:list, training_dir: str, num_models: int, input_file: str = 'input.json'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers first
    logger.handlers.clear()
    
    # Add file handler
    fh = logging.FileHandler('deepmd_training.log')
    fh.setLevel(logging.INFO)
    
    # Add console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Add formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # Save the current working directory to restore later
    original_dir = os.getcwd()
    print(f"Original directory: {original_dir}")
    
    # Create the base training directory if it doesn't exist
    os.makedirs(training_dir, exist_ok=True)
    
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
        # Define the training folder name
        folder_name = f"training_{i}"
        dir_name = os.path.join(training_dir, folder_name)
        
        print("\n========================================================================")
        print(f"  RUNNING TRAINING IN FOLDER ({dir_name}) !")
        print("========================================================================\n")
        
        # Check if the training folder already exists
        if os.path.exists(dir_name):
            print(f"Training folder '{dir_name}' already exists. Using this folder for training.")
        else:
            print(f"Creating new training folder: {dir_name}")
            os.makedirs(dir_name, exist_ok=True)
        
        try:
            # Change to the training directory
            os.chdir(dir_name)
            logger.info(f'Training directory path: {os.getcwd()}')

            # Load the JSON configuration file
            input_path = os.path.join(original_dir, input_file)
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Load the JSON configuration file
            with open(input_path, 'r') as f:
                config_data = json.load(f)
                update_json(config_data, datadir, atom_types)

                
            # Write the updated data back to the file
            with open(input_file, 'w') as f:
                json.dump(config_data, f, indent=4) 
            
            # Check for the existence of the checkpoint file
            if os.path.exists('checkpoint'):
                print("\n" + "*"*80)
                print("[INFO] Checkpoint file found | Resuming training with the current model state")
                print("*"*80 + "\n")

                # Run DeepMD training with the checkpoint
                subprocess.run(['dp', 'train', input_file, '-r', 'model.ckpt'], check=True)
            else:
                # Run DeepMD training without checkpoint
                subprocess.run(['dp', 'train', input_file], check=True)
            
            logger.info("Training completed successfully !")
            
            # Freeze the trained model
            frozen_model_name = f"frozen_model_{i}.pb"
            subprocess.run(['dp', 'freeze', '-o', frozen_model_name], check=True)
            logger.info("Model frozen successfully !")
            
            # Compress the frozen model
            compressed_model = f"frozen_model_compressed_{i}.pb"
            subprocess.run(['dp', 'compress', '-t', input_file, '-i', frozen_model_name, '-o', compressed_model], check=True)
            logger.info("Model compressed successfully !")

        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
        finally:
            # Change back to the original working directory
            os.chdir(original_dir)
    return frozen_model_name
# #===================================================================================================#
# #                                     END OF FILE 
# #===================================================================================================#        
#!/usr/bin/python3
#! deepmd.py
"""
DeepMD module for SPARC package.

This module contains functions for:
1. Setting up DeepPotential calculators for ASE atoms objects
2. Training DeepMD models with various configurations
3. Model freezing and compression

The module provides a unified interface for all DeepMD-related operations.
"""
################################################################
# standard import
import os
import logging
import subprocess
import random
import json
################################################################
# third party import
from ase import Atoms
from deepmd.calculator import DP
################################################################
# Local import
from sparc.src.utils.logger import SparcLog
################################################################
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#===================================================================================================#
# DeepMD Setup Functions
#===================================================================================================#

def setup_DeepPotential(atoms, model_path, model_name="frozen_model.pb"):
    """
    Setup a DeepPotential calculator for an ASE atoms object.

    Args:
        atoms: ase.Atoms
            The atomic structure to assign the DeepPotential model to
        model_path: str
            Path to the directory containing DeepPotential model
        model_name: str, optional
            Name of the DeepPotential model file (default: "frozen_model.pb")

    Returns:
        tuple: (dp_system, dp_calc)
            - dp_system: ASE atoms object with DeepPotential calculator attached
            - dp_calc: DeepPotential calculator object

    Raises:
        Exception: If an issue occurs while setting up or testing the DeepPotential model
    """
    # Construct full path to model file
    dp_model = os.path.join(model_path, model_name)
    dp_calc = DP(model=dp_model)
    
    # Create atoms object with DeepPotential calculator
    dp_system = Atoms(atoms, calculator=dp_calc)
    
    try:
        # Test calculator by computing energy and forces
        potential_energy = dp_system.get_potential_energy()
        forces = dp_system.get_forces()
        
        if potential_energy is not None and forces is not None:
            SparcLog("\n" + "=" * 72)
            SparcLog(f"DeepPotential model successfully loaded and tested: \n {dp_model}")
            SparcLog("=" * 72)
        else:
            SparcLog("\n" + "=" * 72)
            SparcLog("Error: Failed to compute energy and forces with DeepPotential model")
            SparcLog("=" * 72)
            
    except Exception as e:
        SparcLog("\n" + "=" * 72)
        SparcLog("Error: Failed to setup DeepPotential model")
        SparcLog(f"Details: {str(e)}")
        SparcLog("=" * 72)
    
    return dp_system, dp_calc
#===================================================================================================#
# Utility Functions
#===================================================================================================#

import re
import subprocess

def evaluate_model_accuracy(model_path, test_data_path):
    """
    Evaluate the accuracy of a trained DeepMD model against reference data.
    
    Args:
        model_path (str): Path to the DeepMD frozen model.
        test_data_path (str): Path to test data in DeepMD npy format.
    """
    try:
        # Run DeepMD model evaluation
        result = subprocess.run(
            ['dp', 'test', '-m', model_path, '-s', test_data_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True  # ensures the output is decoded to str
        )
        output_lines = result.stdout.splitlines()
        # print("Full DeepMD Test Output:\n", output_lines)
        SparcLog("\n" + "=" * 72)
        SparcLog("DeepMD Model Evaluation Results")
        SparcLog("-" * 72)
        SparcLog(result.stdout.strip())
   
    except subprocess.CalledProcessError as e:
        SparcLog("\n" + "=" * 72)
        SparcLog("Error in model evaluation:")
        SparcLog(f"Details: {str(e)}")
        SparcLog("=" * 72)
        return None

#===================================================================================================#
# DeepMD Training Functions
#===================================================================================================#

def update_json(data, datadir, atom_types):
    """
    Update the DeepMD input JSON configuration with random seeds and proper paths.
    
    Args:
        data: dict
            The loaded JSON configuration data
        datadir: str
            Path to the directory containing training data
        atom_types: list
            List of atomic species in the system
            
    Returns:
        dict: Updated JSON configuration
    """
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
#--------------------------------------------------------------------------------------------------------------------------------------------
def deepmd_training(active_learning: bool, datadir: str, atom_types: list, training_dir: str, num_models: int, input_file: str = 'input.json'):
    """
    Train DeepMD models for molecular potential energy surface representation.
    
    This function handles the complete training workflow:
    1. Setting up training directories
    2. Configuring model parameters with random seeds
    3. Training multiple models with different initializations
    4. Freezing and compressing the trained models
    
    Args:
        active_learning: bool
            Whether this training is part of an active learning cycle
        datadir: str
            Path to directory containing training and validation data
        atom_types: list
            List of atomic species in the system
        training_dir: str
            Path to the directory where models will be trained
        num_models: int
            Number of models to train (minimum: 2)
        input_file: str, optional
            Path to DeepMD input JSON file (default: 'input.json')
            
    Returns:
        str: Name of the frozen model file
    """
    # Clear any existing logger handlers first
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
    SparcLog(f"Original directory: {original_dir}")
    
    # Create the base training directory if it doesn't exist
    os.makedirs(training_dir, exist_ok=True)
    
    # Check if the number of models is at least 2
    if num_models < 2:
        raise ValueError("The number of models must be at least 2.")
    elif num_models > 4:
        SparcLog("*" * 100,
              "\n !! WARNING: More than 4 models will not give additional advantage. Proceed with caution! !!\n",
              "*" * 100)
    
    SparcLog("\n========================================================================")
    SparcLog(f"          DEEPMD WILL TRAIN {num_models} MODELS !")
    SparcLog("========================================================================")
    
    # Loop through the required number of models
    for i in range(1, num_models + 1):
        # Define the training folder name
        folder_name = f"training_{i}"
        dir_name = os.path.join(training_dir, folder_name)
        
        SparcLog("\n========================================================================")
        SparcLog(f"  RUNNING TRAINING IN FOLDER ({dir_name}) !")
        SparcLog("========================================================================\n")
        
        # Check if the training folder already exists
        if os.path.exists(dir_name):
            SparcLog(f"Training folder '{dir_name}' already exists. Using this folder for training.")
        else:
            SparcLog(f"Creating new training folder: {dir_name}")
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
                SparcLog("\n" + "*"*80)
                SparcLog("[INFO] Checkpoint file found | Resuming training with the current model state")
                SparcLog("*"*80 + "\n")

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
    
        # Evaluate Model Accuracy
        evaluate_model_accuracy(f"{dir_name}/{frozen_model_name}", f"{datadir}/validation_data")
    
    return frozen_model_name

#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================# 
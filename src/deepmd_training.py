import os
import sys
import logging
from deepmd_utils.main import main
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

def deepmd_training(training_dir, input_file='input.json'):
    
    # Save the current working directory to restor later
    original_dir = os.getcwd()
    
    # Define the training directory (e.g., '01.train')
    dir_name = training_dir

    try:
        # Create the training directory if it doesn't already exist
        os.makedirs(dir_name, exist_ok=True)
        os.chdir(dir_name)
        logging.info(f'Training directory path: {os.getcwd()}')
        
        # Path to the input file relative to the training directory
        input_path = os.path.join('../..', input_file)
        
        # Check if the input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Set arguments to run the DeepMD training with the provided input file
        sys.argv = ['dp', 'train', input_path]
        main()                                                      # Run DeepMD training
        logging.info("Training completed successfully")
        
        # Freeze the trained model (default: "frozen_model.pb")
        sys.argv = ['dp', 'freeze']
        main()                                                      # Freeze model
        logging.info("Model frozen successfully")
        
        # Compress the frozen model
        sys.argv = ['dp', 'compress', '-t', {input_path}]
        main()                                                      # Compress model
        logging.info("Model compressed successfully")
    
    except Exception as e:
        # Log any errors that occur during training, freezing, or compression
        logging.error(f"An error occurred during training: {str(e)}")
    finally:
        # Change back to the original working directory
        os.chdir(original_dir)
        
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#        
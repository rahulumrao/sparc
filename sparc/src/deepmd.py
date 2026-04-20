#!/usr/bin/python3
# deepmd.py

"""
DeepMD module for SPARC package with DeePMD-kit v2/v3 support.

This module contains functions for:
1. Setting up DeepPotential calculators for ASE atoms objects
2. Training DeepMD models with TensorFlow or PyTorch backends
3. Model freezing and compression
4. Support for DeePMD-GNN (MACE, NequIP models)

Supports DeePMD-kit v2 and v3 with automatic backend detection.
"""

import os, sys
import logging
import subprocess
import random
import json
from pathlib import Path
from typing import Optional, List

################################################################
# Third party imports
from ase import Atoms

################################################################
# Local imports
from sparc.src.utils.logger import SparcLog

################################################################
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

################################################################
# Version Detection
################################################################

# Module-level cache for version detection
_cached_version = None

def get_version():
    """
    Detect installed DeePMD-kit version and backend.

    For v3, backend is determined by testing which one is functional.
    Defaults to TensorFlow if detection fails. Results are cached.

    Returns
    -------
    tuple
        (major_version, backend)
        e.g., (3, 'pytorch') or (2, 'tensorflow')
    """
    global _cached_version
    if _cached_version is not None:
        return _cached_version

    try:
        # Get version
        result = subprocess.run(
            ['dp', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_str = result.stdout.strip()
        
        # Parse version number
        if 'v3' in version_str or version_str.startswith('3'):
            major_version = 3
        else:
            major_version = 2
        
        # For v3, detect backend by testing which one works
        if major_version == 3:
            backend = get_backend()
        else:
            backend = 'tensorflow'
        
        SparcLog(f"Detected DeePMD-kit v{major_version} with {backend.upper()} backend")
        _cached_version = (major_version, backend)
        return _cached_version

    except Exception as e:
        SparcLog(f"Warning: Could not detect DeePMD version: {e}")
        SparcLog("Assuming DeePMD-kit v2 with TensorFlow backend")
        _cached_version = (2, 'tensorflow')
        return _cached_version


def get_backend():
    """
    Detect which backend is functional in DeePMD-kit v3.
    
    Both deepmd.pt and deepmd.tf modules may exist in the environment,
    but only one will actually work. We test which one can be imported.
    Defaults to TensorFlow if detection fails.
    
    Returns
    -------
    str
        'pytorch' or 'tensorflow'
    """
    # Try PyTorch backend first
    try:
        from deepmd.pt.model.model import get_model as get_pt_model
        SparcLog("PyTorch backend is functional")
        return 'pytorch'
    except Exception:
        pass
    
    # Try TensorFlow backend
    try:
        from deepmd.tf.model.model import get_model as get_tf_model
        SparcLog("TensorFlow backend is functional")
        return 'tensorflow'
    except Exception:
        pass
    
    # Default to TensorFlow for v3
    SparcLog("Warning: Could not detect functional backend, defaulting to TensorFlow")
    return 'tensorflow'


################################################################
# DeepMD Setup Functions
################################################################
def setup_DeepPotential(atoms, model_path: str, model_name: Optional[str] = None):
    """
    Setup a DeepPotential calculator for an ASE atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to assign the DeepPotential model to
    model_path : str
        Path to the directory containing DeepPotential model
    model_name : str, optional
        Name of the DeepPotential model file. If None, auto-detects based on version

    Returns
    -------
    tuple
        (dp_system, dp_calc) - ASE atoms object with calculator and the calculator object

    Raises
    ------
    FileNotFoundError
        If model file is not found
    Exception
        If model setup or testing fails
    """
    # Auto-detect version and backend
    version, backend = get_version()

    # Auto-detect model name based on backend
    if model_name is None:
        # Try both extensions and pick whichever exists
        pth_path = os.path.join(model_path, "frozen_model_1.pth")
        pb_path = os.path.join(model_path, "frozen_model_1.pb")

        if os.path.exists(pth_path):
            dp_model = pth_path
        elif os.path.exists(pb_path):
            dp_model = pb_path
        else:
            # Fall back to backend-based guess for clearer error message
            ext = ".pth" if backend == 'pytorch' else ".pb"
            dp_model = os.path.join(model_path, f"frozen_model_1{ext}")
    else:
        # model name is provided - check if it's a full path
        if os.path.isabs(model_name) or os.path.exists(model_name):
            dp_model = model_name
        else:
            dp_model = os.path.join(model_path, model_name)
    
    if not Path(dp_model).exists():
        raise FileNotFoundError(f"DeepPotential model not found: {dp_model}")

    # Validate model format matches installed backend
    from sparc.src.utils.utils import check_backend_mismatch
    check_backend_mismatch(dp_model, backend)

    try:
        # Import calculator (compatible with both v2 and v3)
        from deepmd.calculator import DP

        dp_calc = DP(model=dp_model)

        # Create atoms object with DeepPotential calculator
        dp_system = Atoms(atoms, calculator=dp_calc)

        # Test calculator
        potential_energy = dp_system.get_potential_energy()
        forces = dp_system.get_forces()
        
        if potential_energy is not None and forces is not None:
            SparcLog("-" * 80)
            SparcLog(f"DeepPotential model successfully loaded and tested:")
            SparcLog(f"  Model: {dp_model}")
            SparcLog(f"  Backend: {backend.upper()}")
            SparcLog("-" * 80)
        else:
            raise ValueError("Failed to compute energy and forces")
        
        return dp_system, dp_calc
    
    except Exception as e:
        SparcLog("-" * 80)
        SparcLog("Error: Failed to setup DeepPotential model")
        SparcLog(f"Details: {str(e)}")
        SparcLog("-" * 80)
        raise

################################################################
# Model Evaluation
################################################################

def evaluate_model_accuracy(model_path: str, test_data_path: str, version: int, backend: str):
    """
    Evaluate the accuracy of a trained DeepMD model against reference data.
    
    Parameters
    ----------
    model_path : str
        Path to the DeepMD frozen model
    test_data_path : str
        Path to test data in DeepMD npy format
    version : int
        DeePMD-kit major version (2 or 3)
    backend : str
        Backend ('pytorch' or 'tensorflow')
    """
    if not Path(model_path).exists():
        SparcLog(f"Warning: Model file not found: {model_path}")
        return
    
    if not Path(test_data_path).exists():
        SparcLog(f"Warning: Test data not found: {test_data_path}")
        return
    
    try:
        # For v3, test command doesn't need backend flag
        cmd = ['dp', 'test', '-m', model_path, '-s', test_data_path]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True
        )
        
        SparcLog("=" * 80)
        SparcLog("DeepMD Model Evaluation Results")
        SparcLog("-" * 80)
        SparcLog(result.stdout.strip())
        SparcLog("=" * 80)
        
    except subprocess.CalledProcessError as e:
        SparcLog("=" * 80)
        SparcLog("Error in model evaluation:")
        SparcLog(f"Details: {str(e)}")
        SparcLog("=" * 80)


################################################################
# JSON Configuration Update
################################################################

def update_json(data: dict, datadir: str, atom_types: List[str]):
    """
    Update the DeepMD input JSON configuration with random seeds and proper paths.
    
    Parameters
    ----------
    data : dict
        The loaded JSON configuration data
    datadir : str
        Path to the directory containing training data
    atom_types : list
        List of atomic species in the system
    
    Returns
    -------
    dict
        Updated JSON configuration
    """
    # Generate random seed
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


################################################################
# DeepMD Training
################################################################

def deepmd_training(
    active_learning: bool,
    datadir: str,
    atom_types: List[str],
    training_dir: str,
    num_models: int,
    input_file: str = 'input.json',
    compress_models: bool = False
):
    """
    Train DeepMD models for molecular potential energy surface representation.
    
    Supports both DeePMD-kit v2 (TensorFlow) and v3 (PyTorch/TensorFlow).
    Backend is automatically detected from the environment.
    
    Parameters
    ----------
    active_learning : bool
        Whether this training is part of an active learning cycle
    datadir : str
        Path to directory containing training and validation data
    atom_types : list
        List of atomic species in the system
    training_dir : str
        Path to the directory where models will be trained
    num_models : int
        Number of models to train (minimum: 2)
    input_file : str, optional
        Path to DeepMD input JSON file (default: 'input.json')
    compress_models : bool, optional
        Whether to compress trained models (default: True)
    
    Returns
    -------
    str
        Name of the frozen model file
    
    Raises
    ------
    ValueError
        If num_models < 2
    FileNotFoundError
        If input file not found
    """
    # Clear logger handlers
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
    
    # Save current directory
    original_dir = os.getcwd()
    SparcLog(f"Original directory: {original_dir}")
    
    # Create training directory
    os.makedirs(training_dir, exist_ok=True)
    
    # Validate num_models
    if num_models < 2:
        raise ValueError("The number of models must be at least 2")
    elif num_models > 4:
        SparcLog("*" * 80)
        SparcLog(" WARNING: More than 4 models may not provide additional advantage!", level='WARNING')
        SparcLog("*" * 80)
    
    # Detect DeePMD version and backend
    version, backend = get_version()
    
    # Determine model file extension
    if backend == 'pytorch':
        model_ext = '.pth'
    else:
        model_ext = '.pb'
    
    SparcLog("=" * 80)
    SparcLog(f" DeePMD-kit v{version} Training")
    SparcLog(f" Backend: {backend.upper()}")
    SparcLog("=" * 80)
    
    # Loop through models
    for i in range(1, num_models + 1):
        folder_name = f"training_{i}"
        dir_name = os.path.join(training_dir, folder_name)
        
        SparcLog("=" * 80)
        SparcLog(f" Training Model {i}/{num_models}")
        SparcLog(f" Directory: {dir_name}")
        SparcLog("=" * 80)
        
        # Create training folder
        os.makedirs(dir_name, exist_ok=True)
        
        try:
            logger.info(f'Training directory: {dir_name}')

            # Load and update JSON configuration
            input_path = os.path.join(original_dir, input_file)
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

            with open(input_path, 'r') as f:
                config_data = json.load(f)

            update_json(config_data, datadir, atom_types)

            # Write updated config into training directory
            config_output_path = os.path.join(dir_name, input_file)
            with open(config_output_path, 'w') as f:
                json.dump(config_data, f, indent=4)

            # Check for checkpoint
            has_checkpoint = (os.path.exists(os.path.join(dir_name, 'checkpoint')) or
                            os.path.exists(os.path.join(dir_name, 'model.ckpt')) or
                            os.path.exists(os.path.join(dir_name, 'model.ckpt.pt')))
            
            if has_checkpoint:
                SparcLog("*" * 80)
                SparcLog(" Checkpoint found - Resuming training".center(80))
                SparcLog("*" * 80)
            
            # Build training command
            if version >= 3:
                # v3 requires backend flag
                if backend == 'pytorch':
                    train_cmd = ['dp', '--pt', 'train', input_file]
                else:
                    train_cmd = ['dp', '--tf', 'train', input_file]
            else:
                # v2 doesn't use backend flags
                train_cmd = ['dp', 'train', input_file]
            
            # Add restart flag if checkpoint exists
            if has_checkpoint:
                if version >= 3:
                    train_cmd.extend(['--restart', 'model.ckpt.pt'])
                else:
                    train_cmd.extend(['--restart', 'model.ckpt'])
            
            # Run training
            SparcLog(f" Training Model {i}/{num_models}")
            SparcLog(f"{'Directory':<30} {dir_name}")
            SparcLog(f"Running command: {' '.join(train_cmd)}")
            subprocess.run(train_cmd, check=True, cwd=dir_name)
            logger.info("Training completed successfully")

            # Freeze the model with numbered name
            frozen_model_name = f"frozen_model_{i}{model_ext}"

            if version >= 3:
                if backend == 'pytorch':
                    freeze_cmd = ['dp', '--pt', 'freeze', '-o', frozen_model_name]
                else:
                    freeze_cmd = ['dp', '--tf', 'freeze', '-o', frozen_model_name]
            else:
                freeze_cmd = ['dp', 'freeze', '-o', frozen_model_name]

            SparcLog(f"Freezing model: {' '.join(freeze_cmd)}")
            subprocess.run(freeze_cmd, check=True, cwd=dir_name)
            logger.info("Model frozen successfully")

            # Compress the model if requested
            if compress_models:
                compressed_model = f"frozen_model_compressed_{i}{model_ext}"

                if version >= 3:
                    if backend == 'pytorch':
                        compress_cmd = ['dp', '--pt', 'compress',
                                    '-i', frozen_model_name,
                                    '-o', compressed_model]
                    else:
                        compress_cmd = ['dp', '--tf', 'compress',
                                    '-i', frozen_model_name,
                                    '-o', compressed_model]
                else:
                    compress_cmd = ['dp', 'compress',
                                '-t', input_file,
                                '-i', frozen_model_name,
                                '-o', compressed_model]

                SparcLog(f"Compressing model: {' '.join(compress_cmd)}")
                subprocess.run(compress_cmd, check=True, cwd=dir_name)
                logger.info("Model compressed successfully")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
        # Evaluate model accuracy
        model_path = os.path.join(dir_name, frozen_model_name)
        test_data = os.path.join(datadir, 'validation_data')
        evaluate_model_accuracy(model_path, test_data, version, backend)
    
    return frozen_model_name


################################################################
# END OF FILE
################################################################

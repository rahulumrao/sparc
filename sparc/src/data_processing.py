#!/usr/bin/python3
# data_processing.py

"""
Data processing module for converting ASE trajectories to DeepMD format.
"""

from pathlib import Path
from typing import Optional
import dpdata
import numpy as np

################################################################
# Local Import
from sparc.src.utils.logger import SparcLog
from sparc.src.utils.utils import combine_trajectories

################################################################

def get_data(
    ase_traj: str = 'AseMD.traj',
    dir_name: str = 'Dataset',
    skip_min: int = 0,
    skip_max: Optional[int] = None,
    seed: int = 42,
    train_ratio: float = 0.8
) -> None:
    """
    Process an ASE trajectory file and split the data into training and validation datasets.

    Parameters
    ----------
    ase_traj : str
        ASE trajectory file name (default: 'AseMD.traj')
    dir_name : str
        Path to the directory for saving training and validation datasets
    skip_min : int
        Skip the first n frames
    skip_max : int, optional
        Skip the last n frames (default: None)
    seed : int
        Random seed for reproducible train/validation split (default: 42)
    train_ratio : float
        Fraction of frames used for training; remainder goes to validation
        (default: 0.8, i.e. 80 % training / 20 % validation)

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If trajectory file does not exist
    ValueError
        If trajectory is empty or invalid
    """
    # Sanity check: verify trajectory file exists
    if not Path(ase_traj).exists():
        raise FileNotFoundError(f"Trajectory file not found: {ase_traj}")
    
    # SparcLog(f"Processing trajectory: {ase_traj}")
    
    # Load trajectory
    dt = dpdata.LabeledSystem(ase_traj, 'ase/traj')
    
    # Sanity check: verify trajectory has frames
    if dt.get_nframes() == 0:
        raise ValueError(f"Trajectory file is empty: {ase_traj}")
    
    SparcLog(f"Loaded {dt.get_nframes()} frames")
    
    # Slice data
    data = dt[skip_min:skip_max]
    n_frames = data.get_nframes()
    
    # Sanity check: verify frames remain after slicing
    if n_frames == 0:
        raise ValueError(
            f"No frames remaining after skipping (skip_min={skip_min}, skip_max={skip_max})"
        )
    
    # SparcLog(f"Using {n_frames} frames after skipping")
    
    SparcLog("")
    SparcLog("DATA PROCESSING")
    SparcLog("-----------------")
    SparcLog(f"{'Trajectory file':<30} {ase_traj}")
    SparcLog(f"{'Total frames':<30} {dt.get_nframes()}")
    SparcLog(f"{'Frames used':<30} {n_frames}")

    # Split train-validation
    n_train = int(n_frames * train_ratio)
    n_val = n_frames - n_train
    
    # Randomly select validation indices
    rng = np.random.default_rng(seed=seed)
    index_validation = rng.choice(n_frames, size=n_val, replace=False)
    
    # Remaining are training
    index_training = list(set(range(n_frames)) - set(index_validation))
    
    # Create subsystems
    data_training = data.sub_system(index_training)
    data_validation = data.sub_system(index_validation)
    
    # Create directories
    train_dir = Path(dir_name) / 'training_data'
    val_dir = Path(dir_name) / 'validation_data'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    data_training.to_deepmd_npy(str(train_dir))
    data_validation.to_deepmd_npy(str(val_dir))
    
    # SparcLog(f"Training data: {len(data_training)} frames saved to {train_dir}")
    # SparcLog(f"Validation data: {len(data_validation)} frames saved to {val_dir}")
    SparcLog(f"{'Training frames':<30} {len(data_training)} ({train_ratio*100:.0f}%)")
    SparcLog(f"{'Validation frames':<30} {len(data_validation)} ({(1-train_ratio)*100:.0f}%)")
    SparcLog("")
###########################################################################################
#                                       END OF FILE
###########################################################################################

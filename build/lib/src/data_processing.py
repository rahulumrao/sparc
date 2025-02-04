import dpdata
import os
import numpy as np
#===================================================================================================#
def get_data(ase_traj='md.traj', dir_name='00.data', skip_min=0, skip_max=None):
    """
        Process a ASE trajectory file and split the data into training and validation datasets.
        The training data consists of 80% of the frames, and the validation data consists of 20%. 
        The data is then saved in the specified directory in the DeepMD .npy format.
    
    Args:
    -----
        ase_traj (str): ASE trajectory file name (default: md.traj).
        dir_name (str): Path to the directory for saving training and validation datasets.
        skip_min (int): Skip the first n frames.
        skip_max (int): Skip the last n frames.
    """
    # Load the trajectory (ASE trajectory format) using dpdata.
    dt = dpdata.LabeledSystem(f'{ase_traj}', 'ase/traj')
    
    # Slice the data to get the frames between skip_min and skip_max
    data = dt[skip_min:skip_max]
    
    # Get the number of frames
    n_frames = data.get_nframes()
    print(f"# the dataset contains %d frames" % n_frames)
    
    # Split data into training and validation sets (80% training, 20% validation)
    trr = int(n_frames*0.8)  # training 80%
    val = n_frames - trr      # validation 20%
    
    # Randomly choose index for validation_data
    index_validation = np.random.choice(n_frames,size=val,replace=False)
    
    # The remaining frames are used for training_data
    index_training = list(set(range(n_frames))-set(index_validation))
    
    # Create subsystems for training and validation
    data_training = data.sub_system(index_training)
    data_validation = data.sub_system(index_validation)
    
    # Save data in the specified directory as .npy files
    data_training.to_deepmd_npy(f'{dir_name}/training_data')               
    data_validation.to_deepmd_npy(f'{dir_name}/validation_data')           
    print(f'# the {dir_name}/training data contains %d frames' % len(data_training)) 
    print(f'# the {dir_name}/validation data contains %d frames' % len(data_validation), '\n')

#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#    
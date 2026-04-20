#!/usr/bin/python3
# finetune.py

"""
Fine-tuning module for universal ML potentials.

Supports:

1. DeePMD fine-tuning via DeePMD-kit v3 (dp --pt finetune)
   Works with DPA-1, DPA-2, DPA-3 and other DeePMD models
2. MACE fine-tuning via mace_run_train --foundation_model

Both produce frozen models compatible with the SPARC active learning workflow.
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

################################################################
# Local imports
from sparc.src.utils.logger import SparcLog
from sparc.src.deepmd import get_version, update_json

################################################################
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

################################################################
# DeePMD Fine-tuning (DPA-1, DPA-2, DPA-3, etc.)
################################################################

def deepmd_finetune(
    datadir: str,
    atom_types: List[str],
    training_dir: str,
    num_models: int,
    input_file: str = 'input.json',
    pretrained_model: str = 'DPA3.pt',
    model_branch: Optional[str] = None,
    learning_rate: Optional[float] = None,
    **kwargs
) -> str:
    """
    Fine-tune a DeePMD universal model on system-specific DFT data.

    Uses DeePMD-kit v3 `dp --pt finetune` command which initializes
    from a pre-trained checkpoint (DPA-1, DPA-2, DPA-3, etc.) and
    adapts to local data.

    Parameters
    ----------
    datadir : str
        Path to directory containing training_data/ and validation_data/
    atom_types : list
        List of atomic species (e.g., ['O', 'H'])
    training_dir : str
        Directory where training_1/, training_2/, etc. will be created
    num_models : int
        Number of models to fine-tune (for ensemble/QbC)
    input_file : str
        Path to DeepMD input JSON configuration
    pretrained_model : str
        Path to pre-trained model file (.pt)
    learning_rate : float, optional
        Override starting learning rate for fine-tuning

    Returns
    -------
    str
        Name of the frozen model file (e.g., 'frozen_model_2.pth')
    """
    original_dir = os.getcwd()
    os.makedirs(training_dir, exist_ok=True)

    # Validate pre-trained model
    pretrained_path = os.path.join(original_dir, pretrained_model)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pre-trained model not found: {pretrained_path}\n"
            f"Provide a valid path to a DPA model (.pt file)"
        )

    version, backend = get_version()
    if version < 3:
        raise RuntimeError("DeePMD fine-tuning requires DeePMD-kit v3 or later")
    if backend != 'pytorch':
        raise RuntimeError("DeePMD fine-tuning requires PyTorch backend")

    model_ext = '.pth'
    frozen_model_name = None

    SparcLog("=" * 80)
    SparcLog("DeePMD FINE-TUNING")
    SparcLog(f"  Pre-trained model : {pretrained_model}")
    SparcLog(f"  Number of models  : {num_models}")
    SparcLog(f"  Data directory    : {datadir}")
    SparcLog("=" * 80)

    for i in range(1, num_models + 1):
        folder_name = f"training_{i}"
        dir_name = os.path.join(training_dir, folder_name)
        os.makedirs(dir_name, exist_ok=True)

        SparcLog("-" * 80)
        SparcLog(f"Fine-tuning Model {i}/{num_models}")
        SparcLog(f"  Directory: {dir_name}")
        SparcLog("-" * 80)

        try:
            # Load and update JSON configuration
            input_path = os.path.join(original_dir, input_file)
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

            with open(input_path, 'r') as f:
                config_data = json.load(f)

            update_json(config_data, datadir, atom_types)

            # Override learning rate if specified
            if learning_rate is not None and 'learning_rate' in config_data:
                config_data['learning_rate']['start_lr'] = learning_rate

            # Write updated config
            config_output_path = os.path.join(dir_name, input_file)
            with open(config_output_path, 'w') as f:
                json.dump(config_data, f, indent=4)

            # Build fine-tune command
            # DeePMD-kit v3: dp --pt train input.json --finetune model.pt
            finetune_cmd = [
                'dp', '--pt', 'train',
                input_file,
                '--finetune', pretrained_path
            ]
            if model_branch:
                finetune_cmd.extend(['--model-branch', model_branch])

            SparcLog(f"  Command: {' '.join(finetune_cmd)}")
            # Stream output live so user can monitor training progress
            process = subprocess.Popen(
                finetune_cmd, cwd=dir_name,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                print(line, end='', flush=True)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, finetune_cmd, output=''.join(output_lines)
                )
            logger.info(f"DeePMD fine-tuning completed for model {i}")

            # Freeze the fine-tuned model
            frozen_model_name = f"frozen_model_{i}{model_ext}"
            freeze_cmd = ['dp', '--pt', 'freeze', '-o', frozen_model_name]

            SparcLog(f"  Freezing: {' '.join(freeze_cmd)}")
            subprocess.run(freeze_cmd, check=True, cwd=dir_name)
            logger.info(f"Model {i} frozen: {frozen_model_name}")

        except subprocess.CalledProcessError as e:
            error_output = (e.output or '') + str(e)
            SparcLog("=" * 80, level="ERROR")
            SparcLog(f"ERROR: Fine-tuning failed for model {i}", level="ERROR")
            SparcLog(f"  Details: {str(e)}", level="ERROR")
            if e.output:
                # Print last 10 lines of output for diagnostics
                for line in e.output.strip().splitlines()[-10:]:
                    SparcLog(f"  > {line}", level="ERROR")

            # Check for common errors and provide actionable guidance
            if 'No module named' in error_output:
                missing = error_output.split("No module named")[-1].strip().strip("'\"")
                SparcLog("", level="ERROR")
                SparcLog(f"  MISSING DEPENDENCY: {missing}", level="ERROR")
                SparcLog(f"  Make sure you have activated the correct conda environment", level="ERROR")
                SparcLog(f"  with DeePMD-kit and PyTorch installed.", level="ERROR")
            elif 'unexpected keyword argument' in error_output or 'got an unexpected' in error_output:
                SparcLog("", level="ERROR")
                SparcLog("  This is likely a VERSION MISMATCH between the pre-trained model", level="ERROR")
                SparcLog(f"  and your installed DeePMD-kit (v{version}).", level="ERROR")
                SparcLog("", level="ERROR")
                SparcLog(f"  Pre-trained model: {pretrained_model}", level="ERROR")
                SparcLog("", level="ERROR")
                SparcLog("  Solutions:", level="ERROR")
                SparcLog("    1. Upgrade DeePMD-kit:  pip install --upgrade deepmd-kit[torch]", level="ERROR")
                SparcLog("    2. Use a model compatible with your DeePMD-kit version", level="ERROR")
                SparcLog("       e.g., DPA-2 models work with DeePMD-kit v3.0.x", level="ERROR")
            SparcLog("=" * 80, level="ERROR")
            raise

    SparcLog("=" * 80)
    SparcLog(f"DeePMD fine-tuning complete. {num_models} models trained.")
    SparcLog("=" * 80)

    return frozen_model_name


################################################################
# MACE Fine-tuning
################################################################

def mace_finetune(
    datadir: str,
    atom_types: List[str],
    training_dir: str,
    num_models: int,
    pretrained_model: str = 'medium',
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 4,
    device: str = 'cpu',
    stress_key: str = 'stress',
    **kwargs
) -> str:
    """
    Fine-tune MACE foundation model on system-specific DFT data.

    Uses mace_run_train with --foundation_model flag for transfer learning
    from pre-trained MACE-MP-0 models.

    Parameters
    ----------
    datadir : str
        Path to directory containing training_data/ and validation_data/
    atom_types : list
        List of atomic species (e.g., ['O', 'H'])
    training_dir : str
        Directory where training_1/, training_2/, etc. will be created
    num_models : int
        Number of models to fine-tune (for ensemble/QbC)
    pretrained_model : str
        Foundation model name ('small', 'medium', 'large') or path to .model file
    num_epochs : int
        Number of fine-tuning epochs (default: 100)
    learning_rate : float
        Learning rate for fine-tuning (default: 0.001)
    batch_size : int
        Batch size (default: 4)
    device : str
        Device for training: 'cpu' or 'cuda' (default: 'cpu')

    Returns
    -------
    str
        Name of the fine-tuned model file
    """
    os.makedirs(training_dir, exist_ok=True)

    # Resolve pretrained_model: built-in names pass through, file paths get resolved
    builtin_names = {'small', 'medium', 'large'}
    if pretrained_model not in builtin_names:
        resolved_model = os.path.abspath(pretrained_model)
        if not os.path.exists(resolved_model):
            raise FileNotFoundError(
                f"MACE model not found: {resolved_model}\n"
                f"Provide a valid path or use a built-in name: {builtin_names}"
            )
        pretrained_model = resolved_model

    # Convert DeepMD npy data to extxyz for MACE
    train_xyz = _convert_deepmd_to_extxyz(datadir, 'training_data')
    valid_xyz = _convert_deepmd_to_extxyz(datadir, 'validation_data')

    frozen_model_name = None

    SparcLog("=" * 80)
    SparcLog("MACE FINE-TUNING")
    SparcLog(f"  Foundation model  : {pretrained_model}")
    SparcLog(f"  Number of models  : {num_models}")
    SparcLog(f"  Epochs            : {num_epochs}")
    SparcLog(f"  Device            : {device}")
    SparcLog("=" * 80)

    for i in range(1, num_models + 1):
        folder_name = f"training_{i}"
        dir_name = os.path.join(training_dir, folder_name)
        os.makedirs(dir_name, exist_ok=True)

        model_name = f"mace_finetuned_{i}"
        frozen_model_name = f"frozen_model_{i}.model"

        SparcLog("-" * 80)
        SparcLog(f"Fine-tuning Model {i}/{num_models}")
        SparcLog(f"  Directory: {dir_name}")
        SparcLog("-" * 80)

        try:
            # Build MACE fine-tuning command
            mace_cmd = [
                'mace_run_train',
                '--name', model_name,
                '--foundation_model', pretrained_model,
                '--train_file', str(train_xyz),
                '--valid_file', str(valid_xyz),
                f'--lr={learning_rate}',
                f'--batch_size={batch_size}',
                f'--max_num_epochs={num_epochs}',
                f'--device={device}',
                f'--seed={40 + i}',
                '--energy_key=energy',
                '--forces_key=forces',
                f'--stress_key={stress_key}',
                '--E0s=average',
            ]

            SparcLog(f"  Command: {' '.join(mace_cmd)}")
            # Stream output live so user can monitor training progress
            process = subprocess.Popen(
                mace_cmd, cwd=dir_name,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                print(line, end='', flush=True)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, mace_cmd, output=''.join(output_lines)
                )
            logger.info(f"MACE fine-tuning completed for model {i}")

            # MACE outputs: {model_name}.model and {model_name}_compiled.model
            source_model = os.path.join(dir_name, f"{model_name}.model")
            target_model = os.path.join(dir_name, frozen_model_name)

            if os.path.exists(source_model):
                os.rename(source_model, target_model)
                SparcLog(f"  Model saved: {frozen_model_name}")
            else:
                raise FileNotFoundError(
                    f"Expected MACE model not found: {source_model}"
                )

        except subprocess.CalledProcessError as e:
            SparcLog(f"ERROR: MACE fine-tuning failed for model {i}", level="ERROR")
            SparcLog(f"  Details: {str(e)}", level="ERROR")
            raise

    SparcLog("=" * 80)
    SparcLog(f"MACE fine-tuning complete. {num_models} models trained.")
    SparcLog("=" * 80)

    return frozen_model_name


################################################################
# Data Conversion Utilities
################################################################

def _convert_deepmd_to_extxyz(datadir: str, subset: str) -> Path:
    """
    Convert DeepMD npy data to extended XYZ format for MACE.

    Parameters
    ----------
    datadir : str
        Base data directory containing training_data/ or validation_data/
    subset : str
        Subdirectory name ('training_data' or 'validation_data')

    Returns
    -------
    Path
        Path to the generated .xyz file
    """
    import dpdata
    from ase.io import write as ase_write

    data_path = os.path.join(datadir, subset)
    output_xyz = Path(datadir) / f"{subset}.xyz"

    if output_xyz.exists() and output_xyz.stat().st_size > 0:
        SparcLog(f"  Using existing {output_xyz}")
        return output_xyz

    SparcLog(f"  Converting {data_path} → {output_xyz}")

    # Load DeepMD npy data
    ds = dpdata.LabeledSystem(data_path, fmt='deepmd/npy')

    # Convert to ASE atoms and write extxyz
    # dpdata attaches a SinglePointCalculator with energy/forces automatically
    frames = []
    for i in range(ds.get_nframes()):
        atoms = ds[i].to('ase/structure')[0]
        frames.append(atoms)

    ase_write(str(output_xyz), frames, format='extxyz')
    SparcLog(f"  Converted {len(frames)} frames to {output_xyz}")

    return output_xyz


################################################################
# Dispatcher
################################################################

def finetune_training(
    finetune_config,
    datadir: str,
    atom_types: List[str],
    training_dir: str,
    num_models: int,
    input_file: str = 'input.json',
) -> str:
    """
    Dispatch fine-tuning to the appropriate backend (DeePMD or MACE).

    Parameters
    ----------
    finetune_config : FineTuneConfig
        Fine-tuning configuration from input.yaml
    datadir : str
        Path to training data directory
    atom_types : list
        Atomic species list
    training_dir : str
        Output directory for trained models
    num_models : int
        Number of ensemble models
    input_file : str
        DeepMD JSON config (used by DeePMD fine-tuning only)

    Returns
    -------
    str
        Frozen model filename
    """
    model_type = finetune_config.model_type.lower()

    SparcLog("")
    SparcLog("=" * 80)
    SparcLog(f"UNIVERSAL MODEL FINE-TUNING: {model_type.upper()}")
    SparcLog("=" * 80)

    # Use finetune-specific input file if provided, otherwise fall back to mlip_setup.input_file
    ft_input_file = finetune_config.input_file or input_file

    if model_type == 'deepmd':
        return deepmd_finetune(
            datadir=datadir,
            atom_types=atom_types,
            training_dir=training_dir,
            num_models=num_models,
            input_file=ft_input_file,
            pretrained_model=finetune_config.pretrained_model,
            model_branch=finetune_config.model_branch,
            learning_rate=finetune_config.learning_rate,
        )

    elif model_type == 'mace':
        return mace_finetune(
            datadir=datadir,
            atom_types=atom_types,
            training_dir=training_dir,
            num_models=num_models,
            pretrained_model=finetune_config.pretrained_model,
            num_epochs=finetune_config.num_epochs,
            learning_rate=finetune_config.learning_rate,
            batch_size=finetune_config.batch_size,
            device=finetune_config.device,
        )

    else:
        raise ValueError(
            f"Unknown fine-tune model type: '{model_type}'. "
            f"Supported: 'deepmd', 'mace'"
        )


################################################################
# MACE ASE Calculator Setup
################################################################

def setup_MACE_calculator(atoms, model_path: str):
    """
    Setup a MACE calculator for an ASE atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure
    model_path : str
        Path to the MACE .model file

    Returns
    -------
    tuple
        (atoms_with_calc, calculator)
    """
    from ase import Atoms

    if not Path(model_path).exists():
        raise FileNotFoundError(f"MACE model not found: {model_path}")

    try:
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=model_path, device='cpu')
    except ImportError:
        raise ImportError(
            "MACE not installed. Install with: pip install mace-torch"
        )

    system = Atoms(atoms, calculator=calc)

    # Test calculator
    energy = system.get_potential_energy()
    forces = system.get_forces()

    if energy is not None and forces is not None:
        SparcLog("-" * 80)
        SparcLog(f"MACE model loaded and tested:")
        SparcLog(f"  Model: {model_path}")
        SparcLog(f"  Energy: {energy:.6f} eV")
        SparcLog("-" * 80)
    else:
        raise ValueError("MACE calculator failed to compute energy/forces")

    return system, calc


################################################################
# END OF FILE
################################################################

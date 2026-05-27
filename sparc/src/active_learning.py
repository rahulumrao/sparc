#!/usr/bin/python3
# active_learning.py

"""
Active learning module for SPARC package.

This module implements Query-by-Committee for selecting candidate structures
from MD trajectories that need DFT labeling based on model deviation.
"""

import os
import subprocess
from pathlib import Path

import dpdata

################################################################
# Third party imports
import numpy as np

from sparc.src.labelling import labelling

################################################################
# Local imports
from sparc.src.utils.logger import SparcLog

################################################################


def _get_model_fparam_info(model_file: str):
    """
    Read numb_fparam and default_fparam from a frozen DeepMD model.

    Frozen models (.pth after `dp freeze`, or .pb) are TorchScript / SavedModel
    archives — not plain checkpoint dicts. DeepPot handles both formats correctly
    and exposes get_dim_fparam().

    default_fparam is read from the training input.json that sits alongside the
    frozen model (written there by SPARC's finetune/train step). This is the only
    reliable source after freezing since the TorchScript archive does not carry
    the original training config.

    Parameters
    ----------
    model_file : str
        Path to frozen model (.pth or .pb)

    Returns
    -------
    tuple
        (numb_fparam, default_fparam)
        numb_fparam = 0 means no fparam needed.
    """
    import json

    model_file = str(model_file)

    # ── Step 1: get numb_fparam via DeepPot (works for both .pth and .pb) ──
    try:
        from deepmd.infer import DeepPot

        dp = DeepPot(model_file)
        numb_fparam = dp.get_dim_fparam()
    except Exception as e:
        SparcLog(
            f"Warning: could not load model {Path(model_file).name} to query fparam: {e}"
        )
        return 0, []

    if numb_fparam == 0:
        return 0, []

    # ── Step 2: read default_fparam from out.json ────────────────────────────
    # DeepMD always writes out.json into the training_N/ directory with the full
    # serialized model config, including fitting_net.default_fparam.
    training_dir = Path(model_file).parent
    json_out = training_dir / "out.json"

    if json_out.exists():
        try:
            with open(json_out) as f:
                config = json.load(f)
            _, default_fparam = _search_fparam_in_config(config)
            if default_fparam:
                return numb_fparam, default_fparam
        except Exception as e:
            SparcLog(f"Warning: could not read default_fparam from {json_out}: {e}")

    # Fallback: zeros
    SparcLog(
        f"Warning: out.json not found in {training_dir}, "
        f"using zeros for fparam (numb_fparam={numb_fparam})."
    )
    return numb_fparam, [0.0] * numb_fparam


def _search_fparam_in_config(config):
    """
    Recursively search a nested model config dict for numb_fparam / default_fparam.

    Returns the first (numb_fparam, default_fparam) pair where numb_fparam > 0,
    or (0, []) if not found.
    """
    if isinstance(config, dict):
        if "numb_fparam" in config and config["numb_fparam"] > 0:
            numb_fparam = int(config["numb_fparam"])
            default_fparam = config.get("default_fparam", [0.0] * numb_fparam)
            return numb_fparam, list(default_fparam)
        for v in config.values():
            result = _search_fparam_in_config(v)
            if result[0] > 0:
                return result
    return 0, []


def _apply_fparam_to_dataset(dataset, numb_fparam: int, default_fparam: list):
    """
    Register fparam as a dpdata data type and fill every frame with default_fparam.

    Universal models like DPA-3 embed default_fparam in their config. Using
    these values ensures dp model-devi receives the same conditioning the model
    was designed to fall back on when no explicit fparam is supplied.

    Parameters
    ----------
    dataset : dpdata.LabeledSystem
        Dataset loaded from the MD trajectory
    numb_fparam : int
        Number of frame parameters (from model config)
    default_fparam : list
        Default fparam vector (from model config, e.g. [0.0, 1.0] for DPA-3)
    """
    try:
        from dpdata.data_type import Axis, DataType

        dpdata.LabeledSystem.register_data_type(
            DataType("fparam", np.ndarray, (Axis.NFRAMES, -1), required=False)
        )
    except Exception as e:
        SparcLog(f"Warning: could not register fparam data type with dpdata: {e}")

    n_frames = dataset.get_nframes()
    fparam_array = np.tile(default_fparam, (n_frames, 1)).astype(np.float64)
    dataset.data["fparam"] = fparam_array

    SparcLog(
        f"  fparam set: {default_fparam} × {n_frames} frames (numb_fparam={numb_fparam})"
    )


################################################################


def QueryByCommittee(
    trajfile: str,
    model_path: str,
    num_models: int,
    max_lim: float,
    min_lim: float,
    dpmd_data_path: str,
    iteration: int = 0,
    rmsd_threshold: float = 0.05,
    exclude_hydrogen: bool = True,
):
    """
    Find maximum deviation in atomic forces among multiple models using Query-by-Committee.

    This function evaluates force predictions from an ensemble of models trained on
    the same dataset with different random initializations. Structures with force
    deviations outside [min_lim, max_lim] are selected as candidates for DFT labeling.
    RMSD filtering removes near-duplicate candidates to ensure training set diversity.

    Parameters
    ----------
    trajfile : str
        Path to the ASE trajectory file containing atomic coordinates
    model_path : str
        Path to the directory containing DeepMD training folders
    num_models : int
        Number of models to consider (minimum 2)
    max_lim : float
        Maximum force deviation threshold (eV/Å)
    min_lim : float
        Minimum force deviation threshold (eV/Å)
    dpmd_data_path : str
        Path to the directory where DeePMD npy data will be saved
    iteration : int, optional
        Current iteration number (default: 0)
    rmsd_threshold : float, optional
        RMSD threshold in Å for duplicate filtering; candidates with RMSD < threshold
        relative to the initial frame or any already-accepted candidate are discarded
        (default: 0.05)
    exclude_hydrogen : bool, optional
        Exclude hydrogen atoms when computing RMSD (default: True)

    Returns
    -------
    tuple
        (candidate_found, candidates_file, n_candidates, model_names)
        - candidate_found: bool
        - candidates_file: str, path to candidates.extxyz trajectory
        - n_candidates: int, number of candidate frames
        - model_names: list, model file paths used
    """
    SparcLog("")
    SparcLog("QUERY-BY-COMMITTEE: MODEL DEVIATION ANALYSIS")
    SparcLog("-" * 80)
    SparcLog(f"{'Iteration':<30} {iteration}")
    SparcLog(f"{'Number of models':<30} {num_models}")
    SparcLog(f"{'Deviation range':<30} [{min_lim:.2f}, {max_lim:.2f}] eV/Å")
    SparcLog("-" * 80)

    SparcLog("=" * 80)
    SparcLog(f"Model Path: {model_path}".center(72))
    SparcLog("=" * 80)

    # ── Discover model files first (supports .pb and .pth) ──────────────────
    model_names = []
    for folder in sorted(os.listdir(model_path)):
        folder_path = os.path.join(model_path, folder)
        if folder.startswith("training_") and os.path.isdir(folder_path):
            model_number = folder.split("_")[1]
            for ext in [".pb", ".pth"]:
                model_file = os.path.join(
                    folder_path, f"frozen_model_{model_number}{ext}"
                )
                if os.path.exists(model_file):
                    model_names.append(model_file)
                    break

    model_names = model_names[-num_models:]

    if len(model_names) < num_models:
        SparcLog("=" * 80)
        SparcLog(
            f"Error: Found only {len(model_names)} models, but {num_models} are required".center(
                72
            )
        )
        SparcLog("Check the model_path!".center(72))
        SparcLog("=" * 80)
        raise ValueError(
            f"Found only {len(model_names)} models, but {num_models} are required. "
            f"Check the model_path!"
        )

    SparcLog("=" * 80)
    SparcLog("Using the following models:".center(72))
    SparcLog("=" * 80)
    for model in model_names:
        SparcLog(f"{Path(model).name}".center(72))
    SparcLog("=" * 80)

    # ── Convert trajectory to DeePMD npy format ──────────────────────────────
    # Universal models (e.g. DPA-3) use frame parameters (fparam). Read the
    # model's own default_fparam and apply it before writing so that
    # dp model-devi finds the required fparam.npy in every set directory.
    dataset = dpdata.LabeledSystem(trajfile, fmt="ase/traj")

    numb_fparam, default_fparam = _get_model_fparam_info(model_names[0])
    if numb_fparam > 0:
        SparcLog(
            f"  Model requires fparam (numb_fparam={numb_fparam}), "
            f"applying default_fparam={default_fparam}"
        )
        _apply_fparam_to_dataset(dataset, numb_fparam, default_fparam)

    dataset.to_deepmd_npy(str(dpmd_data_path))

    # ── Run dp model-devi ────────────────────────────────────────────────────
    outfile = f"{str(dpmd_data_path)}/model_dev_{iteration}.out"
    command = (
        ["dp", "model-devi", "-m"]
        + model_names
        + ["-s", str(dpmd_data_path), "-o", str(outfile)]
    )

    try:
        subprocess.run(command, check=True)
        SparcLog("=" * 80)
        SparcLog("Model deviation calculation completed successfully!".center(72))
        SparcLog(f"Results saved in: {outfile}".center(72))
        SparcLog("=" * 80)
    except subprocess.CalledProcessError as e:
        SparcLog("=" * 80)
        SparcLog("Error in dp model-devi command execution".center(72))
        SparcLog(str(e).center(72))
        SparcLog("=" * 80)
        raise

    # ── Select candidates based on force deviation ───────────────────────────
    candidate_found, candidates_file, n_candidates = labelling(
        trajfile,
        str(outfile),
        min_lim,
        max_lim,
        output_dir=f"{str(dpmd_data_path)}/dft_candidates",
        rmsd_threshold=rmsd_threshold,
        exclude_hydrogen=exclude_hydrogen,
    )

    with open("learning_state.log", "a") as f:
        f.write(f"\nIteration {iteration:06d}\n")
        f.write(f"Training data from: {trajfile}\n")
        f.write(f"Model deviation range: [{min_lim:.3f}, {max_lim:.3f}] eV/Å\n")
        f.write(f"Candidates found: {n_candidates if candidate_found else 0}\n")
        f.write("-" * 80 + "\n")

    return candidate_found, candidates_file, n_candidates, model_names


################################################################
# END OF FILE
################################################################

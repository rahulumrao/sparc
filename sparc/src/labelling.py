#!/usr/bin/python3
# labelling.py

"""
This module selects structures for DFT labeling based on force deviations
from an ensemble of machine learning models (Query-by-Committee).

Structures with model deviation (max_devi_f) within a specified range
are selected as candidates. RMSD-based filtering removes duplicate/similar
structures to ensure diverse training data.

Reference: https://doi.org/10.1016/j.cpc.2020.107206
"""

import os
from datetime import datetime

################################################################
# Third party imports
import pandas as pd
from ase.io import read, write

################################################################
# Local imports
from sparc.src.deal import run_deal_filter
from sparc.src.utils.logger import SparcLog
from sparc.src.utils.rmsd import kabsch_rmsd

################################################################


def labelling(
    trajfile: str,
    outfile: str,
    min_lim: float,
    max_lim: float,
    output_dir: str = None,
    **kwargs,
):
    """
    Select and extract structures for labeling based on force deviations.

    This function reads model deviation data from dp model-devi output and
    identifies candidate structures whose maximum force deviation falls within
    the specified range [min_lim, max_lim]. Optionally filters out similar
    structures using RMSD to ensure diverse training data.

    Parameters
    ----------
    trajfile : str
        Path to trajectory file (ASE-readable format)
    outfile : str
        Path to model deviation output file from dp model-devi
    min_lim : float
        Minimum force deviation threshold (eV/Å)
    max_lim : float
        Maximum force deviation threshold (eV/Å)
    output_dir : str, optional
        Path to directory for saving candidate structure files
        If None, uses 'dft_candidates' (default: None)
    **kwargs : dict
        Optional keyword arguments:
        - save_summary (bool): Save summary CSV with metadata (default: True)
        - rmsd_threshold (float): RMSD threshold (Å) for filtering duplicates.
            Structures with RMSD < threshold are SKIPPED (default: None, no filtering)
        - exclude_hydrogen (bool): Exclude H atoms in RMSD calculation (default: False)

    Returns
    -------
    tuple
        (candidate_found, candidates_file, n_candidates)
        - candidate_found: bool, whether any candidates were selected
        - candidates_file: str, path to the combined candidates.extxyz trajectory
        - n_candidates: int, number of candidate frames in the file

    Notes
    -----
    RMSD filtering (when enabled):
    - RMSD < threshold: SKIP (too similar, duplicate)
    - RMSD >= threshold: KEEP (sufficiently different)
    - Higher RMSD is always better (more diverse structures)

    Example:
    >>> # Original usage (backward compatible)
    >>> labelling('traj.traj', 'model_dev.out', 0.05, 0.20)

    >>> # With RMSD filtering
    >>> labelling('traj.traj', 'model_dev.out', 0.05, 0.20,
    ...           rmsd_threshold=0.2, exclude_hydrogen=True)
    """
    # Extract kwargs with defaults
    save_summary = kwargs.get("save_summary", True)
    rmsd_threshold = kwargs.get("rmsd_threshold", 0.05)
    exclude_hydrogen = kwargs.get("exclude_hydrogen", True)
    deal_config = kwargs.get("deal_config", None)
    structure_file = kwargs.get("structure_file", "")
    deepmd_input_json = kwargs.get("deepmd_input_json", "")

    # Set default output directory
    if output_dir is None:
        output_dir = "dft_candidates"

    # Read model deviation file
    names = [
        "step",
        "max_devi_v",
        "min_devi_v",
        "avg_devi_v",
        "max_devi_f",
        "min_devi_f",
        "avg_devi_f",
        "dev_e",
    ]

    try:
        data = pd.read_csv(outfile, sep=r"\s+", comment="#", names=names)
    except Exception as e:
        SparcLog(f"Error reading model deviation file: {e}")
        return False, "", 0

    # Filter structures within deviation range
    candidates = data[(data["max_devi_f"] >= min_lim) & (data["max_devi_f"] <= max_lim)]

    if candidates.empty:
        SparcLog(
            f"No candidates found for labelling within range [{min_lim:.2f}, {max_lim:.2f}] eV/Å"
        )
        return False, "", 0

    # SparcLog("=" * 90)
    # SparcLog(f"Found {len(candidates)} candidates within range [{min_lim:.2f}, {max_lim:.2f}] eV/Å")
    # if rmsd_threshold is not None:
    #     SparcLog(f"RMSD filtering: structures with RMSD < {rmsd_threshold:.3f} Å will be skipped")
    # SparcLog("=" * 90 + "\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read trajectory
    try:
        dptraj = read(trajfile, index=":")
    except Exception as e:
        SparcLog(f"Error reading trajectory file: {e}")
        return False, "", 0

    # Initialize tracking
    candidates_file = os.path.join(output_dir, "candidates.extxyz")
    accepted_structures = []
    candidate_metadata = {
        "frame_indices": [],
        "max_force_deviations": [],
        "avg_force_deviations": [],
        "min_rmsd_to_previous": [],
        "serial_numbers": [],
    }

    serial = 0
    skipped = 0

    excluded_traj_file = os.path.join(output_dir, "excluded_frames.extxyz")

    # ── Build ordered candidate list; embed metadata in frame.info ────────────
    # _step_index, max_devi_f, _avg_devi_f survive the deal_selected.xyz
    # round-trip so labelling.py can recover per-frame stats without id() matching.
    ordered_candidates = []
    for _, row in candidates.iterrows():
        frame_index = int(row["step"])
        frame = dptraj[frame_index]
        frame.info["_step_index"] = frame_index
        frame.info["max_devi_f"] = float(row["max_devi_f"])
        frame.info["_avg_devi_f"] = float(row["avg_devi_f"])
        ordered_candidates.append(
            (frame_index, float(row["max_devi_f"]), float(row["avg_devi_f"]), frame)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 diversity filter — DEAL (SGP) or Kabsch RMSD
    # ══════════════════════════════════════════════════════════════════════════

    if deal_config is not None and deal_config.enabled:
        # ── DEAL SGP path ─────────────────────────────────────────────────────
        candidate_pool = [f for _, _, _, f in ordered_candidates]
        is_periodic = bool(dptraj[0].pbc.any())

        # run_deal_filter writes deal_selected.xyz directly into output_dir
        # via DEALConfig.output_prefix — no post-run file move needed.
        # _step_index / max_devi_f / _avg_devi_f embedded in frame.info above
        # survive the xyz round-trip — no id() matching needed.
        selected_frames = run_deal_filter(
            candidate_frames=candidate_pool,
            sparc_deal_cfg=deal_config,
            is_periodic=is_periodic,
            structure_file=structure_file,
            deepmd_input_json=deepmd_input_json,
            output_dir=output_dir,
        )

        # DEAL already wrote deal_selected.xyz — use it directly as candidates_file
        # so sparc.py reads the same file for DFT labelling. No duplicate write.
        candidates_file = os.path.join(output_dir, "deal_selected.xyz")

        for frame in selected_frames:
            frame_index = int(frame.info.get("_step_index", 0))
            max_f_dev = float(frame.info.get("max_devi_f", 0.0))
            avg_f_dev = float(frame.info.get("_avg_devi_f", 0.0))
            serial += 1
            accepted_structures.append(frame)
            candidate_metadata["frame_indices"].append(frame_index)
            candidate_metadata["max_force_deviations"].append(max_f_dev)
            candidate_metadata["avg_force_deviations"].append(avg_f_dev)
            candidate_metadata["min_rmsd_to_previous"].append(0.0)
            candidate_metadata["serial_numbers"].append(serial)

        skipped = len(candidate_pool) - len(selected_frames)
        n_candidates = len(accepted_structures)

        SparcLog("=" * 80)
        SparcLog("Selection Summary (DEAL SGP filter):")
        SparcLog(
            f"  Candidates in deviation range [{min_lim:.2f}, {max_lim:.2f}] eV/Å: {len(candidates)}"
        )
        SparcLog(f"  Skipped (SGP variance below threshold): {skipped}")
        SparcLog(f"  Final selected structures: {n_candidates}")
        if n_candidates > 0:
            SparcLog(f"  Candidates trajectory: {candidates_file}")
        SparcLog("=" * 90 + "\n")

    else:
        # ── RMSD path (original, unchanged) ──────────────────────────────────
        ref_structure = dptraj[0].get_positions()
        accepted_positions = []
        chem_symbols = dptraj[0].get_chemical_symbols()

        rmsd_log_file = os.path.join(output_dir, "rmsd_filtering.dat")
        with open(rmsd_log_file, "w") as rmsd_log:
            rmsd_log.write("# RMSD Filtering Log\n")
            rmsd_log.write(
                f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            rmsd_log.write(f"# RMSD Threshold: {rmsd_threshold:.3f} Å\n")
            rmsd_log.write(f"# Exclude Hydrogen: {exclude_hydrogen}\n")
            rmsd_log.write("#\n")
            rmsd_log.write(
                f"# {'Status':<8} {'Serial':<8} {'Frame':<8} {'MaxFDev':<12} {'MinRMSD':<12} {'Action':<15}\n"
            )
            rmsd_log.write(f"#{'=' * 80}\n")

            for (
                frame_index,
                max_f_dev,
                avg_f_dev,
                current_structure,
            ) in ordered_candidates:
                min_rmsd = None

                if rmsd_threshold is not None:
                    current_pos = current_structure.get_positions()
                    rmsd_values = []

                    try:
                        rmsd = kabsch_rmsd(
                            current_pos,
                            ref_structure,
                            noH=exclude_hydrogen,
                            symbols=chem_symbols if exclude_hydrogen else None,
                        )
                        rmsd_values.append(rmsd)
                    except Exception as e:
                        rmsd_log.write(
                            f"ERROR    {serial:>7d} {frame_index:>7d} {max_f_dev:>11.4f} "
                            f"{'N/A':<11}  RMSD vs frame0 failed: {e}\n"
                        )
                        continue

                    for ref_pos in accepted_positions:
                        try:
                            rmsd = kabsch_rmsd(
                                current_pos,
                                ref_pos,
                                noH=exclude_hydrogen,
                                symbols=chem_symbols if exclude_hydrogen else None,
                            )
                            rmsd_values.append(rmsd)
                        except Exception:
                            continue

                    min_rmsd = min(rmsd_values)

                    if min_rmsd < rmsd_threshold:
                        skipped += 1
                        write(
                            excluded_traj_file,
                            current_structure,
                            format="extxyz",
                            append=True,
                        )
                        rmsd_log.write(
                            f"SKIP     {'---':>7} {frame_index:>7d} {max_f_dev:>11.4f} "
                            f"{min_rmsd:>11.3f}  RMSD < {rmsd_threshold:.3f}\n"
                        )
                        continue

                serial += 1
                rmsd_str = (
                    f"{min_rmsd:>11.3f}" if min_rmsd is not None else "N/A (first)"
                )
                rmsd_log.write(
                    f"ACCEPT   {serial:>7d} {frame_index:>7d} {max_f_dev:>11.4f} "
                    f"{rmsd_str}  Selected\n"
                )

                if rmsd_threshold is not None:
                    accepted_positions.append(current_pos.copy())

                accepted_structures.append(current_structure)
                candidate_metadata["frame_indices"].append(frame_index)
                candidate_metadata["max_force_deviations"].append(max_f_dev)
                candidate_metadata["avg_force_deviations"].append(avg_f_dev)
                candidate_metadata["min_rmsd_to_previous"].append(
                    min_rmsd if min_rmsd is not None else 0.0
                )
                candidate_metadata["serial_numbers"].append(serial)

        # Write all accepted structures to a single trajectory file
        n_candidates = len(accepted_structures)
        if n_candidates > 0:
            write(candidates_file, accepted_structures, format="extxyz")

        SparcLog("=" * 80)
        SparcLog("Selection Summary (RMSD filter):")
        SparcLog(
            f"  Candidates in deviation range [{min_lim:.2f}, {max_lim:.2f}] eV/Å: {len(candidates)}"
        )
        if rmsd_threshold is not None:
            SparcLog(f"  Skipped (RMSD < {rmsd_threshold:.3f} Å): {skipped}")
        SparcLog(f"  Final selected structures: {n_candidates}")
        if n_candidates > 0:
            SparcLog(f"  Candidates trajectory: {candidates_file}")
        SparcLog(f"  RMSD details written to: {rmsd_log_file}")
        SparcLog("=" * 90 + "\n")

    # Save summary CSV for tracking
    if save_summary and n_candidates > 0:
        summary_df = pd.DataFrame(
            {
                "ID": candidate_metadata["serial_numbers"],
                "Frame": candidate_metadata["frame_indices"],
                "MaxFDev_eV_A": candidate_metadata["max_force_deviations"],
                "AvgFDev_eV_A": candidate_metadata["avg_force_deviations"],
                "RMSD_A": candidate_metadata["min_rmsd_to_previous"],
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # Format Output
        summary_df["MaxFDev_eV_A"] = summary_df["MaxFDev_eV_A"].map("{:.4f}".format)
        summary_df["AvgFDev_eV_A"] = summary_df["AvgFDev_eV_A"].map("{:.4f}".format)
        summary_df["RMSD_A"] = summary_df["RMSD_A"].map("{:.4f}".format)
        # Save metadata file
        summary_file = os.path.join(output_dir, "candidates_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        SparcLog(f"Candidate summary saved to: {summary_file}\n")

    candidate_found = n_candidates > 0

    return candidate_found, candidates_file, n_candidates


################################################################
# END OF FILE
################################################################

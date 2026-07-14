#!/usr/bin/python3
# sparc/src/deal/filter.py

"""
DEAL integration adapter for SPARC.

run_deal_filter() is the single public entry point. It:
  1. Resolves FLARE cutoff (explicit override → auto from DeepMD input.json → default)
  2. Auto-detects element species from the structure file
  3. Attaches dummy SinglePointCalculator to pre-DFT candidate frames
  4. Builds DataConfig / DEALConfig / FlareConfig and runs DEAL.run()
  5. Returns the selected ASE Atoms list (read from deal_selected.xyz)

All FLARE / DEAL imports are inside the function body so the rest of
SPARC never imports flare-pp at module load time — DEAL stays optional.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List

from ase import Atoms

from sparc.src.deal.utils import (
    _extract_rcut_from_deepmd_json,
    _get_species,
    attach_dummy_calc,
)
from sparc.src.utils.logger import SparcLog

if TYPE_CHECKING:
    from sparc.src.utils.read_input import DEALConfig


class _DealScreenPassthrough:
    """Routes DEAL verbose output to screen (nohup.out) only — bypasses Sparc.log.
    Replaces \\r with \\n so progress lines are readable under `tail -f`."""

    def __init__(self, console):
        self._console = console

    def write(self, text):
        self._console.write(text.replace("\r", "\n"))
        self._console.flush()

    def flush(self):
        self._console.flush()


def run_deal_filter(
    candidate_frames: List[Atoms],
    sparc_deal_cfg: "DEALConfig",
    is_periodic: bool,
    structure_file: str,
    deepmd_input_json: str,
    output_dir: str = ".",
) -> List[Atoms]:
    """
    Run DEAL SGP variance filter over force-dev-preselected candidate frames.

    Parameters
    ----------
    candidate_frames : list[Atoms]
        Frames that already passed the QBC [min_lim, max_lim] force-dev gate.
        Caller must store trajectory metadata (_step_index, max_devi_f,
        _avg_devi_f) in frame.info before calling — these survive the
        deal_selected.xyz round-trip and are used by labelling.py for the
        summary CSV without any id()-based object matching.
    sparc_deal_cfg : DEALConfig
        DEAL parameters parsed from input.yaml deal: section.
    is_periodic : bool
        True for periodic systems (VASP/CP2K/QE). False for molecules
        (ORCA/Gaussian/xTB) — disables stress in FLARE.
    structure_file : str
        Path to the input structure file — used for species auto-detection.
    deepmd_input_json : str
        Path to mlip_setup.input_file — used to auto-detect cutoff (rcut for
        DeepMD; r_max for MACE). Override with deal.cutoff in input.yaml.

    Returns
    -------
    list[Atoms]
        Frames selected by SGP variance — read directly from deal_selected.xyz.
        frame.info metadata set by caller is preserved across the xyz round-trip.
        Returns all candidate_frames unchanged if flare-pp is not installed.
    """
    import os

    # flare-pp links Eigen which auto-detects MKL; some conda builds ship
    # an inconsistent libmkl_avx512.so causing a FATAL load error.
    # Setting GNU threading layer avoids the broken AVX-512 code path.
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    # Cap flare-pp OpenMP threads at min(half CPUs, 4) — leaves cores for DFT jobs.
    _omp_threads = min(max((os.cpu_count() or 4) // 2, 1), 4)
    os.environ.setdefault("OMP_NUM_THREADS", str(_omp_threads))

    try:
        from deal.config import DataConfig, FlareConfig
        from deal.config import DEALConfig as DealDEALConfig
        from deal.core import DEAL
    except (ImportError, OSError) as e:
        SparcLog(
            f"WARNING [DEAL]: flare-pp / deal not importable ({e}). "
            "Falling back to RMSD filter. "
            "Install with: pip install flare-pp && pip install git+https://github.com/luigibonati/DEAL"
        )
        return candidate_frames

    if not candidate_frames:
        return []

    # ── Resolve cutoff ────────────────────────────────────────────────────────
    if sparc_deal_cfg.cutoff is not None:
        cutoff = sparc_deal_cfg.cutoff
        SparcLog(f"DEAL: using explicit cutoff = {cutoff} Å (from input.yaml)")
    else:
        cutoff = _extract_rcut_from_deepmd_json(deepmd_input_json)

    # ── Auto-detect species ───────────────────────────────────────────────────
    species = _get_species(structure_file, candidate_frames)

    SparcLog("-" * 80)
    SparcLog("DEAL SGP FILTER: Sparse Gaussian Process diversity selection")
    SparcLog(f"  Input frames   : {len(candidate_frames)}")
    SparcLog(f"  SGP threshold  : {sparc_deal_cfg.threshold}")
    SparcLog(f"  FLARE cutoff   : {cutoff} Å")
    SparcLog(f"  Species        : {species}")
    SparcLog(f"  Periodic       : {is_periodic}")
    SparcLog("-" * 80)

    # ── Attach dummy calc so DEAL._extract_dft() doesn't crash ───────────────
    # Periodic systems: force_only=False → DEAL reads energy + forces + stress.
    # Molecular systems: force_only=True → only forces needed.
    attach_dummy_calc(candidate_frames, periodic=is_periodic)

    # ── Build DEAL config objects ─────────────────────────────────────────────
    from ase.data import atomic_numbers as _ase_z

    species_z = [_ase_z[s] for s in species]

    data_cfg = DataConfig(images=candidate_frames)

    # output_prefix directs DEAL to write deal_selected.xyz inside output_dir
    # instead of CWD. DEAL appends "_selected.xyz" to this prefix.
    deal_output_prefix = os.path.join(output_dir, "deal")

    deal_cfg = DealDEALConfig(
        threshold=sparc_deal_cfg.threshold,
        max_selected=None,  # no hard cap — let DEAL select all diverse frames
        force_only=(not is_periodic),
        train_hyps=False,
        min_steps_with_model=1,
        verbose=True,
        output_prefix=deal_output_prefix,
    )

    flare_cfg = FlareConfig(
        cutoff=cutoff,
        species=species_z,
        kernels=[{"name": "NormalizedDotProduct", "sigma": 2.0, "power": 2}],
        descriptors=[
            {
                "name": "B2",
                "radial_basis": "chebyshev",
                "cutoff_function": "cosine",
                "nmax": sparc_deal_cfg.nmax,
                "lmax": sparc_deal_cfg.lmax,
            }
        ],
        variance_type="local",
    )

    # ── Run DEAL ──────────────────────────────────────────────────────────────
    # DEAL verbose=True writes \r-overwriting progress to stdout. sys.stdout is
    # the SPARC Logger which tees to both Sparc.log and console — raw \r lines
    # produce ^M garbage in the log file. Route DEAL output to console only
    # (skips Sparc.log) and replace \r with \n so tail -f nohup.out is readable.
    from sparc.src.utils.logger import global_logger as _sparc_logger

    selector = DEAL(data_cfg, deal_cfg, flare_cfg)
    _orig_stdout = sys.stdout
    if _sparc_logger is not None:
        sys.stdout = _DealScreenPassthrough(_sparc_logger.console_output)
    try:
        selector.run()
    finally:
        sys.stdout = _orig_stdout

    # ── Read selected frames from DEAL output ─────────────────────────────────
    # DEAL writes to output_prefix + "_selected.xyz" → output_dir/deal_selected.xyz.
    # frame.info metadata (_step_index, max_devi_f, _avg_devi_f) is preserved
    # through the xyz round-trip — no id() matching needed in labelling.py.
    from ase.io import read as ase_read

    deal_output_xyz = os.path.join(output_dir, "deal_selected.xyz")
    if not os.path.exists(deal_output_xyz):
        SparcLog(
            f"WARNING [DEAL]: {deal_output_xyz} not found — returning all candidate frames."
        )
        return candidate_frames

    selected_frames = list(ase_read(deal_output_xyz, index=":"))
    SparcLog(
        f"DEAL: selected {len(selected_frames)} / {len(candidate_frames)} diverse frames."
    )
    return selected_frames

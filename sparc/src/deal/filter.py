#!/usr/bin/python3
# sparc/src/deal/filter.py

"""
DEAL integration adapter for SPARC.

Requires DEAL >= 3.0, which ships its own native sparse-GP extension — a
separate FLARE installation is no longer needed. Older FLARE-backed releases
(<= 2.1) are not supported: they expose ``FlareConfig`` instead of
``SGPConfig`` and import ``flare`` at module load. When an unsupported version
is found, the filter warns and hands the frames back untouched so the caller
falls back to the RMSD filter.

run_deal_filter() is the single public entry point. It:
  1. Resolves the descriptor cutoff (explicit override → auto from the MLIP
     input JSON → default)
  2. Auto-detects element species from the structure file
  3. Attaches dummy SinglePointCalculator to pre-DFT candidate frames
  4. Optionally derives a per-atom candidate mask, so the GP only evaluates
     atoms flagged as interesting by a per-atom uncertainty array
  5. Builds DataConfig / DEALConfig / SGPConfig and runs DEAL — either in
     threshold mode (single pass) or budget mode (incremental passes with a
     decaying threshold until ``max_selected`` frames are collected)
  6. Returns the selected ASE Atoms list (read from deal_selected.xyz)

input.yaml only exposes enabled / threshold / max_selected / cutoff / mask.
Everything else keeps the SPARC defaults collected below and is reachable
through the ``deal.advanced`` mapping, which is routed to the right DEAL
object (or to SPARC's own budget loop) and validated against the installed
DEAL rather than against a hard-coded schema.

All DEAL imports are inside the function body so the rest of SPARC never
imports DEAL at module load time — DEAL stays optional.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ase import Atoms

from sparc.src.deal.utils import (
    _extract_rcut_from_deepmd_json,
    _get_species,
    add_vacuum_box,
    attach_dummy_calc,
    strip_vacuum_box,
)
from sparc.src.utils.logger import SparcLog

if TYPE_CHECKING:
    from sparc.src.utils.read_input import DEALConfig

# DEAL 3.0 replaced the external FLARE backend with its own native sparse-GP
# extension and renamed FlareConfig → SGPConfig. SPARC targets that API only.
_DEAL_MIN_VERSION = (3, 0)

_INSTALL_HINT = (
    "DEAL >= 3.0 is required. Install with:\n"
    "    git clone https://github.com/luigibonati/DEAL.git && cd DEAL\n"
    "    conda env create -f environment-deal.yml && conda activate deal\n"
    "    python -m pip install -e . --no-build-isolation\n"
    "The sparse-GP backend is built during installation; flare-pp is no longer used."
)

# Per-atom force deviation, stored in frame.arrays with one value per atom.
# Deliberately distinct from the per-frame scalar frame.info["max_devi_f"] that
# drives the QbC gate: "per_atom_" marks the array, "max_" marks the scalar.
# Written by `dp model-devi --atomic` (deepmd calls the columns atm_devi_f).
_DEFAULT_MASK_KEY = "per_atom_devi_f"

# ── SPARC defaults for everything input.yaml does not expose ────────────────
# Every group below is overridable through deal.advanced in input.yaml.

# Passed straight to DEAL's DEALConfig.
_DEAL_DEFAULTS: Dict[str, Any] = {
    "update_threshold": None,  # DEAL derives 0.8 * threshold
    "max_atoms_added": 0.2,
    "min_steps_with_model": 1,
    "initial_atoms": None,
    "train_hyps": False,
    "save_gp": False,
    "save_full_trajectory": False,
    "verbose": True,
}

# Shape of the B2 descriptor and of the NormalizedDotProduct kernel.
_DESCRIPTOR_DEFAULTS: Dict[str, Any] = {
    "nmax": 8,
    "lmax": 4,
    "radial_basis": "chebyshev",
    "cutoff_function": "cosine",
}
_KERNEL_DEFAULTS: Dict[str, Any] = {"sigma": 2.0, "power": 2}

# Remaining SGPConfig knobs. "local" is the only variance_type that keeps
# DEAL's local-uncertainty fast path available.
_SGP_DEFAULTS: Dict[str, Any] = {"variance_type": "local"}

# TrajectoryMasker options — used only when deal.mask is set.
_MASK_DEFAULTS: Dict[str, Any] = {
    "mask_threshold": None,  # None → DEAL's automatic per-frame thresholds
    "mask_upper_threshold": None,
    "mask_mode": "above",
    "mask_plot": False,
    "mask_fraction": 0.3,
}

# Handled by SPARC itself, not by DEAL.
_SPARC_DEFAULTS: Dict[str, Any] = {
    "max_iterations": 10,  # budget mode passes
    "threshold_factor": 0.7,  # budget mode threshold decay
    "seed": 24,  # shuffling between budget passes
    "omp_threads": None,  # None → min(half the CPUs, 4)
}

# Owned by the adapter — setting these in advanced would silently fight the
# input.yaml keys or the periodic/molecular handling.
_RESERVED_KEYS = frozenset(
    {
        "threshold",
        "max_selected",
        "mask",
        "mask_key",
        "force_only",
        "output_prefix",
        "cutoff",
        "species",
        "images",
        "files",
    }
)


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


def _split_advanced(advanced: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Route deal.advanced entries to the object each one belongs to.

    Returns the six option groups — deal, descriptor, kernel, sgp, mask and
    sparc — each pre-filled with the module defaults above. Unknown or reserved
    keys raise immediately: a typo here would otherwise change the science
    silently, and the filter runs early enough in an AL iteration for a hard
    failure to be cheap.
    """
    groups = {
        "deal": dict(_DEAL_DEFAULTS),
        "descriptor": dict(_DESCRIPTOR_DEFAULTS),
        "kernel": dict(_KERNEL_DEFAULTS),
        "sgp": dict(_SGP_DEFAULTS),
        "mask": dict(_MASK_DEFAULTS),
        "sparc": dict(_SPARC_DEFAULTS),
    }

    for key, value in (advanced or {}).items():
        if key in _RESERVED_KEYS:
            hint = (
                "set the per-atom array name as deal.mask instead"
                if key == "mask_key"
                else f"set the top-level deal.{key} instead"
            )
            raise ValueError(f"deal.advanced.{key} is controlled by SPARC — {hint}.")
        for group, options in groups.items():
            if key in options:
                options[key] = value
                break
        else:
            known = sorted(k for opts in groups.values() for k in opts)
            raise ValueError(
                f"Unknown option 'deal.advanced.{key}'. Accepted: {', '.join(known)}"
            )

    return groups


def _set_thread_env(omp_threads: Optional[int]) -> None:
    """Set BLAS/OpenMP env before DEAL's native extension is imported.

    The extension links Eigen, compiled with EIGEN_USE_MKL_ALL whenever LAPACK
    resolves to MKL. Some conda builds ship an inconsistent libmkl_avx512.so
    that aborts at load; the GNU threading layer avoids that code path. Threads
    are capped so the GP does not starve concurrent DFT jobs.
    """
    import os

    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    if omp_threads is None:
        omp_threads = min(max((os.cpu_count() or 4) // 2, 1), 4)
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))


def _installed_version() -> Tuple[str, Optional[Tuple[int, ...]]]:
    """Return (version string, parsed numeric tuple or None)."""
    try:
        from importlib.metadata import version as _pkg_version

        raw = _pkg_version("deal")
    except Exception:
        return "unknown", None

    parts: List[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return raw, tuple(parts) if parts else None


def _import_deal() -> Optional[Dict[str, Any]]:
    """Import the DEAL >= 3.0 API, or return None if it is unusable."""
    raw_version, parsed = _installed_version()
    if parsed is not None and parsed < _DEAL_MIN_VERSION:
        SparcLog(
            f"WARNING [DEAL]: found deal {raw_version}, which still relies on an "
            f"external FLARE install. Falling back to RMSD filter.\n{_INSTALL_HINT}"
        )
        return None

    try:
        from deal.config import DataConfig, SGPConfig
        from deal.config import DEALConfig as DealDEALConfig
        from deal.core import DEAL
        from deal.preprocessing import TrajectoryMasker
    except (ImportError, OSError) as e:
        SparcLog(
            f"WARNING [DEAL]: deal (>= 3.0) not importable ({e}). "
            f"Falling back to RMSD filter.\n{_INSTALL_HINT}"
        )
        return None

    return {
        "DataConfig": DataConfig,
        "DEALConfig": DealDEALConfig,
        "SGPConfig": SGPConfig,
        "DEAL": DEAL,
        "TrajectoryMasker": TrajectoryMasker,
        "version": raw_version,
    }


def _mask_key(cfg: "DEALConfig") -> str:
    """Name of the per-atom array the mask is derived from."""
    return cfg.mask if isinstance(cfg.mask, str) else _DEFAULT_MASK_KEY


def _prepare_mask(
    frames: List[Atoms],
    cfg: "DEALConfig",
    mask_opts: Dict[str, Any],
    masker_cls,
    output_dir: str,
) -> Optional[str]:
    """Build the per-atom ``deal_mask`` array; return the array name or None.

    The mask restricts GP evaluation to atoms whose per-atom uncertainty flags
    them as interesting — this is what makes DEAL cheap on reactive, interface
    and defect systems where only a local region drives the selection.

    SPARC does not yet produce a per-atom uncertainty array itself: frames must
    already carry the source array in ``frame.arrays`` (per-atom force
    deviations from ``dp model-devi --atomic``, or any array attached upstream).
    Missing arrays disable masking rather than aborting the run.
    """
    import os

    if not cfg.mask:
        return None

    key = _mask_key(cfg)
    missing = sum(1 for f in frames if key not in f.arrays)
    if missing:
        SparcLog(
            f"WARNING [DEAL]: {missing}/{len(frames)} frames lack the per-atom "
            f"array '{key}' — masking disabled, every atom stays eligible. "
            "Attach per-atom uncertainties to frame.arrays to enable it."
        )
        return None

    plot = mask_opts["mask_plot"]
    if plot is True:
        plot = os.path.join(output_dir, "deal_mask.png")

    masker = masker_cls(
        key=key,
        mask_threshold=mask_opts["mask_threshold"],
        mask_upper_threshold=mask_opts["mask_upper_threshold"],
        mode=mask_opts["mask_mode"],
        mask_fraction=mask_opts["mask_fraction"],
        mask_key="deal_mask",
        plot=plot,
    )
    try:
        summary = masker.apply_to_trajectory(frames)
    except (RuntimeError, ValueError) as e:
        SparcLog(
            f"WARNING [DEAL]: mask preprocessing failed ({e}) — "
            "continuing with all atoms eligible."
        )
        return None

    SparcLog(
        f"DEAL mask: {summary.n_selected_atoms}/{summary.n_atoms} atoms eligible "
        f"({100.0 * summary.selected_fraction:.1f} %), "
        f"{summary.n_frames_with_selection}/{summary.n_frames} frames keep candidates."
    )
    if summary.lower_threshold is not None:
        SparcLog(
            f"DEAL mask: automatic frame window = "
            f"[{summary.lower_threshold:.4f}, {summary.upper_threshold:.4f}] "
            f"on the per-frame maximum of '{key}'."
        )
    if summary.n_frames_with_selection == 0:
        SparcLog(
            "WARNING [DEAL]: mask left no eligible atom in any frame — nothing could "
            "be selected. Disabling the mask; lower deal.mask_threshold to use it."
        )
        return None

    return "deal_mask"


def _build_configs(
    api: Dict[str, Any],
    cfg: "DEALConfig",
    groups: Dict[str, Dict[str, Any]],
    frames: List[Atoms],
    species_z: List[int],
    cutoff: float,
    is_periodic: bool,
    mask_name: Optional[str],
    output_prefix: str,
) -> Tuple[Any, Dict[str, Any], Any]:
    """Return (data_cfg, deal_kwargs, sgp_cfg).

    DEALConfig is returned as a kwargs dict, not an instance: budget mode
    rebuilds it once per pass with a decayed threshold.
    """
    descriptor = dict(groups["descriptor"])
    kernel = dict(groups["kernel"])

    data_cfg = api["DataConfig"](images=frames, seed=groups["sparc"]["seed"])

    deal_kwargs = dict(
        groups["deal"],
        threshold=cfg.threshold,
        mask=mask_name,
        force_only=(not is_periodic),
        output_prefix=output_prefix,
    )

    sgp_cfg = api["SGPConfig"](
        cutoff=cutoff,
        species=species_z,
        kernels=[dict(name="NormalizedDotProduct", **kernel)],
        descriptors=[dict(name="B2", **descriptor)],
        **groups["sgp"],
    )
    return data_cfg, deal_kwargs, sgp_cfg


def _run_threshold_mode(
    api: Dict[str, Any], data_cfg, deal_kwargs: Dict[str, Any], sgp_cfg
) -> None:
    """Single pass at a fixed uncertainty threshold."""
    selector = api["DEAL"](data_cfg, api["DEALConfig"](**deal_kwargs), sgp_cfg)
    selector.run()


def _run_budget_mode(
    api: Dict[str, Any],
    data_cfg,
    deal_kwargs: Dict[str, Any],
    sgp_cfg,
    cfg: "DEALConfig",
    sparc_opts: Dict[str, Any],
) -> None:
    """Incremental selection until ``max_selected`` frames are collected.

    Mirrors DEAL's own incremental CLI mode, which lives in deal/cli.py rather
    than in DEAL.run(). The same DEAL object — and therefore the same GP and
    inducing set — is reused across passes through configure_run(); DEAL only
    resets its selection counters when a new SGPConfig is passed, so
    deal_selected.xyz keeps appending across passes. The threshold decays as
    ``threshold_factor ** (pass + 1)``, and frames already selected leave the
    pool. Frames carry their trajectory index in ``info['original_frame']``,
    which survives the shuffling DEAL applies from the second pass onward.
    """
    import numpy as np

    max_iterations = sparc_opts["max_iterations"]
    threshold_factor = sparc_opts["threshold_factor"]
    seed = sparc_opts["seed"]

    all_images = list(data_cfg.images or [])
    rng = np.random.default_rng(seed)
    selected_ids: set = set()
    selector = None

    SparcLog(
        f"DEAL: budget mode — target {cfg.max_selected} frames, "
        f"up to {max_iterations} passes, threshold_factor={threshold_factor}"
    )

    for iteration in range(max_iterations):
        remaining = [
            a for a in all_images if a.info.get("original_frame") not in selected_ids
        ]
        if not remaining:
            SparcLog("DEAL: budget mode stopped — every input frame was selected.")
            break
        # Shuffle from the second pass on so frame ordering does not bias which
        # of the remaining frames wins the lowered threshold.
        if iteration != 0:
            rng.shuffle(remaining)

        threshold = float(np.round(threshold_factor ** (iteration + 1), 3))
        SparcLog(
            f"DEAL: pass {iteration + 1}/{max_iterations} "
            f"(threshold={threshold}, pool={len(remaining)})"
        )

        # DEAL derives update_threshold as 0.8*threshold when it is None, so it
        # keeps tracking the decay unless the user pinned it in deal.advanced.
        pass_kwargs = dict(deal_kwargs, threshold=threshold)
        pass_data = api["DataConfig"](images=remaining, seed=seed)
        pass_cfg = api["DEALConfig"](**pass_kwargs)

        if selector is None:
            selector = api["DEAL"](pass_data, pass_cfg, sgp_cfg)
        else:
            selector.configure_run(data_cfg=pass_data, deal_cfg=pass_cfg)

        before = len(selected_ids)
        selector.run()
        selected_ids.update(
            f.info["original_frame"]
            for f in selector.selected_frames
            if "original_frame" in f.info
        )
        SparcLog(
            f"DEAL: pass {iteration + 1} added {len(selected_ids) - before} frames "
            f"({len(selected_ids)}/{cfg.max_selected} total)"
        )

        if len(selected_ids) >= cfg.max_selected:
            SparcLog("DEAL: budget reached.")
            break
    else:
        SparcLog(
            f"WARNING [DEAL]: reached max_iterations={max_iterations} with only "
            f"{len(selected_ids)}/{cfg.max_selected} frames selected. Lower "
            "advanced.threshold_factor or raise advanced.max_iterations."
        )


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
        (ORCA/Gaussian/xTB) — the GP then trains on forces only.
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
        Returns all candidate_frames unchanged if DEAL >= 3.0 is not installed.
    """
    import os

    cfg = sparc_deal_cfg
    # Raises on unknown/reserved keys before any expensive work happens.
    groups = _split_advanced(cfg.advanced)
    _set_thread_env(groups["sparc"]["omp_threads"])

    api = _import_deal()
    if api is None:
        return candidate_frames

    if not candidate_frames:
        return []

    # ── Resolve cutoff ────────────────────────────────────────────────────────
    if cfg.cutoff is not None:
        cutoff = cfg.cutoff
        SparcLog(f"DEAL: using explicit cutoff = {cutoff} Å (from input.yaml)")
    else:
        cutoff = _extract_rcut_from_deepmd_json(deepmd_input_json)

    # ── Auto-detect species ───────────────────────────────────────────────────
    species = _get_species(structure_file, candidate_frames)

    budget_mode = cfg.max_selected is not None

    SparcLog("-" * 80)
    SparcLog("DEAL SGP FILTER: Sparse Gaussian Process diversity selection")
    SparcLog(f"  DEAL version   : {api['version']} (native sparse-GP backend)")
    SparcLog(f"  Input frames   : {len(candidate_frames)}")
    if budget_mode:
        SparcLog(
            f"  Mode           : budget (max_selected={cfg.max_selected}, "
            f"threshold_factor={groups['sparc']['threshold_factor']}, "
            f"max_iterations={groups['sparc']['max_iterations']})"
        )
    else:
        SparcLog(f"  Mode           : threshold ({cfg.threshold})")
    SparcLog(
        f"  Cutoff         : {cutoff} Å   "
        f"(B2 nmax={groups['descriptor']['nmax']}, lmax={groups['descriptor']['lmax']})"
    )
    SparcLog(f"  Species        : {species}")
    SparcLog(f"  Periodic       : {is_periodic}")
    SparcLog(f"  Atom mask      : {_mask_key(cfg) if cfg.mask else 'off'}")
    SparcLog("-" * 80)

    # ── Attach dummy calc so DEAL._extract_dft() doesn't crash ───────────────
    # Periodic systems: force_only=False → DEAL reads energy + forces + stress.
    # Molecular systems: force_only=True → only forces needed.
    attach_dummy_calc(candidate_frames, periodic=is_periodic)

    # ── Give molecular frames a vacuum cell ───────────────────────────────────
    # The B2 descriptor builds its neighbour list from the cell. A zero cell
    # (the ASE default for molecules / nopbc MD) yields an empty neighbour list,
    # zero descriptors and therefore zero SGP variance on every atom — nothing
    # exceeds the threshold and only the bootstrap frame is selected. Work on
    # boxed copies; candidate_frames stays untouched for the DFT stage.
    deal_frames = candidate_frames
    boxed = any(frame.cell.rank < 3 for frame in candidate_frames)
    if boxed:
        deal_frames = add_vacuum_box(candidate_frames, cutoff)
        SparcLog(
            f"DEAL: frames have no cell — wrapped in {deal_frames[0].cell[0][0]:.1f} Å "
            "vacuum box for descriptor evaluation."
        )

    # Budget mode drops selected frames between passes and shuffles the rest, so
    # frame identity must travel with the frame instead of with its position.
    for i, frame in enumerate(deal_frames):
        frame.info["original_frame"] = int(frame.info.get("_step_index", i))

    # ── Optional per-atom candidate mask ──────────────────────────────────────
    mask_name = _prepare_mask(
        deal_frames, cfg, groups["mask"], api["TrajectoryMasker"], output_dir
    )

    # ── Build DEAL config objects ─────────────────────────────────────────────
    from ase.data import atomic_numbers as _ase_z

    species_z = [_ase_z[s] for s in species]

    # output_prefix directs DEAL to write deal_selected.xyz inside output_dir
    # instead of CWD. DEAL appends "_selected.xyz" to this prefix.
    deal_output_prefix = os.path.join(output_dir, "deal")

    data_cfg, deal_kwargs, sgp_cfg = _build_configs(
        api,
        cfg,
        groups,
        deal_frames,
        species_z,
        cutoff,
        is_periodic,
        mask_name,
        deal_output_prefix,
    )

    if groups["deal"]["save_gp"] or groups["deal"]["train_hyps"]:
        SparcLog(
            "DEAL: save_gp / train_hyps disable the local-uncertainty fast path "
            "(training labels, Kuf, posterior mean and QR are no longer skipped) — "
            "expect a slower run."
        )

    # ── Run DEAL ──────────────────────────────────────────────────────────────
    # DEAL verbose=True writes \r-overwriting progress to stdout. sys.stdout is
    # the SPARC Logger which tees to both Sparc.log and console — raw \r lines
    # produce ^M garbage in the log file. Route DEAL output to console only
    # (skips Sparc.log) and replace \r with \n so tail -f nohup.out is readable.
    from sparc.src.utils.logger import global_logger as _sparc_logger

    _orig_stdout = sys.stdout
    if _sparc_logger is not None:
        sys.stdout = _DealScreenPassthrough(_sparc_logger.console_output)
    try:
        if budget_mode:
            _run_budget_mode(api, data_cfg, deal_kwargs, sgp_cfg, cfg, groups["sparc"])
        else:
            _run_threshold_mode(api, data_cfg, deal_kwargs, sgp_cfg)
    finally:
        sys.stdout = _orig_stdout

    # ── Read selected frames from DEAL output ─────────────────────────────────
    # DEAL writes to output_prefix + "_selected.xyz" → output_dir/deal_selected.xyz.
    # frame.info metadata (_step_index, max_devi_f, _avg_devi_f) is preserved
    # through the xyz round-trip — no id() matching needed in labelling.py.
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    deal_output_xyz = os.path.join(output_dir, "deal_selected.xyz")
    if not os.path.exists(deal_output_xyz):
        SparcLog(
            f"WARNING [DEAL]: {deal_output_xyz} not found — returning all candidate frames."
        )
        return candidate_frames

    selected_frames = list(ase_read(deal_output_xyz, index=":"))
    rewrite = False

    # The final budget pass can overshoot the target; keep the first
    # max_selected frames — selection order is uncertainty-driven, so the
    # earliest selections are the most informative ones.
    if budget_mode and len(selected_frames) > cfg.max_selected:
        SparcLog(
            f"DEAL: trimming {len(selected_frames)} → {cfg.max_selected} frames "
            "(budget overshoot on the final pass)."
        )
        selected_frames = selected_frames[: cfg.max_selected]
        rewrite = True

    # Remove the artificial vacuum cell so candidates.extxyz (mirrored from this
    # file by labelling.py) and the DFT stage stay molecular.
    if boxed and selected_frames:
        strip_vacuum_box(selected_frames)
        rewrite = True

    if rewrite and selected_frames:
        ase_write(deal_output_xyz, selected_frames, format="extxyz")

    # Diagnostic: all-zero SGP variance means the descriptors carried no
    # information (empty neighbour list, wrong species map, cutoff too small).
    max_unc = [
        float(f.info["max_atomic_uncertainty"])
        for f in selected_frames
        if "max_atomic_uncertainty" in f.info
    ]
    if max_unc and max(max_unc) == 0.0:
        SparcLog(
            "WARNING [DEAL]: every selected frame reports max_atomic_uncertainty = 0. "
            "The SGP saw no atomic neighbours — check deal.cutoff and the frame cell. "
            "Only the bootstrap frame can be selected in this state."
        )

    SparcLog(
        f"DEAL: selected {len(selected_frames)} / {len(candidate_frames)} diverse frames."
    )
    return selected_frames

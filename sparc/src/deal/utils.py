#!/usr/bin/python3
# sparc/src/deal/utils.py

"""
Utility helpers for the DEAL integration.

- _extract_rcut_from_deepmd_json : parse DeepMD input.json for rcut
- _get_species                   : auto-detect element list from structure/frames
- attach_dummy_calc              : add placeholder SinglePointCalculator so
                                   DEAL's _extract_dft() doesn't crash on
                                   pre-DFT candidate frames
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from sparc.src.utils.logger import SparcLog

_DEAL_DEFAULT_CUTOFF = 5.0  # Å — fallback when rcut cannot be parsed


def _extract_rcut_from_deepmd_json(input_json_path: str) -> float:
    """
    Recursively find cutoff radius in a DeepMD or MACE input JSON and return max.

    Handles:
    - DeepMD se_e2_a / se_e3  : model.descriptor.rcut
    - DeepMD DPA-2 / DPA-3    : model.descriptor.repinit.rcut (nested)
    - MACE                     : r_max (top-level or nested)
    - Any future layout        : recursive search covers all cases
    """
    # Keys recognised as cutoff radius across different MLIP codes
    _CUTOFF_KEYS = {"rcut", "r_max"}

    path = Path(input_json_path)
    if not path.exists():
        SparcLog(
            f"WARNING [DEAL]: input JSON not found at '{input_json_path}'. "
            f"Using default FLARE cutoff = {_DEAL_DEFAULT_CUTOFF} Å. "
            "Set mlip_setup.input_file correctly or add deal.cutoff explicitly."
        )
        return _DEAL_DEFAULT_CUTOFF

    def _collect(obj: object, found: list) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _CUTOFF_KEYS and isinstance(v, (int, float)):
                    found.append(float(v))
                else:
                    _collect(v, found)
        elif isinstance(obj, list):
            for item in obj:
                _collect(item, found)

    try:
        with open(path) as f:
            data = json.load(f)
        rcut_values: list[float] = []
        _collect(data, rcut_values)
        if not rcut_values:
            raise ValueError("no 'rcut' or 'r_max' key found in JSON")
        cutoff = max(rcut_values)
        SparcLog(f"DEAL: auto-detected cutoff = {cutoff} Å from {path.name}")
        return cutoff
    except Exception as e:
        SparcLog(
            f"WARNING [DEAL]: Could not parse cutoff from '{input_json_path}' ({e}). "
            f"Using default FLARE cutoff = {_DEAL_DEFAULT_CUTOFF} Å."
        )
        return _DEAL_DEFAULT_CUTOFF


def _get_species(structure_file: str, candidate_frames: List[Atoms]) -> List[str]:
    """
    Return sorted unique element list for FLARE species map.

    Tries structure_file first (authoritative — covers elements not yet in
    candidate frames). Falls back to union of candidate frame symbols.
    """
    from ase.io import read as ase_read

    try:
        ref = ase_read(structure_file, index=0)
        return sorted(set(ref.get_chemical_symbols()))
    except Exception:
        pass

    symbols: set[str] = set()
    for frame in candidate_frames:
        symbols.update(frame.get_chemical_symbols())
    return sorted(symbols)


def attach_dummy_calc(frames: List[Atoms], periodic: bool = False) -> None:
    """
    Attach a zeroed SinglePointCalculator to frames that lack one.

    DEAL's _extract_dft() calls atoms.calc.results['forces'] (and energy/stress
    when force_only=False for periodic systems). Dummy zeros satisfy the
    interface — DEAL selects on SGP variance, not the actual DFT values.
    """
    for frame in frames:
        if frame.calc is None:
            n = len(frame)
            kwargs = dict(energy=0.0, forces=np.zeros((n, 3)))
            if periodic:
                kwargs["stress"] = np.zeros(6)
            frame.calc = SinglePointCalculator(frame, **kwargs)

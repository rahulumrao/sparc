"""
Tests for the DEAL >= 3.0 adapter (sparc/src/deal/filter.py).

DEAL itself is not a test dependency: a stub package is injected into
sys.modules so the adapter's config building, advanced-key routing, mask
handling and the two selection modes can be exercised without the native
sparse-GP extension.

Covers:
- deal.advanced routing to DEALConfig / SGPConfig / masker / SPARC options
- rejection of unknown and SPARC-reserved advanced keys
- threshold mode vs budget mode dispatch and threshold decay
- graceful fallback when DEAL is missing or too old
- mask disabled (with warning) when frames lack the per-atom array
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from ase.build import molecule
from ase.io import write as ase_write
from sparc.src.deal import filter as deal_filter
from sparc.src.utils.read_input import DEALConfig, ValidationError

# ============================================================
# Stub DEAL package
# ============================================================


@dataclass
class StubDataConfig:
    images: Optional[List[Any]] = None
    seed: int = 24


@dataclass
class StubDEALConfig:
    threshold: float = 1.0
    update_threshold: Optional[float] = None
    max_atoms_added: Any = 0.2
    min_steps_with_model: int = 0
    mask: Any = None
    initial_atoms: Any = None
    force_only: bool = True
    train_hyps: bool = False
    output_prefix: str = "deal"
    verbose: Any = False
    save_gp: bool = False
    save_full_trajectory: bool = False


@dataclass
class StubSGPConfig:
    kernels: List[Dict[str, Any]] = field(default_factory=list)
    descriptors: List[Dict[str, Any]] = field(default_factory=list)
    species: Optional[List[int]] = None
    cutoff: float = 4.5
    variance_type: str = "local"


class StubDEAL:
    """Records every pass and writes the frames it is given as 'selected'."""

    runs: List[Dict[str, Any]] = []

    def __init__(self, data_cfg, deal_cfg, sgp_cfg):
        self.data_cfg = data_cfg
        self.deal_cfg = deal_cfg
        self.sgp_cfg = sgp_cfg
        self.selected_frames = []
        self.rebuilds = 0

    def configure_run(self, data_cfg=None, deal_cfg=None, sgp_cfg=None):
        if data_cfg is not None:
            self.data_cfg = data_cfg
        if deal_cfg is not None:
            self.deal_cfg = deal_cfg
        if sgp_cfg is not None:
            self.sgp_cfg = sgp_cfg
            self.selected_frames = []
        self.rebuilds += 1

    def run(self):
        # One frame per pass, so budget mode needs several passes to fill up.
        pool = list(self.data_cfg.images or [])
        picked = pool[:1]
        StubDEAL.runs.append(
            {
                "threshold": self.deal_cfg.threshold,
                "pool": len(pool),
                "mask": self.deal_cfg.mask,
                "deal_cfg": self.deal_cfg,
                "sgp_cfg": self.sgp_cfg,
            }
        )
        self.selected_frames.extend(picked)
        existing = list(self._read_existing())
        ase_write(
            f"{self.deal_cfg.output_prefix}_selected.xyz",
            existing + picked,
            format="extxyz",
        )

    def _read_existing(self):
        import os

        from ase.io import read as ase_read

        path = f"{self.deal_cfg.output_prefix}_selected.xyz"
        if not os.path.exists(path):
            return []
        return ase_read(path, index=":")


class StubMaskSummary:
    def __init__(self, n_frames, n_atoms, n_selected, n_frames_with_selection):
        self.n_frames = n_frames
        self.n_atoms = n_atoms
        self.n_selected_atoms = n_selected
        self.n_frames_with_selection = n_frames_with_selection
        self.lower_threshold = None
        self.upper_threshold = None

    @property
    def selected_fraction(self):
        return self.n_selected_atoms / self.n_atoms if self.n_atoms else 0.0


class StubTrajectoryMasker:
    calls: List[Dict[str, Any]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        StubTrajectoryMasker.calls.append(kwargs)

    def apply_to_trajectory(self, images):
        images = list(images)
        n_atoms = 0
        n_selected = 0
        for atoms in images:
            values = np.asarray(atoms.arrays[self.kwargs["key"]], dtype=float)
            mask = values > (self.kwargs["mask_threshold"] or 0.0)
            atoms.arrays[self.kwargs["mask_key"]] = mask.astype(int)
            n_atoms += len(atoms)
            n_selected += int(mask.sum())
        return StubMaskSummary(len(images), n_atoms, n_selected, len(images))


@pytest.fixture
def stub_deal(monkeypatch):
    """Install a fake `deal` package and report version 3.0.0."""
    StubDEAL.runs = []
    StubTrajectoryMasker.calls = []

    pkg = types.ModuleType("deal")
    config = types.ModuleType("deal.config")
    config.DataConfig = StubDataConfig
    config.DEALConfig = StubDEALConfig
    config.SGPConfig = StubSGPConfig
    core = types.ModuleType("deal.core")
    core.DEAL = StubDEAL
    preprocessing = types.ModuleType("deal.preprocessing")
    preprocessing.TrajectoryMasker = StubTrajectoryMasker

    for name, mod in {
        "deal": pkg,
        "deal.config": config,
        "deal.core": core,
        "deal.preprocessing": preprocessing,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(deal_filter, "_installed_version", lambda: ("3.0.0", (3, 0, 0)))
    return StubDEAL


@pytest.fixture
def frames():
    out = []
    for i in range(6):
        atoms = molecule("H2O")
        atoms.set_cell([9, 9, 9])
        atoms.set_pbc(True)
        atoms.info["_step_index"] = i * 10
        out.append(atoms)
    return out


def _run(frames, cfg, tmp_path, **kwargs):
    return deal_filter.run_deal_filter(
        candidate_frames=frames,
        sparc_deal_cfg=cfg,
        is_periodic=kwargs.get("is_periodic", True),
        structure_file=kwargs.get("structure_file", ""),
        deepmd_input_json=kwargs.get("deepmd_input_json", ""),
        output_dir=str(tmp_path),
    )


# ============================================================
# advanced-key routing
# ============================================================


def test_advanced_defaults_are_complete():
    groups = deal_filter._split_advanced({})
    assert groups["deal"]["max_atoms_added"] == 0.2
    assert groups["descriptor"]["nmax"] == 8
    assert groups["kernel"]["sigma"] == 2.0
    assert groups["sgp"]["variance_type"] == "local"
    assert groups["sparc"]["threshold_factor"] == 0.7


def test_advanced_routes_each_key_to_its_group():
    groups = deal_filter._split_advanced(
        {
            "max_atoms_added": 5,
            "lmax": 6,
            "power": 3,
            "variance_type": "DTC",
            "mask_mode": "between",
            "max_iterations": 3,
        }
    )
    assert groups["deal"]["max_atoms_added"] == 5
    assert groups["descriptor"]["lmax"] == 6
    assert groups["kernel"]["power"] == 3
    assert groups["sgp"]["variance_type"] == "DTC"
    assert groups["mask"]["mask_mode"] == "between"
    assert groups["sparc"]["max_iterations"] == 3


def test_advanced_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unknown option 'deal.advanced.nmaxx'"):
        deal_filter._split_advanced({"nmaxx": 8})


@pytest.mark.parametrize("key", ["threshold", "max_selected", "cutoff", "mask"])
def test_advanced_rejects_reserved_keys(key):
    with pytest.raises(ValueError, match="controlled by SPARC"):
        deal_filter._split_advanced({key: 1})


# ============================================================
# config construction
# ============================================================


def test_threshold_mode_builds_expected_configs(stub_deal, frames, tmp_path):
    cfg = DEALConfig(
        enabled=True,
        threshold=0.02,
        cutoff=6.0,
        advanced={"nmax": 10, "sigma": 1.5, "save_full_trajectory": True},
    )
    selected = _run(frames, cfg, tmp_path)

    assert len(stub_deal.runs) == 1
    run = stub_deal.runs[0]
    assert run["threshold"] == 0.02
    assert run["mask"] is None
    assert run["deal_cfg"].force_only is False  # periodic → energies and stress
    assert run["deal_cfg"].save_full_trajectory is True
    assert run["deal_cfg"].min_steps_with_model == 1  # SPARC default, not DEAL's 0
    assert run["sgp_cfg"].cutoff == 6.0
    assert run["sgp_cfg"].descriptors[0]["nmax"] == 10
    assert run["sgp_cfg"].descriptors[0]["name"] == "B2"
    assert run["sgp_cfg"].kernels[0]["sigma"] == 1.5
    assert run["sgp_cfg"].species == [1, 8]
    assert len(selected) == 1


def test_molecular_frames_use_force_only(stub_deal, tmp_path):
    mols = []
    for i in range(3):
        atoms = molecule("H2O")  # no cell
        atoms.info["_step_index"] = i
        mols.append(atoms)

    cfg = DEALConfig(enabled=True, cutoff=4.0)
    selected = _run(mols, cfg, tmp_path, is_periodic=False)

    assert stub_deal.runs[0]["deal_cfg"].force_only is True
    # The vacuum box added for descriptor evaluation must not leak downstream.
    assert all(frame.cell.rank == 0 for frame in selected)


# ============================================================
# budget mode
# ============================================================


def test_budget_mode_decays_threshold_and_stops_at_target(stub_deal, frames, tmp_path):
    cfg = DEALConfig(
        enabled=True,
        max_selected=3,
        cutoff=5.0,
        advanced={"threshold_factor": 0.5, "max_iterations": 10},
    )
    selected = _run(frames, cfg, tmp_path)

    # Stub selects one frame per pass → exactly max_selected passes.
    assert len(stub_deal.runs) == 3
    assert [r["threshold"] for r in stub_deal.runs] == [0.5, 0.25, 0.125]
    # Selected frames leave the pool between passes.
    assert [r["pool"] for r in stub_deal.runs] == [6, 5, 4]
    assert len(selected) == 3


def test_budget_mode_stops_at_max_iterations(stub_deal, frames, tmp_path):
    cfg = DEALConfig(
        enabled=True,
        max_selected=5,
        cutoff=5.0,
        advanced={"max_iterations": 2},
    )
    selected = _run(frames, cfg, tmp_path)

    assert len(stub_deal.runs) == 2
    assert len(selected) == 2


def test_frames_carry_original_frame_index(stub_deal, frames, tmp_path):
    cfg = DEALConfig(enabled=True, max_selected=2, cutoff=5.0)
    _run(frames, cfg, tmp_path)
    assert [f.info["original_frame"] for f in frames] == [0, 10, 20, 30, 40, 50]


# ============================================================
# mask
# ============================================================


def test_mask_disabled_when_array_missing(stub_deal, frames, tmp_path):
    cfg = DEALConfig(enabled=True, mask=True, cutoff=5.0)
    _run(frames, cfg, tmp_path)

    assert StubTrajectoryMasker.calls == []
    assert stub_deal.runs[0]["mask"] is None


def test_mask_used_when_array_present(stub_deal, frames, tmp_path):
    for atoms in frames:
        atoms.arrays["per_atom_devi_f"] = np.array([0.30, 0.01, 0.02])

    cfg = DEALConfig(
        enabled=True, mask=True, cutoff=5.0, advanced={"mask_threshold": 0.1}
    )
    _run(frames, cfg, tmp_path)

    assert StubTrajectoryMasker.calls[0]["key"] == "per_atom_devi_f"
    assert StubTrajectoryMasker.calls[0]["mask_threshold"] == 0.1
    assert stub_deal.runs[0]["mask"] == "deal_mask"


def test_mask_accepts_custom_array_name(stub_deal, frames, tmp_path):
    for atoms in frames:
        atoms.arrays["my_uncertainty"] = np.array([0.30, 0.01, 0.02])

    cfg = DEALConfig(enabled=True, mask="my_uncertainty", cutoff=5.0)
    _run(frames, cfg, tmp_path)

    assert StubTrajectoryMasker.calls[0]["key"] == "my_uncertainty"


# ============================================================
# fallback behaviour
# ============================================================


def test_missing_deal_returns_frames_unchanged(monkeypatch, frames, tmp_path):
    monkeypatch.setattr(deal_filter, "_import_deal", lambda: None)
    cfg = DEALConfig(enabled=True)
    assert _run(frames, cfg, tmp_path) is frames


def test_old_deal_version_is_refused(monkeypatch, frames, tmp_path):
    monkeypatch.setattr(deal_filter, "_installed_version", lambda: ("2.1.0", (2, 1, 0)))
    cfg = DEALConfig(enabled=True)
    assert deal_filter._import_deal() is None
    assert _run(frames, cfg, tmp_path) is frames


def test_empty_candidate_list(stub_deal, tmp_path):
    cfg = DEALConfig(enabled=True)
    assert _run([], cfg, tmp_path) == []


# ============================================================
# input.yaml validation
# ============================================================


def test_minimal_config_is_one_key():
    cfg = DEALConfig(enabled=True)
    assert cfg.threshold == 0.01
    assert cfg.max_selected is None
    assert cfg.mask is False
    assert cfg.advanced == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.0},
        {"cutoff": -1.0},
        {"max_selected": 0},
        {"max_selected": 1.5},
        {"mask": ""},
        {"advanced": []},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValidationError):
        DEALConfig(enabled=True, **kwargs)

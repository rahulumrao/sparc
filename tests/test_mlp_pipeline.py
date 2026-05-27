"""
Tests for DeepMD training pipeline (sparc/src/deepmd.py, sparc/src/data_processing.py).

Updated for v0.2: filesystem-first model detection (.pth vs .pb).
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.io import read


def _which(name: str) -> str | None:
    return shutil.which(name)


def test_deepmd_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    End-to-end test:
      ASE trajectory -> get_data -> DeepMD npy dataset
      DeepMD input.json -> deepmd_training -> frozen/compressed models

    Skips if DeepMD CLI is not available (dp).
    """
    dp_cli = _which("dp")
    if dp_cli is None:
        pytest.skip("'dp' (DeePMD-kit CLI) not found in PATH")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "data" / "mlp"

    traj_file = data_dir / "AseMD.traj"
    input_json = data_dir / "input.json"

    if not traj_file.exists() or not input_json.exists():
        pytest.skip("MLP test data not found")

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    run_traj = run_dir / "AseMD.traj"
    run_input = run_dir / "input.json"
    shutil.copy(traj_file, run_traj)
    shutil.copy(input_json, run_input)

    monkeypatch.chdir(run_dir)

    np.random.seed(12345)

    atoms0 = read(str(run_traj), index=0)
    seen = set()
    atom_types = []
    for s in atoms0.get_chemical_symbols():
        if s not in seen:
            atom_types.append(s)
            seen.add(s)

    dataset_dir = run_dir / "Dataset"
    training_dir = str(run_dir / "Training")

    with (
        open(os.devnull, "w") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        from sparc.src.data_processing import get_data

        get_data(
            ase_traj=str(run_traj),
            dir_name=str(dataset_dir),
            skip_min=0,
            skip_max=None,
        )

        from sparc.src.deepmd import deepmd_training

        frozen_model_name = deepmd_training(
            active_learning=False,
            datadir=str(dataset_dir),
            atom_types=atom_types,
            training_dir=training_dir,
            num_models=2,
            input_file=str(run_input),
        )

    # Assert dataset
    training_data = dataset_dir / "training_data"
    validation_data = dataset_dir / "validation_data"

    assert training_data.exists()
    assert validation_data.exists()
    assert any(training_data.rglob("*.npy"))
    assert any(validation_data.rglob("*.npy"))

    # Assert training
    train_root = Path(training_dir)
    assert train_root.exists()

    for i in (1, 2):
        model_dir = train_root / f"training_{i}"
        assert model_dir.exists()

        # v0.2: models can be .pth (PyTorch) or .pb (TensorFlow)
        frozen_pth = model_dir / f"frozen_model_{i}.pth"
        frozen_pb = model_dir / f"frozen_model_{i}.pb"
        assert frozen_pth.exists() or frozen_pb.exists(), (
            f"Missing frozen model in {model_dir}: "
            f"neither {frozen_pth.name} nor {frozen_pb.name} found"
        )

    assert isinstance(frozen_model_name, str)
    assert frozen_model_name.endswith(".pth") or frozen_model_name.endswith(".pb")

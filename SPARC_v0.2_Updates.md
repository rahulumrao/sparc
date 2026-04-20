# SPARC v0.2 — Updates and Current State

**Smart Potential with Atomistic Rare Events & Continuous Learning**

---

## Overview

SPARC is an active learning workflow for developing reactive machine learning potentials.
It orchestrates DFT calculations, ML model training, molecular dynamics simulations,
and iterative refinement through Query-by-Committee (QbC) candidate selection.

---

## Workflow Architecture

```
INPUT: YAML config → Load structures

SECTION 1: Ab Initio MD (AIMD)
├── DFT calculator setup (ORCA / VASP / CP2K / xTB)
├── Initialize thermostat (Nose-Hoover / Langevin / NPT Berendsen / NVE)
├── Temperature ramping (optional, linear VASP-style)
├── PLUMED enhanced sampling (optional)
└── Output: AIMD trajectory

SECTION 2: MLIP Training
├── Process AIMD trajectory → DeepMD npy format
├── Fine-tune universal model (DPA-1/2/3, MACE) [optional]
│   └── Or standard DeepMD from-scratch training
├── Train ensemble (N models with different random seeds)
└── Output: frozen_model_1.pth, frozen_model_2.pth, ...

SECTION 3: ML-MD Simulation
├── Load trained models via DeepPotential calculator
├── Initialize thermostat + temperature ramping
├── Run safety-checked MD (energy threshold, distance constraints)
├── Checkpoint save/restore for restart
├── Umbrella sampling (optional via PLUMED)
└── Output: ML-MD trajectory

SECTION 4: Active Learning Loop (iterative)
├── Step 1: DFT Labeling — compute energies/forces for candidates
├── Step 2: Retrain — combine all trajectories, retrain ensemble
├── Step 3: ML-MD — run MD with updated models
├── Step 4: Query-by-Committee — select new candidates
└── Break when no candidates found or max iterations reached
```

---

## New Features & Changes

### 1. Universal Model Fine-Tuning

**New module:** `sparc/src/finetune.py`

Fine-tune pre-trained universal ML potentials on system-specific DFT data instead of
training from scratch. Supports two backends:

| Backend | Command | Models | Output |
|---------|---------|--------|--------|
| DeePMD (DPA) | `dp --pt train --finetune model.pt` | DPA-1, DPA-2, DPA-3 | `.pth` |
| MACE | `mace_run_train --foundation_model` | small, medium, large, custom `.model` | `.model` |

**Key functions:**
- `deepmd_finetune()` — Fine-tune DeePMD models with `--model-branch` support for multi-task models (e.g., DPA-3.2-5M with branches like `Omat24`, `Organic_Reactions`)
- `mace_finetune()` — Fine-tune MACE foundation models with automatic data conversion (DeepMD npy → extxyz)
- `finetune_training()` — Dispatcher that routes to the correct backend
- `setup_MACE_calculator()` — ASE calculator setup for MACE models
- `_convert_deepmd_to_extxyz()` — Data format converter with empty file detection

**Configuration** (top-level in YAML):
```yaml
finetune:
  enabled: True
  model_type: "deepmd"              # "deepmd" or "mace"
  pretrained_model: "DPA-3.2-5M.pt" # Model file or MACE name (small/medium/large)
  model_branch: "Omat24"            # Multi-task branch (DPA-3 only)
  input_file: "input_finetune.json" # Fine-tune JSON config (separate from training JSON)
  learning_rate: 0.001
  batch_size: 4                     # MACE only
  device: "cpu"                     # MACE only
```

**Error diagnostics:**
- Version mismatch detection (e.g., DPA-3 model requiring newer DeePMD-kit)
- Missing dependency detection with environment activation guidance
- Live training output streaming to console

---

### 2. Consolidated Candidate Trajectory

**Changed module:** `sparc/src/labelling.py`

Previously, each candidate structure was written to an individual folder
(`0001/input.extxyz`, `0002/input.extxyz`, ...), causing file sprawl.

**Now:** All accepted candidates are written to a single `candidates.extxyz` trajectory file.

- `labelling()` returns `(candidate_found, candidates_file, n_candidates)` instead of `(candidate_found, labelled_files)`
- DFT labeling reads frames with `read(candidates_file, index=':')`
- Restart recovery counts frames from the single file

---

### 3. ML-MD Checkpoint/Restart

**Changed module:** `sparc/src/ase_md.py`

`ExecuteMlpDynamics` now supports resume from checkpoint:

```python
def ExecuteMlpDynamics(..., restart: bool = False):
```

- Checkpoint file (`md_checkpoint.pkl`) is saved inside the simulation directory (e.g., `02.dpmd/`)
- State includes: positions, velocities, momenta, cell, stress, step count
- On restart, loads checkpoint, sets `dyn.nsteps`, and runs remaining steps
- Checkpoint saved periodically via `dyn.attach()` and at simulation end

**Configuration:**
```yaml
mlip_setup:
  restart: False    # Resume ML-MD from checkpoint
```

---

### 4. Temperature Ramping

**Changed module:** `sparc/src/ase_md.py`

Linear temperature ramping (VASP-style):

```
T(t) = T_start + (T_end - T_start) × (t / t_total)
```

- Works with NVT ensemble only (NVE raises error, NPT prints warning)
- Scales velocities and updates thermostat target temperature
- Supports both Langevin and Nose-Hoover (with compatibility warning)

**Configuration:**
```yaml
aimd_setup:          # or mlip_setup
  temperature: 300   # T_start
  temp_end: 600      # T_end (null = no ramping)
```

---

### 5. Backend Mismatch Detection

**Changed module:** `sparc/src/utils/utils.py`

New function `check_backend_mismatch()` validates that model file format matches
the installed DeePMD backend:

- `.pth` model + TensorFlow backend → `RuntimeError` with installation guidance
- `.pb` model + PyTorch backend → `RuntimeError` with installation guidance

Called from `setup_DeepPotential()` before loading the model.

---

### 6. Filesystem-First Model Detection

**Changed modules:** `sparc/sparc.py`, `sparc/src/deepmd.py`

Model detection no longer relies solely on `get_version()` backend detection.
Instead, it checks the filesystem first:

```python
if os.path.exists("frozen_model_1.pth"):
    dp_model = "frozen_model_1.pth"
elif os.path.exists("frozen_model_1.pb"):
    dp_model = "frozen_model_1.pb"
else:
    # Fall back to backend detection
```

This prevents `FileNotFoundError` when the backend default doesn't match actual model files.

---

### 7. MDLogger Fix (PLUMED Compatibility)

**Changed module:** `sparc/src/ase_md.py`

ASE's built-in `MDLogger` crashed with PLUMED because `Plumed.compute_bias()` returns
`energy_bias = np.zeros((1,))` — a shape `(1,)` array that propagates to `results['energy']`,
causing `TypeError: only 0-dimensional arrays can be converted to Python scalars` in
`MDLogger`'s `%` formatting.

**Fix:** Removed ASE's `MDLogger` entirely from all three MD execution functions.
The existing `log_md_setup()` in `utils.py` already handles array→scalar conversion
properly via `float()` casting.

---

### 8. Fine-Tuning Configuration (Top-Level)

**Changed module:** `sparc/src/utils/read_input.py`

`FineTuneConfig` was moved from inside `MLIPSetupConfig` to a top-level section
in `SparcConfig`, accessed as `config.finetune` instead of `config.mlip_setup.finetune`.

This avoids YAML parsing conflicts and keeps fine-tuning as an independent concern.

**Dataclass:**
```python
@dataclass
class FineTuneConfig:
    enabled: bool = False
    model_type: Literal["deepmd", "mace"] = "deepmd"
    pretrained_model: str = "DPA3.pt"
    model_branch: Optional[str] = "Omat24"
    input_file: Optional[str] = None
    num_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 4
    device: str = "cpu"
```

---

## File Reference

| File | Purpose |
|------|---------|
| `sparc/sparc.py` | Main workflow orchestrator (4-section pipeline + AL loop) |
| `sparc/src/finetune.py` | **NEW** — Universal model fine-tuning (DeePMD/DPA + MACE) |
| `sparc/src/deepmd.py` | DeePMD-kit v2/v3 integration (training, freezing, evaluation) |
| `sparc/src/ase_md.py` | MD engines (NVE/NVT/NPT), temperature ramping, checkpointing |
| `sparc/src/labelling.py` | Candidate selection with RMSD filtering |
| `sparc/src/active_learning.py` | Query-by-Committee model deviation analysis |
| `sparc/src/calculator.py` | DFT calculator setup (ORCA, VASP, CP2K, xTB) |
| `sparc/src/plumed_wrapper.py` | PLUMED enhanced sampling and umbrella sampling |
| `sparc/src/data_processing.py` | Trajectory → DeepMD npy data conversion |
| `sparc/src/utils/read_input.py` | YAML config parsing with dataclass validation |
| `sparc/src/utils/utils.py` | Logging, checkpointing, trajectory management, safety checks |
| `sparc/src/utils/logger.py` | SparcLog logging system |
| `sparc/src/utils/banner.py` | Startup banner |
| `scripts/input.yaml` | Example YAML configuration |

---

## Configuration Reference

### Example YAML (Fine-Tuning with Active Learning)

```yaml
general:
  structure_file: "system.xyz"

dft_calculator:
  engine: "xTB"
  template_file: "xtb_template.inp"
  exe_command: "xtb"

aimd_setup:
  ensemble: "NVT"
  thermostat:
    type: "Langevin"
    friction: 0.15
  temperature: 300
  timestep_fs: 1.0
  steps: 500
  log_frequency: 2
  restart: false
  plumed:
    enabled: False

mlip_setup:
  training: True
  data_dir: "DeePMD_training/00.data"
  input_file: "input.json"            # Standard training JSON
  num_models: 2
  MdSimulation: True
  restart: False
  ensemble: "NVT"
  temperature: 300.0
  thermostat:
    type: "Langevin"
    friction: 0.15
  timestep_fs: 1.0
  md_steps: 50000
  log_frequency: 40
  plumed:
    enabled: False

finetune:
  enabled: True
  model_type: "deepmd"
  pretrained_model: "DPA-3.2-5M.pt"
  model_branch: "Omat24"
  input_file: "input_finetune.json"   # Fine-tune specific JSON
  learning_rate: 0.001

active_learning: True
learning_restart: False
iteration: 5
model_dev:
  f_min_dev: 0.1
  f_max_dev: 0.3

output:
  log_file: "AseMD.log"
  xyz_file: "AseTraj.xyz"
  aimdtraj_file: "AseMD.traj"
  dptraj_file: "dpmd.traj"
```

---

## Supported Pre-Trained Models

### DeePMD (DPA) — Full AL Support
| Model | Source | Branch | Compatible |
|-------|--------|--------|------------|
| DPA-2 | [AIS Square](https://www.aissquare.com/models) | N/A | DeePMD-kit ≥ v3.0 |
| DPA-3.2-5M | [AIS Square](https://www.aissquare.com/models) | Omat24, Organic_Reactions, ... | DeePMD-kit ≥ v3.1 |

### MACE — Fine-Tuning Only (AL not yet wired)
| Model | Source | Notes |
|-------|--------|-------|
| MACE-MP-0 small/medium/large | Built-in names | General materials |
| MACE-MH-0 | [mace-foundations](https://github.com/ACEsuit/mace-foundations) | Materials + hydrogen |

> **Note:** MACE fine-tuning produces `.model` files but the AL loop (ML-MD, QbC via `dp model-devi`)
> is currently wired for DeePMD only. Full MACE AL support requires Python-level ensemble deviation.

---

## Dependencies

- **DeePMD-kit** v3.0+ (v3.1+ for DPA-3 models) with PyTorch backend
- **ASE** (Atomic Simulation Environment)
- **dpdata** for data format conversion
- **PLUMED** (optional, for enhanced sampling)
- **mace-torch** (optional, for MACE fine-tuning)

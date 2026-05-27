# SPARC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/rahulumrao/sparc?style=social)](https://github.com/rahulumrao/sparc/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/rahulumrao/sparc?style=social)](https://github.com/rahulumrao/sparc/network/members)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation Status](https://readthedocs.org/projects/sparc/badge/)](https://docs-sparc.readthedocs.io/en/latest/)
[![CI](https://github.com/rahulumrao/sparc/actions/workflows/ci.yaml/badge.svg)](https://github.com/rahulumrao/sparc/actions)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.09468/status.svg)](https://doi.org/10.21105/joss.09468)

**S**mart **P**otential with **A**tomistic **R**are Events and **C**ontinuous Learning

<!-- ![Alt text](docs/source/sparc_logo.png) -->
<img src="docs/source/sparc_logo.png" alt="drawing" style="width:300px;"/>

For More Information, Please Visit [SPARC Documentation](https://docs-sparc.readthedocs.io/en/latest/).

Try SPARC [Tutorial](https://github.com/rahulumrao/sparc/blob/main/examples/SPARC_Tutorial.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahulumrao/sparc/blob/main/examples/SPARC_Tutorial.ipynb)


## Overview

SPARC is a Python package built around the `ASE` wrapper that implements an automated workflow for developing machine learning interatomic potentials (MLIPs) for reactive chemical systems. It automates the process of identifying new structures in the configurational space through active learning, eliminating the need for long _ab initio_ MD simulations upfront.

## Key Features

- Automated active learning workflow (Query-by-Committee)
- _Ab initio_ molecular dynamics (AIMD) with **VASP**, **CP2K**, **ORCA**, **QE**, **xTB**, and **Gaussian**
- NVE, NVT (Nose-Hoover / Langevin), and NPT (Berendsen) ensembles
- ML potential training with [DeepMD-kit v3](https://github.com/deepmodeling/deepmd-kit) (PyTorch backend)
- Fine-tuning of universal ML potentials (**DPA-2**, **DPA-3**)
- ML/MD simulations and iterative model refinement
- Force deviation monitoring and Query-by-Committee candidate selection
- Reactive trajectory generation with [PLUMED](https://www.plumed.org/) integration (Metadynamics, Umbrella Sampling, etc.)

## Requirements

### Core Dependencies
- Python 3.10
- [DeepMD-kit v3+](https://github.com/deepmodeling/deepmd-kit) (PyTorch backend)
- [ASE](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment)
- [dpdata](https://github.com/deepmodeling/dpdata)

### DFT Engines (one or more required)
- [VASP](https://www.vasp.at/) — periodic systems
- [CP2K](https://www.cp2k.org/) — periodic systems
- [ORCA](https://www.faccts.de/orca/) — molecular systems
- [Quantum ESPRESSO](https://www.quantum-espresso.org/) — periodic systems
- [xTB](https://github.com/grimme-lab/xtb) — semi-empirical
- [Gaussian](https://gaussian.com/) — molecular systems


### Python Package Dependencies
- numpy, pandas, dpdata, scipy

## Installation

#### 1. Create and activate a conda environment

```bash
conda create -n sparc python=3.10
conda activate sparc
```

#### 2. Install DeepMD-kit v3 (PyTorch backend)

```bash
pip install deepmd-kit[torch]
```

For GPU support:
```bash
pip install deepmd-kit[torch,cu12]
```

#### 3. Clone and install SPARC

```bash
git clone https://github.com/rahulumrao/sparc.git
cd sparc
pip install .
```

#### 4. Install PLUMED (optional)

For standard CVs:
```bash
conda install -c conda-forge py-plumed
```

For advanced CVs (e.g., SPRINT), build from source with all modules enabled:
```bash
./configure --enable-mpi=no --enable-modules=all PYTHON_BIN=$(which python) --prefix=$CONDA_PREFIX
make -j$(nproc) && make install
```
> Refer to the official [PLUMED installation page](https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html) for more details.

#### 5. Install [DeePMD-GNN](https://github.com/deepmodeling/deepmd-gnn) plugin for [MACE](https://github.com/ACEsuit/mace) and [NeQUIP](https://github.com/mir-group/nequip) support (Optional)

```bash
conda install deepmd-gnn -c conda-forge

export CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)'):$CONDA_PREFIX"

conda install -c nvidia cuda-toolkit=12.1 cuda-nvcc=12.1 cuda-nvtx=12.1 cuda-nvrtc=12.1 cuda-cudart-dev=12.1 cuda-cupti=12.1 -y

conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

> Refer to the official [DeePMD-GNN](https://docs.deepmodeling.com/projects/deepmd/en/stable/third-party/out-of-deepmd-kit.html) installation page.

#### 6. Install xTB (optional)

See [xtb-python installation docs](https://xtb-python.readthedocs.io/en/latest/installation.html#installation).

```bash
conda config --add channels conda-forge
conda install xtb-python
```

## Environment Setup

```bash
export VASP_PP_PATH=/path/to/vasp/potcar_files    # VASP POTCAR path (VASP only)
```

If PLUMED was installed from source:
```bash
export PLUMED_KERNEL="$CONDA_PREFIX/lib/libplumedKernel.so"
export PYTHONPATH="$CONDA_PREFIX/lib/plumed/python:$PYTHONPATH"
```

## Quick Start

Prepare an `input.yaml` (see `examples/` for full templates):

```yaml
general:
  structure_file: "POSCAR"

dft_calculator:
  engine: "VASP"
  template_file: "INCAR"

aimd_setup:
  ensemble: "NVT"
  temperature: 300
  timestep_fs: 1.0
  steps: 500
  thermostat:
    type: "Nose"
    tdamp: 2.0

mlip_setup:
  training: true
  data_dir: "Training_Data"
  input_file: "input.json"
  num_models: 4
  MdSimulation: true
  ensemble: "NVT"
  temperature: 300
  timestep_fs: 1.0
  md_steps: 2000
  train_ratio: 0.8            # 80% training / 20% validation

finetune:                     # optional: fine-tune a universal model
  enabled: false
  model_type: "deepmd"        # "deepmd" or "mace"
  pretrained_model: "DPA3.pt"
  model_branch: "Omat24"

active_learning: true
iteration: 10
model_dev:
  f_min_dev: 0.05
  f_max_dev: 0.30
```

Run SPARC:
```bash
sparc -i input.yaml
```

## Directory Structure

```
Project Root/
├── POSCAR / input.xyz        (structure file)
├── INCAR                     (DFT template)
├── input.json                (DeepMD training input)
├── input.yaml                (SPARC input)
├── Training_Data/
│   ├── training_data/
│   └── validation_data/
├── iter_000000/
│   ├── 00.dft/               (DFT / AIMD labelling)
│   ├── 01.train/             (MLIP training or fine-tuning)
│   │   ├── training_1/
│   │   └── training_2/
│   └── 02.dpmd/              (ML-MD + model deviation)
├── iter_000001/
│   └── ...
```

## Core Components

### 1. MD Simulation
- NVE, NVT (Nose-Hoover / Langevin), and NPT (Berendsen) ensembles
- _Ab initio_ and ML molecular dynamics via the same ASE interface
- Checkpoint/restart capabilities
- PLUMED integration for accelerated sampling (Metadynamics, Umbrella Sampling)

### 2. MLIP Training
- Automated model training with DeepMD-kit v3 (PyTorch)
- Ensemble model generation for Query-by-Committee
- **Fine-tuning** of universal potentials (DPA-3, MACE-MP) from a pre-trained checkpoint

### 3. Active Learning
- Query-by-Committee for candidate selection based on force deviation
- RMSD-based duplicate filtering
- Automated structure labelling and retraining loop
- `fparam` support for universal models (DPA-3)

## Current Status

- Fixed model update in active learning iterations restart with added keys:
  - `learning_restart: True`
  - `latest_model: 'path/to/frozen_model.pb'`
- Structured log formatting for better readability
- Implemented Umbrella Sampling for reaction study on-the-fly
- Utility tools for analysing model accuracy, active learning status, and structural properties

## Planned Updates

- Support for ORCA, Psi4 and xTB calculators
- Documentation under development

## Known Issues

> [!IMPORTANT]
> Some hardware configurations show issues with `conda` channels when CUDA is not detected:
> ```
> LibMambaUnsatisfiableError: Encountered problems while solving:
>  - nothing provides __cuda needed by libdeepmd-2.2.10-0_cuda10.2_gpu
> ```
> Check [DeePMD documentation](https://deepmd-kit.readthedocs.io/en/latest/install/easy-install.html) for installation guidance.

## Build Document Locally

```bash
pip install sphinx sphinx-autodoc-typehints sphinx_rtd_theme nbsphinx

cd docs/
make html
```

This creates an `html` file in the `build/` folder. Open `build/html/index.html` in any browser.

## License

This project is licensed under the [MIT License](./LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Code Style and Linting

We use [`ruff`](https://docs.astral.sh/ruff/) and [`pre-commit`](https://pre-commit.com/) for code styling and linting. Configurations are defined in [`pyproject.toml`](pyproject.toml) and [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

```bash
pip install ruff pre-commit
```

Run all hooks:

```bash
pre-commit run --all-files
```

## Citation

If you use this software or the dataset in your research, please cite:

```bibtex
@article{joss,
  author  = {Verma, Rahul and Joshi, Nisarg and Pfaendtner, Jim},
  title   = {{SPARC}: An Automated Workflow Toolkit for Accelerated Active Learning of Reactive Machine Learning Interatomic Potentials},
  journal = {Journal of Open Source Software},
  volume  = {11},
  number  = {120},
  pages   = {9468},
  year    = {2026},
  month   = {apr},
  doi     = {10.21105/joss.09468},
  url     = {https://doi.org/10.21105/joss.09468}
}

@software{sparc,
  author  = {Verma, Rahul and Joshi, Nisarg and Pfaendtner, Jim},
  doi     = {https://doi.org/10.5281/zenodo.19389278},
  license = {MIT},
  month   = {Apr},
  title   = {{SPARC}: An Automated Workflow Toolkit for Accelerated Active Learning of Reactive Machine Learning Interatomic Potentials},
  url     = {https://github.com/rahulumrao/sparc},
  year    = {2026}
}

@dataset{sparc_dataset,
  author  = {Verma, Rahul and Joshi, Nisarg and Pfaendtner, Jim},
  doi     = {https://doi.org/10.5281/zenodo.18261342},
  license = {MIT},
  month   = {jan},
  title   = {{SPARC}: An Automated Workflow Toolkit for Accelerated Active Learning of Reactive Machine Learning Interatomic Potentials},
  url     = {https://zenodo.org/records/18261342},
  year    = {2026}
}
```

---
> [!WARNING]
> This package is under active development. Features and APIs may change. \
> Also, this code is designed to work in a Linux environment. It may not be fully compatible with macOS systems.

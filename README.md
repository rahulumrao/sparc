# SPARC

**S**mart **P**otential with **A**tomistic **R**are Events and **C**ontinuous Learning

## Overview

SPARC is an open-source Python package that implements an active learning workflow for developing accurate machine learning potentials. It automates the process of running quantum mechanical calculations, training ML models, and identifying new structures that need quantum mechanical labeling.

## Key Features

- Ab initio Molecular Dynamics (AIMD) using VASP
- Machine learning potential training with DeepMD-kit
- Deep Potential Molecular Dynamics (DPMD) simulations
- Active learning for continuous model improvement
- Enhanced sampling with PLUMED integration

## Requirements

### Core Dependencies
- Python 3.xx
- DeepMD-kit 2.2.10
- ASE (Atomic Simulation Environment)
- VASP (for DFT calculations)
- PLUMED (for enhanced sampling)
- MPI library

### Python Package Dependencies
- numpy
- pandas
- dpdata

## Installation

1. Create and activate a conda environment:
```bash
conda create -n sparc python=3.10
conda activate sparc
```

2. Install PLUMED:
```bash
conda install -c conda-forge py-plumed
```

4. Clone and install SPARC:
```bash
git clone https://github.com/rahulumrao/sparc.git
cd sparc
pip install .
```

## Quick Start

1. Set up VASP POTCAR files path:
```bash
export VASP_PP_PATH=/path/to/vasp/potcar_files
```

2. Create an input file (see example below)

### Example Input File
```yaml
general:
  structure_file: "POSCAR"
  md_steps: 1000
  log_frequency: 10

md_simulation:
  thermostat: "Nose"
  temperature: 300.0
  timestep_fs: 1.0

dft_calculator:
  name: "VASP"
  exe_path: "/path/to/vasp"
  exe_name: "vasp_std"
```

See `scripts/input.yaml` for a complete configuration template.


## Core Components

### 1. MD Simulation
- Supports both _ab initio_ MD (VASP) and DeepPotential MD
- NVT ensemble with Nose-Hoover thermostat
- Checkpoint/restart capabilities
- Optional PLUMED integration for enhanced sampling

### 2. DeepMD Training
- Automated model training
- Multiple model generation for labelling
- Configurable network architecture and training parameters

### 3. Active Learning
- Query by Committee approach for structure selection
- Force-based deviation metrics
- Automated structure labeling and retraining

## Workflow

1. **Initial AIMD**
   - Runs ab initio MD using VASP
   - Generates training data

2. **DeepMD Training**
   - Processes AIMD trajectories
   - Trains multiple DeepMD models
   - Freezes and compresses models

3. **DPMD Simulation**
   - Runs MD using trained DeepPotential models
   - Monitors force deviations

4. **Active Learning**
   - Identifies structures for labeling
   - Performs DFT calculations on selected structures
   - Retrains models with expanded dataset

## Current Status

- ✅ Active learning with candidate labeling implemented
- 🚧 Code refinement in progress
- 🚧 Support for additional DFT calculators planned
- 📝 Documentation under development


> [!IMPORTANT]  
> There are some version dependencies, currently the latest version of `deepmd-kit` is not supported. Check [documentation](https://deepmd-kit.readthedocs.io/en/latest/install/easy-install.html) for installation of older version.

## Limitations

- Currently only supports DeepMD-kit 2.2.10 (newer versions not yet supported)
- Limited to VASP for DFT calculations
- Documentation is still being developed

## License

This project is licensed under the MIT License.

<!-- ## Support -->

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
> [!WARNING]
> This package is under active development. Features and APIs may change.
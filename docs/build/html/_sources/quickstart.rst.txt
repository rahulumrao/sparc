.. _quickstart:

Quick Start Guide
=================

Welcome to the Quick Start Guide for SPARC. This guide walks you through
the basic setup, configuration, and execution steps to run your first
simulation.

Set Environment Variables
--------------------------

.. code-block:: bash

   export VASP_PP_PATH=/path/to/vasp/potcar_files    # POTCAR files path (VASP only)

If PLUMED was installed from source (skip if you used ``conda-forge``):

.. code-block:: bash

   export PLUMED_KERNEL="$CONDA_PREFIX/lib/libplumedKernel.so"
   export PYTHONPATH="$CONDA_PREFIX/lib/plumed/python:$PYTHONPATH"

Basic Usage
-----------

SPARC requires a ``YAML`` input file that defines the structure, DFT
calculator, MD settings, and active learning parameters.

Example Input File
------------------

The example below shows a complete active learning workflow using VASP:

.. code-block:: yaml

    general:
      structure_file: "POSCAR"

    dft_calculator:
      engine: "VASP"
      template_file: "INCAR"
      exe_command: "mpirun -np 4 vasp_std"

    aimd_setup:
      ensemble: "NVT"
      temperature: 300.0
      timestep_fs: 1.0
      steps: 100
      thermostat:
        type: "Nose"
        tdamp: 2.0

    mlip_setup:
      training: true
      data_dir: "Dataset"
      input_file: "input.json"
      num_models: 4
      MdSimulation: true
      ensemble: "NVT"
      temperature: 300.0
      timestep_fs: 1.0
      md_steps: 2000
      train_ratio: 0.8

    active_learning: true
    iteration: 10
    model_dev:
      f_min_dev: 0.05
      f_max_dev: 0.30

See :doc:`yaml` for a full description of all available options.

Running a Simulation
--------------------

.. code-block:: bash

    sparc -i input.yaml

.. _quickstart_directory:

Directory Structure
-------------------

After the first iteration the following layout is created:

.. code-block:: text

    Project Root/
    ├── POSCAR
    ├── INCAR
    ├── input.json
    ├── input.yaml
    ├── Dataset/
    │   ├── training_data/
    │   └── validation_data/
    ├── iter_000000/
    │   ├── 00.dft/           DFT / AIMD labelling
    │   ├── 01.train/         MLIP training
    │   │   ├── training_1/
    │   │   ├── training_2/
    │   │   └── ...
    │   └── 02.dpmd/          ML-MD + model deviation
    ├── iter_000001/
    │   └── ...

- ``00.dft/`` — DFT calculations used to label selected structures
- ``01.train/`` — ML model training; one ``training_N/`` folder per model
- ``02.dpmd/`` — ML-MD simulation and Query-by-Committee model deviation

Sample Output (``Sparc.log``)
------------------------------

.. code-block:: text

    ================================================================================
    BEGIN CALCULATION - 2025-04-08 22:30:32
    ================================================================================

            ######  ########     ###    ########   ######
            ##    ## ##     ##   ## ##   ##     ## ##    ##
            ##       ##     ##  ##   ##  ##     ## ##
            ######  ########  ##     ## ########  ##
                  ## ##        ######### ##   ##   ##
            ##    ## ##        ##     ## ##    ##  ##    ##
            ######  ##        ##     ## ##     ##  ######
            --v0.2.0

    ================================================================================
    Creating Directories for Iteration: 000000
    ================================================================================
    ├── iter_000000
    │   ├── 00.dft
    │   ├── 01.train
    │   └── 02.dpmd

    ================================================================================
    Starting AIMD Simulation [Nose-Hoover]
    ================================================================================
    Step     Epot (eV)    Ekin (eV)    Temp (K)
    --------------------------------------------------------------------------------
    0           -36.0932      0.3102    300.00
    1           -36.1182      0.4385    424.04
    2           -36.1058      0.4062    392.84

    ================================================================================
    MLIP Training — 4 models
    ================================================================================
    RUNNING TRAINING IN FOLDER (iter_000000/01.train/training_1)
    ...
    frozen_model_1.pth saved

    ================================================================================
    Starting ML-MD Simulation
    ================================================================================
    Step     Epot (eV)    Ekin (eV)    Temp (K)
    --------------------------------------------------------------------------------
    0           -29.8049      0.1939    300.00
    5           -29.7611      0.1458    225.61
    10          -29.7915      0.1711    264.75

Core Components
---------------

1. MD Simulation
~~~~~~~~~~~~~~~~

* NVE, NVT (Nose-Hoover / Langevin), and NPT (Berendsen) ensembles
* Supports both *ab initio* (VASP, CP2K, ORCA, QE, xTB, Gaussian) and ML-MD
* Checkpoint/restart capabilities
* PLUMED integration for enhanced sampling (Metadynamics, Umbrella Sampling)

2. MLIP Training
~~~~~~~~~~~~~~~~

* Automated DeepMD-kit training pipeline
* Ensemble model generation for uncertainty quantification
* **Fine-tuning** of universal potentials (DPA-3, MACE-MP) from a pre-trained checkpoint

3. Active Learning
~~~~~~~~~~~~~~~~~~

* Query-by-Committee (QbC) for candidate selection based on force deviation
* RMSD-based duplicate filtering for diverse training data
* Automated DFT labelling and model retraining
* ``fparam`` support for universal models (e.g., DPA-3)

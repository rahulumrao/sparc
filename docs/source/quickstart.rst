.. _quickstart:

Quick Start Guide
=================

Welcome to the Quick Start Guide for setting up and running your first simulation with SPARC. 
This guide will walk you through the basic setup, configuration, and execution steps, allowing you to quickly begin your calculation.
Follow these simple steps to configure your environment, set up an input file. 
An example input file is provided in the Basic Usage section below.

Set Environment Variables:
--------------------------

.. code-block:: bash

  export VASP_PP_PATH=/path/to/vasp/potcar_files    # POTCAR files path

If PLUMED is installed manually (skip: if used ``conda-forge``), we need to set the ``plumed`` environment.

.. code-block:: bash

  export PLUMED_KERNEL="$CONDA_PREFIX/lib/libplumedKernel.so"
  export PYTHONPATH="$CONDA_PREFIX/lib/plumed/python:$PYTHONPATH"

Basic Usage
-----------

``sparc`` code requires a ``YAML`` input file. 
This file contains essential settings that define the parameters for the simulation, 
such as the structure file, the number of molecular dynamics settings, system configuration, 
and the specifics of the DFT calculator (e.g., VASP or other supported tools).

An example YAML input is provided below.

Example Input File
------------------

.. code-block:: yaml

    general:
      structure_file: "POSCAR"

    md_simulation:
      thermostat: "Nose"
      temperature: 300.0
      steps: 100
      timestep_fs: 1.0

    dft_calculator:
      name: "VASP"
      prec: "Normal"
      incar_file: "INCAR"
      exe_command: "mpirun -np 2 /path/to/vasp_std"

Running a Simulation
--------------------

Once the input file is configured, calculation is invoked with the following command:

.. code-block:: bash

    sparc -i input.yaml

.. _quickstart_directory:
Directory Structure
-------------------

Running the first calculation cycle, the following directory structure will be created:

.. code-block:: bash

  >>> Project Root
  ├── POSCAR
  ├── INCAR
  ├── input.json
  ├── input.yaml
  ├── Dataset
  │   ├── training_data
  │   └── validation_data
  ├── iter_000000
  │   ├── 00.dft
  │   ├── 01.train
  │   └── 02.dpmd
  ├── iter_000001
  │   ├── 00.dft
  │   ├── 01.train
  │   └── 02.dpmd

This layout organizes the outputs in a structured format:

  - **POSCAR /  INCAR**: Standard VASP structure and input file
  - **input.yaml**: User-defined configuration for SPARC execution.
  - **input.json**: User-defined configuration deepmd-kit ML trainig.
  - ``Dataset/``: Stores training and validation data for ML training.
  - **iter_000000, iter_000001, ...**: Iteration folders containing

    - ``00.dft``: DFT calculations used to label selected structures
    - ``01.train``: ML model training
    - ``02.dpmd``: Molecular dynamics simulations using the trained ML potential

Each new iteration (e.g., ``iter_000001``, ``iter_000002``, ...) corresponds to an ``active learning cycle``,
where the potential is refined based on new DFT data and retrained models.    

By default this will write all the information in an output file ``sparc.log``. 
The contents of ``sparc.log`` will be like this:

.. code-block:: bash

  ========================================================================
  BEGIN CALCULATION - 2025-04-08 22:30:32
  ========================================================================


          ######  ########     ###    ########   ######
          ##    ## ##     ##   ## ##   ##     ## ##    ##
          ##       ##     ##  ##   ##  ##     ## ##
          ######  ########  ##     ## ########  ##
                ## ##        ######### ##   ##   ##
          ##    ## ##        ##     ## ##    ##  ##    ##
          ######  ##        ##     ## ##     ##  ######
          --v0.1.0

  ========================================================================
  Creating directories for Iteration: 000000
  ========================================================================
  ├── iter_000000/
  │   ├── 00.dft/
  │   ├── 01.train/
  │   └── 02.dpmd/
  ========================================================================

  ! ab-initio MD Simulations will be performed at Temp.: 300K !

  ========================================================================

  ========================================================================
                    Starting AIMD Simulation [Langevin]
  ========================================================================
  Steps: 0, Epot: -23.900369, Ekin: 0.193890, Temp: 300.00
  Steps: 1, Epot: -23.868841, Ekin: 0.163737, Temp: 253.35
  Steps: 2, Epot: -23.809660, Ekin: 0.114040, Temp: 176.45
  Steps: 3, Epot: -23.785705, Ekin: 0.095163, Temp: 147.24
  ========================================================================
            DEEPMD WILL TRAIN 2 MODELS !
  ========================================================================

  ========================================================================
  RUNNING TRAINING IN FOLDER (iter_000000/01.train/training_1) !
  DeepMD Model Evaluation Results:
  ------------------------------------------------------------------------
  DEEPMD INFO    # ---------------output of dp test---------------
  DEEPMD INFO    # testing system : Dataset/validation_data
  DEEPMD INFO    # number of test data : 45
  DEEPMD INFO    Energy MAE         : 1.000640e+01 eV
  DEEPMD INFO    Energy RMSE        : 2.874460e+01 eV
  DEEPMD INFO    Energy MAE/Natoms  : 2.001280e+00 eV
  DEEPMD INFO    Energy RMSE/Natoms : 5.748920e+00 eV
  DEEPMD INFO    Force  MAE         : 7.150255e-02 eV/A
  DEEPMD INFO    Force  RMSE        : 9.012610e-02 eV/A
  DEEPMD INFO    Virial MAE         : 2.775489e-01 eV
  DEEPMD INFO    Virial RMSE        : 3.691740e-01 eV
  DEEPMD INFO    Virial MAE/Natoms  : 5.550977e-02 eV
  DEEPMD INFO    Virial RMSE/Natoms : 7.383479e-02 eV
  DEEPMD INFO    # -----------------------------------------------
  ========================================================================
  DeepPotential model successfully loaded and tested:
  iter_000000/01.train/training_1/frozen_model_1.pb
  ========================================================================

  ========================================================================
            Initializing DeepPotential MD Simulation [Langevin]
  ========================================================================
  Steps: 0, Epot: -29.804899, Ekin: 0.193890, Temp: 300.00
  Steps: 70, Epot: -29.761078, Ekin: 0.145811, Temp: 225.61
  Steps: 140, Epot: -29.791506, Ekin: 0.171109, Temp: 264.75
  Steps: 210, Epot: -29.771736, Ekin: 0.170868, Temp: 264.38
  Steps: 280, Epot: -29.755832, Ekin: 0.133304, Temp: 206.26
  Steps: 350, Epot: -29.801518, Ekin: 0.177911, Temp: 275.28
  Steps: 420, Epot: -29.736048, Ekin: 0.165758, Temp: 256.47
  Steps: 490, Epot: -29.717222, Ekin: 0.161525, Temp: 249.92
  ========================================================================

Core Components
---------------

1. MD Simulation
~~~~~~~~~~~~~~~~
* Supports both *ab initio* MD (VASP) and DeepPotential based MD
* NVT ensemble with Nose-Hoover and Langevin thermostat
* Checkpoint/restart capabilities
* Seamless integration with PLUMED for PES exploration

2. DeepMD Training
~~~~~~~~~~~~~~~~~~
* Automated training pipeline for trainig ML models
* Supports multi-model ensembles for uncertainty quantification
* Customizable neural network architecture and training hyperparameters

3. Active Learning
~~~~~~~~~~~~~~~~~~
* Implements a `Query-by-Committee` (QbC) strategy for data selection
* Uses force deviation metrics to identify high-uncertainty configurations
* Automatically labels new structures with DFT and retrains ML models
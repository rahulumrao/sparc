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

    [SPARC][INFO]================================================================================
    [SPARC][INFO]BEGIN CALCULATION - 2026-10-14 11:39:55
    [SPARC][INFO]================================================================================
          ######    ########     ###    ########   ######
        ##     ##   ##     ##   ## ##   ##     ## ##    ##
        ##          ##     ##  ##   ##  ##     ## ##
          ######    ########  ##     ## ########  ##
                ##  ##        ######### ##   ##   ##
        ##      ##  ##        ##     ## ##    ##  ##    ##
          ######    ##        ##     ## ##     ##  ######
          --v0.2

    [SPARC][INFO] SPARC: Smart Potential with Atomistic Rare Events & Continuous Learning
    [SPARC][INFO] Version: 0.2

    [SPARC][INFO] ================================================================================
    [SPARC][INFO] Working directory: /home/rverma7/My_Calc/NCSU_Work/DeePMD/sparc/PET_decomposition/sparc_V2/Trail_2
    [SPARC][INFO] Configuration file: input.yaml
    [SPARC][INFO]
    [SPARC][INFO] ================================================================================
    [SPARC][INFO]                     YAML configuration loaded successfully!
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] Creating Directories for Iteration: 000000
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] ├── iter_000000
    [SPARC][INFO] │   ├── 00.dft
    [SPARC][INFO] │   ├── 01.train
    [SPARC][INFO] │   └── 02.dpmd
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] Creating Directories for Iteration: 000000
    [SPARC][INFO] ================================================================================
    [SPARC][INFO]├── iter_000000
    [SPARC][INFO] │   ├── 00.dft
    [SPARC][INFO] │   ├── 01.train
    [SPARC][INFO] │   └── 02.dpmd
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] Creating Directories for Iteration: 000000
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] MLIP Training — 4 models
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] Running Training in Folder (iter_000000/01.train/training_1)
    [SPARC][INFO] ...
    [SPARC][INFO] frozen_model_1.pth saved
    [SPARC][INFO] ================================================================================
    [SPARC][INFO]             MACHINE LEARNING POTENTIAL MD SIMULATION FOR [NVT-Nose]
    [SPARC][INFO] Output Logfile: iter_000000/02.dpmd/Iter0_dpmd_s0_r0.log
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] Step     Epot (eV)    Ekin (eV)    Temp (K)   P (GPa)    V (Ang^3)
    [SPARC][INFO] --------------------------------------------------------------------------------
    [SPARC][INFO] ***********************************************************
    [SPARC][INFO] Reference Potential Energy (Step 0): -112.906507 eV
    [SPARC][INFO] Threshold limit: +/- 3.50 eV
    [SPARC][INFO] ***********************************************************
    [SPARC][INFO] 0          -112.9065      0.9501    350.00
    [SPARC][INFO] 400        -112.3256      0.9020    332.30
    [SPARC][INFO] 800        -112.1378      1.0138    373.47
    [SPARC][INFO] 1200       -112.3223      0.8234    303.35
    [SPARC][INFO] 1600       -112.3493      1.0422    383.94
    [SPARC][INFO] 2000       -112.1704      1.0746    395.89
    [SPARC][INFO] QUERY-BY-COMMITTEE: MODEL DEVIATION ANALYSIS
    [SPARC][INFO] ================================================================================
    [SPARC][INFO]                            frozen_model_1.pth
    [SPARC][INFO]                            frozen_model_2.pth
    [SPARC][INFO]                            frozen_model_3.pth
    [SPARC][INFO]                            frozen_model_4.pth
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] ================================================================================
    [SPARC][INFO]           Model deviation calculation completed successfully!
    [SPARC][INFO]          Results saved in: iter_000000/02.dpmd/model_dev_0.out
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] Candidate summary saved to: iter_000000/02.dpmd/dft_candidates/candidates_summary.csv
    [SPARC][INFO] --------------------------------------------------------------------------------
    [SPARC][INFO]                         ACTIVE LEARNING CYCLE : ENABLED
    [SPARC][INFO] --------------------------------------------------------------------------------
    [SPARC][INFO] Maximum Iterations: 20
    [SPARC][INFO] Force Deviation Thresholds: [0.05, 0.4] eV/Å
    [SPARC][INFO] MLIP Ensemble Size: 4 models
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] --------------------------------------------------------------------------------
    [SPARC][INFO]                             WORKFLOW TIMING SUMMARY
    [SPARC][INFO] --------------------------------------------------------------------------------
    [SPARC][INFO]   iter 0 | total 6.2 min | mlmd=4.5 min
    [SPARC][INFO]   iter 1 | total 985.2 min | dft=319.4 min, train=611.1 min, mlmd=35.2 min
    [SPARC][INFO]   iter 2 | total 1249.4 min | dft=618.9 min, train=594.6 min, mlmd=27.8 min
    [SPARC][INFO]   iter 3 | total 1119.7 min | dft=486.0 min, train=600.2 min, mlmd=27.5 min
    [SPARC][INFO]   iter 4 | total 1068.4 min | dft=445.6 min, train=591.4 min, mlmd=27.2 min
    [SPARC][INFO]   iter 5 | total 1504.1 min | dft=420.5 min, train=937.9 min, mlmd=120.2 min
    [SPARC][INFO]   iter 6 | total 1358.2 min | dft=624.5 min, train=592.8 min, mlmd=121.5 min
    [SPARC][INFO]   iter 7 | total 1310.2 min | dft=577.8 min, train=591.5 min, mlmd=121.3 min
    [SPARC][INFO]   Saved to: /home/rverma7/My_Calc/NCSU_Work/DeePMD/sparc/PET_decomposition/sparc_V2/timings.csv
    [SPARC][INFO] --------------------------------------------------------------------------------
    [SPARC][INFO] ================================================================================
    [SPARC][INFO] END CALCULATION - 2026-10-02 11:04:29
    [SPARC][INFO] Total Runtime: 1:44:30.786859
    [SPARC][INFO] ================================================================================

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

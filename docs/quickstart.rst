Quick Start Guide
=================

This guide will help you get started with SPARC quickly.

Basic Usage
-----------

1. Create an input YAML file (see example below)
2. Run SPARC with your input file

Example Input File
------------------

.. code-block:: yaml

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

Running a Simulation
--------------------

To run a simulation:

.. code-block:: bash

    python sparc.py -i input.yaml

Core Components
---------------

1. MD Simulation
~~~~~~~~~~~~~~~~
* Supports both *ab initio* MD (VASP) and DeepPotential MD
* NVT ensemble with Nose-Hoover thermostat
* Checkpoint/restart capabilities
* Optional PLUMED integration

2. DeepMD Training
~~~~~~~~~~~~~~~~~~
* Automated model training
* Multiple model generation
* Configurable network architecture

3. Active Learning
~~~~~~~~~~~~~~~~~~
* Query by Committee approach
* Force-based deviation metrics
* Automated structure labeling 
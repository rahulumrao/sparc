Input File
==========

SPARC input is divided in various sections for different tasks. 
Each section has some required key values and some of them are optional defined by the user.
Since the code is independent from each seaction. 
Therefore, each task (*ab initio*, *MLP training*, *MLP-MD*, *Active Learning*) can be executed separately.
Required/optional input for each sections is explained below.


General Settings
~~~~~~~~~~~~~~~~

This section requires the coordinate structure file to begin the calculation.
Currently only ``POSCAR`` file format is supported.

.. code-block:: yaml

    # General settings
    general:
      structure_file: "POSCAR"  # Input structure [Default]

DFT Calculator
--------------

Here, we define the parameters for the DFT calculator (e.g., VASP), and the executable command path.
For flexibilty we have also have option to directly parse the ``INCAR`` file.

.. code-block:: yaml

    # DFT calculator settings
    dft_calculator:
      name: "VASP"               # DFT package name         [Required]
      prec: "Normal"             # Precision level          [Required]
      kgamma: True               # Gamma point calculation  [Required]
      incar_file: "INCAR"        # Path to INCAR file       [Required]
      exe_command: "mpirun -np 2 /path/to/VASP/bin/vasp_std"  # Full command to run VASP with MPI [Required]


MD Simulation
-------------

The settings for the molecular dynamics (MD) simulation are configured in this section, 
including the type of thermostat (Nose/Langevin), MD steps, and timestep, etc.

.. code-block:: yaml

    # ASE MD simulation settings
    md_simulation:
      ensemble: "NVT"           # Ensemble for MD simulation                  [Required]
      thermostat:               # Thermostat configuration as a dictionary    [Required]
        type: "Nose"            # Thermostat type (Nose-Hoover or Langevin)   [Required]
        tdamp: 10.0              # Damping parameter for Nose-Hoover thermostat  [Required]
      timestep_fs: 1.0          # TimeStep for MD simulation                    [Optional (Default: 1.0)]
      temperature: 300          # Temperature in Kelvin                         [Optional (Default: 300)]
      steps: 10                 # Number of AIMD steps                          [Required]
      restart: False            # Whether to restart MD simulation from a checkpoint file [Optional]
      use_dft_plumed: True      # Use PLUMED for AIMD simulation                 [Optional (Default: False)]
      plumed_file: "plumed_dft.dat"    # PLUMED input file                       


.. note::

    Both the ``Nose-Hoover and Langevin`` thermostats have been implemented. 
    If the ``Nose`` thermostat is selected it requires damping ``tdamp`` parameter. 
    Alternatively, if the ``Langevin`` thermostat is chosen, the ``friction`` parameter must be specified. 
    For further details, refer to the `Molecular Dynamics <asemd_>`_ section in ASE.

    .. code-block:: yaml

      type: Nose
      tdamp: 0.01

      type: Langevin
      friction: 0.01

PLUMED
------

In this section, the settings for PLUMED integration are defined. 
This includes the PLUMED input file and the kT value for simulations involving collective variables.

.. code-block:: yaml

    # Plumed settings
    plumed:
      restart: True             # restart PLUMED        [Optional (Default: False)]
      kT: 0.02585               # kT value PLUMED units [if Plumed=True[Required]]


.. _deepmd_section:
DeepMD
------

This section defines the settings for training a DeepMD model. 
It includes the paths to the training data and input files, as well as the number of models to generate.

.. code-block:: yaml

    # DeepMD settings
    deepmd_setup:
      training: False           # Enable DeepPotential training  [Required]
      data_dir: "DeePMD_training/00.data"  # Path to store training and validation data [Optional]
      input_file: "input.json"  # Input file for DeepMD training  [Required]
      skip_min: 0               # Minimum frames to skip  [Optional] exclude 
      skip_max: null            # Maximum frames to skip  [Optional]
      num_models: 2             # Number of models to be trained  [Required (can not be less than 2)]
      MdSimulation: True        # DeepPotential MD simulation (False/True)  [Required]
      timestep_fs: 1.0          # Timestep (fs) [Optional (Default: 1.0)]
      md_steps: 2000            # MLP-MD steps  [Required]
      multiple_run: 5           # Run multiple MD-MLP run starting from different velocities [Optional (Default: 0)]
      log_frequency: 5          # Output frequency  [Required]
      use_plumed: True          # Default: False  [Optional]
      plumed_file: "plumed.dat" # Optional: defaults to "plumed.dat" if not specified


Active Learning
---------------

Active learning protocol is enabled by setting it to ``True``. 
This section contains the configuration how the learning loop will be executed, 
including the number of iterations and the force deviation metrics. 
AL iteration can be restarted by enabling ``learning_restart`` keyword, which also requires the path to the ``latest model``.

.. code-block:: yaml

    # Active Learning
    active_learning: False      # [Required (Default: False)]
    learning_restart: False     # Restart AL loop from last step  [Optional]
    latest_model: 'iter_000006/01.train/training_1/frozen_model_1.pb'   # Latest Model path to restart [Required (if above True)]
    iteration: 10               # Active Learning iteration steps [Optional (default: 10)]
    model_dev:
      f_min_dev: 0.1            # [Required]
      f_max_dev: 0.8            # [Required]


Metric
------

This is a sanity check to exclude the unphysical structures when running ML/MD, 
since, the initial ML model might not have sufficient information about the potential energy surface.
Also, when the criterian is met the MD simulation will be stopped and the code will move to the next iteration.
This feature is fully option but recommended.

.. code-block:: yaml

    # check metrics [Optional]
    distance_metrics:
      - pair: [0, 3]
        min_distance: 1.2  # Minimum allowed distance in Angstroms
        max_distance: 5.0  # Maximum allowed distance in Angstroms
      - pair: [0, 1]
        min_distance: 1.2  # Minimum allowed distance in Angstroms
        max_distance: 2.0  # Maximum allowed distance in Angstroms
    # If you want to skip distance checks, you can comment out or remove this section.


Output
------

Each iteration will log the output in a file. These filenames are default and are optional to change by user.

.. code-block:: yaml

    # File output settings
    output:
      log_file: "AseMD.log"        # Log file of MD simulation        [Optional (Default: AseMD.log)]
      aimdtraj_file: "AseMD.traj"  # ASE trajectory file for AIMD run [Optional (Default: AseMD.traj)]
      dptraj_file: "dpmd.traj"     # ASE DeepMD trajectory file       [Optional (Default: dpmd.traj)]


``log_file`` contains the MD information, and ``traj`` file stores the trajectory structures.

.. code-block:: bash
  
  Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
  0.0000        -112.0807    -112.8950       0.8143   300.0
  0.0700        -111.6322    -112.7149       1.0828   398.9
  0.1400        -112.4215    -113.3518       0.9303   342.7
  0.2100        -112.9996    -113.6775       0.6779   249.8
  0.2800        -112.6910    -113.7220       1.0310   379.8

Directory structure
-------------------

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

For detailed output see section :ref:`quickstart_directory`.

.. _asemd: https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html
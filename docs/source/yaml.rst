.. _inputfile:

Input File
==========

SPARC input is configured via a single YAML file (``input.yaml``).
It is divided into sections for different tasks — each task
(*ab initio MD*, *MLIP training*, *ML-MD*, *Active Learning*) can be
enabled or disabled independently.

.. code-block:: bash

    python sparc.py -i input.yaml


General Settings
~~~~~~~~~~~~~~~~

Specifies the input structure file. Supports a single file or a list of files
(used for multiple independent MD runs).

.. code-block:: yaml

    general:
      structure_file: "POSCAR"             # single file         [Required]
      # structure_file:                    # or a list of files
      #   - "POSCAR_1"
      #   - "POSCAR_2"

.. note::

    VASP ``POSCAR``, ``xyz``, ``cif``, and any other format supported by ASE can be used.
    For Gaussian and ORCA, periodicity is automatically removed if present.


DFT Calculator
--------------

Defines the DFT engine, template file, and optional executable path.
Supported engines: ``VASP``, ``CP2K``, ``ORCA``, ``xTB``, ``QE``, ``Gaussian``.

.. code-block:: yaml

    dft_calculator:
      engine: "VASP"                       # DFT engine name                  [Required]
      template_file: "INCAR"               # Engine-specific template file     [Required]
      exe_command: "mpirun -np 4 vasp_std" # Executable command (auto-detect if omitted) [Optional]

Each engine reads its template differently:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Engine
     - Template file
     - Notes
   * - ``VASP``
     - ``INCAR``
     - Standard VASP INCAR format
   * - ``CP2K``
     - ``cp2k_template.inp``
     - CP2K input file (comments stripped)
   * - ``ORCA``
     - ``orca_template.inp``
     - ORCA simple input format
   * - ``xTB``
     - ``xtb_template.inp``
     - ``key = value`` file
   * - ``QE``
     - ``qe_template.in``
     - Quantum ESPRESSO ``pw.x`` input; k-points default to Gamma
   * - ``Gaussian``
     - ``gaussian_template.inp``
     - ``key = value`` file; always non-periodic


AB Initio MD (AIMD)
-------------------

Controls the ab initio MD run driven by the DFT calculator.
Set ``steps: 0`` (default) to skip AIMD entirely.

.. code-block:: yaml

    aimd_setup:
      ensemble: "NVT"            # Ensemble: NVT, NVE, or NPT               [Required]
      temperature: 300.0         # Temperature in Kelvin                     [Required]
      temp_end: null             # Ramp temperature to this value (optional) [Optional]
      timestep_fs: 1.0           # MD timestep in femtoseconds               [Optional, Default: 1.0]
      steps: 500                 # Number of AIMD steps (0 = skip AIMD)      [Required]
      log_frequency: 10          # Output frequency in steps                 [Optional, Default: 1]
      restart: false             # Resume from checkpoint                    [Optional, Default: false]

      thermostat:
        type: "Nose"             # Nose-Hoover or Langevin                   [Required]
        tdamp: 2.0               # Damping time for Nose-Hoover (fs)         [Required for Nose]
        # friction: 0.01         # Friction coefficient for Langevin         [Required for Langevin]

      plumed:
        enabled: false           # Enable PLUMED enhanced sampling           [Optional, Default: false]
        plumed_file: "plumed_dft.dat"  # PLUMED input file                   [Required if enabled]
        kT: 0.02585              # kT in eV (300 K ≈ 0.02585)               [Optional]
        restart: false           # Restart PLUMED from checkpoint            [Optional]

**NPT ensemble** requires additional parameters:

.. code-block:: yaml

    aimd_setup:
      ensemble: "NPT"
      temperature: 300.0
      tau_t: 100.0               # Thermostat time constant (fs)             [Required for NPT]
      tau_p: 1000.0              # Barostat time constant (fs)               [Required for NPT]
      pressure: 1.01325          # Target pressure in bar (1 atm = 1.01325 bar) [Required for NPT]
      compressibility: null      # Isothermal compressibility in 1/bar (null = Cu default ~7.1e-7) [Optional]

.. note::

    Temperature ramping (``temp_end``) is supported for NVT/Langevin.
    Nose-Hoover thermostat resists rapid temperature changes — use Langevin for ramping.


.. _deepmd_section:

MLIP Setup
----------

Controls MLIP model training and ML-MD simulation.

.. code-block:: yaml

    mlip_setup:
      # ── Training ──
      training: false             # Enable MLIP training                     [Required]
      data_dir: "Training_Data"   # Directory for training data              [Optional]
      input_file: "input.json"    # DeepMD training input JSON               [Required if training]
      skip_min: 0                 # Skip first N frames from trajectory      [Optional]
      skip_max: null              # Skip frames beyond this index            [Optional]
      train_ratio: 0.8            # Training fraction (0.0, 1.0); rest = validation [Optional, Default: 0.8]
      num_models: 4               # Number of committee models (min 2)       [Required]

      # ── ML-MD ──
      MdSimulation: false         # Enable ML-MD simulation                  [Required]
      ensemble: "NVT"             # Ensemble: NVT, NVE, or NPT               [Required]
      temperature: 300.0          # Temperature in Kelvin                    [Required]
      temp_end: null              # Ramp temperature to this value           [Optional]
      timestep_fs: 1.0            # MD timestep in femtoseconds              [Optional, Default: 1.0]
      md_steps: 2000              # Number of ML-MD steps                    [Required]
      multiple_run: 1             # Independent MD runs (uses structure list) [Optional, Default: 1]
      log_frequency: 5            # Output frequency in steps                [Optional, Default: 5]
      epot_threshold: 2.5         # Stop MD if Epot spike exceeds this (eV)  [Optional]
      restart: false              # Resume ML-MD from checkpoint             [Optional]

      # ── Restart exploration ──
      restart_exploration: false  # Start next iteration from a saved frame  [Optional]
      restart_frame: "candidates" # Frame source: "candidates" or file path  [Optional]

      thermostat:
        type: "Nose"
        tdamp: 2.0
        # friction: 0.01

      plumed:
        enabled: false
        plumed_file: "plumed.dat"
        kT: 0.02585
        restart: false

        umbrella_sampling:
          enabled: false          # Enable umbrella sampling windows         [Optional]
          config_file: "umbrella_sampling.yaml"  # Window definitions file   [Required if enabled]


Fine-Tuning (Universal Models)
------------------------------

Optional section to fine-tune a pre-trained universal DeePMD model (DPA-3)
instead of training from scratch.

.. code-block:: yaml

    finetune:
      enabled: false                       # Enable fine-tuning               [Optional, Default: false]
      model_type: "deepmd"                 # Model backend                    [Required if enabled]
      pretrained_model: "DPA3.pt"          # Path to pre-trained model        [Required if enabled]
      model_branch: "Omat24"               # Model branch for multi-task models [Optional]
      input_file: null                     # Fine-tune JSON (uses mlip_setup.input_file if null) [Optional]
      learning_rate: 0.001                 # Starting learning rate           [Optional]
      device: "cpu"                        # "cpu" or "cuda"                  [Optional]


Active Learning
---------------

Enables the iterative active learning loop. When enabled, SPARC will
repeatedly run ML-MD, select uncertain candidates with Query-by-Committee,
label them with DFT, and retrain the models.

.. code-block:: yaml

    active_learning: false        # Enable active learning loop              [Required]
    learning_restart: false       # Resume AL from last saved checkpoint     [Optional]
    latest_model: null            # Model path to use on restart             [Required if learning_restart]
    iteration: 10                 # Maximum AL iterations                    [Optional, Default: 10]

    model_dev:
      f_min_dev: 0.1              # Lower force deviation threshold (eV/Å)  [Required]
      f_max_dev: 0.8              # Upper force deviation threshold (eV/Å)  [Required]

Structures with force deviation in ``[f_min_dev, f_max_dev]`` are selected as
candidates. Structures below ``f_min_dev`` are well-described; above ``f_max_dev``
are too uncertain and discarded.


Distance Metrics
----------------

Optional sanity check to stop ML-MD when atomic distances become unphysical.
Useful in early AL iterations when the model may not be reliable.

.. code-block:: yaml

    distance_metrics:
      - pair: [0, 3]
        min_distance: 1.2        # Minimum allowed distance (Å)
        max_distance: 5.0        # Maximum allowed distance (Å)
      - pair: [0, 1]
        min_distance: 1.2
        max_distance: 2.0

Atom indices in ``pair`` refer to the 0-based index in the structure file.
The MD will stop and the frame will be discarded if any constraint is violated.


Output
------

Controls output filenames. All fields are optional.

.. code-block:: yaml

    output:
      log_file: "AseMD.log"        # MD log file (time, energies, temperature) [Optional]
      aimdtraj_file: "AseMD.traj"  # AIMD trajectory                           [Optional]
      dptraj_file: "dpmd.traj"     # ML-MD trajectory                          [Optional]
      xyz_file: "AseTraj.xyz"      # XYZ format trajectory                     [Optional]

The ``log_file`` format:

.. code-block:: text

    Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
    0.0000        -112.0807    -112.8950       0.8143   300.0
    0.0700        -111.6322    -112.7149       1.0828   398.9
    0.1400        -112.4215    -113.3518       0.9303   342.7


Directory Structure
-------------------

.. code-block:: text

    Project Root/
    ├── POSCAR               (structure file)
    ├── INCAR                (DFT template)
    ├── input.json           (DeepMD training input)
    ├── input.yaml           (SPARC input)
    ├── Training_Data/       (processed training data)
    ├── iter_000000/
    │   ├── 00.dft/          (DFT / AIMD run)
    │   ├── 01.train/        (model training output)
    │   └── 02.dpmd/         (ML-MD run + model deviation)
    ├── iter_000001/
    │   ├── 00.dft/
    │   ├── 01.train/
    │   └── 02.dpmd/
    └── ...

For a complete worked example see :ref:`quickstart`.

.. _asemd: https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html

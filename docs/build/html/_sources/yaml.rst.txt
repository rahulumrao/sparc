.. _inputfile:

Input File
==========

SPARC input is configured via a single YAML file (``input.yaml``).
It is divided into sections for different tasks — each task
(*ab initio MD*, *MLIP training*, *ML-MD*, *Active Learning*) can be
enabled or disabled independently.

.. code-block:: bash

    sparc -i input.yaml


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

.. note::

    **Bias force correction** (``aimd_setup.plumed.force_correction``) is available for
    PLUMED-biased AIMD trajectories. When PLUMED applies a bias (e.g. metadynamics or
    umbrella sampling), the recorded forces include the bias contribution. Force correction
    subtracts the PLUMED bias forces from each frame before the trajectory is used for
    MLIP training, ensuring models are trained on physical (unbiased) forces only.
    Full documentation will be added in a future release.

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
      seed: 42                    # Random seed for train/validation split   [Optional, Default: 42]
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
      restart_frame: "candidates" # Frame source: "last", "random", "candidates" [Optional]

      thermostat:
        type: "Nose"
        tdamp: 2.0
        # friction: 0.01

      plumed:
        enabled: false
        plumed_file: "plumed.dat"
        kT: 0.02585
        restart: false
        start_iteration: 0        # Apply PLUMED from this AL iteration      [Optional, Default: 0]

        umbrella_sampling:
          enabled: false          # Enable umbrella sampling windows         [Optional]
          config_file: "umbrella_sampling.yaml"  # Window definitions file   [Required if enabled]

.. note::

    **Delayed PLUMED activation** (``mlip_setup.plumed.start_iteration``) lets the first
    AL iterations run as plain ML-MD to build a reliable base model, then switches on
    PLUMED-biased sampling (e.g. umbrella sampling or metadynamics) from the specified
    iteration onward. For example, ``start_iteration: 1`` skips PLUMED in iteration 0
    and enables it from iteration 1. Default ``0`` applies PLUMED from the start.


Restart Exploration
~~~~~~~~~~~~~~~~~~~

By default each AL iteration starts ML-MD from the original input structure.
``restart_exploration`` changes this so each iteration seeds its MD from a
frame saved in the **previous** iteration, helping the model explore new regions
of phase space rather than re-sampling the same starting geometry.

.. code-block:: yaml

    mlip_setup:
      restart_exploration: false   # Seed ML-MD from a previous-iteration frame [Optional, Default: false]
      restart_frame: "candidates"  # Frame selection strategy                   [Optional, Default: "candidates"]

Three strategies are available for ``restart_frame``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - Behaviour
   * - ``"candidates"``
     - Each run starts from a different randomly chosen DFT-labelled candidate
       from the previous iteration. Safest — candidates are already validated by DFT.
   * - ``"last"``
     - All runs start from the last frame of the previous ML-MD trajectory.
       Good when a single long run is used (``multiple_run: 1``).
   * - ``"random"``
     - Each run starts from a different random frame in the previous ML-MD trajectory.
       Broadest phase-space coverage but frames are not DFT-validated.

.. note::

    ``restart_exploration`` has no effect in iteration 0 (no previous trajectory exists).
    It activates from iteration 1 onward.


Fine-Tuning (Universal Models)
------------------------------

Optional section to fine-tune a pre-trained universal DeePMD model (DPA-3)
instead of training from scratch.
See :doc:`finetune` for full details.

.. code-block:: yaml

    finetune:
      enabled: false                       # Enable fine-tuning               [Optional, Default: false]
      model_type: "deepmd"                 # Model backend                    [Required if enabled]
      pretrained_model: "DPA3.pt"          # Path to pre-trained model        [Required if enabled]
      model_branch: "Omat24"               # Branch for multi-task models     [Optional]
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
    min_candidates: 1             # Stop if candidates found < this value    [Optional, Default: 1]

    model_dev:
      f_min_dev: 0.1              # Lower force deviation threshold (eV/Å)  [Required]
      f_max_dev: 0.8              # Upper force deviation threshold (eV/Å)  [Required]
      rmsd_threshold: 0.05        # RMSD duplicate filter (Å)               [Optional, Default: 0.05]
      exclude_hydrogen: true      # Exclude H atoms from RMSD calculation   [Optional, Default: true]

Structures with force deviation in ``[f_min_dev, f_max_dev]`` are selected as
candidates. Structures below ``f_min_dev`` are well-described; above ``f_max_dev``
are too uncertain and discarded.

The AL loop stops when ``iteration`` is reached **or** when the number of candidates
found in an iteration falls below ``min_candidates``. The default is ``min_candidates: 1``, and stops only when zero candidates are found. Set a
higher value to stop earlier when the model is converging and only a handful of
uncertain structures remain.

**RMSD duplicate filtering** removes near-identical candidates before DFT labelling.
Each candidate is compared (via the Kabsch algorithm) against the initial frame and all
already-accepted candidates in the same iteration. Structures with RMSD below
``rmsd_threshold`` are discarded as duplicates. A log of every accept/skip decision is
written to ``dft_candidates/rmsd_filtering.dat``.
See :ref:`rmsd_analysis` for how to compute RMSD on a trajectory.

.. note::

    Set ``rmsd_threshold: 0.0`` to disable RMSD filtering and accept all candidates
    within the force-deviation range. Use ``exclude_hydrogen: false`` to include H atoms
    in the RMSD calculation.


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
    ├── Training_Data/
    │   ├── training_data/   (DeepMD npy sets for training)
    │   └── validation_data/ (DeepMD npy sets for validation)
    ├── iter_000000/
    │   ├── 00.dft/          (DFT / AIMD run)
    │   ├── 01.train/        (model training or fine-tuning)
    │   │   ├── training_1/
    │   │   ├── training_2/
    │   │   └── ...
    │   └── 02.dpmd/         (ML-MD run + model deviation)
    ├── iter_000001/
    │   ├── 00.dft/
    │   ├── 01.train/
    │   └── 02.dpmd/
    └── ...

For a complete worked example see :ref:`quickstart`.

.. _asemd: https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html
.. _kabsch: https://en.wikipedia.org/wiki/Kabsch_algorithm
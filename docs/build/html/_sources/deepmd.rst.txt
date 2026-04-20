DeepMD Setup
============

.. module:: deepmd

Overview
--------

This module provides functionalities to setting up the training of DeepMD ML models,
freezing and compressing them, and evaluating their accuracy.
Inside each ``iter_00000xx`` directory a ``01.train`` directory will be created, and based on the ``num_models``
a subdirectory with name ``traininig_x`` will be created and the training will be performed.
Training will be done based on the user defined ``input_file``, see :ref:`deepmd_section` section.

.. tip::
   Users are encouraged to train several models for better accuracy.

After training each directory should have a file ``frozen_model.pb``.
These different models will be used to perform *Query-by-committee* to find the candidates for labelling.

.. note::
   **Query by committee (QbC)**: Identifies the configurations by measuring the disagreement among an ensemble of model.
   Allows the model to learn only **what it needs to** without wasting resources on redundant data.
   See also Deepmd-kit `model deviation <qbc_>`_ for more details.


If the candidates are found then inside the ``02.dft`` directory a subdirecoty ``dft_candidates`` will be created
with a separate ``POSCAR`` for each candidates.

.. code-block:: bash

   >>> tree dft_candidates
       ├── 0001
       │   └── POSCAR
       ├── 0002
       │   └── POSCAR
       ├── 0003
       │   └── POSCAR
       ├── 0004
       │   └── POSCAR
       ├── 0005
           └── POSCAR


Features:
   - Setup DeepPotential calculators for ASE atoms objects
   - Train DeepMD models with various configurations
   - Freeze and compress trained models
   - Evaluate trained models

Usage Example
-------------

Here is an example of how to use `setup_DeepPotential` to assign a DeepPotential calculator to an ASE atoms object:

.. code-block:: python

   from sparc.src.deepmd import setup_DeepPotential
   from ase import Atoms

   atoms = Atoms("H2O")
   dp_system, dp_calc = setup_DeepPotential(atoms, "path/to/model")
   print(dp_system.get_potential_energy())


Supported Backends
------------------

SPARC automatically detects the installed DeepMD-kit version and selects the
appropriate backend at runtime:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Backend
     - Version
     - Notes
   * - DeepMD v2 (TensorFlow)
     - deepmd-kit 2.x
     - Trains standard DeePMD models; frozen model saved as ``frozen_model.pb``
   * - DeepMD v3 (TensorFlow)
     - deepmd-kit 3.x
     - Trains DeePMD models with TF backend; frozen model saved as ``frozen_model.pb``
   * - DeepMD v3 (PyTorch)
     - deepmd-kit 3.x
     - Trains DeePMD models with PyTorch backend; frozen model saved as ``frozen_model.pth``; supports fine-tuning from universal models
   * - GNN (MACE / NequIP)
     - deepmd-kit 3.x + deepmd-gnn
     - GNN potentials trained through the DeepMD-kit interface; frozen model saved as ``frozen_model.pth``

No changes to ``input.yaml`` or ``input.json`` are required when switching
versions — SPARC falls back automatically based on what is installed.

.. note::
   Fine-tuning from pre-trained universal models (DPA-3) and GNN backends
   (MACE, NequIP) require DeepMD-kit v3 with the PyTorch backend.
   See :doc:`finetune` and :ref:`deepmd_gnn` for details.


Module Contents
---------------
.. (# change the path to sparc.src.deepmd later)
.. automodule:: sparc.src.deepmd
   :members:
   :undoc-members:
   :show-inheritance:


.. _deepmd_gnn:

Graph Neural Network Potentials (GNN)
--------------------------------------

DeepMD-kit v3 (PyTorch backend) supports training graph neural network
potentials via the
`deepmd-gnn <https://github.com/deepmodeling/deepmd-gnn>`_ plugin.
MACE and NequIP models are trained through the same DeepMD-kit interface as
standard DeePMD models — the only difference is the ``model.type`` field in
``input.json``.

Because SPARC drives training by passing ``input.json`` directly to
DeepMD-kit, **MACE and NequIP are fully supported within the SPARC active
learning workflow**. Set ``mlip_setup.input_file`` to a MACE or NequIP
``input.json`` and the rest of the pipeline (training, freezing, ML-MD,
and Query-by-Committee model deviation) works without any other changes.

.. note::

   GNN backends (MACE, NequIP) require DeepMD-kit v3 with the PyTorch
   backend and the `deepmd-gnn <https://github.com/deepmodeling/deepmd-gnn>`_
   plugin. The recommended installation is via conda-forge::

      conda install deepmd-gnn -c conda-forge

   Alternatively, install from source (requires Python 3.9+, a C++ compiler
   supporting C++14/C++17, DeepMD-kit v3.0.0b2+, and PyTorch)::

      pip install deepmd-gnn

   The TensorFlow backend (DeepMD-kit v2.x) does not support GNN models.

**MACE**

`MACE <https://github.com/ACEsuit/mace>`_ is an equivariant GNN potential.
Set ``model.type`` to ``"mace"`` in ``input.json``:

.. code-block:: json

   {
     "model": {
       "type": "mace",
       "type_map": ["O", "H"],
       "r_max": 6.0,
       "sel": "auto",
       "hidden_irreps": "64x0e"
     },
     "learning_rate": {
       "type": "exp",
       "decay_steps": 5000,
       "start_lr": 0.001,
       "stop_lr": 3.51e-8
     },
     "loss": {
       "type": "ener",
       "start_pref_e": 0.02,
       "limit_pref_e": 1,
       "start_pref_f": 1000,
       "limit_pref_f": 1,
       "start_pref_v": 0,
       "limit_pref_v": 0
     },
     "training": {
       "training_data": {
         "systems": ["data/data_0/", "data/data_1/", "data/data_2/"],
         "batch_size": "auto"
       },
       "validation_data": {
         "systems": ["data/data_3"],
         "batch_size": 1,
         "numb_btch": 3
       },
       "numb_steps": 1000000,
       "seed": 10,
       "disp_file": "lcurve.out",
       "disp_freq": 100,
       "save_freq": 1000
     }
   }

**NequIP**

`NequIP <https://github.com/mir-group/nequip>`_ is another equivariant GNN
framework supported by deepmd-gnn. Set ``model.type`` to ``"nequip"``:

.. code-block:: json

   {
     "model": {
       "type": "nequip",
       "type_map": ["O", "H"],
       "r_max": 6.0,
       "sel": "auto",
       "l_max": 1
     },
     "learning_rate": {
       "type": "exp",
       "decay_steps": 5000,
       "start_lr": 0.001,
       "stop_lr": 3.51e-8
     },
     "loss": {
       "type": "ener",
       "start_pref_e": 0.02,
       "limit_pref_e": 1,
       "start_pref_f": 1000,
       "limit_pref_f": 1,
       "start_pref_v": 0,
       "limit_pref_v": 0
     },
     "training": {
       "training_data": {
         "systems": ["data/data_0/", "data/data_1/", "data/data_2/"],
         "batch_size": "auto"
       },
       "validation_data": {
         "systems": ["data/data_3"],
         "batch_size": 1,
         "numb_btch": 3
       },
       "numb_steps": 1000000,
       "seed": 10,
       "disp_file": "lcurve.out",
       "disp_freq": 100,
       "save_freq": 1000
     }
   }

See the `deepmd-gnn examples <https://github.com/deepmodeling/deepmd-gnn/tree/master/examples/water>`_
for full working input files.


References
----------

For more details on DeepMD-Kit, visit: https://github.com/deepmodeling/deepmd-kit

.. _qbc: https://docs.deepmodeling.com/projects/deepmd/en/stable/test/model-deviation.html

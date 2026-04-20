Data Processing
===============

.. module:: data_processing

Overview
--------

The data processing module converts an ASE trajectory file into the
DeepMD ``.npy`` format required for MLIP training. Frames are randomly
split into training and validation sets using a configurable ratio.

The processed data is written to two subdirectories inside ``data_dir``:

.. code-block:: text

    data_dir/
    ├── training_data/      # DeepMD npy sets for training
    └── validation_data/    # DeepMD npy sets for validation

Usage
-----

``get_data`` is called automatically by SPARC during the training step.
The relevant options are set under ``mlip_setup`` in ``input.yaml``:

.. code-block:: yaml

    mlip_setup:
      data_dir: "Training_Data"   # Output directory for processed data
      skip_min: 0                 # Skip first N frames
      skip_max: null              # Skip frames beyond this index (null = keep all)
      train_ratio: 0.8            # Training fraction — rest goes to validation
      seed: 42                    # Random seed for reproducible split

Train / Validation Split
------------------------

Given ``n_frames`` remaining after applying ``skip_min`` / ``skip_max``:

.. code-block:: text

    n_train = int(n_frames × train_ratio)
    n_val   = n_frames − n_train

``n_val`` frame indices are drawn randomly (seeded by ``seed``); the
remaining indices form the training set. Setting ``seed`` to a fixed
value produces a reproducible split across runs.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``train_ratio``
     - ``0.8``
     - Fraction of frames for training. Must be in ``(0.0, 1.0)``.
   * - ``seed``
     - ``42``
     - NumPy random seed for the validation index draw.
   * - ``skip_min``
     - ``0``
     - Number of frames to skip from the start of the trajectory.
   * - ``skip_max``
     - ``null``
     - Skip frames from this index onward (``null`` keeps all frames).

Module Contents
---------------

.. automodule:: sparc.src.data_processing
   :members:
   :undoc-members:
   :show-inheritance:

Reference
---------

See `dpdata <dpdata_>`_ for details on the DeepMD npy format.

.. _dpdata: https://docs.deepmodeling.com/projects/dpdata/en/master/index.html

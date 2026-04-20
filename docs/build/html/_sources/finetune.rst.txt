Fine-Tuning Universal Models
============================

.. module:: finetune

Overview
--------

The fine-tuning module allows you to adapt a pre-trained universal ML potential
to your specific chemical system using a small amount of DFT reference data.
This is an alternative to training from scratch and typically requires far fewer
DFT calculations to reach the same accuracy.

Currently supported backend:

- **DeePMD** — fine-tunes DPA-1, DPA-2, or DPA-3 models via ``dp --pt finetune``
  (requires DeePMD-kit v3 with PyTorch backend)

SPARC trains ``num_models`` independent models (one per ``training_x/``
directory). These are used as a committee for Query-by-Committee uncertainty
estimation during active learning.

.. note::

   Fine-tuning replaces the standard MLIP training step. Set ``finetune.enabled: true``
   in ``input.yaml`` and ensure ``mlip_setup.training: true`` is also enabled.

Configuration
-------------

Enable fine-tuning in ``input.yaml`` under the ``finetune`` section:

.. code-block:: yaml

    finetune:
      enabled: true
      model_type: "deepmd"
      pretrained_model: "DPA3.pt"    # Path to pre-trained model file
      model_branch: "Omat24"         # Multi-task branch for DPA-3 (optional)
      input_file: null               # Fine-tune JSON (uses mlip_setup.input_file if null)
      learning_rate: 0.001           # Starting learning rate
      device: "cpu"                  # "cpu" or "cuda"

All fields except ``enabled`` and ``model_type`` are optional and have sensible defaults.

DeePMD Fine-Tuning
------------------

Uses ``dp --pt train --finetune`` to initialise from a pre-trained checkpoint.

**Requirements**

- DeePMD-kit v3 with PyTorch backend (``pip install deepmd-kit[torch]``)
- A pre-trained model file (``*.pt``) — e.g., DPA-3 from the DeepModeling model hub

**What it does**

For each model ``i`` in ``1 … num_models``:

1. Copies and updates ``input.json`` (data paths, atom types, optional learning rate override)
2. Runs ``dp --pt train input.json --finetune <pretrained_model.pt> [--model-branch BRANCH]``
3. Freezes the trained model: ``dp --pt freeze -o frozen_model_i.pth``

The frozen models are saved inside ``01.train/training_i/frozen_model_i.pth``.

**Multi-task models (DPA-3)**

DPA-3 is a multi-task model trained on several datasets simultaneously. Use
``model_branch`` to select the branch that best matches your system:

.. code-block:: yaml

    finetune:
      model_type: "deepmd"
      pretrained_model: "DPA3.pt"
      model_branch: "Omat24"         # inorganic materials
      # model_branch: "Organic_Reactions"

Leave ``model_branch`` as ``null`` for single-task models (DPA-1, DPA-2).

.. .. MACE Fine-Tuning (not currently integrated with DeepMD workflow)
.. .. ----------------------------------------------------------------
.. .. Uses ``mace_run_train --foundation_model`` for transfer learning from MACE-MP-0.

Integration with Active Learning
---------------------------------

When fine-tuning is enabled, SPARC substitutes it for the standard DeePMD
training step inside the active learning loop:

.. code-block:: text

    Iteration N
    ├── 00.dft/       DFT single-point labelling
    ├── 01.train/     Fine-tuning (replaces from-scratch training)
    │   ├── training_1/
    │   │   ├── input.json
    │   │   └── frozen_model_1.pth
    │   ├── training_2/
    │   └── ...
    └── 02.dpmd/      ML-MD + model deviation (QbC)

The frozen models produced here are picked up automatically by the ML-MD
and model-deviation steps.

Troubleshooting
---------------

**"Pre-trained model not found"**
    Ensure ``pretrained_model`` is a valid path relative to the project root.

**"DeePMD fine-tuning requires v3 or later"**
    Upgrade with ``pip install --upgrade deepmd-kit[torch]``.

**"unexpected keyword argument" during fine-tuning**
    Version mismatch between the pre-trained model and installed DeePMD-kit.
    Either upgrade DeePMD-kit or use a model compatible with your installed version
    (e.g., DPA-2 models work with DeePMD-kit v3.0.x).

Module Contents
---------------

.. automodule:: sparc.src.finetune
   :members:
   :undoc-members:
   :show-inheritance:

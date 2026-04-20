.. _fine_tuning_guide:

Fine-Tuning vs. Training From Scratch
======================================

SPARC supports two MLIP training strategies. Choose based on how much DFT
data you have and whether a suitable pre-trained foundation model exists.


Training from scratch (default)
---------------------------------

SPARC trains ``num_models`` DeePMD models from random initialisation using
the accumulated dataset. Weights are updated entirely from your DFT data.

Use this when:

- No relevant pre-trained model exists for your chemical system
- You have a large DFT dataset (typically > 500 frames)
- Maximum control over the model architecture is required

Configuration — simply leave ``finetune.enabled: false`` (the default) and
configure ``mlip_setup`` normally.


Fine-tuning a DeePMD universal model
--------------------------------------

Set ``finetune.enabled: true`` to initialise from a pre-trained DeePMD
foundation model (DPA-1, DPA-2, or DPA-3). This requires DeePMD-kit v3
with the PyTorch backend.

Fine-tuning typically converges with far fewer DFT calculations than training
from scratch — often 50–200 frames instead of 500+.


When to use fine-tuning
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Situation
     - Recommendation
   * - Small dataset (< 300 frames)
     - Fine-tuning converges faster
   * - Inorganic materials
     - DPA-3 ``Omat24`` branch is a good starting point
   * - Organic / reactive systems
     - DPA-3 ``Organic_Reactions`` branch
   * - Novel system, no related pre-trained model
     - Train from scratch
   * - Using TF/TF2 DeePMD (v2.x)
     - Train from scratch (fine-tuning requires v3+)


For full configuration options, see :doc:`../finetune`.

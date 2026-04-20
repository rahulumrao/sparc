.. _active_learning_guide:

Active Learning
===============

SPARC uses a **Query-by-Committee (QbC)** strategy to select which structures
to send for DFT labelling. An ensemble of ``num_models`` independently trained
models is used; configurations where the models *disagree* are the most
informative for improving the potential.


Force deviation thresholds
--------------------------

The key tuning parameters are in the ``model_dev`` block:

.. code-block:: yaml

    model_dev:
      f_min_dev: 0.1    # eV/Å — lower bound
      f_max_dev: 0.8    # eV/Å — upper bound

The outcome for each frame depends on its maximum atomic force deviation
across the model committee:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Deviation range
     - Outcome
   * - Below ``f_min_dev``
     - Model is confident — frame discarded (already well described)
   * - Between bounds
     - **Selected as candidate** for DFT labelling
   * - Above ``f_max_dev``
     - Model has no knowledge — frame discarded (likely unphysical)

Typical starting values: ``f_min_dev: 0.05–0.1`` eV/Å,
``f_max_dev: 0.3–0.8`` eV/Å. Tighten the range in later iterations as the
model matures.


Number of models
----------------

``mlip_setup.num_models`` controls committee size. A minimum of 2 is required;
4 is recommended for reliable uncertainty estimates. More models increase
training cost but improve candidate selection quality.


Restarting an interrupted run
------------------------------

If the workflow is killed mid-run, set:

.. code-block:: yaml

    active_learning: true
    learning_restart: true
    latest_model: "iter_000005/01.train/training_1/frozen_model_1.pb"

SPARC will skip already-completed iterations and resume from the checkpoint.


Restart exploration
-------------------

To start the next ML-MD from a specific saved frame (e.g., a candidate
structure) rather than the initial structure, use:

.. code-block:: yaml

    mlip_setup:
      restart_exploration: true
      restart_frame: "candidates"   # use last saved candidate frame

This is useful for directed exploration of transition-state regions.

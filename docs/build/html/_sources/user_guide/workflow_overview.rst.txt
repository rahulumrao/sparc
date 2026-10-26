.. _workflow_overview:

Workflow Overview
=================

SPARC implements an active learning loop for machine learning interatomic
potential (MLIP) development. Each iteration follows three stages that map
directly to sub-directories created inside ``iter_xxxxxx/``:

.. code-block:: text

    iter_000000/
    ├── 00.dft/     ← Stage 1: DFT / AIMD labelling
    ├── 01.train/   ← Stage 2: MLIP training
    └── 02.dpmd/    ← Stage 3: ML-MD + Query-by-Committee

.. image:: ../../_static/images/sparc_flowchart.png
   :alt: SPARC workflow diagram
   :width: 700px
   :align: center


Stage 1 — DFT / AIMD (``00.dft``)
-----------------------------------

Ab initio molecular dynamics (AIMD) or single-point DFT calculations are run
to label new candidate structures. This stage is controlled by the
``aimd_setup`` and ``dft_calculator`` sections of ``input.yaml``.

Set ``aimd_setup.steps: 0`` to skip AIMD entirely and jump straight to
training. In this case, place pre-existing trajectory data as
``AseMD.traj`` (ASE trajectory format) inside the ``00.dft/`` directory
of the current iteration before running.


Stage 2 — MLIP Training (``01.train``)
----------------------------------------

``num_models`` independent MLIP models are trained on the accumulated dataset.
Each model is placed in its own ``training_x/`` sub-directory. Controlled by
``mlip_setup.training`` and ``mlip_setup.input_file``.

For fine-tuning from a pre-trained foundation model instead of training from
scratch, see :doc:`fine_tuning`.


Stage 3 — ML-MD + Query-by-Committee (``02.dpmd``)
----------------------------------------------------

ML-driven molecular dynamics explores configuration space using the trained
committee of models. The force deviation across models (``model_dev_*.out``)
identifies uncertain structures as candidates for DFT relabelling in the next
iteration. Controlled by the ``mlip_setup.MdSimulation`` block.


How sections in ``input.yaml`` map to stages
---------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - ``input.yaml`` section
     - Controls
   * - ``general``
     - Input structure file(s)
   * - ``dft_calculator``
     - DFT engine and template for Stage 1
   * - ``aimd_setup``
     - AIMD run in Stage 1
   * - ``mlip_setup``
     - Training (Stage 2) and ML-MD (Stage 3)
   * - ``finetune``
     - Optional fine-tuning instead of from-scratch training (Stage 2)
   * - ``active_learning``
     - Loop control: iterations, deviation thresholds
   * - ``distance_metrics``
     - Optional geometry sanity checks during ML-MD
   * - ``output``
     - Custom output filenames


Example ``input.yaml`` (color-coded by stage)
---------------------------------------------

The block below uses the same section names as a real SPARC input file.  Stage
banners match the ``iter_xxxxxx/`` sub-directories created at runtime.  See
:doc:`yaml` for every available key.

.. raw:: html

   <p class="workflow-yaml-legend">
     <span class="stage1">Stage 1 — 00.dft</span>
     <span class="stage2">Stage 2 — 01.train</span>
     <span class="stage3">Stage 3 — 02.dpmd</span>
     <span class="stageal">Active Learning</span>
   </p>

.. code-block:: sparc-workflow-yaml

    # [Stage 1 — 00.dft] DFT / AIMD labelling
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
      steps: 100                    # set 0 to skip AIMD and use existing traj
      log_frequency: 1
      thermostat:
        type: "Nose"
        tdamp: 2.0

    # [Stage 2 — 01.train] MLIP ensemble training
    mlip_setup:
      training: true
      data_dir: "Dataset"
      input_file: "input.json"
      num_models: 4                 # committee size for QbC
      train_ratio: 0.8

    # [Stage 3 — 02.dpmd] ML-MD exploration + QbC selection
      MdSimulation: true
      ensemble: "NVT"
      temperature: 300.0
      timestep_fs: 1.0
      md_steps: 2000
      log_frequency: 5
      multiple_run: 1
      restart_exploration: false    # seed later iters from QbC pool
      restart_frame: "candidates"

    # [Active Learning] loop control and candidate filtering
    active_learning: true
    iteration: 10
    min_candidates: 1
    model_dev:
      f_min_dev: 0.05               # eV/Å — below: model confident
      f_max_dev: 0.30               # eV/Å — above: discard as unphysical
      rmsd_threshold: 0.05          # Å — duplicate filter

    # [Optional] geometry guards during ML-MD
    distance_metrics:
      - pair: [0, 3]
        min_distance: 1.2
        max_distance: 5.0

    output:
      dptraj_file: "dpmd.traj"


Loop termination
----------------

The loop runs for ``active_learning.iteration`` cycles. It also stops early
if no candidate structures are found in a given cycle (the model has converged
for the sampled region of phase space).

To resume an interrupted run, set ``learning_restart: true`` and supply
``latest_model`` pointing to the last frozen model checkpoint.

.. _active_learning_guide:

Active Learning
===============

SPARC uses a **Query-by-Committee (QbC)** strategy to select which structures
to send for DFT labelling. An ensemble of ``num_models`` independently trained
models is used; configurations where the models *disagree* are the most
informative for improving the potential.


Force Deviation Thresholds
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


Number of Models
----------------

``mlip_setup.num_models`` controls committee size. A minimum of 2 is required;
4 is recommended for reliable uncertainty estimates. More models increase
training cost but improve candidate selection quality.


Restarting an Interrupted Run
------------------------------

SPARC writes a ``progress.json`` file after every DFT candidate in Step 1 of
each AL iteration. If the workflow crashes — during training, ML-MD, or
mid-DFT — set ``learning_restart: true`` to resume automatically:

.. code-block:: yaml

    active_learning: true
    learning_restart: true

``latest_model`` is **deprecated** and ignored. The model used for ML-MD is
always auto-detected from the current iteration's training directory
(``iter_N/01.train``) after retraining completes. It can be omitted.

SPARC reads ``progress.json`` to determine which AL iteration was in progress
and how many candidates had been labelled, then:

1. Re-runs the last DFT candidate (minor redundancy — identical result).
2. Retrains models from the combined trajectory.
3. Runs ML-MD with the freshly retrained models for that iteration.

.. note::

   ``progress.json`` is only updated during the DFT labelling step. A crash
   during ML-MD causes a restart from the beginning of that iteration's DFT
   labelling — already-labelled candidates are re-processed from the last
   saved index.


Restart Exploration
-------------------

By default each AL iteration seeds its ML-MD from the original input
structure. ``restart_exploration`` changes this so each iteration seeds its
MD from a saved frame:

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
     - Use the last DFT-validated candidate from the previous iteration. Safest
       option — the structure has a high-quality DFT label and is known to lie
       in an interesting region. **Recommended default.**
   * - ``"last"``
     - Use the final frame of the previous ML-MD trajectory. Continues
       exploration from exactly where the last run ended. Useful for directed
       sampling but carries the risk that the last frame is unphysical if the
       model collapsed.
   * - ``"random"``
     - Draw a random frame from the previous ML-MD trajectory. Adds stochastic
       diversity across iterations. Most useful when ``restart_exploration`` is
       combined with ``multiple_run > 1``.

.. note::

   ``restart_exploration`` has no effect in iteration 0 (no previous
   trajectory exists).


.. _deal_filter:

DEAL: Sparse GP Diversity Filter
---------------------------------

After the QbC force-deviation gate, a second filter removes redundant
candidates before DFT labelling. By default this uses a geometric RMSD check
(Kabsch algorithm). When ``deal.enabled: true``, the geometric filter is
replaced by a **Sparse Gaussian Process** (SGP) posterior-variance filter
operating in ACE descriptor space.

Mathematical Foundation
^^^^^^^^^^^^^^^^^^^^^^^

**Local environment descriptors (B2 / ACE).**
Each atom :math:`i` in a candidate frame is mapped to a descriptor vector
:math:`\mathbf{d}_i \in \mathbb{R}^{n_\mathrm{feat}}` by the Atomic Cluster
Expansion B2 basis:

.. math::

   B_i^{nl} = \sum_{j \in \mathcal{N}(i)}
               R_n(r_{ij})\, Y_l(\hat{\mathbf{r}}_{ij})

where :math:`R_n` are Chebyshev radial basis functions up to order
``nmax``, :math:`Y_l` are real spherical harmonics up to degree ``lmax``,
and :math:`\mathcal{N}(i)` denotes neighbours within the cutoff radius.
Descriptors are normalised by the NormalizedDotProduct kernel with
hyper-parameters :math:`\sigma=2` and power :math:`p=2`:

.. math::

   K(\mathbf{d}, \mathbf{d}') =
     \sigma^2 \left(\frac{\mathbf{d} \cdot \mathbf{d}'}
                         {|\mathbf{d}||\mathbf{d}'|}\right)^p

**Sparse Gaussian Process.**
Given a set of inducing-point descriptors
:math:`\mathbf{Z} = \{\mathbf{d}_1, \ldots, \mathbf{d}_M\}` accumulated
from selected frames, the SGP posterior variance for a new local environment
:math:`\mathbf{d}^*` is:

.. math::

   \sigma^2(\mathbf{d}^*)
   = K(\mathbf{d}^*, \mathbf{d}^*)
   - \mathbf{k}^{\top}(\mathbf{d}^*)\,
     \mathbf{K}_{ZZ}^{-1}\,
     \mathbf{k}(\mathbf{d}^*)

where :math:`\mathbf{k}(\mathbf{d}^*) = [K(\mathbf{d}^*, \mathbf{d}_j)]_{j=1}^{M}`
is the cross-covariance vector and :math:`\mathbf{K}_{ZZ}` is the
:math:`M \times M` kernel matrix over inducing points. A frame is selected
when any of its atomic environments has variance above ``deal.threshold``;
upon selection its descriptors are added to :math:`\mathbf{Z}` (expanding the
inducing set for subsequent candidates).

This greedy sequential selection guarantees that no two selected frames share
a novel environment — unlike RMSD, which is purely geometric and
coordinate-frame-dependent.

Data Flow Through the Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The figure below traces how array shapes transform at each stage.

.. code-block:: text

   dpmd.traj  (ASE binary, N frames, 3×n_at forces per frame)
       │
       ▼  dpdata.LabeledSystem  →  npy sets (coords, forces, energies)
       │
       ▼  dp model-devi  →  model_dev_K.out  (N rows × 4 columns)
          columns: step | max_devi_e | min_devi_f | max_devi_f
       │
       ▼  Force-deviation gate  [min_lim, max_lim]
          keeps C frames  (C << N)
          → candidates.extxyz  (C × (3n_at + 1) per frame)
       │
       ├──(RMSD path)──────────────────────────────────────────┐
       │  Kabsch RMSD vs. reference + accepted set             │
       │  drop if RMSD < rmsd_threshold                        │
       │  → candidates.extxyz  (C' ≤ C frames)                │
       └──(DEAL path)──────────────────────────────────────────┘
          attach_dummy_calc: zeros energy/forces on each frame
          DataConfig(images=C frames)
          FlareConfig: B2 descriptors, NDP kernel
              nmax × lmax  radial × angular channels
              cutoff auto-read from DeepMD input.json rcut
          DEAL.run():
              for each frame in order:
                  compute d* for every atom
                  if max σ²(d*) > threshold:
                      select frame
                      add d* to inducing set Z
          → selected frames list  (S ≤ C frames)
       │
       ▼  CalculateDFTEnergy × S
          appends to  iter_N/00.dft/AseMD.traj

**Array shapes at the DEAL stage** (``n_at`` = atoms per frame, ``C`` = QbC
candidates):

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Variable
     - Shape
     - Notes
   * - ``candidate_frames``
     - ``list[Atoms]``, length *C*
     - Each frame: positions ``(n_at, 3)``, symbols ``(n_at,)``
   * - B2 descriptor per atom
     - ``(n_feat,)`` where ``n_feat ≈ nmax × (lmax+1)²``
     - Computed inside DEAL / FLARE C++ backend
   * - Kernel matrix :math:`K_{ZZ}`
     - ``(M, M)`` growing as frames are selected
     - Cholesky-updated in-place; :math:`M` starts at 0
   * - ``selected`` (output)
     - ``list[Atoms]``, length *S ≤ C*
     - Returned to ``QueryByCommittee``; written to ``candidates.extxyz``

Edge Cases and Error Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**flare-pp not installed.**
``run_deal_filter`` catches ``ImportError`` and returns the original
``candidate_frames`` list unchanged, falling back to the RMSD filter. A
``WARNING [DEAL]`` message is emitted. No crash.

**Empty candidate list (C = 0).**
``run_deal_filter`` returns ``[]`` immediately. ``QueryByCommittee`` logs
zero candidates and sets ``candidate_found = False``, ending the AL loop
if ``candidate_idx < min_candidates``.

**``deal_selected.xyz`` fallback.**
Some DEAL versions write selected frames to a file rather than returning them
via ``selector.selected_frames``. The adapter checks for
``deal_selected.xyz`` and reads it with ``ase.io.read`` if the attribute is
absent. If neither source is available, all candidate frames are returned
with a warning.

**cutoff auto-detection fails.**
If ``mlip_setup.input_file`` is missing or contains no ``rcut`` key,
``_extract_rcut_from_deepmd_json`` warns and falls back to
``_DEAL_DEFAULT_CUTOFF = 5.0`` Å. Override explicitly with ``deal.cutoff``.

**Species mismatch.**
``_get_species`` reads the reference structure file first (authoritative —
covers elements not yet appearing in candidate frames). If the file cannot be
read, it falls back to the union of species in ``candidate_frames``. This
ensures FLARE's species map is always complete.

**Periodic vs. molecular systems.**
``is_periodic`` is derived from the DFT engine:
``GAUSSIAN / ORCA / XTB → False``, all others ``→ True``.
When ``is_periodic = False``, stress training is disabled in
``FlareConfig`` (``stress_training=False``) and ``DEALConfig``
(``force_only=True``), avoiding FLARE errors on non-periodic cells.

Parallelization Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^

DEAL's SGP is **sequential by design**: each frame's variance is evaluated
against the inducing set built from all previously accepted frames. There is
no embarrassingly parallel variant — the order of evaluation affects which
frames are selected. Do not attempt to batch or shuffle ``candidate_frames``
before calling ``run_deal_filter``.

FLARE's C++ kernel evaluations are multi-threaded internally (OpenMP); the
number of threads is controlled by ``OMP_NUM_THREADS``. Set this before
launching SPARC:

.. code-block:: bash

   export OMP_NUM_THREADS=4
   sparc -i input.yaml

The surrounding SPARC workflow (DFT labelling, DeepMD training) is
single-process; parallelism there comes from MPI-parallel VASP / CP2K and
the multi-GPU training launched by ``dp train``.


Enabling DEAL
^^^^^^^^^^^^^

Add the ``deal`` block to ``input.yaml``:

.. code-block:: yaml

    deal:
      enabled: true
      threshold: 0.01     # SGP posterior variance — tune independently
      # cutoff: 6.0       # Å — omit to auto-read rcut from input.json
      nmax: 8             # radial basis functions  [Optional, Default: 8]
      lmax: 4             # angular channels        [Optional, Default: 4]

``deal.threshold`` is dimensionless SGP posterior variance in ACE descriptor
space. It is **not** comparable to ``f_min_dev`` (eV/Å). Start at ``0.01``;
lower values retain more frames, higher values are more aggressive.

To disable (default), omit the block or set ``enabled: false`` — the RMSD
filter is active:

.. code-block:: yaml

    model_dev:
      rmsd_threshold: 0.025   # Å — RMSD filter used when DEAL is off

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

By default, every active-learning iteration seeds ML-MD from the input
structure.  With ``restart_exploration: true``, iteration ≥ 1 draws its
initial geometry from the **QbC candidate pool** of the previous iteration:

.. code-block:: text

    iter_{N-1}/02.dpmd/dft_candidates/candidates.extxyz

What is a QbC candidate?
^^^^^^^^^^^^^^^^^^^^^^^^

After ML-MD, SPARC evaluates each trajectory frame with Query-by-Committee
(QbC): an ensemble of models predicts atomic forces and their spread
(force deviation) is recorded.  Per frame:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Force deviation
     - Outcome
   * - Below ``f_min_dev``
     - Models agree — frame discarded (already learned)
   * - Within ``[f_min_dev, f_max_dev]``
     - **Kept as candidate** — moderate disagreement, worth labeling
   * - Above ``f_max_dev``
     - Models diverge catastrophically — discarded (likely unphysical)

Surviving frames are deduplicated by RMSD and collected in
``candidates.extxyz``.  This shortlist contains locally distinct,
model-uncertain configurations — not arbitrary points along the MD
trajectory.

Why restart from the QbC pool?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without restart exploration, every iteration begins from the same input
geometry.  The model keeps re-sampling the same region of phase space.
Seeding from the QbC pool directs each new ML-MD run outward from
configurations the committee already flagged as uncertain but not yet
fully learned.

Configuration
^^^^^^^^^^^^^

.. code-block:: yaml

    mlip_setup:
      restart_exploration: false   # Seed from QbC pool [Optional, Default: false]
      restart_frame: "candidates"  # Selection strategy [Optional, Default: "candidates"]

Selection strategies
^^^^^^^^^^^^^^^^^^^^

All three modes read from the same QbC pool.  They differ in how parallel
MD shoots (``multiple_run``) are mapped to pool entries:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``restart_frame``
     - Behaviour
   * - ``"candidates"``
     - **Round-robin (default).**  Shoot *k* → candidate *k mod N*.  Each
       parallel trajectory seeds from a different pool entry; the list wraps
       when ``multiple_run > N``.  Deterministic.  Recommended for
       ``multiple_run > 1``.
   * - ``"last"``
     - All shoots seed from the **final** pool entry — the most recently
       flagged uncertain configuration.  Suitable for ``multiple_run: 1``.
   * - ``"random"``
     - **Stochastic resampling.**  Each shoot draws a uniform random entry
       from the pool.  The same frame may seed multiple trajectories.

Worked example
^^^^^^^^^^^^^^

Suppose iteration 1 produces five QbC candidates ``[A, B, C, D, E]``.
Iteration 2 runs with ``restart_exploration: true`` and ``multiple_run: 4``:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - ``restart_frame``
     - Shoot assignments
     - Notes
   * - ``"candidates"``
     - 1→A, 2→B, 3→C, 4→D
     - Systematic coverage of the shortlist
   * - ``"last"``
     - 1→E, 2→E, 3→E, 4→E
     - All continue from the same geometry
   * - ``"random"``
     - 1→C, 2→A, 3→C, 4→E
     - Draws may repeat (e.g. shoots 1 and 3 both → C)

.. note::

   Active from iteration 1 onward.  Falls back to the input structure if
   ``candidates.extxyz`` is absent.


.. _deal_filter:

DEAL: Sparse GP Diversity Filter
---------------------------------

After the QbC force-deviation gate, a second filter removes redundant
candidates before DFT labelling. By default this uses a geometric RMSD check
(Kabsch algorithm). When ``deal.enabled: true``, the geometric filter is
replaced by a **Sparse Gaussian Process** (SGP) posterior-variance filter
operating in ACE descriptor space.

.. note::

   SPARC targets **DEAL >= 3.0**, which carries its own native sparse-GP
   extension — ``flare`` / ``flare-pp`` is no longer a dependency and
   ``FlareConfig`` was renamed ``SGPConfig``. Older DEAL releases are refused
   with a warning and the run falls back to the RMSD filter. Install with:

   .. code-block:: bash

      git clone https://github.com/luigibonati/DEAL.git && cd DEAL
      conda env create -f environment-deal.yml && conda activate deal
      python -m pip install -e . --no-build-isolation

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

Selection Modes
^^^^^^^^^^^^^^^

**Threshold mode** (default) makes one pass at a fixed ``deal.threshold``. The
number of surviving frames is whatever the trajectory happens to yield, which
is awkward when the DFT budget per AL iteration is the real constraint.

**Budget mode** (``deal.max_selected: N``) instead targets a frame count. It
runs repeated passes over the candidates, lowering the threshold as
:math:`f^{i+1}` (``threshold_factor``, default 0.7) on pass :math:`i`, and stops
as soon as *N* frames are collected or ``max_iterations`` passes are spent:

.. code-block:: text

   pass 1  threshold 0.700   pool 120 frames   →  4 selected
   pass 2  threshold 0.490   pool 116 frames   →  9 selected
   pass 3  threshold 0.343   pool 111 frames   → 20 selected   → budget reached

The **same GP object is reused across passes** (``DEAL.configure_run``), so the
inducing set keeps growing and a frame rejected in pass 1 is re-judged against
everything learned since. Already-selected frames leave the pool; from pass 2
on the remaining pool is shuffled so trajectory order does not bias which
frames win the lowered threshold. Frames carry their trajectory index in
``info['original_frame']`` so bookkeeping survives that shuffle.

Per-Atom Candidate Mask
^^^^^^^^^^^^^^^^^^^^^^^

``deal.mask`` restricts GP evaluation to atoms flagged by a per-atom
uncertainty array. Atoms outside the mask get uncertainty :math:`-1`, so they
can neither trigger a selection nor enter the sparse set. On reactive,
interface and defect systems — where a handful of atoms drive the chemistry and
the rest are well-known bulk — this both cuts prediction time and stops
thermal noise in the bulk from spending DFT slots.

The mask is derived from a per-atom array in ``frame.arrays``:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Name
     - Level
     - Meaning
   * - ``max_devi_f``
     - frame (``info``)
     - Scalar QbC force deviation; drives the ``[f_min_dev, f_max_dev]`` gate
   * - ``per_atom_devi_f``
     - atom (``arrays``)
     - One force deviation per atom; source for the mask
   * - ``deal_mask``
     - atom (``arrays``)
     - Boolean eligibility array DEAL actually reads

.. warning::

   SPARC does not produce ``per_atom_devi_f`` yet — it needs
   ``dp model-devi --atomic``, which is not wired into
   :func:`QueryByCommittee`. Until it is, enable the mask only if you attach
   the array yourself; frames missing it disable masking with a warning
   instead of failing the run. Set ``deal.mask`` to a string to read a
   differently named array.

With ``mask_threshold`` unset, DEAL picks thresholds automatically: per frame,
``mask_fraction × frame_max`` (default 0.3), and frames whose maximum falls
outside ``[1.1 × mean, 4.0 × mean]`` across the trajectory get an empty mask —
those frames can then never be selected.

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
          SGPConfig: B2 descriptors, NDP kernel
              nmax × lmax  radial × angular channels
              cutoff auto-read from DeepMD input.json rcut
          optional mask: deal_mask marks eligible atoms
          DEAL.run():
              for each frame in order:
                  compute d* for every eligible atom
                  if max σ²(d*) > threshold:
                      select frame
                      add d* to inducing set Z
          budget mode: repeat with threshold ×= threshold_factor,
              same GP, pool minus already-selected frames
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
     - Computed inside DEAL's native sparse-GP extension (C++/pybind11)
   * - Kernel matrix :math:`K_{ZZ}`
     - ``(M, M)`` growing as frames are selected
     - Cholesky-updated in-place; :math:`M` starts at 0
   * - ``selected`` (output)
     - ``list[Atoms]``, length *S ≤ C*
     - Returned to ``QueryByCommittee``; written to ``candidates.extxyz``

Edge Cases and Error Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**DEAL missing or older than 3.0.**
``run_deal_filter`` warns, prints the install recipe and returns the original
``candidate_frames`` list unchanged, falling back to the RMSD filter. A DEAL
2.x install is detected from its package version *before* the import is
attempted, so its ``import flare`` failure never reaches the user as a
traceback. No crash either way.

**Empty candidate list (C = 0).**
``run_deal_filter`` returns ``[]`` immediately. ``QueryByCommittee`` logs
zero candidates and sets ``candidate_found = False``, ending the AL loop
if ``candidate_idx < min_candidates``.

**``deal_selected.xyz`` missing.**
DEAL writes its selection to ``<output_prefix>_selected.xyz``, which the
adapter reads back with ``ase.io.read`` — the round-trip is what preserves
``frame.info`` metadata for the summary CSV. If the file is absent, all
candidate frames are returned with a warning.

**Unknown key in ``deal.advanced``.**
Raises ``ValueError`` listing the accepted names, before any GP work starts. A
silently ignored typo would change the selection science, so this one fails
loudly rather than warning.

**cutoff auto-detection fails.**
If ``mlip_setup.input_file`` is missing or contains no ``rcut`` / ``r_max``
key, ``_extract_rcut_from_deepmd_json`` warns and falls back to
``_DEAL_DEFAULT_CUTOFF = 5.0`` Å. Override explicitly with ``deal.cutoff``.

**Species mismatch.**
``_get_species`` reads the reference structure file first (authoritative —
covers elements not yet appearing in candidate frames). If the file cannot be
read, it falls back to the union of species in ``candidate_frames``. This
ensures the GP species map is always complete.

**Periodic vs. molecular systems.**
``is_periodic`` is derived from the DFT engine:
``GAUSSIAN / ORCA / XTB → False``, all others ``→ True``.
When ``is_periodic = False``, ``DEALConfig(force_only=True)`` trains on forces
alone, and molecular frames are wrapped in a vacuum box for the duration of
the descriptor evaluation — a zero cell yields an empty neighbour list and
therefore zero variance on every atom. The box is stripped again before the
frames reach the DFT stage.

**Budget overshoot.**
The final budget pass can push past ``max_selected``; the adapter keeps the
first *N* frames (selection order is uncertainty-driven) and rewrites
``deal_selected.xyz`` to match.

Parallelization Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^

DEAL's SGP is **sequential by design**: each frame's variance is evaluated
against the inducing set built from all previously accepted frames. There is
no embarrassingly parallel variant — the order of evaluation affects which
frames are selected. Do not attempt to batch or shuffle ``candidate_frames``
before calling ``run_deal_filter``.

DEAL's native kernel evaluations are multi-threaded internally (OpenMP). The
adapter caps threads at ``min(half the CPUs, 4)`` so the GP does not starve
concurrent DFT jobs; override with ``advanced.omp_threads`` or by exporting
``OMP_NUM_THREADS`` before launching SPARC (an existing value always wins):

.. code-block:: bash

   export OMP_NUM_THREADS=4
   sparc -i input.yaml

``MKL_THREADING_LAYER=GNU`` is set for the same reason as before: DEAL's
extension links Eigen, which compiles with ``EIGEN_USE_MKL_ALL`` whenever
LAPACK resolves to MKL, and some conda builds ship an inconsistent
``libmkl_avx512.so`` that aborts at load.

The surrounding SPARC workflow (DFT labelling, DeepMD training) is
single-process; parallelism there comes from MPI-parallel VASP / CP2K and
the multi-GPU training launched by ``dp train``.


Enabling DEAL
^^^^^^^^^^^^^

One line is enough:

.. code-block:: yaml

    deal:
      enabled: true

The full user-facing surface is five keys — everything else has a default:

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Key
     - Default
     - Meaning
   * - ``enabled``
     - ``false``
     - ``true`` → SGP filter; ``false`` → Kabsch RMSD filter
   * - ``threshold``
     - ``0.01``
     - SGP posterior variance cut (threshold mode only)
   * - ``max_selected``
     - ``null``
     - Set an integer to switch to budget mode and target that frame count
   * - ``cutoff``
     - ``null``
     - Descriptor cutoff in Å; ``null`` auto-reads ``rcut`` / ``r_max``
   * - ``mask``
     - ``false``
     - ``true`` (or an array name) to restrict the GP to flagged atoms

``deal.threshold`` is dimensionless SGP posterior variance in ACE descriptor
space. It is **not** comparable to ``f_min_dev`` (eV/Å). Start at ``0.01``;
lower values retain more frames, higher values are more aggressive. In budget
mode it is ignored — the decaying threshold is used instead:

.. code-block:: yaml

    deal:
      enabled: true
      max_selected: 20    # 20 DFT calculations per AL iteration, no tuning

Advanced Options
^^^^^^^^^^^^^^^^

The rest of the DEAL API lives under ``deal.advanced``. Keys are routed to the
object each belongs to and validated on entry, so a typo raises instead of
being silently ignored:

.. code-block:: yaml

    deal:
      enabled: true
      max_selected: 20
      advanced:
        threshold_factor: 0.6       # faster decay → more frames per pass
        max_atoms_added: 0.1        # cap environments added per frame
        save_full_trajectory: true  # per-atom uncertainty for every frame

.. list-table::
   :header-rows: 1
   :widths: 16 26 20 38

   * - Group
     - Keys
     - Defaults
     - Notes
   * - GP selection
     - ``update_threshold``, ``max_atoms_added``, ``min_steps_with_model``,
       ``initial_atoms``
     - ``null``, ``0.2``, ``1``, ``null``
     - ``update_threshold: null`` tracks ``0.8 × threshold``, so it keeps
       following the decay in budget mode
   * - Descriptor
     - ``nmax``, ``lmax``, ``radial_basis``, ``cutoff_function``
     - ``8``, ``4``, ``chebyshev``, ``cosine``
     - B2 basis size; cost grows as ``nmax × (lmax+1)²``
   * - Kernel
     - ``sigma``, ``power``
     - ``2.0``, ``2``
     - NormalizedDotProduct hyper-parameters
   * - GP
     - ``variance_type``
     - ``local``
     - ``local`` is the only value that keeps DEAL's fast path
   * - Mask
     - ``mask_threshold``, ``mask_upper_threshold``, ``mask_mode``,
       ``mask_fraction``, ``mask_plot``
     - ``null``, ``null``, ``above``, ``0.3``, ``false``
     - Used only when ``deal.mask`` is set
   * - Training/output
     - ``train_hyps``, ``save_gp``, ``save_full_trajectory``, ``verbose``
     - ``false``, ``false``, ``false``, ``true``
     - ``train_hyps`` and ``save_gp`` both disable the local-uncertainty fast
       path — expect a slower run
   * - Budget/runtime
     - ``max_iterations``, ``threshold_factor``, ``seed``, ``omp_threads``
     - ``10``, ``0.7``, ``24``, ``null``
     - Handled by SPARC's budget loop, not passed to DEAL

``threshold``, ``max_selected``, ``cutoff`` and ``mask`` are rejected inside
``advanced`` — they belong at the top level.

Extra output files land next to ``candidates.extxyz``:
``deal_selected.xyz`` always, plus ``deal_sgp.json`` (``save_gp``),
``deal_trajectory_uncertainty.xyz`` with a per-atom ``atomic_uncertainty``
array (``save_full_trajectory``) and ``deal_mask.png`` (``mask_plot: true``).

To disable (default), omit the block or set ``enabled: false`` — the RMSD
filter is active:

.. code-block:: yaml

    model_dev:
      rmsd_threshold: 0.025   # Å — RMSD filter used when DEAL is off

.. default-role:: math

Plumed
======

.. module:: plumed_wrapper

Overview
--------

SPARC integrates the open-source `PLUMED <plumed_>`_ library via the ASE
`PLUMED wrapper <_asePlumed>`_ to enable enhanced sampling during molecular
dynamics. Any PLUMED collective variable (CV) or bias can be used — the
examples here are illustrative, not prescriptive.

SPRINT Coordinates
------------------

Social Permutation Invariant (``SPRINT``) coordinates are one example of a
CV that works well for reactive and configurational exploration. They are
constructed from a contact matrix built on equilibrium distances between atom
types, and are invariant to atomic permutation. SPRINT is based on graph
theory and provides a universal descriptor of chemical space.

By definition SPRINT coordinates are calculated from the largest eigenvalue
`\lambda` of an `n \times n` adjacency matrix and its corresponding
eigenvector `\bf{V}`:

.. math::

   s_{i} = \sqrt{n} \lambda \mathit{v_i}

.. note::
   ``SPRINT`` is part of the ``adjmat`` module and requires PLUMED to be
   compiled with the correct flag. See the PLUMED section in :ref:`InstalltionGuide`.

.. tip::
   SPRINT coordinates combine well with `Parallel Bias Metadynamics (PBMetaD) <pbmetad>`_
   for self-guided exploration of complex chemical and configurational spaces,
   but any PLUMED bias (metadynamics, harmonic restraints, funnel metadynamics,
   etc.) can be attached in the same way.


Enabling PLUMED in ``input.yaml``
----------------------------------

PLUMED can be enabled for both the AIMD stage and the ML-MD stage
independently. Set ``plumed.enabled: true`` and point to a PLUMED input file:

.. code-block:: yaml

   aimd_setup:
     plumed:
       enabled: true
       plumed_file: "plumed_dft.dat"   # PLUMED input for AIMD stage
       kT: 0.02585                      # kT in eV (300 K)
       restart: false

   mlip_setup:
     plumed:
       enabled: true
       plumed_file: "plumed.dat"        # PLUMED input for ML-MD stage
       kT: 0.02585
       restart: false


Umbrella Sampling
-----------------

SPARC supports umbrella sampling across multiple windows via the
``umbrella_sampling`` block inside ``mlip_setup.plumed``. Each window runs
an independent ML-MD trajectory with its own pre-equilibrated structure and
PLUMED restraint file. Windows execute sequentially; output for window ``N``
is written to ``iter_000000/02.dpmd/window_NNN/``.

Recommended directory layout — keep all window structures and PLUMED files
in a ``configs/`` subdirectory alongside ``input.yaml``:

.. code-block:: text

   Project Root/
   ├── structure.xyz
   ├── input.yaml
   ├── umbrella_sampling.yaml
   └── configs/
       ├── config_14.xyz          (structure near CV = 1.4 Å)
       ├── config_17.xyz
       ├── ...
       ├── plumed_window_000.dat  (restraint at 1.4 Å)
       ├── plumed_window_001.dat  (restraint at 1.7 Å)
       └── ...

**Step 1 — Enable umbrella sampling in** ``input.yaml``

.. code-block:: yaml

   mlip_setup:
     MdSimulation: true
     ensemble: "NVT"
     temperature: 300.0
     timestep_fs: 1.0
     md_steps: 50000                   # steps per window
     log_frequency: 50
     thermostat:
       type: "Nose"
       tdamp: 25.0
     plumed:
       enabled: true
       kT: 0.02585
       umbrella_sampling:
         enabled: true
         config_file: "umbrella_sampling.yaml"

**Step 2 — Define windows in** ``umbrella_sampling.yaml``

Each entry specifies a starting structure and PLUMED file for that window:

.. code-block:: yaml

   umbrella_windows:
     - structure: "configs/config_14.xyz"
       plumed_file: "configs/plumed_window_000.dat"   # CV centre: 1.4 Å
     - structure: "configs/config_17.xyz"
       plumed_file: "configs/plumed_window_001.dat"   # CV centre: 1.7 Å
     - structure: "configs/config_20.xyz"
       plumed_file: "configs/plumed_window_002.dat"   # CV centre: 2.0 Å

**Step 3 — Write a PLUMED restraint file for each window**

Use ``UNITS LENGTH=A TIME=fs ENERGY=eV`` to match SPARC's unit convention.
Label the restraint so its bias energy is accessible in ``PRINT``:

.. code-block:: text

   # plumed_window_001.dat — window centred at 1.7 Å
   UNITS LENGTH=A TIME=fs ENERGY=eV

   d1: DISTANCE ATOMS=1,8

   rest: RESTRAINT ARG=d1 AT=1.7 KAPPA=50.0

   PRINT ARG=d1,rest.bias FILE=COLVAR STRIDE=50

   FLUSH STRIDE=250

Each window gets a different ``AT`` value. Typical ``KAPPA`` range:
10–100 eV/Å² depending on required overlap between adjacent windows.

**Output**

For each window SPARC writes to ``iter_000000/02.dpmd/window_NNN/``:

.. code-block:: text

   window_000/
   ├── input.xyz       (copy of starting structure)
   ├── COLVAR          (CV value and bias energy vs time)
   └── PLUMED.log

COLVAR columns: time [fs], CV [Å], bias [eV]:

.. code-block:: text

   #! FIELDS time d1 rest.bias
    0.000000  1.455013  0.075661
    50.000000  1.454005  0.072914

Collect ``COLVAR`` files from all windows and pass to WHAM or ``pymbar`` to
recover the potential of mean force (PMF).

.. note::
   Umbrella sampling can be combined with active learning by setting
   ``active_learning: true``. SPARC runs QbC model-deviation analysis on each
   window's trajectory and collects uncertain structures for DFT labelling,
   improving model accuracy along the CV path.


Module Contents
---------------

.. automodule:: sparc.src.plumed_wrapper
   :members:
   :undoc-members:
   :show-inheritance:


.. _plumed: https://www.plumed.org/
.. _asePlumed: https://wiki.fysik.dtu.dk/ase//ase/calculators/plumed.html
.. _pbmetad: https://www.plumed.org/doc-v2.9/user-doc/html/_p_b_m_e_t_a_d.html

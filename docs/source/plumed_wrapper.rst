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
an independent ML-MD trajectory with its own structure and PLUMED restraint
file.

**Step 1 — Enable umbrella sampling in** ``input.yaml``

.. code-block:: yaml

   mlip_setup:
     MdSimulation: true
     ensemble: "NVT"
     temperature: 300.0
     timestep_fs: 1.0
     md_steps: 5000
     thermostat:
       type: "Nose"
       tdamp: 2.0
     plumed:
       enabled: true
       kT: 0.02585
       umbrella_sampling:
         enabled: true
         config_file: "umbrella_sampling.yaml"   # window definitions

**Step 2 — Define windows in** ``umbrella_sampling.yaml``

Each entry in ``umbrella_windows`` specifies a starting structure and a
PLUMED input file that applies the restraint for that window:

.. code-block:: yaml

   umbrella_windows:
     - structure: "window_0/input.xyz"
       plumed_file: "window_0/plumed_us.dat"
     - structure: "window_1/input.xyz"
       plumed_file: "window_1/plumed_us.dat"
     - structure: "window_2/input.xyz"
       plumed_file: "window_2/plumed_us.dat"

**Step 3 — Write a PLUMED restraint file for each window**

A typical harmonic restraint on a distance CV:

.. code-block:: text

   # plumed_us.dat — window centred at d = 2.5 Å
   UNITS LENGTH=A ENERGY=eV TIME=fs

   d: DISTANCE ATOMS=1,2

   RESTRAINT ARG=d AT=2.5 KAPPA=100.0

   PRINT ARG=d FILE=colvar.dat STRIDE=10

Each window gets a different ``AT`` value stepping along the CV. The output
``colvar.dat`` files from all windows can then be combined with WHAM or
`pymbar` to recover the free energy profile.

.. note::
   Each window runs sequentially. Output for window ``N`` is written to
   ``02.dpmd/window_NNN/``.


Module Contents
---------------

.. automodule:: sparc.src.plumed_wrapper
   :members:
   :undoc-members:
   :show-inheritance:


.. _plumed: https://www.plumed.org/
.. _asePlumed: https://wiki.fysik.dtu.dk/ase//ase/calculators/plumed.html
.. _pbmetad: https://www.plumed.org/doc-v2.9/user-doc/html/_p_b_m_e_t_a_d.html

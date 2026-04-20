.. _plumed_guide:

Enhanced Sampling with PLUMED
==============================

SPARC integrates `PLUMED <https://www.plumed.org/>`_ as an auxiliary
calculator for both AIMD and ML-MD, enabling biased sampling to explore
regions of configuration space that plain MD would rarely visit.

PLUMED can be enabled independently for each MD stage in ``input.yaml``:

.. code-block:: yaml

    # Enable for AIMD (Stage 1)
    aimd_setup:
      plumed:
        enabled: true
        plumed_file: "plumed_dft.dat"
        kT: 0.02585        # 300 K in eV
        restart: false

    # Enable for ML-MD (Stage 3)
    mlip_setup:
      plumed:
        enabled: true
        plumed_file: "plumed.dat"
        kT: 0.02585
        restart: false


Any PLUMED collective variable or bias can be used
--------------------------------------------------

Because SPARC simply attaches PLUMED as an auxiliary calculator, any
collective variable or enhanced sampling method supported by your PLUMED
installation can be used — metadynamics, umbrella sampling, steered MD,
free energy perturbation, etc.

The :doc:`../tutorial` shows one specific example using **SPRINT**
coordinates with **Parallel Bias Metadynamics (PBMetaD)** for reactive
sampling of the NH\ :sub:`3`\ BH\ :sub:`3` system. This combination
worked well for that case, but it is not a requirement — choose the
collective variables and bias that suit your system.

Example: SPRINT coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SPRINT** (Social PeRmutatIon iNvarianT) coordinates are graph-based
collective variables derived from a contact matrix. They:

- Are permutation-invariant (atom relabelling does not change the coordinate)
- Discriminate chemical environments without prior knowledge of reaction paths
- Generalise across bond-breaking and bond-forming events

By definition, SPRINT coordinates are computed from the largest eigenvalue
:math:`\lambda` of the :math:`n \times n` adjacency matrix and its
corresponding eigenvector :math:`\mathbf{V}`:

.. math::

   s_i = \sqrt{n} \, \lambda \, v_i

.. note::

   ``SPRINT`` is part of the ``adjmat`` PLUMED module and requires PLUMED to
   be compiled with ``--enable-modules=all``. See :ref:`InstalltionGuide`
   for instructions.

.. tip::

   See the :doc:`../tutorial` for a complete ``plumed_dp.dat`` input file
   using SPRINT + PBMetaD on the NH\ :sub:`3`\ BH\ :sub:`3` system.


Umbrella sampling
-----------------

SPARC also supports umbrella sampling windows for free-energy calculations.
Enable via:

.. code-block:: yaml

    mlip_setup:
      plumed:
        enabled: true
        plumed_file: "plumed.dat"
        umbrella_sampling:
          enabled: true
          config_file: "umbrella_sampling.yaml"

See :doc:`../plumed_wrapper` for technical details on the PLUMED wrapper
module.

.. raw:: html

   <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
     <img src="_static/sparc_logo.png" alt="SPARC Logo" style="height: 150px; object-fit: contain;">
     <span style="font-size: 1.5em; font-weight: bold;">Welcome to SPARC's documentation!</span>
   </div>

**SPARC** (**S**\ mart **P**\ otential with **A**\ tomistic **R**\ are Events and **C**\ ontinuous Learning)
is a Python package that implements an automated active learning workflow for developing machine learning
interatomic potentials (MLIPs). It is built on top of `ASE <ase_>`_ and ties together DFT labelling,
MLIP training, and ML-driven molecular dynamics into a single iterative loop.

Scientific Overview
-------------------

SPARC employs an active learning loop to iteratively refine machine learning potentials using
first-principles calculations. Each iteration runs three stages: DFT/AIMD labelling (``00.dft``),
MLIP training (``01.train``), and ML-MD with Query-by-Committee candidate selection (``02.dpmd``).
The loop continues until the model converges for the sampled region of phase space.
See :doc:`user_guide/workflow_overview` for a full description.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   quickstart
   yaml
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Modules

   calculator
   deepmd
   finetune
   md
   DataProcess
   plumed_wrapper
   analysis
   workflow
   ChemView

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   contributing

Key Features
------------

- **Six DFT engines** — VASP, CP2K, ORCA, xTB, Quantum ESPRESSO, and Gaussian via `ASE <ase_>`_ calculators
- **DeepMD-kit v2 and v3** — supports TensorFlow and PyTorch backends; automatically detected at runtime
- **GNN potentials** — MACE and NequIP training via `deepmd-gnn <deepmd_gnn_>`_ using the same workflow
- **Fine-tuning** — initialise from pre-trained DPA-3 universal models instead of training from scratch
- **Active learning** — Query-by-Committee force deviation selects uncertain structures for DFT relabelling
- **Enhanced sampling** — `PLUMED <plumed_>`_ integration for metadynamics, umbrella sampling, and any CV/bias


Indices and Tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _vasp: https://www.vasp.at/
.. _deepmd: https://github.com/deepmodeling/deepmd-kit
.. _deepmd_gnn: https://github.com/deepmodeling/deepmd-gnn
.. _plumed: https://www.plumed.org/
.. _ase: https://wiki.fysik.dtu.dk/ase/

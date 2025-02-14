Welcome to SPARC's documentation!
=================================

**SPARC** (**S**\ mart **P**\ otential with **A**\ tomistic **R**\ are Events and **C**\ ontinuous Learning) 
is a Python package that implements an active learning workflow for developing accurate machine learning potentials.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api/index
   examples/index
   contributing

Features
--------

* Ab initio Molecular Dynamics (AIMD) using VASP
* Machine learning potential training with DeepMD-kit
* Deep Potential Molecular Dynamics (DPMD) simulations
* Active learning for continuous model improvement
* Enhanced sampling with PLUMED integration

Requirements
------------

Core Dependencies:

* Python 3.xx
* DeepMD-kit 2.2.10
* ASE (Atomic Simulation Environment)
* VASP (for DFT calculations)
* PLUMED (for enhanced sampling)
* MPI library

Python Package Dependencies:

* numpy
* pandas
* dpdata

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 
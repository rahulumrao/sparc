.. SPARC documentation master file, created by
   sphinx-quickstart on Tue Mar 25 19:57:53 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
     <img src="_static/sparc_logo.png" alt="SPARC Logo" style="height: 150px; object-fit: contain;">
     <span style="font-size: 1.5em; font-weight: bold;">Welcome to SPARC's documentation!</span>
   </div>


.. Welcome to SPARC's documentation!
.. =================================

**SPARC** (**S**\ mart **P**\ otential with **A**\ tomistic **R**\ are Events and **C**\ ontinuous Learning) 
is a Python package build around the `ASE <ase_>`_ wrapper that implements an automated workflow of developing machine learning potential for reactive chemical systems. 
This package is designed to work seamlessly within the Python framewrok, providing users with powerful tools for efficient simulation and model improvement.

Scientific Overview
--------------------
SPARC workflow employs an active learning protocol to iteratively refine machine learning potentials using first-principles calculations. 
This enables users to develop highly accurate potentials for molecular dynamics simulations, which is essential for studying 
complex materials and chemical processes.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   install.rst
   quickstart
   yaml
   calculator
   deepmd
   md
   DataProcess
   plumed_wrapper
   analysis
   workflow
   user_guide/index
   api/index
   tests/index
   contributing

Key Features
------------
SPARC provides the following core functionalities:

- **First-principles calculations** with `VASP <vasp_>`_ and `CP2K <cp2k_>`_ for first-principle calculations
- **Machine learning model training** with `DeepMD-kit <deepmd_>`_ package
- **Machine learning molecular dynamics (ML/MD)** simulations using `ASE <ase_>`_ MD engine for efficient and scalable modeling
- **Active learning** for continuous model improvement through iterative training and data selection
- **Potential Energy Surface Exploration** through integration with `PLUMED <plumed_>`_ library

Use Cases
---------
Here are some example use cases for SPARC:

1. **Material Property Prediction:** Use SPARC to develop accurate interatomic potentials for molecular dynamics simulations to predict the properties of new materials.
2. **Chemical Reaction Pathways:** Apply active learning to uncover reaction mechanisms by training a potential based on sparse, high-quality data points.
3. **High-Throughput Simulations:** Combine SPARC's active learning protocol with high-throughput simulation to explore large chemical spaces efficiently.

Installation
------------
For detailed installation instructions, please refer to the `Installation Guide <install.html>`_.

Indices and Tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 
  
.. _vasp: https://www.vasp.at/
.. _cp2k: https://www.cp2k.org/
.. _deepmd: https://github.com/deepmodeling/deepmd-kit
.. _plumed: https://www.plumed.org/
.. _ase: https://wiki.fysik.dtu.dk/ase/
Installation
===========

This guide will help you install SPARC and its dependencies.

Prerequisites
-------------

Before installing SPARC, ensure you have the following prerequisites:

* Python 3.xx
* Conda package manager
* MPI library
* VASP (for DFT calculations)
* PLUMED (optional, for enhanced sampling)

Step-by-Step Installation
-------------------------

1. Create and activate a conda environment:

   .. code-block:: bash

      conda create -n sparc python=3.10
      conda activate sparc

2. Install PLUMED:

   .. code-block:: bash

      conda install -c conda-forge py-plumed

3. Install DeepMD-kit:

   .. code-block:: bash

      pip install deepmd-kit==2.2.10

4. Clone and install SPARC:

   .. code-block:: bash

      git clone https://github.com/rahulumrao/sparc.git
      cd sparc
      pip install .

Environment Setup
-----------------

Set up VASP POTCAR files path:

.. code-block:: bash

   export VASP_PP_PATH=/path/to/vasp/potcar_files

Verification
------------

To verify your installation:

.. code-block:: python

   import sparc
   print(sparc.__version__) 
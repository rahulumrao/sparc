.. _InstalltionGuide:
SPARC Installation Guide
========================

This guide provides step-by-step instructions to set up SPARC, an automated workflow for 
training machine learning potential for reactive chemical system
along with its core dependencies such as DeepMD-kit, PLUMED, and VASP.



.. Quick Start
.. -----------

.. For experienced users, the basic setup steps are:

.. .. code-block:: bash

..    conda create -n sparc python=3.10
..    conda activate sparc
..    pip install deepmd-kit[gpu,cu12,lmp]
..    git clone https://github.com/rahulumrao/sparc.git && cd sparc
..    pip install .


Core Software Dependencies:
---------------------------

* Python 3.9 or 3.10 (recommended)
* DeepMD-kit 2.2.10
* ASE (Atomic Simulation Environment)
* VASP (required for DFT calculations)
* PLUMED (for exploration of PES)

Python Package Dependencies:
----------------------------

* numpy
* pandas
* dpdata

Step-by-Step Installation
-------------------------

1. Create and activate a conda environment:

   .. code-block:: bash

      conda create -n sparc python=3.10
      conda activate sparc

2. Install  `DeepMD-kit <dpmd_install_>`_, this command installs the GPU-enabled version of DeepMD-kit:

.. code-block:: bash

   conda install deepmd-kit=2.2.10=*gpu libdeepmd=2.2.10=*gpu lammps horovod -c https://conda.deepmodeling.com -c defaults

3. Clone and install SPARC:

   .. code-block:: bash

      git clone https://github.com/rahulumrao/sparc.git
      cd sparc
      pip install .
      
.. _InstallPlumed:
4. Install PLUMED:

   .. code-block:: bash

      conda install -c conda-forge py-plumed

.. note::
   Some Collective Variables (CVs), such as Generic CVs (e.g., ``SPRINT``), are part of the additional module and are not included in a standard PLUMED installation. 
   To enable them, we need to manually install PLUMED and wrap with Python.


   Download the `PLUMED package <https://www.plumed.org/download>`_ from the official website.
   During installation, PLUMED will detect the Python interpreter from the active ``conda environment`` and enable Python bindings.

   .. code-block:: bash

      ./configure --enable-mpi=no --enable-modules=all PYTHON_BIN=$(which python) --prefix=$CONDA_PREFIX
      
      make -j$(nproc) && make install

   Once the installation is complete, you should see a directory named ``plumed`` inside the ``lib`` folder of your ``conda environment``.

   To verify the installation, run the following command in the terminal:

   .. code-block:: python

      >>> ls /home/user/anaconda3/envs/sparc/lib/plumed

   Expected output:

   .. code-block:: ini

      fortran     patches        plumed-mklib  plumed-partial_tempering  plumed-runtime   plumed-vim2html  src
      modulefile  plumed-config  plumed-newcv  plumed-patch              plumed-selector  scripts          vim

   You can also import the module in Python to confirm installation:

   .. code-block:: python

      >>> from ase.calculators import plumed
      >>> from plumed import Plumed

Environment Setup
-----------------

Set up POTCAR file path:

.. code-block:: bash

   export VASP_PP_PATH=/path/to/vasp/potcar_files

Verification
------------

To verify your installation:

.. code-block:: bash

   >>> sparc -h
      
      sparc [-h] [-i INPUT_FILE]

      options:
      -h, --help            show this help message and exit
      -i INPUT_FILE, --input_file INPUT_FILE
                              Input YAML file


.. important::

   The ``pip install tensorflow[and-cuda]`` may not always detect the GPU due to potential configuration issues. 
   To verify if TensorFlow has successfully recognized the GPU, execute the following command:

   ``python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"``
   
   If the output is an empty list, check:

   - Your NVIDIA driver and CUDA toolkit installation
   - The CUDA version compatibility with TensorFlow
   - That your environment variables (e.g., `LD_LIBRARY_PATH`) are correctly set
   
   Also, refer to the `TensorFlow GPU troubleshooting guide <tf_>`_ for details.

.. _dpmd_install: https://docs.deepmodeling.com/projects/deepmd/en/stable/getting-started/install.html
.. _plumed: https://www.plumed.org/download
.. _tf: https://www.tensorflow.org/install/pip
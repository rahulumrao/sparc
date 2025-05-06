DeepMD Setup
============

.. module:: deepmd

Overview
--------

This module provides functionalities to setting up the training of DeepMD ML models, 
freezing and compressing them, and evaluating their accuracy.
Inside each ``iter_00000xx`` directory a ``01.train`` directory will be created, and based on the ``num_models`` 
a subdirectory with name ``traininig_x`` will be created and the training will be performed. 
Training will be done based on the user defined ``input_file``, see :ref:`deepmd_section` section.

.. tip::
   Users are encouraged to train several models for better accuracy.

After training each directory should have a file ``frozen_model.pb``. 
These different models will be used to perform *Query-by-committee* to find the candidates for labelling.

.. note::
   **Query by committee (QbC)**: Identifies the configurations by measuring the disagreement among an ensemble of model. 
   Allows the model to learn only **what it needs to** without wasting resources on redundant data. 
   See also Deepmd-kit `model deviation <qbc_>`_ for more details. 


If the candidates are found then inside the ``02.dft`` directory a subdirecoty ``dft_candidates`` will be created 
with a separate ``POSCAR`` for each candidates.

.. code-block:: bash

   >>> tree dft_candidates
       ├── 0001
       │   └── POSCAR
       ├── 0002
       │   └── POSCAR
       ├── 0003
       │   └── POSCAR
       ├── 0004
       │   └── POSCAR
       ├── 0005
           └── POSCAR


Features:
   - Setup DeepPotential calculators for ASE atoms objects
   - Train DeepMD models with various configurations
   - Freeze and compress trained models
   - Evaluate trained models

Usage Example
-------------

Here is an example of how to use `setup_DeepPotential` to assign a DeepPotential calculator to an ASE atoms object:

.. code-block:: python

   from deepmd import setup_DeepPotential
   from ase import Atoms

   atoms = Atoms("H2O")
   dp_system, dp_calc = setup_DeepPotential(atoms, "path/to/model")
   print(dp_system.get_potential_energy())



Module Contents
---------------
.. (# change the path to sparc.src.deepmd later)
.. automodule:: src.deepmd 
   :members:
   :undoc-members:
   :show-inheritance:


References
----------

For more details on DeepMD-Kit, visit: https://github.com/deepmodeling/deepmd-kit

.. _qbc: https://docs.deepmodeling.com/projects/deepmd/en/stable/test/model-deviation.html

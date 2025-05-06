Molecular Dynamics
==================

.. module:: ase_md

Overview
--------

This module provides functionalities for running molecular dynamics (MD) simulations using ASE (Atomic Simulation Environment). 
It includes different thermostats and integrators to handle NVT MD simulations.

Each iteration of ML/MD data will be stored in corresponding ``iter_0000xx`` directory. 
Which will have the a corresponding ``log_file`` and ``traj`` file.

.. code-block:: bash

    >>> cat Iter1_dpmd.log
    Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
    0.0000        -112.0807    -112.8950       0.8143   300.0
    0.0700        -111.6322    -112.7149       1.0828   398.9
    0.1400        -112.4215    -113.3518       0.9303   342.7
    0.2100        -112.9996    -113.6775       0.6779   249.8
    0.2800        -112.6910    -113.7220       1.0310   379.8
    0.3500        -112.8007    -113.2903       0.4896   180.4


Features:
---------
- Nose-Hoover and Langevin thermostats for NVT simulations
- *ab-initio* energy calculations
- *ab-initio* and ML molecular dynamics execution

.. - LAMMPS integration for MD simulations

Module Contents
---------------

.. automodule:: src.ase_md
   :members:
   :undoc-members:
   :show-inheritance:

Functions:
----------

.. autofunction:: sparc.src.ase_md.NoseNVT

.. autofunction:: sparc.src.ase_md.LangevinNVT

.. ### CalculateDFTEnergy

.. .. autofunction:: ase_md.CalculateDFTEnergy

.. ### ExecuteAbInitioDynamics

.. .. autofunction:: ase_md.ExecuteAbInitioDynamics

.. ### ExecuteMlpDynamics

.. .. autofunction:: ase_md.ExecuteMlpDynamics

.. ### lammps_md

.. .. autofunction:: ase_md.lammps_md

Usage Examples
--------------

- Nose-Hoover NVT Simulation:

.. code-block:: python

    from ase import Atoms
    from ase_md import NoseNVT

    atoms = Atoms("H2O")
    dyn = NoseNVT(atoms, temperature=300)
    dyn.run(1000)

- Langevin NVT Simulation:
  
.. code-block:: python

    from ase_md import LangevinNVT

    dyn = LangevinNVT(atoms, temperature=300, friction=0.01)
    dyn.run(1000)

- Ab-initio Molecular Dynamics:
  
.. code-block:: python

    from ase_md import ExecuteAbInitioDynamics

    ExecuteAbInitioDynamics(system=atoms, dyn=dyn, steps=500, pace=10, 
                            log_filename="aimd.log", trajfile="aimd.traj", 
                            dir_name="aimd_results", name="DFT_AIMD")

References
----------

For more information on ASE, visit: https://wiki.fysik.dtu.dk/ase/

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

Usage Examples
--------------

The examples below correspond to the ``aimd_setup`` (or ``mlip_setup``)
blocks in ``input.yaml``.  The Python API mirrors those settings directly.

**NVT — Nose-Hoover thermostat**

.. code-block:: yaml

    aimd_setup:
      ensemble: "NVT"
      temperature: 300.0
      timestep_fs: 1.0
      steps: 500
      thermostat:
        type: "Nose"
        tdamp: 2.0

.. code-block:: python

    from ase.io import read
    from sparc.src.ase_md import NoseNVT

    atoms = read("POSCAR")
    dyn = NoseNVT(atoms, temperature=300, tdamp=2.0, timestep=1.0)
    dyn.run(500)

**NVT — Langevin thermostat**

.. code-block:: yaml

    aimd_setup:
      ensemble: "NVT"
      temperature: 300.0
      timestep_fs: 1.0
      steps: 500
      thermostat:
        type: "Langevin"
        friction: 0.01

.. code-block:: python

    from sparc.src.ase_md import LangevinNVT

    dyn = LangevinNVT(atoms, temperature=300, friction=0.01, timestep=1.0)
    dyn.run(500)

**NVT — Temperature ramp (Langevin)**

Temperature ramping linearly from ``temperature`` to ``temp_end`` over the
run. Nose-Hoover resists rapid temperature changes; use Langevin for ramping.

.. code-block:: yaml

    aimd_setup:
      ensemble: "NVT"
      temperature: 300.0
      temp_end: 1000.0
      timestep_fs: 1.0
      steps: 1000
      thermostat:
        type: "Langevin"
        friction: 0.01

.. code-block:: python

    from sparc.src.ase_md import LangevinNVT

    dyn = LangevinNVT(atoms, temperature=300, temp_end=1000,
                      friction=0.01, timestep=1.0)
    dyn.run(1000)

**NVE — microcanonical ensemble**

.. code-block:: yaml

    aimd_setup:
      ensemble: "NVE"
      temperature: 300.0
      timestep_fs: 1.0
      steps: 500

.. code-block:: python

    from sparc.src.ase_md import ExecuteAbInitioDynamics
    from ase.md.verlet import VelocityVerlet
    import ase.units as units

    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
    ExecuteAbInitioDynamics(system=atoms, dyn=dyn, steps=500, pace=10,
                            log_filename="aimd.log", trajfile="aimd.traj",
                            dir_name="aimd_results", name="NVE_AIMD")

**NPT — isothermal-isobaric ensemble**

Pressure is given in **bar** and compressibility in **1/bar**, following
ASE's ``units.bar`` convention (``pressure_au = pressure * units.bar``,
``compressibility_au = compressibility / units.bar``).

.. code-block:: yaml

    aimd_setup:
      ensemble: "NPT"
      temperature: 300.0
      timestep_fs: 1.0
      steps: 500
      tau_t: 100.0               # Berendsen temperature coupling time [fs]
      tau_p: 1000.0              # Berendsen pressure coupling time [fs]
      pressure: 1.01325          # Target pressure [bar]  (1 atm = 1.01325 bar)
      compressibility: 4.57e-5   # Isothermal compressibility [1/bar] (optional; default: Cu ~7.1e-7 bar⁻¹)

.. note::

    For NPT, the ``thermostat.type`` key is **ignored**. ``NPTBerendsen``
    handles both temperature and pressure coupling via ``tau_t`` and ``tau_p``.

.. code-block:: python

    from sparc.src.ase_md import NPT

    dyn = NPT(atoms, timestep=1.0, temperature=300,
              tau_t=100.0, pressure=1.01325, tau_p=1000.0,
              compressibility=4.57e-5)

.. list-table:: NPT parameter units
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Unit
     - Notes
   * - ``pressure``
     - bar
     - 1 atm ≈ 1.01325 bar; 1 GPa = 10 000 bar
   * - ``compressibility``
     - 1/bar
     - Default: 7.1×10⁻⁷ bar⁻¹ (Cu). Water ≈ 4.57×10⁻⁵ bar⁻¹
   * - ``tau_t``
     - fs
     - Berendsen temperature coupling time constant
   * - ``tau_p``
     - fs
     - Berendsen pressure coupling time constant

Module Contents
---------------

.. automodule:: sparc.src.ase_md
   :members:
   :undoc-members:
   :show-inheritance:


References
----------

For more information on ASE, visit: https://wiki.fysik.dtu.dk/ase/

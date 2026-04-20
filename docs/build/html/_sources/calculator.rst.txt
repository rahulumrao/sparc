DFT Calculator
==============

Overview
--------

The ``calculator.py`` module sets up DFT calculators as ASE calculator objects
from the ``dft_calculator`` section of ``input.yaml``. Six engines are supported:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``engine`` key
     - Backend
     - Notes
   * - ``VASP``
     - ``ase.calculators.vasp.Vasp``
     - All parameters read from INCAR template; requires POTCAR files (``VASP_PP_PATH``)
   * - ``CP2K``
     - ``ase.calculators.cp2k.CP2K``
     - Requires ``cp2k_template.inp``
   * - ``ORCA``
     - ``ase.calculators.orca.ORCA``
     - Molecular (non-periodic); periodicity auto-removed from structure
   * - ``xTB``
     - ``ase.calculators.xtb.XTB``
     - Semi-empirical tight-binding; reads ``key = value`` template file; non-periodic — NPT ensemble not supported
   * - ``QE``
     - ``ase.calculators.espresso.Espresso``
     - Quantum ESPRESSO ``pw.x``; k-points default to Gamma-only sampling
   * - ``Gaussian``
     - ``ase.calculators.gaussian.Gaussian``
     - Molecular (non-periodic); periodicity auto-removed from structure


Configuration (``input.yaml``)
-------------------------------

.. code-block:: yaml

    dft_calculator:
      engine: "VASP"                       # Engine name (required)
      template_file: "INCAR"               # Engine-specific input template (required)
      exe_command: "mpirun -np 4 vasp_std" # Executable command (optional; auto-detected if omitted)

Each engine reads its template file differently:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Engine
     - Template file
     - Notes
   * - ``VASP``
     - ``INCAR``
     - Standard VASP INCAR format
   * - ``CP2K``
     - ``cp2k_template.inp``
     - CP2K input file (comments stripped)
   * - ``ORCA``
     - ``orca_template.inp``
     - ORCA simple input format
   * - ``xTB``
     - ``xtb_template.inp``
     - ``key = value`` file
   * - ``QE``
     - ``qe_template.in``
     - Quantum ESPRESSO ``pw.x`` input
   * - ``Gaussian``
     - ``gaussian_template.inp``
     - ``key = value`` file; always non-periodic

.. note::
   For **ORCA** and **Gaussian**, any periodic boundary conditions on the input
   structure are automatically removed before the calculator is attached.

.. note::
   Ensure that the VASP executable and ``VASP_PP_PATH`` are correctly set before
   running. See :ref:`InstalltionGuide` for environment setup.


Template File Reference
-----------------------

Each engine reads its ``template_file`` in a different format. Minimal working
examples are shown below.

**1. VASP** (``INCAR``)

Standard VASP INCAR format. All keys are passed directly to the ASE ``Vasp``
calculator.

.. code-block:: sparc-template

   ISTART = 0
   ICHARG = 2
   ENCUT  = 520
   EDIFF  = 1E-6
   NSW    = 0
   IBRION = -1
   ISMEAR = 0
   SIGMA  = 0.05
   LWAVE  = .FALSE.
   LCHARG = .FALSE.
   NPAR   = 4

**2. CP2K** (``cp2k_template.inp``)

Full CP2K input format. Lines starting with ``#`` or ``!`` are stripped.
ASE injects the ``&COORD`` and ``&CELL`` sections automatically from the
``Atoms`` object.

.. code-block:: sparc-template

   ! This is a comment
   # This is a comment
   &GLOBAL
     ECHO_INPUT
   &END GLOBAL
   &FORCE_EVAL
     &DFT
       CHARGE 0
       MULTIPLICITY 1
       &MGRID
         REL_CUTOFF 50
         NGRIDS 3
       &END MGRID
       &QS
         METHOD GPW
         EPS_DEFAULT 1.0E-10
         EPS_PGF_ORB 1.0E-12
         EXTRAPOLATION ASPC
         EXTRAPOLATION_ORDER 3
       &END QS
       &SCF
         EPS_SCF 1.0E-4
         SCF_GUESS RESTART
         &OT
           PRECONDITIONER FULL_ALL
         &END OT
         &OUTER_SCF
           EPS_SCF 1.0E-4
           MAX_SCF 3
         &END OUTER_SCF
       &END SCF
     &END DFT
     &SUBSYS
       &KIND B
         ELEMENT B
       &END KIND
       &KIND N
         ELEMENT N
       &END KIND
       &KIND H
         ELEMENT H
       &END KIND
     &END SUBSYS
   &END FORCE_EVAL

CP2K parameters (``cp2k:`` section in ``input.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Top-level ASE ``CP2K()`` parameters are supplied in a **separate** ``cp2k:``
section of ``input.yaml`` — not inside ``dft_calculator:``.

**Parameter precedence:** values in ``cp2k:`` override the defaults listed
below. Parameters not present in ``cp2k:`` take their default values.
Settings controlled entirely by the template (``cutoff``,
``pseudo_potential``) default to ``null`` so that ASE does not inject
conflicting values.

.. code-block:: yaml

   dft_calculator:
     engine: "CP2K"
     template_file: "cp2k_template.inp"
     exe_command: "mpirun -np 4 cp2k.psmp"  # optional

   cp2k:
     xc: "PBE"                        # XC functional
     max_scf: 500                      # Max SCF iterations
     basis_set_file: "BASIS_MOLOPT"    # Basis set library file
     potential_file: "GTH_POTENTIALS"  # Pseudopotential library file

.. list-table:: ``cp2k:`` parameters and defaults
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Default
     - Description
   * - ``xc``
     - ``PBE``
     - Exchange-correlation functional
   * - ``max_scf``
     - ``500``
     - Maximum number of outer SCF cycles
   * - ``basis_set_file``
     - ``BASIS_MOLOPT``
     - Filename of the basis set library (``BASIS_SET_FILE_NAME`` in ``&DFT``)
   * - ``basis_set``
     - ``null``
     - Global basis set. When ``null``, specify ``BASIS_SET`` per species
       inside each ``&KIND`` block of the template
   * - ``potential_file``
     - ``GTH_POTENTIALS``
     - Filename of the pseudopotential library (``POTENTIAL_FILE_NAME`` in ``&DFT``)
   * - ``pseudo_potential``
     - ``null``
     - Global pseudopotential. When ``null``, specify ``POTENTIAL`` per species
       inside each ``&KIND`` block of the template
   * - ``label``
     - ``cp2k/job``
     - Output file prefix and working subdirectory

.. note::
   ``basis_set`` and ``pseudo_potential`` default to ``null``. Define ``BASIS_SET``
   and ``POTENTIAL`` inside each ``&KIND`` block of the template to assign
   different basis sets and pseudopotentials per species.

**3. ORCA** (``orca_template.inp``)

Requires one ``!`` keyword line. Charge and multiplicity are read from the
``*xyz`` line. The ``%pal`` block sets parallelism.

.. note::

   ORCA uses MPI for parallel execution (``%pal nprocs N end``). Ensure that
   MPI binaries (``mpirun`` / ``mpiexec``) are on your ``PATH`` and that the
   correct MPI libraries are loaded before running, otherwise ORCA will fall
   back to serial execution or fail to start.

.. code-block:: sparc-template

   ! B3LYP 6-31G(d,p) Engrad
   %scf Convergence tight
    maxiter 300 end
   %pal nprocs 4 end
   *xyz 0 1

**4. xTB** (``xtb_template.inp``)

Simple ``key = value`` file. Comments start with ``#``. Supported keys:

.. code-block:: sparc-template

   # xTB template for SPARC
   method               = GFN2-xTB
   charge               = 0
   multiplicity         = 1
   accuracy             = 1.0
   electronic_temperature = 300.0
   max_iterations       = 250
   solvent              = none
   solvent_method       = none
   directory            = xtb_run
   label                = xtb

**5. Quantum ESPRESSO** (``qe_template.in``)

Standard ``pw.x`` input. ASE injects atomic positions and cell parameters.
The ``ATOMIC_SPECIES`` card maps elements to pseudopotential files.
K-points default to Gamma-only if the ``K_POINTS gamma`` card is used or
the card is omitted.

.. note::
   ``calculation``, ``tprnfor``, and ``tstress`` are always overridden by
   SPARC to ``scf``, ``.true.``, and ``.true.`` respectively — they may be
   included in the template for clarity but have no effect.

.. code-block:: sparc-template

   &CONTROL
     calculation  = 'scf'
     tprnfor      = .true.
     tstress      = .true.
     pseudo_dir   = '/path/to/pseudo'
     outdir       = './qe_out'
     verbosity    = 'low'
   /

   &SYSTEM
     ibrav            = 0
     ecutwfc          = 40
     ecutrho          = 320
     occupations      = 'fixed'
   /

   &ELECTRONS
     conv_thr         = 1.0d-5
     mixing_beta      = 0.3
     electron_maxstep = 100
   /

   ATOMIC_SPECIES
     B   10.811  b_pbe_v1.4.uspp.F.UPF
     N   14.007  N.pbe-n-radius_5.UPF
     H    1.008  H.pbe-rrkjus_psl.1.0.0.UPF

   K_POINTS gamma

**6. Gaussian** (``gaussian_template.inp``)

Simple ``key = value`` file. Comments start with ``#`` or ``!``.
``method`` and ``basis`` are required. ASE always runs a force calculation.

.. code-block:: sparc-template

   # Gaussian template for SPARC
   method       = HF
   basis        = STO-3G
   charge       = 0
   multiplicity = 1
   mem          = 2GB
   nproc        = 2
   scf          = qc,maxcycle=200


Module Contents
---------------

.. automodule:: sparc.src.calculator
   :members:
   :undoc-members:
   :show-inheritance:

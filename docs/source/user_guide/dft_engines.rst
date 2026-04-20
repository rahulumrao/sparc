.. _dft_engines_guide:

Choosing a DFT Engine
=====================

SPARC supports six DFT engines. Set ``dft_calculator.engine`` in
``input.yaml`` to select one. See :doc:`../calculator` for the full API
reference.


VASP
----

Best for periodic solids and surfaces. Requires a commercial VASP licence.

**Requirements:**

- VASP binary (``vasp_std``, ``vasp_gam``, etc.)
- POTCAR files — set ``export VASP_PP_PATH=/path/to/potcar_files``

**Template file:** ``INCAR``

.. code-block:: yaml

    dft_calculator:
      engine: "VASP"
      template_file: "INCAR"
      exe_command: "mpirun -np 16 vasp_std"


CP2K
----

Open-source alternative for periodic systems (solids, surfaces, liquids).

**Requirements:**

- CP2K binary + basis-set/pseudopotential library
- ``cp2k_template.inp`` in the project root

.. code-block:: yaml

    dft_calculator:
      engine: "CP2K"
      template_file: "cp2k_template.inp"


ORCA
----

Suitable for molecular systems requiring high-level correlated methods
(e.g., CCSD(T), hybrid DFT). Non-periodic; SPARC automatically removes
periodic boundary conditions from the structure.

.. code-block:: yaml

    dft_calculator:
      engine: "ORCA"
      template_file: "orca_template.inp"
      exe_command: "/path/to/orca"


xTB
---

Semi-empirical tight-binding (GFN1-xTB, GFN2-xTB). Very fast — useful for
large molecules or rapid screening before committing to DFT.

.. warning::
   xTB is a **non-periodic** calculator. Periodic boundary conditions are
   automatically removed from the structure at runtime. As a result, the
   **NPT ensemble is not supported** with xTB — the barostat requires a
   periodic cell to compute stress and apply volume coupling. Use NVE or
   NVT instead. SPARC will raise an error if NPT is requested with xTB.

.. code-block:: yaml

    dft_calculator:
      engine: "xTB"
      template_file: "xtb_template.inp"


Quantum ESPRESSO (QE)
---------------------

Open-source plane-wave DFT for periodic systems. k-points default to
Gamma-only sampling; override by specifying ``kpts`` in the template.

.. code-block:: yaml

    dft_calculator:
      engine: "QE"
      template_file: "qe_template.in"
      exe_command: "mpirun -np 4 pw.x"


Gaussian
--------

For molecular (non-periodic) systems. Periodicity is auto-removed. Uses a
``key = value`` template file.

.. code-block:: yaml

    dft_calculator:
      engine: "Gaussian"
      template_file: "gaussian_template.inp"
      exe_command: "g16"

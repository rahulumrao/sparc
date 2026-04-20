.. _chemview:

========
ChemView
========

``ChemView`` is a wrapper around `Chemiscope <https://chemiscope.org/>`_ for
visualising ASE trajectories in Jupyter notebooks. It computes geometric and
atomic properties from ``ase.Atoms`` objects and renders them as a linked
scatter plot and 3-D structure viewer.

.. code-block:: python

   from sparc.src.utils.chemview import ChemView

For worked examples, see the companion notebook below.

.. toctree::
   :maxdepth: 1
   :hidden:

   chemview.ipynb


Requirements
============

``chemiscope`` and ``numpy`` must be installed alongside ASE:

.. code-block:: bash

   pip install chemiscope numpy


Parameters
==========

**frames** : ``ase.Atoms`` or ``list[ase.Atoms]``
   Input structures. A single ``Atoms`` object or a list. Must be non-empty.

**specs** : ``list[str]``
   Properties to compute from the trajectory. See `Property Specs`_ below.

**x, y** : ``str``
   Property names for the scatter-plot axes. Defaults to ``"frame"`` and
   ``"energy"`` when those specs are present.

**z** : ``str``
   Optional third axis for 3-D scatter plots.

**color** : ``str``
   Property used to colour scatter-plot points.

**size** : ``str``
   Property used to scale point size.

**symbol** : ``str``
   Property used for point symbols.

**names** : ``list[str]``
   Custom names for each spec. Length must match ``specs``.

**color_atoms** : ``"element"`` | ``"atom_index"`` | ``None``
   Colouring scheme for atoms in the 3-D viewer. Default: ``"element"``.

**labels** : ``bool``
   Show element symbols on atoms in the 3-D viewer.

**show_index** : ``bool``
   Clicking an atom centres the view on it and shows its properties in the
   info panel.

**env_cutoff** : ``float``
   Cutoff radius in Å for the atom environment sphere. Default: 3.5.

**plot** : ``bool``
   Show the scatter-plot panel alongside the 3-D viewer. Default: ``True``.


.. _property-specs:

Property Specs
==============

Each spec is a string passed in the ``specs`` list.

Structure-level
---------------

One value per frame.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Spec
     - Unit
     - Description
   * - ``"frame"``
     -
     - Frame index (0, 1, 2, …)
   * - ``"energy"``
     - eV
     - Potential energy from ``atoms.get_potential_energy()``
   * - ``"distance:i,j"``
     - Å
     - Distance between atoms *i* and *j*
   * - ``"angle:i,j,k"``
     - deg
     - Angle at atom *j* in the *i–j–k* triplet
   * - ``"dihedral:i,j,k,l"``
     - deg
     - Dihedral angle for the *i–j–k–l* quadruplet
   * - ``"volume"``
     - Å³
     - Cell volume from ``atoms.get_volume()``
   * - ``"cell_a"``, ``"cell_b"``, ``"cell_c"``
     - Å
     - Lattice vector lengths
   * - ``"cell_alpha"``, ``"cell_beta"``, ``"cell_gamma"``
     - deg
     - Cell angles

Atom-level
----------

One value per atom per frame. All frames must have the same number of atoms.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Spec
     - Unit
     - Description
   * - ``"x_pos"``, ``"y_pos"``, ``"z_pos"``
     - Å
     - Cartesian position components
   * - ``"symbol"``
     -
     - Element symbol
   * - ``"atomic_number"``
     -
     - Atomic number
   * - ``"force_x"``, ``"force_y"``, ``"force_z"``
     - eV/Å
     - Force components
   * - ``"force_norm"``
     - eV/Å
     - Force magnitude

If forces are not available on a frame, values are filled with ``NaN``.

.. note::

   ``atom_index`` is always included as a default property and does not need
   to be listed in ``specs``.


Auto-generated Names
====================

When ``names`` is not provided, property keys are generated from the spec
string:

.. code-block:: text

   "frame"            → "frame"
   "energy"           → "energy"
   "distance:0,4"     → "distance_0_4"
   "angle:0,1,2"      → "angle_0_1_2"
   "dihedral:0,1,2,3" → "dihedral_0_1_2_3"
   "force_norm"       → "force_norm"

These keys are used for ``x``, ``y``, ``color``, etc. When the same spec
type appears more than once, use ``names`` to assign distinct keys.


Constraints
===========

- All frames must contain the same number of atoms.
- ``energy`` and force-based specs require a calculator or pre-stored arrays
  on the ``Atoms`` objects. Missing values are filled with ``NaN``.
- Duplicate property names are not allowed. Use ``names`` to disambiguate.


API Reference
=============

.. autofunction:: sparc.src.utils.chemview.ChemView
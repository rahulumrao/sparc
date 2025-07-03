.. .. role:: raw-math(raw)
..     :format: latex html

.. default-role:: math

Plumed
======

.. module:: plumed_wrapper

Overview
--------

We used the open source `Plumed <plumed_>`_ library to accelerate the exploration of potential energy surface.
The ASE build-in `plumed-wrapper <_asePlumed>`_ was attached with the existing calculator.

We used the Social Permutation Invariant (``SPRINT``) coordinates together with Parellel Bias Metdynamics to accelerate the molecular dynamics simulation.
SPRINT coordinates are computed based on the equilibrium distances between atom types and the distances between 
each of the atoms in a system to construct a contact matrix. SPRINT is a generic coordinate based on 
the graph theory which has the universal discrimination of a chemical space. This allows the exploration of the potential energy space
much quicker. 

By definition SPRINT coordinate are calculated from the largest eiugenvalue, `\lambda` of an `n \times n` 
adjency matrix and its corresponding eigenvector, `\bf{V}`, using:

.. math::

   s_{i} = \sqrt{n} \lambda \mathit{v_i}

.. note::
    ``SPRINT`` coordinate is part of the ``adjmat`` module, therefore we need to compile Plumed with correct flag. 
    Please see the PLUMED section in the :ref:`InstalltionGuide`.

.. tip:: 
    Since the package incorporates PLUMED as an auxiliary calculator, 
    it enables the use of advanced enhanced sampling techniques to accelerate the exploration of the potential energy landscape. 
    We particularly recommend combining the ``SPRINT`` coordinates with `Parallel Bias Metadynamics (PBMetaD) <pbmetad>`_ ,
    as this approach offers efficient, self-guided exploration of complex chemical and configurational spaces.

.. Function:
.. ~~~~~~~~~

.. .. autofunction:: sparc.src.plumed_wrapper.modify_forces


.. automodule:: sparc.src.plumed_wrapper
   :members:
   :undoc-members:
   :show-inheritance:



.. References
.. ~~~~~~~~~~

.. _plumed: https://www.plumed.org/
.. _asePlumed: https://wiki.fysik.dtu.dk/ase//ase/calculators/plumed.html
.. _pbmetad:: https://www.plumed.org/doc-v2.9/user-doc/html/_p_b_m_e_t_a_d.html
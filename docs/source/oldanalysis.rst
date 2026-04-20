.. _AnalysisSection:

Analysis
========

.. module:: analysis

Overview
--------

During the execution of the active learning workflow for accelerated exploration of the phase space, 
it is critical to monitor the training progress and predictive reliability of the machine learning potential.
This can be achieved by systematically analyzing key indicators that reflect the current state of the model's 
learning and generalization performance.

To do this, a suite of specialized modules has been implemented to visualize a range of physical and 
statistical properties. These include, but are not limited to, learning curves 
(e.g., energy and force loss evolution), parity plots comparing predicted and reference quantities, 
model uncertainty estimations (such as ensemble variance or deviation metrics), and physical observables 
derived from molecular dynamics simulations (e.g., temperature fluctuations, and sample trajectory).

Model deviation in Forces
-------------------------

This metric tells how much different model in an ensamble ``disagree`` about the forces acting on a given atom in a specific configuration.
A large deviation means the model is uncertain and that more training data is required in that region of phase space.


Mathematically, the `force deviation <modeldevi_>`_  for atom :math:`i` is defined as:

.. math::

   \epsilon_{\mathbf{F}, i}(\mathbf{x}) = \sqrt{ \frac{1}{n_m} \sum_{k=1}^{n_m} \left\| \mathbf{F}_i^{(k)} - \bar{\mathbf{F}}_i \right\|^2 }

where:

- :math:`\mathbf{F}_i^{(k)}` is the force on atom :math:`i` predicted by model :math:`k`,
- :math:`\bar{\mathbf{F}}_i = \frac{1}{n_m} \sum_{k=1}^{n_m} \mathbf{F}_i^{(k)}` is the average force over all models,
- :math:`n_m` is the number of models in the ensemble,
- and :math:`\| \cdot \|` is the Euclidean norm.

In simple terms:

1. Predict the force on atom :math:`i` using multiple models.
2. Compute the average force.
3. Measure how much each model's prediction deviates from the average.
4. Compute the root mean square of those deviations.

This value quantifies how much the models **disagree** about the force, serving as a proxy for uncertainty.

.. _modeldevi: https://docs.deepmodeling.com/projects/deepmd/en/master/test/model-deviation.html

API Reference:
--------------

.. autofunction:: sparc.src.utils.plot_utils.PlotForceDeviation

.. image:: images/model_devi.jpg
   :alt: Visualization of force model deviation
   :align: center
   :width: 700px
   :target: ../_static/model_devi.jpg

.. automodule:: sparc.src.utils.plot_utils
   .. :members: WorkFlowAnalysis
   .. :exclude-members: 

 .. toctree::
..    :maxdepth: 1

..    notebooks/analysisAmmoniaBorate.ipynb
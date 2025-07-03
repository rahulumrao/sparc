Workflow
========


Function:
~~~~~~~~~

.. autofunction:: sparc.src.utils.workflow.WorkFlowAnalysis


This Python module provides an interactive interface for analyzing workflow outputs across multiple iterations. 
The script is especially useful for to inspect energetics and geometric properties over training iterations.

It is designed for Jupyter Notebook environments using ``ipywidgets``, ``matplotlib``, and ``ASE``.

Features
--------
- Load and visualize per-iteration properties from the trajectory file (e.g., temperature, energy).
- Interactive widgets for selecting root directory, trajectory paths, and specific iterations.
- Geometry analysis tab for plotting bond lengths or angles for user-defined atomic indices.

Dependencies
------------
- numpy
- matplotlib
- ASE (Atomic Simulation Environment)

Quick Start
-----------
1. Launch a Jupyter Notebook.
2. Import and call the `WorkFlowAnalysis()` function from sparc library:

   .. code-block:: python

      from sparc.src.utils.workflow import WorkFlowAnalysis
      WorkFlowAnalysis()

3. The interface will appear with tabs for:

   - Temperature
   - Total Energy
   - Potential Energy
   - Kinetic Energy
   - Geometry (Bond / Angle)

4. For each tab:

   - Set the root directory containing `iter_xxxxxx` folders.
   - Specify subfolder (default: `02.dpmd`) and trajectory file (default: `dpmd.traj`).
   - Click "Refresh Iterations" to load available folders.
   - Select iterations to plot.
   - Click the plot button (e.g., "Plot Temperature").

Example Directory Structure
---------------------------

.. code-block:: text

   project_root/
   ├── iter_000000/
   │   └── 02.dpmd/
   │       └── dpmd.traj
   ├── iter_000001/
   │   └── 02.dpmd/
   │       └── dpmd.traj
   └── iter_000002/
       └── 02.dpmd/
           └── dpmd.traj

Geometry Tab Details
---------------------
- Choose "Bond" or "Angle" type.
- Provide indices for atoms (e.g., `0 1` or `0 1 2`).
- The y-axis label will automatically render the proper bond/angle symbols with subscripts.


.. image:: _static/WorkflowAnalysis.gif
   :alt: Workflow Analysis Animation
   :width: 600px



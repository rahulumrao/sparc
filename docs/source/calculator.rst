DFT Calculator
==============

Overview
--------

The `calculator.py` module is designed to set up Density Functional Theory (DFT) calculators for VASP and CP2K using ASE atom objects. 
It provides a command-line interface to configure and initialize these calculators based on a YAML configuration file.

Class : SetupDFTCalculator
~~~~~~~~~~~~~~~~~~~~~~~~~~

A class to set up DFT calculators for VASP and CP2K.

**Parameters:**

- `input_config` (dict): Dictionary containing configuration for the DFT calculator.

**Methods:**

- `vasp()`: Set up the VASP calculator for ASE atom objects.
- `cp2k()`: Set up the CP2K calculator for ASE atom objects.

dft_calculator
~~~~~~~~~~~~~~

Helper function to set up the DFT calculator based on configuration.

**Parameters:**

- `config` (dict): Dictionary containing the full DFT configuration.

**Returns:**

- ASE Calculator: The configured ASE calculator instance.

.. note:: 
    Ensure that the VASP executable and INCAR file paths are correctly specified in the ``yaml`` file. 
    The CP2K setup requires a user defined template file named ``cp2k_template.inp``

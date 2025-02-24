# src/__init__.py
"""
SPARC - Structure Prediction and Active Learning with ReaxFF and DFT Calculations
"""

import sys

# Check Python version
if sys.version_info[:2] < (3, 6):
    raise ImportError(
        "Python >=3.6 is required to use SPARC. "
        f"Current version: {'.'.join(map(str, sys.version_info[:3]))}"
    )

try:
    import ase
except ImportError:
    raise ImportError(
        "ASE is required to use SPARC. Please install it with:\n"
        "pip install ase"
    )

try:
    import deepmd
except ImportError:
    raise ImportError(
        "DeePMD-kit is required to use SPARC. Please install it with:\n"
        "pip install deepmd-kit"
    )

__version__ = "0.1.0"
__author__ = "Rahul Verma"
__email__ = "rverma7@ncsu.edu"

from .sparc import main

# Expose commonly used functions at package level
from .ase_md import NoseNVT, run_aimd, run_Ase_DPMD
from .deepmd_training import deepmd_training
from .active_learning import QueryByCommittee

__all__ = [
    'main',
    'NoseNVT',
    'run_aimd',
    'run_Ase_DPMD',
    'deepmd_training',
    'QueryByCommittee'
]
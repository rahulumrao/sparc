"""
Plotting module for SPARC workflow analysis.

Supports matplotlib backend with shared utilities.

Usage:
    from plotting import matplot
    from plotting.main import get_iteration_dirs, load_trajectory
"""

from . import matplot
from .main import (
    get_iteration_dirs,
    load_trajectory,
    extract_iteration_number,
    compute_rmse,
    compute_mae,
    ReadColvar,
    get_2dSurface,
    get_1dSurface,
    ViewTraj
)

__all__ = [
    'matplot',
    'get_iteration_dirs',
    'load_trajectory',
    'extract_iteration_number',
    'compute_rmse',
    'compute_mae',
    'ReadColvar',
    'get_2dSurface',
    'get_1dSurface',
    'ViewTraj'
]

__version__ = '0.2.0'

"""
Plotting module for SPARC workflow analysis.

Supports matplotlib backend with shared utilities.

Usage:
    from plotting import matplot
    from plotting import chemview
    from plotting.main import get_iteration_dirs, load_trajectory
"""

from . import chemview, matplot
from .main import (
    ReadColvar,
    ViewTraj,
    compute_mae,
    compute_rmse,
    extract_iteration_number,
    get_1dSurface,
    get_2dSurface,
    get_iteration_dirs,
    load_trajectory,
)
from .matplot import PlotForceError

__all__ = [
    "matplot",
    "chemview",
    "get_iteration_dirs",
    "load_trajectory",
    "extract_iteration_number",
    "compute_rmse",
    "compute_mae",
    "ReadColvar",
    "get_2dSurface",
    "get_1dSurface",
    "ViewTraj",
    "PlotForceError",
]

__version__ = "0.2"

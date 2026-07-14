"""
Plotting module for SPARC workflow analysis.

Supports matplotlib backend with shared utilities.

Usage:
    from plotting import matplot
    from plotting import chemview
    from plotting.main import get_iteration_dirs, load_trajectory
"""

try:
    from . import chemview
except ImportError:
    chemview = None  # type: ignore[assignment]

try:
    from . import matplot
    from .matplot import PlotForceError, PlotWorkflowTiming
except ImportError:
    matplot = None  # type: ignore[assignment]
    PlotForceError = None  # type: ignore[assignment]
    PlotWorkflowTiming = None  # type: ignore[assignment]

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
    "PlotWorkflowTiming",
]

__version__ = "0.2"

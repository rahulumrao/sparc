# Compatibility shim — moved to sparc.src.utils.plotting.plot_utils
# This file will be removed in a future release.
from sparc.src.utils.plotting.plot_utils import (
    compute_rmse,
    compute_mae,
    ParityPlot,
    PlotLcurve,
    PlotForceDeviation,
    PlotPotentialEnergy,
    PlotDistribution,
    PlotPES,
    PlotTemp,
    ReadColvar,
    get_2dSurface,
    get_1dSurface,
    ViewTraj,
)
from sparc.src.utils.plotting.matplot import PlotForceError

__all__ = [
    "compute_rmse",
    "compute_mae",
    "ParityPlot",
    "PlotLcurve",
    "PlotForceDeviation",
    "PlotForceError",
    "PlotPotentialEnergy",
    "PlotDistribution",
    "PlotPES",
    "PlotTemp",
    "ReadColvar",
    "get_2dSurface",
    "get_1dSurface",
    "ViewTraj",
]

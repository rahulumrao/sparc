"""
Plotting helpers for AL workflow timing data.

Three grouped bars per iteration — mlmd + qbc are merged into Exploration:
  - DFT Labelling  (#55A868 green)
  - Training       (#4C72B0 blue)
  - Exploration    (#DD8452 orange)  ← mlmd + qbc combined

Broken y-axis triggers automatically when any single step is >> the others
(typical case: Training dominates at 10-15 h while DFT + Exploration < 1 h).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.patches import Patch

    _MATPLOTLIB_AVAILABLE = True
except ImportError as _matplotlib_import_err:
    plt = None  # type: ignore[assignment]
    Axes = None  # type: ignore[assignment]
    Patch = None  # type: ignore[assignment]
    _MATPLOTLIB_AVAILABLE = False
    _matplotlib_import_err_msg = str(_matplotlib_import_err)


def _require_matplotlib() -> None:
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Timing plots require matplotlib.\n"
            "Install with: pip install matplotlib\n"
            f"(original error: {_matplotlib_import_err_msg})"
        )


# ── Column / label / color definitions (post-merge) ──────────────────────────
PLOT_COLS = ("dft_wall_h", "train_wall_h", "sampling_wall_h")
PLOT_LABELS = ("DFT Labelling", "Training", "Exploration")
PLOT_COLORS = ("#55A868", "#4C72B0", "#DD8452")
BAR_WIDTH = 0.28
BAR_OFFSETS = (-BAR_WIDTH, 0.0, BAR_WIDTH)

_DEFAULT_RC: dict = {
    "font.family": "serif",
    "font.serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "font.size": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
}


# ── Data preparation ──────────────────────────────────────────────────────────


def _prepare_df(raw: pd.DataFrame, unit: str = "h") -> pd.DataFrame:
    """
    Pivot raw timings.csv into a wide per-iteration DataFrame.

    mlmd + qbc are summed into sampling_wall_h so the three display columns
    match DFT Labelling / Training / Exploration.

    Always pivots ``duration_h``, then scales to minutes when ``unit == "m"``.
    (Previously ``unit="m"`` incorrectly used ``duration_s`` and multiplied by 60,
    which produced seconds×60 instead of minutes.)
    """
    wide = raw.pivot_table(
        index="iteration",
        columns="step",
        values="duration_h",
        fill_value=0.0,
    )

    def _get(col: str) -> pd.Series:
        return wide[col] if col in wide.columns else pd.Series(0.0, index=wide.index)

    # DFT candidate count — only 'dft' rows carry a non-empty count
    counts = (
        raw.loc[raw["step"] == "dft"]
        .set_index("iteration")["count"]
        .apply(lambda v: int(float(v)) if pd.notna(v) and str(v).strip() != "" else 0)
    )

    df = pd.DataFrame(
        {
            "iteration": wide.index,
            "dft_wall_h": _get("dft"),
            "train_wall_h": _get("train"),
            "sampling_wall_h": _get("mlmd") + _get("qbc"),
            "n_configs_labeled": counts.reindex(wide.index).fillna(0).astype(int),
        }
    ).reset_index(drop=True)

    if unit == "m":
        # hours → minutes
        for col in ("dft_wall_h", "train_wall_h", "sampling_wall_h"):
            df[col] = df[col] * 60.0

    return df


# ── Broken-axis helpers ───────────────────────────────────────────────────────


def _needs_broken_axis(df: pd.DataFrame, ratio: float = 2.5) -> bool:
    """True when one step is >> all others across any iteration."""
    col_max = df[list(PLOT_COLS)].max()
    dominant = float(col_max.max())
    others = col_max[col_max < dominant]
    if others.empty or float(others.max()) <= 0:
        return False
    return dominant > float(others.max()) * ratio


def _broken_limits(
    df: pd.DataFrame, pad: float = 0.15
) -> Tuple[float, float, float, float]:
    """
    Compute y-axis limits for a broken-axis plot.

    Returns (bottom_lo, bottom_hi, top_lo, top_hi).
    The dominant step goes into the top panel; all others into the bottom.
    """
    col_max = df[list(PLOT_COLS)].max()
    dom_col = col_max.idxmax()
    dom_val = float(col_max[dom_col])
    rest_max = float(col_max[col_max.index != dom_col].max())

    bottom_hi = rest_max * (1 + pad)
    bottom_hi = float(np.ceil(bottom_hi * 4) / 4)  # snap to nearest 0.25

    top_lo = max(dom_val * 0.85, bottom_hi + 0.5)
    top_hi = dom_val * (1 + pad * 0.5)
    return 0.0, bottom_hi, top_lo, top_hi


# ── Drawing helpers ───────────────────────────────────────────────────────────


def _style_ax(ax: Axes, grid_ls: str = "-.", grid_alpha: float = 0.7) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.85", lw=0.6, ls=grid_ls, alpha=grid_alpha)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def _break_marks(ax_top: Axes, ax_bot: Axes) -> None:
    kw = dict(
        marker=[(-1, -0.5), (1, 0.5)], ms=10, ls="none", color="k", mew=1, clip_on=False
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kw)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kw)


def _draw_bars(
    ax_top: Axes,
    ax_bot: Axes,
    df: pd.DataFrame,
    threshold: float,
    x: np.ndarray,
    counts: np.ndarray,
    unit: str = "h",
) -> None:
    """
    Draw grouped bars on a broken y-axis.

    Matplotlib ``bar(height=v)`` always starts at y=0. If a tall bar is drawn
    *only* on the upper panel, the part below ``ylim[0]`` is clipped and the
    visible stub looks like it is floating ("hanging") above the break.

    Values that cross the break are therefore drawn on **both** panels: the
    bottom panel shows 0 → break, the top panel shows break → value. Short
    bars stay on the bottom panel only. Numeric labels are placed once, on
    the panel that contains the bar top.
    """
    fmt = "%.1f" if unit == "m" else "%.2f"
    for offset, col, color in zip(BAR_OFFSETS, PLOT_COLS, PLOT_COLORS):
        vals = df[col].to_numpy()
        for i, val in enumerate(vals):
            xpos = x[i] + offset
            crosses = val > threshold

            # Always draw on the bottom panel (clipped by ylim).
            b_bot = ax_bot.bar(
                xpos, val, BAR_WIDTH, color=color, edgecolor="white", lw=0.6
            )

            if crosses:
                # Same bar on the top panel → only the tip above the break shows.
                b_top = ax_top.bar(
                    xpos, val, BAR_WIDTH, color=color, edgecolor="white", lw=0.6
                )
                if val > 1e-6:
                    ax_top.bar_label(
                        b_top, fmt=fmt, padding=3, fontsize=10, color="0.25"
                    )
                if col == "dft_wall_h":
                    _annotate_counts(ax_top, [b_top[0]], [counts[i]])
                    _annotate_counts(ax_bot, [b_bot[0]], [counts[i]])
            else:
                if val > 1e-6:
                    ax_bot.bar_label(
                        b_bot, fmt=fmt, padding=3, fontsize=10, color="0.25"
                    )
                if col == "dft_wall_h":
                    _annotate_counts(ax_bot, [b_bot[0]], [counts[i]])


def _annotate_counts(ax: Axes, bars: list, counts) -> None:
    """Write candidate count text inside the visible part of DFT bars."""
    lo, hi = ax.get_ylim()
    for bar, n in zip(bars, counts):
        if not n:
            continue
        h = float(bar.get_height())
        # Visible segment of this bar inside the panel.
        y_lo = max(lo, 0.0)
        y_hi = min(hi, h)
        if y_hi <= y_lo:
            continue
        y = 0.5 * (y_lo + y_hi)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{int(n):,}",
            rotation=45,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )


def _legend_patches() -> list:
    return [
        Patch(facecolor=c, edgecolor="white", label=lbl)
        for c, lbl in zip(PLOT_COLORS, PLOT_LABELS)
    ]


# ── Public API ────────────────────────────────────────────────────────────────


def PlotStepTimings(
    df: pd.DataFrame,
    unit: str = "h",
    save_fig: Optional[str] = None,
    show: bool = True,
    rcparams: Optional[dict] = None,
    Ycmin: Optional[Tuple[float, float]] = None,
    Ycmax: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (9, 5),
    grid_ls: str = "-.",
    grid_alpha: float = 0.7,
) -> None:
    """
    Grouped-bar workflow timing chart from timings.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Raw timings.csv loaded with pandas (or sparc.src.utils.timing.load_workflow_timing).
        Required columns: iteration, step, duration_h, count.
    unit : str
        'h' hours (default) or 'm' minutes.
    save_fig : str, optional
        File path to save (e.g. 'timing.png').
    show : bool
        Call plt.show() after rendering.
    rcparams : dict, optional
        Extra matplotlib rcParam overrides.
    Ycmin : (float, float), optional
        Explicit (lo, hi) for the bottom panel of the broken axis, e.g. (0, 4).
        Auto-computed when None.
    Ycmax : (float, float), optional
        Explicit (lo, hi) for the top panel of the broken axis, e.g. (6, 16).
        Auto-computed when None.
    figsize : (float, float)
        Matplotlib figure size in inches (default ``(9, 5)``).
    """
    _require_matplotlib()
    rc = dict(_DEFAULT_RC)
    if rcparams:
        rc.update(rcparams)
    plt.rcParams.update(rc)

    wdf = _prepare_df(df, unit=unit)
    ylabel = "Wall time (h)" if unit == "h" else "Wall time (min)"
    fmt = "%.1f" if unit == "m" else "%.2f"
    x = np.arange(len(wdf))
    iters = wdf["iteration"].to_numpy()
    counts = wdf["n_configs_labeled"].to_numpy()
    patches = _legend_patches()

    if _needs_broken_axis(wdf) or (Ycmin is not None or Ycmax is not None):
        b_lo, b_hi, t_lo, t_hi = _broken_limits(wdf)
        if Ycmin is not None:
            b_lo, b_hi = Ycmin
        if Ycmax is not None:
            t_lo, t_hi = Ycmax

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=figsize,
            dpi=300,
            gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05},
        )
        ax1.set_ylim(t_lo, t_hi)
        ax2.set_ylim(b_lo, b_hi)

        for ax in (ax1, ax2):
            _style_ax(ax, grid_ls=grid_ls, grid_alpha=grid_alpha)

        ax1.spines["bottom"].set_visible(False)
        ax1.tick_params(bottom=False)
        _break_marks(ax1, ax2)

        _draw_bars(ax1, ax2, wdf, threshold=b_hi, x=x, counts=counts, unit=unit)

        ax2.set_xticks(x)
        ax2.set_xticklabels(iters)
        ax2.set_xlabel("Iteration")
        # Leave room so the shared ylabel does not sit on the y-tick labels.
        fig.subplots_adjust(left=0.16, right=0.97, top=0.9, bottom=0.12)
        fig.supylabel(ylabel, x=0.02)
        fig.legend(
            handles=patches,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.65, 0.99),
        )

    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        _style_ax(ax, grid_ls=grid_ls, grid_alpha=grid_alpha)
        dft_bars: list = []

        for offset, col, color in zip(BAR_OFFSETS, PLOT_COLS, PLOT_COLORS):
            vals = wdf[col].to_numpy()
            b = ax.bar(
                x + offset, vals, BAR_WIDTH, color=color, edgecolor="white", lw=0.6
            )
            labels = [fmt % v if v > 1e-6 else "" for v in vals]
            ax.bar_label(b, labels=labels, padding=3, fontsize=10, color="0.25")
            if col == "dft_wall_h":
                dft_bars = list(b)

        _annotate_counts(ax, dft_bars, counts)
        ymax = float(wdf[list(PLOT_COLS)].max().max()) * 1.15
        ax.set_ylim(0, max(ymax, 0.5))
        ax.set_xticks(x)
        ax.set_xticklabels(iters)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        fig.legend(
            handles=patches,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.65, 0.99),
        )
        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.1)

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    elif save_fig:
        plt.close()


def PlotStepTimingLine(
    df: pd.DataFrame,
    unit: str = "h",
    save_fig: Optional[str] = None,
    show: bool = True,
    rcparams: Optional[dict] = None,
    figsize: Tuple[float, float] = (9, 5),
) -> None:
    """Line plot of per-step wall time vs iteration."""
    _require_matplotlib()
    rc = dict(_DEFAULT_RC)
    if rcparams:
        rc.update(rcparams)
    plt.rcParams.update(rc)

    wdf = _prepare_df(df, unit=unit)
    ylabel = "Wall time (h)" if unit == "h" else "Wall time (min)"
    x = wdf["iteration"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    _style_ax(ax)

    for col, label, color in zip(PLOT_COLS, PLOT_LABELS, PLOT_COLORS):
        ax.plot(
            x, wdf[col].to_numpy(), marker="o", lw=2, ms=6, color=color, label=label
        )

    ax.set_xticks(x)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.1)

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    elif save_fig:
        plt.close()

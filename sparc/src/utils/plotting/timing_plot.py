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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

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
    """
    duration_col = "duration_h" if unit == "h" else "duration_s"

    wide = raw.pivot_table(
        index="iteration",
        columns="step",
        values=duration_col,
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
) -> Tuple[list, list]:
    """
    Draw all bars, routing each to ax_top if val > threshold else ax_bot.

    Returns (dft_bars_on_top, dft_bars_on_bot) for count annotation.
    """
    dft_top: list = []
    dft_bot: list = []

    for offset, col, color in zip(BAR_OFFSETS, PLOT_COLS, PLOT_COLORS):
        vals = df[col].to_numpy()
        for i, val in enumerate(vals):
            ax = ax_top if val > threshold else ax_bot
            b = ax.bar(
                x[i] + offset, val, BAR_WIDTH, color=color, edgecolor="white", lw=0.6
            )
            if val > 1e-6:  # skip label on zero-height bars (e.g. iter 0 DFT/Training)
                ax.bar_label(b, fmt="%.2f", padding=2, fontsize=10, color="0.25")
            if col == "dft_wall_h":
                (dft_top if val > threshold else dft_bot).append(b[0])

    return dft_top, dft_bot


def _annotate_counts(ax: Axes, bars: list, counts: np.ndarray) -> None:
    """Write candidate count text inside DFT bars."""
    lo, hi = ax.get_ylim()
    for bar, n in zip(bars, counts):
        if n == 0:
            continue
        h = bar.get_height()
        if not (lo < h <= hi):
            continue
        y = h / 2 if lo < h / 2 < hi else (lo + hi) / 2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{n:,}",
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
    """
    rc = dict(_DEFAULT_RC)
    if rcparams:
        rc.update(rcparams)
    plt.rcParams.update(rc)

    wdf = _prepare_df(df, unit=unit)
    ylabel = "Wall time (h)" if unit == "h" else "Wall time (min)"
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
            figsize=(9, 5),
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

        dft_top, dft_bot = _draw_bars(ax1, ax2, wdf, threshold=b_hi, x=x)
        _annotate_counts(ax1, dft_top, counts)
        _annotate_counts(ax2, dft_bot, counts)

        ax2.set_xticks(x)
        ax2.set_xticklabels(iters)
        ax2.set_xlabel("Iteration")
        fig.supylabel(ylabel, x=0.04)
        fig.legend(
            handles=patches,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.65, 0.99),
        )
        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.1)

    else:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        _style_ax(ax, grid_ls=grid_ls, grid_alpha=grid_alpha)
        dft_bars: list = []

        for offset, col, color in zip(BAR_OFFSETS, PLOT_COLS, PLOT_COLORS):
            vals = wdf[col].to_numpy()
            b = ax.bar(
                x + offset, vals, BAR_WIDTH, color=color, edgecolor="white", lw=0.6
            )
            labels = [f"{v:.2f}" if v > 1e-6 else "" for v in vals]
            ax.bar_label(b, labels=labels, padding=2, fontsize=10, color="0.25")
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
) -> None:
    """Line plot of per-step wall time vs iteration."""
    rc = dict(_DEFAULT_RC)
    if rcparams:
        rc.update(rcparams)
    plt.rcParams.update(rc)

    wdf = _prepare_df(df, unit=unit)
    ylabel = "Wall time (h)" if unit == "h" else "Wall time (min)"
    x = wdf["iteration"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
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

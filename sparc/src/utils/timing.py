"""
Workflow timing utilities for SPARC active learning runs.

Records per-step wall-clock times in a plot-ready CSV file:

    timings.csv   — one row per step (load directly with pandas)

Each step completion also writes a contextual [INFO] timing line to Sparc.log.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from sparc.src.utils.logger import SparcLog

STEP_ORDER = ("dft", "train", "mlmd", "qbc")

STEP_LABELS = {
    "dft": "DFT Labelling",
    "train": "Training",
    "mlmd": "Exploration",
    "qbc": "Query-by-Committee",
}

CSV_COLUMNS = (
    "iteration",
    "step",
    "step_dir",
    "duration_s",
    "duration_h",
    "count",
)


@dataclass
class StepHandle:
    """Token returned by start_step; pass to end_step to record duration."""

    iteration: int
    step: str
    step_dir: str
    count: Optional[int]
    t0: float


def _format_log_duration(seconds: float) -> str:
    """Format duration in minutes for Sparc.log timing lines."""
    return f"{seconds / 60:.1f} min"


def _format_step_time_line(
    iteration: int,
    step: str,
    duration_s: float,
    count: Optional[Union[int, str]] = None,
) -> str:
    """Build a contextual INFO timing line for a completed workflow step."""
    label = STEP_LABELS.get(step, step)
    msg = f"Step timing | iter {iteration} | {label} time: {_format_log_duration(duration_s)}"
    if count not in ("", None):
        msg += f" | n={count}"
    return msg


def _normalize_record(row: dict) -> dict:
    """Coerce CSV string values back to typed fields."""
    count = row.get("count", "")
    if count in ("", None):
        count_value: Union[int, str] = ""
    else:
        try:
            count_value = int(float(count))
        except (TypeError, ValueError):
            count_value = ""

    return {
        "iteration": int(row["iteration"]),
        "step": row["step"],
        "step_dir": row["step_dir"],
        "duration_s": float(row["duration_s"]),
        "duration_h": float(row["duration_h"]),
        "count": count_value,
    }


class WorkflowTimer:
    """Track and persist wall-clock time for each AL workflow step."""

    def __init__(
        self,
        output_dir: Union[str, Path] = ".",
        csv_name: str = "timings.csv",
    ):
        self.output_dir = Path(output_dir)
        self.csv_path = self.output_dir / csv_name
        self.records: List[dict] = self._load_existing_records()

    def _load_existing_records(self) -> List[dict]:
        if not self.csv_path.exists():
            return []
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                return [_normalize_record(row) for row in csv.DictReader(f)]
        except OSError:
            return []

    def start_step(
        self,
        iteration: int,
        step: str,
        step_dir: str = "",
        count: Optional[int] = None,
    ) -> StepHandle:
        """Record step start time; timing is logged on end_step."""
        return StepHandle(
            iteration=iteration,
            step=step,
            step_dir=step_dir,
            count=count,
            t0=time.perf_counter(),
        )

    def end_step(self, handle: StepHandle) -> dict:
        """Compute elapsed time since start_step and write CSV/log entry."""
        duration_s = time.perf_counter() - handle.t0
        row = self.record(
            iteration=handle.iteration,
            step=handle.step,
            step_dir=handle.step_dir,
            duration_s=duration_s,
            count=handle.count,
        )
        SparcLog("")
        SparcLog(
            _format_step_time_line(
                handle.iteration,
                handle.step,
                row["duration_s"],
                row.get("count"),
            )
        )
        return row

    def record(
        self,
        iteration: int,
        step: str,
        step_dir: str,
        duration_s: float,
        count: Optional[int] = None,
    ) -> dict:
        """Append a timing record to timings.csv."""
        row = {
            "iteration": int(iteration),
            "step": step,
            "step_dir": step_dir,
            "duration_s": round(float(duration_s), 3),
            "duration_h": round(float(duration_s) / 3600.0, 6),
            "count": "" if count is None else int(count),
        }
        self.records.append(row)
        self._append_csv_row(row)
        return row

    def _append_csv_row(self, row: dict) -> None:
        write_header = not self.csv_path.exists()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({col: row[col] for col in CSV_COLUMNS})

    def log_summary(self) -> None:
        """Print a compact per-iteration timing summary to Sparc.log."""
        if not self.records:
            return

        SparcLog("")
        SparcLog("-" * 80)
        SparcLog("WORKFLOW TIMING SUMMARY".center(80))
        SparcLog("-" * 80)

        by_iter: dict[int, dict[str, float]] = {}
        for row in self.records:
            by_iter.setdefault(row["iteration"], {})
            by_iter[row["iteration"]][row["step"]] = float(row["duration_s"])

        plot_steps = ("dft", "train", "mlmd")
        for iteration in sorted(by_iter):
            parts = by_iter[iteration]
            total = sum(parts.values())
            step_parts = ", ".join(
                f"{step}={parts[step] / 60:.1f} min"
                for step in plot_steps
                if step in parts
            )
            SparcLog(
                f"  iter {iteration} | total {_format_log_duration(total)} | {step_parts}"
            )

        SparcLog(f"  Saved to: {self.csv_path}")
        SparcLog("-" * 80)


def load_workflow_timing(
    path: Union[str, Path] = "timings.csv",
):
    """
    Load workflow timing records as a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to ``timings.csv``.

    Returns
    -------
    pandas.DataFrame
        Columns: iteration, step, step_dir, duration_s, duration_h, count.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Timing file not found: {path}")
    return pd.read_csv(path)

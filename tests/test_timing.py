from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sparc.src.utils.timing import (
    CSV_COLUMNS,
    WorkflowTimer,
    load_workflow_timing,
)


def _use_headless_backend():
    """Plotting deps are optional for a SPARC run, so skip instead of failing."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_workflow_timer_writes_plot_ready_csv(tmp_path):
    timer = WorkflowTimer(tmp_path)

    dft = timer.start_step(0, "dft", "00.dft")
    timer.end_step(dft)
    train = timer.start_step(0, "train", "01.train")
    timer.end_step(train)

    csv_path = tmp_path / "timings.csv"
    assert csv_path.exists()
    assert not (tmp_path / "timings.json").exists()

    df = pd.read_csv(csv_path)
    assert list(df.columns) == list(CSV_COLUMNS)
    assert len(df) == 2
    assert set(df["step"]) == {"dft", "train"}
    assert (df["duration_s"] >= 0).all()
    assert (df["duration_h"] >= 0).all()


def test_workflow_timer_records_dft_count(tmp_path):
    timer = WorkflowTimer(tmp_path)
    handle = timer.start_step(1, "dft", "00.dft", count=169)
    timer.end_step(handle)

    df = pd.read_csv(tmp_path / "timings.csv")
    assert int(df.iloc[0]["count"]) == 169


def test_workflow_timer_logs_step_time_on_end(tmp_path):
    timer = WorkflowTimer(tmp_path)
    messages = []

    with patch(
        "sparc.src.utils.timing.SparcLog", side_effect=lambda m: messages.append(m)
    ):
        handle = timer.start_step(2, "train", "01.train")
        assert messages == []
        timer.end_step(handle)

    assert len(messages) == 2
    assert messages[0] == ""
    assert "Step timing |" in messages[1]
    assert "Training time:" in messages[1]
    assert " min" in messages[1]
    assert "iter 2 |" in messages[1]


def test_workflow_timer_logs_dft_count_in_time_line(tmp_path):
    timer = WorkflowTimer(tmp_path)
    messages = []

    with patch(
        "sparc.src.utils.timing.SparcLog", side_effect=lambda m: messages.append(m)
    ):
        handle = timer.start_step(1, "dft", "00.dft", count=169)
        timer.end_step(handle)

    assert len(messages) == 2
    assert messages[0] == ""
    assert "Step timing |" in messages[1]
    assert "DFT Labelling time:" in messages[1]
    assert "n=169" in messages[1]
    assert " min" in messages[1]


def test_workflow_timer_appends_on_restart(tmp_path):
    timer = WorkflowTimer(tmp_path)
    handle = timer.start_step(0, "mlmd", "02.dpmd")
    timer.end_step(handle)

    timer2 = WorkflowTimer(tmp_path)
    handle2 = timer2.start_step(1, "dft", "00.dft")
    timer2.end_step(handle2)

    df = pd.read_csv(tmp_path / "timings.csv")
    assert len(df) == 2
    assert list(df["iteration"]) == [0, 1]
    assert len(timer2.records) == 2


def test_load_workflow_timing(tmp_path):
    timer = WorkflowTimer(tmp_path)
    timer.record(2, "qbc", "02.dpmd", 12.5)

    df = load_workflow_timing(tmp_path / "timings.csv")
    assert len(df) == 1
    assert df.iloc[0]["step"] == "qbc"
    assert df.iloc[0]["duration_s"] == 12.5


def test_load_workflow_timing_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_workflow_timing(tmp_path / "missing.csv")


def test_prepare_df_minutes_from_hours():
    """unit='m' must be hours×60, not seconds×60."""
    from sparc.src.utils.plotting.timing_plot import _prepare_df

    df = pd.DataFrame(
        [
            {
                "iteration": 1,
                "step": "train",
                "duration_h": 2.45,
                "duration_s": 2.45 * 3600,
                "count": None,
            },
            {
                "iteration": 1,
                "step": "mlmd",
                "duration_h": 0.10,
                "duration_s": 0.10 * 3600,
                "count": None,
            },
            {
                "iteration": 1,
                "step": "dft",
                "duration_h": 0.25,
                "duration_s": 0.25 * 3600,
                "count": 10,
            },
        ]
    )
    mins = _prepare_df(df, unit="m")
    assert mins["train_wall_h"].iloc[0] == pytest.approx(2.45 * 60)
    assert mins["dft_wall_h"].iloc[0] == pytest.approx(0.25 * 60)
    assert mins["sampling_wall_h"].iloc[0] == pytest.approx(0.10 * 60)


def test_plot_grouped_broken_axis(tmp_path):
    _use_headless_backend()
    from sparc.src.utils.plotting.timing_plot import (
        PlotStepTimings,
        _needs_broken_axis,
        _prepare_df,
    )

    df = pd.DataFrame(
        [
            {"iteration": 1, "step": "train", "duration_h": 2.42, "count": None},
            {"iteration": 1, "step": "mlmd", "duration_h": 0.13, "count": None},
            {"iteration": 1, "step": "dft", "duration_h": 15.0, "count": 4067},
            {"iteration": 2, "step": "train", "duration_h": 2.54, "count": None},
            {"iteration": 2, "step": "mlmd", "duration_h": 0.17, "count": None},
            {"iteration": 2, "step": "dft", "duration_h": 0.66, "count": 169},
        ]
    )
    assert _needs_broken_axis(_prepare_df(df))

    out = tmp_path / "grouped.png"
    PlotStepTimings(df, save_fig=str(out), show=False)
    assert out.exists()


@pytest.mark.parametrize("plot_type", ["grouped", "line"])
def test_plot_workflow_timing(tmp_path, plot_type):
    _use_headless_backend()
    from sparc.src.utils.plotting.matplot import PlotWorkflowTiming

    template_csv = (
        Path(__file__).resolve().parents[1]
        / "examples/workflow_timing_template/timings_template.csv"
    )
    out = tmp_path / f"{plot_type}.png"
    PlotWorkflowTiming(
        timing_file=str(template_csv), plot_type=plot_type, save_fig=str(out)
    )
    assert out.exists()

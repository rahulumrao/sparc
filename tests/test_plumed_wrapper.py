"""Tests for PLUMED input rewriting in sparc.src.plumed_wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
from sparc.src.plumed_wrapper import (
    plumed_block,
    read_block,
    rewrite_file_kwargs,
)


class TestReadBlock:
    def test_joins_multiline_action_without_ellipsis(self):
        lines = [
            "fk: FKERNELABF ...",
            "ARG=diff",
            "KAPPA=8000.0",
            "CZARSTRIDE=1000",
            "...",
            "PRINT ARG=* FILE=COLVAR",
        ]
        out = read_block(lines)
        assert out == [
            "fk: FKERNELABF ARG=diff KAPPA=8000.0 CZARSTRIDE=1000",
            "PRINT ARG=* FILE=COLVAR",
        ]

    def test_skips_comments_and_blanks(self):
        lines = [
            "# comment",
            "",
            "UNITS LENGTH=A",
            "  # indented comment inside file",
            "d1: DISTANCE ATOMS=1,2",
        ]
        assert read_block(lines) == [
            "UNITS LENGTH=A",
            "d1: DISTANCE ATOMS=1,2",
        ]

    def test_unclosed_block_raises(self):
        with pytest.raises(ValueError, match="Unclosed"):
            read_block(["fk: FKERNELABF ...", "ARG=d1"])


class TestRewriteFileKwargs:
    def test_state_wfile_not_corrupted_by_file_substring(self, tmp_path: Path):
        out_dir = tmp_path / "iter_000000" / "02.dpmd"
        line = "opes: OPES_METAD ARG=d1 STATE_WFILE=STATE STATE_WSTRIDE=10000"
        out = rewrite_file_kwargs(line, out_dir)
        assert f"STATE_WFILE={out_dir}/STATE" in out
        assert "STATE_WSTRIDE=10000" in out
        assert out.index("STATE_WFILE=") < out.index("STATE_WSTRIDE=")

    def test_print_file_prefixed(self, tmp_path: Path):
        out_dir = tmp_path / "iter_000000" / "02.dpmd"
        out = rewrite_file_kwargs("PRINT ARG=* FILE=COLVAR", out_dir)
        assert out == f"PRINT ARG=* FILE={out_dir}/COLVAR"

    def test_outdir_rewritten_to_out_dir(self, tmp_path: Path):
        out_dir = tmp_path / "iter_000001" / "02.dpmd"
        out = rewrite_file_kwargs("fk: FKERNELABF ARG=d1 OUTDIR=/tmp/old", out_dir)
        assert f"OUTDIR={out_dir}" in out
        assert "/tmp/old" not in out


class TestPlumedBlock:
    def test_load_relative_absolutized(self, tmp_path: Path):
        lib = tmp_path / "fk-eabf" / "src" / "forcekernel.so"
        lib.parent.mkdir(parents=True)
        lib.write_bytes(b"\x00")
        plumed_dir = tmp_path
        out_dir = tmp_path / "iter_000000" / "02.dpmd"
        out = plumed_block("LOAD FILE=fk-eabf/src/forcekernel.so", plumed_dir, out_dir)
        assert out == f"LOAD FILE={lib.resolve()}"
        assert "iter_000000" not in out

    def test_load_absolute_unchanged(self, tmp_path: Path):
        lib = tmp_path / "forcekernel.so"
        lib.write_bytes(b"\x00")
        out_dir = tmp_path / "iter_000000" / "02.dpmd"
        out = plumed_block(f"LOAD FILE={lib}", tmp_path, out_dir)
        assert out == f"LOAD FILE={lib.resolve()}"

    def test_load_missing_raises(self, tmp_path: Path):
        out_dir = tmp_path / "iter_000000" / "02.dpmd"
        with pytest.raises(FileNotFoundError, match="PLUMED LOAD"):
            plumed_block("LOAD FILE=missing.so", tmp_path, out_dir)

    def test_fkernelabf_gets_outdir(self, tmp_path: Path):
        out_dir = tmp_path / "iter_000000" / "02.dpmd"
        out = plumed_block("fk: FKERNELABF ARG=d1 KAPPA=5000.0", tmp_path, out_dir)
        assert out.endswith(f"OUTDIR={out_dir}")
        assert "ARG=d1" in out
        assert "KAPPA=5000.0" in out

    def test_fkernelabf_replaces_existing_outdir(self, tmp_path: Path):
        out_dir = tmp_path / "iter_000002" / "02.dpmd"
        out = plumed_block(
            "fk: FKERNELABF ARG=d1 OUTDIR=/old/path KAPPA=1.0",
            tmp_path,
            out_dir,
        )
        assert out.count("OUTDIR=") == 1
        assert f"OUTDIR={out_dir}" in out
        assert "/old/path" not in out

    def test_metad_no_outdir_inject(self, tmp_path: Path):
        out_dir = tmp_path / "iter_000000" / "02.dpmd"
        out = plumed_block(
            "metad: METAD ARG=d1 SIGMA=0.1 HEIGHT=1.0 PACE=10 FILE=HILLS",
            tmp_path,
            out_dir,
        )
        assert "OUTDIR=" not in out
        assert f"FILE={out_dir}/HILLS" in out

    def test_multiline_then_outdir(self, tmp_path: Path):
        out_dir = tmp_path / "iter_000000" / "02.dpmd"
        joined = read_block(
            [
                "fk: FKERNELABF ...",
                "ARG=diff",
                "KAPPA=8000.0",
                "CZARSTRIDE=1000",
                "...",
            ]
        )[0]
        out = plumed_block(joined, tmp_path, out_dir)
        assert "..." not in out
        assert out.startswith("fk: FKERNELABF ARG=diff KAPPA=8000.0 CZARSTRIDE=1000")
        assert out.endswith(f"OUTDIR={out_dir}")

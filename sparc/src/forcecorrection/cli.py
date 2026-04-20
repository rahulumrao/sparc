"""
CLI entry point for standalone PLUMED bias force correction.

Run from the directory containing the PLUMED output files (cv_force, cv_derivs).

Usage
-----
sparc-correct --traj AseMD.traj --cv-atoms 1,8
sparc-correct --traj AseTraj.xyz --cv-atoms 1,8 --cv-atoms 1,2
sparc-correct --traj AseMD.traj --cv-atoms 1,8 --cv-force-file cv_force --cv-derivs-file cv_derivs
sparc-correct --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sparc.src.forcecorrection.corrector import correct_aimd_forces


def _parse_cv_atoms(value: str) -> list[int]:
    """Parse a comma-separated list of 1-based atom indices, e.g. '1,8'."""
    try:
        indices = [int(x) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"cv-atoms must be comma-separated integers (e.g. '1,8'), got: {value!r}"
        )
    if any(i < 1 for i in indices):
        raise argparse.ArgumentTypeError(
            f"Atom indices must be 1-based (got {indices})"
        )
    return indices


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sparc-correct",
        description=(
            "Remove PLUMED bias forces from an ASE AIMD trajectory.\n\n"
            "Run from the directory containing the PLUMED output files\n"
            "(cv_force and cv_derivs). Supports ASE binary (.traj) and\n"
            "extxyz (.xyz) trajectory formats. The file is corrected in-place."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  sparc-correct --traj AseMD.traj --cv-atoms 1,8\n"
            "  sparc-correct --traj AseTraj.xyz --cv-atoms 1,8 --cv-atoms 1,2\n"
            "  sparc-correct --traj AseMD.traj --cv-atoms 1,4,7 \\\n"
            "                --cv-force-file cv_force --cv-derivs-file cv_derivs\n"
        ),
    )
    p.add_argument(
        "--traj", "-t",
        required=True,
        metavar="FILE",
        help="Trajectory file to correct (.traj or .xyz). Overwritten in-place.",
    )
    p.add_argument(
        "--cv-atoms",
        required=True,
        action="append",
        type=_parse_cv_atoms,
        metavar="I,J,...",
        help=(
            "1-based atom indices for one CV, comma-separated (e.g. 1,8). "
            "Repeat once per CV in the same order as ARG= in plumed.dat."
        ),
    )
    p.add_argument(
        "--cv-force-file",
        default="cv_force",
        metavar="FILE",
        help="PLUMED DUMPFORCES output filename (default: cv_force).",
    )
    p.add_argument(
        "--cv-derivs-file",
        default="cv_derivs",
        metavar="FILE",
        help="PLUMED DUMPDERIVATIVES output filename (default: cv_derivs).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    traj_path = Path(args.traj)
    if not traj_path.exists():
        parser.error(f"Trajectory file not found: {traj_path}")

    correct_aimd_forces(
        traj_path=traj_path,
        cv_atoms=args.cv_atoms,
        cv_force_file=args.cv_force_file,
        cv_derivs_file=args.cv_derivs_file,
        # dft_dir defaults to traj_path.parent (set inside correct_aimd_forces)
    )


if __name__ == "__main__":
    main()

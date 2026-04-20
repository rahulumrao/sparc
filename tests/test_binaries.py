from __future__ import annotations

import shutil
import warnings


def _which(name: str) -> str | None:
    return shutil.which(name)


def test_dft_backend():
    """
    Check if external executables exist in PATH.
    """
    checks = {
        "vasp_std": "VASP (vasp_std)",
        "cp2k_shell.psmp": "CP2K (cp2k_shell.psmp)",
        "pw.x": "Quantum ESPRESSO (pw.x)",
        "g16": "Gaussian 16 (g16)",
        "g09": "Gaussian 09 (g09)",
    }

    found = {}

    for exe, label in checks.items():
        path = _which(exe)
        if path is not None:
            found[exe] = path
            print(f"{label} found at: {path}")

    # If no backend is available throw warning
    if not found:
        warnings.warn(
            "No DFT backend (VASP, CP2K, QE, Gaussian) found in PATH.\n"
            "Make sure to set the environment before running 'sparc'",
            RuntimeWarning,
        )

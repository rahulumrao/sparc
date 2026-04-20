from __future__ import annotations

import importlib.util
from typing import Dict, List
import sparc

# required dependencies for sparc
REQUIRED_DEPENDENCIES: List[str] = ["ase", "deepmd", "plumed"]


def test_import_sparc():
    """
    Ensure sparc can be imported without error
    """
    try:
        import sparc
    except Exception as e:
        raise AssertionError(
            f"Error: {type(e).__name__}: {e}"
        ) from e


def has_package(name: str) -> bool:
    """
    Safely check if a Python package is available without importing it.
    """
    return importlib.util.find_spec(name) is not None


def dependency_status() -> Dict[str, bool]:
    """
    Return availability of key dependencies.
    """
    return {name: has_package(name) for name in REQUIRED_DEPENDENCIES}


def test_libraries():
    """
    Report dependency availability.
    """
    deps = dependency_status()

    print("\nStatus:")
    for name, ok in deps.items():
        print(f"  {name:<8}: {'FOUND' if ok else 'NOT FOUND'}")

    missing = [name for name in REQUIRED_DEPENDENCIES if not deps.get(name, False)]

    assert not missing, (
        "Missing required dependencies:\n"
        + "\n".join(f"  - {name}" for name in missing)
    )

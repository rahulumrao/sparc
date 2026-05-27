from sparc.src.utils.logger import SparcLog


# ==================================================================================
def xtb_template(template_path):
    """
    Parse a simple key=value text template for XTB.
    Lines starting with '#' and blank lines are ignored.
    Returns a dict with properly-typed values.
    """

    def _as_none(s: str):
        return None if s.strip().lower() in {"", "none", "null"} else s

    def _as_int(s: str):
        return int(s.strip())

    def _as_float(s: str):
        return float(s.strip())

    def _as_str(s: str):
        return s.strip()

    # Map of expected keys -> (cast_fn)
    casters = {
        "method": _as_str,
        "charge": _as_int,
        "multiplicity": _as_int,
        "accuracy": _as_float,
        "electronic_temperature": _as_float,
        "max_iterations": _as_int,
        "solvent": _as_none,  # None means gas-phase
        "solvent_method": _as_none,  # e.g. alpb / gbsa or None
        "directory": _as_str,
        "label": _as_str,
    }

    params = {}
    with open(template_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "=" not in ln:
                continue
            key, val = ln.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key in casters:
                try:
                    params[key] = casters[key](val)
                except Exception:
                    # fall back to raw string if casting fails
                    params[key] = val
            else:
                # Unknown key: keep as string but warn
                SparcLog(
                    f"[XTB template] Unknown key '{key}' -> ignored", level="WARNING"
                )
    return params

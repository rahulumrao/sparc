#--------------------------------------------------------------------------------------
import numpy as np
#--------------------------------------------------------------------------------------
# soft import
try:
    import chemiscope
except Exception as e:
    print(
        "[ChemView] Missing dependency 'chemiscope'. "
        "Install with: pip install chemiscope  — docs: https://chemiscope.org/\n"
        f"Import error was: {e}"
    )
    chemiscope = None

# =========================
# helper functions
# =========================

def _require_chemiscope():
    if chemiscope is None:
        raise RuntimeError(
            "ChemView requires 'chemiscope'. Please install using:\n"
            "    pip install chemiscope\n"
            "Docs: https://chemiscope.org/"
        )

def _parse_spec(spec):
    """Parse a list of properties eg: 'distance:0,7' -> ('distance', (0,7))."""
    s = str(spec).strip()
    if ":" in s:
        kind, arg = s.split(":", 1)
        idx = tuple(int(x.strip()) for x in arg.split(",") if x.strip() != "")
    else:
        kind, idx = s, tuple()
    return kind.lower(), idx

def _auto_name(kind, idx):
    """Return the property key ('distance', 'angle', ...)."""
    if kind == "frame":
        return "frame"
    return kind 

def _ensure_atom_index(frames, props, key="atom_index"):
    #
    if key in props:
        return
    frames = list(frames)
    idx = np.concatenate([np.arange(len(f)) for f in frames]).reshape(-1, 1)
    props[key] = {
        "target": "atom",
        "values": idx,
        "units": "",
        "description": "Atom index (start from 0)",
    }

def _resolve_axis_name(properties, name):
    """
    resolve distance:0,7' -> ('distance', (0,7)) and return name key.
    """
    if name in properties:
        return name
    candidates = [k for k in properties.keys() if k == name or k.startswith(name + "_")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise KeyError(f"Property '{name}' not found. Available keys: {list(properties.keys())}")
    raise KeyError(
        f"Ambiguous base name '{name}'. Candidates: {candidates}. "
        f"Use one explicitly or pass 'names=[...]' when building."
    )

# =========================
# main ChemView function
# =========================

def ChemView(
    *,
    frames,                  # sequence of ase.Atoms
    specs,                   # e.g. ["frame","energy","distance:0,7", "aangle:0,1,2"]
    x,                       # axis property (structure)
    y,                       # axis property (structure)
    z=None,                  # [optional] axis property (structure)
    names=None,              # rename keys for specs (same length as specs)
    units_override=None,
    **kwargs,                # meta_name, color_atoms, labels, map_color(plot), plot
):
    """
    Build properties from user defined specs and launch a Chemiscope viewer.

    Parameters
    ----------
    frames : sequence of ase.Atoms
    specs  : list   ("frame", "energy", "distance:i,j", "angle:i,j,k", "dihedral:i,j,k,l")
    x, y   : str    structure property names ('energy', 'distance',...)
    z      : str or None
    names  : list or None
    units_override : dict or None

    **kwargs
    --------
    meta_name="ChemView"
    color_atoms="element"    # None | 'element' | 'atom_index'   (3D structure coloring)
    labels=False             # True → show element symbols on atoms
    map_color=None           # None | one of {x, y, z (if provided)}  (plot coloring)
    plot=True                # True → show plot panel (plot + structure)

    Usage
    -----

    ``python
    from sparc.src.utils.chemview import ChemView
    ChemView(frames=traj,
        specs=["frame", "energy", "angle:0,1,7"],
        x="frame",
        y="energy",
        # z="angle",
        map_color='energy')
    ```
    """
    _require_chemiscope()

    frames = list(frames)
    if not frames:
        raise ValueError("`frames` is empty")
    if names is not None and len(names) != len(specs):
        raise ValueError("`names` length must match `specs` length")

    # ---- build structure-level properties from specs ----
    props = {}
    for k, spec in enumerate(specs):
        kind, idx = _parse_spec(spec)
        key = names[k] if names is not None else _auto_name(kind, idx)

        if kind == "frame":
            vals, unit = list(range(len(frames))), ""
        elif kind == "energy":
            vals, unit = [at.get_potential_energy() for at in frames], "eV"
        elif kind == "distance":
            if len(idx) != 2:
                raise ValueError("distance requires i,j")
            vals = [at.get_distance(idx[0], idx[1]) for at in frames]; unit =  "Å"
        elif kind == "angle":
            if len(idx) != 3:
                raise ValueError("angle requires i,j,k")
            vals = [at.get_angle(idx[0], idx[1], idx[2]) for at in frames]; unit = "deg"
        elif kind == "dihedral":
            if len(idx) != 4:
                raise ValueError("dihedral requires i,j,k,l")
            vals = [at.get_dihedral(idx[0], idx[1], idx[2], idx[3]) for at in frames]; unit = "deg"
        else:
            raise ValueError(f"Unsupported spec kind: {kind!r}")

        unit = (units_override or {}).get(key, unit)
        props[key] = {"target": "structure", "values": vals, "units": unit}

    # add atom_index for selection panel
    _ensure_atom_index(frames, props)

    # axis names ('distance')
    x = _resolve_axis_name(props, x)
    y = _resolve_axis_name(props, y)
    if z is not None:
        z = _resolve_axis_name(props, z)

    map_color = kwargs.get("map_color", None)
    if map_color is not None:
        map_color = _resolve_axis_name(props, map_color)

    # ---- viewer settings / options ----
    meta_name   = kwargs.get("meta_name", "ChemView")
    color_atoms = kwargs.get("color_atoms", "element")
    show_labels = bool(kwargs.get("labels", False))
    show_map    = bool(kwargs.get("plot", True))

    structure_settings = {
        "unitCell": False,                      # True : show the cell
        "spaceFilling": False,
        "atomLabels": show_labels,              # True: element symbols as labels
        "environments": {"activated": True},    # enable click-to-select atoms
    }
    if color_atoms in ("element", "atom_index"):
        structure_settings["color"] = {"property": color_atoms}

    settings = {"target": "structure", "structure": [structure_settings]}
    mode = "structure"

    if show_map:
        map_settings = {"x": {"property": x}, "y": {"property": y}, "z": {"property": (z or "")}}
        valid_colors = {x, y}
        if z:
            valid_colors.add(z)
        if map_color in valid_colors:
            map_settings["color"] = {"property": map_color}
        settings["map"] = map_settings
        mode = "default"   # plot + structure

    return chemiscope.show(
        frames=frames,
        properties=props,
        meta=dict(name=meta_name),
        environments=chemiscope.all_atomic_environments(frames),
        settings=settings,
        mode=mode,
    )
#--------------------------------------------------------------------------------------
# End of File
#--------------------------------------------------------------------------------------
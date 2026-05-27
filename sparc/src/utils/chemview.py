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
    vals = [str(i) for f in frames for i in range(len(f))]
    props[key] = {
        "target": "atom",
        "values": vals,
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
    specs,                   # e.g. ["frame","energy","distance:0,7", "angle:0,1,2"]
    x,                       # axis property (structure)
    y,                       # axis property (structure)
    z=None,                  # [optional] axis property (structure)
    names=None,              # rename keys for specs (same length as specs)
    units_override=None,
    **kwargs,                # meta_name, color_atoms, labels, show_index, env_cutoff, map_color, plot
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

    Other Parameters
    ----------------
    meta_name : str, optional
        Dataset label shown in Chemiscope. Default ``"ChemView"``.
    color_atoms : str or None, optional
        Atom colouring in the 3-D viewer: ``"element"``, ``"atom_index"``, or ``None``.
        Default ``"element"``.
    labels : bool, optional
        Show element symbols on atoms. Default ``False``.
    show_index : bool, optional
        Show atom index numbers on atoms. Default ``False``.
    env_cutoff : float, optional
        Cutoff radius (Å) for atom environment selection. Default ``3.5``.
    map_color : str or None, optional
        Property used to colour scatter-plot points. Must be one of ``x``, ``y``, or ``z``.
        Default ``None``.
    plot : bool, optional
        Show the scatter-plot panel alongside the 3-D viewer. Default ``True``.

    Examples
    --------
    Basic usage::

        from sparc.src.utils.chemview import ChemView

        ChemView(frames=traj,
            specs=["frame", "energy", "angle:0,1,7"],
            x="frame",
            y="energy",
            map_color='energy')

    Show atom indices and click-to-select::

        ChemView(frames=traj,
            specs=["frame", "energy"],
            x="frame", y="energy",
            show_index=True,
            color_atoms='atom_index')
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
            vals = [float(np.asarray(at.get_potential_energy()).flat[0]) for at in frames]
            unit = "eV"
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
    show_index  = bool(kwargs.get("show_index", False))
    show_map    = bool(kwargs.get("plot", True))
    env_cutoff  = float(kwargs.get("env_cutoff", 3.5))

    # labels=True  → floating element symbols on atoms
    # show_index   → atom_index property visible on click (no floating labels)
    structure_settings = {
        "unitCell": False,
        "spaceFilling": False,
        "atomLabels": show_labels,
        "environments": {
            "activated": True,
            "center": False,
            "cutoff": env_cutoff,
            "bgColor": "grey",
            "bgStyle": "licorice",
        },
    }
    if color_atoms in ("element", "atom_index"):
        structure_settings["color"] = {"property": color_atoms}

    settings = {"target": "structure", "structure": [structure_settings]}
    mode = "structure"

    if show_map:
        map_settings = {"x": {"property": x}, "y": {"property": y}}
        if z:
            map_settings["z"] = {"property": z}
        valid_colors = {x, y}
        if z:
            valid_colors.add(z)
        if map_color in valid_colors:
            map_settings["color"] = {"property": map_color}
        settings["map"] = map_settings
        mode = "default"   # plot + structure

    import inspect
    _show_params = inspect.signature(chemiscope.show).parameters
    _frames_key  = "frames"    if "frames"    in _show_params else "structures"
    _meta_key    = "meta"      if "meta"      in _show_params else "metadata"
    return chemiscope.show(
        **{_frames_key: frames},
        properties=props,
        **{_meta_key: dict(name=meta_name)},
        environments=chemiscope.all_atomic_environments(frames),
        settings=settings,
        mode=mode,
    )
#--------------------------------------------------------------------------------------
# End of File
#--------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
import numpy as np

# --------------------------------------------------------------------------------------
# soft import
try:
    import chemiscope
except Exception:
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
    if kind == "frame":
        return "frame"
    if idx:
        return kind + "_" + "_".join(str(i) for i in idx)
    return kind


def _ensure_atom_index(frames, props, key="atom_index"):
    if key in props:
        return
    frames = list(frames)
    vals = [i for f in frames for i in range(len(f))]
    props[key] = {
        "target": "atom",
        "values": vals,
        "units": "",
        "description": "Atom index (start from 0)",
    }


def _resolve_axis_name(properties, name):
    """Resolve a property key for map/axis selection."""
    if name in properties:
        return name
    candidates = [k for k in properties.keys() if k == name or k.startswith(name + "_")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise KeyError(
            f"Property '{name}' not found. Available keys: {list(properties.keys())}"
        )
    raise KeyError(
        f"Ambiguous base name '{name}'. Candidates: {candidates}. "
        f"Use one explicitly or pass 'names=[...]' when building."
    )


def _build_properties(frames, specs, names=None, units_override=None):
    """Build Chemiscope property dictionary from ChemView specs."""
    props = {}
    for k, spec in enumerate(specs):
        kind, idx = _parse_spec(spec)
        key = names[k] if names is not None else _auto_name(kind, idx)

        if kind == "frame":
            vals, unit = list(range(len(frames))), ""
        elif kind == "energy":
            vals = [
                float(np.asarray(at.get_potential_energy()).flat[0]) for at in frames
            ]
            unit = "eV"
        elif kind == "distance":
            if len(idx) != 2:
                raise ValueError("distance requires i,j")
            vals = [at.get_distance(idx[0], idx[1]) for at in frames]
            unit = "Å"
        elif kind == "angle":
            if len(idx) != 3:
                raise ValueError("angle requires i,j,k")
            vals = [at.get_angle(idx[0], idx[1], idx[2]) for at in frames]
            unit = "deg"
        elif kind == "dihedral":
            if len(idx) != 4:
                raise ValueError("dihedral requires i,j,k,l")
            vals = [at.get_dihedral(idx[0], idx[1], idx[2], idx[3]) for at in frames]
            unit = "deg"
        elif kind == "volume":
            vals = [float(at.get_volume()) for at in frames]
            unit = "Å³"
        elif kind in ("cell_a", "cell_b", "cell_c"):
            axis = {"cell_a": 0, "cell_b": 1, "cell_c": 2}[kind]
            vals = [float(np.linalg.norm(at.cell[axis])) for at in frames]
            unit = "Å"
        elif kind in ("cell_alpha", "cell_beta", "cell_gamma"):
            from ase.geometry import cell_to_cellpar

            param = {"cell_alpha": 3, "cell_beta": 4, "cell_gamma": 5}[kind]
            vals = [float(cell_to_cellpar(at.cell)[param]) for at in frames]
            unit = "deg"
        elif kind in ("x_pos", "y_pos", "z_pos"):
            col = {"x_pos": 0, "y_pos": 1, "z_pos": 2}[kind]
            vals = [
                float(at.positions[i, col]) for at in frames for i in range(len(at))
            ]
            unit = "Å"
            unit = (units_override or {}).get(key, unit)
            props[key] = {"target": "atom", "values": vals, "units": unit}
            continue
        elif kind == "force_norm":
            _vals = []
            for at in frames:
                try:
                    f = at.get_forces()
                    _vals.extend(float(v) for v in np.linalg.norm(f, axis=1))
                except Exception:
                    _vals.extend([float("nan")] * len(at))
            vals = _vals
            unit = "eV/Å"
            unit = (units_override or {}).get(key, unit)
            props[key] = {"target": "atom", "values": vals, "units": unit}
            continue
        elif kind == "symbol":
            vals = [
                at.get_chemical_symbols()[i] for at in frames for i in range(len(at))
            ]
            unit = ""
            unit = (units_override or {}).get(key, unit)
            props[key] = {"target": "atom", "values": vals, "units": unit}
            continue
        elif kind == "atomic_number":
            vals = [
                int(at.get_atomic_numbers()[i]) for at in frames for i in range(len(at))
            ]
            unit = ""
            unit = (units_override or {}).get(key, unit)
            props[key] = {"target": "atom", "values": vals, "units": unit}
            continue
        elif kind in ("force_x", "force_y", "force_z"):
            col = {"force_x": 0, "force_y": 1, "force_z": 2}[kind]
            _vals = []
            for at in frames:
                try:
                    f = at.get_forces()
                    _vals.extend(float(v) for v in f[:, col])
                except Exception:
                    _vals.extend([float("nan")] * len(at))
            vals = _vals
            unit = "eV/Å"
            unit = (units_override or {}).get(key, unit)
            props[key] = {"target": "atom", "values": vals, "units": unit}
            continue
        elif kind in ("max_devi_f", "maxfdev"):
            vals = [float(at.info.get("max_devi_f", float("nan"))) for at in frames]
            unit = "eV/Å"
        elif kind in ("avg_devi_f", "avgfdev"):
            vals = [
                float(
                    at.info.get(
                        "_avg_devi_f",
                        at.info.get("avg_devi_f", float("nan")),
                    )
                )
                for at in frames
            ]
            unit = "eV/Å"
        elif kind in ("step_index", "md_frame"):
            vals = [
                int(at.info.get("_step_index", at.info.get("frame", index)))
                for index, at in enumerate(frames)
            ]
            unit = ""
        elif kind == "deal_uncertainty":
            vals = [
                float(at.info.get("max_atomic_uncertainty", float("nan")))
                for at in frames
            ]
            unit = ""
        elif kind in ("dft_energy", "labelled_energy"):
            vals = [float(at.info.get("dft_energy", float("nan"))) for at in frames]
            unit = "eV"
        else:
            raise ValueError(f"Unsupported spec kind: {kind!r}")

        unit = (units_override or {}).get(key, unit)
        props[key] = {"target": "structure", "values": vals, "units": unit}

    _ensure_atom_index(frames, props)
    return props


def build_chemview_dataset(
    *,
    frames,
    specs,
    x,
    y,
    z=None,
    names=None,
    units_override=None,
    **kwargs,
):
    """
    Build a Chemiscope dataset dict from ChemView-style arguments.

    Returns
    -------
    dataset : dict
        Chemiscope dataset suitable for ``chemiscope.show`` or Streamlit viewer.
    mode : str
        Chemiscope display mode (``default``, ``structure``, or ``map``).
    """
    _require_chemiscope()

    frames = list(frames)
    if not frames:
        raise ValueError("`frames` is empty")
    if names is not None and len(names) != len(specs):
        raise ValueError("`names` length must match `specs` length")

    props = _build_properties(frames, specs, names=names, units_override=units_override)

    x = _resolve_axis_name(props, x)
    y = _resolve_axis_name(props, y)
    if z is not None:
        z = _resolve_axis_name(props, z)

    map_color = kwargs.get("map_color")
    if map_color is not None:
        map_color = _resolve_axis_name(props, map_color)

    meta_name = kwargs.get("meta_name", "ChemView")
    color_atoms = kwargs.get("color_atoms", "element")
    show_labels = bool(kwargs.get("labels", False))
    show_map = bool(kwargs.get("plot", True))
    env_cutoff = float(kwargs.get("env_cutoff", 3.5))
    include_environments = bool(kwargs.get("include_environments", True))

    structure_settings = {
        "unitCell": False,
        "spaceFilling": False,
        "atomLabels": show_labels,
        "environments": {
            "activated": include_environments,
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
        if map_color is not None and map_color in props:
            map_settings["color"] = {"property": map_color}
        settings["map"] = map_settings
        mode = "default"

    environments = (
        chemiscope.all_atomic_environments(frames) if include_environments else None
    )
    dataset = chemiscope.create_input(
        structures=frames,
        properties=props,
        metadata={"name": meta_name},
        environments=environments,
        settings=settings,
    )
    return dataset, mode


def _show_chemview_dataset(
    dataset, mode, *, warning_timeout=10000, cache_structures=True
):
    """
    Display a dataset already produced by ``create_input``.

    ``chemiscope.show`` runs ``create_input`` again, which breaks when
    ``environments`` are already normalized dicts (structure index becomes the
    literal string ``'structure'``).  Pass the finalized dict straight to the
    Jupyter widget instead.
    """
    from chemiscope.jupyter import ChemiscopeWidget, MapWidget, StructureWidget

    widget_classes = {
        "default": ChemiscopeWidget,
        "structure": StructureWidget,
        "map": MapWidget,
    }
    if mode not in widget_classes:
        raise ValueError(
            f"invalid ChemView mode {mode!r}; expected 'default', 'structure', or 'map'"
        )

    meta = dataset.get("meta") or {}
    has_metadata = meta.get("name", " ") != " "

    payload = dict(dataset)
    properties = dict(payload.get("properties") or {})

    if mode == "structure" and "index" not in properties:
        properties["index"] = {
            "target": "structure",
            "values": list(range(len(payload.get("structures", [])))),
        }
        payload["properties"] = properties

    if mode != "structure" and len(properties) < 2:
        raise ValueError("Need at least two properties to visualize a map widget")

    return widget_classes[mode](
        payload,
        has_metadata=has_metadata,
        warning_timeout=warning_timeout,
        cache_structures=cache_structures,
    )


def ChemView(
    *,
    frames,
    specs,
    x,
    y,
    z=None,
    names=None,
    units_override=None,
    **kwargs,
):
    """Launch Chemiscope viewer in Jupyter using ChemView-style arguments."""
    dataset, mode = build_chemview_dataset(
        frames=frames,
        specs=specs,
        x=x,
        y=y,
        z=z,
        names=names,
        units_override=units_override,
        **kwargs,
    )
    return _show_chemview_dataset(
        dataset,
        mode,
        warning_timeout=kwargs.get("warning_timeout", 10000),
        cache_structures=kwargs.get("cache_structures", True),
    )


# --------------------------------------------------------------------------------------
# End of File
# --------------------------------------------------------------------------------------

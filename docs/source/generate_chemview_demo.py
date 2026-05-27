"""
Generate a self-contained ChemView demo HTML for the Sphinx docs.

Usage (requires ASE):
    python docs/source/generate_chemview_demo.py [path/to/file.traj]

Outputs:
    docs/source/_static/chemview_demo.html   — standalone viewer (no CDN)
    docs/source/_static/chemview_demo.json   — raw chemiscope dataset

The bundled chemiscope JS/CSS must be present in _static/:
    chemiscope.min.js, chemiscope-sphinx.js, chemiscope-sphinx.css

Copy them from an installed chemiscope package:
    CHEM=$(python -c "import chemiscope, os; print(os.path.dirname(chemiscope.__file__))")
    cp $CHEM/sphinx/static/chemiscope.min.js    docs/source/_static/
    cp $CHEM/sphinx/static/chemiscope-sphinx.js  docs/source/_static/
    cp $CHEM/sphinx/static/chemiscope-sphinx.css docs/source/_static/
"""
import json
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRAJ = PROJECT_ROOT / "examples/CP2K/workflow/TrajCombined.traj"
STATIC_DIR   = Path(__file__).resolve().parent / "_static"
OUTPUT_HTML  = STATIC_DIR / "chemview_demo.html"
OUTPUT_JSON  = STATIC_DIR / "chemview_demo.json"


def _build_dataset(frames):
    structures, energies, bn_distances, force_norms = [], [], [], []

    for atoms in frames:
        pos = atoms.positions
        structures.append({
            "size":  len(atoms),
            "names": atoms.get_chemical_symbols(),
            "x":     pos[:, 0].tolist(),
            "y":     pos[:, 1].tolist(),
            "z":     pos[:, 2].tolist(),
            **({"cell": atoms.cell.flatten().tolist()} if atoms.pbc.any() else {}),
        })

        energies.append(float(np.asarray(atoms.get_potential_energy()).flat[0]))
        bn_distances.append(float(atoms.get_distance(0, 7)))  # B=0, N=7

        try:
            f = atoms.get_forces()
            force_norms.extend(np.linalg.norm(f, axis=1).tolist())
        except Exception:
            force_norms.extend([None] * len(atoms))

    n = len(frames)
    e0 = energies[0]

    properties = {
        "frame": {
            "target": "structure",
            "values": list(range(n)),
            "units":  "",
            "description": "Frame index",
        },
        "energy": {
            "target": "structure",
            "values": energies,
            "units":  "eV",
            "description": "Potential energy",
        },
        "relative_energy": {
            "target": "structure",
            "values": [(e - e0) * 1000 for e in energies],
            "units":  "meV",
            "description": "Energy relative to first frame",
        },
        "BN_distance": {
            "target": "structure",
            "values": bn_distances,
            "units":  "Angstrom",
            "description": "B-N bond distance",
        },
        "force_norm": {
            "target": "atom",
            "values": force_norms,
            "units":  "eV/Angstrom",
            "description": "Force magnitude per atom",
        },
        "atom_index": {
            "target": "atom",
            "values": [str(i) for f in frames for i in range(len(f))],
            "units":  "",
            "description": "Atom index (0-based)",
        },
    }

    environments = [
        {"structure": i, "center": j, "cutoff": 3.5}
        for i in range(n)
        for j in range(len(frames[i]))
    ]

    settings = {
        "target": "structure",
        "map": {
            "x":     {"property": "frame"},
            "y":     {"property": "BN_distance"},
            "color": {"property": "relative_energy"},
        },
        "structure": [{"unitCell": False, "atomLabels": False, "spaceFilling": False}],
    }

    return {
        "meta":         {"name": "ChemView Demo - Ammonia Borane"},
        "structures":   structures,
        "properties":   properties,
        "environments": environments,
        "settings":     settings,
    }


def _render_html(dataset_json: str) -> str:
    # Correct chemiscope API (from chemiscope-sphinx.js):
    #   config = { map: id, info: id, meta: id, structure: id }   (DOM element IDs)
    #   Chemiscope.DefaultVisualizer.load(config, dataset, warnings)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ChemView Demo</title>
  <link rel="stylesheet" href="chemiscope-sphinx.css">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ height: 100%; background: #f5f7fa; font-family: system-ui, sans-serif; }}
    #root {{ height: 100vh; display: flex; flex-direction: column; }}
    #banner {{
      flex-shrink: 0;
      background: rgba(44,62,80,.9); color: #ecf0f1;
      padding: 6px 14px; font-size: 13px;
      display: flex; align-items: center; gap: 12px;
    }}
    #banner strong {{ color: #fff; }}
    #viewer-root {{ flex: 1; overflow: hidden; }}
    #viewer-root .chemiscope-sphinx {{ height: 100%; }}
    #viewer-root .visualizer-container {{ height: 100%; }}
    #err {{
      padding: 2em; color: #c0392b; background: #fdf3f2;
      border-left: 4px solid #c0392b; margin: 1em;
    }}
  </style>
</head>
<body>
<div id="root">
  <div id="banner">
    <strong>ChemView Demo</strong>
    Ammonia borane (NH₃BH₃) &middot; 264 frames &middot; B&ndash;N distance vs frame, coloured by &Delta;E
  </div>
  <div id="viewer-root">
    <div class="chemiscope-sphinx">
      <div class="visualizer-container">
        <div class="visualizer-column-right">
          <div id="cv-meta"></div>
          <div id="cv-map"></div>
        </div>
        <div class="visualizer-column">
          <div id="cv-structure" class="visualizer-item"></div>
          <div id="cv-info"      class="visualizer-info"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>const DATASET = {dataset_json};</script>
<script src="chemiscope.min.js"></script>
<script>
(async () => {{
  const config = {{
    map:       'cv-map',
    info:      'cv-info',
    meta:      'cv-meta',
    structure: 'cv-structure',
  }};
  try {{
    const warnings = new Chemiscope.Warnings();
    await Chemiscope.DefaultVisualizer.load(config, DATASET, warnings);
  }} catch (err) {{
    const div = document.createElement('div');
    div.id = 'err';
    div.innerHTML = '<strong>ChemView failed to load:</strong><br><code>' + err + '</code>';
    document.getElementById('viewer-root').replaceWith(div);
  }}
}})();
</script>
</body>
</html>"""


def main():
    traj_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRAJ
    if not traj_path.exists():
        sys.exit(f"Trajectory not found: {traj_path}")

    try:
        from ase.io import read
    except ImportError:
        sys.exit("ASE is required: pip install ase")

    print(f"Reading {traj_path} ...")
    frames = read(str(traj_path), index=":")
    print(f"  {len(frames)} frames, {len(frames[0])} atoms/frame")

    dataset = _build_dataset(frames)
    dataset_json = json.dumps(dataset)

    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_JSON.write_text(dataset_json, encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}  ({OUTPUT_JSON.stat().st_size // 1024} KB)")

    OUTPUT_HTML.write_text(_render_html(dataset_json), encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML}  ({OUTPUT_HTML.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

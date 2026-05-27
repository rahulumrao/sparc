"""
Generate a self-contained ChemView demo HTML for the Sphinx docs.

Usage (requires ASE):
    python docs/source/generate_chemview_demo.py [path/to/file.traj]

Output: docs/source/_static/chemview_demo.html
"""
import json
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRAJ = PROJECT_ROOT / "examples/CP2K/workflow/TrajCombined.traj"
OUTPUT_HTML = Path(__file__).resolve().parent / "_static/chemview_demo.html"

CHEMISCOPE_CSS = "https://chemiscope.org/chemiscope.min.css"
CHEMISCOPE_JS  = "https://chemiscope.org/chemiscope.min.js"


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
            "units":  "Å",
            "description": "B–N bond distance",
        },
        "force_norm": {
            "target": "atom",
            "values": force_norms,
            "units":  "eV/Å",
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
        "meta":         {"name": "ChemView Demo — Ammonia Borane"},
        "structures":   structures,
        "properties":   properties,
        "environments": environments,
        "settings":     settings,
    }


def _render_html(dataset_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ChemView Demo</title>
  <link rel="stylesheet" href="{CHEMISCOPE_CSS}">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #f5f7fa; font-family: system-ui, sans-serif; height: 100vh; }}
    #chemview {{ width: 100%; height: 100vh; }}
    #banner {{
      position: absolute; top: 0; left: 0; right: 0; z-index: 10;
      background: rgba(44,62,80,.85); color: #ecf0f1;
      padding: 6px 14px; font-size: 13px;
      display: flex; align-items: center; gap: 12px;
    }}
    #banner a {{ color: #3498db; }}
    #loading {{
      position: absolute; top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      font-size: 1.1em; color: #555;
    }}
  </style>
</head>
<body>
  <div id="banner">
    <strong>ChemView Demo</strong>
    Ammonia borane (NH₃BH₃) · 264 frames · B–N distance vs frame, coloured by ΔE
    &nbsp;|&nbsp;
    <a href="https://chemiscope.org" target="_blank">chemiscope.org</a>
  </div>
  <div id="loading">Loading viewer…</div>
  <div id="chemview"></div>

  <script>const DATASET = {dataset_json};</script>
  <script src="{CHEMISCOPE_JS}"></script>
  <script>
    document.addEventListener('DOMContentLoaded', async () => {{
      document.getElementById('loading').remove();
      try {{
        await Chemiscope.DefaultVisualizer.load({{
          element: document.getElementById('chemview'),
          dataset:  DATASET,
        }});
      }} catch (err) {{
        document.getElementById('chemview').innerHTML =
          '<p style="padding:2em;color:#c0392b;">' +
          'Could not load the chemiscope widget.<br>' +
          'Make sure you are connected to the internet (loads JS from chemiscope.org).<br>' +
          '<code>' + err + '</code></p>';
      }}
    }});
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
    dataset_json = json.dumps(dataset, allow_nan=False)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(_render_html(dataset_json), encoding="utf-8")
    size_kb = OUTPUT_HTML.stat().st_size // 1024
    print(f"Wrote {OUTPUT_HTML}  ({size_kb} KB)")


if __name__ == "__main__":
    main()

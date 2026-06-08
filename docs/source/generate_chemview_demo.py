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
            "values": [i for f in frames for i in range(len(f))],
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


def _render_example_html(title: str, dataset_json: str) -> str:
    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="chemiscope-sphinx.css">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}html,body{{height:100%;background:#f5f7fa;font-family:system-ui,sans-serif}}
#root{{height:100vh;display:flex;flex-direction:column}}
#hdr{{flex-shrink:0;background:rgba(44,62,80,.9);color:#ecf0f1;padding:5px 12px;font-size:12px}}
#hdr strong{{color:#fff}}#vr{{flex:1;overflow:hidden}}
#vr .chemiscope-sphinx{{height:100%}}#vr .visualizer-container{{height:100%}}
</style></head><body>
<div id="root">
<div id="hdr"><strong>{title}</strong> &nbsp;&middot;&nbsp; Ammonia borane NH&#x2083;BH&#x2083; &middot; 264 frames</div>
<div id="vr"><div class="chemiscope-sphinx"><div class="visualizer-container">
<div class="visualizer-column-right"><div id="cv-meta"></div><div id="cv-map"></div></div>
<div class="visualizer-column"><div id="cv-structure" class="visualizer-item"></div><div id="cv-info" class="visualizer-info"></div>
</div></div></div></div></div>
<script>const DATASET={dataset_json};</script>
<script src="chemiscope.min.js"></script>
<script>(async()=>{{const cfg={{map:'cv-map',info:'cv-info',meta:'cv-meta',structure:'cv-structure'}};
try{{await Chemiscope.DefaultVisualizer.load(cfg,DATASET,new Chemiscope.Warnings());}}
catch(e){{document.getElementById('vr').innerHTML='<p style="padding:1em;color:red">'+e+'</p>';}}
}})();</script></body></html>"""


EXAMPLES = [
    # (filename_stem, extra_structure_props, extra_atom_props, settings, title)
    (
        "chemview_ex1",
        {},
        {},
        {"target":"structure","map":{"x":{"property":"frame"},"y":{"property":"energy"},"color":{"property":"energy"}},"structure":[{"unitCell":False}]},
        "Example 1 — Energy vs Frame",
    ),
    (
        "chemview_ex2",
        {"d_BN": lambda frames: [float(at.get_distance(0,7)) for at in frames]},
        {},
        {"target":"structure","map":{"x":{"property":"frame"},"y":{"property":"d_BN"},"color":{"property":"energy"}},"structure":[{"unitCell":False}]},
        "Example 2 — B–N Distance vs Frame",
    ),
    (
        "chemview_ex3",
        {},
        {"force_norm": lambda frames: [float(v) for at in frames for v in __import__('numpy').linalg.norm(at.get_forces(), axis=1)]},
        {"target":"structure","map":{"x":{"property":"frame"},"y":{"property":"energy"},"color":{"property":"energy"}},"structure":[{"unitCell":False}]},
        "Example 3 — Atom-level Force Norms",
    ),
    (
        "chemview_ex4",
        {
            "d_BN":      lambda frames: [float(at.get_distance(0,7)) for at in frames],
            "d_BH":      lambda frames: [float(at.get_distance(0,1)) for at in frames],
            "angle_HBN": lambda frames: [float(at.get_angle(1,0,7))  for at in frames],
        },
        {},
        {"target":"structure","map":{"x":{"property":"d_BN"},"y":{"property":"angle_HBN"},"color":{"property":"energy"}},"structure":[{"unitCell":False}]},
        "Example 4 — Multi-property (dᴮₙ, angle H–B–N)",
    ),
]


def _build_example_dataset(frames, extra_struct, extra_atom, settings):
    n = len(frames)
    structures = []
    for at in frames:
        pos = at.positions
        s = {"size":len(at),"names":at.get_chemical_symbols(),
             "x":pos[:,0].tolist(),"y":pos[:,1].tolist(),"z":pos[:,2].tolist()}
        if at.pbc.any(): s["cell"] = at.cell.flatten().tolist()
        structures.append(s)

    energies = [float(np.asarray(at.get_potential_energy()).flat[0]) for at in frames]
    props = {
        "frame":  {"target":"structure","values":list(range(n)),"units":"","description":"Frame index"},
        "energy": {"target":"structure","values":energies,"units":"eV","description":"Potential energy"},
    }
    for key, fn in extra_struct.items():
        vals = fn(frames)
        props[key] = {"target":"structure","values":vals,"units":"","description":key}
    for key, fn in extra_atom.items():
        vals = fn(frames)
        props[key] = {"target":"atom","values":vals,"units":"","description":key}
    props["atom_index"] = {"target":"atom","values":[i for f in frames for i in range(len(f))],"units":"","description":"Atom index"}

    envs = [{"structure":i,"center":j,"cutoff":3.5} for i in range(n) for j in range(len(frames[i]))]
    return {"meta":{"name":"ChemView Demo"},"structures":structures,"properties":props,"environments":envs,"settings":settings}


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

    # generate per-example files for the notebook page
    for stem, extra_struct, extra_atom, settings, title in EXAMPLES:
        ds = _build_example_dataset(frames, extra_struct, extra_atom, settings)
        dj = json.dumps(ds)
        out = STATIC_DIR / f"{stem}.html"
        out.write_text(_render_example_html(title, dj), encoding="utf-8")
        print(f"Wrote {out.name}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

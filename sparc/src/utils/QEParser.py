#!/usr/bin/python3
# QEParser.py

"""
Parser for Quantum ESPRESSO template files (pw.x input format).

Reads a QE input file and extracts namelists (&CONTROL, &SYSTEM, &ELECTRONS,
&IONS, &CELL) and cards (K_POINTS, ATOMIC_SPECIES) into a dictionary suitable
for the ASE Espresso calculator.

Expected template structure
---------------------------
A standard pw.x input file. Coordinates and cell parameters are ignored
(ASE provides them from the Atoms object). The parser extracts:

  - input_data   : dict of all namelist parameters (flat)
  - kpts         : k-point grid as (k1, k2, k3) tuple, or None for Gamma-only (default)
  - koffset      : k-point offset as (o1, o2, o3) tuple — only read from template if present
  - pseudopotentials : {element: filename} mapping from ATOMIC_SPECIES card

Example template (qe_template.inp)
-----------------------------------
    &CONTROL
      calculation = 'scf'
      tprnfor = .true.
      tstress = .true.
      pseudo_dir = './pseudo'
    /
    &SYSTEM
      ecutwfc = 60
      ecutrho = 480
      occupations = 'smearing'
      smearing = 'cold'
      degauss = 0.01
    /
    &ELECTRONS
      conv_thr = 1.0d-8
      mixing_beta = 0.35
    /
    ATOMIC_SPECIES
      Si  28.085  Si.pbe-n-rrkjus_psl.1.0.0.UPF
      O   15.999  O.pbe-n-rrkjus_psl.1.0.0.UPF

By default, Gamma-only k-points are used (~50% less memory/CPU).
To use a k-point grid, add the following to the template:

    K_POINTS automatic
      4 4 4  0 0 0
              ↑ offset (optional, only needed if required)
"""

import re
from pathlib import Path
from sparc.src.utils.logger import SparcLog


def qe_template(template_path: str) -> dict:
    """
    Parse a Quantum ESPRESSO pw.x input template file.

    Parameters
    ----------
    template_path : str
        Path to the QE template file.

    Returns
    -------
    dict with keys:
        'input_data'        : dict  — flat dictionary of namelist parameters
        'pseudopotentials'  : dict  — {element: pseudopotential_filename}
        'pseudo_dir'        : str or None — pseudo_dir if found in CONTROL
        'kpts'              : tuple of 3 ints, or None (default: None = Gamma-only)
        'koffset'           : tuple of 3 ints, or None (only set if template has offset)
    """
    p = Path(template_path)
    if not p.exists():
        raise FileNotFoundError(f"QE template not found: {template_path}")

    with p.open() as f:
        raw = f.read()

    # ── Parse namelists (&CONTROL ... /, &SYSTEM ... /, etc.) ──
    input_data = {}
    namelist_re = re.compile(
        r'&(\w+)\s*\n(.*?)\n\s*/', re.DOTALL | re.IGNORECASE
    )
    for match in namelist_re.finditer(raw):
        section_name = match.group(1).upper()
        body = match.group(2)
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            # Handle comma-separated parameters on the same line
            for part in line.split(','):
                part = part.strip()
                if '=' not in part:
                    continue
                key, val = part.split('=', 1)
                key = key.strip().lower()
                val = _convert_qe_value(val.strip())
                input_data[key] = val

    # Extract pseudo_dir before passing to ASE (ASE handles it via profile)
    pseudo_dir = input_data.pop('pseudo_dir', None)
    if isinstance(pseudo_dir, str):
        pseudo_dir = pseudo_dir.strip("'\"")

    # ── Parse ATOMIC_SPECIES card ──
    pseudopotentials = {}
    species_re = re.compile(
        r'ATOMIC_SPECIES\s*\n(.*?)(?=\n\s*(?:ATOMIC_POSITIONS|K_POINTS|'
        r'CELL_PARAMETERS|CONSTRAINTS|OCCUPATIONS|ATOMIC_FORCES|&|\Z))',
        re.DOTALL | re.IGNORECASE
    )
    m = species_re.search(raw)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                element = parts[0]
                pp_file = parts[2]
                pseudopotentials[element] = pp_file

    # ── Parse K_POINTS card ──
    # Default: Gamma-only (kpts=None triggers QE's optimised Γ-point routines)
    # koffset is only set when the template explicitly provides offset values
    kpts = None
    koffset = None
    kpts_re = re.compile(
        r'K_POINTS\s*[\{\(]?\s*(\w+)\s*[\}\)]?\s*\n(.*?)(?=\n\s*(?:&|\Z|[A-Z_]+\s))',
        re.DOTALL | re.IGNORECASE
    )
    km = kpts_re.search(raw)
    if km:
        kpts_type = km.group(1).lower()
        kpts_body = km.group(2).strip()
        if kpts_type in ('automatic', 'auto'):
            parts = kpts_body.split()
            if len(parts) >= 3:
                kpts = (int(parts[0]), int(parts[1]), int(parts[2]))
            if len(parts) >= 6:
                koffset = (int(parts[3]), int(parts[4]), int(parts[5]))
        elif kpts_type == 'gamma':
            kpts = None  # ASE uses kpts=None for gamma-only

    return {
        'input_data': input_data,
        'pseudopotentials': pseudopotentials,
        'pseudo_dir': pseudo_dir,
        'kpts': kpts,
        'koffset': koffset,
    }


def _convert_qe_value(val: str):
    """
    Convert a QE namelist value string to a Python type.

    Handles Fortran-style booleans (.true./.false.), integers, floats
    (including 'd' exponent notation), and quoted strings.
    """
    v = val.strip().rstrip(',')

    # Fortran booleans
    if v.lower() in ('.true.', '.t.'):
        return True
    if v.lower() in ('.false.', '.f.'):
        return False

    # Quoted string
    if (v.startswith("'") and v.endswith("'")) or \
       (v.startswith('"') and v.endswith('"')):
        return v[1:-1]

    # Fortran 'd' exponent -> 'e'
    v_num = v.lower().replace('d', 'e')

    # Try int first, then float
    try:
        return int(v_num)
    except ValueError:
        pass
    try:
        return float(v_num)
    except ValueError:
        pass

    # Return as string (strip quotes if any)
    return v.strip("'\"")

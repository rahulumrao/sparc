#!/usr/bin/python3
# GaussianParser.py

"""
Parser for Gaussian template files.

Reads a key = value template and returns parameters to pass directly
to ASE's Gaussian calculator. ASE handles Link0 (%mem, %chk, %nprocshared)
vs route section sorting internally.

The calculator always runs as a force calculation (force=None is set
in calculator.py) since ASE drives the MD.

Expected template structure
---------------------------
A keyword = value file (one per line). Comments start with '#' or '!'.
Use 'nproc' or 'nprocshared' for processor count — both are accepted.

Example template (gaussian_template.inp)
----------------------------------------
    # Gaussian template for SPARC
    method    = b3lyp
    basis     = 6-31G*
    charge    = 0
    multiplicity = 1
    mem       = 4GB
    nprocshared = 8
    chk       = job.chk
    scf       = qc,maxcycle=200
"""

from pathlib import Path
from sparc.src.utils.logger import SparcLog


def gaussian_template(template_path: str) -> dict:
    """
    Parse a Gaussian template file into kwargs for ASE's Gaussian calculator.

    Parameters
    ----------
    template_path : str
        Path to the Gaussian template file.

    Returns
    -------
    dict
        Flat dictionary of keyword arguments ready to pass to Gaussian().
        'charge' and 'multiplicity' are converted to 'charge' and 'mult'.
    """
    p = Path(template_path)
    if not p.exists():
        raise FileNotFoundError(f"Gaussian template not found: {template_path}")

    params = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('!'):
                continue
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip().lower()
            val = val.strip()
            params[key] = val

    # ── Required parameters ──
    if 'method' not in params:
        raise ValueError("Gaussian template missing 'method' (e.g. b3lyp, mp2, hf)")
    if 'basis' not in params:
        raise ValueError("Gaussian template missing 'basis' (e.g. 6-31G*, cc-pVDZ)")

    # ── Normalize keys for ASE ──
    # ASE expects 'mult' not 'multiplicity'
    if 'multiplicity' in params:
        params['mult'] = int(params.pop('multiplicity'))
    if 'charge' in params:
        params['charge'] = int(params.pop('charge'))

    # ASE recognises 'nprocshared' as Link0 — map 'nproc' to 'nprocshared'
    if 'nproc' in params:
        params['nprocshared'] = params.pop('nproc')

    # Try numeric conversion for remaining values
    for k in list(params):
        if k in ('method', 'basis', 'mem', 'chk', 'oldchk', 'scf',
                 'extra', 'save', 'charge', 'mult', 'nprocshared'):
            continue
        v = params[k]
        try:
            params[k] = int(v)
        except ValueError:
            try:
                params[k] = float(v)
            except ValueError:
                pass

    return params

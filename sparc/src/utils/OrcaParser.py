#!/usr/bin/python3
# OrcaParser.py
################################################################
import re
import shutil
from pathlib import Path

from sparc.src.utils.logger import SparcLog

################################################################
orca_notice = False


# ==================================================================================
def parse_orca_template(template_path: str):
    """
    Parse a minimal ORCA template file.

    Expected structure (order flexible):
      - One line starting with '!'  -> orcasimpleinput
      - Zero or more %... blocks   -> orcablocks (including 'end' lines)
      - One '*xyz charge mult' line -> charge, mult (coords optional and ignored here)

    Returns
    -------
    dict with keys:
      'orcasimpleinput': str
      'orcablocks': str (possibly empty)
      'charge': int (default 0 if not found)
      'mult': int   (default 1 if not found)
    """
    p = Path(template_path)
    if not p.exists():
        raise FileNotFoundError(f"ORCA template not found: {template_path}")

    keyword = None
    blocks_lines = []
    charge = 0
    mult = 1

    with p.open() as f:
        lines = f.readlines()

    # 1) ! line (simpleinput)
    for line in lines:
        s = line.strip()
        if s.startswith("!"):
            # keep everything after '!' as-is
            keyword = s[1:].strip()
            break

    # 2) % blocks (collect all lines that belong to blocks)
    in_block = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%"):
            in_block = True
            blocks_lines.append(line.rstrip("\n"))
            continue
        if in_block:
            blocks_lines.append(line.rstrip("\n"))
            # Heuristic: many ORCA blocks end with an 'end' on its own line.
            # We *do not* force-close at 'end' in case multiple lines follow;
            # we stop when we hit a new section (e.g., '*xyz').
            if stripped.lower().startswith("*xyz"):
                # back up: don't include *xyz in blocks; end the block before it
                blocks_lines.pop()
                in_block = False
                break

    # 3) charge mult
    cm_pat = re.compile(r"^\*+\s*xyz\s+(-?\d+)\s+(\d+)", re.IGNORECASE)
    for line in lines:
        m = cm_pat.match(line.strip())
        if m:
            charge = int(m.group(1))
            mult = int(m.group(2))
            break

    if keyword is None:
        raise ValueError("Template missing the '!' ORCA keyword line.")

    blocks = ""
    if blocks_lines:
        # Trim trailing empty lines and keep user formatting
        while blocks_lines and not blocks_lines[-1].strip():
            blocks_lines.pop()
        blocks = "\n".join(blocks_lines)

    NPROCS_RE = re.compile(r"%\s*pal\b.*?\bnprocs?\s+(\d+)", re.IGNORECASE | re.DOTALL)
    np_match = NPROCS_RE.search(blocks or "")
    global orca_notice
    if np_match and not orca_notice:
        nprocs = int(np_match.group(1))
        if nprocs > 1:
            mpirun = shutil.which("mpirun")
            lvl = "WARNING" if not mpirun else "INFO"
            # ---------------------------------------------------------------------
            SparcLog(f" ORCA parallel requested (nprocs={nprocs})", level=lvl)
            SparcLog(f" mpirun: {mpirun or 'NOT FOUND'}", level=lvl)
            SparcLog(" Ensure load MPI libraries before proceeding.", level=lvl)
            # ---------------------------------------------------------------------
            orca_notice = True

    return {
        "orcasimpleinput": keyword,
        "orcablocks": blocks,
        "charge": charge,
        "multi": mult,
    }


#  Please verify the MPI configuration to ensure proper parallelization before continuing.",

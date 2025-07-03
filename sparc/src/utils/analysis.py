#!/usr/bin/env python3

"""
SPARC Analysis Utility

This script acts as the entry point for running various post-processing
or analysis routines in SPARC, independent of the main workflow.
"""
#--------------------------------------------------------------------------------------#
import argparse
import sys

# Import analysis modules
from sparc.src.utils.logger import SparcLog
from sparc.src.utils.mlp_pes import get_energies
# (Later can be add more: rdf_analysis, mean_force, diffusion, etc.)
#--------------------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser(description='SPARC Analysis Tools')
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available analysis commands")

    # ---------------------------
    # Energy Calculate CLI
    # ---------------------------
    cmp_parser = subparsers.add_parser("get_energies", help="Calculate DFT and ML potential energies for frames in a trajectory")
    cmp_parser.add_argument("--dft_file", type=str, required=True, help="Path to the reference trajectory file containing DFT energies (e.g., OUTCAR).")
    cmp_parser.add_argument("--ifmt", type=str, default="vasp-out", help="ASE-compatible format for reading the input trajectory [default: vasp-out]")
    cmp_parser.add_argument("--skip", type=int, default=1, help="Stride for frame extraction: analyze every nth frame [default: 1]")
    cmp_parser.add_argument("--model", type=str, default='training_2/frozen_model_2.pb', help="Name of frozen model [default: 'frozen_model_2.pb']")
    cmp_parser.add_argument("--bond", type=int, nargs=2, default=[0, 1], help="Bond indices to monitor [default: 0 1]")
    cmp_parser.add_argument("--npar", type=int, default=4, help="Number of processors for parellel execution [default: 4]")
    cmp_parser.add_argument("--out", type=str, default="energies.csv", help="File to store the DFT and ML energies in CSV format [default: 'energies.csv']")

    args = parser.parse_args()

    if args.command == "get_energies":
        get_energies(
            dft_file=args.dft_file, 
            ifmt=args.ifmt,
            skip=args.skip,
            model=args.model,
            bond=args.bond,
            out=args.out,
            npar=args.npar
        )
    else:
        SparcLog(f"Unknown analysis command: {args.command}", level="ERROR", origin="ANALYSIS")
        sys.exit(1)

if __name__ == "__main__":
    main()
#===================================================================================================#
#                                     END OF FILE 
#===================================================================================================#   

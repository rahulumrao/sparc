#!/usr/bin/python3
# rmsd.py
################################################################
import yaml
import os
import sys
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Iterable
from sparc.src.utils.logger import SparcLog
################################################################
# Third Party Import
from ase.io import read, write
################################################################
"""
    This code calculate Root Mean Square Deviation (RMSD) between two structural configuraition, P and Q.
    The RMSD is calculated by finding a best fit for rotation between vector P and Q using `Kabsch algorithm`.

    Functions
    ---------
    kabsch_rmsd(P, Q) -> float
        Return RMSD after aligning P onto Q
        P, Q: (N, 3) array

    kabsch_align(P, Q) ->
        Align P onto Q using Kabsch using `scipy`
        Return
        ~~~~~~
            - Paligned  : (N, 3) = rot.apply(P, tP)
            - rot       : scipy.spatial.transform.Rotation
            - tP, tQ    : (3, ) vrnroids of P and Q

    kabsch_rmsd(P, Q, symbol) -> (P_noH, Q_noH)
        Remove hydrogen ('H') from P and Q while calculating RMSD.

    Examples
    --------
        # Basic Usage
        >>> from ase.io import read
        >>> from rmsd import kabsch_rmsd
        >>> A = read('structure_1.xyz')
        >>> B = read('structure_2.xyz')
        >>> r = kabsch_rmsd(A.get_positions(), B.get_positions())
        >>> print(f"RMSD : {r:.3f}")
        >>> r = kabsch_rmsd(A.get_positions(), B.get_positions(), noH=True, symbols=A.get_chemical_symbols())
        >>> print(f"RMSD : {r:.3f}")
"""

#---------------------------------------
#  Translate both vector to the center #
#---------------------------------------
def t_center(X):
    '''
        Translate coordinates to the center so that their centriod concides
    '''
    X = np.asarray(X, float)
    c = X.mean(axis=0)
    return X - c, c

def v_align(P ,Q, noH=False, symbols=None):
    '''
        Align both P and Q vector.

        Return
        ------
            tuple: (aligned_P, rotation, centroid_P, centroid_Q, rmsd)
    '''
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.shape[1] !=3:
        raise ValueError("P and Q must have shape (N, 3).")
    
    #----------------------------------------
    #       RMSD without H-atoms            #
    #----------------------------------------
    if noH == True and symbols != None :
        sym = np.asarray(list(symbols))
        mask = sym != "H"
        if mask.sum() == 0:
            raise ValueError("All atoms are hydrogen nothing left in the system.")
        P, Q = P[mask], Q[mask]
    elif noH == True and symbols == None:
        SparcLog(f"Could not find atomic symbols!")
    
    Pc, tP = t_center(P)
    Qc, tQ = t_center(Q)

    # Suppress UserWarning for poorly defined rotations (common for similar structures)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Optimal rotation is not uniquely or poorly defined")
        rot, _ = Rotation.align_vectors(Qc, Pc)

    # rot, _ = Rotation.align_vectors(Qc, Pc)
    Paligned = rot.apply(Pc)

    # RMSD
    residual = ((Paligned - Qc)**2).sum(axis = 1)
    rmsd = float(np.sqrt(residual.mean()))

    return Paligned, rot, tP, tQ, rmsd

def kabsch_rmsd(P, Q, noH=False, symbols=None):
    '''
        Returns the RMSD after aligining both P and Q vector.
    '''
    return v_align(P, Q, noH, symbols)[-1]

#--------------------------
#  CLI interface for RMSD #
#--------------------------
def rmsd():

    import argparse
    from ase.io import read

    arg = argparse.ArgumentParser(description="Calculate RMSD using Kabsch Algorithm")
    arg.add_argument("strA", help='Conformation 1 in ASE readable format (eg: XYZ)')
    arg.add_argument("strB", help='Conformation 2 in ASE readable format (eg: XYZ)')
    arg.add_argument("--no-hydrogen", action="store_true", help="Remove Hydrogen before computing RMSD")
    arg.add_argument("--tol", type=float, default=0.10, help="tolerance for conformer")
    args = arg.parse_args()

    A, B = read(args.strA), read(args.strB)
    P, Q = A.get_positions(), B.get_positions()

    if P.shape != Q.shape:
        SparcLog(f"Different atom count: {P.shape[0]} vs {Q.shape[0]}")
        return

    if args.no_hydrogen:
        symsA, symsB = A.get_chemical_symbols(), B.get_chemical_symbols()
        if sorted(x for x in symsA if x != "H") != sorted(x for x in symsB if x != "H"):
            SparcLog(f"Different heavy atom composition -> can't compare!")
            return

        # P, Q = v_align(P, Q, True, symsA)
        _, rot, tP, tQ, r = v_align(P, Q, True, symsA)
    else:
        _, rot, tP, tQ, r = v_align(P, Q, False, None)

    SparcLog(rf"RMSD : {r:.6f} Ang.")
    SparcLog("Appears to be:","Same Conformers" if r <= args.tol else "Different Conformers")
    SparcLog("Rotation matrix R:\n", rot.as_matrix())
    SparcLog(f"Centroid A: {tP}")
    SparcLog(f"Centroid B: {tQ}")

if __name__ == "__main__":
    rmsd()
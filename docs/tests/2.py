from ase.calculators.lj import LennardJones
from ase.calculators.plumed import Plumed
#from ase.calculators.plumed import Plumed as pl
#from plumed import Plumed as pl
from ase.constraints import FixedPlane
from ase.md.langevin import Langevin
from ase.io import read
from ase import units

timestep = 0.005
ps = 1000 * units.fs

#setup = open("plumed.dat", "r").read().splitlines()
with open("plumed.dat", "r") as file:
    setup = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]

#setup = [f" UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}",
#         "d1: DISTANCE ATOMS=1,4",
#         "mat: CONTACT_MATRIX ATOMS=1-10 SWITCH={RATIONAL R_0=1.0}",
#         "PRINT ARG=c1.* STRIDE=100 FILE=COLVAR"]

#setup = [f" UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}",
#	"mat: CONTACT_MATRIX ATOMS=1-7 SWITCH={{RATIONAL R_0=1.0}}",
#	"rsums: COLUMNSUMS MATRIX=mat MEAN",
#	"PRINT ARG=rsums.* FILE=COLVAR"]

atoms = read('isomer.xyz')
cons = [FixedPlane(i, [0, 0, 1]) for i in range(7)]
atoms.set_constraint(cons)
atoms.set_masses([1, 1, 1, 1, 1, 1, 1])

atoms.calc = Plumed(calc=LennardJones(rc=2.5, r0=3.),
                    input=setup,
                    timestep=timestep,
                    atoms=atoms,
                    kT=0.1)

dyn = Langevin(atoms, timestep, temperature_K=0.1/units.kB, friction=1,
               fixcm=False, trajectory='UnbiasMD.xyz')

dyn.run(1000)


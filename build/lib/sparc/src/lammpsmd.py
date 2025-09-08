
import os
import glob
import shutil
import numpy as np
from datetime import datetime
from copy import deepcopy
from ase import units
from ase.io import read, write
from ase.md.md import MolecularDynamics
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.plumed import Plumed
import yaml


class DeepMDLammpsCalculator(LAMMPS):
    def __init__(self, model_files, label, command="lmp", tmp_dir="tmp", deepmd_opts=None, **kwargs):
        if not isinstance(model_files, list):
            model_files = [model_files]

        self.model_files = model_files
        self.label = label
        self.tmp_dir = tmp_dir
        self.deepmd_opts = deepmd_opts or []
        kwargs.setdefault("keep_tmp_files", True)
        kwargs.setdefault("files", model_files)
        kwargs.setdefault("tmp_dir", tmp_dir)
        kwargs.setdefault("label", label)

        super().__init__(command=command, **kwargs)

    def set_deepmd_inputs(self):
        model_str = " ".join(self.model_files)
        pair_style_line = f"pair_style deepmd {model_str}"
        if self.deepmd_opts:
            pair_style_line += " " + " ".join(self.deepmd_opts)

        self.parameters.update({
            "pair_style": pair_style_line,
            "pair_coeff": ["* *"],
        })

    def calculate(self, atoms=None, properties=None, system_changes=all_changes, set_atoms=True):
        self.set_deepmd_inputs()
        super().calculate(atoms, properties, system_changes)


class DeepMDLammpsCalculatorWithPlumed(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, plumed_input, **kwargs):
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.plumed_input = plumed_input
        self.plumed = Plumed(input=plumed_input)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        self.base_calc.calculate(atoms, properties, system_changes)
        self.plumed.atoms = atoms
        self.plumed.calculate(atoms)

        self.results = dict(self.base_calc.results)
        if "forces" in self.base_calc.results:
            self.results["forces"] += self.plumed.results.get("forces", 0.0)
        if "energy" in self.plumed.results:
            self.results["energy"] += self.plumed.results["energy"]


class DeepMDMD(MolecularDynamics):
    def __init__(self, atoms, model_files, timestep_fs, command, specorder,
                 output_prefix="deepmd", deepmd_opts=None, plumed_input=None, **kwargs):
        self.specorder = specorder
        self.model_files = model_files
        self.output_prefix = output_prefix
        self.command = command
        self.deepmd_opts = deepmd_opts or []
        self.plumed_input = plumed_input
        self.thermo_file = f"{output_prefix}_thermo.txt"
        self.traj_file = f"{output_prefix}_traj.xyz"
        self.nsteps = 0
        self.curr_atoms = atoms.copy()
        self.params = deepcopy(kwargs)
        self.params.setdefault("timestep", timestep_fs / 1000 / units.fs)
        self.initial_params = deepcopy(self.params)

        super().__init__(atoms, timestep_fs, trajectory=None)

    def run_until(self, std_tolerance, max_steps):
        now = datetime.now()
        label = now.strftime("%Y%m%d_%H%M%S")

        params = deepcopy(self.initial_params)
        dump_file = f"trj_{label}.bin"

        compute_uncertainty_cmds = [
            "unc all deepmd/std/atom",
            "MaxUnc all reduce max c_unc"
        ]
        dump_cmd = f"dump dmp all custom {params['dump_period']} tmp/{dump_file} id type x y z vx vy vz fx fy fz c_unc"
        thermo_cmd = f'thermoprint all print {params["dump_period"]} "$(step) $(temp) $(pe) $(pxx) $(pyy) $(pzz)" append tmp/{self.thermo_file}'

        params["compute"] = params.get("compute", []) + compute_uncertainty_cmds
        params["dump"] = [dump_cmd]
        params["fix"] = params.get("fix", []) + [thermo_cmd]
        params["run"] = (
            f"{max_steps} upto every {params['dump_period']} "
            f"\"if '$(c_MaxUnc) > {std_tolerance}' then 'write_restart restart_*.dat' quit\""
        )

        base_calc = DeepMDLammpsCalculator(
            model_files=self.model_files,
            label=label,
            command=self.command,
            specorder=self.specorder,
            deepmd_opts=self.deepmd_opts,
            **params
        )

        if self.plumed_input:
            calc = DeepMDLammpsCalculatorWithPlumed(base_calc, self.plumed_input)
        else:
            calc = base_calc

        calc.calculate(self.curr_atoms, set_atoms=True)

        trj = read(
            f"tmp/{dump_file}",
            format="lammps-dump-binary",
            colnames="id type x y z vx vy vz fx fy fz c_unc".split(),
            specorder=self.specorder,
            index=":",
        )
        self.nsteps += len(trj) - 1
        self.curr_atoms = trj[-1]
        self._backup_trajectory(trj)

    def _backup_trajectory(self, trj):
        if os.path.exists(self.traj_file):
            existing = read(self.traj_file, index=":")
            combined = existing + trj
        else:
            combined = trj

        write(self.traj_file, combined, format="extxyz")


def load_from_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    atoms_file = config["general"]["structure_file"]
    atoms = read(atoms_file)
    model_files = config["deepmd"]["model_files"]
    timestep = config["md_simulation"]["timestep_fs"]
    command = config["general"].get("lammps_cmd", "lmp")
    specorder = config["deepmd"]["specorder"]
    output_prefix = config["general"].get("output_prefix", "deepmd")
    dump_period = config["md_simulation"].get("dump_period", 10)
    std_tol = config["active_learning"]["f_max_dev"]
    nsteps = config["general"].get("md_steps", 1000)
    plumed_input = config.get("plumed", {}).get("input_file")
    deepmd_opts = config["deepmd"].get("deepmd_opts", [])

    return DeepMDMD(
        atoms=atoms,
        model_files=model_files,
        timestep_fs=timestep,
        command=command,
        specorder=specorder,
        output_prefix=output_prefix,
        dump_period=dump_period,
        deepmd_opts=deepmd_opts,
        plumed_input=plumed_input
    ), std_tol, nsteps

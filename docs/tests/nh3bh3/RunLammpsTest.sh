#!/bin/bash

module load LAMMPS/deepmd

for i in `seq 1 4`
do
dir="iter_00000${i}/02.dpmd"
echo "Running MD for $dir"

mkdir -p "$dir/MetaD"
cp lmp_run/MetaD/without_uwall/conf.lmp $dir/MetaD
cp lmp_run/MetaD/without_uwall/input.lammps $dir/MetaD
cp lmp_run/MetaD/without_uwall/plumed.dat $dir/MetaD

cd $dir/MetaD
ln -s ../../01.train/training_1/frozen_model_1.pb frozen_model_1.pb
ln -s ../../01.train/training_2/frozen_model_2.pb frozen_model_2.pb

CUDA_VISIBLE_DEVICES=1 mpirun -np 1 dpmd_lmp -i input.lammps

cd ../../../

done

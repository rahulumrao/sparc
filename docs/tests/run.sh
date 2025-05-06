#!/bin/bash

#export PLUMED_KERNEL=/home/rverma7/anaconda3/envs/PLUMED/lib/libplumedKernel.so
#export PYTHONPATH=/home/rverma7/anaconda3/envs/PLUMED/lib/plumed/python:$PYTHONPATH
export PLUMED_KERNEL=/home/rverma7/anaconda3/envs/BaseAsePlumed/lib/libplumedKernel.so
export PYTHONPATH="/home/rverma7/anaconda3/envs/BaseAsePlumed/lib/plumed/python:$PYTHONPATH"


python 2.py

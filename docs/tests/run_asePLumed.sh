#!/bin/bash
export KMP_WARNINGS=0
export KMP_AFFINITY=none
export TF_CPP_MIN_LOG_LEVEL=3

export VASP_PP_PATH=/home/prg/Softwares/VASP_gcc/POTCAR_FILES
export CUDA_VISIBLE_DEVICES=0
sparc -i input.yaml
#PYTHONUNBUFFERED=1 sparc -i input.yaml > output.log #2>&1

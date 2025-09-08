#!/bin/bash
#BSUB -n 16 
#BSUB -W 72:00 #hh-mm
#BSUB -q pfaendtner #_gpu #gpu
#BSUB -R "rusage[mem=40GB]"
##BSUB -R "select[rtx2080 ] | a10 | a100 | a30 | gtx1080 | p100]"
##BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J nh3bh3
#
source /usr/share/Modules/init/bash
#
module purge
#
module load VASP/vasp_cpu_gcc
conda activate /rs1/researchers/w/wjpfaend/rahul/MLP/sparc_CPU  
ulimit -s unlimited

export KMP_WARNINGS=0
export KMP_AFFINITY=none
export TF_CPP_MIN_LOG_LEVEL=3

export VASP_PP_PATH=/rs1/researchers/w/wjpfaend/codes/POTCAR_FILES

#
date

sparc -i input.yaml

date

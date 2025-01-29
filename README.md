## SPARC - Smart Potential with Atomistic Rare Events and Continious Learning

<!-- <p align="center">
  <img width="527" height="527" src="docs/SPARC.jpeg?raw=true">
</p> -->

 <!-- #src="https://github.com/rahulumrao/sparc/docs/SPARC.jpeg?raw=true"> -->

SPARC is a open-source python code for building meachine learned interatomic potentials on the fly.


## Software Requirements

* VASP package
* Python 3
* ASE (Atomic Simulation Environment)
* Plumed
* MPI library

## Installation

Create conda environment :

```bash
conda create -n $ENV_NAME python==$PYTHON_VERSION
conda activate $ENV_NAME

conda install conda-forge py-plumed
```
> **_NOTE:_** Remember to change the $ENV_NAME (environment name) according to your conda environment name.

Download repository and install package :
```bash
git clone https://github.com/rahulumrao/sparc.git
cd sparc
pip install .
```

## Usage
```bash
export VASP_PP_PATH=/path/to/vasp//POTCAR_FILES
sparc -i input.yaml
```




# SPARC MLIP Workflow Dataset

The dataset is archived on Zenodo which includes all necessary inputs, configuration, and corresponding output data.
Files are organized to re-run the workflow from scratch and inspect the results generated in this work.

### Contents

The Zenodo archive contains following directory tree:

```bash
.
├── Results                 # Output data generatedin this work
│   └── nh3bh3
│       ├── iter_000000
│       ├── iter_000001
│       ├── iter_000002
│       ├── iter_000003
│       ├── iter_000004
│       └── nohup.out
└── configs                 # Input structure and starting configs
    ├── INCAR
    ├── POSCAR
    ├── input.json
    ├── input.yaml
    ├── iter_000000
    └── plumed_dp.dat
```
- configs/: Contains input scripts and reference data to initialize the MLIP workflow.
- Results/: Output of the workflow run performed in this study.

## Fetching the Dataset from Zenodo

The dataset is hosted on Zenodo and can be downloaded using either a web browser or the command line.

### Download via browser

1. Navigate to the [Zenodo record](https://zenodo.org/records/18261342),
2. Download the compressed archive,
3. Extract the archive:

```bash
tar -xvf AmmoniaBorate.tar.gz 
```

## Download via command line

```bash
wget https://zenodo.org/records/18261342/files/AmmoniaBorate.tar.gz?download=1
tar -xvf AmmoniaBorate.tar.gz
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{sparc,
  author = {Verma, Rahul and Joshi, Nisarg and Pfaendtner, Jim},
  doi    = {https://doi.org/10.5281/zenodo.18261342},
  license = {MIT},
  month  = {jan},
  title  = {{SPARC}: An Automated Workflow Toolkit for Accelerated Active Learning of Reactive Machine Learning Interatomic Potentials},
  url    = {https://zenodo.org/records/18261342},
  year   = {2026}
}
```


#!/bin/bash
# This script installs Anaconda, ASE, PLUMED, and essential Python packages on a Linux system.
# Refer to official documentation for more details:
# PLUMED: https://www.plumed.org/doc-v2.8/user-doc/html/_installation.html#installingpython
# ASE: https://wiki.fysik.dtu.dk/ase/install.html#installation-from-source

# Make the script executable and run it: chmod +x install.sh && ./install.sh

LOGFILE="install.log"
exec > >(tee -a "$LOGFILE") 2>&1

# Define variables
ENV_NAME="ASE_PLUMED"
PYTHON_VERSION="3.10.13"
ANACONDA_VERSION="2024.10.1"
ANACONDA_SCRIPT="Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh"
ANACONDA_URL="https://repo.anaconda.com/archive/${ANACONDA_SCRIPT}"
ANACONDA_PATH="$HOME/anaconda3"
PLUMED_CHANNEL="conda-forge"
PYTHON_PACKAGES=("ase" "matplotlib" "notebook" "numpy" "pandas")

# Install Anaconda
install_anaconda() {
    echo -e "\n\n----------------- INSTALL ANACONDA -------------------\n\n"
    cd ~/
    wget $ANACONDA_URL
    if [ $? -ne 0 ]; then
        echo "Failed to download Anaconda. Please check your internet connection or URL."
        exit 1
    fi

    sha256sum $ANACONDA_SCRIPT
    bash $ANACONDA_SCRIPT
    rm $ANACONDA_SCRIPT
    $ANACONDA_PATH/bin/conda init
    eval "$($ANACONDA_PATH/bin/conda shell.bash hook)"
}

# Create and activate Conda environment
create_conda_environment() {
    echo -e "\n\n------------- CREATE CONDA ENVIRONMENT ---------------\n\n"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    conda activate $ENV_NAME
}

# Install PLUMED
install_plumed() {
    echo -e "\n\n------------------- INSTALL PLUMED -------------------\n\n"
    conda install -c $PLUMED_CHANNEL py-plumed -y
}

# Install Python packages
install_python_packages() {
    echo -e "\n\n----------------- INSTALL PYTHON PACKAGES -------------\n\n"
    for package in "${PYTHON_PACKAGES[@]}"; do
        pip install $package
    done
}

# Main script
if [ ! -d "$ANACONDA_PATH" ]; then
    install_anaconda
else
    echo "Anaconda is already installed."
    eval "$($ANACONDA_PATH/bin/conda shell.bash hook)"
fi

create_conda_environment
install_plumed
install_python_packages

echo -e "\n\n------------------ INSTALLATION COMPLETE -----------------\n\n"
echo -e "\n------------------ Written by: Rahul Verma ------------------\n"
echo -e "          Dept. of Chemical and BioMolecular Engineering         "
echo -e "                   NC State University                           "
echo -e "                 Email: rverma7@ncsu.edu                         "
echo -e "\n-------------------------------------------------------------\n"



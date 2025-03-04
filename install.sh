#!/bin/bash

ENV_NAME="sparc_env"
ENV_FILE="environment.yml"

echo -e "\033[92mChecking Conda installation...\033[0m"
if ! command -v conda &> /dev/null
then
    echo -e "\033[91mError: Conda is not installed. Please install Miniconda or Anaconda first.\033[0m"
    exit 1
fi

# Check if the environment exists
if conda env list | grep -q "$ENV_NAME"; then
    echo -e "\033[92mConda environment '$ENV_NAME' already exists. Skipping...\033[0m"
else
    echo -e "\033[92mCreating Conda environment from $ENV_FILE...\033[0m"
    conda env create -f "$ENV_FILE"
fi


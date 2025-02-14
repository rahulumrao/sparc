from setuptools import setup, find_packages
import subprocess
import sys

def build_docs():
    """Build the documentation."""
    try:
        subprocess.run(['make', 'html'], cwd='docs', check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        sys.exit(1)


def has_cuda():
    """Check if CUDA is available on the system"""
    try:
        # Try to run nvidia-smi to check for CUDA
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Determine which deepmd-kit version to install
if has_cuda():
    # For systems with CUDA, install GPU version
    deepmd_requirement = "deepmd-kit[gpu,cu12,lmp]==2.2.10"  # Using CUDA 12 as default
else:
    print("\033[93mWarning: No GPU detected. Installing CPU-only version of deepmd-kit.\033[0m")
    # For systems without CUDA, install CPU version
    deepmd_requirement = "deepmd-kit[cpu,lmp]==2.2.10"

setup(
    name="sparc",
    version="0.1",
    packages=find_packages(include=["src"]),
    install_requires=[
        "ase",
        "numpy",
        "pandas",
        "scipy",
        "dpdata",
        deepmd_requirement,  
        "pytest",
    ],
    author="Rahul Verma",
    author_email="rverma7@ncsu.edu",
    description="A package for DeepMD training and on-the-fly learning with VASP",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.0",
    keywords="",
    url="",
    docs_url="",
    entry_points={
        "console_scripts": [
            "sparc = src.sparc:main",
        ],
    },
    extras_require={
        "docs": ["sphinx", "sphinx-rtd-theme", "sphinx-autodoc-typehints"],
    },
    cmdclass={
        "build_docs": build_docs,
    },
)

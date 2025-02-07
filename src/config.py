# New file for centralized configuration management
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class MDConfig:
    """Configuration for Molecular Dynamics simulation parameters.
    
    Attributes:
        temperature: Target temperature in Kelvin
        timestep_fs: Timestep in femtoseconds
        tdamp: Damping time for Nose-Hoover thermostat in femtoseconds
        thermostat: Type of thermostat to use (default: "Nose")
        use_plumed: Whether to use PLUMED for enhanced sampling
        log_frequency: How often to write logs
    """
    temperature: float
    timestep_fs: float
    tdamp: Optional[float] = None
    thermostat: str = "Nose"
    use_plumed: bool = False
    log_frequency: int = 100

@dataclass
class DFTConfig:
    """Configuration for DFT calculations.
    
    Attributes:
        calculator: Name of DFT calculator (e.g., "VASP")
        exe_path: Absolute path to executable
        exe_name: Name of executable
        precision: Calculation precision
        encut: Plane-wave energy cutoff
        ediff: Convergence criteria
        kgamma: Whether to use gamma-point only k-sampling
    """
    calculator: str
    exe_path: str
    exe_name: str
    precision: str = "Normal"
    encut: float = 400.0
    ediff: float = 1e-6
    kgamma: bool = True

@dataclass
class DeepMDConfig:
    """Configuration for DeepMD-kit training and simulation.
    
    Attributes:
        training: Whether to perform training
        num_models: Number of models to train
        data_dir: Directory for training data
        train_dir: Directory for model training
        md_steps: Number of MD steps for DPMD simulation
        log_frequency: How often to write logs
    """
    training: bool = False
    num_models: int = 4
    data_dir: str = "DeePMD_training/00.data"
    train_dir: str = "DeePMD_training/01.train"
    md_steps: int = 1000
    log_frequency: int = 100

@dataclass
class SPARCConfig:
    """Main configuration class for SPARC.
    
    This class brings together all configuration components and provides
    validation methods.
    
    Attributes:
        md: Molecular dynamics configuration
        dft: DFT calculation configuration
        deepmd: DeepMD-kit configuration
        structure_file: Input structure file path
        output_dir: Directory for output files
    """
    md: MDConfig
    dft: DFTConfig
    deepmd: DeepMDConfig
    structure_file: str = "POSCAR"
    output_dir: str = "output"

    def validate(self) -> None:
        """Validate the configuration.
        
        Raises:
            ValueError: If any configuration parameters are invalid
        """
        if self.md.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.md.timestep_fs <= 0:
            raise ValueError("Timestep must be positive")
        if self.md.thermostat == "Nose" and self.md.tdamp is None:
            raise ValueError("tdamp is required for Nose thermostat") 
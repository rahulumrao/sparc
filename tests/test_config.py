"""
Tests for v0.2 configuration system (dataclass-based SparcConfig).

Covers:
- YAML parsing with SparcConfig.from_yaml()
- Top-level FineTuneConfig
- MLIPSetupConfig.restart flag
- Temperature ramping (temp_end)
- Validation errors for invalid inputs
- Distance metrics parsing
- QE and Gaussian engine recognition
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from sparc.src.utils.read_input import (
    SparcConfig,
    FineTuneConfig,
    MLIPSetupConfig,
    AIMDSetupConfig,
    ThermostatConfig,
    PlumedConfig,
    MLIPPlumedConfig,
    UmbrellaSamplingConfig,
    ModelDeviationConfig,
    DistanceMetric,
    DFTCalculatorConfig,
    GeneralConfig,
    OutputConfig,
    ConfigurationError,
    ValidationError,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def minimal_yaml(tmp_path):
    """Write a minimal valid YAML config and return its path."""
    struct = tmp_path / "water.xyz"
    struct.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.8 0.6 0.0\nH -0.8 0.6 0.0\n")
    template = tmp_path / "template.inp"
    template.write_text("! PBE def2-SVP\n")

    yaml_path = tmp_path / "input.yaml"
    yaml_path.write_text(f"""
general:
  structure_file: "{struct}"
dft_calculator:
  engine: "ORCA"
  template_file: "{template}"
aimd_setup:
  ensemble: "NVT"
  thermostat:
    type: "Nose"
    tdamp: 2.0
  temperature: 300
  steps: 10
mlip_setup:
  training: False
  num_models: 2
  md_steps: 100
  restart: True
output:
  log_file: "test.log"
""")
    return yaml_path


@pytest.fixture
def finetune_yaml(tmp_path):
    """YAML with top-level finetune section."""
    struct = tmp_path / "system.xyz"
    struct.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.8 0.6 0.0\nH -0.8 0.6 0.0\n")
    template = tmp_path / "template.inp"
    template.write_text("! PBE\n")

    yaml_path = tmp_path / "input.yaml"
    yaml_path.write_text(f"""
general:
  structure_file: "{struct}"
dft_calculator:
  engine: "xTB"
  template_file: "{template}"
aimd_setup:
  ensemble: "NVT"
  thermostat:
    type: "Langevin"
    friction: 0.15
  temperature: 300
  steps: 0
mlip_setup:
  training: False
  num_models: 2
  md_steps: 100
finetune:
  enabled: True
  model_type: "deepmd"
  pretrained_model: "DPA-3.2-5M.pt"
  model_branch: "Omat24"
  input_file: "input_finetune.json"
  learning_rate: 0.001
active_learning: True
iteration: 5
model_dev:
  f_min_dev: 0.1
  f_max_dev: 0.3
output:
  log_file: "test.log"
""")
    return yaml_path


# ============================================================
# FineTuneConfig (top-level)
# ============================================================

class TestFineTuneConfig:
    """Test top-level finetune configuration."""

    def test_defaults(self):
        cfg = FineTuneConfig()
        assert cfg.enabled is False
        assert cfg.model_type == "deepmd"
        assert cfg.pretrained_model == "DPA3.pt"
        assert cfg.model_branch == "Omat24"
        assert cfg.input_file is None
        assert cfg.learning_rate == 0.001
        assert cfg.batch_size == 4
        assert cfg.device == "cpu"

    def test_mace_config(self):
        cfg = FineTuneConfig(
            enabled=True,
            model_type="mace",
            pretrained_model="medium",
            batch_size=8,
            device="cuda",
            num_epochs=200,
        )
        assert cfg.model_type == "mace"
        assert cfg.pretrained_model == "medium"
        assert cfg.batch_size == 8
        assert cfg.device == "cuda"
        assert cfg.num_epochs == 200

    def test_deepmd_with_branch(self):
        cfg = FineTuneConfig(
            enabled=True,
            model_type="deepmd",
            pretrained_model="DPA-3.2-5M.pt",
            model_branch="Organic_Reactions",
            input_file="input_finetune.json",
        )
        assert cfg.model_branch == "Organic_Reactions"
        assert cfg.input_file == "input_finetune.json"

    def test_from_yaml_top_level(self, finetune_yaml):
        """Finetune is parsed as a top-level section, not nested in mlip_setup."""
        os.chdir(finetune_yaml.parent)
        config = SparcConfig.from_yaml(str(finetune_yaml))

        assert config.finetune.enabled is True
        assert config.finetune.model_type == "deepmd"
        assert config.finetune.pretrained_model == "DPA-3.2-5M.pt"
        assert config.finetune.model_branch == "Omat24"
        assert config.finetune.input_file == "input_finetune.json"
        assert config.finetune.learning_rate == 0.001


# ============================================================
# MLIPSetupConfig — restart flag
# ============================================================

class TestMLIPRestart:
    """Test ML-MD restart flag in config."""

    def test_restart_default_false(self):
        cfg = MLIPSetupConfig()
        assert cfg.restart is False

    def test_restart_from_yaml(self, minimal_yaml):
        os.chdir(minimal_yaml.parent)
        config = SparcConfig.from_yaml(str(minimal_yaml))
        assert config.mlip_setup.restart is True


# ============================================================
# Temperature ramping
# ============================================================

class TestTemperatureRamping:
    """Test temp_start / temp_end fields."""

    def test_aimd_temp_end(self):
        cfg = AIMDSetupConfig(temperature=300, temp_end=600)
        assert cfg.temp_end == 600

    def test_mlip_temp_end(self):
        cfg = MLIPSetupConfig(temperature=300, temp_end=500)
        assert cfg.temp_end == 500

    def test_temp_end_default_none(self):
        cfg = AIMDSetupConfig(temperature=300)
        assert cfg.temp_end is None


# ============================================================
# DFT Engine recognition (QE, Gaussian)
# ============================================================

class TestDFTEngines:
    """Test that new DFT engines are accepted in config."""

    def test_qe_engine(self, tmp_path):
        template = tmp_path / "qe.in"
        template.write_text("&CONTROL\n/\n")
        cfg = DFTCalculatorConfig(engine="QE", template_file=str(template))
        assert cfg.engine == "QE"

    def test_gaussian_engine(self, tmp_path):
        template = tmp_path / "gaussian.gjf"
        template.write_text("#p HF/STO-3G\n")
        cfg = DFTCalculatorConfig(engine="Gaussian", template_file=str(template))
        assert cfg.engine == "Gaussian"


# ============================================================
# Validation errors
# ============================================================

class TestValidation:
    """Test that invalid configs raise appropriate errors."""

    def test_num_models_too_low(self):
        with pytest.raises(ValidationError):
            MLIPSetupConfig(num_models=1)

    def test_negative_timestep(self):
        with pytest.raises(ValidationError):
            AIMDSetupConfig(timestep_fs=-0.5)

    def test_timestep_too_large(self):
        with pytest.raises(ValidationError):
            AIMDSetupConfig(timestep_fs=10.0)

    def test_negative_temperature(self):
        with pytest.raises(ValidationError):
            AIMDSetupConfig(temperature=-100)

    def test_model_dev_min_ge_max(self):
        with pytest.raises(ValidationError):
            ModelDeviationConfig(f_min_dev=0.5, f_max_dev=0.3)

    def test_distance_metric_wrong_pair(self):
        with pytest.raises(ValidationError):
            DistanceMetric(pair=[0, 1, 2], min_distance=1.0, max_distance=3.0)

    def test_distance_metric_min_gt_max(self):
        with pytest.raises(ValidationError):
            DistanceMetric(pair=[0, 1], min_distance=5.0, max_distance=3.0)

    def test_iteration_zero(self, tmp_path):
        struct = tmp_path / "s.xyz"
        struct.write_text("1\nH\nH 0 0 0\n")
        template = tmp_path / "t.inp"
        template.write_text("!\n")
        with pytest.raises(ValidationError):
            SparcConfig(
                general=GeneralConfig(structure_file=str(struct)),
                dft_calculator=DFTCalculatorConfig(engine="ORCA", template_file=str(template)),
                iteration=0,
            )

    def test_npt_missing_params(self):
        with pytest.raises(ValidationError):
            AIMDSetupConfig(ensemble="NPT", temperature=300)

    def test_missing_yaml_file(self):
        with pytest.raises(ConfigurationError):
            SparcConfig.from_yaml("nonexistent.yaml")


# ============================================================
# Full YAML round-trip
# ============================================================

class TestYAMLRoundTrip:
    """Test config load → dict → save → reload."""

    def test_to_dict(self, minimal_yaml):
        os.chdir(minimal_yaml.parent)
        config = SparcConfig.from_yaml(str(minimal_yaml))
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "general" in d
        assert "dft_calculator" in d
        assert "aimd_setup" in d
        assert "mlip_setup" in d
        assert "finetune" in d
        assert "output" in d

    def test_distance_metrics_parsing(self, tmp_path):
        struct = tmp_path / "sys.xyz"
        struct.write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
        template = tmp_path / "t.inp"
        template.write_text("! HF\n")
        yaml_path = tmp_path / "input.yaml"
        yaml_path.write_text(f"""
general:
  structure_file: "{struct}"
dft_calculator:
  engine: "ORCA"
  template_file: "{template}"
distance_metrics:
  - pair: [0, 1]
    min_distance: 0.5
    max_distance: 2.0
output:
  log_file: "t.log"
""")
        os.chdir(tmp_path)
        config = SparcConfig.from_yaml(str(yaml_path))
        assert len(config.distance_metrics) == 1
        assert config.distance_metrics[0].pair == [0, 1]
        assert config.distance_metrics[0].min_distance == 0.5

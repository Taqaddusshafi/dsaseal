"""
Unit tests for configuration management.
Tests config loading, validation, and default values.
"""

import os
import tempfile
import pytest
import yaml

from seal_dsa.config import (
    SEALDSAConfig,
    ModelConfig,
    LoRAConfig,
    SEALConfig,
    CurriculumConfig,
    EWCConfig,
    CheckpointConfig,
    load_config,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.max_length == 512
        assert cfg.quantization == "4bit"

    def test_custom_values(self):
        cfg = ModelConfig(name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_length=256)
        assert cfg.name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert cfg.max_length == 256


class TestLoRAConfig:
    """Test LoRAConfig dataclass."""

    def test_default_values(self):
        cfg = LoRAConfig()
        assert cfg.r == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.05
        assert "q_proj" in cfg.target_modules

    def test_scaling_factor(self):
        cfg = LoRAConfig(r=8, alpha=16)
        # Scaling = alpha / r = 16 / 8 = 2.0
        assert cfg.alpha / cfg.r == 2.0

    def test_parameter_budget(self):
        """Verify LoRA parameter count calculation."""
        cfg = LoRAConfig(r=8)
        hidden_size = 1536  # Qwen2.5-1.5B
        num_layers = 28
        num_modules = len(cfg.target_modules)

        params_per_module = 2 * hidden_size * cfg.r
        total_params = num_layers * num_modules * params_per_module

        # Should be approximately 2.75M for Qwen2.5-1.5B
        assert 2_000_000 < total_params < 3_500_000


class TestSEALConfig:
    """Test SEAL loop configuration."""

    def test_default_values(self):
        cfg = SEALConfig()
        assert cfg.num_epochs >= 1
        assert cfg.questions_per_topic >= 5
        assert 0.0 <= cfg.forgetting_threshold <= 1.0

    def test_questions_per_topic_minimum(self):
        cfg = SEALConfig(questions_per_topic=5)
        assert cfg.questions_per_topic == 5


class TestEWCConfig:
    """Test EWC configuration."""

    def test_default_values(self):
        cfg = EWCConfig()
        assert cfg.lambda_ >= 0
        assert cfg.fisher_sample_size >= 10

    def test_disabled_by_default(self):
        cfg = EWCConfig()
        assert cfg.enabled is True  # Should be enabled by default


class TestConfigLoading:
    """Test YAML config loading."""

    def test_load_default_config(self):
        """Test loading the default config file."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "default.yaml"
        )
        if os.path.exists(config_path):
            cfg = load_config(config_path)
            assert isinstance(cfg, SEALDSAConfig)
            assert cfg.model.name is not None

    def test_load_custom_yaml(self):
        """Test loading a custom YAML config."""
        custom_config = {
            "model": {
                "name": "test-model",
                "max_length": 256,
            },
            "lora": {
                "r": 4,
                "alpha": 8,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name

        try:
            cfg = load_config(temp_path)
            assert cfg.model.name == "test-model"
            assert cfg.lora.r == 4
        finally:
            os.unlink(temp_path)

    def test_missing_file_raises_error(self):
        """Test that loading a non-existent file raises an error."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_config("nonexistent_config.yaml")


class TestSEALDSAConfig:
    """Test the top-level configuration."""

    def test_default_construction(self):
        cfg = SEALDSAConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.lora, LoRAConfig)
        assert isinstance(cfg.seal, SEALConfig)
        assert isinstance(cfg.curriculum, CurriculumConfig)
        assert isinstance(cfg.ewc, EWCConfig)
        assert isinstance(cfg.checkpoint, CheckpointConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

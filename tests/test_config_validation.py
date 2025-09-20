import pytest

from charlm.config import TrainingConfig


def test_valid_configuration():
    cfg = TrainingConfig.from_mapping(
        {
            "model": {
                "d_model": 64,
                "n_layer": 2,
                "n_head": 4,
                "n_kv_head": 2,
                "seq_len": 32,
            },
            "batch_size_tokens": 1024,
            "micro_batch_size": 4,
            "scheduler": {"max_steps": 10},
        }
    )
    assert cfg.grad_accum_steps == 8


def test_invalid_head_relationship():
    with pytest.raises(ValueError):
        TrainingConfig.from_mapping(
            {
                "model": {
                    "d_model": 64,
                    "n_layer": 2,
                    "n_head": 3,
                    "n_kv_head": 2,
                    "seq_len": 32,
                },
                "batch_size_tokens": 1024,
                "micro_batch_size": 4,
                "scheduler": {"max_steps": 10},
            }
        )


def test_invalid_batch_accumulation():
    with pytest.raises(ValueError):
        TrainingConfig.from_mapping(
            {
                "model": {
                    "d_model": 64,
                    "n_layer": 2,
                    "n_head": 4,
                    "n_kv_head": 2,
                    "seq_len": 30,
                },
                "batch_size_tokens": 1000,
                "micro_batch_size": 4,
                "scheduler": {"max_steps": 10},
            }
        )

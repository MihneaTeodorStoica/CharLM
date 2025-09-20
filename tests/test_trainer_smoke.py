import numpy as np

from charlm.config import TrainingConfig
from charlm.trainer import Trainer


def test_trainer_smoke(tmp_path):
    data = np.arange(256, dtype=np.uint8)
    train_bin = tmp_path / "train.bin"
    train_bin.write_bytes(data.tobytes())
    train_idx = tmp_path / "train.idx"
    np.array([0, len(data)], dtype=np.int64).tofile(train_idx)

    cfg = TrainingConfig()
    cfg.model.d_model = 64
    cfg.model.n_layer = 1
    cfg.model.n_head = 4
    cfg.model.n_kv_head = 2
    cfg.model.seq_len = 32
    cfg.batch_size_tokens = 256
    cfg.micro_batch_size = 4
    cfg.data_workers = 0
    cfg.scheduler.max_steps = 4
    cfg.scheduler.warmup_steps = 1
    cfg.eval_interval = 0
    cfg.checkpoint.out_dir = str(tmp_path / "ckpt")
    cfg.checkpoint.save_interval = 2
    cfg.logging.log_interval = 10
    cfg.validate()

    trainer = Trainer(
        cfg,
        train_bin=str(train_bin),
        train_idx=str(train_idx),
        val_bin=None,
        val_idx=None,
        device="cpu",
    )
    trainer.fit()

    assert any(tmp_path.joinpath("ckpt").glob("step-*.pt"))

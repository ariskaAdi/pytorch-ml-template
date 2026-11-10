from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def make_cfg(tmp_path):
    """Build the real Hydra config with small, fast test settings."""

    def _make(*overrides: str, name: str = "run"):
        run_dir = (tmp_path / name).as_posix()
        defaults = [
            f"run_dir='{run_dir}'",
            "train.max_steps=20",
            "train.batch_size=64",
            "train.log_every=1000",
            "train.eval_every=1000",
            "train.save_every=1000",
            "train.warmup_steps=5",
            "data.num_samples=1024",
        ]
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            return compose(config_name="train", overrides=[*defaults, *overrides])

    return _make

import json
from pathlib import Path

import pytest
import torch

from mltemplate.tasks import build_task
from mltemplate.training.checkpoint import load_checkpoint
from mltemplate.training.distributed import setup_distributed
from mltemplate.training.trainer import Trainer
from mltemplate.utils.seed import seed_everything
from mltemplate.utils.tracking import Tracker
from scripts.train import run


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_training_learns_and_writes_outputs(make_cfg, precision):
    cfg = make_cfg(f"train.precision={precision}", "train.max_steps=60", "train.eval_every=30")
    metrics = run(cfg)

    assert metrics["val/accuracy"] > 0.7  # Bayes-optimal is ~0.84
    run_dir = cfg.run_dir
    for f in ["config.yaml", "run_info.json", "metrics.jsonl"]:
        assert (Path(run_dir) / f).exists(), f
    assert (Path(run_dir) / "checkpoints" / "last.pt").exists()
    assert (Path(run_dir) / "checkpoints" / "best.pt").exists()

    info = json.loads((Path(run_dir) / "run_info.json").read_text())
    assert info[-1]["env"]["torch"] == torch.__version__


def test_grad_accumulation_matches_bigger_batch(make_cfg):
    """2 x 32 accumulated must give the same update as 1 x 64 (no dropout, fp32)."""
    results = []
    for overrides in (
        ["train.batch_size=64", "train.grad_accum_steps=1"],
        ["train.batch_size=32", "train.grad_accum_steps=2"],
    ):
        cfg = make_cfg("train.precision=fp32", *overrides, name="_".join(overrides))
        seed_everything(cfg.seed)
        task = build_task(cfg)
        trainer = Trainer(cfg, task, setup_distributed(), Tracker(cfg, cfg.run_dir, False))
        # Feed the exact same 64 samples to both setups.
        x, y = task.train_dataset.x[:64], task.train_dataset.y[:64]
        mb = 64 // cfg.train.grad_accum_steps
        batches = iter([{"x": x[i : i + mb], "y": y[i : i + mb]} for i in range(0, 64, mb)])
        trainer.train_step(batches)
        results.append([p.detach().clone() for p in task.model.parameters()])

    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_resume_is_exact(make_cfg):
    """Stop at step 10, restart, finish at 20 == one uninterrupted 20-step run.

    Uses dropout and crosses an epoch boundary mid-run, so RNG, data order and
    optimizer/scheduler state must all be restored correctly.
    """
    common = [
        "train.precision=fp32",
        "model.dropout=0.1",
        "train.batch_size=128",
        "train.grad_accum_steps=2",
    ]  # 7 batches/epoch -> step 10 is mid-epoch

    # A: uninterrupted.
    cfg_a = make_cfg(*common, name="a")
    run(cfg_a)
    final_a = load_checkpoint(f"{cfg_a.run_dir}/checkpoints/last.pt")

    # B: 10 steps, "crash", then a fresh process-like restart that resumes.
    cfg_b = make_cfg(*common, name="b")
    seed_everything(cfg_b.seed)
    trainer = Trainer(
        cfg_b, build_task(cfg_b), setup_distributed(), Tracker(cfg_b, cfg_b.run_dir, False)
    )
    batches = trainer._batches()
    for _ in range(10):
        trainer.train_step(batches)
    trainer.save("last.pt")
    del trainer, batches

    run(cfg_b)  # resume=true by default
    final_b = load_checkpoint(f"{cfg_b.run_dir}/checkpoints/last.pt")

    assert final_b["step"] == final_a["step"] == 20
    for k in final_a["model"]:
        assert torch.equal(final_a["model"][k], final_b["model"][k]), k

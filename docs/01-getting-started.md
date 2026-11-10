# 1. Getting started

## Install

PyTorch has different builds for CPU, CUDA and ROCm, so install it first, then the template.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1

# CPU only (laptop, CI):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU: pick the command for your CUDA version at https://pytorch.org/get-started

pip install -e ".[dev,serve]"        # add ",tracking" for Weights & Biases
pre-commit install                   # run ruff on every commit
```

With [uv](https://docs.astral.sh/uv/) the same thing is faster:

```bash
uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,serve]"
```

Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `dev` | pytest, ruff, pre-commit, httpx | tests and linting |
| `serve` | fastapi, uvicorn | the inference API |
| `tracking` | wandb | `tracking.backend=wandb` |

## Check the install

```bash
pytest -m "not slow"     # fast tests, ~20 s on a laptop CPU
pytest                   # also the 2-process DDP test
```

## First training run

```bash
python scripts/train.py experiment_name=first_run
```

The example task classifies synthetic Gaussian clusters with a small MLP. It runs on CPU in
a few seconds. Expected output (numbers vary slightly):

```
step     10 | train/loss 1.354 | train/accuracy 0.5156 | train/grad_norm 0.2846 | perf/samples_per_sec 3038 | train/lr 0.00066 | train/epoch 0
step     20 | train/loss 1.227 | train/accuracy 0.7031 | train/grad_norm 0.3177 | perf/samples_per_sec 2867 | train/lr 0.00126 | train/epoch 0
...
step    100 | val/loss 0.4329 | val/accuracy 0.839
...
step    500 | val/loss 0.4258 | val/accuracy 0.8288
```

The clusters overlap on purpose: the best possible accuracy is about 84%, so the curves
behave like a real run (including a little overfitting near the end).

## What a run produces

Everything for one experiment goes to `run_dir` (default `outputs/<experiment_name>/`):

```
outputs/first_run/
├── config.yaml          # fully resolved config used for this run
├── run_info.json        # git commit, command, environment (one entry per launch)
├── metrics.jsonl        # every logged metric, one JSON object per line
└── checkpoints/
    ├── last.pt          # latest state, used for resume
    └── best.pt          # lowest validation loss so far
```

Running the same command again **resumes** from `last.pt` instead of starting over
(see [04](04-checkpoint-and-resume.md)). Use a new `experiment_name` for a fresh run.

## Next steps

- Evaluate: `python scripts/evaluate.py --checkpoint outputs/first_run/checkpoints/best.pt`
- Change settings: [02 Configuration](02-configuration.md)
- Understand the loop: [03 Training flow](03-training-flow.md)

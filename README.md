# pytorch-ml-template

A reusable PyTorch training template. Every project in this portfolio (GPT from scratch,
LoRA summarizer, road-damage detector) starts from it, so the boring-but-critical parts
are built and tested once.

| Feature | Where |
|---|---|
| Hydra config with command-line overrides | [configs/](configs/), [docs/02](docs/02-configuration.md) |
| Generic trainer: bf16/fp16 AMP, grad accumulation, grad clipping, warmup + cosine LR, `torch.compile` | [trainer.py](src/mltemplate/training/trainer.py), [docs/03](docs/03-training-flow.md) |
| Exact checkpoint/resume (model, optimizer, scheduler, RNG, data position) | [checkpoint.py](src/mltemplate/training/checkpoint.py), [docs/04](docs/04-checkpoint-and-resume.md) |
| W&B or local tracking + reproducibility record (config, git commit, environment) | [tracking.py](src/mltemplate/utils/tracking.py), [docs/05](docs/05-tracking-and-reproducibility.md) |
| DDP via `torchrun` (NCCL on GPU, gloo on CPU) | [distributed.py](src/mltemplate/training/distributed.py), [docs/06](docs/06-distributed-training.md) |
| Tests incl. overfit-one-batch, exact-resume and 2-process DDP; ruff; GitHub Actions | [tests/](tests/), [docs/07](docs/07-testing-and-ci.md) |
| Docker images (train + slim serve) and RunPod setup | [docker/](docker/), [docs/08](docs/08-docker-and-runpod.md) |
| FastAPI inference service | [app/serve.py](app/serve.py), [docs/09](docs/09-serving.md) |

## Quick start

```bash
# 1. Install (pick the PyTorch build for your machine first: https://pytorch.org/get-started)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,serve]"

# 2. Run the tests
pytest -m "not slow"

# 3. Train the example model (synthetic data, runs on CPU in seconds)
python scripts/train.py experiment_name=first_run

# 4. Evaluate and serve it
python scripts/evaluate.py --checkpoint outputs/first_run/checkpoints/best.pt
CHECKPOINT_PATH=outputs/first_run/checkpoints/best.pt uvicorn app.serve:app --port 8000
```

## Start a new project from this template

1. On GitHub: **Use this template → Create a new repository** (e.g. `indo-gpt-from-scratch`), then clone it.
2. Check the template works before changing anything: `pip install -e ".[dev]" && pytest`.
3. Add your project-specific parts. Everything else (trainer, checkpointing, DDP, tracking,
   Docker, CI) is reused as is:

   | Add | Where |
   |---|---|
   | Model | `src/mltemplate/models/` |
   | Dataset | `src/mltemplate/data/` |
   | Task: model + data + `loss_fn(model, batch) -> (loss, metrics)` | `src/mltemplate/tasks/` + one line in `tasks/__init__.py` |
   | Configs | `configs/model/`, `configs/data/` |
   | Tests: shape, overfit-one-batch, model-specific checks | `tests/` |

4. Train with the new parts:

   ```bash
   python scripts/train.py task=language_model model=gpt_small data=indo_wiki
   ```

Full guide with a GPT example and notes for LoRA and YOLO projects:
[docs/10-new-project.md](docs/10-new-project.md).

## Project layout

```
configs/                # Hydra configs: train.yaml + model/ + data/
src/mltemplate/
├── tasks/              # model + data + loss for one project (the part you replace)
├── models/             # architectures
├── data/               # datasets
├── training/           # trainer, checkpointing, schedulers, DDP helpers
├── evaluation/         # metrics
├── inference/          # load a checkpoint and predict
└── utils/              # seeding, tracking, reproducibility
scripts/                # train.py, evaluate.py, spawn_ddp.py (Windows DDP)
app/                    # FastAPI service
tests/                  # pytest
docker/                 # Dockerfile.train, Dockerfile.serve
docs/                   # one document per workflow
.github/workflows/      # CI
```

## Documentation

1. [Getting started](docs/01-getting-started.md): install, first run, what a run produces
2. [Configuration](docs/02-configuration.md): Hydra config tree, overrides, sweeps
3. [Training flow](docs/03-training-flow.md): what happens from `train.py` to the last step
4. [Checkpoint & resume](docs/04-checkpoint-and-resume.md): surviving a stopped pod
5. [Tracking & reproducibility](docs/05-tracking-and-reproducibility.md): W&B, metrics, run records
6. [Distributed training](docs/06-distributed-training.md): DDP with `torchrun`
7. [Testing & CI](docs/07-testing-and-ci.md): what each test proves
8. [Docker & RunPod](docs/08-docker-and-runpod.md): images, network volumes, pod workflow
9. [Serving](docs/09-serving.md): inference API
10. [Starting a new project](docs/10-new-project.md): turn the template into GPT / LoRA / detection

## Platform notes

- Tested locally on Windows 11 (CPU, PyTorch 2.15) and designed for Linux GPU pods (RunPod).
- On Windows, `torchrun` fails because PyTorch's Windows build lacks libuv; use
  `python scripts/spawn_ddp.py --nproc 2 ...` for local DDP checks
  (see [docs/06](docs/06-distributed-training.md#windows)).

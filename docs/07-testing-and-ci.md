# 7. Testing and CI

```bash
pytest                    # everything (~30 s on a laptop CPU)
pytest -m "not slow"      # skip the multi-process DDP test
pytest tests/test_overfit.py -v
ruff check . && ruff format --check .
```

## What each test proves

| Test | Proves | Catches |
|---|---|---|
| `test_model.py::test_forward_shape` | Output shape is right | Wrong reshapes / heads |
| `test_model.py::test_backward_reaches_all_params` | Every parameter gets a gradient | Detached tensors, unused layers |
| `test_overfit.py::test_overfit_single_batch` | Model + loss can memorize one batch (loss < 0.01) | Wrong loss, broken graph, bad init. **Run this before any GPU spend.** |
| `test_data.py` | Split sizes, dtypes, determinism, batch shapes | Data pipeline bugs |
| `test_scheduler.py` | Warmup is linear, cosine ends at `min_lr_ratio` | Off-by-one LR schedules |
| `test_trainer.py::test_training_learns_and_writes_outputs` | End-to-end run in fp32 and bf16 learns and writes all outputs | Integration bugs, AMP problems |
| `test_trainer.py::test_grad_accumulation_matches_bigger_batch` | 2 × 32 accumulated == 1 × 64 | Wrong loss scaling in accumulation |
| `test_trainer.py::test_resume_is_exact` | Stop at 10 + resume == uninterrupted 20 steps, bit-identical (with dropout, mid-epoch) | Missing optimizer/scheduler/RNG/data-position state |
| `test_ddp.py` (slow) | Real 2-process DDP (gloo) trains and saves | Distributed setup bugs |
| `test_serve.py` | API returns valid probabilities and rejects bad input | Serving regressions |

The resume test was checked by deliberately breaking the RNG restore and the batch skip:
both changes make it fail.

## Tests use the real config

`tests/conftest.py` builds the config with Hydra's `compose` API from the real
`configs/` folder, plus small overrides (few steps, small data, temp `run_dir`). If a
config change breaks training, the tests notice.

## Writing tests for a new project

Keep the same pattern for each new model:

- shape test for the forward pass
- **overfit-one-batch** test with the real loss
- one model-specific invariant, e.g. for GPT: changing token *t+1* must not change the
  output at position *t* (causal mask)
- a short end-to-end `run(cfg)` with tiny settings

## CI

[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs on every push and pull request:

1. Install the CPU PyTorch wheel (small, fast) and the project with `dev,serve` extras
2. `ruff check .` and `ruff format --check .`
3. `pytest -q` (including the DDP test)

## Pre-commit

```bash
pre-commit install          # once
pre-commit run --all-files  # manually
```

Hooks: ruff (lint + fix), ruff-format, trailing whitespace, end-of-file, YAML check, and a
1 MB file-size limit so checkpoints and datasets never end up in git.

# 5. Tracking and reproducibility

## Experiment tracking

Code: [tracking.py](../src/mltemplate/utils/tracking.py).

Every metric is written to `run_dir/metrics.jsonl` and printed to the console. With
`tracking.backend=wandb` it also goes to Weights & Biases.

```bash
pip install -e ".[tracking]"
wandb login                                  # once per machine (or set WANDB_API_KEY)
python scripts/train.py tracking.backend=wandb tracking.project=indo-gpt experiment_name=run1
```

- The W&B run id is the `experiment_name`, with `resume="allow"`. A resumed training run
  continues the **same** W&B chart instead of creating a new one.
- On RunPod, set `WANDB_API_KEY` as a pod environment variable (or RunPod secret), then
  watch runs from your phone.

### Reading `metrics.jsonl` without W&B

```python
import pandas as pd

df = pd.read_json("outputs/first_run/metrics.jsonl", lines=True)
df.dropna(subset=["val/loss"]).plot(x="step", y="val/loss")
```

## Reproducibility record

Code: [repro.py](../src/mltemplate/utils/repro.py). At the start of every launch rank 0
writes:

- `config.yaml`: the fully resolved config.
- `run_info.json`: a list with one entry per launch (resumes add entries):

```json
{
  "started_at": "2026-11-10T07:41:55+00:00",
  "command": "scripts/train.py experiment_name=demo",
  "world_size": 1,
  "git": {"commit": "3f2a9c1...", "dirty": false},
  "env": {"python": "3.12.10", "torch": "2.5.1", "cuda_available": true,
          "cuda": "12.4", "gpus": ["NVIDIA RTX A6000"]}
}
```

If the working tree has uncommitted changes, `dirty` is `true` and a warning is printed:
the run cannot be rebuilt from the commit alone. Commit before important runs.

## Seeding

`seed_everything(seed, deterministic)` seeds Python, NumPy and PyTorch (CPU + all GPUs).

| Setting | Result |
|---|---|
| `deterministic: false` (default) | Same seed → same data order and init; GPU kernels may still differ in the last digits |
| `deterministic: true` | Requests deterministic kernels; slower, use it for debugging or final reported runs |

The synthetic dataset uses its own seeded generator, so data is identical across processes
and runs regardless of the global RNG.

## Checklist for a reportable result

- [ ] Committed code (`dirty: false`)
- [ ] Config and `run_info.json` saved next to the result
- [ ] Hardware, batch size and precision written in the results table
- [ ] Ideally 2–3 seeds, reporting mean ± std

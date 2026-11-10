# 2. Configuration

Configs use [Hydra](https://hydra.cc/). One main file composes smaller ones:

```
configs/
├── train.yaml           # experiment name, seed, training + tracking settings
├── model/
│   └── mlp.yaml         # selected by  model=mlp
└── data/
    └── synthetic.yaml   # selected by  data=synthetic
```

`train.yaml` starts with a `defaults` list that picks one file from each group:

```yaml
defaults:
  - model: mlp
  - data: synthetic
  - _self_
```

## Overriding from the command line

Any value can be changed without editing files:

```bash
# change one value
python scripts/train.py train.lr=1.0e-3

# several values + a name for the run
python scripts/train.py experiment_name=big_mlp model.hidden_dim=512 train.max_steps=2000

# swap a whole group (after you add configs/model/transformer.yaml)
python scripts/train.py model=transformer

# add a key that does not exist yet
python scripts/train.py +train.label_smoothing=0.1
```

Tip: write floats as `1.0e-3`, not `1e-3`. Plain YAML reads `1e-3` as a string.

## Interpolation

Values can reference other values with `${...}`. The data config reuses the model sizes so
they can never disagree:

```yaml
# configs/data/synthetic.yaml
in_dim: ${model.in_dim}
num_classes: ${model.num_classes}
```

`run_dir: outputs/${experiment_name}` works the same way.

## Key settings

| Key | Meaning |
|---|---|
| `task` | Which task builder to use (`src/mltemplate/tasks/__init__.py`) |
| `experiment_name` | Run name; also the W&B run id and the output folder |
| `run_dir` | Where checkpoints and logs go. On RunPod, point it at the network volume |
| `seed`, `deterministic` | Reproducibility, see [05](05-tracking-and-reproducibility.md) |
| `train.max_steps` | Number of **optimizer** steps |
| `train.batch_size` | Micro-batch size **per GPU** |
| `train.grad_accum_steps` | Micro-batches per optimizer step |
| `train.precision` | `fp32`, `bf16` (recommended on Ampere/Hopper), `fp16` |
| `train.compile` | Wrap the model with `torch.compile` |
| `train.resume` | Continue from `last.pt` if it exists |
| `tracking.backend` | `console` or `wandb` |

Effective batch size = `batch_size × grad_accum_steps × number of GPUs`.

## Sweeps

Hydra's multirun runs one job per combination:

```bash
python scripts/train.py --multirun train.lr=1.0e-3,3.0e-3,1.0e-2 \
    'experiment_name=lr_${train.lr}'
```

Give each job its own `experiment_name`, otherwise they share `run_dir` and resume each
other's checkpoints. Results go to `outputs/lr_0.001/`, `outputs/lr_0.01/`, ... (Hydra also
creates an empty `multirun/` folder, which is git-ignored). For larger sweeps use W&B sweeps that call `scripts/train.py`.

## Why Hydra's own output folders are disabled

By default Hydra creates a timestamped folder per launch. With `torchrun`, every process
gets a slightly different timestamp, so each rank would write to a different folder. The
template turns that off (`hydra.run.dir: .`, `output_subdir: null`) and uses `run_dir`
as the single output location, which also makes resuming trivial.

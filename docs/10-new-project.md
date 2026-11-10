# 10. Starting a new project from the template

## 1. Create the repository

On GitHub, mark this repo as a **template** (Settings → Template repository). Then for each
project: **Use this template → Create a new repository**, e.g. `indo-gpt-from-scratch`.

Locally:

```bash
git clone https://github.com/<user>/indo-gpt-from-scratch.git
cd indo-gpt-from-scratch
pip install -e ".[dev]"
pytest            # the template must be green before you change anything
```

Optionally rename the package (`mltemplate` → `indogpt`) in `pyproject.toml`, the `src/`
folder and the imports.

## 2. What to replace and what to keep

| Keep as is | Replace / add |
|---|---|
| `training/` (trainer, checkpoint, DDP, scheduler) | `models/` your architecture |
| `utils/` (seed, tracking, repro) | `data/` your dataset |
| `scripts/train.py`, `scripts/evaluate.py` | `tasks/` a new task wiring them together |
| `docker/`, CI, pre-commit | `configs/model/`, `configs/data/` |
| Test patterns | Tests for your model and data |

## 3. Add a task

A task is the only connection between your code and the trainer:

```python
# src/mltemplate/tasks/language_model.py
import torch.nn.functional as F
from mltemplate.tasks.base import Task


def lm_loss(model, batch):
    logits = model(batch["input_ids"])  # (B, T, vocab)
    loss = F.cross_entropy(logits.flatten(0, 1).float(), batch["labels"].flatten())
    return loss, {}  # extra metrics optional


def build_lm_task(cfg) -> Task:
    model = GPT(cfg.model)
    train_ds = TokenDataset(cfg.data.train_bin, cfg.data.block_size)
    val_ds = TokenDataset(cfg.data.val_bin, cfg.data.block_size)
    return Task(model=model, train_dataset=train_ds, val_dataset=val_ds, loss_fn=lm_loss)
```

Register it in `src/mltemplate/tasks/__init__.py`:

```python
TASKS = {
    "classification": build_classification_task,
    "language_model": build_lm_task,
}
```

Add `configs/model/gpt_small.yaml` and `configs/data/indo_wiki.yaml`, then:

```bash
python scripts/train.py task=language_model model=gpt_small data=indo_wiki
```

Tip: make a config per experiment in `configs/experiment/` and select it with
`+experiment=shakespeare_char`, so the Tiny Shakespeare reproduction is one command.

## 4. Per-phase notes

### Phase 1: GPT from scratch
- Dataset items: `{"input_ids": (T,), "labels": (T,)}` from a `np.memmap` of token ids.
  For a memmap, random windows per item are fine; resume then relies on the seeded RNG.
- Add metric: tokens/sec = `samples_per_sec × block_size`.
- Tests: causal-mask invariant, overfit one batch, Shakespeare loss target in an
  experiment config.

### Phase 2: LoRA summarization
- Model from `transformers`; loss can be `model(**batch).loss`.
- Only LoRA parameters have `requires_grad=True`; `build_optimizer` already skips frozen
  ones.
- Evaluation (ROUGE, generation) is slow: run it in `scripts/evaluate.py` on `best.pt`,
  not every `eval_every` steps.

### Phase 3: road-damage detection
- Ultralytics YOLO has its own training loop. Reuse the template for configs, tracking,
  reproducibility, Docker, CI and the dataset/split code, and call Ultralytics from a
  script instead of the `Trainer`.
- Keep the split-by-sequence/city logic in `data/` with tests that prove no sequence appears
  in two splits.

## 5. Checklist for a new project

- [ ] Template tests pass before any change
- [ ] Task, model and data added, with an overfit-one-batch test
- [ ] Config files for model, data and at least one experiment
- [ ] Short end-to-end run on CPU passes in CI
- [ ] README: one-line summary, results table, exact reproduce commands

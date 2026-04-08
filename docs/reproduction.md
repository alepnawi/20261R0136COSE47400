# Reproduction

Run commands from the repository root.

## Install

```bash
python -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -e .
```

If the target model is not already cached locally, export one of these first:

```bash
export HF_TOKEN=...
# or
export HUGGINGFACE_HUB_TOKEN=...
```

## Reproduce the imported PAnDa development preset

```bash
./.venv/bin/python scripts/run_truthfulqa.py
```

This wrapper uses the imported historical PAnDa preset and writes outputs to
`results/dev/stage12_panda_truthfulqa_sanity10`.

## Reproduce the imported alpha-switch baseline

```bash
./.venv/bin/python scripts/run_alpha_switch_baseline.py
```

This wrapper uses the imported `stage11c_alpha_switch` preset and writes
outputs to `results/dev/stage11c_alpha_switch_sanity10_v3`.

## Plot a saved summary

```bash
./.venv/bin/python scripts/plot_results.py \
  results/dev/stage12_panda_truthfulqa_sanity10/summary.csv \
  --output results/dev/stage12_panda_truthfulqa_sanity10/overview.png \
  --title "PAnDa TruthfulQA sanity-10"
```

## Notes

- The JSON files in `configs/` are provenance snapshots copied from the
  original artifact metadata. They document the imported runs but are not yet
  consumed as declarative config files by the wrapper scripts.
- The current PAnDa artifact was historically developed under the internal
  codename `jaca`; public-facing exported labels in this repo use `PAnDa`.
- The evaluator reads `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` when loading remote
  Hugging Face assets.
- If your assets are already cached locally, append `--local-files-only` to the
  runner command.
- Depending on your environment, you may need locally cached model weights or
  access to Hugging Face-hosted assets.

## Build the paper

```bash
cd paper
latexmk -pdf panda_note.tex
```

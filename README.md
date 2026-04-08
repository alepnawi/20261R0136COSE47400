# PAnDa

PAnDa is the public-facing research repository for `Parallel-block Adaptive Contrast DoLa`, extracted from the broader `KeelNetV2` workspace into a cleaner, reproducibility-oriented layout.

## Status

- This repo currently reflects the imported Stage 12 PAnDa development artifact on `TruthfulQA sanity-10`.
- Historical `stage*` labels are intentionally preserved inside the imported code and metadata so they still match the original saved experiments.
- This is a cleaned research release, not a full mirror of the internal working directory.

## What is included

- `src/panda/eval.py`: the imported evaluation driver that contains the current PAnDa and baseline decoding logic.
- `scripts/`: lightweight entry points for rerunning the imported development presets and plotting saved summaries.
- `configs/`: provenance metadata snapshots for the imported baseline and PAnDa runs.
- `results/dev/`: saved metadata, summaries, pairwise summaries, and raw predictions for the imported baseline comparison and current PAnDa artifact.
- `paper/`: the current PAnDa paper draft and bibliography.
- `docs/`: method, reproduction, and limitation notes for the cleaned repo.

## Quick Start

Create an environment and install the package from the repository root:

```bash
python -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -e .
```

If the selected model is gated or not cached locally, set one of:

```bash
export HF_TOKEN=...
# or
export HUGGINGFACE_HUB_TOKEN=...
```

Run the imported PAnDa development preset:

```bash
./.venv/bin/python scripts/run_truthfulqa.py
```

Run the imported alpha-switch baseline preset:

```bash
./.venv/bin/python scripts/run_alpha_switch_baseline.py
```

Generate an overview plot from a saved summary:

```bash
./.venv/bin/python scripts/plot_results.py \
  results/dev/stage12_panda_truthfulqa_sanity10/summary.csv \
  --output results/dev/stage12_panda_truthfulqa_sanity10/overview.png \
  --title "PAnDa TruthfulQA sanity-10"
```

## Layout

```text
PAnDa/
  configs/
  docs/
  paper/
  results/
  scripts/
  src/panda/
```

## Reproducibility Notes

- The imported preset expects the same model family used in the original metadata snapshots. See `configs/stage12_panda_sanity10.json`.
- The evaluator reads `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` when loading remote Hugging Face assets.
- If your model and dataset assets are already cached, you can add `--local-files-only` to avoid network access.
- The current repo tracks compact summaries and metadata, not the full internal cache tree or every exploratory run from `KeelNetV2`.
- Some experiments may require locally cached Hugging Face assets or authentication, depending on the model you choose.
- The current PAnDa artifact was historically developed under the internal codename `jaca`; public-facing exported labels in this repo use `PAnDa`.

## Paper

Build the paper from the `paper/` directory with:

```bash
latexmk -pdf panda_note.tex
```

The current `panda_note.tex` uses an inline boxed figure. The files under
`paper/figures/` are kept as editable or archival assets and are not required
for the current TeX build.

## Current Caveat

The saved PAnDa evidence in this repo is still a development artifact rather than a publication-final benchmark package. See `docs/limitations.md` before presenting the current results as a final claim.

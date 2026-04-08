# Method

PAnDa stands for `Parallel-block Adaptive Contrast DoLa`.

The central idea is to keep two fixed contrast regimes inside the same
block-parallel decoding pass:

- a safer low-alpha contrast view
- a stronger truth-seeking high-alpha contrast view

Instead of generating two full responses and reranking them afterward, PAnDa
compares those two views locally inside each speculative block, detects the
first meaningful divergence point, and only applies truth-biased arbitration
from that point onward.

## Where the implementation lives

- The imported PAnDa block logic lives in `src/panda/eval.py`.
- The key block-parallel method is `run_jaca_block`, which is the historical
  internal function name for the current PAnDa block decoder.
- The main generation entry point is `generate_with_decoder`.
- The response-level comparison baseline remains in
  `generate_with_alpha_switch_car_dola`.

## Why the code still uses stage labels

This repository was extracted from a larger research workspace. The imported
implementation keeps the original stage naming so the code still lines up with
the saved metadata and result files under `configs/` and `results/dev/`.

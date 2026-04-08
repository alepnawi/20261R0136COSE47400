#!/usr/bin/env python3
"""Run the imported PAnDa TruthfulQA development preset."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from panda.eval import main


DEFAULT_ARGS = [
    "--comparison-preset",
    "stage12_jaca",
    "--mode",
    "sanity",
    "--save-results",
    "--results-dir",
    str(REPO_ROOT / "results" / "dev" / "stage12_panda_truthfulqa_sanity10"),
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    main()

#!/usr/bin/env python3
"""Run the imported alpha-switch baseline development preset."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from panda.eval import main


DEFAULT_ARGS = [
    "--comparison-preset",
    "stage11c_alpha_switch",
    "--mode",
    "sanity",
    "--save-results",
    "--results-dir",
    str(REPO_ROOT / "results" / "dev" / "stage11c_alpha_switch_sanity10_v3"),
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    main()

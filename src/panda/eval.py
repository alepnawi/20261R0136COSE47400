"""PAnDa evaluation driver imported from the KeelNetV2 research workspace.

The implementation is intentionally kept close to the original single-file
experiment runner so the saved artifacts in this repository remain traceable to
their source runs. Historical "stage" labels are preserved for provenance.
"""

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from datasets import load_dataset
except ImportError as exc:
    raise RuntimeError(
        "The 'datasets' package is required for evaluation. "
        "Install it with: python -m pip install datasets"
    ) from exc


torch.set_grad_enabled(False)

DEFAULT_STRATEGYQA_DATASET = "tasksource/bigbench"
DEFAULT_STRATEGYQA_CONFIG = "strategyqa"
DEFAULT_STRATEGYQA_SPLIT = "validation"
DEFAULT_STRATEGYQA_SOURCE = (
    f"{DEFAULT_STRATEGYQA_DATASET}/{DEFAULT_STRATEGYQA_CONFIG}:{DEFAULT_STRATEGYQA_SPLIT}"
)
DEFAULT_ALPACAEVAL_DATASET = "tatsu-lab/alpaca_eval"
DEFAULT_ALPACAEVAL_CONFIG = "alpaca_eval_gpt4_baseline"
DEFAULT_ALPACAEVAL_SPLIT = "eval"
DEFAULT_ALPACAEVAL_SOURCE = (
    f"{DEFAULT_ALPACAEVAL_DATASET}/{DEFAULT_ALPACAEVAL_CONFIG}:{DEFAULT_ALPACAEVAL_SPLIT}"
)


@dataclass
class DynDoLaConfig:
    shallow_bucket: list
    tau: float = 0.5
    lambda_margin: float = 1.0
    beta: float = 0.9
    gamma: float = 2.5
    alpha_min: float = 0.1
    alpha_max: float = 0.8
    update_every: int = 4
    instability_temperature: float = 10.0
    trisla_window: int = 4
    trisla_lambda_current: float = 0.6
    trisla_lambda_window: float = 0.3
    trisla_lambda_prefix: float = 0.1
    trisla_jsd_temperature: float = 1.0
    trisla_margin_epsilon: float = 0.001
    trisla_switch_margin: float = 0.0
    tqla_window: int = 4
    tqla_lambda_current: float = 0.6
    tqla_lambda_window: float = 0.3
    tqla_lambda_prefix: float = 0.1
    tqla_utility_epsilon: float = 0.001
    tqla_baseline_margin_threshold: float | None = None
    tqla_layer_deviation_penalty: float = 0.0
    tqla_flip_penalty: float = 0.0
    tqla_verify_epsilon: float = 0.0
    tqla_require_positive_signals: bool = False
    tqla_verify_top1_override: bool = False
    calibration_gate_ema_decay: float = 0.7
    calibration_gate_confidence_on: float = 0.55
    calibration_gate_confidence_off: float = 0.65
    calibration_gate_margin_on: float = 5.0
    calibration_gate_hold_steps: int = 2
    soft_decay_trigger_threshold: float = 0.15
    pressure_alpha_momentum: float = 0.8
    jacobi_window_size: int = 4
    jacobi_max_iters: int = 2
    jaca_divergence_threshold: float = 0.05
    jaca_truth_bias: float = 0.02
    jaca_early_agreement_shortcut: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="PAnDa evaluation driver imported from KeelNetV2.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", choices=("sanity", "subset", "full"), default="sanity")
    parser.add_argument("--truthfulqa-limit", type=str, default=None)
    parser.add_argument("--strategyqa-limit", type=str, default=None)
    parser.add_argument("--gsm8k-limit", type=str, default=None)
    parser.add_argument("--halueval-limit", type=str, default=None)
    parser.add_argument("--alpacaeval-limit", type=str, default=None)
    parser.add_argument(
        "--comparison-preset",
        choices=(
            "stage5_fixed_alpha_check",
            "stage6_trisla",
            "stage6_trisla_rescue",
            "stage7_tqla",
            "stage7_tqla_rescue",
            "stage8_soft_decay",
            "stage9_pressure_linear",
            "stage10_jacobi",
            "stage11_calibration_gated_fixed_alpha",
            "stage11b_car_dola",
            "stage11c_alpha_switch",
            "stage12_jaca",
            "stage14_jaca_eas",
        ),
        default=None,
        help="Apply a predefined decoder comparison setup.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--results-dir", default="results/dev/run")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--strict-eval", action="store_true")
    parser.add_argument("--dola-relative-top", type=float, default=0.1)
    parser.add_argument("--dola-relative-top-value", type=float, default=-1000.0)
    parser.add_argument("--skip-truthfulqa", action="store_true")
    parser.add_argument("--skip-strategyqa", action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    parser.add_argument("--include-halueval", action="store_true")
    parser.add_argument("--include-alpacaeval", action="store_true")
    parser.add_argument("--include-gsm8k-sequence", action="store_true")
    parser.add_argument("--alpacaeval-max-new-tokens", type=int, default=256)
    parser.add_argument("--halueval-root", type=str, default=None)
    parser.add_argument(
        "--halueval-tasks",
        type=str,
        default="qa,dialogue,summarization",
        help="Comma-separated HaluEval task files to use from the official release root.",
    )
    parser.add_argument("--sequence-max-new-tokens", type=int, default=160)
    parser.add_argument("--strategyqa-dataset", type=str, default=None)
    parser.add_argument("--strategyqa-config", type=str, default=None)
    parser.add_argument("--strategyqa-split", type=str, default=None)
    parser.add_argument("--include-no-ema-ablation", action="store_true")
    parser.add_argument("--include-fixed-alpha-ablation", action="store_true")
    parser.add_argument("--fixed-alpha-value", type=float, default=0.5)
    parser.add_argument("--include-fixed-layer-ablation", action="store_true")
    parser.add_argument("--fixed-layer-index", type=int, default=None)
    parser.add_argument("--include-trisla", action="store_true")
    parser.add_argument("--include-tqla", action="store_true")
    parser.add_argument("--include-soft-decay", action="store_true")
    parser.add_argument("--include-pressure-linear", action="store_true")
    parser.add_argument("--include-jacobi", action="store_true")
    parser.add_argument("--include-jaca", action="store_true")
    parser.add_argument("--include-calibration-gated-fixed-alpha", action="store_true")
    parser.add_argument("--include-car-dola", action="store_true")
    parser.add_argument("--include-alpha-switch-car-dola", action="store_true")
    parser.add_argument("--exclude-full-dyndola", action="store_true")
    parser.add_argument("--trisla-window", type=int, default=4)
    parser.add_argument("--trisla-lambda-current", type=float, default=0.6)
    parser.add_argument("--trisla-lambda-window", type=float, default=0.3)
    parser.add_argument("--trisla-lambda-prefix", type=float, default=0.1)
    parser.add_argument(
        "--trisla-jsd-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used only for TriSLA JSD arbitration/history. Default 1.0 avoids JSD saturation.",
    )
    parser.add_argument(
        "--trisla-margin-epsilon",
        type=float,
        default=0.001,
        help="Minimum TriSLA relative-advantage margin required before selected-layer history is reinforced.",
    )
    parser.add_argument(
        "--trisla-switch-margin",
        type=float,
        default=0.0,
        help="Minimum arbitration-score gain required before TriSLA switches away from the incumbent layer.",
    )
    parser.add_argument(
        "--trisla-window-grid",
        type=str,
        default=None,
        help="Comma-separated TriSLA window sweep, e.g. 2,4,6",
    )
    parser.add_argument(
        "--trisla-weight-grid",
        type=str,
        default=None,
        help="Semicolon-separated TriSLA weight triples current,window,prefix, e.g. 0.6,0.3,0.1;0.8,0.15,0.05",
    )
    parser.add_argument("--tqla-window", type=int, default=4)
    parser.add_argument("--tqla-lambda-current", type=float, default=0.6)
    parser.add_argument("--tqla-lambda-window", type=float, default=0.3)
    parser.add_argument("--tqla-lambda-prefix", type=float, default=0.1)
    parser.add_argument(
        "--tqla-utility-epsilon",
        type=float,
        default=0.001,
        help="Minimum TqLA utility score required before overriding the DoLa-FixedAlpha fallback path.",
    )
    parser.add_argument(
        "--tqla-baseline-margin-threshold",
        type=float,
        default=None,
        help="Only allow TqLA overrides when the DoLa-FixedAlpha baseline margin is below this threshold.",
    )
    parser.add_argument(
        "--tqla-layer-deviation-penalty",
        type=float,
        default=0.0,
        help="Penalty applied per layer of deviation away from the DoLa-FixedAlpha selected layer.",
    )
    parser.add_argument(
        "--tqla-flip-penalty",
        type=float,
        default=0.0,
        help="Penalty applied when a TqLA candidate changes the baseline top-1 token before verification.",
    )
    parser.add_argument(
        "--tqla-verify-epsilon",
        type=float,
        default=0.0,
        help="Minimum one-step verified utility gain required before a top-1-changing TqLA override is accepted.",
    )
    parser.add_argument(
        "--tqla-require-positive-signals",
        action="store_true",
        help="Require both margin gain and entropy gain to be positive before a TqLA candidate is eligible.",
    )
    parser.add_argument(
        "--tqla-verify-top1-override",
        action="store_true",
        help="Run an extra one-step fixed-alpha verification pass before accepting a TqLA top-1 token change.",
    )
    parser.add_argument(
        "--calibration-gate-ema-decay",
        type=float,
        default=0.7,
        help="EMA decay used to smooth the Stage 11 confidence signal before hysteretic gating.",
    )
    parser.add_argument(
        "--calibration-gate-confidence-on",
        type=float,
        default=0.55,
        help="Turn fixed-alpha contrast on when smoothed confidence drops below this threshold.",
    )
    parser.add_argument(
        "--calibration-gate-confidence-off",
        type=float,
        default=0.65,
        help="Turn fixed-alpha contrast off when smoothed confidence rises above this threshold.",
    )
    parser.add_argument(
        "--calibration-gate-margin-on",
        type=float,
        default=5.0,
        help="Only activate fixed-alpha contrast when the base-model top1-top2 margin is below this threshold.",
    )
    parser.add_argument(
        "--calibration-gate-hold-steps",
        type=int,
        default=2,
        help="Minimum number of steps to keep the Stage 11 gate active after it turns on.",
    )
    parser.add_argument(
        "--calibration-gate-confidence-source",
        choices=("base_top1_prob",),
        default="base_top1_prob",
        help="Confidence signal source for Stage 11. The current pilot uses base-model top-1 probability.",
    )
    parser.add_argument(
        "--calibration-rerank-delta",
        type=float,
        default=0.02,
        help="Minimum confidence margin required before CaR-DoLa overrides the DoLa-FixedAlpha default.",
    )
    parser.add_argument(
        "--calibration-confidence-max-new-tokens",
        type=int,
        default=12,
        help="Maximum tokens to generate when querying the self-calibrated model for response-level confidence.",
    )
    parser.add_argument(
        "--alpha-switch-low",
        type=float,
        default=0.2,
        help="Low fixed-alpha candidate used by the alpha-switch reranker.",
    )
    parser.add_argument(
        "--alpha-switch-high",
        type=float,
        default=0.8,
        help="High fixed-alpha candidate used by the alpha-switch reranker.",
    )
    parser.add_argument(
        "--soft-decay-alpha-base",
        type=float,
        default=None,
        help="Base alpha for the soft-decay decoder. Defaults to --fixed-alpha-value when omitted.",
    )
    parser.add_argument(
        "--soft-decay-alpha-peak",
        type=float,
        default=0.8,
        help="Peak alpha restored when the soft-decay decoder sees a risk trigger.",
    )
    parser.add_argument(
        "--soft-decay-half-life",
        type=float,
        default=6.0,
        help="Token half-life for exponential alpha decay back toward the base value.",
    )
    parser.add_argument(
        "--soft-decay-trigger-threshold",
        type=float,
        default=0.15,
        help="Trigger when normalized instability meets or exceeds this threshold.",
    )
    parser.add_argument(
        "--pressure-alpha-base",
        type=float,
        default=None,
        help="Base alpha for the pressure-linear decoder. Defaults to --fixed-alpha-value when omitted.",
    )
    parser.add_argument(
        "--pressure-alpha-peak",
        type=float,
        default=0.8,
        help="Peak alpha for the pressure-linear decoder when pressure saturates at 1.",
    )
    parser.add_argument(
        "--pressure-alpha-momentum",
        type=float,
        default=0.8,
        help="EMA momentum for the pressure-linear decoder's risk state.",
    )
    parser.add_argument(
        "--jacobi-window-size",
        type=int,
        default=4,
        help="Speculative window size for DoLa-Fixed-Jacobi block refinement.",
    )
    parser.add_argument(
        "--jacobi-max-iters",
        type=int,
        default=2,
        help="Maximum Jacobi refinement passes per speculative block.",
    )
    parser.add_argument(
        "--jaca-divergence-threshold",
        type=float,
        default=0.05,
        help="Minimum low/high regime JSD required before JACA treats a block position as a local disagreement.",
    )
    parser.add_argument(
        "--jaca-truth-bias",
        type=float,
        default=0.02,
        help="Truth-bias margin used by JACA when deciding whether to keep the high-alpha local update.",
    )
    parser.add_argument(
        "--jaca-early-agreement-shortcut",
        action="store_true",
        help="Commit the leading JACA block prefix when low/high regimes do not produce a genuine conflict.",
    )
    parser.add_argument(
        "--shallow-bucket",
        type=str,
        default=None,
        help="Comma-separated global shallow bucket override, e.g. 0,2,4,6,8",
    )
    parser.add_argument(
        "--truthfulqa-shallow-bucket",
        type=str,
        default=None,
        help="Comma-separated TruthfulQA-only shallow bucket override, e.g. 12,14,16,18",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_limit(raw_value, mode, default_value):
    if raw_value is not None:
        if raw_value.lower() in {"none", "all", "full"}:
            return None
        return int(raw_value)
    if mode == "sanity":
        return default_value
    if mode == "subset":
        return 20
    return None


def apply_comparison_preset(args):
    if args.comparison_preset == "stage5_fixed_alpha_check":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = False
        args.include_soft_decay = False
        args.exclude_full_dyndola = False
    elif args.comparison_preset == "stage6_trisla":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = True
        args.include_tqla = False
        args.include_soft_decay = False
        args.exclude_full_dyndola = True
        args.strict_eval = True
    elif args.comparison_preset == "stage6_trisla_rescue":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = True
        args.include_tqla = False
        args.include_soft_decay = False
        args.include_pressure_linear = False
        args.include_jacobi = False
        args.exclude_full_dyndola = True
        args.strict_eval = True
        if float(args.trisla_switch_margin) == 0.0:
            args.trisla_switch_margin = 0.001
    elif args.comparison_preset == "stage7_tqla":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = True
        args.include_soft_decay = False
        args.exclude_full_dyndola = True
        args.strict_eval = True
    elif args.comparison_preset == "stage7_tqla_rescue":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = True
        args.include_soft_decay = False
        args.include_pressure_linear = False
        args.include_jacobi = False
        args.exclude_full_dyndola = True
        args.strict_eval = True
        if float(args.tqla_utility_epsilon) == 0.001:
            args.tqla_utility_epsilon = 0.01
        if args.tqla_baseline_margin_threshold is None:
            args.tqla_baseline_margin_threshold = 5.0
        if float(args.tqla_layer_deviation_penalty) == 0.0:
            args.tqla_layer_deviation_penalty = 0.05
        if float(args.tqla_flip_penalty) == 0.0:
            args.tqla_flip_penalty = 0.1
        if float(args.tqla_verify_epsilon) == 0.0:
            args.tqla_verify_epsilon = 0.01
        if not args.tqla_require_positive_signals:
            args.tqla_require_positive_signals = True
        if not args.tqla_verify_top1_override:
            args.tqla_verify_top1_override = True
    elif args.comparison_preset == "stage8_soft_decay":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = False
        args.include_soft_decay = True
        args.include_pressure_linear = False
        args.exclude_full_dyndola = True
        args.strict_eval = True
    elif args.comparison_preset == "stage9_pressure_linear":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = False
        args.include_soft_decay = False
        args.include_pressure_linear = True
        args.include_jacobi = False
        args.exclude_full_dyndola = True
        args.strict_eval = True
    elif args.comparison_preset == "stage10_jacobi":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = False
        args.include_soft_decay = False
        args.include_pressure_linear = False
        args.include_jacobi = True
        args.exclude_full_dyndola = True
        args.strict_eval = True
    elif args.comparison_preset == "stage11_calibration_gated_fixed_alpha":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = False
        args.include_soft_decay = False
        args.include_pressure_linear = False
        args.include_jacobi = False
        args.include_calibration_gated_fixed_alpha = True
        args.exclude_full_dyndola = True
        args.strict_eval = True
        if args.model_name == "Qwen/Qwen2.5-3B-Instruct":
            args.model_name = "HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration"
    elif args.comparison_preset == "stage11b_car_dola":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = False
        args.include_soft_decay = False
        args.include_pressure_linear = False
        args.include_jacobi = False
        args.include_calibration_gated_fixed_alpha = False
        args.include_car_dola = True
        args.exclude_full_dyndola = True
        args.strict_eval = True
        args.skip_strategyqa = True
        args.skip_gsm8k = True
        if args.model_name == "Qwen/Qwen2.5-3B-Instruct":
            args.model_name = "HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration"
    elif args.comparison_preset == "stage11c_alpha_switch":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = True
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = False
        args.include_soft_decay = False
        args.include_pressure_linear = False
        args.include_jacobi = False
        args.include_calibration_gated_fixed_alpha = False
        args.include_car_dola = False
        args.include_alpha_switch_car_dola = True
        args.exclude_full_dyndola = True
        args.strict_eval = True
        args.skip_strategyqa = True
        args.skip_gsm8k = True
        if float(args.alpha_switch_low) == 0.2:
            args.alpha_switch_low = 0.1
        if float(args.alpha_switch_high) == 0.8:
            args.alpha_switch_high = 0.95
        if args.model_name == "Qwen/Qwen2.5-3B-Instruct":
            args.model_name = "HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration"
    elif args.comparison_preset == "stage12_jaca":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = False
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = False
        args.include_soft_decay = False
        args.include_pressure_linear = False
        args.include_jacobi = False
        args.include_jaca = True
        args.include_calibration_gated_fixed_alpha = False
        args.include_car_dola = False
        args.include_alpha_switch_car_dola = True
        args.exclude_full_dyndola = True
        args.strict_eval = True
        args.skip_strategyqa = True
        args.skip_gsm8k = True
        if float(args.alpha_switch_low) == 0.2:
            args.alpha_switch_low = 0.1
        if float(args.alpha_switch_high) == 0.8:
            args.alpha_switch_high = 0.95
        if args.model_name == "Qwen/Qwen2.5-3B-Instruct":
            args.model_name = "HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration"
    elif args.comparison_preset == "stage14_jaca_eas":
        args.include_no_ema_ablation = False
        args.include_fixed_alpha_ablation = False
        args.include_fixed_layer_ablation = False
        args.include_trisla = False
        args.include_tqla = False
        args.include_soft_decay = False
        args.include_pressure_linear = False
        args.include_jacobi = False
        args.include_jaca = True
        args.include_calibration_gated_fixed_alpha = False
        args.include_car_dola = False
        args.include_alpha_switch_car_dola = True
        args.exclude_full_dyndola = True
        args.strict_eval = True
        args.skip_strategyqa = True
        args.skip_gsm8k = True
        args.jaca_early_agreement_shortcut = True
        if float(args.alpha_switch_low) == 0.2:
            args.alpha_switch_low = 0.1
        if float(args.alpha_switch_high) == 0.8:
            args.alpha_switch_high = 0.95
        if float(args.jaca_divergence_threshold) == 0.05:
            args.jaca_divergence_threshold = 0.15
        if args.model_name == "Qwen/Qwen2.5-3B-Instruct":
            args.model_name = "HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration"
    return args


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def parse_bucket_spec(spec):
    if spec is None:
        return None
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError(f"Invalid empty bucket spec: {spec!r}")
    return values


def parse_int_grid(spec):
    if spec is None:
        return None
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError(f"Invalid empty int grid: {spec!r}")
    return values


def parse_weight_grid(spec):
    if spec is None:
        return None
    triples = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid TriSLA weight triple: {chunk!r}")
        triples.append(tuple(float(part) for part in parts))
    if not triples:
        raise ValueError(f"Invalid empty weight grid: {spec!r}")
    return triples


def sample_candidate_rows(candidates, limit, rng):
    population_size = len(candidates)
    if population_size == 0:
        return [], {
            "usable_row_count": 0,
            "selected_row_count": 0,
            "sampling_mode": "empty",
            "selected_source_indices": [],
        }

    if limit is None or limit >= population_size:
        selected_positions = list(range(population_size))
        sampling_mode = "all_usable_rows"
    else:
        selected_positions = sorted(rng.sample(range(population_size), int(limit)))
        sampling_mode = "seeded_random_subset"

    selected_rows = []
    selected_source_indices = []
    for position in selected_positions:
        row = dict(candidates[position])
        selected_source_indices.append(int(row.pop("source_idx")))
        selected_rows.append(row)

    manifest = {
        "usable_row_count": population_size,
        "selected_row_count": len(selected_rows),
        "sampling_mode": sampling_mode,
    }
    if sampling_mode == "seeded_random_subset":
        manifest["selected_source_indices"] = selected_source_indices
    return selected_rows, manifest


def make_sampling_rng(seed, dataset_name):
    return random.Random(f"{int(seed)}:{dataset_name}")


def canonicalize_number_text(text):
    text = str(text).strip().replace(",", "")
    final_answer_matches = re.findall(
        r"final\s+answer\s*:\s*([-+]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_answer_matches:
        return final_answer_matches[-1].rstrip(". ")
    if text.lower().startswith("final answer:"):
        text = text.split(":", 1)[1].strip()
    text = text.rstrip(". ")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return text
    return None


def canonicalize_yes_no_label(value):
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return canonicalize_yes_no_label(value[0])
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return "yes" if value == 1 else "no" if value == 0 else None

    text = str(value).strip().lower()
    text = re.sub(r"^[\"'\[\(\s]+|[\"'\]\)\s]+$", "", text)
    text = re.sub(r"[.?!]+$", "", text).strip()
    label_map = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "0": "no",
    }
    return label_map.get(text)


def get_base_model(causal_lm):
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(causal_lm, attr):
            return getattr(causal_lm, attr)
    raise ValueError("Unsupported architecture. Add a base-model accessor for this model class.")


def apply_final_norm(base_model, hidden_state):
    for attr in ("norm", "final_layernorm", "ln_f"):
        if hasattr(base_model, attr):
            return getattr(base_model, attr)(hidden_state)
    return hidden_state


def get_decoder_names(args):
    if args.comparison_preset in {"stage12_jaca", "stage14_jaca_eas"}:
        return (
            "greedy",
            "dola",
            "alpha_switch_car_dola",
            "jaca",
        )
    if args.comparison_preset == "stage11c_alpha_switch":
        return (
            "greedy",
            "dola",
            "dyndola_fixed_alpha_low",
            "dyndola_fixed_alpha",
            "dyndola_fixed_alpha_high",
            "alpha_switch_car_dola",
        )
    if args.comparison_preset == "stage8_soft_decay":
        return ("dyndola_fixed_alpha", "soft_decay_alpha")
    if args.comparison_preset == "stage9_pressure_linear":
        return ("dyndola_fixed_alpha", "pressure_linear_alpha")
    if args.comparison_preset == "stage10_jacobi":
        return ("dyndola_fixed_alpha", "dola_fixed_jacobi")
    decoders = ["greedy", "dola"]
    if not args.exclude_full_dyndola:
        decoders.append("dyndola")
    if args.include_no_ema_ablation:
        decoders.append("dyndola_no_ema")
    if args.include_fixed_alpha_ablation:
        decoders.append("dyndola_fixed_alpha")
    if args.include_fixed_layer_ablation:
        decoders.append("dyndola_fixed_layer")
    if args.include_trisla:
        decoders.append("trisla")
    if args.include_tqla:
        decoders.append("tqla")
    if args.include_soft_decay:
        decoders.append("soft_decay_alpha")
    if args.include_pressure_linear:
        decoders.append("pressure_linear_alpha")
    if args.include_jacobi:
        decoders.append("dola_fixed_jacobi")
    if args.include_jaca:
        decoders.append("jaca")
    if args.include_calibration_gated_fixed_alpha:
        decoders.append("calibration_gated_fixed_alpha")
    if args.include_car_dola:
        decoders.append("car_dola")
    if args.include_alpha_switch_car_dola:
        decoders.extend(("dyndola_fixed_alpha_low", "dyndola_fixed_alpha_high", "alpha_switch_car_dola"))
    return tuple(decoders)


DECODER_LABELS = {
    "greedy": "Greedy",
    "dola": "DoLa",
    "dyndola": "Full DynDoLa",
    "dyndola_no_ema": "DynDoLa-NoEMA",
    "dyndola_fixed_alpha": "DoLa-FixedAlpha",
    "dyndola_fixed_alpha_low": "DoLa-FixedAlpha (Low)",
    "dyndola_fixed_alpha_high": "DoLa-FixedAlpha (High)",
    "dyndola_fixed_layer": "DynDoLa-FixedLayer",
    "trisla": "TriSLA",
    "tqla": "Token-quality Layer Arbitration",
    "soft_decay_alpha": "Soft-Decay Alpha",
    "pressure_linear_alpha": "Pressure-Linear Alpha",
    "dola_fixed_jacobi": "DoLa-Fixed-Jacobi",
    "jaca": "JACA",
    "calibration_gated_fixed_alpha": "Calibration-Gated Fixed-Alpha DoLa",
    "car_dola": "Calibration-Reranked Fixed-Alpha DoLa",
    "alpha_switch_car_dola": "Alpha-Switch Calibration-Reranked DoLa",
}


def get_decoder_label(decoder_name):
    return DECODER_LABELS.get(decoder_name, decoder_name)


class Stage4Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.use_chat_template = not args.no_chat_template
        self.decoder_names = get_decoder_names(args)
        self.decoder_labels = {name: get_decoder_label(name) for name in self.decoder_names}
        self.fixed_alpha_value = float(args.fixed_alpha_value)
        self.soft_decay_alpha_base = (
            self.fixed_alpha_value if args.soft_decay_alpha_base is None else float(args.soft_decay_alpha_base)
        )
        self.soft_decay_alpha_peak = float(args.soft_decay_alpha_peak)
        self.soft_decay_half_life = float(args.soft_decay_half_life)
        self.pressure_alpha_base = (
            self.fixed_alpha_value if args.pressure_alpha_base is None else float(args.pressure_alpha_base)
        )
        self.pressure_alpha_peak = float(args.pressure_alpha_peak)
        self.pressure_alpha_momentum = float(args.pressure_alpha_momentum)
        self.jacobi_window_size = int(args.jacobi_window_size)
        self.jacobi_max_iters = int(args.jacobi_max_iters)
        self.jaca_divergence_threshold = float(args.jaca_divergence_threshold)
        self.jaca_truth_bias = float(args.jaca_truth_bias)
        self.jaca_early_agreement_shortcut = bool(args.jaca_early_agreement_shortcut)
        self.alpacaeval_max_new_tokens = int(args.alpacaeval_max_new_tokens)
        self.halueval_root = args.halueval_root
        self.halueval_tasks = tuple(
            part.strip() for part in str(args.halueval_tasks).split(",") if part.strip()
        )
        self.calibration_gate_confidence_source = str(args.calibration_gate_confidence_source)
        self.calibration_rerank_delta = float(args.calibration_rerank_delta)
        self.calibration_confidence_max_new_tokens = int(args.calibration_confidence_max_new_tokens)
        self.alpha_switch_low = float(args.alpha_switch_low)
        self.alpha_switch_high = float(args.alpha_switch_high)
        self.dola_relative_top = float(args.dola_relative_top)
        self.dola_relative_top_value = float(args.dola_relative_top_value)
        self.global_bucket_override = parse_bucket_spec(args.shallow_bucket)
        self.truthfulqa_bucket_override = parse_bucket_spec(args.truthfulqa_shallow_bucket)
        if args.trisla_window < 1:
            raise ValueError("--trisla-window must be >= 1")
        if args.trisla_jsd_temperature <= 0.0:
            raise ValueError("--trisla-jsd-temperature must be > 0")
        if args.trisla_margin_epsilon < 0.0:
            raise ValueError("--trisla-margin-epsilon must be >= 0")
        if args.trisla_switch_margin < 0.0:
            raise ValueError("--trisla-switch-margin must be >= 0")
        if args.tqla_window < 1:
            raise ValueError("--tqla-window must be >= 1")
        if args.tqla_utility_epsilon < 0.0:
            raise ValueError("--tqla-utility-epsilon must be >= 0")
        if args.tqla_baseline_margin_threshold is not None and args.tqla_baseline_margin_threshold < 0.0:
            raise ValueError("--tqla-baseline-margin-threshold must be >= 0")
        if args.tqla_layer_deviation_penalty < 0.0:
            raise ValueError("--tqla-layer-deviation-penalty must be >= 0")
        if args.tqla_flip_penalty < 0.0:
            raise ValueError("--tqla-flip-penalty must be >= 0")
        if args.tqla_verify_epsilon < 0.0:
            raise ValueError("--tqla-verify-epsilon must be >= 0")
        if not 0.0 <= args.calibration_gate_ema_decay < 1.0:
            raise ValueError("--calibration-gate-ema-decay must be in [0, 1)")
        if not 0.0 <= args.calibration_gate_confidence_on <= 1.0:
            raise ValueError("--calibration-gate-confidence-on must be in [0, 1]")
        if not 0.0 <= args.calibration_gate_confidence_off <= 1.0:
            raise ValueError("--calibration-gate-confidence-off must be in [0, 1]")
        if args.calibration_gate_confidence_on > args.calibration_gate_confidence_off:
            raise ValueError("--calibration-gate-confidence-on must be <= --calibration-gate-confidence-off")
        if args.calibration_gate_margin_on < 0.0:
            raise ValueError("--calibration-gate-margin-on must be >= 0")
        if args.calibration_gate_hold_steps < 0:
            raise ValueError("--calibration-gate-hold-steps must be >= 0")
        if self.calibration_rerank_delta < 0.0:
            raise ValueError("--calibration-rerank-delta must be >= 0")
        if self.calibration_confidence_max_new_tokens < 1:
            raise ValueError("--calibration-confidence-max-new-tokens must be >= 1")
        if self.alpha_switch_low < 0.0:
            raise ValueError("--alpha-switch-low must be >= 0")
        if self.alpha_switch_high < 0.0:
            raise ValueError("--alpha-switch-high must be >= 0")
        if self.alpha_switch_low > self.alpha_switch_high:
            raise ValueError("--alpha-switch-low must be <= --alpha-switch-high")
        if self.soft_decay_alpha_base < 0.0:
            raise ValueError("--soft-decay-alpha-base must be >= 0")
        if self.soft_decay_alpha_peak < self.soft_decay_alpha_base:
            raise ValueError("--soft-decay-alpha-peak must be >= --soft-decay-alpha-base")
        if self.soft_decay_half_life <= 0.0:
            raise ValueError("--soft-decay-half-life must be > 0")
        if self.pressure_alpha_base < 0.0:
            raise ValueError("--pressure-alpha-base must be >= 0")
        if self.pressure_alpha_peak < self.pressure_alpha_base:
            raise ValueError("--pressure-alpha-peak must be >= --pressure-alpha-base")
        if not 0.0 <= self.pressure_alpha_momentum < 1.0:
            raise ValueError("--pressure-alpha-momentum must be in [0, 1)")
        if self.jacobi_window_size < 1:
            raise ValueError("--jacobi-window-size must be >= 1")
        if self.jacobi_max_iters < 1:
            raise ValueError("--jacobi-max-iters must be >= 1")
        if self.jaca_divergence_threshold < 0.0:
            raise ValueError("--jaca-divergence-threshold must be >= 0")
        if self.alpacaeval_max_new_tokens < 1:
            raise ValueError("--alpacaeval-max-new-tokens must be >= 1")
        if args.include_halueval and not self.halueval_root:
            raise ValueError("--halueval-root is required when --include-halueval is enabled")
        if args.include_halueval and not self.halueval_tasks:
            raise ValueError("--halueval-tasks must contain at least one task when --include-halueval is enabled")
        self.soft_decay_decay_rho = math.pow(0.5, 1.0 / self.soft_decay_half_life)
        trisla_weight_total = (
            float(args.trisla_lambda_current)
            + float(args.trisla_lambda_window)
            + float(args.trisla_lambda_prefix)
        )
        if trisla_weight_total <= 0.0:
            raise ValueError("At least one TriSLA weight must be positive.")
        tqla_weight_total = (
            float(args.tqla_lambda_current)
            + float(args.tqla_lambda_window)
            + float(args.tqla_lambda_prefix)
        )
        if tqla_weight_total <= 0.0:
            raise ValueError("At least one TqLA weight must be positive.")

        print(
            {
                "model_name": args.model_name,
                "device": self.device,
                "dtype": str(self.dtype),
                "hf_token_visible": self.hf_token is not None,
                "local_files_only": args.local_files_only,
                "mode": args.mode,
                "decoders": self.decoder_labels,
                "fixed_alpha_value": self.fixed_alpha_value,
                "soft_decay_config": {
                    "alpha_base": self.soft_decay_alpha_base,
                    "alpha_peak": self.soft_decay_alpha_peak,
                    "half_life": self.soft_decay_half_life,
                    "decay_rho": self.soft_decay_decay_rho,
                    "trigger_threshold": args.soft_decay_trigger_threshold,
                },
                "pressure_linear_config": {
                    "alpha_base": self.pressure_alpha_base,
                    "alpha_peak": self.pressure_alpha_peak,
                    "momentum": self.pressure_alpha_momentum,
                    "risk_source": "max(0, normalized_instability)",
                },
                "jacobi_config": {
                    "window_size": self.jacobi_window_size,
                    "max_iters": self.jacobi_max_iters,
                    "init_strategy": "repeat_last",
                    "commit_strategy": "stable_prefix_then_fallback_1",
                },
                "jaca_config": {
                    "divergence_threshold": self.jaca_divergence_threshold,
                    "truth_bias": self.jaca_truth_bias,
                    "low_alpha": self.alpha_switch_low,
                    "high_alpha": self.alpha_switch_high,
                    "local_score": "top1_confidence",
                    "early_agreement_shortcut": self.jaca_early_agreement_shortcut,
                },
                "alpacaeval_config": {
                    "include_alpacaeval": args.include_alpacaeval,
                    "max_new_tokens": self.alpacaeval_max_new_tokens,
                    "official_scorer_installed": importlib.util.find_spec("alpaca_eval") is not None,
                },
                "halueval_config": {
                    "include_halueval": args.include_halueval,
                    "root": self.halueval_root,
                    "tasks": self.halueval_tasks,
                },
                "strict_eval": args.strict_eval,
                "dola_relative_top": self.dola_relative_top,
                "dola_relative_top_value": self.dola_relative_top_value,
                "trisla_window": args.trisla_window,
                "trisla_weights": {
                    "current": args.trisla_lambda_current,
                    "window": args.trisla_lambda_window,
                    "prefix": args.trisla_lambda_prefix,
                },
                "trisla_jsd_temperature": args.trisla_jsd_temperature,
                "trisla_margin_epsilon": args.trisla_margin_epsilon,
                "trisla_switch_margin": args.trisla_switch_margin,
                "tqla_window": args.tqla_window,
                "tqla_weights": {
                    "current": args.tqla_lambda_current,
                    "window": args.tqla_lambda_window,
                    "prefix": args.tqla_lambda_prefix,
                },
                "tqla_utility_epsilon": args.tqla_utility_epsilon,
                "tqla_baseline_margin_threshold": args.tqla_baseline_margin_threshold,
                "tqla_layer_deviation_penalty": args.tqla_layer_deviation_penalty,
                "tqla_flip_penalty": args.tqla_flip_penalty,
                "tqla_verify_epsilon": args.tqla_verify_epsilon,
                "tqla_require_positive_signals": args.tqla_require_positive_signals,
                "tqla_verify_top1_override": args.tqla_verify_top1_override,
                "calibration_gate": {
                    "ema_decay": args.calibration_gate_ema_decay,
                    "confidence_on": args.calibration_gate_confidence_on,
                    "confidence_off": args.calibration_gate_confidence_off,
                    "margin_on": args.calibration_gate_margin_on,
                    "hold_steps": args.calibration_gate_hold_steps,
                    "confidence_source": self.calibration_gate_confidence_source,
                },
                "calibration_rerank": {
                    "delta": self.calibration_rerank_delta,
                    "confidence_max_new_tokens": self.calibration_confidence_max_new_tokens,
                },
                "alpha_switch": {
                    "low": self.alpha_switch_low,
                    "high": self.alpha_switch_high,
                },
                "include_tqla": args.include_tqla,
                "include_calibration_gated_fixed_alpha": args.include_calibration_gated_fixed_alpha,
                "include_car_dola": args.include_car_dola,
                "include_alpha_switch_car_dola": args.include_alpha_switch_car_dola,
            }
        )

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            token=self.hf_token,
            local_files_only=args.local_files_only,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            token=self.hf_token,
            local_files_only=args.local_files_only,
        )
        self.model.eval()
        self.model_input_device = next(self.model.parameters()).device
        self._confidence_allowed_token_ids = None
        self._confidence_first_token_ids = None

        num_layers = getattr(self.model.config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Could not infer num_hidden_layers from model.config.")
        default_bucket = self.global_bucket_override
        if default_bucket is None:
            default_bucket = list(range(0, max(1, num_layers // 4), 2))
            if not default_bucket:
                default_bucket = [0]
        default_bucket = [idx for idx in default_bucket if 0 <= idx < num_layers]
        if not default_bucket:
            raise ValueError("No valid shallow bucket indices remain after filtering against model depth.")
        self.mature_layer_index = num_layers - 1
        self.default_fixed_layer = default_bucket[0]
        self.fixed_layer_index = (
            int(args.fixed_layer_index) if args.fixed_layer_index is not None else self.default_fixed_layer
        )
        self.default_bucket = list(default_bucket)
        self.truthfulqa_bucket = (
            [idx for idx in self.truthfulqa_bucket_override if 0 <= idx < num_layers]
            if self.truthfulqa_bucket_override is not None
            else list(self.default_bucket)
        )
        if not self.truthfulqa_bucket:
            raise ValueError("No valid TruthfulQA bucket indices remain after filtering against model depth.")
        self.cfg = DynDoLaConfig(
            shallow_bucket=list(self.default_bucket),
            trisla_window=int(args.trisla_window),
            trisla_lambda_current=float(args.trisla_lambda_current),
            trisla_lambda_window=float(args.trisla_lambda_window),
            trisla_lambda_prefix=float(args.trisla_lambda_prefix),
            trisla_jsd_temperature=float(args.trisla_jsd_temperature),
            trisla_margin_epsilon=float(args.trisla_margin_epsilon),
            trisla_switch_margin=float(args.trisla_switch_margin),
            tqla_window=int(args.tqla_window),
            tqla_lambda_current=float(args.tqla_lambda_current),
            tqla_lambda_window=float(args.tqla_lambda_window),
            tqla_lambda_prefix=float(args.tqla_lambda_prefix),
            tqla_utility_epsilon=float(args.tqla_utility_epsilon),
            tqla_baseline_margin_threshold=(
                None
                if args.tqla_baseline_margin_threshold is None
                else float(args.tqla_baseline_margin_threshold)
            ),
            tqla_layer_deviation_penalty=float(args.tqla_layer_deviation_penalty),
            tqla_flip_penalty=float(args.tqla_flip_penalty),
            tqla_verify_epsilon=float(args.tqla_verify_epsilon),
            tqla_require_positive_signals=bool(args.tqla_require_positive_signals),
            tqla_verify_top1_override=bool(args.tqla_verify_top1_override),
            calibration_gate_ema_decay=float(args.calibration_gate_ema_decay),
            calibration_gate_confidence_on=float(args.calibration_gate_confidence_on),
            calibration_gate_confidence_off=float(args.calibration_gate_confidence_off),
            calibration_gate_margin_on=float(args.calibration_gate_margin_on),
            calibration_gate_hold_steps=int(args.calibration_gate_hold_steps),
            soft_decay_trigger_threshold=float(args.soft_decay_trigger_threshold),
            pressure_alpha_momentum=float(args.pressure_alpha_momentum),
            jacobi_window_size=self.jacobi_window_size,
            jacobi_max_iters=self.jacobi_max_iters,
            jaca_divergence_threshold=self.jaca_divergence_threshold,
            jaca_truth_bias=self.jaca_truth_bias,
            jaca_early_agreement_shortcut=self.jaca_early_agreement_shortcut,
        )

        print(
            {
                "model_input_device": str(self.model_input_device),
                "num_layers": num_layers,
                "default_shallow_bucket": self.default_bucket,
                "truthfulqa_shallow_bucket": self.truthfulqa_bucket,
                "dola_mature_layer": self.mature_layer_index,
                "default_fixed_layer": self.default_fixed_layer,
                "fixed_layer_index": self.fixed_layer_index,
            }
        )

    def use_default_bucket(self):
        self.cfg.shallow_bucket = list(self.default_bucket)

    def use_truthfulqa_bucket(self):
        self.cfg.shallow_bucket = list(self.truthfulqa_bucket)

    def set_trisla_config(self, window_size, lambda_current, lambda_window, lambda_prefix):
        self.cfg.trisla_window = int(window_size)
        self.cfg.trisla_lambda_current = float(lambda_current)
        self.cfg.trisla_lambda_window = float(lambda_window)
        self.cfg.trisla_lambda_prefix = float(lambda_prefix)

    def init_tqla_state(self):
        return {
            "selected_layer": self.cfg.shallow_bucket[0],
            "step": 0,
            "window_history": {layer_idx: [] for layer_idx in self.cfg.shallow_bucket},
            "prefix_sums": {layer_idx: 0.0 for layer_idx in self.cfg.shallow_bucket},
            "prefix_counts": {layer_idx: 0 for layer_idx in self.cfg.shallow_bucket},
            "fallback_state": self.init_dyndola_state(),
        }

    def init_soft_decay_state(self):
        return {
            "selected_layer": self.cfg.shallow_bucket[0],
            "step": 0,
            "alpha": self.soft_decay_alpha_base,
        }

    def init_pressure_linear_state(self):
        return {
            "selected_layer": self.cfg.shallow_bucket[0],
            "step": 0,
            "pressure": 0.0,
        }

    def init_calibration_gate_state(self):
        return {
            "step": 0,
            "gate_active": 0,
            "hold_remaining": 0,
            "confidence_ema": None,
            "base_state": self.init_dyndola_state(),
        }

    def init_jacobi_state(self):
        return {"block_index": 0}

    def init_jaca_state(self):
        return {"block_index": 0}

    @staticmethod
    def common_prefix_length(left_tokens, right_tokens):
        left_list = left_tokens[0].tolist()
        right_list = right_tokens[0].tolist()
        prefix_len = 0
        for left_token, right_token in zip(left_list, right_list):
            if int(left_token) != int(right_token):
                break
            prefix_len += 1
        return prefix_len

    @staticmethod
    def repeat_last_token_buffer(generated, window_size):
        return generated[:, -1:].repeat(1, window_size)

    @staticmethod
    def logits_entropy(logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return float((-(probs * log_probs).sum(dim=-1)).item())

    @staticmethod
    def top1_confidence(logits):
        probs = F.softmax(logits, dim=-1)
        return float(torch.max(probs, dim=-1).values.item())

    @staticmethod
    def parse_confidence_value(text):
        text = str(text).strip()
        if not text:
            return None
        first_line = text.splitlines()[0].strip()
        candidate_patterns = (
            first_line,
            text,
        )
        for candidate in candidate_patterns:
            exact_match = re.fullmatch(
                r"(?i)(?:confidence\s*[:=]\s*)?([01](?:\.\d+)?)%?",
                candidate.strip(),
            )
            if exact_match:
                value = float(exact_match.group(1))
                if candidate.strip().endswith("%"):
                    value /= 100.0
                if 0.0 <= value <= 1.0:
                    return value
            json_match = re.search(
                r'"confidence"\s*:\s*([01](?:\.\d+)?)',
                candidate,
            )
            if json_match:
                value = float(json_match.group(1))
                if 0.0 <= value <= 1.0:
                    return value
        matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", first_line)
        for match in matches:
            try:
                value = float(match)
            except ValueError:
                continue
            if 0.0 <= value <= 1.0:
                return value
            if 1.0 < value <= 100.0 and "%" in first_line:
                return value / 100.0
        return None

    @staticmethod
    def build_confidence_query_prompt(prompt, candidate_text):
        return (
            "Estimate how likely the candidate answer is to be correct for the question.\n"
            "Do not think aloud.\n\n"
            f"Question:\n{prompt}\n\n"
            f"Candidate answer:\n{candidate_text}\n\n"
            "Reply with exactly one decimal number between 0 and 1 on the first line.\n"
            "Examples:\n0.12\n0.73\n0.98\n"
            "Do not output any words, labels, explanation, or punctuation other than the decimal number.\n"
        )

    def get_confidence_allowed_token_ids(self):
        if self._confidence_allowed_token_ids is not None and self._confidence_first_token_ids is not None:
            return self._confidence_allowed_token_ids, self._confidence_first_token_ids

        allowed_chars = set("0123456789. \n\r\t")
        allowed_token_ids = []
        first_token_ids = []
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        vocab_size = len(self.tokenizer)

        for token_id in range(vocab_size):
            if token_id in special_ids:
                continue
            piece = self.decode_token(token_id)
            if not piece:
                continue
            if any(ch not in allowed_chars for ch in piece):
                continue
            allowed_token_ids.append(int(token_id))
            stripped = piece.lstrip()
            if stripped and stripped[0] in "01":
                first_token_ids.append(int(token_id))

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and eos_token_id not in allowed_token_ids:
            allowed_token_ids.append(int(eos_token_id))

        self._confidence_allowed_token_ids = tuple(sorted(set(allowed_token_ids)))
        self._confidence_first_token_ids = tuple(sorted(set(first_token_ids)))
        return self._confidence_allowed_token_ids, self._confidence_first_token_ids

    def generate_numeric_confidence_query(self, prompt):
        generated = self.prepare_prompt(prompt)
        prompt_length = generated.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        trace = []
        generated_steps = 0
        forward_passes = 0
        allowed_token_ids, first_token_ids = self.get_confidence_allowed_token_ids()
        allowed_tensor = torch.tensor(allowed_token_ids, device=generated.device, dtype=torch.long)
        first_tensor = (
            torch.tensor(first_token_ids, device=generated.device, dtype=torch.long)
            if first_token_ids
            else allowed_tensor
        )
        saw_numeric_digit = False

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for step_idx in range(self.calibration_confidence_max_new_tokens):
            scores, _, trace_row, _ = self.decoder_step_logits("greedy", generated, None)
            extra_forward_passes = 0 if trace_row is None else int(trace_row.get("extra_forward_passes") or 0)
            forward_passes += 1 + extra_forward_passes

            current_allowed = first_tensor if step_idx == 0 else allowed_tensor
            constrained_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
            constrained_scores[:, current_allowed] = scores[:, current_allowed]
            if eos_token_id is not None and saw_numeric_digit:
                constrained_scores[:, eos_token_id] = scores[:, eos_token_id]

            next_token = torch.argmax(constrained_scores, dim=-1, keepdim=True)
            generated_steps += 1
            token_text = self.decode_token(next_token.item())
            saw_numeric_digit = saw_numeric_digit or any(ch.isdigit() for ch in token_text)
            if trace_row is not None:
                row = dict(trace_row)
                row["step"] = len(trace)
                row["token_id"] = int(next_token.item())
                row["token_text"] = token_text
                row["ablation_mode"] = "confidence_numeric_query"
                trace.append(row)
            generated = torch.cat([generated, next_token.to(generated.device)], dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            if saw_numeric_digit and "\n" in token_text and step_idx > 0:
                break

        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time
        return self.decode_continuation(generated, prompt_length), trace, make_runtime_summary(
            elapsed,
            generated_steps,
            forward_passes=forward_passes,
            generated_tokens=generated_steps,
        )

    def query_self_calibrated_confidence(self, prompt, candidate_text):
        confidence_prompt = self.build_confidence_query_prompt(prompt, candidate_text)
        raw_output, trace, runtime_summary = self.generate_numeric_confidence_query(
            confidence_prompt,
        )
        confidence_value = self.parse_confidence_value(raw_output)
        return {
            "confidence": confidence_value,
            "raw_output": str(raw_output).strip(),
            "trace": trace,
            "runtime_summary": runtime_summary,
            "parse_valid": confidence_value is not None,
        }

    def tqla_local_utility(self, logits):
        return float(self.top1_top2_margin(logits) - self.logits_entropy(logits))

    def tqla_fixed_alpha_followup(self, generated, token_id, fallback_state):
        next_token = torch.tensor([[int(token_id)]], device=generated.device)
        next_generated = torch.cat([generated, next_token], dim=-1)
        next_layer_logits, next_final_logits = self.forward_with_layer_logits(next_generated)
        follow_logits, next_state, follow_trace = self.compute_dyndola_step(
            "dyndola_fixed_alpha",
            next_final_logits,
            next_layer_logits,
            fallback_state,
        )
        return {
            "utility": self.tqla_local_utility(follow_logits),
            "selected_layer": int(follow_trace["selected_layer"]),
            "margin": float(self.top1_top2_margin(follow_logits)),
            "entropy": float(self.logits_entropy(follow_logits)),
            "state": next_state,
        }

    def verify_tqla_override(self, generated, baseline_token_id, candidate_token_id, fallback_state):
        baseline_follow = self.tqla_fixed_alpha_followup(generated, baseline_token_id, fallback_state)
        candidate_follow = self.tqla_fixed_alpha_followup(generated, candidate_token_id, fallback_state)
        verified_gain = float(candidate_follow["utility"] - baseline_follow["utility"])
        return {
            "baseline_follow": baseline_follow,
            "candidate_follow": candidate_follow,
            "verified_gain": verified_gain,
            "extra_forward_passes": 2,
        }

    def select_dynamic_layer(self, step, selected_layer, layer_logits, p_final):
        if step % self.cfg.update_every != 0:
            return selected_layer
        best_score = -float("inf")
        for candidate_idx in self.cfg.shallow_bucket:
            if candidate_idx >= len(layer_logits):
                continue
            candidate_logits = layer_logits[candidate_idx]
            candidate_probs = F.softmax(candidate_logits / self.cfg.tau, dim=-1)
            score = self.js_divergence(p_final, candidate_probs)
            if score > best_score:
                best_score = score
                selected_layer = candidate_idx
        return selected_layer

    def compute_instability_terms(self, final_logits, shallow_logits, p_final):
        p_shallow = F.softmax(shallow_logits / self.cfg.tau, dim=-1)
        divergence = self.kl_divergence(p_final, p_shallow)
        margin = self.top1_top2_margin(final_logits)
        instability = divergence - self.cfg.lambda_margin * margin
        normalized_instability = torch.tanh(
            torch.tensor(instability / self.cfg.instability_temperature, dtype=torch.float32)
        ).item()
        return divergence, margin, instability, normalized_instability

    def get_fixed_alpha_for_decoder(self, decoder_name):
        if decoder_name == "dyndola_fixed_alpha_low":
            return self.alpha_switch_low
        if decoder_name == "dyndola_fixed_alpha_high":
            return self.alpha_switch_high
        return self.fixed_alpha_value

    def compute_dyndola_step(self, decoder_name, final_logits, layer_logits, dy_state):
        if dy_state is None:
            dy_state = self.init_dyndola_state()

        step = dy_state["step"]
        selected_layer = dy_state["selected_layer"]
        state_value = dy_state["state"]
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)

        if decoder_name == "dyndola_fixed_layer":
            selected_layer = self.fixed_layer_index
        else:
            selected_layer = self.select_dynamic_layer(step, selected_layer, layer_logits, p_final)

        shallow_logits = layer_logits[selected_layer]
        divergence, margin, instability, normalized_instability = self.compute_instability_terms(
            final_logits, shallow_logits, p_final
        )
        if decoder_name == "dyndola_no_ema":
            state_value = normalized_instability
        else:
            state_value = self.cfg.beta * state_value + (1.0 - self.cfg.beta) * normalized_instability
        if decoder_name in {"dyndola_fixed_alpha", "dyndola_fixed_alpha_low", "dyndola_fixed_alpha_high"}:
            requested_alpha = self.get_fixed_alpha_for_decoder(decoder_name)
            alpha_cap = max(self.cfg.alpha_max, requested_alpha)
            alpha = max(self.cfg.alpha_min, min(alpha_cap, requested_alpha))
        else:
            alpha = torch.sigmoid(torch.tensor(self.cfg.gamma * state_value, dtype=torch.float32)).item()
            alpha = max(self.cfg.alpha_min, min(self.cfg.alpha_max, alpha))
        logits = final_logits - alpha * shallow_logits

        trace_row = {
            "step": step,
            "selected_layer": int(selected_layer),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "normalized_instability": float(normalized_instability),
            "state": float(state_value),
            "alpha": float(alpha),
            "ablation_mode": (
                "no_ema"
                if decoder_name == "dyndola_no_ema"
                else "fixed_alpha"
                if decoder_name == "dyndola_fixed_alpha"
                else "fixed_alpha_low"
                if decoder_name == "dyndola_fixed_alpha_low"
                else "fixed_alpha_high"
                if decoder_name == "dyndola_fixed_alpha_high"
                else "fixed_layer"
                if decoder_name == "dyndola_fixed_layer"
                else "full"
            ),
        }
        next_state = {"selected_layer": selected_layer, "state": state_value, "step": step + 1}
        return logits, next_state, trace_row

    def compute_soft_decay_step(self, final_logits, layer_logits, dy_state):
        if dy_state is None:
            dy_state = self.init_soft_decay_state()

        step = int(dy_state["step"])
        selected_layer = int(dy_state["selected_layer"])
        previous_alpha = float(dy_state["alpha"])
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)
        selected_layer = self.select_dynamic_layer(step, selected_layer, layer_logits, p_final)

        shallow_logits = layer_logits[selected_layer]
        divergence, margin, instability, normalized_instability = self.compute_instability_terms(
            final_logits, shallow_logits, p_final
        )
        risk_triggered = float(normalized_instability >= self.cfg.soft_decay_trigger_threshold)
        if risk_triggered:
            alpha = self.soft_decay_alpha_peak
        else:
            alpha = self.soft_decay_alpha_base + self.soft_decay_decay_rho * (
                previous_alpha - self.soft_decay_alpha_base
            )
        alpha = max(self.soft_decay_alpha_base, min(self.soft_decay_alpha_peak, alpha))
        logits = final_logits - alpha * shallow_logits

        trace_row = {
            "step": step,
            "selected_layer": int(selected_layer),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "normalized_instability": float(normalized_instability),
            "state": None,
            "alpha": float(alpha),
            "ablation_mode": "soft_decay_alpha",
            "risk_triggered": float(risk_triggered),
            "risk_score": float(normalized_instability),
        }
        next_state = {
            "selected_layer": selected_layer,
            "step": step + 1,
            "alpha": alpha,
        }
        return logits, next_state, trace_row

    def compute_pressure_linear_step(self, final_logits, layer_logits, dy_state):
        if dy_state is None:
            dy_state = self.init_pressure_linear_state()

        step = int(dy_state["step"])
        selected_layer = int(dy_state["selected_layer"])
        previous_pressure = float(dy_state["pressure"])
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)
        selected_layer = self.select_dynamic_layer(step, selected_layer, layer_logits, p_final)

        shallow_logits = layer_logits[selected_layer]
        divergence, margin, instability, normalized_instability = self.compute_instability_terms(
            final_logits, shallow_logits, p_final
        )
        risk_score = max(0.0, min(1.0, float(normalized_instability)))
        pressure = self.cfg.pressure_alpha_momentum * previous_pressure + (
            1.0 - self.cfg.pressure_alpha_momentum
        ) * risk_score
        pressure = max(0.0, min(1.0, pressure))
        alpha = self.pressure_alpha_base + (self.pressure_alpha_peak - self.pressure_alpha_base) * pressure
        alpha = max(self.pressure_alpha_base, min(self.pressure_alpha_peak, alpha))
        logits = final_logits - alpha * shallow_logits

        trace_row = {
            "step": step,
            "selected_layer": int(selected_layer),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "normalized_instability": float(normalized_instability),
            "state": float(pressure),
            "alpha": float(alpha),
            "ablation_mode": "pressure_linear_alpha",
            "risk_triggered": None,
            "risk_score": float(risk_score),
        }
        next_state = {
            "selected_layer": selected_layer,
            "step": step + 1,
            "pressure": pressure,
        }
        return logits, next_state, trace_row

    def prepare_prompt(self, prompt, add_generation_prompt=True):
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            if isinstance(prompt, (list, tuple)):
                messages = list(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]
            tokenized_output = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
            )
            input_ids = tokenized_output if isinstance(tokenized_output, torch.Tensor) else tokenized_output.input_ids
        else:
            if isinstance(prompt, (list, tuple)):
                prompt = "\n\n".join(
                    str(message.get("content", "")).strip()
                    for message in prompt
                    if str(message.get("content", "")).strip()
                )
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.to(self.model_input_device)

    def forward_with_layer_logits(self, input_ids):
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        base_model = get_base_model(self.model)
        lm_head = self.model.get_output_embeddings()
        layer_logits = []
        for hidden_state in outputs.hidden_states[1:]:
            normalized = apply_final_norm(base_model, hidden_state)
            logits = lm_head(normalized[:, -1, :]).float()
            layer_logits.append(logits)
        final_logits = outputs.logits[:, -1, :].float()
        return layer_logits, final_logits

    def forward_with_window_layer_logits(self, input_ids, window_size):
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        base_model = get_base_model(self.model)
        lm_head = self.model.get_output_embeddings()
        window_slice = slice(-(window_size + 1), -1)
        layer_logits = []
        for hidden_state in outputs.hidden_states[1:]:
            normalized = apply_final_norm(base_model, hidden_state)
            logits = lm_head(normalized[:, window_slice, :]).float()
            layer_logits.append(logits)
        final_logits = outputs.logits[:, window_slice, :].float()
        return layer_logits, final_logits

    @staticmethod
    def top1_top2_margin(logits):
        top2 = torch.topk(logits, k=2, dim=-1).values[0]
        return (top2[0] - top2[1]).item()

    @staticmethod
    def js_divergence(p, q, eps=1e-8):
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=-1)
        kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=-1)
        return (0.5 * (kl_pm + kl_qm)).item()

    @staticmethod
    def kl_divergence(p, q, eps=1e-8):
        return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1).item()

    def select_dynamic_layers_for_window(self, final_logits, layer_logits):
        window_size = int(final_logits.shape[1])
        selected_layers = []
        jsd_scores = []
        final_probs = F.softmax(final_logits / self.cfg.tau, dim=-1)
        for position_idx in range(window_size):
            p_final = final_probs[:, position_idx, :]
            best_score = -float("inf")
            selected_layer = self.cfg.shallow_bucket[0]
            for candidate_idx in self.cfg.shallow_bucket:
                if candidate_idx >= len(layer_logits):
                    continue
                candidate_logits = layer_logits[candidate_idx][:, position_idx, :]
                candidate_probs = F.softmax(candidate_logits / self.cfg.tau, dim=-1)
                score = self.js_divergence(p_final, candidate_probs)
                if score > best_score:
                    best_score = score
                    selected_layer = candidate_idx
            selected_layers.append(int(selected_layer))
            jsd_scores.append(float(best_score))
        return selected_layers, jsd_scores

    def run_jacobi_block(self, generated, window_size):
        window_size = int(window_size)
        alpha = max(self.cfg.alpha_min, min(self.cfg.alpha_max, self.fixed_alpha_value))
        buffer = self.repeat_last_token_buffer(generated, window_size)
        previous_buffer = buffer.clone()
        final_rows = []
        first_scores = None
        converged = False
        passes_used = 0
        agreement_prefix_len = 0

        for iteration_idx in range(self.jacobi_max_iters):
            input_ids = torch.cat([generated, buffer], dim=-1)
            layer_logits, final_logits = self.forward_with_window_layer_logits(input_ids, window_size)
            selected_layers, jsd_scores = self.select_dynamic_layers_for_window(final_logits, layer_logits)
            next_tokens = []
            current_rows = []
            current_first_scores = None

            for position_idx, (selected_layer, jsd_score) in enumerate(zip(selected_layers, jsd_scores)):
                final_logits_pos = final_logits[:, position_idx, :]
                p_final = F.softmax(final_logits_pos / self.cfg.tau, dim=-1)
                shallow_logits = layer_logits[selected_layer][:, position_idx, :]
                divergence, margin, instability, normalized_instability = self.compute_instability_terms(
                    final_logits_pos,
                    shallow_logits,
                    p_final,
                )
                contrast_logits = final_logits_pos - alpha * shallow_logits
                next_token = torch.argmax(contrast_logits, dim=-1)
                if position_idx == 0:
                    current_first_scores = contrast_logits
                next_tokens.append(next_token)
                current_rows.append(
                    {
                        "step": None,
                        "selected_layer": int(selected_layer),
                        "divergence": float(divergence),
                        "margin": float(margin),
                        "instability": float(instability),
                        "normalized_instability": float(normalized_instability),
                        "state": None,
                        "alpha": float(alpha),
                        "ablation_mode": "fixed_jacobi",
                        "risk_triggered": None,
                        "risk_score": None,
                        "jsd_current": float(jsd_score),
                        "jsd_window": None,
                        "jsd_prefix": None,
                        "selection_margin": None,
                        "history_window_signal": None,
                        "history_prefix_signal": None,
                        "history_credit": None,
                        "selection_score": float(jsd_score),
                        "fallback_used": None,
                        "baseline_margin": None,
                        "jacobi_position": int(position_idx),
                        "jacobi_window_size": int(window_size),
                        "jacobi_pass_index": int(iteration_idx),
                    }
                )

            new_buffer = torch.stack(next_tokens, dim=1)
            passes_used = iteration_idx + 1
            final_rows = current_rows
            first_scores = current_first_scores
            previous_buffer = buffer
            converged = torch.equal(new_buffer, buffer)
            buffer = new_buffer
            if converged:
                break

        stable_prefix_len = window_size if converged else self.common_prefix_length(previous_buffer, buffer)
        commit_len = stable_prefix_len if stable_prefix_len > 0 else 1
        for row in final_rows:
            row["jacobi_passes_used"] = int(passes_used)
            row["jacobi_converged"] = float(converged)
            row["jacobi_stable_prefix_len"] = int(stable_prefix_len)
            row["jacobi_commit_len"] = int(commit_len)

        return {
            "buffer": buffer,
            "first_scores": first_scores,
            "position_rows": final_rows,
            "forward_passes": int(passes_used),
            "converged": bool(converged),
            "stable_prefix_len": int(stable_prefix_len),
            "commit_len": int(commit_len),
        }

    def run_jaca_block(self, generated, window_size):
        window_size = int(window_size)
        low_alpha = max(self.cfg.alpha_min, min(max(self.cfg.alpha_max, self.alpha_switch_low), self.alpha_switch_low))
        high_alpha = max(
            self.cfg.alpha_min,
            min(max(self.cfg.alpha_max, self.alpha_switch_high), self.alpha_switch_high),
        )
        buffer = self.repeat_last_token_buffer(generated, window_size)
        previous_buffer = buffer.clone()
        final_rows = []
        first_scores = None
        converged = False
        passes_used = 0

        for iteration_idx in range(self.jacobi_max_iters):
            input_ids = torch.cat([generated, buffer], dim=-1)
            layer_logits, final_logits = self.forward_with_window_layer_logits(input_ids, window_size)
            selected_layers, jsd_scores = self.select_dynamic_layers_for_window(final_logits, layer_logits)

            candidate_rows = []
            first_divergence_idx = None
            for position_idx, (selected_layer, jsd_score) in enumerate(zip(selected_layers, jsd_scores)):
                final_logits_pos = final_logits[:, position_idx, :]
                p_final = F.softmax(final_logits_pos / self.cfg.tau, dim=-1)
                shallow_logits = layer_logits[selected_layer][:, position_idx, :]
                divergence, margin, instability, normalized_instability = self.compute_instability_terms(
                    final_logits_pos,
                    shallow_logits,
                    p_final,
                )
                low_scores = final_logits_pos - low_alpha * shallow_logits
                high_scores = final_logits_pos - high_alpha * shallow_logits
                low_token = torch.argmax(low_scores, dim=-1)
                high_token = torch.argmax(high_scores, dim=-1)
                safe_confidence = self.top1_confidence(low_scores)
                truth_confidence = self.top1_confidence(high_scores)
                token_mismatch = int(int(low_token.item()) != int(high_token.item()))
                if self.jaca_early_agreement_shortcut and not token_mismatch:
                    regime_jsd = 0.0
                    disagreement = 0
                else:
                    low_probs = F.softmax(low_scores / self.cfg.tau, dim=-1)
                    high_probs = F.softmax(high_scores / self.cfg.tau, dim=-1)
                    regime_jsd = self.js_divergence(low_probs, high_probs)
                    disagreement = int(token_mismatch and regime_jsd >= float(self.jaca_divergence_threshold))
                if disagreement and first_divergence_idx is None:
                    first_divergence_idx = int(position_idx)
                candidate_rows.append(
                    {
                        "position_idx": int(position_idx),
                        "selected_layer": int(selected_layer),
                        "divergence": float(divergence),
                        "margin": float(margin),
                        "instability": float(instability),
                        "normalized_instability": float(normalized_instability),
                        "jsd_current": float(jsd_score),
                        "low_scores": low_scores,
                        "high_scores": high_scores,
                        "low_token": low_token,
                        "high_token": high_token,
                        "jaca_token_mismatch": float(token_mismatch),
                        "jaca_divergence": float(regime_jsd),
                        "jaca_safe_confidence": float(safe_confidence),
                        "jaca_truth_confidence": float(truth_confidence),
                        "jaca_disagreement": float(disagreement),
                    }
                )

            agreement_prefix_len = 0
            for row in candidate_rows:
                if int(row["jaca_disagreement"]) != 0:
                    break
                agreement_prefix_len += 1

            next_tokens = []
            current_rows = []
            current_first_scores = None
            for row in candidate_rows:
                position_idx = int(row["position_idx"])
                arbitration_active = first_divergence_idx is not None and position_idx >= first_divergence_idx
                use_truth = arbitration_active and (
                    float(row["jaca_truth_confidence"])
                    > float(row["jaca_safe_confidence"]) - float(self.jaca_truth_bias)
                )
                selected_scores = row["high_scores"] if use_truth else row["low_scores"]
                selected_token = row["high_token"] if use_truth else row["low_token"]
                if position_idx == 0:
                    current_first_scores = selected_scores
                next_tokens.append(selected_token)
                current_rows.append(
                    {
                        "step": None,
                        "selected_layer": int(row["selected_layer"]),
                        "divergence": float(row["divergence"]),
                        "margin": float(row["margin"]),
                        "instability": float(row["instability"]),
                        "normalized_instability": float(row["normalized_instability"]),
                        "state": None,
                        "alpha": float(high_alpha if use_truth else low_alpha),
                        "ablation_mode": "jaca",
                        "risk_triggered": float(row["jaca_disagreement"]),
                        "risk_score": float(row["jaca_divergence"]),
                        "jsd_current": float(row["jsd_current"]),
                        "jsd_window": None,
                        "jsd_prefix": None,
                        "selection_margin": float(
                            float(row["jaca_truth_confidence"]) - float(row["jaca_safe_confidence"])
                        ),
                        "history_window_signal": None,
                        "history_prefix_signal": None,
                        "history_credit": None,
                        "selection_score": float(
                            row["jaca_truth_confidence"] if use_truth else row["jaca_safe_confidence"]
                        ),
                        "fallback_used": float(not use_truth),
                        "baseline_margin": float(row["margin"]),
                        "jacobi_position": int(position_idx),
                        "jacobi_window_size": int(window_size),
                        "jacobi_pass_index": int(iteration_idx),
                        "jaca_divergence": float(row["jaca_divergence"]),
                        "jaca_safe_confidence": float(row["jaca_safe_confidence"]),
                        "jaca_truth_confidence": float(row["jaca_truth_confidence"]),
                        "jaca_token_mismatch": float(row["jaca_token_mismatch"]),
                        "jaca_disagreement": float(row["jaca_disagreement"]),
                        "jaca_selected_truth": float(use_truth),
                        "jaca_arbitration_active": float(arbitration_active),
                        "jaca_first_divergence_position": (
                            int(first_divergence_idx) if first_divergence_idx is not None else None
                        ),
                        "jaca_agreement_prefix_len": int(agreement_prefix_len),
                    }
                )

            new_buffer = torch.stack(next_tokens, dim=1)
            passes_used = iteration_idx + 1
            final_rows = current_rows
            first_scores = current_first_scores
            previous_buffer = buffer
            converged = torch.equal(new_buffer, buffer)
            buffer = new_buffer
            if converged:
                break

        stable_prefix_len = window_size if converged else self.common_prefix_length(previous_buffer, buffer)
        if self.jaca_early_agreement_shortcut:
            commit_len = max(
                stable_prefix_len if stable_prefix_len > 0 else 1,
                agreement_prefix_len if agreement_prefix_len > 0 else 1,
            )
        else:
            commit_len = stable_prefix_len if stable_prefix_len > 0 else 1
        for row in final_rows:
            row["jacobi_passes_used"] = int(passes_used)
            row["jacobi_converged"] = float(converged)
            row["jacobi_stable_prefix_len"] = int(stable_prefix_len)
            row["jacobi_commit_len"] = int(commit_len)

        return {
            "buffer": buffer,
            "first_scores": first_scores,
            "position_rows": final_rows,
            "forward_passes": int(passes_used),
            "converged": bool(converged),
            "stable_prefix_len": int(stable_prefix_len),
            "commit_len": int(commit_len),
        }

    @staticmethod
    def get_relative_top_filter(scores, relative_top=0.1, min_tokens_to_keep=1):
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, _ = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + math.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    @staticmethod
    def official_dola_js_divergence(mature_logits, premature_logits):
        softmax_mature_layer = F.softmax(mature_logits, dim=-1)
        softmax_premature_layer = F.softmax(premature_logits, dim=-1)
        average_distribution = 0.5 * (softmax_mature_layer + softmax_premature_layer)
        log_softmax_mature_layer = F.log_softmax(mature_logits, dim=-1)
        log_softmax_premature_layer = F.log_softmax(premature_logits, dim=-1)
        kl1 = F.kl_div(log_softmax_mature_layer, average_distribution, reduction="none").mean(-1)
        kl2 = F.kl_div(log_softmax_premature_layer, average_distribution, reduction="none").mean(-1)
        return float((0.5 * (kl1 + kl2)).mean().item())

    def build_official_dola_scores(self, final_logits, layer_logits):
        candidate_metrics = []
        for candidate_idx in self.cfg.shallow_bucket:
            if candidate_idx >= len(layer_logits) or candidate_idx == self.mature_layer_index:
                continue
            candidate_logits = layer_logits[candidate_idx]
            candidate_score = self.official_dola_js_divergence(final_logits, candidate_logits)
            candidate_metrics.append(
                {
                    "layer": int(candidate_idx),
                    "jsd_current": float(candidate_score),
                }
            )
        if not candidate_metrics:
            raise ValueError("DoLa could not score any candidate layers in the current shallow bucket.")

        best_candidate = max(candidate_metrics, key=lambda row: row["jsd_current"])
        selected_layer = int(best_candidate["layer"])
        premature_logits = layer_logits[selected_layer]
        mature_log_probs = F.log_softmax(final_logits, dim=-1)
        premature_log_probs = F.log_softmax(premature_logits, dim=-1)
        contrast_scores = mature_log_probs - premature_log_probs
        contrast_scores = F.log_softmax(contrast_scores, dim=-1)
        if self.dola_relative_top > 0.0:
            relative_top_mask = self.get_relative_top_filter(mature_log_probs, self.dola_relative_top)
            contrast_scores = torch.where(
                relative_top_mask,
                torch.full_like(contrast_scores, self.dola_relative_top_value),
                contrast_scores,
            )
        trace_row = {
            "step": None,
            "selected_layer": selected_layer,
            "divergence": float(best_candidate["jsd_current"]),
            "margin": None,
            "instability": None,
            "normalized_instability": None,
            "state": None,
            "alpha": None,
            "ablation_mode": "official_dola",
            "jsd_current": float(best_candidate["jsd_current"]),
            "jsd_window": None,
            "jsd_prefix": None,
            "selection_score": float(best_candidate["jsd_current"]),
        }
        return contrast_scores, trace_row

    def decode_token(self, token_id):
        return self.tokenizer.decode([int(token_id)], skip_special_tokens=False)

    def decode_continuation(self, full_sequence, prompt_length):
        continuation = full_sequence[0, prompt_length:]
        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()

    @staticmethod
    def synchronize_cuda():
        if not torch.cuda.is_available():
            return
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_idx)

    def init_dyndola_state(self):
        return {"selected_layer": self.cfg.shallow_bucket[0], "state": 0.0, "step": 0}

    def init_trisla_state(self):
        return {
            "selected_layer": self.cfg.shallow_bucket[0],
            "step": 0,
            "window_history": {layer_idx: [] for layer_idx in self.cfg.shallow_bucket},
            "prefix_sums": {layer_idx: 0.0 for layer_idx in self.cfg.shallow_bucket},
            "prefix_counts": {layer_idx: 0 for layer_idx in self.cfg.shallow_bucket},
        }

    def decoder_step_logits(self, decoder_name, generated, dy_state=None):
        if decoder_name == "greedy":
            logits = self.model(input_ids=generated, use_cache=False).logits[:, -1, :].float()
            return logits, dy_state, None, False

        layer_logits, final_logits = self.forward_with_layer_logits(generated)

        if decoder_name == "dola":
            contrast_scores, trace_row = self.build_official_dola_scores(final_logits, layer_logits)
            return contrast_scores, dy_state, trace_row, True

        if decoder_name == "soft_decay_alpha":
            logits, next_state, trace_row = self.compute_soft_decay_step(final_logits, layer_logits, dy_state)
            return logits, next_state, trace_row, False

        if decoder_name == "pressure_linear_alpha":
            logits, next_state, trace_row = self.compute_pressure_linear_step(final_logits, layer_logits, dy_state)
            return logits, next_state, trace_row, False

        if decoder_name == "calibration_gated_fixed_alpha":
            if dy_state is None:
                dy_state = self.init_calibration_gate_state()

            base_logits = final_logits
            base_margin = self.top1_top2_margin(base_logits)
            base_confidence = self.top1_confidence(base_logits)
            fixed_alpha_logits, next_base_state, fixed_alpha_trace = self.compute_dyndola_step(
                "dyndola_fixed_alpha",
                final_logits,
                layer_logits,
                dy_state.get("base_state"),
            )

            previous_gate_active = bool(dy_state.get("gate_active", 0))
            previous_hold_remaining = int(dy_state.get("hold_remaining", 0))
            previous_confidence_ema = dy_state.get("confidence_ema")
            confidence_ema = (
                float(base_confidence)
                if previous_confidence_ema is None
                else self.cfg.calibration_gate_ema_decay * float(previous_confidence_ema)
                + (1.0 - self.cfg.calibration_gate_ema_decay) * float(base_confidence)
            )
            confidence_triggered = float(confidence_ema < self.cfg.calibration_gate_confidence_on)
            margin_triggered = float(base_margin < self.cfg.calibration_gate_margin_on)

            gate_active = previous_gate_active
            hold_remaining = previous_hold_remaining
            gate_switched_on = 0.0
            gate_switched_off = 0.0

            if not previous_gate_active:
                if confidence_triggered and margin_triggered:
                    gate_active = True
                    hold_remaining = int(self.cfg.calibration_gate_hold_steps)
                    gate_switched_on = 1.0
            else:
                if hold_remaining > 0:
                    hold_remaining -= 1
                elif confidence_ema > self.cfg.calibration_gate_confidence_off:
                    gate_active = False
                    hold_remaining = 0
                    gate_switched_off = 1.0

            if gate_active:
                logits = fixed_alpha_logits
                selected_layer = int(fixed_alpha_trace["selected_layer"])
                applied_alpha = float(fixed_alpha_trace["alpha"])
                fallback_used = 0.0
                ablation_mode = "calibration_gated_fixed_alpha_on"
            else:
                logits = base_logits
                selected_layer = int(fixed_alpha_trace["selected_layer"])
                applied_alpha = 0.0
                fallback_used = 1.0
                ablation_mode = "calibration_gated_fixed_alpha_off"

            trace_row = {
                "step": int(dy_state.get("step", 0)),
                "selected_layer": selected_layer,
                "divergence": float(fixed_alpha_trace["divergence"]),
                "margin": float(base_margin),
                "instability": float(fixed_alpha_trace["instability"]),
                "normalized_instability": float(fixed_alpha_trace["normalized_instability"]),
                "state": None,
                "alpha": float(applied_alpha),
                "ablation_mode": ablation_mode,
                "fallback_used": float(fallback_used),
                "baseline_margin": float(base_margin),
                "calibration_confidence": float(base_confidence),
                "calibration_confidence_ema": float(confidence_ema),
                "gate_active": float(gate_active),
                "confidence_triggered": float(confidence_triggered),
                "margin_triggered": float(margin_triggered),
                "gate_switch_on": float(gate_switched_on),
                "gate_switch_off": float(gate_switched_off),
                "gate_hold_remaining": int(hold_remaining),
            }
            next_state = {
                "step": int(dy_state.get("step", 0)) + 1,
                "gate_active": int(gate_active),
                "hold_remaining": int(hold_remaining),
                "confidence_ema": float(confidence_ema),
                "base_state": next_base_state,
            }
            return logits, next_state, trace_row, False

        if decoder_name == "trisla":
            if dy_state is None:
                dy_state = self.init_trisla_state()

            step = dy_state["step"]
            p_final = F.softmax(final_logits / self.cfg.trisla_jsd_temperature, dim=-1)
            window_history = {
                int(layer_idx): list(values) for layer_idx, values in dy_state["window_history"].items()
            }
            prefix_sums = {int(layer_idx): float(value) for layer_idx, value in dy_state["prefix_sums"].items()}
            prefix_counts = {int(layer_idx): int(value) for layer_idx, value in dy_state["prefix_counts"].items()}

            candidate_metrics = []
            for candidate_idx in self.cfg.shallow_bucket:
                if candidate_idx >= len(layer_logits):
                    continue
                candidate_logits = layer_logits[candidate_idx]
                candidate_probs = F.softmax(candidate_logits / self.cfg.trisla_jsd_temperature, dim=-1)
                jsd_current = self.js_divergence(p_final, candidate_probs)
                prior_window = window_history.get(candidate_idx, [])
                history_window_signal = (
                    statistics.mean(prior_window[-self.cfg.trisla_window :]) if prior_window else 0.0
                )
                prefix_count = prefix_counts.get(candidate_idx, 0)
                history_prefix_signal = prefix_sums.get(candidate_idx, 0.0) / prefix_count if prefix_count > 0 else 0.0
                candidate_metrics.append(
                    {
                        "layer": int(candidate_idx),
                        "jsd_current": float(jsd_current),
                        "history_window_signal": float(history_window_signal),
                        "history_prefix_signal": float(history_prefix_signal),
                    }
                )

            if not candidate_metrics:
                raise ValueError("TriSLA could not score any candidate layers in the current shallow bucket.")

            if len(candidate_metrics) == 1:
                candidate_metrics[0]["selection_margin"] = float(candidate_metrics[0]["jsd_current"])
            else:
                jsd_by_layer = {row["layer"]: row["jsd_current"] for row in candidate_metrics}
                for row in candidate_metrics:
                    best_other_jsd = max(
                        value for layer_idx, value in jsd_by_layer.items() if layer_idx != row["layer"]
                    )
                    row["selection_margin"] = float(row["jsd_current"] - best_other_jsd)

            for row in candidate_metrics:
                arbitration_score = (
                    self.cfg.trisla_lambda_current * row["selection_margin"]
                    + self.cfg.trisla_lambda_window * row["history_window_signal"]
                    + self.cfg.trisla_lambda_prefix * row["history_prefix_signal"]
                )
                row["arbitration_score"] = float(arbitration_score)

            candidate_by_layer = {row["layer"]: row for row in candidate_metrics}
            best_candidate = max(candidate_metrics, key=lambda row: row["arbitration_score"])
            incumbent_layer = int(dy_state.get("selected_layer", self.cfg.shallow_bucket[0]))
            incumbent_candidate = candidate_by_layer.get(incumbent_layer)
            switch_gap = None
            switch_blocked = 0.0
            if incumbent_candidate is not None and int(best_candidate["layer"]) != incumbent_layer:
                switch_gap = float(best_candidate["arbitration_score"] - incumbent_candidate["arbitration_score"])
                if switch_gap < float(self.cfg.trisla_switch_margin):
                    best_candidate = incumbent_candidate
                    switch_blocked = 1.0
            selected_layer = int(best_candidate["layer"])
            shallow_logits = layer_logits[selected_layer]
            p_shallow = F.softmax(shallow_logits / self.cfg.tau, dim=-1)
            divergence = self.kl_divergence(p_final, p_shallow)
            margin = self.top1_top2_margin(final_logits)
            instability = divergence - self.cfg.lambda_margin * margin
            normalized_instability = torch.tanh(
                torch.tensor(instability / self.cfg.instability_temperature, dtype=torch.float32)
            ).item()
            alpha = max(self.cfg.alpha_min, min(self.cfg.alpha_max, self.fixed_alpha_value))
            logits = final_logits - alpha * shallow_logits

            next_window_history = {
                int(layer_idx): list(values) for layer_idx, values in window_history.items()
            }
            next_prefix_sums = dict(prefix_sums)
            next_prefix_counts = dict(prefix_counts)
            history_credit = max(
                0.0,
                float(best_candidate["selection_margin"]) - float(self.cfg.trisla_margin_epsilon),
            )
            if history_credit > 0.0:
                selected_history = next_window_history.setdefault(selected_layer, [])
                selected_history.append(history_credit)
                if len(selected_history) > self.cfg.trisla_window:
                    selected_history.pop(0)
                next_prefix_sums[selected_layer] = next_prefix_sums.get(selected_layer, 0.0) + history_credit
                next_prefix_counts[selected_layer] = next_prefix_counts.get(selected_layer, 0) + 1

            trace_row = {
                "step": step,
                "selected_layer": selected_layer,
                "divergence": float(divergence),
                "margin": float(margin),
                "instability": float(instability),
                "normalized_instability": float(normalized_instability),
                "state": None,
                "alpha": float(alpha),
                "ablation_mode": "trisla",
                "jsd_current": float(best_candidate["jsd_current"]),
                "jsd_window": float(best_candidate["history_window_signal"]),
                "jsd_prefix": float(best_candidate["history_prefix_signal"]),
                "selection_margin": float(best_candidate["selection_margin"]),
                "history_window_signal": float(best_candidate["history_window_signal"]),
                "history_prefix_signal": float(best_candidate["history_prefix_signal"]),
                "history_credit": float(history_credit),
                "selection_score": float(best_candidate["arbitration_score"]),
                "switch_blocked": float(switch_blocked),
                "switch_gap": float(switch_gap) if switch_gap is not None else None,
            }
            next_state = {
                "selected_layer": selected_layer,
                "step": step + 1,
                "window_history": next_window_history,
                "prefix_sums": next_prefix_sums,
                "prefix_counts": next_prefix_counts,
            }
            return logits, next_state, trace_row, False

        if decoder_name == "tqla":
            if dy_state is None:
                dy_state = self.init_tqla_state()

            fallback_logits, next_fallback_state, fallback_trace = self.compute_dyndola_step(
                "dyndola_fixed_alpha",
                final_logits,
                layer_logits,
                dy_state.get("fallback_state"),
            )
            step = int(dy_state["step"])
            alpha = float(fallback_trace["alpha"])
            baseline_selected_layer = int(fallback_trace["selected_layer"])
            baseline_token_id = int(torch.argmax(fallback_logits, dim=-1).item())
            baseline_margin = self.top1_top2_margin(fallback_logits)
            baseline_entropy = self.logits_entropy(fallback_logits)
            baseline_guarded = float(
                self.cfg.tqla_baseline_margin_threshold is not None
                and baseline_margin >= float(self.cfg.tqla_baseline_margin_threshold)
            )
            window_history = {
                int(layer_idx): list(values) for layer_idx, values in dy_state["window_history"].items()
            }
            prefix_sums = {int(layer_idx): float(value) for layer_idx, value in dy_state["prefix_sums"].items()}
            prefix_counts = {int(layer_idx): int(value) for layer_idx, value in dy_state["prefix_counts"].items()}

            candidate_metrics = []
            for candidate_idx in self.cfg.shallow_bucket:
                if candidate_idx >= len(layer_logits):
                    continue
                candidate_logits = layer_logits[candidate_idx]
                contrast_logits = final_logits - alpha * candidate_logits
                contrast_margin = self.top1_top2_margin(contrast_logits)
                contrast_entropy = self.logits_entropy(contrast_logits)
                margin_gain = contrast_margin - baseline_margin
                entropy_gain = baseline_entropy - contrast_entropy
                utility_current = float(margin_gain + entropy_gain)
                candidate_token_id = int(torch.argmax(contrast_logits, dim=-1).item())
                prior_window = window_history.get(candidate_idx, [])
                history_window_signal = (
                    statistics.mean(prior_window[-self.cfg.tqla_window :]) if prior_window else 0.0
                )
                prefix_count = prefix_counts.get(candidate_idx, 0)
                history_prefix_signal = prefix_sums.get(candidate_idx, 0.0) / prefix_count if prefix_count > 0 else 0.0
                positive_signals_ok = (
                    margin_gain > 0.0 and entropy_gain > 0.0
                ) if self.cfg.tqla_require_positive_signals else True
                layer_penalty = float(
                    self.cfg.tqla_layer_deviation_penalty * abs(int(candidate_idx) - baseline_selected_layer)
                )
                flip_penalty = float(
                    self.cfg.tqla_flip_penalty if candidate_token_id != baseline_token_id else 0.0
                )
                arbitration_score = (
                    self.cfg.tqla_lambda_current * utility_current
                    + self.cfg.tqla_lambda_window * history_window_signal
                    + self.cfg.tqla_lambda_prefix * history_prefix_signal
                    - layer_penalty
                    - flip_penalty
                )
                candidate_metrics.append(
                    {
                        "layer": int(candidate_idx),
                        "contrast_logits": contrast_logits,
                        "utility_current": utility_current,
                        "margin_gain": float(margin_gain),
                        "entropy_gain": float(entropy_gain),
                        "history_window_signal": float(history_window_signal),
                        "history_prefix_signal": float(history_prefix_signal),
                        "candidate_token_id": candidate_token_id,
                        "positive_signals_ok": bool(positive_signals_ok),
                        "layer_penalty": layer_penalty,
                        "flip_penalty": flip_penalty,
                        "arbitration_score": float(arbitration_score),
                    }
                )

            if not candidate_metrics:
                raise ValueError("TqLA could not score any candidate layers in the current shallow bucket.")

            best_candidate = max(candidate_metrics, key=lambda row: row["arbitration_score"])
            eligible_candidates = [row for row in candidate_metrics if row["positive_signals_ok"]]
            best_eligible_candidate = (
                max(eligible_candidates, key=lambda row: row["arbitration_score"]) if eligible_candidates else None
            )
            candidate_for_trace = best_eligible_candidate or best_candidate

            next_window_history = {
                int(layer_idx): list(values) for layer_idx, values in window_history.items()
            }
            next_prefix_sums = dict(prefix_sums)
            next_prefix_counts = dict(prefix_counts)
            fallback_used = True
            verification_required = 0.0
            verification_passed = None
            verified_gain = None
            extra_forward_passes = 0

            if baseline_guarded:
                ablation_mode = "tqla_baseline_guard"
            elif best_eligible_candidate is None:
                ablation_mode = "tqla_signal_guard"
            elif float(best_eligible_candidate["arbitration_score"]) < float(self.cfg.tqla_utility_epsilon):
                ablation_mode = "tqla_fallback"
            else:
                verification_required = float(
                    self.cfg.tqla_verify_top1_override
                    and int(best_eligible_candidate["candidate_token_id"]) != baseline_token_id
                )
                if verification_required:
                    verification_result = self.verify_tqla_override(
                        generated,
                        baseline_token_id,
                        int(best_eligible_candidate["candidate_token_id"]),
                        next_fallback_state,
                    )
                    extra_forward_passes = int(verification_result["extra_forward_passes"])
                    verified_gain = float(verification_result["verified_gain"])
                    verification_passed = float(verified_gain > float(self.cfg.tqla_verify_epsilon))
                    fallback_used = verification_passed == 0.0
                    ablation_mode = "tqla_verify_reject" if fallback_used else "tqla_rescue"
                else:
                    fallback_used = False
                    ablation_mode = "tqla_rescue"

            if fallback_used:
                selected_layer = baseline_selected_layer
                logits = fallback_logits
                proposed_layer = int(candidate_for_trace["layer"])
                utility_current = float(candidate_for_trace["utility_current"])
                margin_gain = float(candidate_for_trace["margin_gain"])
                entropy_gain = float(candidate_for_trace["entropy_gain"])
                history_window_signal = float(candidate_for_trace["history_window_signal"])
                history_prefix_signal = float(candidate_for_trace["history_prefix_signal"])
                history_credit = 0.0
                selection_score = float(candidate_for_trace["arbitration_score"])
                layer_penalty = float(candidate_for_trace["layer_penalty"])
                flip_penalty = float(candidate_for_trace["flip_penalty"])
                positive_signals_ok = float(candidate_for_trace["positive_signals_ok"])
                divergence = float(fallback_trace["divergence"])
                margin = float(fallback_trace["margin"])
                instability = float(fallback_trace["instability"])
                normalized_instability = float(fallback_trace["normalized_instability"])
                state_value = None
            else:
                selected_layer = int(best_eligible_candidate["layer"])
                proposed_layer = selected_layer
                logits = best_eligible_candidate["contrast_logits"]
                utility_current = float(best_eligible_candidate["utility_current"])
                margin_gain = float(best_eligible_candidate["margin_gain"])
                entropy_gain = float(best_eligible_candidate["entropy_gain"])
                history_window_signal = float(best_eligible_candidate["history_window_signal"])
                history_prefix_signal = float(best_eligible_candidate["history_prefix_signal"])
                layer_penalty = float(best_eligible_candidate["layer_penalty"])
                flip_penalty = float(best_eligible_candidate["flip_penalty"])
                positive_signals_ok = float(best_eligible_candidate["positive_signals_ok"])
                history_credit_source = float(verified_gain) if verification_required else utility_current
                history_credit_threshold = (
                    float(self.cfg.tqla_verify_epsilon)
                    if verification_required
                    else float(self.cfg.tqla_utility_epsilon)
                )
                history_credit = max(
                    0.0,
                    history_credit_source - history_credit_threshold,
                )
                if history_credit > 0.0:
                    selected_history = next_window_history.setdefault(selected_layer, [])
                    selected_history.append(history_credit)
                    if len(selected_history) > self.cfg.tqla_window:
                        selected_history.pop(0)
                    next_prefix_sums[selected_layer] = next_prefix_sums.get(selected_layer, 0.0) + history_credit
                    next_prefix_counts[selected_layer] = next_prefix_counts.get(selected_layer, 0) + 1
                selection_score = float(best_eligible_candidate["arbitration_score"])
                p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)
                p_shallow = F.softmax(layer_logits[selected_layer] / self.cfg.tau, dim=-1)
                divergence = self.kl_divergence(p_final, p_shallow)
                margin = self.top1_top2_margin(final_logits)
                instability = divergence - self.cfg.lambda_margin * margin
                normalized_instability = torch.tanh(
                    torch.tensor(instability / self.cfg.instability_temperature, dtype=torch.float32)
                ).item()
                state_value = None
                ablation_mode = "tqla_rescue"

            trace_row = {
                "step": step,
                "selected_layer": selected_layer,
                "divergence": float(divergence),
                "margin": float(margin),
                "instability": float(instability),
                "normalized_instability": float(normalized_instability),
                "state": state_value,
                "alpha": float(alpha),
                "ablation_mode": ablation_mode,
                "jsd_current": None,
                "jsd_window": None,
                "jsd_prefix": None,
                "selection_margin": float(utility_current),
                "margin_gain": float(margin_gain),
                "entropy_gain": float(entropy_gain),
                "history_window_signal": float(history_window_signal),
                "history_prefix_signal": float(history_prefix_signal),
                "history_credit": float(history_credit),
                "selection_score": float(selection_score),
                "fallback_used": float(fallback_used),
                "baseline_selected_layer": baseline_selected_layer,
                "baseline_margin": float(baseline_margin),
                "proposed_layer": int(proposed_layer),
                "baseline_guarded": float(baseline_guarded),
                "positive_signals_ok": float(positive_signals_ok),
                "layer_penalty": float(layer_penalty),
                "flip_penalty": float(flip_penalty),
                "verification_required": float(verification_required),
                "verification_passed": verification_passed,
                "verified_gain": verified_gain,
                "extra_forward_passes": int(extra_forward_passes),
            }
            next_state = {
                "selected_layer": selected_layer,
                "step": step + 1,
                "window_history": next_window_history,
                "prefix_sums": next_prefix_sums,
                "prefix_counts": next_prefix_counts,
                "fallback_state": next_fallback_state,
            }
            return logits, next_state, trace_row, False

        if decoder_name not in {
            "dyndola",
            "dyndola_no_ema",
            "dyndola_fixed_alpha",
            "dyndola_fixed_alpha_low",
            "dyndola_fixed_alpha_high",
            "dyndola_fixed_layer",
        }:
            raise ValueError(f"Unknown decoder: {decoder_name}")
        logits, next_state, trace_row = self.compute_dyndola_step(decoder_name, final_logits, layer_logits, dy_state)
        return logits, next_state, trace_row, False

    def generate_with_jacobi_decoder(self, prompt, max_new_tokens=96, stop_on_eos=True):
        generated = self.prepare_prompt(prompt)
        prompt_length = generated.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        trace = []
        generated_steps = 0
        forward_passes = 0
        jacobi_state = self.init_jacobi_state()

        self.synchronize_cuda()
        start_time = time.perf_counter()
        while generated_steps < max_new_tokens:
            remaining_tokens = max_new_tokens - generated_steps
            block_window_size = min(self.jacobi_window_size, remaining_tokens)
            block_result = self.run_jacobi_block(generated, block_window_size)
            forward_passes += int(block_result["forward_passes"])

            commit_len = min(int(block_result["commit_len"]), remaining_tokens)
            commit_tokens = block_result["buffer"][:, :commit_len]
            if stop_on_eos and eos_token_id is not None:
                eos_positions = (commit_tokens[0] == eos_token_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    commit_len = int(eos_positions[0].item()) + 1
                    commit_tokens = commit_tokens[:, :commit_len]

            for position_idx in range(commit_len):
                row = dict(block_result["position_rows"][position_idx])
                row["step"] = len(trace)
                row["token_id"] = int(commit_tokens[0, position_idx].item())
                row["token_text"] = self.decode_token(commit_tokens[0, position_idx].item())
                row["jacobi_block_index"] = int(jacobi_state["block_index"])
                trace.append(row)

            generated = torch.cat([generated, commit_tokens.to(generated.device)], dim=-1)
            generated_steps += commit_len
            jacobi_state["block_index"] += 1

            if stop_on_eos and eos_token_id is not None and commit_tokens[0, -1].item() == eos_token_id:
                break

        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return self.decode_continuation(generated, prompt_length), trace, make_runtime_summary(
            elapsed,
            generated_steps,
            forward_passes=forward_passes,
            generated_tokens=generated_steps,
        )

    def generate_with_jaca_decoder(self, prompt, max_new_tokens=96, stop_on_eos=True):
        generated = self.prepare_prompt(prompt)
        prompt_length = generated.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        trace = []
        generated_steps = 0
        forward_passes = 0
        jaca_state = self.init_jaca_state()

        self.synchronize_cuda()
        start_time = time.perf_counter()
        while generated_steps < max_new_tokens:
            remaining_tokens = max_new_tokens - generated_steps
            block_window_size = min(self.jacobi_window_size, remaining_tokens)
            block_result = self.run_jaca_block(generated, block_window_size)
            forward_passes += int(block_result["forward_passes"])

            commit_len = min(int(block_result["commit_len"]), remaining_tokens)
            commit_tokens = block_result["buffer"][:, :commit_len]
            if stop_on_eos and eos_token_id is not None:
                eos_positions = (commit_tokens[0] == eos_token_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    commit_len = int(eos_positions[0].item()) + 1
                    commit_tokens = commit_tokens[:, :commit_len]

            for position_idx in range(commit_len):
                row = dict(block_result["position_rows"][position_idx])
                row["step"] = len(trace)
                row["token_id"] = int(commit_tokens[0, position_idx].item())
                row["token_text"] = self.decode_token(commit_tokens[0, position_idx].item())
                row["jacobi_block_index"] = int(jaca_state["block_index"])
                trace.append(row)

            generated = torch.cat([generated, commit_tokens.to(generated.device)], dim=-1)
            generated_steps += commit_len
            jaca_state["block_index"] += 1

            if stop_on_eos and eos_token_id is not None and commit_tokens[0, -1].item() == eos_token_id:
                break

        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return self.decode_continuation(generated, prompt_length), trace, make_runtime_summary(
            elapsed,
            generated_steps,
            forward_passes=forward_passes,
            generated_tokens=generated_steps,
        )

    def generate_with_decoder(self, prompt, decoder_name, max_new_tokens=96, stop_on_eos=True):
        if decoder_name == "dola_fixed_jacobi":
            return self.generate_with_jacobi_decoder(prompt, max_new_tokens=max_new_tokens, stop_on_eos=stop_on_eos)
        if decoder_name == "jaca":
            return self.generate_with_jaca_decoder(prompt, max_new_tokens=max_new_tokens, stop_on_eos=stop_on_eos)

        generated = self.prepare_prompt(prompt)
        prompt_length = generated.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        trace = []
        generated_steps = 0
        forward_passes = 0
        dy_state = (
            self.init_dyndola_state()
            if decoder_name
            in {
                "dyndola",
                "dyndola_no_ema",
                "dyndola_fixed_alpha",
                "dyndola_fixed_alpha_low",
                "dyndola_fixed_alpha_high",
                "dyndola_fixed_layer",
            }
            else self.init_soft_decay_state()
            if decoder_name == "soft_decay_alpha"
            else self.init_pressure_linear_state()
            if decoder_name == "pressure_linear_alpha"
            else self.init_calibration_gate_state()
            if decoder_name == "calibration_gated_fixed_alpha"
            else self.init_trisla_state()
            if decoder_name == "trisla"
            else self.init_tqla_state()
            if decoder_name == "tqla"
            else None
        )

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for _ in range(max_new_tokens):
            scores, dy_state, trace_row, _ = self.decoder_step_logits(decoder_name, generated, dy_state)
            extra_forward_passes = 0 if trace_row is None else int(trace_row.get("extra_forward_passes") or 0)
            forward_passes += 1 + extra_forward_passes
            next_token = torch.argmax(scores, dim=-1, keepdim=True)
            generated_steps += 1
            if trace_row is not None:
                row = dict(trace_row)
                row["step"] = len(trace)
                row["token_id"] = int(next_token.item())
                row["token_text"] = self.decode_token(next_token.item())
                trace.append(row)
            generated = torch.cat([generated, next_token.to(generated.device)], dim=-1)
            if stop_on_eos and eos_token_id is not None and next_token.item() == eos_token_id:
                break
        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return self.decode_continuation(generated, prompt_length), trace, make_runtime_summary(
            elapsed,
            generated_steps,
            forward_passes=forward_passes,
            generated_tokens=generated_steps,
        )

    def candidate_to_ids(self, choice_text):
        if not choice_text.startswith((" ", "\n")):
            choice_text = " " + choice_text
        ids = self.tokenizer(choice_text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
        if not ids:
            raise ValueError(f"Empty tokenization for choice: {choice_text!r}")
        return ids

    def score_candidate_with_decoder(self, prompt, decoder_name, choice_text):
        if decoder_name == "dola_fixed_jacobi":
            generated = self.prepare_prompt(prompt)
            total_logprob = 0.0
            token_ids = self.candidate_to_ids(choice_text)
            trace = []
            forward_passes = 0

            self.synchronize_cuda()
            start_time = time.perf_counter()
            for token_idx, token_id in enumerate(token_ids):
                remaining_tokens = len(token_ids) - token_idx
                block_window_size = min(self.jacobi_window_size, remaining_tokens)
                block_result = self.run_jacobi_block(generated, block_window_size)
                forward_passes += int(block_result["forward_passes"])
                logprobs = torch.log_softmax(block_result["first_scores"], dim=-1)
                total_logprob += float(logprobs[0, token_id].item())
                row = dict(block_result["position_rows"][0])
                row["step"] = len(trace)
                row["token_id"] = int(token_id)
                row["token_text"] = self.decode_token(token_id)
                row["jacobi_block_index"] = int(token_idx)
                trace.append(row)
                next_token = torch.tensor([[token_id]], device=generated.device)
                generated = torch.cat([generated, next_token], dim=-1)
            self.synchronize_cuda()
            elapsed = time.perf_counter() - start_time

            return total_logprob, trace, make_runtime_summary(
                elapsed,
                len(token_ids),
                forward_passes=forward_passes,
                generated_tokens=len(token_ids),
            )

        if decoder_name == "jaca":
            generated = self.prepare_prompt(prompt)
            total_logprob = 0.0
            token_ids = self.candidate_to_ids(choice_text)
            trace = []
            forward_passes = 0

            self.synchronize_cuda()
            start_time = time.perf_counter()
            for token_idx, token_id in enumerate(token_ids):
                remaining_tokens = len(token_ids) - token_idx
                block_window_size = min(self.jacobi_window_size, remaining_tokens)
                block_result = self.run_jaca_block(generated, block_window_size)
                forward_passes += int(block_result["forward_passes"])
                logprobs = torch.log_softmax(block_result["first_scores"], dim=-1)
                total_logprob += float(logprobs[0, token_id].item())
                row = dict(block_result["position_rows"][0])
                row["step"] = len(trace)
                row["token_id"] = int(token_id)
                row["token_text"] = self.decode_token(token_id)
                row["jacobi_block_index"] = int(token_idx)
                trace.append(row)
                next_token = torch.tensor([[token_id]], device=generated.device)
                generated = torch.cat([generated, next_token], dim=-1)
            self.synchronize_cuda()
            elapsed = time.perf_counter() - start_time

            return total_logprob, trace, make_runtime_summary(
                elapsed,
                len(token_ids),
                forward_passes=forward_passes,
                generated_tokens=len(token_ids),
            )

        generated = self.prepare_prompt(prompt)
        dy_state = (
            self.init_dyndola_state()
            if decoder_name
            in {
                "dyndola",
                "dyndola_no_ema",
                "dyndola_fixed_alpha",
                "dyndola_fixed_alpha_low",
                "dyndola_fixed_alpha_high",
                "dyndola_fixed_layer",
            }
            else self.init_soft_decay_state()
            if decoder_name == "soft_decay_alpha"
            else self.init_pressure_linear_state()
            if decoder_name == "pressure_linear_alpha"
            else self.init_calibration_gate_state()
            if decoder_name == "calibration_gated_fixed_alpha"
            else self.init_trisla_state()
            if decoder_name == "trisla"
            else self.init_tqla_state()
            if decoder_name == "tqla"
            else None
        )
        total_logprob = 0.0
        token_ids = self.candidate_to_ids(choice_text)
        trace = []
        forward_passes = 0

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for token_id in token_ids:
            scores, dy_state, trace_row, scores_are_logprobs = self.decoder_step_logits(
                decoder_name, generated, dy_state
            )
            extra_forward_passes = 0 if trace_row is None else int(trace_row.get("extra_forward_passes") or 0)
            forward_passes += 1 + extra_forward_passes
            if scores_are_logprobs:
                total_logprob += float(scores[0, token_id].item())
            else:
                logprobs = torch.log_softmax(scores, dim=-1)
                total_logprob += float(logprobs[0, token_id].item())
            next_token = torch.tensor([[token_id]], device=generated.device)
            if trace_row is not None:
                row = dict(trace_row)
                row["step"] = len(trace)
                row["token_id"] = int(token_id)
                row["token_text"] = self.decode_token(token_id)
                trace.append(row)
            generated = torch.cat([generated, next_token], dim=-1)
        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return total_logprob, trace, make_runtime_summary(
            elapsed,
            len(token_ids),
            forward_passes=forward_passes,
            generated_tokens=len(token_ids),
        )


def build_truthfulqa_prompt(question):
    return question + "\nAnswer with one of the provided answer options only."


def build_strategyqa_prompt(question):
    return question + "\nAnswer with exactly one word: yes or no."


def build_gsm8k_prompt(question):
    return question + "\nAnswer with only the final number."


def build_gsm8k_sequence_prompt(question):
    return (
        question
        + "\nSolve this step by step in 2-4 short lines."
        + "\nEnd with exactly: Final answer: <number>"
    )


def build_halueval_prompt(task_name, knowledge, user_input, response_text, label_yes="A", label_no="B"):
    answer_instruction = (
        "Does the answer contain hallucinations?\n"
        f"Reply with exactly one label: {label_yes} = yes, {label_no} = no."
    )
    response_instruction = (
        "Does the response contain hallucinations?\n"
        f"Reply with exactly one label: {label_yes} = yes, {label_no} = no."
    )
    summary_instruction = (
        "Does the summary contain hallucinations?\n"
        f"Reply with exactly one label: {label_yes} = yes, {label_no} = no."
    )
    if task_name == "qa":
        return (
            f"Knowledge: {knowledge}\n"
            f"Question: {user_input}\n"
            f"Answer: {response_text}\n"
            f"{answer_instruction}"
        )
    if task_name == "dialogue":
        return (
            f"Knowledge: {knowledge}\n"
            f"Dialogue history: {user_input}\n"
            f"Response: {response_text}\n"
            f"{response_instruction}"
        )
    if task_name == "summarization":
        return (
            f"Document: {knowledge}\n"
            f"Summary: {response_text}\n"
            f"{summary_instruction}"
        )
    return (
        f"Context: {knowledge}\n"
        f"Prompt: {user_input}\n"
        f"Response: {response_text}\n"
        f"{response_instruction}"
    )


def mean_or_none(values):
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return statistics.mean(numeric_values)


def make_runtime_summary(latency_seconds, decoder_steps, forward_passes=None, generated_tokens=None):
    latency_seconds = float(latency_seconds)
    decoder_steps = int(decoder_steps)
    forward_passes = decoder_steps if forward_passes is None else int(forward_passes)
    generated_tokens = decoder_steps if generated_tokens is None else int(generated_tokens)
    latency_per_step_ms = (1000.0 * latency_seconds / decoder_steps) if decoder_steps > 0 else None
    steps_per_second = (decoder_steps / latency_seconds) if latency_seconds > 0.0 else None
    latency_per_forward_ms = (1000.0 * latency_seconds / forward_passes) if forward_passes > 0 else None
    tokens_per_forward = (generated_tokens / forward_passes) if forward_passes > 0 else None
    return {
        "latency_seconds": latency_seconds,
        "decoder_steps": decoder_steps,
        "forward_passes": forward_passes,
        "generated_tokens": generated_tokens,
        "latency_per_step_ms": latency_per_step_ms,
        "latency_per_forward_ms": latency_per_forward_ms,
        "steps_per_second": steps_per_second,
        "tokens_per_forward": tokens_per_forward,
        "factual_speedup": tokens_per_forward,
    }


def merge_runtime_summaries(runtime_summaries):
    total_latency = 0.0
    total_steps = 0
    total_forward_passes = 0
    total_generated_tokens = 0
    for runtime_summary in runtime_summaries:
        if not runtime_summary:
            continue
        total_latency += float(runtime_summary.get("latency_seconds") or 0.0)
        total_steps += int(runtime_summary.get("decoder_steps") or 0)
        total_forward_passes += int(runtime_summary.get("forward_passes") or 0)
        total_generated_tokens += int(runtime_summary.get("generated_tokens") or 0)
    return make_runtime_summary(
        total_latency,
        total_steps,
        forward_passes=total_forward_passes,
        generated_tokens=total_generated_tokens,
    )


def trisla_sweep_id(window_size, lambda_current, lambda_window, lambda_prefix):
    return (
        f"w{window_size}_"
        f"c{lambda_current:.3f}_"
        f"rw{lambda_window:.3f}_"
        f"p{lambda_prefix:.3f}"
    )


def build_trisla_sweep_configs(args):
    weight_grid = parse_weight_grid(args.trisla_weight_grid)
    window_grid = parse_int_grid(args.trisla_window_grid)
    if weight_grid is None and window_grid is None:
        return None

    if not args.include_trisla and args.comparison_preset != "stage6_trisla":
        raise ValueError("TriSLA sweep requested without enabling TriSLA.")

    if weight_grid is None:
        weight_grid = [(
            float(args.trisla_lambda_current),
            float(args.trisla_lambda_window),
            float(args.trisla_lambda_prefix),
        )]
    if window_grid is None:
        window_grid = [int(args.trisla_window)]

    configs = []
    seen = set()
    for window_size in window_grid:
        if window_size < 1:
            raise ValueError("All TriSLA sweep windows must be >= 1.")
        for lambda_current, lambda_window, lambda_prefix in weight_grid:
            if lambda_current + lambda_window + lambda_prefix <= 0.0:
                raise ValueError("Each TriSLA sweep weight triple must have positive total weight.")
            config = (
                int(window_size),
                float(lambda_current),
                float(lambda_window),
                float(lambda_prefix),
            )
            if config in seen:
                continue
            seen.add(config)
            configs.append(
                {
                    "sweep_id": trisla_sweep_id(*config),
                    "trisla_window": config[0],
                    "trisla_lambda_current": config[1],
                    "trisla_lambda_window": config[2],
                    "trisla_lambda_prefix": config[3],
                }
            )
    return configs


def summarize_trace(trace):
    if not trace:
        return {
            "dyn_steps": 0,
            "avg_alpha": None,
            "alpha_std": None,
            "avg_selected_layer": None,
            "switch_rate": None,
            "avg_instability": None,
            "avg_risk_score": None,
            "trigger_rate": None,
            "avg_jsd_current": None,
            "avg_jsd_window": None,
            "avg_jsd_prefix": None,
            "avg_selection_margin": None,
            "avg_history_window_signal": None,
            "avg_history_prefix_signal": None,
            "avg_history_credit": None,
            "avg_selection_score": None,
            "fallback_rate": None,
            "avg_baseline_margin": None,
            "baseline_guard_rate": None,
            "positive_signal_rate": None,
            "verification_required_rate": None,
            "verification_pass_rate": None,
            "avg_verified_gain": None,
            "avg_layer_penalty": None,
            "avg_flip_penalty": None,
            "avg_calibration_confidence": None,
            "avg_calibration_confidence_ema": None,
            "gate_active_rate": None,
            "confidence_trigger_rate": None,
            "margin_trigger_rate": None,
            "gate_switch_on_rate": None,
            "gate_switch_off_rate": None,
            "avg_gate_hold_remaining": None,
            "switch_blocked_rate": None,
            "avg_switch_gap": None,
            "avg_jacobi_passes": None,
            "avg_jacobi_window_size": None,
            "avg_jacobi_stable_prefix": None,
            "avg_jacobi_commit_len": None,
            "jacobi_convergence_rate": None,
            "jaca_disagreement_rate": None,
            "jaca_truth_selection_rate": None,
            "avg_jaca_divergence": None,
            "avg_jaca_safe_confidence": None,
            "avg_jaca_truth_confidence": None,
            "avg_jaca_agreement_prefix": None,
            "jaca_arbitration_rate": None,
        }
    alphas = [float(row["alpha"]) for row in trace if row.get("alpha") is not None]
    layers = [int(row["selected_layer"]) for row in trace if row.get("selected_layer") is not None]
    instabilities = [float(row["instability"]) for row in trace if row.get("instability") is not None]
    switches = sum(int(layers[i] != layers[i - 1]) for i in range(1, len(layers))) if len(layers) > 1 else 0
    switch_rate = switches / max(1, len(layers) - 1) if layers else None
    return {
        "dyn_steps": len(trace),
        "avg_alpha": statistics.mean(alphas) if alphas else None,
        "alpha_std": statistics.pstdev(alphas) if len(alphas) > 1 else 0.0 if alphas else None,
        "avg_selected_layer": statistics.mean(layers) if layers else None,
        "switch_rate": switch_rate,
        "avg_instability": statistics.mean(instabilities) if instabilities else None,
        "avg_risk_score": mean_or_none(row.get("risk_score") for row in trace),
        "trigger_rate": mean_or_none(row.get("risk_triggered") for row in trace),
        "avg_jsd_current": mean_or_none(row.get("jsd_current") for row in trace),
        "avg_jsd_window": mean_or_none(row.get("jsd_window") for row in trace),
        "avg_jsd_prefix": mean_or_none(row.get("jsd_prefix") for row in trace),
        "avg_selection_margin": mean_or_none(row.get("selection_margin") for row in trace),
        "avg_history_window_signal": mean_or_none(row.get("history_window_signal") for row in trace),
        "avg_history_prefix_signal": mean_or_none(row.get("history_prefix_signal") for row in trace),
        "avg_history_credit": mean_or_none(row.get("history_credit") for row in trace),
        "avg_selection_score": mean_or_none(row.get("selection_score") for row in trace),
        "fallback_rate": mean_or_none(row.get("fallback_used") for row in trace),
        "avg_baseline_margin": mean_or_none(row.get("baseline_margin") for row in trace),
        "baseline_guard_rate": mean_or_none(row.get("baseline_guarded") for row in trace),
        "positive_signal_rate": mean_or_none(row.get("positive_signals_ok") for row in trace),
        "verification_required_rate": mean_or_none(row.get("verification_required") for row in trace),
        "verification_pass_rate": mean_or_none(row.get("verification_passed") for row in trace),
        "avg_verified_gain": mean_or_none(row.get("verified_gain") for row in trace),
        "avg_layer_penalty": mean_or_none(row.get("layer_penalty") for row in trace),
        "avg_flip_penalty": mean_or_none(row.get("flip_penalty") for row in trace),
        "avg_calibration_confidence": mean_or_none(row.get("calibration_confidence") for row in trace),
        "avg_calibration_confidence_ema": mean_or_none(row.get("calibration_confidence_ema") for row in trace),
        "gate_active_rate": mean_or_none(row.get("gate_active") for row in trace),
        "confidence_trigger_rate": mean_or_none(row.get("confidence_triggered") for row in trace),
        "margin_trigger_rate": mean_or_none(row.get("margin_triggered") for row in trace),
        "gate_switch_on_rate": mean_or_none(row.get("gate_switch_on") for row in trace),
        "gate_switch_off_rate": mean_or_none(row.get("gate_switch_off") for row in trace),
        "avg_gate_hold_remaining": mean_or_none(row.get("gate_hold_remaining") for row in trace),
        "switch_blocked_rate": mean_or_none(row.get("switch_blocked") for row in trace),
        "avg_switch_gap": mean_or_none(row.get("switch_gap") for row in trace),
        "avg_jacobi_passes": mean_or_none(row.get("jacobi_passes_used") for row in trace),
        "avg_jacobi_window_size": mean_or_none(row.get("jacobi_window_size") for row in trace),
        "avg_jacobi_stable_prefix": mean_or_none(row.get("jacobi_stable_prefix_len") for row in trace),
        "avg_jacobi_commit_len": mean_or_none(row.get("jacobi_commit_len") for row in trace),
        "jacobi_convergence_rate": mean_or_none(row.get("jacobi_converged") for row in trace),
        "jaca_disagreement_rate": mean_or_none(row.get("jaca_disagreement") for row in trace),
        "jaca_truth_selection_rate": mean_or_none(row.get("jaca_selected_truth") for row in trace),
        "avg_jaca_divergence": mean_or_none(row.get("jaca_divergence") for row in trace),
        "avg_jaca_safe_confidence": mean_or_none(row.get("jaca_safe_confidence") for row in trace),
        "avg_jaca_truth_confidence": mean_or_none(row.get("jaca_truth_confidence") for row in trace),
        "avg_jaca_agreement_prefix": mean_or_none(row.get("jaca_agreement_prefix_len") for row in trace),
        "jaca_arbitration_rate": mean_or_none(row.get("jaca_arbitration_active") for row in trace),
    }


def compute_choice_score_details(choice_scores, correct_choice=None):
    sorted_rows = sorted(((str(choice), float(score)) for choice, score in choice_scores.items()), key=lambda row: row[1], reverse=True)
    prediction = sorted_rows[0][0]
    prediction_score = sorted_rows[0][1]
    runner_up_score = sorted_rows[1][1] if len(sorted_rows) > 1 else None
    decision_margin = prediction_score - runner_up_score if runner_up_score is not None else None

    details = {
        "prediction": prediction,
        "prediction_score": prediction_score,
        "runner_up_score": runner_up_score,
        "decision_margin": decision_margin,
        "correct_choice_score": None,
        "best_false_score": None,
        "correct_margin": None,
    }
    if correct_choice is None:
        return details

    correct_choice_norm = normalize_text(correct_choice)
    matching_scores = [score for choice, score in sorted_rows if normalize_text(choice) == correct_choice_norm]
    false_scores = [score for choice, score in sorted_rows if normalize_text(choice) != correct_choice_norm]
    if matching_scores:
        details["correct_choice_score"] = matching_scores[0]
    if false_scores:
        details["best_false_score"] = max(false_scores)
    if details["correct_choice_score"] is not None and details["best_false_score"] is not None:
        details["correct_margin"] = details["correct_choice_score"] - details["best_false_score"]
    return details


def score_choices_with_decoder(evaluator, prompt, choices, decoder_name):
    scored_rows = []
    runtime_summaries = []
    for choice in choices:
        sequence_logprob, trace, runtime_summary = evaluator.score_candidate_with_decoder(prompt, decoder_name, choice)
        scored_rows.append({"choice": choice, "sequence_logprob": sequence_logprob, "trace": trace})
        runtime_summaries.append(runtime_summary)
    best_row = max(scored_rows, key=lambda row: row["sequence_logprob"])
    return scored_rows, best_row["trace"], merge_runtime_summaries(runtime_summaries)


def build_pairwise_preference_prompt(prompt, candidate_a, candidate_b):
    return (
        "Compare two candidate answers to the same question.\n"
        "Choose the answer that is more likely to be correct.\n\n"
        f"Question:\n{prompt}\n\n"
        f"Candidate A:\n{candidate_a}\n\n"
        f"Candidate B:\n{candidate_b}\n\n"
        "Reply with exactly one label:\n"
        "A = Candidate A is more likely correct\n"
        "B = Candidate B is more likely correct\n"
    )


def query_pairwise_candidate_preference(evaluator, prompt, candidate_a, candidate_b):
    preference_prompt = build_pairwise_preference_prompt(prompt, candidate_a, candidate_b)
    label_choices = ["A", "B"]
    scored_rows, trace, runtime_summary = score_choices_with_decoder(
        evaluator,
        preference_prompt,
        label_choices,
        "greedy",
    )
    sorted_rows = sorted(scored_rows, key=lambda row: row["sequence_logprob"], reverse=True)
    scores = [row["sequence_logprob"] for row in scored_rows]
    probs = softmax_over_scores(scores)
    prob_map = {row["choice"]: float(prob) for row, prob in zip(scored_rows, probs)}
    score_map = {row["choice"]: row["sequence_logprob"] for row in scored_rows}
    selected_choice = sorted_rows[0]["choice"]
    return {
        "selected_choice": selected_choice,
        "prob_a": prob_map["A"],
        "prob_b": prob_map["B"],
        "prob_tie": 0.0,
        "choice_scores": score_map,
        "choice_probs": prob_map,
        "trace": trace,
        "runtime_summary": runtime_summary,
    }


def choose_pairwise_rerank_branch(
    evaluator,
    prompt,
    candidate_a_prediction,
    candidate_b_prediction,
    candidate_a_decoder,
    candidate_b_decoder,
    default_decoder,
):
    candidate_a_norm = normalize_text(candidate_a_prediction)
    candidate_b_norm = normalize_text(candidate_b_prediction)
    same_prediction = candidate_a_norm == candidate_b_norm
    confidence_base = None
    confidence_fixed = None
    confidence_base_raw = None
    confidence_fixed_raw = None
    confidence_base_valid = None
    confidence_fixed_valid = None
    confidence_runtime = None
    pairwise_choice = None
    pairwise_choice_scores = None
    pairwise_choice_probs = None
    pairwise_tie_prob = None

    selected_decoder = default_decoder
    if not same_prediction:
        preference_query = query_pairwise_candidate_preference(
            evaluator,
            prompt,
            candidate_a_prediction,
            candidate_b_prediction,
        )
        confidence_base = float(preference_query["prob_a"])
        confidence_fixed = float(preference_query["prob_b"])
        confidence_base_raw = f"{confidence_base:.6f}"
        confidence_fixed_raw = f"{confidence_fixed:.6f}"
        confidence_base_valid = 1.0
        confidence_fixed_valid = 1.0
        confidence_runtime = preference_query["runtime_summary"]
        pairwise_choice = preference_query["selected_choice"]
        pairwise_choice_scores = json.dumps(preference_query["choice_scores"], ensure_ascii=True)
        pairwise_choice_probs = json.dumps(preference_query["choice_probs"], ensure_ascii=True)
        pairwise_tie_prob = float(preference_query["prob_tie"])
        if pairwise_choice == "A" and float(confidence_base) > float(confidence_fixed) + evaluator.calibration_rerank_delta:
            selected_decoder = candidate_a_decoder
        elif pairwise_choice == "B" and float(confidence_fixed) > float(confidence_base) + evaluator.calibration_rerank_delta:
            selected_decoder = candidate_b_decoder

    return {
        "selected_decoder": selected_decoder,
        "selected_base": float(selected_decoder == candidate_a_decoder),
        "same_prediction": float(same_prediction),
        "confidence_base": confidence_base,
        "confidence_fixed_alpha": confidence_fixed,
        "confidence_base_raw": confidence_base_raw,
        "confidence_fixed_alpha_raw": confidence_fixed_raw,
        "confidence_base_valid": confidence_base_valid,
        "confidence_fixed_alpha_valid": confidence_fixed_valid,
        "confidence_runtime": confidence_runtime,
        "pairwise_choice": pairwise_choice,
        "pairwise_choice_scores": pairwise_choice_scores,
        "pairwise_choice_probs": pairwise_choice_probs,
        "pairwise_tie_prob": pairwise_tie_prob,
    }


def choose_car_dola_branch(evaluator, prompt, base_prediction, fixed_prediction):
    return choose_pairwise_rerank_branch(
        evaluator,
        prompt,
        base_prediction,
        fixed_prediction,
        "greedy",
        "dyndola_fixed_alpha",
        "dyndola_fixed_alpha",
    )


def choose_alpha_switch_branch(evaluator, prompt, low_prediction, high_prediction):
    return choose_pairwise_rerank_branch(
        evaluator,
        prompt,
        low_prediction,
        high_prediction,
        "dyndola_fixed_alpha_low",
        "dyndola_fixed_alpha_high",
        "dyndola_fixed_alpha_low",
    )


def generate_with_car_dola(evaluator, prompt, max_new_tokens=96, stop_on_eos=True):
    base_prediction, base_trace, base_runtime = evaluator.generate_with_decoder(
        prompt,
        "greedy",
        max_new_tokens=max_new_tokens,
        stop_on_eos=stop_on_eos,
    )
    fixed_prediction, fixed_trace, fixed_runtime = evaluator.generate_with_decoder(
        prompt,
        "dyndola_fixed_alpha",
        max_new_tokens=max_new_tokens,
        stop_on_eos=stop_on_eos,
    )
    branch = choose_car_dola_branch(
        evaluator,
        prompt,
        base_prediction,
        fixed_prediction,
    )
    use_base = branch["selected_decoder"] == "greedy"
    prediction = base_prediction if use_base else fixed_prediction
    trace = base_trace if use_base else fixed_trace
    extra = {
        "scoring_mode": "generation_calibration_reranked_fixed_alpha",
        "choice_scores": None,
        "car_selected_decoder": branch["selected_decoder"],
        "car_selected_base": branch["selected_base"],
        "car_same_prediction": branch["same_prediction"],
        "car_confidence_base": branch["confidence_base"],
        "car_confidence_fixed_alpha": branch["confidence_fixed_alpha"],
        "car_confidence_base_valid": branch["confidence_base_valid"],
        "car_confidence_fixed_alpha_valid": branch["confidence_fixed_alpha_valid"],
        "car_confidence_base_raw": branch["confidence_base_raw"],
        "car_confidence_fixed_alpha_raw": branch["confidence_fixed_alpha_raw"],
        "car_pairwise_choice": branch["pairwise_choice"],
        "car_pairwise_choice_scores": branch["pairwise_choice_scores"],
        "car_pairwise_choice_probs": branch["pairwise_choice_probs"],
        "car_pairwise_tie_prob": branch["pairwise_tie_prob"],
    }
    extra.update(
        merge_runtime_summaries(
            (base_runtime, fixed_runtime, branch["confidence_runtime"])
        )
    )
    return prediction, trace, extra


def generate_with_alpha_switch_car_dola(evaluator, prompt, max_new_tokens=96, stop_on_eos=True):
    low_prediction, low_trace, low_runtime = evaluator.generate_with_decoder(
        prompt,
        "dyndola_fixed_alpha_low",
        max_new_tokens=max_new_tokens,
        stop_on_eos=stop_on_eos,
    )
    high_prediction, high_trace, high_runtime = evaluator.generate_with_decoder(
        prompt,
        "dyndola_fixed_alpha_high",
        max_new_tokens=max_new_tokens,
        stop_on_eos=stop_on_eos,
    )
    branch = choose_alpha_switch_branch(
        evaluator,
        prompt,
        low_prediction,
        high_prediction,
    )
    use_low = branch["selected_decoder"] == "dyndola_fixed_alpha_low"
    prediction = low_prediction if use_low else high_prediction
    trace = low_trace if use_low else high_trace
    extra = {
        "scoring_mode": "generation_alpha_switch_calibration_reranked_fixed_alpha",
        "choice_scores": None,
        "car_selected_decoder": branch["selected_decoder"],
        "car_selected_base": branch["selected_base"],
        "car_same_prediction": branch["same_prediction"],
        "car_confidence_base": branch["confidence_base"],
        "car_confidence_fixed_alpha": branch["confidence_fixed_alpha"],
        "car_confidence_base_valid": branch["confidence_base_valid"],
        "car_confidence_fixed_alpha_valid": branch["confidence_fixed_alpha_valid"],
        "car_confidence_base_raw": branch["confidence_base_raw"],
        "car_confidence_fixed_alpha_raw": branch["confidence_fixed_alpha_raw"],
        "car_pairwise_choice": branch["pairwise_choice"],
        "car_pairwise_choice_scores": branch["pairwise_choice_scores"],
        "car_pairwise_choice_probs": branch["pairwise_choice_probs"],
        "car_pairwise_tie_prob": branch["pairwise_tie_prob"],
    }
    extra.update(
        merge_runtime_summaries(
            (low_runtime, high_runtime, branch["confidence_runtime"])
        )
    )
    return prediction, trace, extra


def predict_choice(evaluator, question, choices, decoder_name, prompt_builder):
    prompt = prompt_builder(question)
    if decoder_name == "car_dola":
        base_rows, base_trace, base_runtime = score_choices_with_decoder(evaluator, prompt, choices, "greedy")
        fixed_rows, fixed_trace, fixed_runtime = score_choices_with_decoder(
            evaluator, prompt, choices, "dyndola_fixed_alpha"
        )
        base_sorted_rows = sorted(base_rows, key=lambda row: row["sequence_logprob"], reverse=True)
        fixed_sorted_rows = sorted(fixed_rows, key=lambda row: row["sequence_logprob"], reverse=True)
        branch = choose_car_dola_branch(
            evaluator,
            prompt,
            base_sorted_rows[0]["choice"],
            fixed_sorted_rows[0]["choice"],
        )
        use_base = branch["selected_decoder"] == "greedy"
        chosen_rows = base_sorted_rows if use_base else fixed_sorted_rows
        best_trace = base_trace if use_base else fixed_trace
        runtime_summary = merge_runtime_summaries(
            (base_runtime, fixed_runtime, branch["confidence_runtime"])
        )
        prediction = chosen_rows[0]["choice"]
        choice_scores = {row["choice"]: row["sequence_logprob"] for row in chosen_rows}
        extra = {
            "scoring_mode": "choice_sequence_logprob_calibration_reranked_fixed_alpha",
            "choice_scores": choice_scores,
            "car_selected_decoder": branch["selected_decoder"],
            "car_selected_base": branch["selected_base"],
            "car_same_prediction": branch["same_prediction"],
            "car_confidence_base": branch["confidence_base"],
            "car_confidence_fixed_alpha": branch["confidence_fixed_alpha"],
            "car_confidence_base_valid": branch["confidence_base_valid"],
            "car_confidence_fixed_alpha_valid": branch["confidence_fixed_alpha_valid"],
            "car_confidence_base_raw": branch["confidence_base_raw"],
            "car_confidence_fixed_alpha_raw": branch["confidence_fixed_alpha_raw"],
            "car_pairwise_choice": branch["pairwise_choice"],
            "car_pairwise_choice_scores": branch["pairwise_choice_scores"],
            "car_pairwise_choice_probs": branch["pairwise_choice_probs"],
            "car_pairwise_tie_prob": branch["pairwise_tie_prob"],
        }
        extra.update(runtime_summary)
        return prediction, best_trace, extra
    if decoder_name == "alpha_switch_car_dola":
        low_rows, low_trace, low_runtime = score_choices_with_decoder(
            evaluator, prompt, choices, "dyndola_fixed_alpha_low"
        )
        high_rows, high_trace, high_runtime = score_choices_with_decoder(
            evaluator, prompt, choices, "dyndola_fixed_alpha_high"
        )
        low_sorted_rows = sorted(low_rows, key=lambda row: row["sequence_logprob"], reverse=True)
        high_sorted_rows = sorted(high_rows, key=lambda row: row["sequence_logprob"], reverse=True)
        branch = choose_alpha_switch_branch(
            evaluator,
            prompt,
            low_sorted_rows[0]["choice"],
            high_sorted_rows[0]["choice"],
        )
        use_low = branch["selected_decoder"] == "dyndola_fixed_alpha_low"
        chosen_rows = low_sorted_rows if use_low else high_sorted_rows
        best_trace = low_trace if use_low else high_trace
        runtime_summary = merge_runtime_summaries(
            (low_runtime, high_runtime, branch["confidence_runtime"])
        )
        prediction = chosen_rows[0]["choice"]
        choice_scores = {row["choice"]: row["sequence_logprob"] for row in chosen_rows}
        extra = {
            "scoring_mode": "choice_sequence_logprob_alpha_switch_calibration_reranked_fixed_alpha",
            "choice_scores": choice_scores,
            "car_selected_decoder": branch["selected_decoder"],
            "car_selected_base": branch["selected_base"],
            "car_same_prediction": branch["same_prediction"],
            "car_confidence_base": branch["confidence_base"],
            "car_confidence_fixed_alpha": branch["confidence_fixed_alpha"],
            "car_confidence_base_valid": branch["confidence_base_valid"],
            "car_confidence_fixed_alpha_valid": branch["confidence_fixed_alpha_valid"],
            "car_confidence_base_raw": branch["confidence_base_raw"],
            "car_confidence_fixed_alpha_raw": branch["confidence_fixed_alpha_raw"],
            "car_pairwise_choice": branch["pairwise_choice"],
            "car_pairwise_choice_scores": branch["pairwise_choice_scores"],
            "car_pairwise_choice_probs": branch["pairwise_choice_probs"],
            "car_pairwise_tie_prob": branch["pairwise_tie_prob"],
        }
        extra.update(runtime_summary)
        return prediction, best_trace, extra
    scored_rows, best_trace, runtime_summary = score_choices_with_decoder(evaluator, prompt, choices, decoder_name)
    sorted_rows = sorted(scored_rows, key=lambda row: row["sequence_logprob"], reverse=True)
    prediction = sorted_rows[0]["choice"]
    choice_scores = {row["choice"]: row["sequence_logprob"] for row in sorted_rows}
    extra = {"scoring_mode": "choice_sequence_logprob", "choice_scores": choice_scores}
    extra.update(runtime_summary)
    return prediction, best_trace, extra


def softmax_over_scores(scores):
    max_score = max(scores)
    shifted = [math.exp(score - max_score) for score in scores]
    denom = sum(shifted)
    return [value / denom for value in shifted]


def compute_truthfulqa_mc_metrics(mc1_rows, mc1_labels, mc2_rows, mc2_labels):
    mc1_scores = [row["sequence_logprob"] for row in mc1_rows]
    mc1_true_scores = [score for score, label in zip(mc1_scores, mc1_labels) if int(label) == 1]
    mc1_false_scores = [score for score, label in zip(mc1_scores, mc1_labels) if int(label) == 0]
    if not mc1_true_scores or not mc1_false_scores:
        raise ValueError("TruthfulQA MC1 requires at least one true and one false answer.")
    mc1 = float(max(mc1_true_scores) > max(mc1_false_scores))
    mc1_margin = max(mc1_true_scores) - max(mc1_false_scores)

    mc2_scores = [row["sequence_logprob"] for row in mc2_rows]
    mc2_probs = softmax_over_scores(mc2_scores)
    mc2_true_indices = [idx for idx, label in enumerate(mc2_labels) if int(label) == 1]
    mc2_false_scores = [score for score, label in zip(mc2_scores, mc2_labels) if int(label) == 0]
    if not mc2_true_indices or not mc2_false_scores:
        raise ValueError("TruthfulQA MC2/MC3 requires at least one true and one false answer.")
    mc2 = sum(mc2_probs[idx] for idx in mc2_true_indices)
    false_cutoff = max(mc2_false_scores)
    true_scores = [mc2_scores[idx] for idx in mc2_true_indices]
    mc3 = statistics.mean(float(score > false_cutoff) for score in true_scores)
    mc2_margin = max(true_scores) - false_cutoff
    mc3_margin = statistics.mean(true_scores) - false_cutoff

    sorted_rows = sorted(mc2_rows, key=lambda row: row["sequence_logprob"], reverse=True)
    return {
        "prediction": sorted_rows[0]["choice"],
        "mc1": mc1,
        "mc2": mc2,
        "mc3": mc3,
        "mc1_margin": mc1_margin,
        "mc2_margin": mc2_margin,
        "mc3_margin": mc3_margin,
        "choice_scores": {row["choice"]: row["sequence_logprob"] for row in sorted_rows},
    }


def predict_truthfulqa_mc(evaluator, question, mc1_choices, mc1_labels, mc2_choices, mc2_labels, decoder_name):
    prompt = build_truthfulqa_prompt(question)
    if decoder_name == "car_dola":
        mc1_rows_base, mc1_trace_base, mc1_runtime_base = score_choices_with_decoder(
            evaluator, prompt, mc1_choices, "greedy"
        )
        mc2_rows_base, _, mc2_runtime_base = score_choices_with_decoder(
            evaluator, prompt, mc2_choices, "greedy"
        )
        metrics_base = compute_truthfulqa_mc_metrics(mc1_rows_base, mc1_labels, mc2_rows_base, mc2_labels)

        mc1_rows_fixed, mc1_trace_fixed, mc1_runtime_fixed = score_choices_with_decoder(
            evaluator, prompt, mc1_choices, "dyndola_fixed_alpha"
        )
        mc2_rows_fixed, _, mc2_runtime_fixed = score_choices_with_decoder(
            evaluator, prompt, mc2_choices, "dyndola_fixed_alpha"
        )
        metrics_fixed = compute_truthfulqa_mc_metrics(mc1_rows_fixed, mc1_labels, mc2_rows_fixed, mc2_labels)

        branch = choose_car_dola_branch(
            evaluator,
            prompt,
            metrics_base["prediction"],
            metrics_fixed["prediction"],
        )
        use_base = branch["selected_decoder"] == "greedy"
        metrics = metrics_base if use_base else metrics_fixed
        best_trace = mc1_trace_base if use_base else mc1_trace_fixed
        extra = {
            "scoring_mode": "truthfulqa_mc_sequence_logprob_calibration_reranked_fixed_alpha",
            "choice_scores": metrics["choice_scores"],
            "car_selected_decoder": branch["selected_decoder"],
            "car_selected_base": branch["selected_base"],
            "car_same_prediction": branch["same_prediction"],
            "car_confidence_base": branch["confidence_base"],
            "car_confidence_fixed_alpha": branch["confidence_fixed_alpha"],
            "car_confidence_base_valid": branch["confidence_base_valid"],
            "car_confidence_fixed_alpha_valid": branch["confidence_fixed_alpha_valid"],
            "car_confidence_base_raw": branch["confidence_base_raw"],
            "car_confidence_fixed_alpha_raw": branch["confidence_fixed_alpha_raw"],
            "car_pairwise_choice": branch["pairwise_choice"],
            "car_pairwise_choice_scores": branch["pairwise_choice_scores"],
            "car_pairwise_choice_probs": branch["pairwise_choice_probs"],
            "car_pairwise_tie_prob": branch["pairwise_tie_prob"],
        }
        extra.update(
            merge_runtime_summaries(
                (
                    mc1_runtime_base,
                    mc2_runtime_base,
                    mc1_runtime_fixed,
                    mc2_runtime_fixed,
                    branch["confidence_runtime"],
                )
            )
        )
        return metrics, best_trace, extra
    if decoder_name == "alpha_switch_car_dola":
        mc1_rows_low, mc1_trace_low, mc1_runtime_low = score_choices_with_decoder(
            evaluator, prompt, mc1_choices, "dyndola_fixed_alpha_low"
        )
        mc2_rows_low, _, mc2_runtime_low = score_choices_with_decoder(
            evaluator, prompt, mc2_choices, "dyndola_fixed_alpha_low"
        )
        metrics_low = compute_truthfulqa_mc_metrics(mc1_rows_low, mc1_labels, mc2_rows_low, mc2_labels)

        mc1_rows_high, mc1_trace_high, mc1_runtime_high = score_choices_with_decoder(
            evaluator, prompt, mc1_choices, "dyndola_fixed_alpha_high"
        )
        mc2_rows_high, _, mc2_runtime_high = score_choices_with_decoder(
            evaluator, prompt, mc2_choices, "dyndola_fixed_alpha_high"
        )
        metrics_high = compute_truthfulqa_mc_metrics(mc1_rows_high, mc1_labels, mc2_rows_high, mc2_labels)

        if normalize_text(metrics_low["prediction"]) == normalize_text(metrics_high["prediction"]):
            branch = {
                "selected_decoder": "dyndola_fixed_alpha_high",
                "selected_base": 0.0,
                "same_prediction": 1.0,
                "confidence_base": None,
                "confidence_fixed_alpha": None,
                "confidence_base_raw": None,
                "confidence_fixed_alpha_raw": None,
                "confidence_base_valid": None,
                "confidence_fixed_alpha_valid": None,
                "confidence_runtime": None,
                "pairwise_choice": None,
                "pairwise_choice_scores": None,
                "pairwise_choice_probs": None,
                "pairwise_tie_prob": None,
            }
        else:
            branch = choose_alpha_switch_branch(
                evaluator,
                prompt,
                metrics_low["prediction"],
                metrics_high["prediction"],
            )
        use_low = branch["selected_decoder"] == "dyndola_fixed_alpha_low"
        metrics = metrics_low if use_low else metrics_high
        best_trace = mc1_trace_low if use_low else mc1_trace_high
        extra = {
            "scoring_mode": "truthfulqa_mc_sequence_logprob_alpha_switch_calibration_reranked_fixed_alpha",
            "choice_scores": metrics["choice_scores"],
            "car_selected_decoder": branch["selected_decoder"],
            "car_selected_base": branch["selected_base"],
            "car_same_prediction": branch["same_prediction"],
            "car_confidence_base": branch["confidence_base"],
            "car_confidence_fixed_alpha": branch["confidence_fixed_alpha"],
            "car_confidence_base_valid": branch["confidence_base_valid"],
            "car_confidence_fixed_alpha_valid": branch["confidence_fixed_alpha_valid"],
            "car_confidence_base_raw": branch["confidence_base_raw"],
            "car_confidence_fixed_alpha_raw": branch["confidence_fixed_alpha_raw"],
            "car_pairwise_choice": branch["pairwise_choice"],
            "car_pairwise_choice_scores": branch["pairwise_choice_scores"],
            "car_pairwise_choice_probs": branch["pairwise_choice_probs"],
            "car_pairwise_tie_prob": branch["pairwise_tie_prob"],
        }
        extra.update(
            merge_runtime_summaries(
                (
                    mc1_runtime_low,
                    mc2_runtime_low,
                    mc1_runtime_high,
                    mc2_runtime_high,
                    branch["confidence_runtime"],
                )
            )
        )
        return metrics, best_trace, extra
    mc1_rows, best_trace, mc1_runtime = score_choices_with_decoder(evaluator, prompt, mc1_choices, decoder_name)
    mc2_rows, _, mc2_runtime = score_choices_with_decoder(evaluator, prompt, mc2_choices, decoder_name)
    metrics = compute_truthfulqa_mc_metrics(mc1_rows, mc1_labels, mc2_rows, mc2_labels)
    extra = {"scoring_mode": "truthfulqa_mc_sequence_logprob", "choice_scores": metrics["choice_scores"]}
    extra.update(merge_runtime_summaries((mc1_runtime, mc2_runtime)))
    return metrics, best_trace, extra


def predict_gsm8k(evaluator, question, decoder_name):
    prompt = build_gsm8k_prompt(question)
    if decoder_name == "car_dola":
        return generate_with_car_dola(evaluator, prompt, max_new_tokens=96, stop_on_eos=True)
    if decoder_name == "alpha_switch_car_dola":
        return generate_with_alpha_switch_car_dola(evaluator, prompt, max_new_tokens=96, stop_on_eos=True)
    prediction, trace, runtime_summary = evaluator.generate_with_decoder(prompt, decoder_name)
    extra = {"scoring_mode": "strict_generation", "choice_scores": None}
    extra.update(runtime_summary)
    return prediction, trace, extra


def predict_gsm8k_sequence(evaluator, question, decoder_name, max_new_tokens):
    prompt = build_gsm8k_sequence_prompt(question)
    if decoder_name == "car_dola":
        return generate_with_car_dola(
            evaluator,
            prompt,
            max_new_tokens=max_new_tokens,
            stop_on_eos=True,
        )
    if decoder_name == "alpha_switch_car_dola":
        return generate_with_alpha_switch_car_dola(
            evaluator,
            prompt,
            max_new_tokens=max_new_tokens,
            stop_on_eos=True,
        )
    prediction, trace, runtime_summary = evaluator.generate_with_decoder(
        prompt,
        decoder_name,
        max_new_tokens=max_new_tokens,
    )
    extra = {"scoring_mode": "sequence_generation", "choice_scores": None}
    extra.update(runtime_summary)
    return prediction, trace, extra


def load_truthfulqa_rows(limit, rng):
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    candidates = []
    for source_idx, row in enumerate(dataset):
        mc1_targets = row.get("mc1_targets") or {}
        mc2_targets = row.get("mc2_targets") or {}
        mc1_choices = list(mc1_targets.get("choices", [])) if isinstance(mc1_targets, dict) else []
        mc1_labels = list(mc1_targets.get("labels", [])) if isinstance(mc1_targets, dict) else []
        mc2_choices = list(mc2_targets.get("choices", [])) if isinstance(mc2_targets, dict) else []
        mc2_labels = list(mc2_targets.get("labels", [])) if isinstance(mc2_targets, dict) else []
        if not mc1_choices or not mc1_labels or not mc2_choices or not mc2_labels:
            continue
        if sum(int(label) == 1 for label in mc1_labels) != 1:
            continue
        if sum(int(label) == 1 for label in mc2_labels) < 1:
            continue
        candidates.append(
            {
                "source_idx": int(source_idx),
                "benchmark": "truthfulqa",
                "question": row["question"],
                "mc1_choices": mc1_choices,
                "mc1_labels": mc1_labels,
                "mc2_choices": mc2_choices,
                "mc2_labels": mc2_labels,
            }
        )
    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    return rows, "truthful_qa/multiple_choice", manifest


def load_strategyqa_rows(limit, rng):
    forced_dataset = getattr(load_strategyqa_rows, "_forced_dataset", None)
    forced_config = getattr(load_strategyqa_rows, "_forced_config", None)
    forced_split = getattr(load_strategyqa_rows, "_forced_split", None)

    if forced_dataset is not None:
        dataset_name = forced_dataset
        config_name = forced_config
        split_name = forced_split or DEFAULT_STRATEGYQA_SPLIT
    else:
        dataset_name = DEFAULT_STRATEGYQA_DATASET
        config_name = DEFAULT_STRATEGYQA_CONFIG
        split_name = DEFAULT_STRATEGYQA_SPLIT

    try:
        kwargs = {"path": dataset_name}
        if config_name is not None:
            kwargs["name"] = config_name
        dataset = load_dataset(**kwargs, split=split_name)
    except Exception as exc:
        raise RuntimeError(
            "StrategyQA loader failed for the configured source: "
            f"dataset={dataset_name!r} config={config_name!r} split={split_name!r}: {exc}"
        ) from exc

    candidates = []
    for source_idx, row in enumerate(dataset):
        question = row.get("inputs") or row.get("question")
        target_norm = canonicalize_yes_no_label(row.get("targets"))
        if target_norm is None:
            target_norm = canonicalize_yes_no_label(row.get("answer"))
        if target_norm is None:
            target_norm = canonicalize_yes_no_label(row.get("label"))
        if target_norm is None:
            mc_targets = row.get("multiple_choice_targets")
            mc_scores = row.get("multiple_choice_scores")
            if isinstance(mc_targets, list) and isinstance(mc_scores, list) and mc_targets and mc_scores:
                best_idx = max(range(min(len(mc_targets), len(mc_scores))), key=lambda idx: float(mc_scores[idx]))
                target_norm = canonicalize_yes_no_label(mc_targets[best_idx])
        if target_norm not in {"yes", "no"} or not str(question).strip():
            continue
        candidates.append(
            {
                "source_idx": int(source_idx),
                "benchmark": "strategyqa",
                "question": str(question).strip(),
                "choices": ["yes", "no"],
                "correct_choice": target_norm,
            }
        )

    if forced_dataset is not None and not candidates:
        raise RuntimeError(
            "StrategyQA loader found the forced dataset but produced 0 usable rows. "
            f"dataset={forced_dataset!r} config={forced_config!r} split={forced_split or 'validation'!r}"
        )
    if not candidates:
        raise RuntimeError(
            "StrategyQA loader produced 0 usable rows for the configured source. "
            f"dataset={dataset_name!r} config={config_name!r} split={split_name!r}"
        )
    source_name = dataset_name if config_name is None else f"{dataset_name}/{config_name}"
    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    return rows, f"{source_name}:{split_name}", manifest


def load_gsm8k_rows(limit, rng):
    dataset = load_dataset("gsm8k", "main", split="test")
    candidates = []
    for source_idx, row in enumerate(dataset):
        answer_text = row.get("answer", "")
        match = re.findall(r"####\s*([-0-9.,]+)", answer_text)
        if not match:
            continue
        candidates.append(
            {
                "source_idx": int(source_idx),
                "benchmark": "gsm8k",
                "question": row["question"],
                "correct_choice": match[-1].replace(",", "").strip(),
            }
        )
    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    return rows, "gsm8k/main", manifest


def load_alpacaeval_rows(limit, rng):
    dataset = load_dataset(
        DEFAULT_ALPACAEVAL_DATASET,
        DEFAULT_ALPACAEVAL_CONFIG,
        split=DEFAULT_ALPACAEVAL_SPLIT,
    )
    candidates = []
    for source_idx, row in enumerate(dataset):
        instruction = str(row.get("instruction") or "").strip()
        reference_output = str(row.get("output") or "").strip()
        dataset_name = str(row.get("dataset") or "alpaca_eval").strip()
        generator_name = str(row.get("generator") or "gpt4_turbo").strip()
        if not instruction or not reference_output:
            continue
        candidates.append(
            {
                "source_idx": int(source_idx),
                "benchmark": "alpacaeval",
                "instruction": instruction,
                "reference_output": reference_output,
                "reference_generator": generator_name,
                "dataset_name": dataset_name,
            }
        )
    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    return rows, DEFAULT_ALPACAEVAL_SOURCE, manifest


def load_halueval_rows(limit, rng, root_path, tasks):
    root = Path(root_path).expanduser()
    if not root.exists():
        raise RuntimeError(f"HaluEval root does not exist: {root}")

    task_to_filename = {
        "qa": "qa_data.json",
        "dialogue": "dialogue_data.json",
        "summarization": "summarization_data.json",
        "general": "general_data.json",
    }

    candidates = []
    for task_name in tasks:
        if task_name not in task_to_filename:
            raise RuntimeError(f"Unsupported HaluEval task: {task_name!r}")
        data_path = root / task_to_filename[task_name]
        if not data_path.exists():
            raise RuntimeError(f"Missing HaluEval file for task {task_name!r}: {data_path}")
        raw_text = data_path.read_text(encoding="utf-8")
        try:
            rows = json.loads(raw_text)
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        for source_idx, row in enumerate(rows):
            knowledge = str(row.get("knowledge") or row.get("document") or row.get("context") or "").strip()
            user_input = str(
                row.get("question") or row.get("dialogue_history") or row.get("user_query") or row.get("input") or ""
            ).strip()
            right_response = str(
                row.get("right_answer") or row.get("right_response") or row.get("right_summary") or ""
            ).strip()
            hallucinated_response = str(
                row.get("hallucinated_answer")
                or row.get("hallucinated_response")
                or row.get("hallucinated_summary")
                or ""
            ).strip()
            if not right_response or not hallucinated_response:
                continue
            use_hallucinated = bool(rng.random() < 0.5)
            response_text = hallucinated_response if use_hallucinated else right_response
            yes_label, no_label = ("A", "B") if bool(rng.random() < 0.5) else ("B", "A")
            correct_choice = yes_label if use_hallucinated else no_label
            prompt_text = build_halueval_prompt(
                task_name,
                knowledge,
                user_input,
                response_text,
                label_yes=yes_label,
                label_no=no_label,
            )
            candidates.append(
                {
                    "source_idx": int(source_idx),
                    "benchmark": f"halueval_{task_name}",
                    "task_name": task_name,
                    "question": prompt_text,
                    "choices": ["A", "B"],
                    "correct_choice": correct_choice,
                    "halueval_yes_label": yes_label,
                    "halueval_no_label": no_label,
                    "halueval_has_hallucination": use_hallucinated,
                    "knowledge": knowledge,
                    "user_input": user_input,
                    "response_text": response_text,
                    "halueval_source_file": str(data_path),
                }
            )

    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    source_name = f"local:{root}"
    return rows, source_name, manifest


def assert_eval_sources(args, truthfulqa_source, strategyqa_source, gsm8k_source, alpacaeval_source="disabled"):
    if not args.strict_eval:
        return

    expected_truthfulqa = "truthful_qa/multiple_choice"
    expected_gsm8k = "gsm8k/main"

    if not args.skip_truthfulqa and truthfulqa_source != expected_truthfulqa:
        raise RuntimeError(
            f"Strict evaluation requires {expected_truthfulqa} for TruthfulQA, got {truthfulqa_source!r}."
        )
    if not args.skip_gsm8k and gsm8k_source != expected_gsm8k:
        raise RuntimeError(f"Strict evaluation requires {expected_gsm8k} for GSM8K, got {gsm8k_source!r}.")
    if not args.skip_strategyqa and strategyqa_source != DEFAULT_STRATEGYQA_SOURCE:
        raise RuntimeError(
            f"Strict evaluation requires {DEFAULT_STRATEGYQA_SOURCE} for StrategyQA, "
            f"got {strategyqa_source!r}."
        )
    if args.include_alpacaeval and alpacaeval_source != DEFAULT_ALPACAEVAL_SOURCE:
        raise RuntimeError(
            f"Strict evaluation requires {DEFAULT_ALPACAEVAL_SOURCE} for AlpacaEval, "
            f"got {alpacaeval_source!r}."
        )


def export_alpacaeval_outputs(evaluator, rows, decoder_names, results_dir, artifact_prefix):
    results_dir.mkdir(parents=True, exist_ok=True)
    reference_rows = [
        {
            "instruction": row["instruction"],
            "output": row["reference_output"],
            "generator": row["reference_generator"],
            "dataset": row["dataset_name"],
        }
        for row in rows
    ]
    reference_path = results_dir / f"{artifact_prefix}_alpacaeval_reference_outputs.json"
    reference_path.write_text(json.dumps(reference_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    generated_paths = {}
    for decoder_name in decoder_names:
        exported_rows = []
        for example_idx, row in enumerate(rows, start=1):
            if example_idx == 1 or example_idx == len(rows):
                print(f"[alpacaeval:{decoder_name}] {example_idx}/{len(rows)}")
            if decoder_name == "car_dola":
                output_text, _, extra = generate_with_car_dola(
                    evaluator,
                    row["instruction"],
                    max_new_tokens=evaluator.alpacaeval_max_new_tokens,
                    stop_on_eos=True,
                )
            elif decoder_name == "alpha_switch_car_dola":
                output_text, _, extra = generate_with_alpha_switch_car_dola(
                    evaluator,
                    row["instruction"],
                    max_new_tokens=evaluator.alpacaeval_max_new_tokens,
                    stop_on_eos=True,
                )
            else:
                output_text, _, extra = evaluator.generate_with_decoder(
                    row["instruction"],
                    decoder_name,
                    max_new_tokens=evaluator.alpacaeval_max_new_tokens,
                    stop_on_eos=True,
                )
            exported_rows.append(
                {
                    "instruction": row["instruction"],
                    "output": output_text,
                    "generator": get_decoder_label(decoder_name),
                    "dataset": row["dataset_name"],
                    "latency_seconds": extra.get("latency_seconds"),
                }
            )
        output_path = results_dir / f"{artifact_prefix}_alpacaeval_{decoder_name}_model_outputs.json"
        output_path.write_text(json.dumps(exported_rows, indent=2, ensure_ascii=True), encoding="utf-8")
        generated_paths[decoder_name] = str(output_path)

    command_lines = [
        "# Official AlpacaEval 2.0 scoring uses the alpaca_eval package.",
        "# Default annotator: weighted_alpaca_eval_gpt4_turbo",
    ]
    for decoder_name, output_path in generated_paths.items():
        command_lines.append(
            "alpaca_eval "
            f"--model_outputs '{output_path}' "
            f"--reference_outputs '{reference_path}' "
            f"--output_path '{results_dir / f'{artifact_prefix}_alpacaeval_{decoder_name}_scored'}'"
        )
    command_path = results_dir / f"{artifact_prefix}_alpacaeval_commands.txt"
    command_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    return {
        "reference_outputs_path": str(reference_path),
        "model_outputs_paths": generated_paths,
        "command_path": str(command_path),
        "official_scorer_installed": importlib.util.find_spec("alpaca_eval") is not None,
    }


def evaluate_truthfulqa(evaluator, rows, progress_every=1, decoder_names=None):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    evaluator.use_truthfulqa_bucket()
    try:
        for example_idx, row in enumerate(rows, start=1):
            if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
                print(f"[truthfulqa] {example_idx}/{total}")
            for decoder_name in decoder_names:
                metrics, trace, extra = predict_truthfulqa_mc(
                    evaluator,
                    row["question"],
                    row["mc1_choices"],
                    row["mc1_labels"],
                    row["mc2_choices"],
                    row["mc2_labels"],
                    decoder_name,
                )
                trace_summary = summarize_trace(trace)
                metric_margins = {
                    "mc1": metrics["mc1_margin"],
                    "mc2": metrics["mc2_margin"],
                    "mc3": metrics["mc3_margin"],
                }
                for metric_name in ("mc1", "mc2", "mc3"):
                    result = {
                        "benchmark": "truthfulqa",
                        "metric_name": metric_name,
                        "example_idx": example_idx - 1,
                        "decoder": decoder_name,
                        "decoder_label": get_decoder_label(decoder_name),
                        "question": row["question"],
                        "correct_choice": None,
                        "prediction": metrics["prediction"],
                        "score": float(metrics[metric_name]),
                        "score_detail": f"{metric_name}={metrics[metric_name]:.4f}",
                        "scoring_mode": extra["scoring_mode"],
                        "choice_scores": json.dumps(extra["choice_scores"], ensure_ascii=True),
                        "decision_margin": float(metric_margins[metric_name]),
                        "correct_margin": float(metric_margins[metric_name]),
                        "format_valid": None,
                        "latency_seconds": extra["latency_seconds"],
                        "decoder_steps": extra["decoder_steps"],
                        "forward_passes": extra["forward_passes"],
                        "latency_per_step_ms": extra["latency_per_step_ms"],
                        "latency_per_forward_ms": extra["latency_per_forward_ms"],
                        "steps_per_second": extra["steps_per_second"],
                        "tokens_per_forward": extra["tokens_per_forward"],
                        "factual_speedup": extra["factual_speedup"],
                        "car_selected_decoder": extra.get("car_selected_decoder"),
                        "car_selected_base": extra.get("car_selected_base"),
                        "car_same_prediction": extra.get("car_same_prediction"),
                        "car_confidence_base": extra.get("car_confidence_base"),
                        "car_confidence_fixed_alpha": extra.get("car_confidence_fixed_alpha"),
                        "car_confidence_base_valid": extra.get("car_confidence_base_valid"),
                        "car_confidence_fixed_alpha_valid": extra.get("car_confidence_fixed_alpha_valid"),
                        "car_confidence_base_raw": extra.get("car_confidence_base_raw"),
                        "car_confidence_fixed_alpha_raw": extra.get("car_confidence_fixed_alpha_raw"),
                        "car_pairwise_choice": extra.get("car_pairwise_choice"),
                        "car_pairwise_choice_scores": extra.get("car_pairwise_choice_scores"),
                        "car_pairwise_choice_probs": extra.get("car_pairwise_choice_probs"),
                        "car_pairwise_tie_prob": extra.get("car_pairwise_tie_prob"),
                    }
                    result.update(trace_summary)
                    results.append(result)
    finally:
        evaluator.use_default_bucket()
    return results


def evaluate_strategyqa(evaluator, rows, progress_every=1, decoder_names=None):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[strategyqa] {example_idx}/{total}")
        for decoder_name in decoder_names:
            prediction, trace, extra = predict_choice(
                evaluator, row["question"], row["choices"], decoder_name, build_strategyqa_prompt
            )
            score_details = compute_choice_score_details(extra["choice_scores"], row["correct_choice"])
            score = float(normalize_text(prediction) == normalize_text(row["correct_choice"]))
            result = {
                "benchmark": "strategyqa",
                "metric_name": "accuracy",
                "example_idx": example_idx - 1,
                "decoder": decoder_name,
                "decoder_label": get_decoder_label(decoder_name),
                "question": row["question"],
                "correct_choice": row["correct_choice"],
                "prediction": prediction,
                "score": score,
                "score_detail": f"pred={normalize_text(prediction)} gold={normalize_text(row['correct_choice'])}",
                "scoring_mode": extra["scoring_mode"],
                "choice_scores": json.dumps(extra["choice_scores"], ensure_ascii=True),
                "decision_margin": score_details["decision_margin"],
                "correct_margin": score_details["correct_margin"],
                "format_valid": None,
                "latency_seconds": extra["latency_seconds"],
                "decoder_steps": extra["decoder_steps"],
                "forward_passes": extra["forward_passes"],
                "latency_per_step_ms": extra["latency_per_step_ms"],
                "latency_per_forward_ms": extra["latency_per_forward_ms"],
                "steps_per_second": extra["steps_per_second"],
                "tokens_per_forward": extra["tokens_per_forward"],
                "factual_speedup": extra["factual_speedup"],
                "car_selected_decoder": extra.get("car_selected_decoder"),
                "car_selected_base": extra.get("car_selected_base"),
                "car_same_prediction": extra.get("car_same_prediction"),
                "car_confidence_base": extra.get("car_confidence_base"),
                "car_confidence_fixed_alpha": extra.get("car_confidence_fixed_alpha"),
                "car_confidence_base_valid": extra.get("car_confidence_base_valid"),
                "car_confidence_fixed_alpha_valid": extra.get("car_confidence_fixed_alpha_valid"),
                "car_confidence_base_raw": extra.get("car_confidence_base_raw"),
                "car_confidence_fixed_alpha_raw": extra.get("car_confidence_fixed_alpha_raw"),
                "car_pairwise_choice": extra.get("car_pairwise_choice"),
                "car_pairwise_choice_scores": extra.get("car_pairwise_choice_scores"),
                "car_pairwise_choice_probs": extra.get("car_pairwise_choice_probs"),
                "car_pairwise_tie_prob": extra.get("car_pairwise_tie_prob"),
            }
            result.update(summarize_trace(trace))
            results.append(result)
    return results


def evaluate_halueval(evaluator, rows, progress_every=1, decoder_names=None):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[{row['benchmark']}] {example_idx}/{total}")
        for decoder_name in decoder_names:
            prediction, trace, extra = predict_choice(
                evaluator,
                row["question"],
                row["choices"],
                decoder_name,
                lambda text: text,
            )
            score_details = compute_choice_score_details(extra["choice_scores"], row["correct_choice"])
            score = float(normalize_text(prediction) == normalize_text(row["correct_choice"]))
            result = {
                "benchmark": row["benchmark"],
                "metric_name": "accuracy",
                "example_idx": example_idx - 1,
                "decoder": decoder_name,
                "decoder_label": get_decoder_label(decoder_name),
                "question": row["question"],
                "correct_choice": row["correct_choice"],
                "prediction": prediction,
                "halueval_yes_label": row.get("halueval_yes_label"),
                "halueval_no_label": row.get("halueval_no_label"),
                "halueval_has_hallucination": row.get("halueval_has_hallucination"),
                "score": score,
                "score_detail": f"pred={normalize_text(prediction)} gold={normalize_text(row['correct_choice'])}",
                "scoring_mode": extra["scoring_mode"],
                "choice_scores": json.dumps(extra["choice_scores"], ensure_ascii=True),
                "decision_margin": score_details["decision_margin"],
                "correct_margin": score_details["correct_margin"],
                "format_valid": None,
                "latency_seconds": extra["latency_seconds"],
                "decoder_steps": extra["decoder_steps"],
                "forward_passes": extra["forward_passes"],
                "latency_per_step_ms": extra["latency_per_step_ms"],
                "latency_per_forward_ms": extra["latency_per_forward_ms"],
                "steps_per_second": extra["steps_per_second"],
                "tokens_per_forward": extra["tokens_per_forward"],
                "factual_speedup": extra["factual_speedup"],
                "car_selected_decoder": extra.get("car_selected_decoder"),
                "car_selected_base": extra.get("car_selected_base"),
                "car_same_prediction": extra.get("car_same_prediction"),
                "car_confidence_base": extra.get("car_confidence_base"),
                "car_confidence_fixed_alpha": extra.get("car_confidence_fixed_alpha"),
                "car_confidence_base_valid": extra.get("car_confidence_base_valid"),
                "car_confidence_fixed_alpha_valid": extra.get("car_confidence_fixed_alpha_valid"),
                "car_confidence_base_raw": extra.get("car_confidence_base_raw"),
                "car_confidence_fixed_alpha_raw": extra.get("car_confidence_fixed_alpha_raw"),
                "car_pairwise_choice": extra.get("car_pairwise_choice"),
                "car_pairwise_choice_scores": extra.get("car_pairwise_choice_scores"),
                "car_pairwise_choice_probs": extra.get("car_pairwise_choice_probs"),
                "car_pairwise_tie_prob": extra.get("car_pairwise_tie_prob"),
            }
            result.update(summarize_trace(trace))
            results.append(result)
    return results


def evaluate_gsm8k(evaluator, rows, progress_every=1, decoder_names=None):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[gsm8k] {example_idx}/{total}")
        for decoder_name in decoder_names:
            prediction, trace, extra = predict_gsm8k(evaluator, row["question"], decoder_name)
            pred_num = canonicalize_number_text(prediction)
            gold_num = canonicalize_number_text(row["correct_choice"])
            score = float(pred_num == gold_num)
            format_valid = float(pred_num is not None)
            result = {
                "benchmark": "gsm8k",
                "metric_name": "accuracy",
                "example_idx": example_idx - 1,
                "decoder": decoder_name,
                "decoder_label": get_decoder_label(decoder_name),
                "question": row["question"],
                "correct_choice": row["correct_choice"],
                "prediction": prediction,
                "score": score,
                "score_detail": f"pred={pred_num} gold={gold_num} format_valid={bool(format_valid)}",
                "scoring_mode": extra["scoring_mode"],
                "choice_scores": None,
                "decision_margin": None,
                "correct_margin": None,
                "format_valid": format_valid,
                "latency_seconds": extra["latency_seconds"],
                "decoder_steps": extra["decoder_steps"],
                "forward_passes": extra["forward_passes"],
                "latency_per_step_ms": extra["latency_per_step_ms"],
                "latency_per_forward_ms": extra["latency_per_forward_ms"],
                "steps_per_second": extra["steps_per_second"],
                "tokens_per_forward": extra["tokens_per_forward"],
                "factual_speedup": extra["factual_speedup"],
                "car_selected_decoder": extra.get("car_selected_decoder"),
                "car_selected_base": extra.get("car_selected_base"),
                "car_same_prediction": extra.get("car_same_prediction"),
                "car_confidence_base": extra.get("car_confidence_base"),
                "car_confidence_fixed_alpha": extra.get("car_confidence_fixed_alpha"),
                "car_confidence_base_valid": extra.get("car_confidence_base_valid"),
                "car_confidence_fixed_alpha_valid": extra.get("car_confidence_fixed_alpha_valid"),
                "car_confidence_base_raw": extra.get("car_confidence_base_raw"),
                "car_confidence_fixed_alpha_raw": extra.get("car_confidence_fixed_alpha_raw"),
                "car_pairwise_choice": extra.get("car_pairwise_choice"),
                "car_pairwise_choice_scores": extra.get("car_pairwise_choice_scores"),
                "car_pairwise_choice_probs": extra.get("car_pairwise_choice_probs"),
                "car_pairwise_tie_prob": extra.get("car_pairwise_tie_prob"),
            }
            result.update(summarize_trace(trace))
            results.append(result)
    return results


def evaluate_gsm8k_sequence(evaluator, rows, progress_every=1, decoder_names=None, max_new_tokens=160):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[gsm8k_sequence] {example_idx}/{total}")
        for decoder_name in decoder_names:
            prediction, trace, extra = predict_gsm8k_sequence(
                evaluator,
                row["question"],
                decoder_name,
                max_new_tokens=max_new_tokens,
            )
            pred_num = canonicalize_number_text(prediction)
            gold_num = canonicalize_number_text(row["correct_choice"])
            score = float(pred_num == gold_num)
            format_valid = float(pred_num is not None)
            result = {
                "benchmark": "gsm8k_sequence",
                "metric_name": "accuracy",
                "example_idx": example_idx - 1,
                "decoder": decoder_name,
                "decoder_label": get_decoder_label(decoder_name),
                "question": row["question"],
                "correct_choice": row["correct_choice"],
                "prediction": prediction,
                "score": score,
                "score_detail": f"pred={pred_num} gold={gold_num} format_valid={bool(format_valid)}",
                "scoring_mode": extra["scoring_mode"],
                "choice_scores": None,
                "decision_margin": None,
                "correct_margin": None,
                "format_valid": format_valid,
                "latency_seconds": extra["latency_seconds"],
                "decoder_steps": extra["decoder_steps"],
                "forward_passes": extra["forward_passes"],
                "latency_per_step_ms": extra["latency_per_step_ms"],
                "latency_per_forward_ms": extra["latency_per_forward_ms"],
                "steps_per_second": extra["steps_per_second"],
                "tokens_per_forward": extra["tokens_per_forward"],
                "factual_speedup": extra["factual_speedup"],
                "car_selected_decoder": extra.get("car_selected_decoder"),
                "car_selected_base": extra.get("car_selected_base"),
                "car_same_prediction": extra.get("car_same_prediction"),
                "car_confidence_base": extra.get("car_confidence_base"),
                "car_confidence_fixed_alpha": extra.get("car_confidence_fixed_alpha"),
                "car_confidence_base_valid": extra.get("car_confidence_base_valid"),
                "car_confidence_fixed_alpha_valid": extra.get("car_confidence_fixed_alpha_valid"),
                "car_confidence_base_raw": extra.get("car_confidence_base_raw"),
                "car_confidence_fixed_alpha_raw": extra.get("car_confidence_fixed_alpha_raw"),
                "car_pairwise_choice": extra.get("car_pairwise_choice"),
                "car_pairwise_choice_scores": extra.get("car_pairwise_choice_scores"),
                "car_pairwise_choice_probs": extra.get("car_pairwise_choice_probs"),
                "car_pairwise_tie_prob": extra.get("car_pairwise_tie_prob"),
            }
            result.update(summarize_trace(trace))
            results.append(result)
    return results


def build_pairwise_summary(results_df):
    pairwise_rows = []
    score_df = results_df[["benchmark", "metric_name", "example_idx", "decoder_label", "score"]].copy()
    group_columns = ["benchmark", "metric_name"]
    if "sweep_id" in results_df.columns:
        score_df["sweep_id"] = results_df["sweep_id"]
        group_columns = ["sweep_id"] + group_columns
    for group_key, group_df in score_df.groupby(group_columns):
        if len(group_columns) == 3:
            sweep_id, benchmark, metric_name = group_key
        else:
            benchmark, metric_name = group_key
            sweep_id = None
        wide_df = group_df.pivot_table(index="example_idx", columns="decoder_label", values="score", aggfunc="first")
        decoder_labels = list(wide_df.columns)
        for left_label in decoder_labels:
            for right_label in decoder_labels:
                if left_label == right_label:
                    continue
                delta = (wide_df[left_label] - wide_df[right_label]).dropna()
                if delta.empty:
                    continue
                pairwise_rows.append(
                    {
                        "sweep_id": sweep_id,
                        "benchmark": benchmark,
                        "metric_name": metric_name,
                        "left_decoder_label": left_label,
                        "right_decoder_label": right_label,
                        "num_examples": int(delta.shape[0]),
                        "mean_score_delta": float(delta.mean()),
                        "median_score_delta": float(delta.median()),
                        "win_rate": float((delta > 0).mean()),
                        "tie_rate": float((delta == 0).mean()),
                        "loss_rate": float((delta < 0).mean()),
                    }
                )
    if not pairwise_rows:
        return pd.DataFrame()
    sort_columns = ["benchmark", "metric_name", "left_decoder_label", "right_decoder_label"]
    if any(row["sweep_id"] is not None for row in pairwise_rows):
        sort_columns = ["sweep_id"] + sort_columns
    return pd.DataFrame(pairwise_rows).sort_values(sort_columns)


def infer_artifact_prefix(args):
    if args.comparison_preset == "stage14_jaca_eas":
        return "stage14_full_eval"
    if args.comparison_preset == "stage12_jaca" or args.include_jaca:
        return "stage12_full_eval"
    if args.comparison_preset == "stage11c_alpha_switch" or args.include_alpha_switch_car_dola:
        return "stage11c_full_eval"
    if args.comparison_preset == "stage11b_car_dola" or args.include_car_dola:
        return "stage11b_full_eval"
    if args.comparison_preset == "stage10_jacobi":
        return "stage10_full_eval"
    if args.comparison_preset == "stage9_pressure_linear":
        return "stage9_full_eval"
    if args.comparison_preset == "stage8_soft_decay":
        return "stage8_full_eval"
    if args.comparison_preset == "stage11_calibration_gated_fixed_alpha" or args.include_calibration_gated_fixed_alpha:
        return "stage11_full_eval"
    if args.comparison_preset == "stage7_tqla" or args.include_tqla:
        return "stage7_full_eval"
    if args.comparison_preset == "stage6_trisla" or args.include_trisla:
        return "stage6_full_eval"
    return "stage4_full_eval"


def save_outputs(results_df, summary_df, pairwise_df, metadata, results_dir, artifact_prefix):
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / f"{artifact_prefix}_raw_predictions.csv"
    summary_path = results_dir / f"{artifact_prefix}_summary.csv"
    pairwise_path = results_dir / f"{artifact_prefix}_pairwise_summary.csv"
    metadata_path = results_dir / f"{artifact_prefix}_metadata.json"
    if pd is not None:
        results_df.to_csv(raw_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        pairwise_df.to_csv(pairwise_path, index=False)
    else:
        (results_dir / f"{artifact_prefix}_raw_predictions.json").write_text(
            json.dumps(results_df, indent=2), encoding="utf-8"
        )
        (results_dir / f"{artifact_prefix}_summary.json").write_text(
            json.dumps(summary_df, indent=2), encoding="utf-8"
        )
        (results_dir / f"{artifact_prefix}_pairwise_summary.json").write_text(
            json.dumps(pairwise_df, indent=2), encoding="utf-8"
        )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        {
            "raw_predictions_path": str(raw_path),
            "summary_path": str(summary_path),
            "pairwise_summary_path": str(pairwise_path),
            "metadata_path": str(metadata_path),
        }
    )


def main():
    args = apply_comparison_preset(parse_args())
    torch.manual_seed(args.seed)
    load_strategyqa_rows._forced_dataset = args.strategyqa_dataset
    load_strategyqa_rows._forced_config = args.strategyqa_config
    load_strategyqa_rows._forced_split = args.strategyqa_split
    artifact_prefix = infer_artifact_prefix(args)
    trisla_sweep_configs = build_trisla_sweep_configs(args)

    truthfulqa_limit = resolve_limit(args.truthfulqa_limit, args.mode, 5)
    strategyqa_limit = resolve_limit(args.strategyqa_limit, args.mode, 5)
    gsm8k_limit = resolve_limit(args.gsm8k_limit, args.mode, 5)
    halueval_limit = resolve_limit(args.halueval_limit, args.mode, 5)
    alpacaeval_limit = resolve_limit(args.alpacaeval_limit, args.mode, 5)

    print(
        {
            "truthfulqa_limit": truthfulqa_limit,
            "strategyqa_limit": strategyqa_limit,
            "gsm8k_limit": gsm8k_limit,
            "halueval_limit": halueval_limit,
            "alpacaeval_limit": alpacaeval_limit,
            "include_gsm8k_sequence": args.include_gsm8k_sequence,
            "include_halueval": args.include_halueval,
            "include_alpacaeval": args.include_alpacaeval,
            "save_results": args.save_results,
            "trisla_sweep": trisla_sweep_configs,
        }
    )

    evaluator = Stage4Evaluator(args)

    truthfulqa_rows, truthfulqa_source, truthfulqa_manifest = ([], "disabled", {"sampling_mode": "disabled"})
    strategyqa_rows, strategyqa_source, strategyqa_manifest = ([], "disabled", {"sampling_mode": "disabled"})
    gsm8k_rows, gsm8k_source, gsm8k_manifest = ([], "disabled", {"sampling_mode": "disabled"})
    halueval_rows, halueval_source, halueval_manifest = ([], "disabled", {"sampling_mode": "disabled"})
    alpacaeval_rows, alpacaeval_source, alpacaeval_manifest = ([], "disabled", {"sampling_mode": "disabled"})

    if not args.skip_truthfulqa:
        truthfulqa_rows, truthfulqa_source, truthfulqa_manifest = load_truthfulqa_rows(
            truthfulqa_limit,
            make_sampling_rng(args.seed, "truthfulqa"),
        )
    if not args.skip_strategyqa:
        strategyqa_rows, strategyqa_source, strategyqa_manifest = load_strategyqa_rows(
            strategyqa_limit,
            make_sampling_rng(args.seed, "strategyqa"),
        )
    if not args.skip_gsm8k:
        gsm8k_rows, gsm8k_source, gsm8k_manifest = load_gsm8k_rows(
            gsm8k_limit,
            make_sampling_rng(args.seed, "gsm8k"),
        )
    if args.include_halueval:
        halueval_rows, halueval_source, halueval_manifest = load_halueval_rows(
            halueval_limit,
            make_sampling_rng(args.seed, "halueval"),
            args.halueval_root,
            tuple(part.strip() for part in str(args.halueval_tasks).split(",") if part.strip()),
        )
    if args.include_alpacaeval:
        alpacaeval_rows, alpacaeval_source, alpacaeval_manifest = load_alpacaeval_rows(
            alpacaeval_limit,
            make_sampling_rng(args.seed, "alpacaeval"),
        )

    assert_eval_sources(args, truthfulqa_source, strategyqa_source, gsm8k_source, alpacaeval_source)

    print(
        {
            "truthfulqa_source": truthfulqa_source,
            "truthfulqa_sampling": truthfulqa_manifest,
            "strategyqa_source": strategyqa_source,
            "strategyqa_sampling": strategyqa_manifest,
            "gsm8k_source": gsm8k_source,
            "gsm8k_sampling": gsm8k_manifest,
            "halueval_source": halueval_source,
            "halueval_sampling": halueval_manifest,
            "alpacaeval_source": alpacaeval_source,
            "alpacaeval_sampling": alpacaeval_manifest,
            "truthfulqa_examples": len(truthfulqa_rows),
            "strategyqa_examples": len(strategyqa_rows),
            "gsm8k_examples": len(gsm8k_rows),
            "halueval_examples": len(halueval_rows),
            "alpacaeval_examples": len(alpacaeval_rows),
        }
    )

    all_results = []
    start_time = time.perf_counter()
    if trisla_sweep_configs is None:
        if truthfulqa_rows:
            all_results.extend(evaluate_truthfulqa(evaluator, truthfulqa_rows, args.progress_every))
        if strategyqa_rows:
            all_results.extend(evaluate_strategyqa(evaluator, strategyqa_rows, args.progress_every))
        if gsm8k_rows:
            all_results.extend(evaluate_gsm8k(evaluator, gsm8k_rows, args.progress_every))
            if args.include_gsm8k_sequence:
                all_results.extend(
                    evaluate_gsm8k_sequence(
                        evaluator,
                        gsm8k_rows,
                        args.progress_every,
                        max_new_tokens=args.sequence_max_new_tokens,
                    )
                )
        if halueval_rows:
            all_results.extend(evaluate_halueval(evaluator, halueval_rows, args.progress_every))
    else:
        baseline_decoder_names = tuple(decoder_name for decoder_name in evaluator.decoder_names if decoder_name != "trisla")
        baseline_results = []
        if baseline_decoder_names:
            print({"baseline_decoder_names": baseline_decoder_names})
            if truthfulqa_rows:
                baseline_results.extend(
                    evaluate_truthfulqa(
                        evaluator,
                        truthfulqa_rows,
                        args.progress_every,
                        decoder_names=baseline_decoder_names,
                    )
                )
            if strategyqa_rows:
                baseline_results.extend(
                    evaluate_strategyqa(
                        evaluator,
                        strategyqa_rows,
                        args.progress_every,
                        decoder_names=baseline_decoder_names,
                    )
                )
            if gsm8k_rows:
                baseline_results.extend(
                    evaluate_gsm8k(
                        evaluator,
                        gsm8k_rows,
                        args.progress_every,
                        decoder_names=baseline_decoder_names,
                    )
                )
            if halueval_rows:
                baseline_results.extend(
                    evaluate_halueval(
                        evaluator,
                        halueval_rows,
                        args.progress_every,
                        decoder_names=baseline_decoder_names,
                    )
                )
                if args.include_gsm8k_sequence:
                    baseline_results.extend(
                        evaluate_gsm8k_sequence(
                            evaluator,
                            gsm8k_rows,
                            args.progress_every,
                            decoder_names=baseline_decoder_names,
                            max_new_tokens=args.sequence_max_new_tokens,
                        )
                    )

        for sweep_config in trisla_sweep_configs:
            print({"trisla_sweep_config": sweep_config})
            evaluator.set_trisla_config(
                sweep_config["trisla_window"],
                sweep_config["trisla_lambda_current"],
                sweep_config["trisla_lambda_window"],
                sweep_config["trisla_lambda_prefix"],
            )

            run_results = [copy.deepcopy(row) for row in baseline_results]
            trisla_results = []
            if truthfulqa_rows:
                trisla_results.extend(
                    evaluate_truthfulqa(
                        evaluator,
                        truthfulqa_rows,
                        args.progress_every,
                        decoder_names=("trisla",),
                    )
                )
            if strategyqa_rows:
                trisla_results.extend(
                    evaluate_strategyqa(
                        evaluator,
                        strategyqa_rows,
                        args.progress_every,
                        decoder_names=("trisla",),
                    )
                )
            if gsm8k_rows:
                trisla_results.extend(
                    evaluate_gsm8k(
                        evaluator,
                        gsm8k_rows,
                        args.progress_every,
                        decoder_names=("trisla",),
                    )
                )
            if halueval_rows:
                trisla_results.extend(
                    evaluate_halueval(
                        evaluator,
                        halueval_rows,
                        args.progress_every,
                        decoder_names=("trisla",),
                    )
                )
                if args.include_gsm8k_sequence:
                    trisla_results.extend(
                        evaluate_gsm8k_sequence(
                            evaluator,
                            gsm8k_rows,
                            args.progress_every,
                            decoder_names=("trisla",),
                            max_new_tokens=args.sequence_max_new_tokens,
                        )
                    )
            run_results.extend(trisla_results)
            for row in run_results:
                row["sweep_id"] = sweep_config["sweep_id"]
                row["trisla_window"] = sweep_config["trisla_window"]
                row["trisla_lambda_current"] = sweep_config["trisla_lambda_current"]
                row["trisla_lambda_window"] = sweep_config["trisla_lambda_window"]
                row["trisla_lambda_prefix"] = sweep_config["trisla_lambda_prefix"]
            all_results.extend(run_results)
    elapsed = time.perf_counter() - start_time
    print({"evaluation_seconds": elapsed, "rows": len(all_results)})

    alpacaeval_export_metadata = None
    if args.include_alpacaeval and alpacaeval_rows:
        alpacaeval_export_metadata = export_alpacaeval_outputs(
            evaluator,
            alpacaeval_rows,
            evaluator.decoder_names,
            Path(args.results_dir),
            artifact_prefix,
        )
        print({"alpacaeval_export": alpacaeval_export_metadata})

    if pd is not None:
        results_df = pd.DataFrame(all_results)
        summary_group_columns = ["benchmark", "metric_name", "decoder", "decoder_label"]
        if "sweep_id" in results_df.columns:
            summary_group_columns = [
                "sweep_id",
                "trisla_window",
                "trisla_lambda_current",
                "trisla_lambda_window",
                "trisla_lambda_prefix",
            ] + summary_group_columns
        summary_df = (
            results_df.groupby(summary_group_columns, as_index=False)
            .agg(
                score_mean=("score", "mean"),
                score_std=("score", "std"),
                num_examples=("score", "size"),
                decision_margin_mean=("decision_margin", "mean"),
                correct_margin_mean=("correct_margin", "mean"),
                format_valid_rate=("format_valid", "mean"),
                latency_seconds_mean=("latency_seconds", "mean"),
                decoder_steps_mean=("decoder_steps", "mean"),
                forward_passes_mean=("forward_passes", "mean"),
                latency_per_step_ms_mean=("latency_per_step_ms", "mean"),
                latency_per_forward_ms_mean=("latency_per_forward_ms", "mean"),
                steps_per_second_mean=("steps_per_second", "mean"),
                tokens_per_forward_mean=("tokens_per_forward", "mean"),
                factual_speedup_mean=("factual_speedup", "mean"),
                avg_alpha=("avg_alpha", "mean"),
                alpha_std=("alpha_std", "mean"),
                avg_selected_layer=("avg_selected_layer", "mean"),
                switch_rate=("switch_rate", "mean"),
                avg_instability=("avg_instability", "mean"),
                avg_risk_score=("avg_risk_score", "mean"),
                trigger_rate=("trigger_rate", "mean"),
                avg_jsd_current=("avg_jsd_current", "mean"),
                avg_jsd_window=("avg_jsd_window", "mean"),
                avg_jsd_prefix=("avg_jsd_prefix", "mean"),
                avg_selection_margin=("avg_selection_margin", "mean"),
                avg_history_window_signal=("avg_history_window_signal", "mean"),
                avg_history_prefix_signal=("avg_history_prefix_signal", "mean"),
                avg_history_credit=("avg_history_credit", "mean"),
                avg_selection_score=("avg_selection_score", "mean"),
                fallback_rate=("fallback_rate", "mean"),
                avg_baseline_margin=("avg_baseline_margin", "mean"),
                baseline_guard_rate=("baseline_guard_rate", "mean"),
                positive_signal_rate=("positive_signal_rate", "mean"),
                verification_required_rate=("verification_required_rate", "mean"),
                verification_pass_rate=("verification_pass_rate", "mean"),
                avg_verified_gain=("avg_verified_gain", "mean"),
                avg_layer_penalty=("avg_layer_penalty", "mean"),
                avg_flip_penalty=("avg_flip_penalty", "mean"),
                avg_calibration_confidence=("avg_calibration_confidence", "mean"),
                avg_calibration_confidence_ema=("avg_calibration_confidence_ema", "mean"),
                car_selected_base_rate=("car_selected_base", "mean"),
                car_same_prediction_rate=("car_same_prediction", "mean"),
                car_confidence_base_mean=("car_confidence_base", "mean"),
                car_confidence_fixed_alpha_mean=("car_confidence_fixed_alpha", "mean"),
                car_confidence_base_valid_rate=("car_confidence_base_valid", "mean"),
                car_confidence_fixed_alpha_valid_rate=("car_confidence_fixed_alpha_valid", "mean"),
                car_pairwise_tie_prob_mean=("car_pairwise_tie_prob", "mean"),
                gate_active_rate=("gate_active_rate", "mean"),
                confidence_trigger_rate=("confidence_trigger_rate", "mean"),
                margin_trigger_rate=("margin_trigger_rate", "mean"),
                gate_switch_on_rate=("gate_switch_on_rate", "mean"),
                gate_switch_off_rate=("gate_switch_off_rate", "mean"),
                avg_gate_hold_remaining=("avg_gate_hold_remaining", "mean"),
                switch_blocked_rate=("switch_blocked_rate", "mean"),
                avg_switch_gap=("avg_switch_gap", "mean"),
                avg_jacobi_passes=("avg_jacobi_passes", "mean"),
                avg_jacobi_window_size=("avg_jacobi_window_size", "mean"),
                avg_jacobi_stable_prefix=("avg_jacobi_stable_prefix", "mean"),
                avg_jacobi_commit_len=("avg_jacobi_commit_len", "mean"),
                jacobi_convergence_rate=("jacobi_convergence_rate", "mean"),
                jaca_disagreement_rate=("jaca_disagreement_rate", "mean"),
                jaca_truth_selection_rate=("jaca_truth_selection_rate", "mean"),
                avg_jaca_divergence=("avg_jaca_divergence", "mean"),
                avg_jaca_safe_confidence=("avg_jaca_safe_confidence", "mean"),
                avg_jaca_truth_confidence=("avg_jaca_truth_confidence", "mean"),
                avg_jaca_agreement_prefix=("avg_jaca_agreement_prefix", "mean"),
                jaca_arbitration_rate=("jaca_arbitration_rate", "mean"),
            )
            .sort_values(summary_group_columns)
        )
        summary_df["score_std"] = summary_df["score_std"].fillna(0.0)
        summary_df["score_sem"] = summary_df["score_std"] / summary_df["num_examples"].pow(0.5)
        pairwise_df = build_pairwise_summary(results_df)
        print("\nSummary")
        summary_print_columns = [
            "benchmark",
            "metric_name",
            "decoder_label",
            "score_mean",
            "score_sem",
            "num_examples",
            "decision_margin_mean",
            "latency_seconds_mean",
            "latency_per_step_ms_mean",
            "latency_per_forward_ms_mean",
            "steps_per_second_mean",
            "forward_passes_mean",
            "factual_speedup_mean",
            "avg_alpha",
            "alpha_std",
            "avg_selected_layer",
            "switch_rate",
            "avg_risk_score",
            "trigger_rate",
            "fallback_rate",
            "baseline_guard_rate",
            "positive_signal_rate",
            "verification_required_rate",
            "verification_pass_rate",
            "avg_verified_gain",
            "avg_calibration_confidence",
            "avg_calibration_confidence_ema",
            "car_selected_base_rate",
            "car_same_prediction_rate",
            "car_confidence_base_mean",
            "car_confidence_fixed_alpha_mean",
            "car_confidence_base_valid_rate",
            "car_confidence_fixed_alpha_valid_rate",
            "car_pairwise_tie_prob_mean",
            "gate_active_rate",
            "confidence_trigger_rate",
            "margin_trigger_rate",
            "gate_switch_on_rate",
            "gate_switch_off_rate",
            "switch_blocked_rate",
            "avg_switch_gap",
            "avg_jacobi_passes",
            "avg_jacobi_stable_prefix",
            "avg_jacobi_commit_len",
            "jacobi_convergence_rate",
            "jaca_disagreement_rate",
            "jaca_truth_selection_rate",
            "avg_jaca_divergence",
            "avg_jaca_safe_confidence",
            "avg_jaca_truth_confidence",
            "avg_jaca_agreement_prefix",
            "jaca_arbitration_rate",
        ]
        if "sweep_id" in summary_df.columns:
            summary_print_columns = [
                "sweep_id",
                "trisla_window",
                "trisla_lambda_current",
                "trisla_lambda_window",
                "trisla_lambda_prefix",
            ] + summary_print_columns
        print(summary_df[summary_print_columns].to_string(index=False))
        if not pairwise_df.empty:
            print("\nPairwise Score Deltas")
            pairwise_print_columns = [
                "benchmark",
                "metric_name",
                "left_decoder_label",
                "right_decoder_label",
                "mean_score_delta",
                "win_rate",
                "tie_rate",
                "loss_rate",
                "num_examples",
            ]
            if "sweep_id" in pairwise_df.columns:
                pairwise_print_columns = ["sweep_id"] + pairwise_print_columns
            print(pairwise_df[pairwise_print_columns].to_string(index=False))
    else:
        results_df = all_results
        summary_df = []
        pairwise_df = []
        print("pandas is missing; skipping DataFrame summary.")

    if args.save_results:
        metadata = {
            "model_name": args.model_name,
            "mode": args.mode,
            "seed": args.seed,
            "local_files_only": args.local_files_only,
            "strict_eval": args.strict_eval,
            "artifact_prefix": artifact_prefix,
            "dola_algorithm": "official_dynamic_dola",
            "dola_mature_layer": evaluator.mature_layer_index,
            "dola_relative_top": evaluator.dola_relative_top,
            "dola_relative_top_value": evaluator.dola_relative_top_value,
            "latency_measurement": "wall_clock_seconds_with_cuda_synchronize",
            "decoders": list(evaluator.decoder_names),
            "decoder_labels": evaluator.decoder_labels,
            "trisla_sweep_configs": trisla_sweep_configs,
            "default_shallow_bucket": evaluator.default_bucket,
            "truthfulqa_shallow_bucket": evaluator.truthfulqa_bucket,
            "include_no_ema_ablation": args.include_no_ema_ablation,
            "include_fixed_alpha_ablation": args.include_fixed_alpha_ablation,
            "fixed_alpha_value": args.fixed_alpha_value,
            "include_fixed_layer_ablation": args.include_fixed_layer_ablation,
            "fixed_layer_index": evaluator.fixed_layer_index,
            "include_trisla": args.include_trisla,
            "include_tqla": args.include_tqla,
            "include_soft_decay": args.include_soft_decay,
            "include_pressure_linear": args.include_pressure_linear,
            "include_jacobi": args.include_jacobi,
            "include_jaca": args.include_jaca,
            "include_calibration_gated_fixed_alpha": args.include_calibration_gated_fixed_alpha,
            "include_car_dola": args.include_car_dola,
            "include_alpha_switch_car_dola": args.include_alpha_switch_car_dola,
            "exclude_full_dyndola": args.exclude_full_dyndola,
            "include_gsm8k_sequence": args.include_gsm8k_sequence,
            "sequence_max_new_tokens": args.sequence_max_new_tokens,
            "trisla_window": args.trisla_window,
            "trisla_lambda_current": args.trisla_lambda_current,
            "trisla_lambda_window": args.trisla_lambda_window,
            "trisla_lambda_prefix": args.trisla_lambda_prefix,
            "trisla_jsd_temperature": args.trisla_jsd_temperature,
            "trisla_margin_epsilon": args.trisla_margin_epsilon,
            "trisla_switch_margin": args.trisla_switch_margin,
            "tqla_window": args.tqla_window,
            "tqla_lambda_current": args.tqla_lambda_current,
            "tqla_lambda_window": args.tqla_lambda_window,
            "tqla_lambda_prefix": args.tqla_lambda_prefix,
            "tqla_utility_epsilon": args.tqla_utility_epsilon,
            "tqla_baseline_margin_threshold": args.tqla_baseline_margin_threshold,
            "tqla_layer_deviation_penalty": args.tqla_layer_deviation_penalty,
            "tqla_flip_penalty": args.tqla_flip_penalty,
            "tqla_verify_epsilon": args.tqla_verify_epsilon,
            "tqla_require_positive_signals": args.tqla_require_positive_signals,
            "tqla_verify_top1_override": args.tqla_verify_top1_override,
            "soft_decay_alpha_base": evaluator.soft_decay_alpha_base,
            "soft_decay_alpha_peak": evaluator.soft_decay_alpha_peak,
            "soft_decay_half_life": evaluator.soft_decay_half_life,
            "soft_decay_decay_rho": evaluator.soft_decay_decay_rho,
            "soft_decay_trigger_threshold": args.soft_decay_trigger_threshold,
            "pressure_alpha_base": evaluator.pressure_alpha_base,
            "pressure_alpha_peak": evaluator.pressure_alpha_peak,
            "pressure_alpha_momentum": evaluator.pressure_alpha_momentum,
            "pressure_alpha_risk_source": "max(0, normalized_instability)",
            "calibration_gate_ema_decay": args.calibration_gate_ema_decay,
            "calibration_gate_confidence_on": args.calibration_gate_confidence_on,
            "calibration_gate_confidence_off": args.calibration_gate_confidence_off,
            "calibration_gate_margin_on": args.calibration_gate_margin_on,
            "calibration_gate_hold_steps": args.calibration_gate_hold_steps,
            "calibration_gate_confidence_source": evaluator.calibration_gate_confidence_source,
            "calibration_gate_signal_note": (
                "pilot_uses_base_top1_probability_from_self_calibrated_model_logits_as_confidence_signal"
            ),
            "calibration_rerank_delta": evaluator.calibration_rerank_delta,
            "calibration_confidence_max_new_tokens": evaluator.calibration_confidence_max_new_tokens,
            "alpha_switch_low": evaluator.alpha_switch_low,
            "alpha_switch_high": evaluator.alpha_switch_high,
            "calibration_rerank_note": (
                "alpha_switch_car_dola_uses_high_alpha_for_same-prediction_truthfulqa_mc_cases_and_forced_pairwise_A_B_candidate_comparison_between_low_and_high_fixed_alpha_when_their_predictions_disagree"
                if args.comparison_preset == "stage11c_alpha_switch" or args.include_alpha_switch_car_dola
                else "car_dola_uses_forced_pairwise_A_B_candidate_comparison_scored_by_sequence_logprob_only_when_base_and_fixed_alpha_disagree"
            ),
            "jacobi_window_size": evaluator.jacobi_window_size,
            "jacobi_max_iters": evaluator.jacobi_max_iters,
            "jacobi_init_strategy": "repeat_last",
            "jacobi_commit_strategy": "stable_prefix_then_fallback_1",
            "jaca_divergence_threshold": evaluator.jaca_divergence_threshold,
            "jaca_truth_bias": evaluator.jaca_truth_bias,
            "jaca_note": (
                "jacobi_block_decoding_with_shared_low_high_fixed_alpha_views_and_local_truth_biased_arbitration"
            ),
            "jaca_early_agreement_shortcut": evaluator.jaca_early_agreement_shortcut,
            "truthfulqa_source": truthfulqa_source,
            "strategyqa_source": strategyqa_source,
            "gsm8k_source": gsm8k_source,
            "halueval_source": halueval_source,
            "alpacaeval_source": alpacaeval_source,
            "truthfulqa_sampling": truthfulqa_manifest,
            "strategyqa_sampling": strategyqa_manifest,
            "gsm8k_sampling": gsm8k_manifest,
            "halueval_sampling": halueval_manifest,
            "alpacaeval_sampling": alpacaeval_manifest,
            "truthfulqa_limit": truthfulqa_limit,
            "strategyqa_limit": strategyqa_limit,
            "gsm8k_limit": gsm8k_limit,
            "halueval_limit": halueval_limit,
            "alpacaeval_limit": alpacaeval_limit,
            "include_halueval": args.include_halueval,
            "halueval_root": args.halueval_root,
            "halueval_tasks": tuple(part.strip() for part in str(args.halueval_tasks).split(",") if part.strip()),
            "include_alpacaeval": args.include_alpacaeval,
            "alpacaeval_export": alpacaeval_export_metadata,
        }
        save_outputs(results_df, summary_df, pairwise_df, metadata, Path(args.results_dir), artifact_prefix)


if __name__ == "__main__":
    main()

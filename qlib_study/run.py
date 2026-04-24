"""
qlib_study/run.py — one-stop runner for the Alpha158 baseline / Alpha158Enhanced
variants, with a metrics table that slots next to your v6 report.

Usage:
    python qlib_study/run.py baseline
    python qlib_study/run.py enhanced
    python qlib_study/run.py both           # runs both and diffs the metrics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CONFIGS = {
    "baseline": HERE / "configs" / "workflow_lightgbm_alpha158.yaml",
    "enhanced": HERE / "configs" / "workflow_lightgbm_enhanced.yaml",
}

METRIC_KEYS = [
    "IC", "ICIR", "Rank IC", "Rank ICIR",
    "excess_return_without_cost.annualized_return",
    "excess_return_without_cost.information_ratio",
    "excess_return_without_cost.max_drawdown",
    "excess_return_with_cost.annualized_return",
    "excess_return_with_cost.information_ratio",
    "excess_return_with_cost.max_drawdown",
]


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _init_qlib(cfg: dict) -> None:
    import qlib
    init_kwargs = cfg["qlib_init"]
    init_kwargs["provider_uri"] = str(Path(init_kwargs["provider_uri"]).expanduser())
    qlib.init(**init_kwargs)


def _run_task(cfg: dict, experiment_name: str) -> dict:
    """
    Execute the task dict via qlib's R (Recorder) API. Returns the flat metrics
    dict pulled from the finished recorder.
    """
    from qlib.model.trainer import task_train
    from qlib.workflow import R

    task = cfg["task"]
    recorder = task_train(task, experiment_name=experiment_name)
    metrics = recorder.list_metrics()
    logger.info("recorder id=%s", recorder.id)
    return metrics


def _flatten_metrics(m: dict) -> dict:
    return {k: m.get(k, float("nan")) for k in METRIC_KEYS}


def _print_report(label: str, metrics: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)
    for k, v in metrics.items():
        try:
            print(f"  {k:<55} {v: .4f}")
        except (TypeError, ValueError):
            print(f"  {k:<55} {v}")


def _diff_report(baseline: dict, enhanced: dict) -> None:
    print("\n" + "=" * 70)
    print("  baseline vs enhanced — Δ (enhanced − baseline)")
    print("=" * 70)
    rows = []
    for k in METRIC_KEYS:
        b, e = baseline.get(k, float("nan")), enhanced.get(k, float("nan"))
        try:
            delta = e - b
            rows.append((k, b, e, delta))
        except TypeError:
            rows.append((k, b, e, None))

    df = pd.DataFrame(rows, columns=["metric", "baseline", "enhanced", "delta"])
    print(df.to_string(index=False, float_format=lambda x: f"{x: .4f}"))


def run(variant: str) -> dict:
    cfg = _load_config(CONFIGS[variant])
    _init_qlib(cfg)
    return _flatten_metrics(_run_task(cfg, experiment_name=f"qlib_study_{variant}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["baseline", "enhanced", "both"])
    args = ap.parse_args()

    # make the handler module importable when qlib loads it by path
    sys.path.insert(0, str(HERE.parent))

    if args.variant == "both":
        b = run("baseline")
        _print_report("baseline (Alpha158)", b)
        e = run("enhanced")
        _print_report("enhanced (Alpha158 + fundamentals)", e)
        _diff_report(b, e)
    else:
        m = run(args.variant)
        _print_report(f"{args.variant}", m)


if __name__ == "__main__":
    main()

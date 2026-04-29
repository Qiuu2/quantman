"""
qlib_study/scripts/mlflow_diff.py

Run baseline (Alpha158) and enhanced (Alpha158Enhanced) end-to-end and produce
a three-panel diagnostic comparing them. Skips training when a recent recorder
already exists; pass --rerun to force fresh runs.

Panels (left → right):
    1. cumulative excess return (with cost)   — equity-curve view
    2. running drawdown of that excess return — risk view
    3. rolling-window IC (default 21d)        — signal stability view

Usage:
    python qlib_study/scripts/mlflow_diff.py
    python qlib_study/scripts/mlflow_diff.py --rerun --window 42
    python qlib_study/scripts/mlflow_diff.py --out reports/qlib_diff/may.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

VARIANTS = {
    "baseline": "qlib_study_baseline",
    "enhanced": "qlib_study_enhanced",
}
COLORS = {"baseline": "#1f77b4", "enhanced": "#d62728"}


def _init_qlib() -> None:
    import qlib
    qlib.init(provider_uri=str(Path.home() / ".qlib/qlib_data/us_data"), region="us")


def _latest_recorder_or_none(exp_name: str):
    from qlib.workflow import R
    try:
        exp = R.get_exp(experiment_name=exp_name, create=False)
    except Exception:
        return None
    recs = list(exp.list_recorders().values())
    if not recs:
        return None
    recs.sort(key=lambda r: r.info.get("start_time", ""), reverse=True)
    return recs[0]


def _ensure_run(variant: str, force: bool):
    exp_name = VARIANTS[variant]
    rec = _latest_recorder_or_none(exp_name)
    if rec is not None and not force:
        print(f"[{variant}] reusing recorder {rec.id[:8]}…")
        return rec

    print(f"[{variant}] training (this can take a while)…")
    from qlib_study.run import run as qlib_run
    qlib_run(variant)
    rec = _latest_recorder_or_none(exp_name)
    if rec is None:
        raise RuntimeError(f"training finished but no recorder found under {exp_name}")
    return rec


def _load_run_data(rec) -> dict:
    """Pull the three series we need from one recorder."""
    port = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
    ic = rec.load_object("sig_analysis/ic.pkl")
    metrics = rec.list_metrics()

    excess = port["return"] - port["bench"] - port.get("cost", 0.0)
    cum = (1 + excess).cumprod() - 1
    running_max = (1 + cum).cummax()
    drawdown = (1 + cum) / running_max - 1

    return {
        "excess_daily": excess,
        "cum_excess": cum,
        "drawdown": drawdown,
        "ic_daily": ic,
        "metrics": metrics,
    }


def _summary_row(variant: str, data: dict) -> dict:
    m = data["metrics"]
    return {
        "variant": variant,
        "ann_ret(w/cost)": m.get("1day.excess_return_with_cost.annualized_return"),
        "IR(w/cost)":      m.get("1day.excess_return_with_cost.information_ratio"),
        "MDD(w/cost)":     m.get("1day.excess_return_with_cost.max_drawdown"),
        "IC":              m.get("IC"),
        "ICIR":            m.get("ICIR"),
        "Rank IC":         m.get("Rank IC"),
    }


def _plot(runs: dict[str, dict], window: int, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_cum, ax_dd, ax_ic = axes

    for variant, data in runs.items():
        color = COLORS[variant]
        m = data["metrics"]

        # panel 1: cumulative excess return
        cum = data["cum_excess"]
        ann = m.get("1day.excess_return_with_cost.annualized_return", float("nan"))
        ax_cum.plot(cum.index, cum.values * 100, label=f"{variant}  (ann {ann:.2%})",
                    color=color, linewidth=1.6)

        # panel 2: drawdown
        dd = data["drawdown"]
        mdd = m.get("1day.excess_return_with_cost.max_drawdown", float("nan"))
        ax_dd.fill_between(dd.index, dd.values * 100, 0, color=color, alpha=0.25)
        ax_dd.plot(dd.index, dd.values * 100, label=f"{variant}  (MDD {mdd:.2%})",
                   color=color, linewidth=1.2)

        # panel 3: rolling IC
        ic = data["ic_daily"].dropna()
        roll = ic.rolling(window).mean()
        ic_mean = m.get("IC", float("nan"))
        ax_ic.plot(roll.index, roll.values, label=f"{variant}  (IC {ic_mean:.4f})",
                   color=color, linewidth=1.4)

    ax_cum.set_title("cumulative excess return  (with cost)")
    ax_cum.set_ylabel("cumulative excess (%)")
    ax_cum.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax_cum.legend(loc="upper left")
    ax_cum.grid(True, alpha=0.3)

    ax_dd.set_title("running drawdown of excess return")
    ax_dd.set_ylabel("drawdown (%)")
    ax_dd.legend(loc="lower left")
    ax_dd.grid(True, alpha=0.3)

    ax_ic.set_title(f"rolling IC  (window={window} days)")
    ax_ic.set_ylabel("rolling-mean IC")
    ax_ic.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax_ic.legend(loc="upper left")
    ax_ic.grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("baseline vs enhanced — Qlib mlflow diff", fontsize=13, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


def _print_summary_table(runs: dict[str, dict]) -> None:
    rows = [_summary_row(v, d) for v, d in runs.items()]
    df = pd.DataFrame(rows).set_index("variant")

    # diff row: enhanced − baseline
    if "baseline" in df.index and "enhanced" in df.index:
        df.loc["Δ (enh−base)"] = df.loc["enhanced"] - df.loc["baseline"]

    print("\n" + "=" * 76)
    print("  summary  (negative MDD is bigger drawdown)")
    print("=" * 76)
    print(df.to_string(float_format=lambda x: f"{x: .4f}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true",
                    help="force fresh training even if a recorder exists")
    ap.add_argument("--window", type=int, default=21,
                    help="rolling window for IC panel (days)")
    ap.add_argument("--out", type=Path, default=None,
                    help="PNG output path; default reports/qlib_diff/diff_<ts>.png")
    args = ap.parse_args()

    _init_qlib()

    runs = {}
    for variant in ["baseline", "enhanced"]:
        rec = _ensure_run(variant, force=args.rerun)
        runs[variant] = _load_run_data(rec)

    _print_summary_table(runs)

    out = args.out or PROJECT_ROOT / "reports" / "qlib_diff" / \
          f"diff_{datetime.now():%Y%m%d_%H%M%S}.png"
    _plot(runs, window=args.window, out=out)


if __name__ == "__main__":
    main()

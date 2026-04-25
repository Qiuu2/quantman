"""
Read a Qlib mlflow recorder and print a flat v6-style report.

Usage:
    python qlib_study/scripts/read_recorder.py                 # latest baseline run
    python qlib_study/scripts/read_recorder.py --exp qlib_study_enhanced
    python qlib_study/scripts/read_recorder.py --recorder_id <hex>
    python qlib_study/scripts/read_recorder.py --csv reports/qlib_baseline.csv

Layout cheatsheet (where each metric lives in mlruns/<exp>/<run>/):
    metrics/IC                              | scalar IC (mean of daily IC)
    metrics/ICIR                            | IC / std(IC) * sqrt(252)
    metrics/Rank IC, Rank ICIR              | rank-IC variants
    metrics/1day.excess_return_with_cost.*  | annualized_return / IR / max_drawdown
    artifacts/pred.pkl                      | DataFrame, idx=(datetime,instrument), col=score
    artifacts/label.pkl                     | DataFrame, true LABEL0 next-day returns
    artifacts/sig_analysis/ic.pkl           | daily IC time series
    artifacts/sig_analysis/ric.pkl          | daily Rank IC time series
    artifacts/portfolio_analysis/
        report_normal_1day.pkl              | daily return/bench/cost/turnover columns
        positions_normal_1day.pkl           | daily holdings
        port_analysis_1day.pkl              | final risk_analysis table (Sharpe, MDD, IR)
        indicator_analysis_1day.pkl         | trade indicators (FFR, PA, PS)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)


def _init_qlib_default() -> None:
    import qlib
    qlib.init(provider_uri=str(Path.home() / ".qlib/qlib_data/us_data"), region="us")


def _latest_recorder(experiment_name: str, recorder_id: str | None):
    from qlib.workflow import R
    exp = R.get_exp(experiment_name=experiment_name)
    if recorder_id:
        return exp.get_recorder(recorder_id=recorder_id)
    recs = list(exp.list_recorders().values())
    if not recs:
        raise SystemExit(f"no recorders under experiment {experiment_name!r}; did you run yet?")
    # newest first by start_time
    recs.sort(key=lambda r: r.info.get("start_time", ""), reverse=True)
    return recs[0]


def _safe_load(rec, name: str, sub: str | None = None):
    try:
        return rec.load_object(f"{sub}/{name}" if sub else name)
    except Exception as e:
        return None


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _kv(k: str, v, fmt: str = "{: .4f}") -> None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        print(f"  {k:<48} {'(missing)':>16}")
    elif isinstance(v, (int, float)):
        print(f"  {k:<48} {fmt.format(v):>16}")
    else:
        print(f"  {k:<48} {str(v):>16}")


def report(rec, csv_path: Path | None) -> None:
    metrics = rec.list_metrics()

    _section(f"recorder  {rec.id}   experiment={rec.info.get('experiment_id')}")
    _kv("status", rec.info.get("status"))
    _kv("start_time", rec.info.get("start_time"), fmt="{}")

    # --- signal metrics (IC family) ----------------------------------------
    _section("signal metrics  (predictive power, no costs)")
    for k in ["IC", "ICIR", "Rank IC", "Rank ICIR"]:
        _kv(k, metrics.get(k))

    # --- portfolio metrics (TopkDropoutStrategy backtest) ------------------
    _section("portfolio metrics  (after costs)  ←→ v6's 'Return / Sharpe / MaxDD'")
    pf_keys = [
        ("annualized_return (no cost)", "1day.excess_return_without_cost.annualized_return"),
        ("information_ratio (no cost)", "1day.excess_return_without_cost.information_ratio"),
        ("max_drawdown      (no cost)", "1day.excess_return_without_cost.max_drawdown"),
        ("annualized_return (w/  cost)", "1day.excess_return_with_cost.annualized_return"),
        ("information_ratio (w/  cost)", "1day.excess_return_with_cost.information_ratio"),
        ("max_drawdown      (w/  cost)", "1day.excess_return_with_cost.max_drawdown"),
    ]
    for label, key in pf_keys:
        _kv(label, metrics.get(key))

    # --- daily IC time series -> mean/std/positive-rate, the v6 'IC quote' --
    ic_ts = _safe_load(rec, "ic.pkl", sub="sig_analysis")
    if ic_ts is not None:
        s = ic_ts.dropna()
        _section("IC time series  ←→ v6 'IC summary'")
        _kv("daily IC mean",  s.mean())
        _kv("daily IC std",   s.std())
        _kv("annualized ICIR", s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else float("nan"))
        _kv("positive IC ratio", (s > 0).mean())

    # --- daily portfolio report -> turnover, cost, hit rate ----------------
    port = _safe_load(rec, "report_normal_1day.pkl", sub="portfolio_analysis")
    if port is not None:
        _section("daily portfolio  ←→ v6 'Turnover / TC'")
        _kv("avg daily return", port["return"].mean())
        _kv("avg daily bench",  port["bench"].mean())
        _kv("avg daily cost",   port.get("cost", pd.Series([np.nan])).mean())
        if "turnover" in port.columns:
            _kv("avg daily turnover", port["turnover"].mean())
            _kv("annualized turnover", port["turnover"].sum() * 252 / len(port))

    # --- predictions head/tail (for sanity) --------------------------------
    pred = _safe_load(rec, "pred.pkl")
    if pred is not None:
        _section("pred.pkl preview  (idx=(datetime, instrument))")
        print(pred.head(5).to_string())
        print("  ...")
        print(pred.tail(5).to_string())
        _kv("rows", len(pred), fmt="{}")
        _kv("date range", f"{pred.index.get_level_values('datetime').min()} → "
                          f"{pred.index.get_level_values('datetime').max()}", fmt="{}")
        _kv("unique instruments", pred.index.get_level_values("instrument").nunique(), fmt="{}")

    # --- indicator analysis (trade execution stats) ------------------------
    ind_ana = _safe_load(rec, "indicator_analysis_1day.pkl", sub="portfolio_analysis")
    if ind_ana is not None:
        _section("trade indicators  (FFR / PA / PS)")
        for name, val in ind_ana["value"].items():
            _kv(str(name), val)

    # --- export to CSV -----------------------------------------------------
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"metric": k, "value": v} for k, v in metrics.items()]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nwrote flat metrics → {csv_path}")

    # --- pointers -----------------------------------------------------------
    _section("how to dig deeper")
    print("  mlflow ui --backend-store-uri ./mlruns        # web UI on :5000")
    print("  python -c \"import pickle; print(pickle.load(open('mlruns/.../pred.pkl','rb')).head())\"")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="qlib_study_baseline",
                    help="experiment name (qlib_study_baseline | qlib_study_enhanced)")
    ap.add_argument("--recorder_id", default=None, help="specific recorder id; default = latest")
    ap.add_argument("--csv", type=Path, default=None, help="export flat metrics here")
    args = ap.parse_args()

    _init_qlib_default()
    rec = _latest_recorder(args.exp, args.recorder_id)
    report(rec, args.csv)


if __name__ == "__main__":
    main()

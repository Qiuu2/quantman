"""
Export v6 fundamental / sentiment / macro panels into Qlib's bin format.

Qlib stores each (instrument, field) timeseries as a little-endian float32 bin
file under provider_uri/features/<lower_ticker>/<lower_field>.day.bin. The
canonical way to write one is via qlib's scripts/dump_bin.py which takes a
directory of per-instrument CSVs.

Pipeline:
  1. Build fundamental panels using the existing enhanced_factors module.
  2. Emit one CSV per ticker to a staging dir:
       staging/AAPL.csv  →  date,roe,gross_margin,eps_surprise,term_spread,...
  3. Invoke qlib's dump_bin.py (update mode) to merge these into the
     existing us_data bin directory alongside $close, $volume, etc.

Run:
    python qlib_study/scripts/export_fundamentals_to_qlib.py \
        --qlib_dir ~/.qlib/qlib_data/us_data

Idempotent: rerunning overwrites just the fundamental fields.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# project root on path so we can import enhanced_factors / finnhub_client / ...
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

FUNDAMENTAL_FIELDS = [
    "roe", "roa", "gross_margin", "operating_margin", "net_margin",
    "pe", "pb", "ps", "debt_to_equity", "current_ratio", "eps",
    "eps_surprise", "analyst_consensus", "target_upside",
    "term_spread", "macro_regime",
]


def load_panels(dates: pd.DatetimeIndex, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Build the combined panel dict: {field_name: DataFrame(index=date, columns=ticker)}.

    Delegates to the existing enhanced_factors + finnhub_client + fred_client so
    we reuse every cache and data-cleaning rule already baked into v5/v6.
    """
    from enhanced_factors import (
        build_fundamental_panel_from_finnhub,
        build_sentiment_panel_from_finnhub,
        build_macro_panel,
    )
    from finnhub_client import (
        fetch_all_fundamentals,
        fetch_all_eps_surprises,
        fetch_all_analyst_ratings,
        fetch_all_price_targets,
    )
    from fred_client import fetch_fred_data

    raw_fund = fetch_all_fundamentals(tickers)
    fund_panel = build_fundamental_panel_from_finnhub(raw_fund, dates, tickers)

    eps = fetch_all_eps_surprises(tickers)
    ratings = fetch_all_analyst_ratings(tickers)
    targets = fetch_all_price_targets(tickers)
    sent_panel = build_sentiment_panel_from_finnhub(eps, ratings, targets, dates, tickers)

    fred_df = fetch_fred_data(start=dates.min(), end=dates.max())
    macro_panel = build_macro_panel(fred_df, dates)

    # macro is Series → broadcast to all tickers
    macro_broadcast = {}
    for name, s in macro_panel.items():
        macro_broadcast[name] = pd.DataFrame({t: s for t in tickers}, index=dates)

    combined = {**fund_panel, **sent_panel, **macro_broadcast}
    return {k: v for k, v in combined.items() if k in FUNDAMENTAL_FIELDS}


def emit_per_ticker_csvs(panels: dict[str, pd.DataFrame], out_dir: Path) -> int:
    """Pivot panel dict to per-ticker CSVs with schema [date, field1, field2, ...]."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tickers = set()
    for df in panels.values():
        all_tickers.update(df.columns)

    written = 0
    for ticker in sorted(all_tickers):
        cols = {}
        for field, df in panels.items():
            if ticker in df.columns:
                cols[field] = df[ticker]
        if not cols:
            continue
        ticker_df = pd.DataFrame(cols)
        ticker_df.index.name = "date"
        ticker_df = ticker_df.reset_index()
        ticker_df.to_csv(out_dir / f"{ticker}.csv", index=False)
        written += 1

    logger.info(f"wrote {written} per-ticker CSVs to {out_dir}")
    return written


def run_dump_bin(staging_dir: Path, qlib_dir: Path, mode: str = "update") -> None:
    """
    Invoke qlib's scripts/dump_bin.py. Two-step shell call for transparency:
        dump_bin.py dump_update --csv_path=... --qlib_dir=... --freq=day ...
    """
    try:
        import qlib
    except ImportError as e:
        raise SystemExit("pyqlib not installed; run setup.sh first") from e

    dump_bin = Path(qlib.__file__).resolve().parent.parent / "scripts" / "dump_bin.py"
    if not dump_bin.exists():
        # pyqlib wheel sometimes ships this under a different path
        candidates = list(Path(qlib.__file__).resolve().parent.parent.rglob("dump_bin.py"))
        if not candidates:
            raise FileNotFoundError("dump_bin.py not found; install qlib from source")
        dump_bin = candidates[0]

    cmd = [
        sys.executable, str(dump_bin),
        f"dump_{mode}",
        f"--csv_path={staging_dir}",
        f"--qlib_dir={qlib_dir}",
        "--freq=day",
        "--date_field_name=date",
        "--file_suffix=.csv",
        f"--include_fields={','.join(FUNDAMENTAL_FIELDS)}",
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib_dir", type=Path, default=Path.home() / ".qlib/qlib_data/us_data")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--mode", choices=["all", "update"], default="update",
                    help="all = rebuild everything (destructive), update = merge fields")
    ap.add_argument("--keep_staging", action="store_true")
    args = ap.parse_args()

    dates = pd.bdate_range(args.start, args.end)

    logger.info("loading sp500 tickers from data_loader.get_sp500_tickers()")
    from data_loader import get_sp500_tickers
    tickers = get_sp500_tickers()

    logger.info("building fundamental/sentiment/macro panels for %d tickers", len(tickers))
    panels = load_panels(dates, tickers)
    logger.info("panels built: %s", sorted(panels.keys()))

    staging = Path(tempfile.mkdtemp(prefix="qlib_fund_csv_"))
    try:
        emit_per_ticker_csvs(panels, staging)
        run_dump_bin(staging, args.qlib_dir, mode=args.mode)
    finally:
        if not args.keep_staging:
            shutil.rmtree(staging, ignore_errors=True)
        else:
            logger.info("kept staging dir: %s", staging)

    logger.info("done. verify with:")
    logger.info("  ls %s/features/aapl/ | grep -E 'roe|gross_margin|eps_surprise'", args.qlib_dir)


if __name__ == "__main__":
    main()

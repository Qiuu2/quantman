"""
v5_main.py - QuantMan v5: Enhanced Data Sources

v0: 原始 12-1 动量（基线）
v3b: 动量(0.6) + 质量(0.4)（旧基本面快照）
v5a: v3b + EPS 超预期情绪信号（月度更新）
v5b: v3b + 月度基本面过滤 + EPS 超预期

核心策略：
- 日频动量/质量因子负责选股排序（不变）
- 基本面因子做月度过滤器（排除基本面差的股票）
- EPS 超预期做短期情绪信号
"""

import logging
from pathlib import Path

import dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from data_loader import load_or_download, get_close_prices, load_config
from factors import momentum_12_1, combine_factors
from fundamental_factors import fetch_fundamental_data, build_fundamental_panel
from portfolio import construct_portfolio, get_rebalance_dates
from backtest import compute_returns, run_backtest, run_benchmark
from metrics import print_report, evaluate

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v5_main")


def apply_monthly_filter(
    daily_factor: pd.DataFrame,
    filter_panel: pd.DataFrame,
    direction: str = "positive",
    threshold_pct: float = 0.3,
) -> pd.DataFrame:
    """
    用月度更新的基本面过滤器筛选股票。

    逻辑：在每个月末，计算过滤器因子的截面排名，
    只保留排名前 threshold_pct（或后，如果 direction=negative）的股票。
    日频因子中不在筛选范围内的股票设为 NaN。

    Args:
        daily_factor: 日频因子面板 (Date x Ticker)
        filter_panel: 过滤器面板 (Date x Ticker)，可以是稀疏的
        direction: "positive"=保留高排名, "negative"=保留低排名
        threshold_pct: 保留比例

    Returns:
        DataFrame: 过滤后的日频因子面板
    """
    logger.info(f"应用月度过滤器 (direction={direction}, threshold={threshold_pct:.0%})...")

    filtered = daily_factor.copy()
    rebalance_dates = get_rebalance_dates(daily_factor.index, freq="ME")

    for rebal_date in rebalance_dates:
        mask = filter_panel.index <= rebal_date
        if mask.sum() == 0:
            continue

        latest_filter_idx = filter_panel.index[mask][-1]
        filter_row = filter_panel.loc[latest_filter_idx].dropna()

        if len(filter_row) < 20:
            continue

        ranked = filter_row.rank(pct=True)

        if direction == "positive":
            passing = ranked[ranked >= (1 - threshold_pct)].index
        else:
            passing = ranked[ranked <= threshold_pct].index

        # Apply filter for the next month
        future_mask = rebalance_dates > rebal_date
        if future_mask.any():
            next_rebal = rebalance_dates[future_mask].iloc[0]
            apply_mask = (daily_factor.index > rebal_date) & (daily_factor.index <= next_rebal)
        else:
            apply_mask = daily_factor.index > rebal_date

        non_passing = [t for t in filtered.columns if t not in passing]
        filtered.loc[apply_mask, non_passing] = np.nan

    n_remaining = filtered.notna().sum(axis=1).mean()
    n_total = len(filtered.columns)
    logger.info(f"月度过滤后: 平均每日 {n_remaining:.0f}/{n_total} 只股票有效")

    return filtered


def plot_v5_comparison(results, benchmark_cumulative, save_dir="reports"):
    """Generate multi-strategy comparison chart"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle("QuantMan v0 vs v3b vs v5: Enhanced Data Sources",
                 fontsize=16, fontweight="bold", y=0.98)

    colors = {"v0": "#9E9E9E", "v3b": "#4CAF50", "v5a": "#2196F3", "v5b": "#E91E63"}

    # Panel 1: Cumulative
    ax1 = axes[0]
    for name, r in results.items():
        style = "--" if name == "v0" else "-"
        lw = 1.5 if name == "v0" else 2.2
        ax1.plot(r["cumulative_returns"].index, r["cumulative_returns"].values,
                 label=r["label"], color=colors.get(name, "#000"),
                 linewidth=lw, linestyle=style)
    if benchmark_cumulative is not None:
        ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values,
                 label="SPY (Buy & Hold)", color="#FF9800", linewidth=1.5, alpha=0.8)
    ax1.set_ylabel("Cumulative Returns")
    ax1.set_title("Cumulative Performance")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.tick_params(axis="x", rotation=30)

    # Panel 2: Drawdown
    ax2 = axes[1]
    for name, r in results.items():
        ax2.fill_between(r["drawdown"].index, r["drawdown"].values, 0,
                         color=colors.get(name, "#000"), alpha=0.4, label=r["label"])
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown Comparison")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.tick_params(axis="x", rotation=30)

    # Panel 3: Rolling Sharpe
    ax3 = axes[2]
    for name, r in results.items():
        ret = r["portfolio_returns"]
        rs = ret.rolling(60).mean() / ret.rolling(60).std() * np.sqrt(252)
        style = "--" if name == "v0" else "-"
        lw = 0.8 if name == "v0" else 1.2
        ax3.plot(rs.index, rs.values, color=colors.get(name, "#000"),
                 linewidth=lw, linestyle=style, label=r["label"])
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.axhline(y=1, color="#4CAF50", linestyle=":", alpha=0.3)
    ax3.set_ylabel("Sharpe (60D)")
    ax3.set_title("Rolling 60-Day Sharpe Ratio")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax3.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    report_file = save_path / "v0_vs_v3b_vs_v5_comparison.png"
    fig.savefig(report_file, dpi=150, bbox_inches="tight")
    logger.info(f"Report saved: {report_file}")
    plt.close(fig)
    return report_file


def main():
    logger.info("=" * 60)
    logger.info("  QuantMan v5: Enhanced Data Sources")
    logger.info("  v0:  Raw 12-1 Momentum")
    logger.info("  v3b: Momentum(0.6) + Quality(0.4) [snapshot]")
    logger.info("  v5a: v3b + EPS Surprise (monthly signal)")
    logger.info("  v5b: v3b + Monthly Fundamental Filter + EPS")
    logger.info("=" * 60)

    # Load .env
    env_path = Path(".env")
    if env_path.exists():
        dotenv.load_dotenv(env_path)

    config = load_config("config.json")
    benchmark_ticker = config["data"]["benchmark_ticker"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    logger.info(f"Backtest: {start_date} ~ {end_date}")

    # Step 1: Price data
    logger.info("\n[Step 1/6] Loading price data...")
    fields = load_or_download("config.json")
    close = get_close_prices(fields)
    stock_close = close.drop(columns=[benchmark_ticker], errors="ignore")
    tickers = stock_close.columns.tolist()
    dates = stock_close.index
    all_returns = compute_returns(close)

    # Step 2: Momentum factor (v0 baseline)
    logger.info("\n[Step 2/6] Computing momentum factor...")
    lookback = config["strategy"]["momentum_lookback"]
    skip = config["strategy"]["momentum_skip"]
    momentum = momentum_12_1(stock_close, lookback=lookback, skip=skip)

    # Step 3: Fundamental snapshot (v3b)
    logger.info("\n[Step 3/6] Loading fundamental snapshots (v3b)...")
    fund = fetch_fundamental_data(tickers)
    quality_panel = build_fundamental_panel(fund, dates, "quality")

    # v3b composite
    composite_v3b = combine_factors(
        {"momentum": momentum, "quality": quality_panel},
        {"momentum": 0.6, "quality": 0.4},
    )

    # Step 4: Fetch Finnhub data
    logger.info("\n[Step 4/6] Loading Finnhub data...")
    from finnhub_client import FinnhubClient

    finnhub_key = dotenv.get_key(env_path, "FINNHUB_API_KEY")
    finnhub_client = None
    eps_df = pd.DataFrame()
    ratings_df = pd.DataFrame()
    targets_df = pd.DataFrame()
    fundamental_panels = {}

    if finnhub_key:
        finnhub_client = FinnhubClient(finnhub_key)

        logger.info("  4a. Loading EPS surprises...")
        eps_df = finnhub_client.fetch_all_eps_surprises(tickers)
        logger.info(f"  EPS data: {len(eps_df)} records")

        logger.info("  4b. Loading analyst ratings...")
        ratings_df = finnhub_client.fetch_all_analyst_ratings(tickers)
        logger.info(f"  Ratings data: {len(ratings_df)} records")

        logger.info("  4c. Loading price targets...")
        targets_df = finnhub_client.fetch_all_price_targets(tickers)
        logger.info(f"  Targets data: {len(targets_df)} records")

        logger.info("  4d. Loading annual fundamentals...")
        raw_fundamentals = finnhub_client.fetch_all_fundamentals(tickers)
        logger.info(f"  Financial data: {len(raw_fundamentals)} records")

        # Build fundamental panels
        from enhanced_factors import build_fundamental_panel_from_finnhub
        fundamental_panels = build_fundamental_panel_from_finnhub(
            raw_fundamentals, dates, tickers
        )

    # Step 5: Build sentiment + filter signals
    logger.info("\n[Step 5/6] Building sentiment & filter signals...")
    from enhanced_factors import (
        build_sentiment_panel_from_finnhub,
        build_macro_panel,
    )

    sentiment_panels = {}
    macro_panels = {}

    if finnhub_client is not None:
        sentiment_panels = build_sentiment_panel_from_finnhub(
            eps_df, ratings_df, targets_df, dates, tickers
        )

    # FRED macro data
    from fred_client import FredClient
    fred_key = dotenv.get_key(env_path, "FRED_API_KEY")
    if fred_key:
        try:
            fred_client = FredClient(fred_key)
            fred_df = fred_client.get_multiple_series(
                start_date=start_date, end_date=end_date
            )
            macro_panels = build_macro_panel(fred_df, dates)
        except Exception as e:
            logger.warning(f"FRED data fetch failed: {e}")

    # Step 5.5: Build v5a and v5b composites
    logger.info("\n[Step 5.5/6] Building v5 composites...")

    # v5a: v3b base + eps_surprise as additional weight
    # Use EPS surprise as a monthly-rebalanced additive signal
    v5a_factors = {"momentum": momentum, "quality": quality_panel}
    v5a_weights = {"momentum": 0.6, "quality": 0.3}

    if "eps_surprise" in sentiment_panels:
        # Scale EPS surprise to have similar magnitude as momentum ranks
        eps_panel = sentiment_panels["eps_surprise"]
        eps_nonzero = eps_panel[eps_panel != 0]
        if eps_nonzero.notna().sum().sum() > 100:
            v5a_factors["eps_surprise"] = eps_panel
            v5a_weights["eps_surprise"] = 0.1
            logger.info("  v5a: added eps_surprise (weight=0.1)")

    composite_v5a = combine_factors(v5a_factors, v5a_weights)

    # v5b: v3b + monthly fundamental filter + eps surprise
    # Strategy: Apply fundamental quality as a pre-filter before portfolio construction
    # Keep stocks with best margins and lowest leverage
    v5b_composite = composite_v3b.copy()  # Start with v3b base

    # Apply gross margin filter: keep top 70% (exclude worst margins)
    if "gross_margin" in fundamental_panels:
        v5b_composite = apply_monthly_filter(
            v5b_composite,
            fundamental_panels["gross_margin"],
            direction="positive",
            threshold_pct=0.7,
        )
        logger.info("  v5b: applied gross_margin filter (top 70%)")

    # Apply debt-to-equity filter: keep bottom 70% (exclude most leveraged)
    if "debt_to_equity" in fundamental_panels:
        v5b_composite = apply_monthly_filter(
            v5b_composite,
            fundamental_panels["debt_to_equity"],
            direction="negative",
            threshold_pct=0.7,
        )
        logger.info("  v5b: applied debt_to_equity filter (bottom 70%)")

    # Step 6: Backtest all
    logger.info("\n[Step 6/6] Running backtests...")
    bm = run_benchmark(all_returns, benchmark_ticker, 100000)

    results = {}

    # v0
    logger.info("\n--- v0: Raw Momentum ---")
    w0 = construct_portfolio(momentum, top_pct=0.2, bottom_pct=0.2)
    r0 = run_backtest(w0, all_returns, 10, 100000)
    results["v0"] = {**r0, "label": "v0: Raw Momentum"}

    # v3b
    logger.info("\n--- v3b: M(0.6)+Q(0.4) ---")
    w3b = construct_portfolio(composite_v3b, top_pct=0.2, bottom_pct=0.2)
    r3b = run_backtest(w3b, all_returns, 10, 100000)
    results["v3b"] = {**r3b, "label": "v3b: M0.6+Q0.4"}

    # v5a
    logger.info("\n--- v5a: v3b + EPS Surprise ---")
    w5a = construct_portfolio(composite_v5a, top_pct=0.2, bottom_pct=0.2)
    r5a = run_backtest(w5a, all_returns, 10, 100000)
    results["v5a"] = {**r5a, "label": "v5a: +EPS Sentiment"}

    # v5b
    logger.info("\n--- v5b: v3b + Fund. Filter ---")
    w5b = construct_portfolio(v5b_composite, top_pct=0.2, bottom_pct=0.2)
    r5b = run_backtest(w5b, all_returns, 10, 100000)
    results["v5b"] = {**r5b, "label": "v5b: +Fund Filter"}

    # Evaluate
    logger.info("\n" + "=" * 75)
    logger.info("  Performance Summary")
    logger.info("=" * 75)

    rf = config["backtest"]["risk_free_rate"]
    bm_returns = bm["returns"] if bm else None
    bm_cumulative = bm["cumulative"] if bm else None

    evaluations = {}
    for name, r in results.items():
        logger.info(f"\n--- {r['label']} ---")
        print_report(
            strategy_returns=r["portfolio_returns"],
            strategy_cumulative=r["cumulative_returns"],
            risk_free_rate=rf,
            benchmark_returns=bm_returns,
            benchmark_cumulative=bm_cumulative,
            strategy_name=r["label"],
            benchmark_name=benchmark_ticker,
        )
        evaluations[name] = evaluate(
            r["portfolio_returns"], r["cumulative_returns"], rf, bm_returns
        )

    # Summary table
    logger.info(f"\n{'=' * 80}")
    logger.info("  Summary")
    logger.info(f"{'=' * 80}")
    header = "{:<25} {:>10} {:>8} {:>10} {:>10} {:>8}".format(
        "Version", "Return", "Sharpe", "MaxDD", "Turnover", "TC"
    )
    logger.info(header)
    logger.info("-" * 80)

    for name in ["v0", "v3b", "v5a", "v5b"]:
        e = evaluations[name]
        r = results[name]
        turnover_avg = (
            float(r["turnover"].mean() * 100)
            if hasattr(r["turnover"], "mean")
            else float(r["turnover"])
        )
        line = "{:<25} {:>10.2%} {:>8.2f} {:>10.2%} {:>10.1f}% {:>8.2%}".format(
            results[name]["label"],
            float(e["年化收益率"]),
            float(e["夏普比率"]),
            float(e["最大回撤"]),
            turnover_avg,
            float(r["total_tc"]),
        )
        logger.info(line)

    # Compare improvements
    v0_s = evaluations["v0"]["夏普比率"]
    v3b_s = evaluations["v3b"]["夏普比率"]
    v5a_s = evaluations["v5a"]["夏普比率"]
    v5b_s = evaluations["v5b"]["夏普比率"]

    logger.info(f"\n{'=' * 80}")
    logger.info(f"  v0   Sharpe: {v0_s:.2f} (baseline)")
    logger.info(f"  v3b  Sharpe: {v3b_s:.2f} ({v3b_s - v0_s:+.2f} vs v0)")
    logger.info(f"  v5a  Sharpe: {v5a_s:.2f} ({v5a_s - v3b_s:+.2f} vs v3b)")
    logger.info(f"  v5b  Sharpe: {v5b_s:.2f} ({v5b_s - v3b_s:+.2f} vs v3b)")
    logger.info(f"{'=' * 80}")

    # Plot
    report_file = plot_v5_comparison(results, bm_cumulative)
    logger.info(f"\nDone! Report: {report_file}")

    return results, evaluations


if __name__ == "__main__":
    main()

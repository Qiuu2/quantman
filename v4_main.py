"""
v4_main.py - QuantMan v0 vs v3b vs v4 对比

v0: 原始 12-1 动量，等权多空
v3b: 动量(0.6) + 质量(0.4) 线性加权
v4:  LightGBM 多因子合成
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from data_loader import load_or_download, get_close_prices, load_config
from factors import momentum_12_1, combine_factors
from fundamental_factors import (
    fetch_fundamental_data,
    build_fundamental_panel,
)
from ml_factors import build_feature_matrix, train_and_predict
from portfolio import construct_portfolio
from backtest import compute_returns, run_backtest, run_benchmark
from metrics import print_report, evaluate

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v4_main")


def plot_v4_comparison(
    results: dict,
    benchmark_cumulative,
    save_dir: str = "reports",
):
    """生成 v0 / v3b / v4 三策略对比图表"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle(
        "QuantMan — v0 vs v3b vs v4: ML Factor Comparison",
        fontsize=16, fontweight="bold", y=0.98,
    )

    colors = {
        "v0": "#9E9E9E",
        "v3b": "#4CAF50",
        "v4": "#E91E63",
    }

    # Panel 1: 累计收益
    ax1 = axes[0]
    for name, r in results.items():
        style = "--" if name == "v0" else "-"
        lw = 1.5 if name == "v0" else 2.2
        ax1.plot(
            r["cumulative_returns"].index,
            r["cumulative_returns"].values,
            label=r["label"],
            color=colors.get(name, "#000"),
            linewidth=lw,
            linestyle=style,
        )
    if benchmark_cumulative is not None:
        ax1.plot(
            benchmark_cumulative.index,
            benchmark_cumulative.values,
            label="SPY (Buy & Hold)",
            color="#FF9800",
            linewidth=1.5,
            alpha=0.8,
        )
    ax1.set_ylabel("Cumulative Returns")
    ax1.set_title("Cumulative Performance")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.tick_params(axis="x", rotation=30)

    # Panel 2: 回撤
    ax2 = axes[1]
    for name, r in results.items():
        alpha = 0.3 if name == "v0" else 0.4
        ax2.fill_between(
            r["drawdown"].index, r["drawdown"].values, 0,
            color=colors.get(name, "#000"), alpha=alpha, label=r["label"],
        )
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown Comparison")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.tick_params(axis="x", rotation=30)

    # Panel 3: 滚动夏普
    ax3 = axes[2]
    for name, r in results.items():
        ret = r["portfolio_returns"]
        rs = ret.rolling(60).mean() / ret.rolling(60).std() * np.sqrt(252)
        style = "--" if name == "v0" else "-"
        lw = 0.8 if name == "v0" else 1.2
        ax3.plot(
            rs.index, rs.values,
            color=colors.get(name, "#000"),
            linewidth=lw, linestyle=style, label=r["label"],
        )
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
    report_file = save_path / "v0_vs_v3b_vs_v4_comparison.png"
    fig.savefig(report_file, dpi=150, bbox_inches="tight")
    logger.info(f"对比报告已保存: {report_file}")
    plt.close(fig)
    return report_file


def main():
    """v0 vs v3b vs v4 对比主流程"""
    logger.info("=" * 60)
    logger.info("  QuantMan — v0 vs v3b vs v4")
    logger.info("  v0:  Raw 12-1 Momentum")
    logger.info("  v3b: Momentum(0.6) + Quality(0.4)")
    logger.info("  v4:  LightGBM Multi-Factor")
    logger.info("=" * 60)

    config = load_config("config.json")
    benchmark_ticker = config["data"]["benchmark_ticker"]
    logger.info(f"回测区间: {config['data']['start_date']} ~ {config['data']['end_date']}")

    # Step 1: 获取数据
    logger.info("\n[Step 1/6] 获取价格数据...")
    fields = load_or_download("config.json")
    close = get_close_prices(fields)
    stock_close = close.drop(columns=[benchmark_ticker], errors="ignore")

    # Step 2: 计算动量因子 (v0)
    logger.info("\n[Step 2/6] 计算动量因子 (v0)...")
    lookback = config["strategy"]["momentum_lookback"]
    skip = config["strategy"]["momentum_skip"]
    momentum = momentum_12_1(stock_close, lookback=lookback, skip=skip)

    # Step 3: 获取基本面数据 (v3b)
    logger.info("\n[Step 3/6] 获取基本面数据 (v3b)...")
    tickers = stock_close.columns.tolist()
    fund = fetch_fundamental_data(tickers)
    quality_panel = build_fundamental_panel(fund, stock_close.index, "quality")

    # v3b 合成
    logger.info("\n[Step 3.5/6] v3b 多因子合成...")
    composite_v3b = combine_factors(
        {"momentum": momentum, "quality": quality_panel},
        {"momentum": 0.6, "quality": 0.4},
    )

    # Step 4: LightGBM 特征工程 + 训练 (v4)
    logger.info("\n[Step 4/6] LightGBM 特征工程 + Rolling 训练 (v4)...")
    features_3d, feature_names = build_feature_matrix(stock_close, fund)
    ml_predictions = train_and_predict(
        close=stock_close,
        features_3d=features_3d,
        feature_names=feature_names,
        train_window=504,  # 2 年
        forward_days=21,
        rebalance_freq=21,  # 月度调仓
        sample_ratio=0.15,  # 每个截面采样 15% 股票
        smooth_alpha=0.6,   # 预测平滑: 60% 新 + 40% 旧
    )

    # Step 5: 回测
    logger.info("\n[Step 5/6] 运行回测...")
    all_returns = compute_returns(close)
    bm = run_benchmark(all_returns, benchmark_ticker, 100000)

    results = {}

    # v0: 原始动量
    logger.info("\n--- v0: 原始动量 ---")
    w0 = construct_portfolio(momentum, top_pct=0.2, bottom_pct=0.2)
    r0 = run_backtest(w0, all_returns, 10, 100000)
    results["v0"] = {**r0, "label": "v0: Raw Momentum"}

    # v3b: 动量 + 质量
    logger.info("\n--- v3b: M(0.6)+Q(0.4) ---")
    w3b = construct_portfolio(composite_v3b, top_pct=0.2, bottom_pct=0.2)
    r3b = run_backtest(w3b, all_returns, 10, 100000)
    results["v3b"] = {**r3b, "label": "v3b: M0.6+Q0.4"}

    # v4: LightGBM
    logger.info("\n--- v4: LightGBM ---")
    w4 = construct_portfolio(ml_predictions, top_pct=0.2, bottom_pct=0.2)
    r4 = run_backtest(w4, all_returns, 10, 100000)
    results["v4"] = {**r4, "label": "v4: LightGBM"}

    # Step 6: 绩效对比
    logger.info("\n[Step 6/6] 绩效评估...")
    rf = config["backtest"]["risk_free_rate"]
    bm_returns = bm["returns"] if bm else None
    bm_cumulative = bm["cumulative"] if bm else None

    evaluations = {}
    for name, r in results.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  {r['label']}")
        logger.info(f"{'=' * 60}")
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

    # 汇总
    logger.info(f"\n{'=' * 75}")
    logger.info("  Summary: v0 vs v3b vs v4")
    logger.info(f"{'=' * 75}")
    header = "{:<20} {:>10} {:>8} {:>10} {:>10} {:>8}".format(
        "Version", "Return", "Sharpe", "MaxDD", "Turnover", "TC"
    )
    logger.info(header)
    logger.info("-" * 75)

    for name in ["v0", "v3b", "v4"]:
        e = evaluations[name]
        r = results[name]
        turnover_avg = (
            float(r["turnover"].mean() * 100)
            if hasattr(r["turnover"], "mean")
            else float(r["turnover"])
        )
        line = "{:<20} {:>10.2%} {:>8.2f} {:>10.2%} {:>10.1f}% {:>8.2%}".format(
            results[name]["label"],
            float(e["年化收益率"]),
            float(e["夏普比率"]),
            float(e["最大回撤"]),
            turnover_avg,
            float(r["total_tc"]),
        )
        logger.info(line)

    # 对比提升
    v0_sharpe = evaluations["v0"]["夏普比率"]
    v3b_sharpe = evaluations["v3b"]["夏普比率"]
    v4_sharpe = evaluations["v4"]["夏普比率"]

    logger.info(f"\n{'=' * 75}")
    logger.info(f"  v0   夏普: {v0_sharpe:.2f} (基线)")
    logger.info(f"  v3b  夏普: {v3b_sharpe:.2f} ({v3b_sharpe - v0_sharpe:+.2f} vs v0)")
    logger.info(f"  v4   夏普: {v4_sharpe:.2f} ({v4_sharpe - v0_sharpe:+.2f} vs v0)")
    logger.info(f"{'=' * 75}")

    # 出图
    report_file = plot_v4_comparison(results, bm_cumulative)
    logger.info(f"\nDone! Report: {report_file}")

    return results, evaluations


if __name__ == "__main__":
    main()

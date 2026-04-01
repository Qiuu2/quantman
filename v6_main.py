"""
v6/v6_main.py - v6 主入口

编排完整流程：
数据加载 → 特征工程 → LambdaRank 训练 → 回测 → 诊断 → v3b 对比

用法：
    cd quantman
    python -m v6.v6_main
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# 确保能 import 上级目录的模块
import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from portfolio import construct_portfolio
from backtest import run_backtest, compute_returns, run_benchmark
from metrics import print_report, evaluate

from v6.config import *
from v6.data_loader import load_all_data
from v6.features import build_all_features, build_labels_weekly
from v6.training import purged_walk_forward_train
from v6.analysis import run_full_diagnostics

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v6_main")


def plot_v6_vs_v3b(
    results: dict,
    benchmark_cumulative: pd.Series,
    save_dir: Path = REPORTS_DIR,
) -> Path:
    """绘制 v6 vs v3b 对比图表"""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle(
        "QuantMan v6 vs v3b: ML Factor Synthesis Comparison",
        fontsize=16, fontweight="bold", y=0.98,
    )

    colors = {"v3b": "#4CAF50", "v6": "#E91E63"}

    # Panel 1: 累计收益
    ax1 = axes[0]
    for name, r in results.items():
        ax1.plot(
            r["cumulative_returns"].index,
            r["cumulative_returns"].values,
            label=r["label"],
            color=colors.get(name, "#000"),
            linewidth=2.2,
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
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax1.tick_params(axis="x", rotation=30)

    # Panel 2: 回撤
    ax2 = axes[1]
    for name, r in results.items():
        ax2.fill_between(
            r["drawdown"].index, r["drawdown"].values, 0,
            color=colors.get(name, "#000"), alpha=0.4, label=r["label"],
        )
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown Comparison")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax2.tick_params(axis="x", rotation=30)

    # Panel 3: 滚动夏普
    ax3 = axes[2]
    for name, r in results.items():
        ret = r["portfolio_returns"]
        # 用 252 日滚动（虽然我们的权重是周频，但回测引擎用日频）
        rs = ret.rolling(252).mean() / ret.rolling(252).std() * np.sqrt(252)
        ax3.plot(
            rs.index, rs.values,
            color=colors.get(name, "#000"),
            linewidth=1.2, label=r["label"],
        )
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.axhline(y=1, color="#4CAF50", linestyle=":", alpha=0.3)
    ax3.set_ylabel("Sharpe (252D)")
    ax3.set_title("Rolling 252-Day Sharpe Ratio")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax3.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    report_file = save_dir / "v6_vs_v3b_comparison.png"
    fig.savefig(report_file, dpi=150, bbox_inches="tight")
    logger.info(f"对比报告已保存: {report_file}")
    plt.close(fig)

    return report_file


def main():
    """v6 主流程"""
    logger.info("=" * 60)
    logger.info("  QuantMan v6 - ML Factor Synthesis")
    logger.info("  SimFin Fundamentals + LambdaRank + Russell 3000")
    logger.info("=" * 60)

    # ── Step 1: 数据加载 ──
    logger.info("\n[Step 1/6] 数据加载...")
    data = load_all_data()
    close_weekly = data["close_weekly"]
    close_daily = data["close_daily"]
    simfin = data["simfin"]
    universe = data["universe"]

    # ── Step 2: 特征工程 ──
    logger.info("\n[Step 2/6] 特征工程...")
    features_3d, feature_names = build_all_features(
        close_weekly=close_weekly,
        simfin_data=simfin,
        universe=universe,
        close_daily_for_pb=close_daily,
    )
    labels_2d = build_labels_weekly(close_weekly, forward_weeks=FORWARD_WEEKS)

    # ── Step 3: LambdaRank 训练 ──
    logger.info("\n[Step 3/6] LightGBM LambdaRank 训练...")
    predictions, diagnostics = purged_walk_forward_train(
        features_3d=features_3d,
        labels_2d=labels_2d,
        dates=close_weekly.index,
        tickers=close_weekly.columns.tolist(),
        feature_names=feature_names,
    )

    # ── Step 4: 回测准备 ──
    logger.info("\n[Step 4/6] 回测准备...")

    # 将周频预测映射到日频（前向填充）
    # 需要将 predictions (周频 index) 扩展到日频
    # 回测引擎用日频权重 × 日频收益

    # 对齐：取预测和日频价格的交集
    common_tickers = list(set(predictions.columns) & set(close_daily.columns) - {BENCHMARK_TICKER})
    predictions_daily = predictions[common_tickers].reindex(close_daily.index).ffill()

    # 构建权重矩阵（用 construct_portfolio 需要日频因子 DataFrame）
    weights_v6 = construct_portfolio(
        predictions_daily, top_pct=TOP_PCT, bottom_pct=BOTTOM_PCT
    )

    # 日频收益
    all_returns = compute_returns(close_daily)

    # Benchmark
    bm = run_benchmark(all_returns, BENCHMARK_TICKER, INITIAL_CAPITAL)

    # ── Step 5: 运行回测 ──
    logger.info("\n[Step 5/6] 运行回测...")

    # v6 回测
    r_v6 = run_backtest(
        weights=weights_v6,
        returns=all_returns,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_capital=INITIAL_CAPITAL,
    )

    # v3b 对比（使用 v6 的股票池和动量因子）
    # 简化：用 momentum_12_1 + quality 线性组合
    # 需要 reindex predictions 的日频版本
    # 这里简化为直接用 ML 预测中的动量因子部分
    # 实际应该用 v3b 的 M0.6+Q0.4 权重重新构建

    # 从特征矩阵中提取动量和质量因子做 v3b 对照
    # 由于 features_3d 已经标准化，直接用 momentum_12_1 和 quality/roe 列
    mom_idx = feature_names.index("momentum_12_1") if "momentum_12_1" in feature_names else 0
    quality_idx = feature_names.index("roe") if "roe" in feature_names else -1

    if quality_idx >= 0:
        v3b_score = 0.6 * features_3d[:, :, mom_idx] + 0.4 * features_3d[:, :, quality_idx]
        v3b_df = pd.DataFrame(
            v3b_score, index=close_weekly.index, columns=close_weekly.columns
        )
        v3b_daily = v3b_df[common_tickers].reindex(close_daily.index).ffill()
        weights_v3b = construct_portfolio(v3b_daily, top_pct=TOP_PCT, bottom_pct=BOTTOM_PCT)
        r_v3b = run_backtest(
            weights=weights_v3b,
            returns=all_returns,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            initial_capital=INITIAL_CAPITAL,
        )
    else:
        logger.warning("无法计算 v3b 对照（缺少质量因子）")
        r_v3b = None

    # ── Step 6: 绩效评估 + 诊断 ──
    logger.info("\n[Step 6/6] 绩效评估 + 诊断...")

    rf = RISK_FREE_RATE
    bm_returns = bm["returns"] if bm else None
    bm_cumulative = bm["cumulative"] if bm else None

    results = {}
    evaluations = {}

    # v6 评估
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  v6: LightGBM LambdaRank")
    logger.info(f"{'=' * 60}")
    print_report(
        strategy_returns=r_v6["portfolio_returns"],
        strategy_cumulative=r_v6["cumulative_returns"],
        risk_free_rate=rf,
        benchmark_returns=bm_returns,
        benchmark_cumulative=bm_cumulative,
        strategy_name="v6: LightGBM LambdaRank",
        benchmark_name=BENCHMARK_TICKER,
    )
    ev_v6 = evaluate(r_v6["portfolio_returns"], r_v6["cumulative_returns"], rf, bm_returns)
    results["v6"] = {**r_v6, "label": "v6: LightGBM LambdaRank"}
    evaluations["v6"] = ev_v6

    # v3b 评估
    if r_v3b:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  v3b: M(0.6) + Q(0.4)")
        logger.info(f"{'=' * 60}")
        print_report(
            strategy_returns=r_v3b["portfolio_returns"],
            strategy_cumulative=r_v3b["cumulative_returns"],
            risk_free_rate=rf,
            benchmark_returns=bm_returns,
            benchmark_cumulative=bm_cumulative,
            strategy_name="v3b: M0.6+Q0.4",
            benchmark_name=BENCHMARK_TICKER,
        )
        ev_v3b = evaluate(r_v3b["portfolio_returns"], r_v3b["cumulative_returns"], rf, bm_returns)
        results["v3b"] = {**r_v3b, "label": "v3b: M0.6+Q0.4"}
        evaluations["v3b"] = ev_v3b

    # 汇总对比
    logger.info(f"\n{'=' * 75}")
    logger.info(f"  Summary: v6 vs v3b")
    logger.info(f"{'=' * 75}")
    header = "{:<20} {:>10} {:>8} {:>10} {:>10} {:>8}".format(
        "Version", "Return", "Sharpe", "MaxDD", "Turnover", "TC"
    )
    logger.info(header)
    logger.info("-" * 75)

    for name in ["v3b", "v6"]:
        if name not in evaluations:
            continue
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

    # 提升分析
    if "v3b" in evaluations and "v6" in evaluations:
        v3b_sharpe = evaluations["v3b"]["夏普比率"]
        v6_sharpe = evaluations["v6"]["夏普比率"]
        improvement = v6_sharpe - v3b_sharpe

        logger.info(f"\n{'=' * 75}")
        logger.info(f"  v3b 夏普: {v3b_sharpe:.2f} (基线)")
        logger.info(f"  v6  夏普: {v6_sharpe:.2f} ({improvement:+.2f} vs v3b)")

        if improvement > 0.1:
            logger.info(f"  ✅ v6 显著优于 v3b (夏普提升 {improvement:+.2f})")
        elif improvement > 0:
            logger.info(f"  ⚡ v6 略优于 v3b (夏普提升 {improvement:+.2f})")
        else:
            logger.info(f"  ❌ v6 未跑赢 v3b (夏普降低 {improvement:+.2f})")
        logger.info(f"{'=' * 75}")

    # 对比图表
    plot_v6_vs_v3b(results, bm_cumulative)

    # 诊断分析
    run_full_diagnostics(
        features_3d=features_3d,
        labels_2d=labels_2d,
        feature_names=feature_names,
        dates=close_weekly.index,
        diagnostics=diagnostics,
        weights=weights_v6,
    )

    logger.info(f"\nDone! 报告保存在: {REPORTS_DIR}")

    return results, evaluations


if __name__ == "__main__":
    main()

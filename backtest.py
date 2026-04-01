"""
backtest.py - 向量化回测引擎

纯 pandas 向量化计算，不依赖第三方回测框架。
支持：多空组合、交易成本、换手率统计。
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    """
    计算日收益率矩阵。

    Args:
        close: 收盘价 DataFrame, index=Date, columns=ticker

    Returns:
        DataFrame: index=Date, columns=ticker, values=daily returns
    """
    return close.pct_change()


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost_bps: float = 10,
    initial_capital: float = 100000,
) -> dict:
    """
    运行向量化回测。

    核心公式:
        daily_pnl = sum(weights_t-1 * returns_t) - transaction_cost * turnover

    Args:
        weights: 权重矩阵, index=Date, columns=ticker
                 正值=多头, 负值=空头, 0=不持有
        returns: 日收益率矩阵, index=Date, columns=ticker
        transaction_cost_bps: 交易成本（基点，10bps = 0.1%）
        initial_capital: 初始资金

    Returns:
        dict: 回测结果，包含各种时间序列和统计指标
    """
    logger.info("开始回测...")

    # 对齐日期
    common_idx = weights.index.intersection(returns.index)
    weights = weights.loc[common_idx]
    returns = returns.loc[common_idx]

    # 对齐列（股票代码）
    common_cols = weights.columns.intersection(returns.columns)
    weights = weights[common_cols]
    returns = returns[common_cols]

    # 计算换手率
    weight_changes = weights.diff()
    turnover = weight_changes.abs().sum(axis=1) / 2  # 双边换手率

    # 计算交易成本（基于换手率）
    tc_rate = transaction_cost_bps / 10000
    daily_tc = turnover * tc_rate

    # 计算组合日收益（用 T-1 日的权重 × T 日的收益）
    # weights 先 shift 一天，因为今天的持仓是昨天决定的
    lagged_weights = weights.shift(1)

    portfolio_returns = (lagged_weights * returns).sum(axis=1) - daily_tc
    portfolio_returns.iloc[0] = 0.0  # 第一天没有收益

    # 累计净值曲线
    cumulative = (1 + portfolio_returns).cumprod()
    nav = cumulative * initial_capital

    # 回撤
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max

    # 统计
    total_turnover = turnover.sum()
    total_tc = daily_tc.sum()

    logger.info(f"回测完成: {len(portfolio_returns)} 个交易日")
    logger.info(f"年化收益率: {portfolio_returns.mean() * 252 * 100:.2f}%")
    logger.info(f"总换手率: {total_turnover:.2f}, 总交易成本: {total_tc*100:.2f}%")

    return {
        "portfolio_returns": portfolio_returns,
        "cumulative_returns": cumulative,
        "nav": nav,
        "drawdown": drawdown,
        "turnover": turnover,
        "daily_tc": daily_tc,
        "weights": weights,
        "lagged_weights": lagged_weights,
        "total_turnover": total_turnover,
        "total_tc": total_tc,
        "initial_capital": initial_capital,
    }


def run_benchmark(
    returns: pd.DataFrame,
    benchmark: str = "SPY",
    initial_capital: float = 100000,
) -> dict:
    """
    计算 benchmark 的收益曲线。

    Args:
        returns: 日收益率 DataFrame（columns=ticker，需包含 benchmark）
        benchmark: 基准代码
        initial_capital: 初始资金

    Returns:
        dict: benchmark 累计收益等
    """
    if benchmark not in returns.columns:
        logger.warning(f"Benchmark {benchmark} 不在数据中")
        return None

    bm_returns = returns[benchmark].copy()
    bm_returns.iloc[0] = 0.0
    bm_cumulative = (1 + bm_returns).cumprod()
    bm_nav = bm_cumulative * initial_capital

    return {
        "returns": bm_returns,
        "cumulative": bm_cumulative,
        "nav": bm_nav,
    }

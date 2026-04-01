"""
portfolio.py - 组合构建模块

基于因子值构建多空组合：
- 做多动量排名前 top_pct% 的股票
- 做空动量排名后 bottom_pct% 的股票
- 等权分配
"""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def get_rebalance_dates(
    dates: pd.DatetimeIndex,
    freq: str = "ME",  # 月末
) -> pd.DatetimeIndex:
    """
    获取调仓日期列表。

    使用 pandas 的日期偏移来获取每月的调仓日。
    使用月末而非月初，因为我们在月末看到完整数据后决定下月持仓。

    Args:
        dates: 交易日的 DatetimeIndex
        freq: 调仓频率 ('ME'=月末, 'QE'=季末)

    Returns:
        DatetimeIndex: 调仓日期
    """
    # 找到每个月最后一个交易日作为调仓日
    # 实际持仓从下一个交易日开始
    monthly_groups = dates.to_series().groupby(pd.Grouper(freq=freq))
    rebalance_dates = monthly_groups.last().dropna()
    rebalance_dates = rebalance_dates[rebalance_dates.index.isin(dates)]
    logger.info(f"调仓日数量: {len(rebalance_dates)}")
    return rebalance_dates


def construct_portfolio(
    factor: pd.DataFrame,
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    构建多空组合的权重矩阵。

    Args:
        factor: 动量因子 DataFrame, index=Date, columns=ticker
        top_pct: 做多比例（前20%）
        bottom_pct: 做空比例（后20%）
        min_stocks: 最少持仓数量（不足则不调仓）

    Returns:
        DataFrame: index=Date, columns=ticker, values=权重
                   正值=多头，负值=空头，0=不持有
    """
    logger.info(f"构建多空组合 (Top {top_pct*100:.0f}% / Bottom {bottom_pct*100:.0f}%)...")

    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)

    for date, row in factor.iterrows():
        valid = row.dropna()
        n_valid = len(valid)
        n_top = max(int(n_valid * top_pct), min_stocks)
        n_bottom = max(int(n_valid * bottom_pct), min_stocks)

        if n_valid < min_stocks * 2:
            continue  # 股票太少，不调仓

        sorted_values = valid.sort_values(ascending=False)

        # 做多头
        long_stocks = sorted_values.index[:n_top]
        long_weight = 1.0 / n_top
        weights.loc[date, long_stocks] = long_weight

        # 做空头
        short_stocks = sorted_values.index[-n_bottom:]
        short_weight = -1.0 / n_bottom
        weights.loc[date, short_stocks] = short_weight

    # 向前填充权重（调仓日之间保持不变）
    weights = weights.ffill()

    # 第一天（没有因子值之前）权重为0
    weights.iloc[0] = 0.0

    long_count = (weights > 0).sum(axis=1).mean()
    short_count = (weights < 0).sum(axis=1).mean()
    logger.info(f"平均多头: {long_count:.0f} 只, 平均空头: {short_count:.0f} 只")

    return weights


def construct_long_only_portfolio(
    factor: pd.DataFrame,
    top_pct: float = 0.20,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    构建纯多头组合（不做空）。

    适用于不支持做空的场景或初始测试。

    Args:
        factor: 动量因子 DataFrame
        top_pct: 持仓比例
        min_stocks: 最少持仓数量

    Returns:
        DataFrame: 权重矩阵（0~1/N）
    """
    logger.info(f"构建纯多头组合 (Top {top_pct*100:.0f}%)...")

    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)

    for date, row in factor.iterrows():
        valid = row.dropna()
        n_valid = len(valid)
        n_top = max(int(n_valid * top_pct), min_stocks)

        if n_valid < min_stocks:
            continue

        sorted_values = valid.sort_values(ascending=False)
        long_stocks = sorted_values.index[:n_top]
        long_weight = 1.0 / n_top
        weights.loc[date, long_stocks] = long_weight

    weights = weights.ffill()
    weights.iloc[0] = 0.0

    avg_holdings = (weights > 0).sum(axis=1).mean()
    logger.info(f"平均持仓: {avg_holdings:.0f} 只")

    return weights

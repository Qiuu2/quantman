"""
liquidity_filter.py - 流动性过滤与加权模块

v2 改进：
  - 动态流动性过滤（时变阈值，排除每个截面上流动性最低的股票）
  - 流动性加权（替代等权，让高流动性股票获得更大权重）
  - Amihud 非流动性指标（衡量价格冲击成本）
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# 流动性指标计算
# ============================================================================


def avg_daily_turnover(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    计算日均成交额（滚动均值）。
    成交额 = 价格 × 成交量。
    """
    turnover = close * volume
    return turnover.rolling(window=window).mean()


def amihud_illiquidity(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    计算 Amihud 非流动性指标。

    公式：ILLIQ = mean(|daily_return| / daily_dollar_volume)
    含义：每1美元交易能推动多少百分比的价格变化。
    值越大 = 流动性越差。
    """
    daily_return = close.pct_change().abs()
    daily_dollar_volume = (close * volume).replace(0, np.nan)
    illiq_daily = daily_return / daily_dollar_volume
    return illiq_daily.rolling(window=window).mean()


# ============================================================================
# 流动性过滤
# ============================================================================


def filter_by_turnover(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    min_turnover: float = 5_000_000,
    window: int = 63,
) -> pd.DataFrame:
    """
    硬阈值过滤：排除日均成交额低于阈值的股票。
    低流动性股票的值设为 NaN。
    """
    avg_turn = avg_daily_turnover(close, volume, window)
    mask = avg_turn >= min_turnover
    filtered = close.where(mask, np.nan)
    removed = (~mask).sum(axis=1)
    logger.info(
        f"硬阈值过滤 (>={min_turnover/1e6:.0f}M/天): "
        f"平均每天排除 {removed.mean():.1f} 只"
    )
    return filtered


def filter_bottom_pct(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 63,
    bottom_pct: float = 0.10,
) -> pd.DataFrame:
    """
    百分位过滤：排除每个截面上流动性最低的 N% 股票。
    动态阈值，随市场环境变化。
    """
    avg_turn = avg_daily_turnover(close, volume, window)
    threshold = avg_turn.quantile(bottom_pct, axis=1)
    mask = avg_turn.ge(threshold, axis=0)
    filtered = close.where(mask, np.nan)
    removed = (~mask).sum(axis=1)
    logger.info(
        f"百分位过滤 (底部{bottom_pct:.0%}): "
        f"平均每天排除 {removed.mean():.1f} 只"
    )
    return filtered


# ============================================================================
# 流动性加权
# ============================================================================


def liquidity_weights(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    基于流动性的权重矩阵。

    用成交额平方根作为权重（温和的流动性偏好），
    使得高流动性股票在组合中占更大权重。
    """
    avg_turn = avg_daily_turnover(close, volume, window)
    # 平方根平滑极端差异（NVDA 344亿 vs NWS 2700万 → 差1300倍，sqrt后差36倍）
    weights = np.sqrt(avg_turn)
    return weights

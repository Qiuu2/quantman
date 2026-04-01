"""
factors.py - 因子计算模块

v1 改进：
  - 波动率调整动量（Vol-Adjusted Momentum）
  - 多时间窗口合成（3m/6m/12m）
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# 基础动量因子
# ============================================================================


def momentum_12_1(
    close: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    计算 12-1 动量因子（原始版本，作为基准）。

    动量 = (今日收盘价 / lookback天前收盘价) / (今日收盘价 / skip天前收盘价)
          = skip天前收盘价 / lookback天前收盘价

    跳过最近1个月是为了避免短期反转效应。

    Args:
        close: 收盘价 DataFrame, index=Date, columns=ticker
        lookback: 回看天数（默认252≈12个月）
        skip: 跳过天数（默认21≈1个月）

    Returns:
        DataFrame: index=Date, columns=ticker, values=momentum score
    """
    logger.info(f"计算 12-1 动量因子 (lookback={lookback}, skip={skip})...")

    # 价格偏移
    price_skip_ago = close.shift(skip)  # 1个月前的价格
    price_lookback_ago = close.shift(lookback)  # 12个月前的价格

    momentum = price_skip_ago / price_lookback_ago - 1

    # 统计
    valid_count = momentum.notna().sum(axis=1)
    logger.info(f"因子计算完成，平均每日 {valid_count.mean():.0f} 只股票有有效值")

    return momentum


# ============================================================================
# v1 改进：波动率调整动量
# ============================================================================


def volatility_adjusted_momentum(
    close: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    vol_window: int = 63,
) -> pd.DataFrame:
    """
    波动率调整动量（Vol-Adjusted Momentum）。

    原理：同样涨了40%，波动率低的股票信号质量更高。
    公式：momentum = return_12_1 / rolling_volatility

    Args:
        close: 收盘价 DataFrame
        lookback: 动量回看天数（默认252）
        skip: 跳过天数（默认21）
        vol_window: 波动率计算窗口（默认63≈3个月）

    Returns:
        DataFrame: 波动率调整后的动量因子
    """
    logger.info(f"计算波动率调整动量 (vol_window={vol_window})...")

    # 计算原始 12-1 动量
    price_skip_ago = close.shift(skip)
    price_lookback_ago = close.shift(lookback)
    raw_momentum = price_skip_ago / price_lookback_ago - 1

    # 计算滚动波动率（年化）
    daily_returns = close.pct_change()
    rolling_vol = daily_returns.rolling(window=vol_window).std() * np.sqrt(252)

    # 波动率调整
    vol_adjusted_momentum = raw_momentum / rolling_vol

    # 统计
    valid_count = vol_adjusted_momentum.notna().sum(axis=1)
    logger.info(f"波动率调整完成，平均每日 {valid_count.mean():.0f} 只股票有有效值")

    return vol_adjusted_momentum


# ============================================================================
# v1 改进：多时间窗口动量合成
# ============================================================================


def multi_window_momentum(
    close: pd.DataFrame,
    skip: int = 21,
    windows: dict = None,
) -> pd.DataFrame:
    """
    多时间窗口动量合成。

    同时计算 3个月/6个月/12个月 动量，加权合成。
    不同窗口捕捉不同强度的趋势惯性。

    Args:
        close: 收盘价 DataFrame
        skip: 跳过天数（默认21）
        windows: 窗口配置，格式为 {天数: 权重}，默认 {63: 0.4, 126: 0.3, 252: 0.3}

    Returns:
        DataFrame: 合成后的动量因子
    """
    if windows is None:
        windows = {63: 0.4, 126: 0.3, 252: 0.3}  # 3m:40%, 6m:30%, 12m:30%

    logger.info(f"计算多窗口动量合成，窗口: {windows}...")

    composite = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    composite[:] = np.nan

    for window, weight in windows.items():
        # 计算该窗口的动量
        price_skip_ago = close.shift(skip)
        price_window_ago = close.shift(window)
        mom_window = price_skip_ago / price_window_ago - 1

        # 加权累加
        composite = composite.add(mom_window * weight, fill_value=0)

    # 统计
    valid_count = composite.notna().sum(axis=1)
    logger.info(f"多窗口合成完成，平均每日 {valid_count.mean():.0f} 只股票有有效值")

    return composite


# ============================================================================
# v1 最终版本：波动率调整 + 多窗口合成
# ============================================================================


def momentum_v1(
    close: pd.DataFrame,
    skip: int = 21,
    windows: dict = None,
    vol_window: int = 63,
    smooth_span: int = 42,
) -> pd.DataFrame:
    """
    v1 改进版动量因子：波动率调整 + 多窗口合成 + 信号平滑。

    流程：
    1. 计算多窗口合成动量
    2. 用波动率调整
    3. 指数加权平滑（减少换手率）

    Args:
        close: 收盘价 DataFrame
        skip: 跳过天数（默认21）
        windows: 窗口配置（默认 {126: 0.4, 252: 0.6}，更保守）
        vol_window: 波动率计算窗口（默认63）
        smooth_span: 平滑窗口（默认42≈2个月，越大越平滑）

    Returns:
        DataFrame: v1 动量因子
    """
    if windows is None:
        # 更保守的权重：只用6m和12m，降低短期噪音
        windows = {126: 0.4, 252: 0.6}

    logger.info("=" * 50)
    logger.info("计算 v1 改进版动量因子")
    logger.info("=" * 50)

    # Step 1: 多窗口合成动量
    composite = multi_window_momentum(close, skip=skip, windows=windows)

    # Step 2: 计算波动率
    daily_returns = close.pct_change()
    rolling_vol = daily_returns.rolling(window=vol_window).std() * np.sqrt(252)

    # Step 3: 波动率调整
    vol_adjusted = composite / rolling_vol

    # Step 4: 信号平滑（关键改进：减少换手率）
    smoothed = vol_adjusted.ewm(span=smooth_span, min_periods=1).mean()

    # 统计
    valid_count = smoothed.notna().sum(axis=1)
    logger.info(f"v1 因子计算完成，平均每日 {valid_count.mean():.0f} 只股票有有效值")
    logger.info(f"信号平滑: ewm(span={smooth_span})")
    logger.info("=" * 50)

    return smoothed


# ============================================================================
# 辅助函数：横截面标准化
# ============================================================================


def cross_sectional_rank(factor: pd.DataFrame) -> pd.DataFrame:
    """
    横截面标准化：将因子值转化为截面排名（0~1）。

    每个截面（每一天）内独立排名，减少极端值影响。

    Args:
        factor: 原始因子值 DataFrame

    Returns:
        DataFrame: 排名标准化后的因子（0~1）
    """
    return factor.rank(axis=1, pct=True)


def cross_sectional_zscore(factor: pd.DataFrame) -> pd.DataFrame:
    """
    横截面 Z-Score 标准化。

    每个截面内独立计算均值和标准差。

    Args:
        factor: 原始因子值 DataFrame

    Returns:
        DataFrame: Z-Score 标准化后的因子
    """
    mean = factor.mean(axis=1)
    std = factor.std(axis=1)
    return factor.sub(mean, axis=0).div(std, axis=0)


def combine_factors(
    factors_dict: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    多因子加权合成。

    将多个因子按权重线性组合，生成综合得分。
    所有因子先用截面排名标准化到 [0, 1]，确保量纲一致。

    Args:
        factors_dict: {"momentum": DataFrame, "value": DataFrame, "quality": DataFrame}
        weights: {"momentum": 0.4, "value": 0.3, "quality": 0.3}

    Returns:
        DataFrame: 综合因子得分
    """
    logger.info("=" * 50)
    logger.info(f"多因子合成，权重: {weights}")
    logger.info("=" * 50)

    # 确保权重和为1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # 对每个因子做截面排名
    ranked_factors = {}
    for name, factor in factors_dict.items():
        ranked = cross_sectional_rank(factor)
        ranked_factors[name] = ranked
        valid = ranked.notna().sum(axis=1).mean()
        logger.info(f"  {name}: 排名完成, 平均 {valid:.0f} 只有效")

    # 对齐索引和列（取交集）
    common_index = None
    common_columns = None
    for name, ranked in ranked_factors.items():
        if common_index is None:
            common_index = ranked.index
            common_columns = ranked.columns
        else:
            common_index = common_index.intersection(ranked.index)
            common_columns = common_columns.intersection(ranked.columns)

    if common_index is None or len(common_index) == 0:
        raise ValueError("因子之间没有共同的时间索引")

    logger.info(f"共同区间: {len(common_index)} 个交易日, {len(common_columns)} 只股票")

    # 加权合成
    composite = pd.DataFrame(0.0, index=common_index, columns=common_columns)

    for name, weight in weights.items():
        if name in ranked_factors:
            aligned = ranked_factors[name].reindex(index=common_index, columns=common_columns)
            composite += aligned * weight
            logger.info(f"  {name}: 权重 {weight:.2f}")

    valid_count = composite.notna().sum(axis=1).mean()
    logger.info(f"多因子合成完成: 平均 {valid_count:.0f} 只股票有有效值")
    logger.info("=" * 50)

    return composite


def winsorize(factor: pd.DataFrame, limits: float = 0.01) -> pd.DataFrame:
    """
    去极值处理（MAD法）。

    将超出 ±3个MAD的值截断到边界。

    Args:
        factor: 原始因子值 DataFrame
        limits: 尾部比例

    Returns:
        DataFrame: 去极值后的因子
    """
    median = factor.median(axis=1)
    mad = (factor.sub(median, axis=0)).abs().median(axis=1)

    lower = median - 3 * 1.4826 * mad
    upper = median + 3 * 1.4826 * mad

    return factor.clip(lower, upper, axis=0)

"""
enhanced_factors.py - v5 增强因子模块

基于新数据源（Finnhub + FRED）构建历史时间序列因子，替代 yfinance 快照。

Finnhub 数据格式（来自 stock/metric 端点）：
  DataFrame columns=[ticker, date, field, value]
  field 可选值: roeTTM, roaTTM, grossMargin, operatingMargin, netMargin,
               peTTM, pb, psTTM, totalDebtToEquity, currentRatio, bookValue, eps

因子列表：
  基本面（时序）：
    - roe_trend: ROE 变化趋势（季度环比）
    - margin_trend: 毛利率变化趋势
    - roa: 总资产收益率
    - debt_to_equity: 资产负债率
    - eps_growth: EPS 季度同比增长

  分析师情绪：
    - eps_surprise: 最近一次 EPS 超预期百分比
    - analyst_consensus: 分析师共识评分（买入=5, 卖出=1）
    - target_upside: 目标价上涨空间

  宏观环境：
    - term_spread: 收益率曲线斜率（10Y-2Y）
    - macro_regime: 宏观环境标签（牛市/熊市/震荡）
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data")

# Finnhub field names -> our factor names
FIELD_RENAME = {
    "roeTTM": "roe",
    "roaTTM": "roa",
    "grossMargin": "gross_margin",
    "operatingMargin": "operating_margin",
    "netMargin": "net_margin",
    "peTTM": "pe",
    "pb": "pb",
    "psTTM": "ps",
    "totalDebtToEquity": "debt_to_equity",
    "currentRatio": "current_ratio",
    "bookValue": "book_value",
    "eps": "eps",
}


# ============================================================================
# 基本面时间序列因子
# ============================================================================


def build_fundamental_panel_from_finnhub(
    raw_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    """
    从 Finnhub stock/metric 原始数据构建基本面因子面板。

    Args:
        raw_df: fetch_all_fundamentals() 返回的 DataFrame
                columns=[ticker, date, field, value]
        dates: 回测区间交易日
        tickers: 股票列表

    Returns:
        dict: {"roe": DataFrame, "roe_trend": DataFrame, ...}
              每个 DataFrame: index=Date, columns=Ticker
    """
    logger.info("=" * 60)
    logger.info("  从 Finnhub 数据构建基本面因子面板")
    logger.info("=" * 60)

    if raw_df.empty:
        logger.warning("基本面数据为空，跳过因子构建")
        return {}

    panel = {}

    # Pivot each field into a (date x ticker) panel
    for field_name, factor_name in FIELD_RENAME.items():
        subset = raw_df[raw_df["field"] == field_name].copy()
        if subset.empty:
            continue

        # Build date x ticker matrix
        factor_df = subset.pivot_table(
            index="date", columns="ticker", values="value", aggfunc="last"
        )

        # Reindex to trading dates, forward fill, back fill
        factor_df = factor_df.reindex(dates).ffill().bfill(limit=5)

        # Only keep tickers in our universe
        factor_df = factor_df[[c for c in factor_df.columns if c in tickers]]

        if factor_df.empty:
            continue

        valid = factor_df.notna().sum().sum()
        total = factor_df.size
        logger.info(f"  {factor_name}: 有效值 {valid}/{total} ({valid/total:.1%})")

        panel[factor_name] = factor_df

    # Derived factors
    n_raw = len(panel)
    logger.info(f"原始基本面因子: {n_raw} 个")

    # ROE trend (QoQ change in ROE)
    if "roe" in panel:
        panel["roe_trend"] = panel["roe"].diff(periods=1)
        logger.info("  roe_trend: ROE 季度环比变化")

    # Margin trend
    if "gross_margin" in panel:
        panel["margin_trend"] = panel["gross_margin"].diff(periods=1)
        logger.info("  margin_trend: 毛利率季度环比变化")

    # EPS growth (QoQ change)
    if "eps" in panel:
        panel["eps_growth"] = panel["eps"].pct_change(periods=1)
        logger.info("  eps_growth: EPS 季度环比增长率")

    # Profitability composite: ROE + gross_margin combined
    if "roe" in panel and "gross_margin" in panel:
        # Cross-sectional rank each, then average
        panel["profitability"] = (
            panel["roe"].rank(axis=1, pct=True)
            + panel["gross_margin"].rank(axis=1, pct=True)
        ) / 2
        logger.info("  profitability: ROE+毛利率综合排名")

    n_total = len(panel)
    logger.info(f"基本面因子构建完成: {n_total} 个因子 ({n_total - n_raw} 个衍生)")

    return panel


# ============================================================================
# 分析师情绪因子
# ============================================================================


def build_sentiment_panel_from_finnhub(
    eps_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    price_targets_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    """
    从 Finnhub 分析师数据构建情绪因子面板。

    Args:
        eps_df: fetch_all_eps_surprises() 的结果
        ratings_df: fetch_all_analyst_ratings() 的结果
        price_targets_df: fetch_all_price_targets() 的结果
        dates: 回测区间交易日
        tickers: 股票列表

    Returns:
        dict: {"eps_surprise": DataFrame, "analyst_consensus": DataFrame, ...}
    """
    logger.info("=" * 60)
    logger.info("  从 Finnhub 数据构建分析师情绪因子")
    logger.info("=" * 60)

    panel = {}

    # 1. EPS Surprise
    eps_panel = _build_eps_surprise_panel(eps_df, dates, tickers)
    if eps_panel is not None:
        panel["eps_surprise"] = eps_panel

    # 2. Analyst Consensus
    ratings_panel = _build_analyst_consensus_panel(ratings_df, dates, tickers)
    if ratings_panel is not None:
        panel["analyst_consensus"] = ratings_panel

    # 3. Target Price Upside
    target_panel = _build_target_upside_panel(price_targets_df, dates, tickers)
    if target_panel is not None:
        panel["target_upside"] = target_panel

    n_factors = len(panel)
    logger.info(f"分析师情绪因子构建完成: {n_factors} 个因子")
    for name, df in panel.items():
        valid = df.notna().sum().sum()
        total = df.size
        logger.info(f"  {name}: 有效值 {valid}/{total} ({valid/total:.1%})")

    return panel


def _build_eps_surprise_panel(
    eps_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame | None:
    """
    构建 EPS 超预期面板。

    在财报公布后的 21 个交易日内，使用超预期百分比作为信号值，线性衰减。
    """
    if eps_df.empty:
        return None

    panel = pd.DataFrame(0.0, index=dates, columns=tickers)

    for _, row in eps_df.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        surprise_pct = row["surprise_pct"]

        if pd.isna(date) or pd.isna(surprise_pct) or ticker not in tickers:
            continue

        # Find trading days within 30 calendar days after earnings date
        mask = (dates >= date) & (dates < date + pd.Timedelta(days=30))
        if mask.sum() == 0:
            continue

        # Linear decay over 21 trading days
        trading_days = dates[mask]
        for i, d in enumerate(trading_days):
            decay = max(0, 1 - i / 21)
            if d in panel.index:
                panel.at[d, ticker] = surprise_pct * decay

    return panel


def _build_analyst_consensus_panel(
    ratings_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame | None:
    """
    构建分析师共识评分面板。

    评分 = (5*strongBuy + 4*buy + 3*hold + 2*sell + 1*strongSell) / total
    """
    if ratings_df.empty:
        return None

    ratings_df = ratings_df.copy()
    ratings_df["total"] = (
        ratings_df["strongBuy"] + ratings_df["buy"]
        + ratings_df["hold"] + ratings_df["sell"]
        + ratings_df["strongSell"]
    )
    ratings_df = ratings_df[ratings_df["total"] > 0]
    ratings_df["consensus"] = (
        5 * ratings_df["strongBuy"]
        + 4 * ratings_df["buy"]
        + 3 * ratings_df["hold"]
        + 2 * ratings_df["sell"]
        + 1 * ratings_df["strongSell"]
    ) / ratings_df["total"]

    # Map period to days ago
    period_map = {
        "0m": 0, "1m": 30, "3m": 90, "6m": 180, "12m": 365,
    }
    ratings_df["days_ago"] = ratings_df["period"].map(period_map).fillna(365)

    panel = pd.DataFrame(np.nan, index=dates, columns=tickers)
    end_date = dates[-1]

    for _, row in ratings_df.iterrows():
        ticker = row["ticker"]
        if ticker not in tickers:
            continue

        rating_date = end_date - pd.Timedelta(days=int(row["days_ago"]))
        score = row["consensus"]

        # Forward fill from rating date onwards
        mask = dates >= rating_date
        panel.loc[mask, ticker] = score

    panel = panel.ffill()

    return panel


def _build_target_upside_panel(
    targets_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame | None:
    """
    构建目标价上涨空间面板。

    upside = (targetMean - currentPrice) / currentPrice
    """
    if targets_df.empty:
        return None

    panel = pd.DataFrame(np.nan, index=dates, columns=tickers)

    for ticker, row in targets_df.iterrows():
        if ticker not in tickers:
            continue

        target = row.get("targetMean")
        current = row.get("currentPrice")

        if pd.isna(target) or pd.isna(current) or current == 0:
            continue

        upside = (target - current) / current
        panel[ticker] = upside

    return panel


# ============================================================================
# 宏观因子
# ============================================================================


def build_macro_panel(
    fred_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """
    从 FRED 数据构建宏观因子面板。

    Returns:
        dict: {"term_spread": Series, "macro_regime": Series, ...}
              宏观数据对所有股票相同，以 Series 形式返回
    """
    logger.info("=" * 60)
    logger.info("  从 FRED 数据构建宏观因子面板")
    logger.info("=" * 60)

    panel = {}

    # Align to trading dates
    fred_aligned = fred_df.reindex(dates).ffill()

    # 1. Term spread (yield curve slope)
    if "term_spread" in fred_aligned.columns:
        panel["term_spread"] = fred_aligned["term_spread"]

    # 2. Fed rate level
    if "fed_rate" in fred_aligned.columns:
        panel["fed_rate"] = fred_aligned["fed_rate"]

    # 3. Macro regime (0=bull, 1=neutral, 2=bear)
    panel["macro_regime"] = _compute_macro_regime(fred_aligned)

    # 4. Inflation momentum
    if "inflation" in fred_aligned.columns:
        panel["inflation_momentum"] = fred_aligned["inflation"].pct_change(21)

    n_factors = len(panel)
    logger.info(f"宏观因子构建完成: {n_factors} 个因子")
    for name, s in panel.items():
        valid = s.notna().sum()
        logger.info(f"  {name}: 有效值 {valid}/{len(s)} ({valid/len(s):.1%})")

    return panel


def _compute_macro_regime(macro_df: pd.DataFrame) -> pd.Series:
    """
    计算宏观环境标签。

    简单规则：
    - term_spread < 0 -> bear (收益率曲线倒挂 = 衰退信号)
    - term_spread > 1% -> bull
    - otherwise -> neutral

    Returns:
        Series: 0=bull, 1=neutral, 2=bear
    """
    if "term_spread" not in macro_df.columns:
        return pd.Series(1, index=macro_df.index)

    spread = macro_df["term_spread"]
    regime = pd.Series(1, index=spread.index, dtype=float)  # neutral

    regime[spread > 1.0] = 0    # bull
    regime[spread < 0.0] = 2    # bear

    return regime


# ============================================================================
# 截面标准化
# ============================================================================


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面排名到 [0, 1]"""
    return df.rank(axis=1, pct=True)


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面 Z-Score"""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)

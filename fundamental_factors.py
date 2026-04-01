"""
fundamental_factors.py - 基本面因子模块（价值 + 质量）

v3 多因子策略的核心：
- 价值因子（PB）：买入低估值股票
- 质量因子（ROE）：买入盈利能力强的公司

数据来源：yfinance Ticker.info（当前快照）
处理方式：截面排名 + 前向填充到回测区间

注意：yfinance 免费版只提供当前时点的基本面快照，无法获取历史时间序列。
      我们用截面排名的方式处理——这是"因子截面有效性验证"的标准做法。
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# 缓存路径
CACHE_DIR = Path("data")
FUNDAMENTAL_CACHE = CACHE_DIR / "fundamental_snapshots.parquet"


def fetch_fundamental_data(
    tickers: list[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    批量获取基本面数据（PB、ROE 等）。

    Args:
        tickers: 股票代码列表
        force_refresh: 是否强制重新获取

    Returns:
        DataFrame: index=ticker, columns=['pb', 'roe', 'pe', 'sector', ...]
    """
    # 尝试加载缓存
    if FUNDAMENTAL_CACHE.exists() and not force_refresh:
        logger.info(f"从缓存加载基本面数据: {FUNDAMENTAL_CACHE}")
        df = pd.read_parquet(FUNDAMENTAL_CACHE)
        logger.info(f"缓存数据: {len(df)} 只股票")
        return df

    logger.info(f"开始获取 {len(tickers)} 只股票的基本面数据...")
    logger.info("预计耗时约 2 分钟（yfinance 频率限制）")

    records = []
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info

            pb = info.get("priceToBook")
            roe = info.get("returnOnEquity")
            pe = info.get("trailingPE")
            sector = info.get("sector", "Unknown")
            market_cap = info.get("marketCap")
            gross_margins = info.get("grossMargins")
            operating_margins = info.get("operatingMargins")
            profit_margins = info.get("profitMargins")
            revenue_growth = info.get("revenueGrowth")
            earnings_growth = info.get("earningsGrowth")

            records.append({
                "ticker": ticker,
                "pb": pb,
                "roe": roe,
                "pe": pe,
                "sector": sector,
                "market_cap": market_cap,
                "gross_margins": gross_margins,
                "operating_margins": operating_margins,
                "profit_margins": profit_margins,
                "revenue_growth": revenue_growth,
                "earnings_growth": earnings_growth,
            })

        except Exception as e:
            failed.append(ticker)
            logger.debug(f"获取 {ticker} 基本面数据失败: {e}")

        # 每 50 只股票打印进度
        if (i + 1) % 50 == 0:
            logger.info(f"  进度: {i + 1}/{len(tickers)}")

        # 避免频率限制
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    df = pd.DataFrame(records).set_index("ticker")

    if failed:
        logger.warning(f"{len(failed)} 只股票获取失败: {failed[:5]}{'...' if len(failed) > 5 else ''}")

    # 保存缓存
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FUNDAMENTAL_CACHE)
    logger.info(f"基本面数据已缓存: {len(df)} 只股票, 保存至 {FUNDAMENTAL_CACHE}")

    return df


def compute_value_factor(df_fund: pd.DataFrame) -> pd.Series:
    """
    计算价值因子：基于 PB（市净率）的截面排名。

    PB 越低 = 越便宜 = 得分越高（取负号排名）。

    处理：
    1. 过滤 PB 缺失或 <= 0 的股票（金融行业的特殊情况）
    2. 用中位数绝对偏差（MAD）去极值
    3. 截面排名到 [0, 1]

    Args:
        df_fund: fetch_fundamental_data 返回的 DataFrame

    Returns:
        Series: 价值因子得分（0~1），越大表示越"便宜"
    """
    logger.info("=" * 50)
    logger.info("计算价值因子（PB）")
    logger.info("=" * 50)

    pb = df_fund["pb"].copy()

    # 过滤无效值
    valid_mask = pb.notna() & (pb > 0)
    n_invalid = (~valid_mask).sum()
    logger.info(f"PB 有效: {valid_mask.sum()} 只, 无效: {n_invalid} 只")

    pb_valid = pb[valid_mask]

    # MAD 去极值
    median = pb_valid.median()
    mad = (pb_valid - median).abs().median()
    lower = median - 3 * 1.4826 * mad
    upper = median + 3 * 1.4826 * mad
    pb_clean = pb_valid.clip(lower, upper)

    # 排名：PB 越低越好 → 取负号排名
    value_score = (-pb_clean).rank(pct=True)

    # 合并回完整 Series
    result = pd.Series(np.nan, index=df_fund.index)
    result[valid_mask] = value_score

    logger.info(f"价值因子计算完成: 均值={result.mean():.3f}, 标准差={result.std():.3f}")
    return result


def compute_quality_factor(df_fund: pd.DataFrame) -> pd.Series:
    """
    计算质量因子：基于 ROE（净资产收益率）的截面排名。

    ROE 越高 = 盈利能力越强 = 得分越高。

    处理：
    1. 过滤 ROE 缺失的股票
    2. MAD 去极值
    3. 截面排名到 [0, 1]

    Args:
        df_fund: fetch_fundamental_data 返回的 DataFrame

    Returns:
        Series: 质量因子得分（0~1），越大表示盈利能力越强
    """
    logger.info("=" * 50)
    logger.info("计算质量因子（ROE）")
    logger.info("=" * 50)

    roe = df_fund["roe"].copy()

    # 过滤无效值
    valid_mask = roe.notna()
    n_invalid = (~valid_mask).sum()
    logger.info(f"ROE 有效: {valid_mask.sum()} 只, 无效: {n_invalid} 只")

    roe_valid = roe[valid_mask]

    # MAD 去极值
    median = roe_valid.median()
    mad = (roe_valid - median).abs().median()
    lower = median - 3 * 1.4826 * mad
    upper = median + 3 * 1.4826 * mad
    roe_clean = roe_valid.clip(lower, upper)

    # 排名：ROE 越高越好
    quality_score = roe_clean.rank(pct=True)

    # 合并回完整 Series
    result = pd.Series(np.nan, index=df_fund.index)
    result[valid_mask] = quality_score

    logger.info(f"质量因子计算完成: 均值={result.mean():.3f}, 标准差={result.std():.3f}")
    return result


def compute_composite_factor(
    df_fund: pd.DataFrame,
    weights: dict = None,
) -> pd.Series:
    """
    计算多因子综合得分 = w1 × 价值 + w2 × 质量。

    各因子先做截面排名（已处理极端值），再加权合成。

    Args:
        df_fund: 基本面数据 DataFrame
        weights: 因子权重，默认 {"value": 0.5, "quality": 0.5}

    Returns:
        Series: 综合因子得分
    """
    if weights is None:
        weights = {"value": 0.5, "quality": 0.5}

    value = compute_value_factor(df_fund)
    quality = compute_quality_factor(df_fund)

    composite = value * weights["value"] + quality * weights["quality"]

    logger.info(f"综合因子: 权重={weights}, 均值={composite.mean():.3f}")

    return composite


def build_fundamental_panel(
    df_fund: pd.DataFrame,
    dates: pd.DatetimeIndex,
    factor_name: str = "composite",
    weights: dict = None,
) -> pd.DataFrame:
    """
    将截面因子数据扩展为时间序列面板。

    由于 yfinance 只提供当前快照，我们用当前截面排名作为最近期的因子值，
    前向填充到整个回测区间。这是"因子截面有效性验证"的标准做法。

    注意：这意味着回测中所有时间点使用相同的 PB/ROE 排名。
          这会高估历史表现（假设这些公司的相对排名在过去不变）。
          但对于验证"多因子 vs 单因子"的相对优劣是足够的。

    Args:
        df_fund: 基本面数据 DataFrame
        dates: 回测区间的交易日 DatetimeIndex
        factor_name: "value", "quality", 或 "composite"
        weights: 多因子权重（仅 composite 需要）

    Returns:
        DataFrame: index=Date, columns=ticker, values=factor score
    """
    logger.info(f"构建基本面因子面板 ({factor_name})...")

    if factor_name == "value":
        scores = compute_value_factor(df_fund)
    elif factor_name == "quality":
        scores = compute_quality_factor(df_fund)
    elif factor_name == "composite":
        scores = compute_composite_factor(df_fund, weights)
    else:
        raise ValueError(f"未知因子: {factor_name}")

    # 构建面板：每个日期使用相同的截面排名
    panel = pd.DataFrame(
        np.nan,
        index=dates,
        columns=df_fund.index,
    )

    for ticker in scores.dropna().index:
        if ticker in panel.columns:
            panel[ticker] = scores[ticker]

    # 前向填充（实际上所有行都一样，但保持接口一致）
    panel = panel.ffill()

    valid_count = panel.notna().sum(axis=1).mean()
    logger.info(f"因子面板构建完成: {len(dates)} 个交易日, 平均 {valid_count:.0f} 只股票有有效值")

    return panel


def check_factor_correlations(df_fund: pd.DataFrame) -> pd.DataFrame:
    """
    检查各因子之间的截面相关性。

    这是多因子模型的核心诊断——低相关因子组合才能有效分散风险。

    Args:
        df_fund: 基本面数据 DataFrame

    Returns:
        DataFrame: 相关系数矩阵
    """
    logger.info("=" * 50)
    logger.info("因子相关性分析")
    logger.info("=" * 50)

    value = compute_value_factor(df_fund)
    quality = compute_quality_factor(df_fund)

    corr_df = pd.DataFrame({
        "value": value,
        "quality": quality,
    }).corr()

    logger.info("\n因子相关系数矩阵:")
    for i in corr_df.index:
        for j in corr_df.columns:
            r = corr_df.loc[i, j]
            if i != j:
                status = "✅ 低" if abs(r) < 0.3 else ("⚠️ 中" if abs(r) < 0.5 else "❌ 高")
                logger.info(f"  {i} vs {j}: {r:.3f} ({status})")

    return corr_df

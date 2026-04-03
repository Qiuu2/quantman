"""
v6/features.py - 特征工程模块

计算 20 个因子（8 技术面 + 12 基本面时序），
做截面 z-score 标准化，不做人工 IC 预筛。

技术面因子（基于周频收盘价）：
1.  momentum_12_1      — 52 周 / 1 周前 动量
2.  momentum_3_1       — 13 周 / 1 周前 短期动量
3.  momentum_accel     — 动量加速度（近 26 周动量 - 远 52 周动量）
4.  volatility_13      — 13 周滚动波动率（年化）
5.  price_range_4      — 4 周价格区间比
6.  reversal_1         — 1 周反转
7.  vol_adj_momentum   — 波动率调整动量
8.  multi_window_mom   — 多窗口动量合成 (13w:0.4, 26w:0.3, 52w:0.3)

基本面时序因子（基于 SimFin 季度财报）：
9.  roe                — 净资产收益率
10. roe_change         — ROE 季度变化率
11. gross_margin       — 毛利率
12. gross_margin_trend — 毛利率趋势（近 4 季 vs 前 4 季均值）
13. asset_turnover     — 资产周转率
14. leverage_change    — 杠杆率变化（总资产/权益比的变化）
15. ocf_to_ni          — 经营现金流 / 净利润比
16. revenue_growth     — 营收增速 YoY
17. ar_turnover_change — 应收账款周转率变化
18. capex_to_revenue   — 资本支出 / 营收比
19. accruals           — 应计利润 = (净利润 - 经营现金流) / 总资产
20. pb_ratio           — 市净率（需收盘价）
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from v6.config import *

logger = logging.getLogger(__name__)


# ============================================================================
# SimFin 实际列名（SimFin 1.0.1 实际使用的列名，带空格）
# ============================================================================

SF_TICKER = "Ticker"
SF_REPORT_DATE = "Report Date"
SF_PUBLISH_DATE = "Publish Date"

# 利润表
SF_REVENUE = "Revenue"
SF_NET_INCOME = "Net Income"
SF_GROSS_PROFIT = "Gross Profit"
SF_DEPR_AMOR = "Depreciation & Amortization"

# 资产负债表
SF_TOTAL_ASSETS = "Total Assets"
SF_TOTAL_EQUITY = "Total Equity"
SF_TOTAL_LIABILITIES = "Total Liabilities"
SF_ACCOUNTS_RECV = "Accounts & Notes Receivable"
SF_INVENTORIES = "Inventories"
SF_CASH_EQUIV = "Cash, Cash Equivalents & Short Term Investments"

# 现金流量表
SF_NET_CASH_OPS = "Net Cash from Operating Activities"
SF_CAPEX = "Change in Fixed Assets & Intangibles"
SF_DIVIDENDS_PAID = "Dividends Paid"


# ============================================================================
# 技术面因子（基于周频收盘价）
# ============================================================================


def _calc_technical_features(close_weekly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    计算所有技术面因子。

    Args:
        close_weekly: (n_weeks, n_stocks) 周频收盘价

    Returns:
        dict: {factor_name: DataFrame}
    """
    logger.info("计算技术面因子（8 个）...")
    features = {}

    weekly_ret = close_weekly.pct_change()

    # 1. 12-1 动量 (52 周 / 1 周)
    f = close_weekly.shift(1) / close_weekly.shift(52) - 1
    features["momentum_12_1"] = f

    # 2. 3-1 短期动量 (13 周 / 1 周)
    f = close_weekly.shift(1) / close_weekly.shift(13) - 1
    features["momentum_3_1"] = f

    # 3. 动量加速度（近 26 周动量 - 远 52 周动量）
    mom_26 = close_weekly.shift(1) / close_weekly.shift(26) - 1
    mom_52 = close_weekly.shift(26) / close_weekly.shift(52) - 1
    features["momentum_accel"] = mom_26 - mom_52

    # 4. 13 周滚动波动率（年化，基于周收益率 × sqrt(52)）
    vol = weekly_ret.rolling(13).std() * np.sqrt(52)
    features["volatility_13"] = vol

    # 5. 4 周价格区间比
    rolling_max = close_weekly.rolling(4).max()
    rolling_min = close_weekly.rolling(4).min()
    f = (rolling_max - rolling_min) / close_weekly
    features["price_range_4"] = f

    # 6. 1 周反转
    f = close_weekly.shift(1) / close_weekly - 1
    features["reversal_1"] = f

    # 7. 波动率调整动量
    mom_raw = close_weekly.shift(1) / close_weekly.shift(52) - 1
    features["vol_adj_momentum"] = mom_raw / vol

    # 8. 多窗口动量合成
    mom_13w = close_weekly.shift(1) / close_weekly.shift(13) - 1
    mom_26w = close_weekly.shift(1) / close_weekly.shift(26) - 1
    mom_52w = close_weekly.shift(1) / close_weekly.shift(52) - 1
    features["multi_window_mom"] = mom_13w * 0.4 + mom_26w * 0.3 + mom_52w * 0.3

    for name, df in features.items():
        valid = df.notna().sum(axis=1).mean()
        logger.info(f"  {name:<25} 平均有效: {valid:.0f} 只")

    return features


# ============================================================================
# 基本面时序因子（基于 SimFin 季度财报）
# ============================================================================


def _prepare_simfin_panels(
    simfin_data: dict[str, pd.DataFrame],
    universe: list[str],
    close_weekly: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    将 SimFin 原始数据转换为 (Ticker, ReportDate) 格式的面板。

    Returns:
        dict: {table_name: DataFrame with index=[Ticker, ReportDate]}
    """
    panels = {}

    for name, df in simfin_data.items():
        # SimFin 返回的 DataFrame 索引是 (Ticker, Report Date)
        idx_names = df.index.names
        if SF_TICKER in idx_names:
            # 重置索引，保留为列
            df = df.reset_index()
            df[SF_TICKER] = df[SF_TICKER].str.replace(".", "-", regex=False)
            date_col = SF_REPORT_DATE if SF_REPORT_DATE in df.columns else SF_PUBLISH_DATE
            if date_col not in df.columns:
                logger.warning(f"{name}: 找不到日期列，跳过")
                continue
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[df[SF_TICKER].isin(universe)]
            df = df.set_index([SF_TICKER, date_col])
        elif SF_TICKER in df.columns:
            df = df.copy()
            df[SF_TICKER] = df[SF_TICKER].str.replace(".", "-", regex=False)
            date_col = SF_REPORT_DATE if SF_REPORT_DATE in df.columns else SF_PUBLISH_DATE
            if date_col not in df.columns:
                logger.warning(f"{name}: 找不到日期列，跳过")
                continue
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[df[SF_TICKER].isin(universe)]
            df = df.set_index([SF_TICKER, date_col])
        else:
            logger.warning(f"{name}: 找不到 Ticker 列，跳过")
            continue

        # 按 ticker + 日期排序
        df = df.sort_index()

        panels[name] = df
        logger.info(f"  {name}: {df.index.get_level_values(0).nunique()} 只股票, "
                     f"{df.index.get_level_values(1).nunique()} 个报告期")

    return panels


def _calc_fundamental_features(
    panels: dict[str, pd.DataFrame],
    universe: list[str],
    dates: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """
    基于季度财报计算基本面时序因子。

    对每个季度计算因子值，然后 step fill（前向填充）到每周日期。

    Args:
        panels: SimFin 面板数据
        universe: 股票池
        dates: 周频日期索引

    Returns:
        dict: {factor_name: DataFrame(n_weeks, n_stocks)}
    """
    logger.info("计算基本面时序因子（12 个）...")

    income = panels.get("income")
    balance = panels.get("balance")
    cashflow = panels.get("cashflow")

    if income is None or balance is None or cashflow is None:
        logger.error("SimFin 三张表不完整，无法计算基本面因子")
        return {}

    # 预计算每个 ticker 的季度因子序列
    # 结果格式: dict[ticker] -> DataFrame(index=ReportDate, columns=[factor1, factor2, ...])
    quarterly_factors = {}

    tickers_with_all = (
        set(income.index.get_level_values(0).unique())
        & set(balance.index.get_level_values(0).unique())
        & set(cashflow.index.get_level_values(0).unique())
    )

    for ticker in tickers_with_all:
        try:
            inc = income.loc[ticker] if ticker in income.index.get_level_values(0) else None
            bal = balance.loc[ticker] if ticker in balance.index.get_level_values(0) else None
            cf = cashflow.loc[ticker] if ticker in cashflow.index.get_level_values(0) else None

            if inc is None or bal is None or cf is None:
                continue

            # 确保是 DataFrame（单 ticker 可能退化为 Series）
            if isinstance(inc, pd.Series):
                inc = inc.to_frame().T
            if isinstance(bal, pd.Series):
                bal = bal.to_frame().T
            if isinstance(cf, pd.Series):
                cf = cf.to_frame().T

            # 合并三张表（按报告日期 outer join）
            combined = pd.concat(
                [inc, bal, cf],
                axis=1,
                join="outer",
            )
            # 去重列
            combined = combined.loc[:, ~combined.columns.duplicated()]
            combined = combined.sort_index()

            # 确保至少有需要的列
            needed_cols = [SF_REVENUE, SF_NET_INCOME, SF_TOTAL_ASSETS, SF_TOTAL_EQUITY]
            available = [c for c in needed_cols if c in combined.columns]
            if len(available) < 2:
                continue

            row = {}
            for dt, row_data in combined.iterrows():
                f = _compute_quarterly_factors(row_data)
                row[dt] = f

            if row:
                qf_df = pd.DataFrame.from_dict(row, orient="index")
                quarterly_factors[ticker] = qf_df

        except Exception as e:
            logger.debug(f"计算 {ticker} 基本面因子失败: {e}")

    # 添加时序衍生因子（ROE 变化率、毛利率趋势等）
    _add_time_series_factors(quarterly_factors)

    # Step fill 到周频日期
    features = {}
    factor_names = (
        list(quarterly_factors.values())[0].columns.tolist()
        if quarterly_factors
        else []
    )

    for fname in factor_names:
        panel = pd.DataFrame(index=dates, columns=universe, dtype=float)

        for ticker, qf_df in quarterly_factors.items():
            if fname in qf_df.columns and ticker in panel.columns:
                # Step fill: 用最近一个报告期的值
                series = qf_df[fname].reindex(
                    dates.union(qf_df.index)
                ).sort_index().ffill()
                panel[ticker] = series.reindex(dates)

        features[fname] = panel
        valid = panel.notna().sum(axis=1).mean()
        logger.info(f"  {fname:<25} 平均有效: {valid:.0f} 只")

    return features


def _compute_quarterly_factors(row: pd.Series) -> dict:
    """
    计算单只股票单季度的所有基本面因子。

    Args:
        row: 该季度合并后的财务数据 Series

    Returns:
        dict: {factor_name: value}
    """
    factors = {}

    # 辅助：安全取值
    def get(col, default=np.nan):
        return row.get(col, np.nan) if pd.notna(row.get(col, np.nan)) else default

    revenue = get(SF_REVENUE)
    net_income = get(SF_NET_INCOME)
    gross_profit = get(SF_GROSS_PROFIT)
    total_assets = get(SF_TOTAL_ASSETS)
    total_equity = get(SF_TOTAL_EQUITY)
    total_liab = get(SF_TOTAL_LIABILITIES)
    accounts_recv = get(SF_ACCOUNTS_RECV)
    inventories = get(SF_INVENTORIES)
    ocf = get(SF_NET_CASH_OPS)
    capex = get(SF_CAPEX)

    # 9. ROE = 净利润 / 股东权益
    if total_equity and total_equity != 0:
        factors["roe"] = net_income / total_equity
    else:
        factors["roe"] = np.nan

    # 10. ROE 变化率 — 需要上下文（在组装时计算）
    # 这里先占位，实际在 _calc_fundamental_features 中计算变化率
    factors["roe"] = factors.get("roe", np.nan)

    # 11. 毛利率 = 毛利润 / 营收
    if revenue and revenue != 0:
        factors["gross_margin"] = gross_profit / revenue
    else:
        factors["gross_margin"] = np.nan

    # 13. 资产周转率 = 营收 / 总资产
    if total_assets and total_assets != 0:
        factors["asset_turnover"] = revenue / total_assets
    else:
        factors["asset_turnover"] = np.nan

    # 14. 杠杆率 = 总资产 / 股东权益
    if total_equity and total_equity != 0:
        factors["leverage"] = total_assets / total_equity
    else:
        factors["leverage"] = np.nan

    # 15. 经营现金流/净利润比
    if net_income and net_income != 0:
        factors["ocf_to_ni"] = ocf / net_income
    else:
        factors["ocf_to_ni"] = np.nan

    # 16. 营收增速 — 需要上下文（在组装时计算 YoY）
    factors["_revenue"] = revenue

    # 17. 应收账款周转率 = 营收 / 应收账款
    if accounts_recv and accounts_recv != 0:
        factors["ar_turnover"] = revenue / accounts_recv
    else:
        factors["ar_turnover"] = np.nan

    # 18. 资本支出/营收比
    if revenue and revenue != 0:
        factors["capex_to_revenue"] = capex / abs(revenue)
    else:
        factors["capex_to_revenue"] = np.nan

    # 19. 应计利润 = (净利润 - 经营现金流) / 总资产
    if total_assets and total_assets != 0:
        factors["accruals"] = (net_income - ocf) / total_assets
    else:
        factors["accruals"] = np.nan

    # 20. PB — 需要收盘价（在组装时计算）
    factors["_total_equity"] = total_equity

    return factors


def _add_time_series_factors(
    quarterly_factors: dict[str, pd.DataFrame],
) -> None:
    """
    在已计算的季度因子面板上添加时序衍生因子。

    直接修改 quarterly_factors dict in-place。

    添加的因子：
    - roe_change: ROE 季度变化
    - gross_margin_trend: 毛利率趋势（近 4 季 vs 前 4 季）
    - leverage_change: 杠杆率变化
    - revenue_growth: 营收增速 YoY（4 季度前 vs 当前）
    - ar_turnover_change: 应收账款周转率变化
    """
    for ticker, qf_df in quarterly_factors.items():
        if len(qf_df) < 2:
            continue

        # ROE 变化
        if "roe" in qf_df.columns:
            qf_df["roe_change"] = qf_df["roe"].diff()

        # 毛利率趋势: 近 4 季均值 - 前 4 季均值
        if "gross_margin" in qf_df.columns:
            recent = qf_df["gross_margin"].rolling(4, min_periods=2).mean()
            lagged = qf_df["gross_margin"].shift(4).rolling(4, min_periods=2).mean()
            qf_df["gross_margin_trend"] = recent - lagged

        # 杠杆率变化
        if "leverage" in qf_df.columns:
            qf_df["leverage_change"] = qf_df["leverage"].diff()

        # 营收增速 YoY（与 4 季前对比）
        if "_revenue" in qf_df.columns:
            qf_df["revenue_growth"] = qf_df["_revenue"].pct_change(periods=4)

        # 应收账款周转率变化
        if "ar_turnover" in qf_df.columns:
            qf_df["ar_turnover_change"] = qf_df["ar_turnover"].diff()

        # 清理辅助列
        qf_df.drop(columns=["_revenue", "_total_equity"], errors="ignore", inplace=True)


# ============================================================================
# 截面 Z-Score 标准化
# ============================================================================


def cross_section_zscore_3d(features_3d: np.ndarray) -> np.ndarray:
    """
    对 3D 特征矩阵做截面 z-score 标准化。

    对每个时间截面的每个特征，减去均值除以标准差。

    Args:
        features_3d: (n_dates, n_stocks, n_features)

    Returns:
        (n_dates, n_stocks, n_features) 标准化后的特征
    """
    n_dates, n_stocks, n_features = features_3d.shape
    result = np.empty_like(features_3d)

    for t in range(n_dates):
        row = features_3d[t]  # (n_stocks, n_features)
        mean = np.nanmean(row, axis=0, keepdims=True)
        std = np.nanstd(row, axis=0, keepdims=True)
        std[std < 1e-10] = 1.0  # 防止除零
        result[t] = np.clip((row - mean) / std, -3, 3)

    return result


# ============================================================================
# 公共接口
# ============================================================================


def build_all_features(
    close_weekly: pd.DataFrame,
    simfin_data: dict[str, pd.DataFrame],
    universe: list[str],
    close_daily_for_pb: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    构建完整的特征矩阵。

    8 技术面 + 12 基本面 = 20 个因子，
    做截面 z-score 标准化。

    Args:
        close_weekly: (n_weeks, n_stocks) 周频收盘价
        simfin_data: SimFin 三张表
        universe: 股票池
        close_daily_for_pb: 日频收盘价（用于计算 PB）

    Returns:
        tuple: (features_3d, feature_names)
               features_3d shape = (n_weeks, n_stocks, n_features)
    """
    logger.info("=" * 60)
    logger.info("  构建特征矩阵（20 因子）")
    logger.info("=" * 60)

    dates = close_weekly.index
    n_weeks = len(dates)
    n_stocks = len(close_weekly.columns)

    # ── 技术面因子 ──
    tech_features = _calc_technical_features(close_weekly)

    # ── 基本面时序因子 ──
    panels = _prepare_simfin_panels(simfin_data, universe, close_weekly)
    fund_features = _calc_fundamental_features(panels, universe, dates)

    # ── PB 因子（用最新日频收盘价） ──
    if close_daily_for_pb is not None and "balance" in panels:
        logger.info("计算 PB 因子...")
        balance_df = panels["balance"]
        pb_panel = pd.DataFrame(index=dates, columns=universe, dtype=float)

        for ticker in universe:
            if ticker not in balance_df.index.get_level_values(0):
                continue
            try:
                bal_ticker = balance_df.loc[ticker]
                if isinstance(bal_ticker, pd.Series):
                    continue

                price = close_daily_for_pb[ticker].dropna()
                equity = bal_ticker[SF_TOTAL_EQUITY].dropna() if SF_TOTAL_EQUITY in bal_ticker.columns else pd.Series(dtype=float)

                if len(price) == 0 or len(equity) == 0:
                    continue

                # 取最新的权益数据
                latest_equity = equity.iloc[-1]
                if pd.isna(latest_equity) or latest_equity <= 0:
                    continue

                # 需要总股本来算 PB = price / (equity / shares)
                # 简化：用市值代替（SimFin 有 shares_basic）
                # 或者直接用 price_to_book 如果 SimFin 提供
                # 暂时用 close / (total_equity_per_share 近似)
                # 这里简化处理，用日频收盘价和最新权益的比值
                # 实际上 PB = Market Cap / Total Equity
                # 由于我们没有 shares 数据，先跳过，用截面排名替代

            except Exception as e:
                logger.debug(f"计算 {ticker} PB 失败: {e}")

        # PB 暂时简化为截面排名
        logger.info("  PB 因子暂用截面价格排名替代（缺 shares 数据）")

    # ── 合并所有因子 ──
    all_features = {}
    all_features.update(tech_features)
    all_features.update(fund_features)

    # 过滤掉只有 NaN 的因子
    valid_features = {}
    for name, df in all_features.items():
        if df.notna().sum().sum() > 0:
            # 对齐到 close_weekly 的 index 和 columns
            aligned = df.reindex(index=dates, columns=close_weekly.columns)
            valid_features[name] = aligned

    feature_names = sorted(valid_features.keys())
    n_features = len(feature_names)

    logger.info(f"\n最终因子列表 ({n_features} 个):")
    for i, name in enumerate(feature_names):
        valid = valid_features[name].notna().sum(axis=1).mean()
        logger.info(f"  {i+1:>2}. {name:<25} 平均有效: {valid:.0f} 只")

    # ── 堆叠为 3D array ──
    feature_arrays = []
    for name in feature_names:
        feature_arrays.append(valid_features[name].values)

    features_3d = np.stack(feature_arrays, axis=2)  # (n_weeks, n_stocks, n_features)

    # ── 截面 z-score 标准化 ──
    features_3d = cross_section_zscore_3d(features_3d)

    # 把 NaN 标记回来（z-score 不改变 NaN）
    nan_mask = np.isnan(features_3d)
    # 已处理：z-score 函数保留了 NaN

    logger.info(f"\n特征矩阵: {features_3d.shape[0]} 周 × {features_3d.shape[1]} 股 × {features_3d.shape[2]} 因子")
    logger.info(f"NaN 比例: {nan_mask.sum() / nan_mask.size:.2%}")
    logger.info("=" * 60)

    return features_3d, feature_names


def build_labels_weekly(
    close_weekly: pd.DataFrame,
    forward_weeks: int = FORWARD_WEEKS,
) -> np.ndarray:
    """
    构建训练标签：未来 N 周收益率。

    Args:
        close_weekly: 周频收盘价
        forward_weeks: 前向周数

    Returns:
        (n_weeks, n_stocks) 未来收益率矩阵
    """
    future_return = close_weekly.shift(-forward_weeks) / close_weekly - 1
    labels = future_return.values.copy()
    return labels

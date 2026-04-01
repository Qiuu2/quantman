"""
v6/data_loader.py - 数据获取与对齐模块

两大数据源：
1. SimFin - 批量下载季度财报（利润表、资产负债表、现金流量表）
2. yfinance - Russell 3000 日线价格（batch + 限速 + 重试 + parquet 缓存）

核心功能：
- 获取 Russell 3000 成分股列表
- SimFin 数据批量加载与 ticker 对齐
- 股票池过滤（按数据完整度）
- 日频 → 周频降采样
"""

import io
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import simfin as sf

from v6.config import *

logger = logging.getLogger(__name__)

# SimFin 实际列名（SimFin 1.0.1）
SF_TICKER = "Ticker"
SF_REPORT_DATE = "Report Date"
SF_PUBLISH_DATE = "Publish Date"

# ── Russell 3000 获取 ──

RUSSELL3000_WIKI_URL = "https://en.wikipedia.org/wiki/Russell_3000_Index"

_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def _fetch_tickers_from_wikipedia(url: str, ticker_col: str = "Symbol") -> list[str]:
    """从 Wikipedia 表格中提取 ticker 列表"""
    resp = requests.get(url, headers=_WIKI_HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    for table in tables:
        cols_lower = [str(c).lower() for c in table.columns]
        if ticker_col.lower() in cols_lower:
            col = table.columns[cols_lower.index(ticker_col.lower())]
            tickers = table[col].dropna().astype(str).str.strip().tolist()
            tickers = [t.replace(".", "-") for t in tickers]
            return tickers

    return []


def get_russell3000_tickers() -> list[str]:
    """
    获取扩展股票池列表。

    策略（按优先级）：
    1. Wikipedia Russell 3000 页面
    2. 合并 SP500 + SP400 + SP600 ≈ S&P 1500
    3. 回退到 SP500
    """
    logger.info("获取股票池列表（目标: Russell 3000）...")

    # 方案 1: 尝试 Wikipedia Russell 3000 页面
    try:
        resp = requests.get(RUSSELL3000_WIKI_URL, headers=_WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))

        for i, table in enumerate(tables):
            cols_lower = [str(c).lower() for c in table.columns]
            for target in ["ticker", "symbol", "ticker symbol"]:
                if target in cols_lower:
                    col = table.columns[cols_lower.index(target)]
                    tickers = table[col].dropna().astype(str).str.strip().tolist()
                    tickers = [t.replace(".", "-") for t in tickers]
                    tickers = [t for t in tickers if 1 <= len(t) <= 5 and t.replace("-", "").isalpha()]
                    if len(tickers) > 500:
                        logger.info(f"获取到 {len(tickers)} 只成分股 (Wikipedia)")
                        return tickers
    except Exception as e:
        logger.debug(f"Wikipedia Russell 3000 尝试失败: {e}")

    # 方案 2: 合并 SP500 + SP400 + SP600
    logger.info("尝试合并 S&P 500/400/600 构建扩展股票池...")
    all_tickers = set()

    for name, url, col in [
        ("SP500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
        ("SP400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol"),
        ("SP600", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol"),
    ]:
        try:
            t = _fetch_tickers_from_wikipedia(url, col)
            all_tickers.update(t)
            logger.info(f"  {name}: {len(t)} 只")
        except Exception:
            logger.debug(f"{name} 获取失败")

    if len(all_tickers) > 500:
        tickers = sorted(all_tickers)
        logger.info(f"合并股票池: {len(tickers)} 只")
        return tickers

    # 方案 3: 回退到 SP500
    logger.warning("回退到 SP500 成分股列表...")
    try:
        return _fetch_tickers_from_wikipedia(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"
        )
    except Exception as e:
        logger.error(f"SP500 也获取失败: {e}")
        return []


# ── yfinance 批量下载 ──


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    benchmark_ticker: str = BENCHMARK_TICKER,
    cache_path: Optional[Path] = None,
    batch_size: int = YF_BATCH_SIZE,
    sleep_seconds: float = YF_SLEEP_SECONDS,
    max_retries: int = YF_MAX_RETRIES,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    批量下载股票日线数据，带限速、重试和本地缓存。

    Args:
        tickers: 股票代码列表
        start_date / end_date: 日期范围
        benchmark_ticker: 基准代码
        cache_path: parquet 缓存路径
        batch_size: 每批下载数量
        sleep_seconds: 批次间等待秒数
        max_retries: 失败重试次数
        force_refresh: 是否强制重新下载

    Returns:
        DataFrame: index=Date, columns=ticker, 收盘价
    """
    if cache_path is None:
        cache_path = DATA_DIR / "russell3000_close.parquet"

    # 尝试加载缓存
    if cache_path.exists() and not force_refresh:
        logger.info(f"从缓存加载价格数据: {cache_path}")
        close = pd.read_parquet(cache_path)
        logger.info(f"缓存: {close.shape[0]} 个交易日, {close.shape[1]} 只股票")
        return close

    all_tickers = list(set(tickers + [benchmark_ticker]))
    logger.info(f"批量下载 {len(all_tickers)} 只股票 ({start_date} ~ {end_date})...")
    logger.info(f"批次大小: {batch_size}, 批次间间隔: {sleep_seconds}s")

    close_frames = []
    n_batches = (len(all_tickers) + batch_size - 1) // batch_size

    for batch_i in range(n_batches):
        start_idx = batch_i * batch_size
        end_idx = min(start_idx + batch_size, len(all_tickers))
        batch_tickers = all_tickers[start_idx:end_idx]

        for retry in range(max_retries):
            try:
                logger.info(
                    f"  批次 {batch_i + 1}/{n_batches} "
                    f"(tickers {start_idx + 1}-{end_idx}), "
                    f"尝试 {retry + 1}/{max_retries}..."
                )
                raw = yf.download(
                    batch_tickers,
                    start=start_date,
                    end=end_date,
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    timeout=30,
                )

                # yfinance 新版: MultiIndex (ticker, field)
                if isinstance(raw.columns, pd.MultiIndex):
                    for field in ["Close", "close"]:
                        try:
                            df_field = raw.xs(field, axis=1, level=1)
                            break
                        except KeyError:
                            continue
                    else:
                        logger.warning(f"  批次 {batch_i + 1}: 无法提取收盘价，跳过")
                        continue
                else:
                    df_field = raw

                df_field.index = pd.to_datetime(df_field.index)
                df_field = df_field.dropna(axis=1, how="all")
                df_field.columns = [c.replace(".", "-") for c in df_field.columns]

                close_frames.append(df_field)
                logger.info(
                    f"    ✓ 获取 {len(df_field.columns)} 只股票, "
                    f"{len(df_field)} 个交易日"
                )
                break

            except Exception as e:
                logger.warning(f"    ✗ 失败: {e}")
                if retry == max_retries - 1:
                    logger.error(f"    批次 {batch_i + 1} 下载失败，跳过")

        # 批次间等待
        if batch_i < n_batches - 1:
            time.sleep(sleep_seconds)

    # 合并所有批次
    if not close_frames:
        raise RuntimeError("所有批次下载失败！")

    # 用 concat 合并（不同批次的列不同，用 outer join）
    close = pd.concat(close_frames, axis=1)
    close = close.sort_index()

    # 去重列（可能有些 ticker 在多个批次中）
    close = close.loc[:, ~close.columns.duplicated()]

    # 保存缓存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(cache_path)
    logger.info(f"价格数据已缓存: {close.shape[0]} 日 × {close.shape[1]} 股 → {cache_path}")

    return close


# ── SimFin 数据加载 ──


def load_simfin_data(
    api_key: str = SIMFIN_API_KEY,
    cache_dir: Path = SIMFIN_CACHE_DIR,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    批量加载 SimFin 季度财报数据。

    SimFin 使用批量下载模式，3 次调用获取全美股三张表。
    数据自动缓存到本地磁盘。

    Returns:
        dict: {
            "income": 利润表 DataFrame,
            "balance": 资产负债表 DataFrame,
            "cashflow": 现金流量表 DataFrame,
        }
    """
    # 配置 SimFin
    sf.set_api_key(api_key=api_key)
    sf.set_data_dir(str(cache_dir))

    logger.info("=" * 60)
    logger.info("  加载 SimFin 季度财报数据")
    logger.info("=" * 60)

    cache_files = {
        "income": cache_dir / "income_quarterly.parquet",
        "balance": cache_dir / "balance_quarterly.parquet",
        "cashflow": cache_dir / "cashflow_quarterly.parquet",
    }

    results = {}

    # 利润表
    if cache_files["income"].exists() and not force_refresh:
        logger.info("从缓存加载利润表...")
        results["income"] = pd.read_parquet(cache_files["income"])
    else:
        logger.info("下载季度利润表 (SimFin)...")
        income = sf.load_income(variant="quarterly", market="us")
        results["income"] = income
        income.to_parquet(cache_files["income"])
        logger.info(f"  利润表: {len(income)} 行")

    # 资产负债表
    if cache_files["balance"].exists() and not force_refresh:
        logger.info("从缓存加载资产负债表...")
        results["balance"] = pd.read_parquet(cache_files["balance"])
    else:
        logger.info("下载季度资产负债表 (SimFin)...")
        balance = sf.load_balance(variant="quarterly", market="us")
        results["balance"] = balance
        balance.to_parquet(cache_files["balance"])
        logger.info(f"  资产负债表: {len(balance)} 行")

    # 现金流量表
    if cache_files["cashflow"].exists() and not force_refresh:
        logger.info("从缓存加载现金流量表...")
        results["cashflow"] = pd.read_parquet(cache_files["cashflow"])
    else:
        logger.info("下载季度现金流量表 (SimFin)...")
        cashflow = sf.load_cashflow(variant="quarterly", market="us")
        results["cashflow"] = cashflow
        cashflow.to_parquet(cache_files["cashflow"])
        logger.info(f"  现金流量表: {len(cashflow)} 行")

    # 统计覆盖公司数
    for name, df in results.items():
        if SF_TICKER in df.index.names:
            n_companies = df.index.get_level_values(SF_TICKER).nunique()
        elif SF_TICKER in df.columns:
            n_companies = df[SF_TICKER].nunique()
        else:
            n_companies = df.index.nunique()
        logger.info(f"  {name}: {len(df)} 条记录, {n_companies} 家公司")

    logger.info("=" * 60)
    return results


def align_ticker_format(simfin_df: pd.DataFrame) -> pd.DataFrame:
    """
    统一 SimFin 的 ticker 格式与 yfinance 一致。

    SimFin 使用原始格式（BRK.B, BF.B），
    yfinance 使用连字符（BRK-B, BF-B）。
    处理 Ticker 在列或索引中的情况。
    """
    df = simfin_df.copy()

    if SF_TICKER in df.index.names:
        # Ticker 在多级索引中 → 重置、替换、重新设索引
        df = df.reset_index()
        df[SF_TICKER] = df[SF_TICKER].str.replace(".", "-", regex=False)
        # 不能直接 set_index 因为其他索引名可能需要保留
        # 但在 filter 中我们会单独处理
        return df

    if SF_TICKER in df.columns:
        df[SF_TICKER] = df[SF_TICKER].str.replace(".", "-", regex=False)

    return df


def build_ticker_map(
    price_tickers: list[str],
    simfin_data: dict[str, pd.DataFrame],
) -> set[str]:
    """
    找出同时存在于价格数据和 SimFin 基本面数据中的 ticker。

    Returns:
        set: 可用 ticker 集合
    """
    # 收集 SimFin 中的所有 ticker
    simfin_tickers = set()
    for name, df in simfin_data.items():
        df = align_ticker_format(df)
        if SF_TICKER in df.columns:
            simfin_tickers.update(df[SF_TICKER].dropna().unique())
        elif SF_TICKER in df.index.names:
            simfin_tickers.update(df.index.get_level_values(SF_TICKER).dropna().unique())

    price_set = set(price_tickers)
    intersection = price_set & simfin_tickers

    logger.info(f"价格数据: {len(price_set)} 只, SimFin: {len(simfin_tickers)} 只")
    logger.info(f"交集: {len(intersection)} 只 ({len(intersection)/len(price_set)*100:.1f}%)")

    return intersection


def filter_universe_by_data_completeness(
    tickers: set[str],
    simfin_data: dict[str, pd.DataFrame],
    min_quarters: int = MIN_QUARTERS,
    start_date: str = START_DATE,
) -> list[str]:
    """
    按数据完整度过滤股票池。

    条件：在 start_date 之前至少有 min_quarters 个季度的完整三张表
    （这些数据将用于计算回测开始时的基本面因子）。

    Args:
        tickers: 候选 ticker 集合
        simfin_data: SimFin 三张表
        min_quarters: 最少季度数
        start_date: 回测起始日期

    Returns:
        list: 过滤后的 ticker 列表
    """
    logger.info(f"按数据完整度过滤股票池 (最少 {min_quarters} 季度)...")

    cutoff = pd.Timestamp(start_date)

    # 对每张表，统计每个 ticker 在截止日期前的季度数
    ticker_quarter_counts = {}

    for name, df in simfin_data.items():
        idx_names = list(df.index.names)

        if SF_TICKER in idx_names:
            df_reset = df.reset_index()
            df_reset[SF_REPORT_DATE] = pd.to_datetime(df_reset[SF_REPORT_DATE])
            df_filtered = df_reset[df_reset[SF_REPORT_DATE] < cutoff]

            for ticker in tickers:
                n = len(df_filtered[df_filtered[SF_TICKER] == ticker])
                if ticker not in ticker_quarter_counts:
                    ticker_quarter_counts[ticker] = {}
                ticker_quarter_counts[ticker][name] = n
        else:
            logger.warning(f"{name}: 无法识别索引结构，跳过")
            continue

    # 过滤：三张表都必须满足最少季度数
    required_tables = {"income", "balance", "cashflow"}
    qualified = []

    for ticker in tickers:
        counts = ticker_quarter_counts.get(ticker, {})
        if set(counts.keys()) >= required_tables:
            min_count = min(counts[t] for t in required_tables)
            if min_count >= min_quarters:
                qualified.append(ticker)

    logger.info(f"满足条件的股票: {len(qualified)} / {len(tickers)}")
    if qualified:
        logger.info(f"  样本: {qualified[:10]}")
    logger.info(f"  ({len(tickers) - len(qualified)} 只被过滤掉)")

    return sorted(qualified)


# ── 周频降采样 ──


def resample_to_weekly(close_daily: pd.DataFrame) -> pd.DataFrame:
    """
    将日频收盘价降采样到周频（每周最后一个交易日）。

    Args:
        close_daily: index=Date, columns=ticker, 日频收盘价

    Returns:
        DataFrame: index=Date(每周五), columns=ticker, 周频收盘价
    """
    logger.info(f"日频降采样到周频: {close_daily.shape} → ...")

    # 按周重采样取最后一个交易日
    close_weekly = close_daily.resample("W-FRI").last()
    close_weekly = close_weekly.dropna(how="all")

    logger.info(f"  → {close_weekly.shape} (约 {close_weekly.shape[0] / 52:.1f} 年)")
    return close_weekly


# ── 数据加载主入口 ──


def load_all_data(
    force_refresh_prices: bool = False,
    force_refresh_simfin: bool = False,
) -> dict:
    """
    数据加载主入口。编排完整的数据获取流程。

    Returns:
        dict: {
            "close_weekly": 周频收盘价 DataFrame,
            "close_daily": 日频收盘价 DataFrame (用于回测引擎),
            "simfin": SimFin 三张表 dict,
            "universe": 最终股票池 list,
        }
    """
    logger.info("=" * 60)
    logger.info("  v6 数据加载")
    logger.info(f"  回测区间: {START_DATE} ~ {END_DATE}")
    logger.info(f"  股票池: {UNIVERSE}")
    logger.info("=" * 60)

    # Step 1: 获取 Russell 3000 列表
    logger.info("\n[Step 1/5] 获取 Russell 3000 成分股列表...")
    russell_tickers = get_russell3000_tickers()

    # Step 2: 加载 SimFin 基本面数据
    logger.info("\n[Step 2/5] 加载 SimFin 季度财报...")
    simfin_data = load_simfin_data(force_refresh=force_refresh_simfin)

    # Step 3: Ticker 对齐 + 股票池过滤
    logger.info("\n[Step 3/5] Ticker 对齐 + 股票池过滤...")
    price_cache = DATA_DIR / "russell3000_close.parquet"

    # 先做 ticker 对齐（用 SimFin 数据判断交集）
    # 价格数据中可能有些 ticker 不在 SimFin 中，所以先下载再过滤
    ticker_intersection = build_ticker_map(russell_tickers, simfin_data)

    qualified_tickers = filter_universe_by_data_completeness(
        ticker_intersection, simfin_data, min_quarters=MIN_QUARTERS, start_date=START_DATE
    )

    if len(qualified_tickers) < 100:
        logger.warning(
            f"过滤后只剩 {len(qualified_tickers)} 只股票，可能数据不足。"
            f"降低 min_quarters 到 4 重试..."
        )
        qualified_tickers = filter_universe_by_data_completeness(
            ticker_intersection, simfin_data, min_quarters=4, start_date=START_DATE
        )

    # Step 4: 下载价格数据（只下载过滤后的 ticker，大幅减少下载量）
    logger.info(f"\n[Step 4/5] 下载价格数据 ({len(qualified_tickers)} 只股票)...")
    close_daily = download_prices(
        tickers=qualified_tickers,
        start_date=START_DATE,
        end_date=END_DATE,
        cache_path=price_cache,
        force_refresh=force_refresh_prices,
    )

    # Step 5: 周频降采样
    logger.info("\n[Step 5/5] 日频 → 周频降采样...")
    close_weekly = resample_to_weekly(close_daily)

    # 确保最终的股票池是价格数据和基本面数据的交集
    final_universe = sorted(set(qualified_tickers) & set(close_weekly.columns))
    close_weekly = close_weekly[final_universe]
    close_daily = close_daily[
        [t for t in final_universe if t in close_daily.columns] + [BENCHMARK_TICKER]
    ]
    # 确保 benchmark 也在
    if BENCHMARK_TICKER not in close_daily.columns:
        logger.warning(f"Benchmark {BENCHMARK_TICKER} 不在数据中，单独下载...")
        bm_close = download_prices(
            tickers=[],
            start_date=START_DATE,
            end_date=END_DATE,
            benchmark_ticker=BENCHMARK_TICKER,
            cache_path=DATA_DIR / "benchmark_close.parquet",
        )
        close_daily[BENCHMARK_TICKER] = bm_close[BENCHMARK_TICKER]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  数据加载完成")
    logger.info(f"  股票池: {len(final_universe)} 只")
    logger.info(f"  周频: {close_weekly.shape[0]} 周 × {close_weekly.shape[1]} 股")
    logger.info(f"  日频: {close_daily.shape[0]} 天 × {close_daily.shape[1]} 股")
    logger.info(f"{'=' * 60}")

    return {
        "close_weekly": close_weekly,
        "close_daily": close_daily,
        "simfin": simfin_data,
        "universe": final_universe,
    }

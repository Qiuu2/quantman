"""
data_loader.py - 数据获取模块

获取标普500成分股列表和历史日线OHLCV数据。
支持本地缓存，避免重复下载。

注意：yfinance 新版 MultiIndex 为 (ticker, field)，
      本模块统一转换为 (field, ticker) 并小写，与旧逻辑兼容。
"""

import json
import io
import logging
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def load_config(config_path: str = "config.json") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sp500_tickers() -> list[str]:
    """
    从 Wikipedia 获取当前标普500成分股代码列表。
    使用 requests 模拟浏览器 User-Agent，避免 403。
    """
    logger.info("获取标普500成分股列表...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].str.strip().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    logger.info(f"获取到 {len(tickers)} 只成分股")
    return tickers


def download_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    benchmark_ticker: str = "SPY",
) -> dict[str, pd.DataFrame]:
    """
    批量下载股票日线数据，返回以 field 为 key 的字典。

    Returns:
        dict: {"close": DataFrame, "volume": DataFrame, ...}
              每个 DataFrame 的 index=Date, columns=ticker
    """
    all_tickers = list(set(tickers + [benchmark_ticker]))
    logger.info(f"下载 {len(all_tickers)} 只股票数据 ({start_date} ~ {end_date})...")

    raw = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=True,
        progress=True,
    )

    # yfinance 新版: MultiIndex (ticker, field)
    # 转换为 {field: DataFrame(index=date, columns=ticker)}
    fields = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for field in raw.columns.get_level_values(1).unique():
            field_lower = field.lower()
            df_field = raw.xs(field, axis=1, level=1)
            df_field.index = pd.to_datetime(df_field.index)
            df_field = df_field.dropna(axis=1, how="all")
            fields[field_lower] = df_field
    else:
        # 只有一只股票的边界情况
        raw.index = pd.to_datetime(raw.index)
        for col in raw.columns:
            fields[col.lower()] = raw[[col]].rename(columns={col: all_tickers[0]})

    n_tickers = len(fields.get("close", pd.DataFrame()).columns)
    logger.info(f"下载完成: {fields['close'].shape[0]} 个交易日, {n_tickers} 只股票")
    return fields


def load_or_download(
    config_path: str = "config.json",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    加载缓存数据或下载新数据。

    Returns:
        dict: {"close": DataFrame, "volume": DataFrame, ...}
    """
    config = load_config(config_path)
    cache_dir = Path(config["data"]["data_dir"])
    close_cache = cache_dir / "close.parquet"
    volume_cache = cache_dir / "volume.parquet"

    if close_cache.exists() and not force_refresh:
        logger.info(f"从缓存加载数据: {cache_dir}")
        close = pd.read_parquet(close_cache)
        volume = pd.read_parquet(volume_cache) if volume_cache.exists() else None
        logger.info(f"缓存数据: {close.shape[0]} 个交易日, {close.shape[1]} 只股票")
        return {"close": close, "volume": volume}

    # 下载
    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers = get_sp500_tickers()
    fields = download_price_data(
        tickers=tickers,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        benchmark_ticker=config["data"]["benchmark_ticker"],
    )

    # 保存缓存
    fields["close"].to_parquet(close_cache)
    if "volume" in fields:
        fields["volume"].to_parquet(volume_cache)
    logger.info(f"数据已缓存到: {cache_dir}")
    return fields


def get_close_prices(fields: dict) -> pd.DataFrame:
    """提取收盘价 DataFrame: index=Date, columns=ticker"""
    return fields["close"].copy()


def get_benchmark_returns(fields: dict, benchmark: str = "SPY") -> pd.Series:
    """计算基准日收益率"""
    close = fields["close"]
    if benchmark not in close.columns:
        raise KeyError(f"Benchmark {benchmark} 不在数据中")
    return close[benchmark].pct_change()

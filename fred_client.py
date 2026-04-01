"""
fred_client.py - FRED (Federal Reserve Economic Data) API 客户端

免费版：
- 120,000 次/天（非常充裕）
- 限制：每次请求最多返回 1000 个数据点（月度数据够用 80+ 年）

数据来源：https://fred.stlouisfed.org/
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data")

# 常用宏观指标 series ID
FRED_SERIES = {
    # 利率
    "FEDFUNDS": "联邦基金有效利率",
    "DGS10": "10 年期国债收益率",
    "DGS2": "2 年期国债收益率",
    "DGS3MO": "3 月期国债收益率",
    "T10Y2Y": "10Y-2Y 国债利差（收益率曲线）",
    # 通胀
    "CPIAUCSL": "CPI 消费者物价指数",
    "CPILFESL": "核心 CPI（除食品和能源）",
    "PCEPI": "PCE 物价指数",
    # 就业
    "UNRATE": "失业率",
    "PAYEMS": "非农就业人数",
    # 经济景气
    "MANEMP": "制造业就业人数",
    "INDPRO": "工业生产指数",
    # 货币
    "M2SL": "M2 货币供应量",
    # PMI
    "MANEMP": "制造业就业",
    "UMCSENT": "密歇根大学消费者信心指数",
}

# 量化因子常用的宏观变量
QUANT_MACRO_SERIES = {
    "T10Y2Y": "term_spread",       # 收益率曲线斜率（衰退预测指标）
    "FEDFUNDS": "fed_rate",        # 联邦基金利率（货币政策松紧）
    "DGS10": "risk_free_rate",     # 无风险利率
    "CPIAUCSL": "inflation",       # 通胀水平
    "UNRATE": "unemployment",      # 失业率
    "M2SL": "money_supply",        # 货币供应
    "UMCSENT": "consumer_sentiment",  # 消费者信心
    "INDPRO": "industrial_production",  # 工业生产
}


class FredClient:
    """FRED API 客户端"""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._last_request_time = 0

    def _rate_limit(self):
        """限速：FRED 额度非常充裕，但还是做个简单限速"""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)
        self._last_request_time = time.time()

    def _get(self, params: dict) -> dict:
        self._rate_limit()
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        resp = requests.get(
            f"{self.BASE_URL}/series/observations",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_series(
        self,
        series_id: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.Series:
        """
        获取单个宏观指标时间序列。

        Args:
            series_id: FRED series ID（如 "DGS10", "UNRATE"）
            start_date: "2021-01-01"
            end_date: "2026-01-01"

        Returns:
            pd.Series: index=Date, values=指标值
        """
        params = {
            "series_id": series_id,
            "observation_start": start_date or "2021-01-01",
            "observation_end": end_date or "2026-01-01",
        }

        data = self._get(params)
        observations = data.get("observations", [])

        if not observations:
            logger.warning(f"获取 {series_id} 无数据")
            return pd.Series(dtype=float)

        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        series = df.set_index("date")["value"]
        series.name = series_id

        return series

    def get_multiple_series(
        self,
        series_map: dict = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        批量获取多个宏观指标。

        Args:
            series_map: {series_id: column_name}
                       默认使用 QUANT_MACRO_SERIES
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: index=Date, columns=column_name
        """
        if series_map is None:
            series_map = QUANT_MACRO_SERIES

        cache_file = CACHE_DIR / "fred_macro.parquet"
        if cache_file.exists():
            logger.info(f"从缓存加载宏观数据: {cache_file}")
            cached = pd.read_parquet(cache_file)
            # 检查缓存是否包含所有需要的 series
            if all(name in cached.columns for name in series_map.values()):
                return cached

        logger.info(f"从 FRED 获取 {len(series_map)} 个宏观指标...")

        result = {}
        for series_id, col_name in series_map.items():
            desc = FRED_SERIES.get(series_id, "")
            logger.info(f"  {series_id}: {desc} ({col_name})")
            try:
                series = self.get_series(series_id, start_date, end_date)
                result[col_name] = series
            except requests.exceptions.HTTPError as e:
                logger.warning(f"  {series_id}: 请求失败 ({e.response.status_code}), 跳过")
            except Exception as e:
                logger.warning(f"  {series_id}: 获取失败 ({e}), 跳过")

        df = pd.DataFrame(result)

        # 前向填充（宏观数据多为月度/季度，需要填充到日频）
        df = df.asfreq("D").ffill()
        df = df.dropna(how="all")

        # 保存缓存
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        logger.info(f"宏观数据已缓存: {df.shape[0]} 天, {df.shape[1]} 个指标")

        return df

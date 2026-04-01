"""
finnhub_client.py - Finnhub API 客户端

免费版限制：
- 60 次/分钟
- 美国市场数据
- 基本面：30 年历史财务报表
- 分析师：评级、目标价、EPS 超预期
- 新闻：1 年历史 + 实时

数据来源：https://finnhub.io/
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# SEC XBRL taxonomy -> our field mapping (simplified)
FINANCIAL_FIELDS = {
    "revenue": ["Revenue", "Sales", "Net Sales", "Total Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Operating Revenue"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "total_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "eps": ["Basic EPS", "Earnings Per Share", "Diluted EPS"],
    "total_debt": ["Total Debt", "Long Term Debt"],
    "cash": ["Cash And Cash Equivalents"],
    "shares_outstanding": ["Ordinary Shares Number"],
}

CACHE_DIR = Path("data")


class FinnhubClient:
    """Finnhub API 客户端（带缓存和限速）"""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.params["token"] = api_key
        self._last_request_time = 0
        self._min_interval = 1.1  # ~55 req/min (留余量)

    def _rate_limit(self):
        """限速：确保不超过 60 次/分钟"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """通用 GET 请求"""
        self._rate_limit()
        url = f"{self.BASE_URL}{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # =========================================================================
    # 基本面：财务报表（历史时间序列）
    # =========================================================================

    def get_income_statement(self, ticker: str, freq: str = "annual") -> list[dict]:
        """
        获取利润表。

        Args:
            ticker: 股票代码
            freq: "annual" (10-K) 或 "quarterly" (10-Q)

        Returns:
            list of dict, 每个元素是一个报告期
        """
        data = self._get(f"/stock/fundamental/{freq}", params={"symbol": ticker})
        return data.get("financials", [])

    def get_balance_sheet(self, ticker: str, freq: str = "annual") -> list[dict]:
        """获取资产负债表"""
        data = self._get(f"/stock/fundamental/{freq}-bs", params={"symbol": ticker})
        return data.get("financials", [])

    def get_cash_flow(self, ticker: str, freq: str = "annual") -> list[dict]:
        """获取现金流量表"""
        data = self._get(f"/stock/fundamental/{freq}-cf", params={"symbol": ticker})
        return data.get("financials", [])

    def get_basic_financials(self, ticker: str) -> dict:
        """获取公司基本财务指标（当前快照）"""
        return self._get("/stock/metric", params={
            "symbol": ticker,
            "metric": "all",
        })

    def get_company_profile(self, ticker: str) -> dict:
        """获取公司概况（行业、市值等）"""
        return self._get("/stock/profile2", params={"symbol": ticker})

    # =========================================================================
    # 分析师数据
    # =========================================================================

    def get_analyst_recommendations(self, ticker: str) -> list[dict]:
        """
        获取分析师评级趋势（买入/持有/卖出 人数）。

        Returns:
            list of dict: [{"period": "1m", "buy": 10, "hold": 5, "sell": 1, ...}, ...]
        """
        data = self._get("/stock/recommendation", params={"symbol": ticker})
        return data if isinstance(data, list) else []

    def get_price_target(self, ticker: str) -> dict:
        """
        获取分析师目标价共识。

        Returns:
            dict: {"targetHigh": 200, "targetLow": 150, "targetMean": 175,
                   "targetMedian": 170, "currentPrice": 160}
        """
        return self._get("/stock/price-target", params={"symbol": ticker})

    def get_earnings_surprises(self, ticker: str) -> list[dict]:
        """
        获取 EPS 超预期历史（最近 4 个季度）。

        Returns:
            list of dict: [{"date": "2024-01-01", "actual": 1.5,
                           "estimate": 1.3, "surprise": 0.2,
                           "surprisePercent": 15.38}, ...]
        """
        data = self._get("/stock/earnings", params={"symbol": ticker})
        return data if isinstance(data, list) else []

    def get_earnings_calendar(
        self, from_date: str, to_date: str
    ) -> list[dict]:
        """
        获取财报日历（所有股票）。

        Args:
            from_date: "2024-01-01"
            to_date: "2024-12-31"

        Returns:
            list of dict: [{"symbol": "AAPL", "date": "2024-01-25", ...}, ...]
        """
        data = self._get("/calendar/earnings", params={
            "from": from_date,
            "to": to_date,
        })
        return data.get("earningsCalendar", [])

    # =========================================================================
    # 新闻
    # =========================================================================

    def get_company_news(
        self, ticker: str, from_date: str, to_date: str
    ) -> list[dict]:
        """
        获取公司新闻（免费版限 1 年历史）。

        Returns:
            list of dict: [{"datetime": 1704067200, "headline": "...",
                           "source": "Reuters", "sentiment": 0.5, ...}, ...]
        """
        data = self._get("/company-news", params={
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
        })
        return data if isinstance(data, list) else []

    # =========================================================================
    # 批量获取（带缓存）
    # =========================================================================

    def fetch_all_fundamentals(
        self,
        tickers: list[str],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        批量获取所有股票的年度基本面历史时间序列（使用 stock/metric 端点）。

        免费版 quarter series 为空（付费功能），
        但 series.annual 包含 26-41 年历史年度数据，格式：
          [{"period": "2025-09-27", "v": 0.4691}, ...]

        同时提取 metric 中的当前快照字段补充。

        Returns:
            DataFrame: columns=[ticker, date, field, value]
        """
        cache_file = CACHE_DIR / "finnhub_fundamentals_annual.parquet"

        if cache_file.exists() and not force_refresh:
            df = pd.read_parquet(cache_file)
            if not df.empty:
                logger.info(f"从缓存加载基本面数据: {cache_file} ({len(df)} records)")
                return df

        logger.info(f"从 Finnhub stock/metric 批量获取 {len(tickers)} 只股票的年度基本面...")
        logger.info(f"预计耗时: {len(tickers) * 1.5 / 60:.0f} 分钟（限速 60 req/min）")

        # Annual series fields we want
        annual_fields = [
            "grossMargin", "operatingMargin", "netMargin",
            "pb", "ps", "totalDebtToEquity", "currentRatio",
            "bookValue", "eps", "roaTTM",
        ]

        # Metric snapshot fields with quarterly context
        metric_snapshot_fields = [
            "roeTTM", "roaTTM", "grossMarginTTM", "operatingMarginTTM",
            "netMarginTTM", "peTTM", "pbQ", "psTTM",
            "totalDebt/EquityQuarterly", "currentRatioQuarterly",
            "epsGrowthQuarterlyYoy", "revenueGrowthQuarterlyYoy",
            "bookValuePerShareQuarterly", "epsTTM",
        ]

        records = []
        failed = []

        for i, ticker in enumerate(tickers):
            try:
                metrics = self.get_basic_financials(ticker)

                # 1. Annual series (historical time series)
                series = metrics.get("series", {})
                annual = series.get("annual", {})

                for field_name in annual_fields:
                    items = annual.get(field_name, [])
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        period = item.get("period")
                        value = item.get("v")
                        if period and value is not None:
                            try:
                                records.append({
                                    "ticker": ticker,
                                    "date": period,
                                    "field": field_name,
                                    "value": float(value),
                                })
                            except (ValueError, TypeError):
                                pass

                # 2. Metric snapshot (current point-in-time)
                metric = metrics.get("metric", {})
                for field_name in metric_snapshot_fields:
                    value = metric.get(field_name)
                    if value is not None:
                        try:
                            records.append({
                                "ticker": ticker,
                                "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                                "field": field_name,
                                "value": float(value),
                            })
                        except (ValueError, TypeError):
                            pass

            except Exception as e:
                failed.append(ticker)
                logger.debug(f"获取 {ticker} 基本面失败: {e}")

            if (i + 1) % 50 == 0:
                logger.info(f"  进度: {i + 1}/{len(tickers)}")

        df = pd.DataFrame(records)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])

        if failed:
            logger.warning(
                f"{len(failed)} 只股票获取失败: {failed[:5]}"
                f"{'...' if len(failed) > 5 else ''}"
            )

        # 保存缓存
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        n_records = len(df) if not df.empty else 0
        n_tickers = df["ticker"].nunique() if not df.empty else 0
        logger.info(f"基本面数据已缓存: {n_records} 条记录, {n_tickers} 只股票, 保存至 {cache_file}")

        return df

    def fetch_all_eps_surprises(
        self,
        tickers: list[str],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        批量获取 EPS 超预期数据。

        Returns:
            DataFrame: columns=[ticker, date, actual, estimate, surprise, surprise_pct]
        """
        cache_file = CACHE_DIR / "finnhub_eps_surprises.parquet"

        if cache_file.exists() and not force_refresh:
            logger.info(f"从缓存加载 EPS 超预期数据: {cache_file}")
            return pd.read_parquet(cache_file)

        logger.info(f"批量获取 {len(tickers)} 只股票的 EPS 超预期数据...")

        records = []
        failed = []

        for ticker in tickers:
            try:
                surprises = self.get_earnings_surprises(ticker)
                for s in surprises:
                    # Finnhub returns date as "period" field
                    record = {
                        "ticker": ticker,
                        "date": s.get("period") or s.get("date"),
                        "actual": s.get("actual"),
                        "estimate": s.get("estimate"),
                        "surprise": s.get("surprise"),
                        "surprise_pct": s.get("surprisePercent"),
                    }
                    if record["date"] is not None:
                        records.append(record)
            except Exception as e:
                failed.append(ticker)
                logger.debug(f"获取 {ticker} EPS 失败: {e}")

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        if failed:
            logger.warning(f"{len(failed)} 只股票 EPS 获取失败")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        logger.info(f"EPS 数据已缓存: {len(df)} 条记录")

        return df

    def fetch_all_analyst_ratings(
        self,
        tickers: list[str],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        批量获取分析师评级趋势。

        Returns:
            DataFrame: columns=[ticker, period, buy, hold, sell, strongBuy, strongSell]
        """
        cache_file = CACHE_DIR / "finnhub_analyst_ratings.parquet"

        if cache_file.exists() and not force_refresh:
            logger.info(f"从缓存加载分析师评级: {cache_file}")
            return pd.read_parquet(cache_file)

        logger.info(f"批量获取 {len(tickers)} 只股票的分析师评级...")

        records = []
        failed = []

        for ticker in tickers:
            try:
                ratings = self.get_analyst_recommendations(ticker)
                for r in ratings:
                    records.append({
                        "ticker": ticker,
                        "period": r.get("period"),
                        "buy": r.get("buy", 0),
                        "hold": r.get("hold", 0),
                        "sell": r.get("sell", 0),
                        "strongBuy": r.get("strongBuy", 0),
                        "strongSell": r.get("strongSell", 0),
                    })
            except Exception as e:
                failed.append(ticker)
                logger.debug(f"获取 {ticker} 评级失败: {e}")

        df = pd.DataFrame(records)

        if failed:
            logger.warning(f"{len(failed)} 只股票评级获取失败")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        logger.info(f"分析师评级已缓存: {len(df)} 条记录")

        return df

    def fetch_all_price_targets(
        self,
        tickers: list[str],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        批量获取分析师目标价。

        Returns:
            DataFrame: columns=[ticker, targetHigh, targetLow, targetMean,
                               targetMedian, currentPrice]
        """
        cache_file = CACHE_DIR / "finnhub_price_targets.parquet"

        if cache_file.exists() and not force_refresh:
            logger.info(f"从缓存加载目标价: {cache_file}")
            return pd.read_parquet(cache_file)

        logger.info(f"批量获取 {len(tickers)} 只股票的目标价...")

        records = []
        failed = []

        for ticker in tickers:
            try:
                pt = self.get_price_target(ticker)
                records.append({
                    "ticker": ticker,
                    "targetHigh": pt.get("targetHigh"),
                    "targetLow": pt.get("targetLow"),
                    "targetMean": pt.get("targetMean"),
                    "targetMedian": pt.get("targetMedian"),
                    "currentPrice": pt.get("currentPrice"),
                })
            except Exception as e:
                failed.append(ticker)
                logger.debug(f"获取 {ticker} 目标价失败: {e}")

        if not records:
            logger.warning("未获取到任何目标价数据")
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("ticker")

        if failed:
            logger.warning(f"{len(failed)} 只股票目标价获取失败")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        logger.info(f"目标价已缓存: {len(df)} 只股票")

        return df

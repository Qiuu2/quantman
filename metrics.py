"""
metrics.py - 绩效评估模块

计算量化策略的核心绩效指标：
年化收益率、年化波动率、夏普比率、最大回撤、
胜率、信息比率、Calmar 比率等。
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化收益率"""
    total = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化波动率"""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    """夏普比率"""
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    vol = excess.std()
    if vol == 0:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def max_drawdown(cumulative: pd.Series) -> float:
    """最大回撤"""
    rolling_max = cumulative.cummax()
    dd = (cumulative - rolling_max) / rolling_max
    return float(dd.min())


def calmar_ratio(returns: pd.Series, cumulative: pd.Series) -> float:
    """Calmar 比率 = 年化收益 / 最大回撤"""
    mdd = max_drawdown(cumulative)
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns) / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """胜率（日度正收益占比）"""
    return float((returns > 0).sum() / len(returns.dropna()))


def monthly_win_rate(returns: pd.Series) -> float:
    """月度胜率"""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return float((monthly > 0).sum() / len(monthly.dropna()))


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    信息比率 = 超额收益均值 / 超额收益标准差

    Args:
        strategy_returns: 策略日收益
        benchmark_returns: 基准日收益

    Returns:
        float: 信息比率（年化）
    """
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    excess = strategy_returns.loc[common_idx] - benchmark_returns.loc[common_idx]
    tracking_error = excess.std()
    if tracking_error == 0:
        return 0.0
    return float(excess.mean() / tracking_error * np.sqrt(252))


def beta_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """
    计算 Alpha 和 Beta。

    Returns:
        (alpha, beta): 年化 alpha 和 beta
    """
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    s = strategy_returns.loc[common_idx]
    b = benchmark_returns.loc[common_idx]

    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_s = s - rf_daily
    excess_b = b - rf_daily

    beta = float(excess_s.cov(excess_b) / excess_b.var())
    alpha = float(excess_s.mean() - beta * excess_b.mean()) * periods_per_year

    return alpha, beta


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    """Sortino 比率（只惩罚下行波动）"""
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def evaluate(
    returns: pd.Series,
    cumulative: pd.Series,
    risk_free_rate: float = 0.04,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """
    综合绩效评估。

    Args:
        returns: 策略日收益 Series
        cumulative: 累计收益 Series
        risk_free_rate: 无风险利率
        benchmark_returns: 基准日收益（可选）

    Returns:
        dict: 所有绩效指标
    """
    results = {
        "年化收益率": annualized_return(returns),
        "年化波动率": annualized_volatility(returns),
        "夏普比率": sharpe_ratio(returns, risk_free_rate),
        "Sortino 比率": sortino_ratio(returns, risk_free_rate),
        "最大回撤": max_drawdown(cumulative),
        "Calmar 比率": calmar_ratio(returns, cumulative),
        "日胜率": win_rate(returns),
        "月胜率": monthly_win_rate(returns),
    }

    if benchmark_returns is not None:
        results["信息比率 (vs Benchmark)"] = information_ratio(returns, benchmark_returns)
        alpha, beta = beta_alpha(returns, benchmark_returns, risk_free_rate)
        results["Alpha (年化)"] = alpha
        results["Beta"] = beta

        # 累计超额收益
        common_idx = returns.index.intersection(benchmark_returns.index)
        excess_cum = (1 + (returns.loc[common_idx] - benchmark_returns.loc[common_idx])).cumprod() - 1
        results["累计超额收益"] = float(excess_cum.iloc[-1]) if len(excess_cum) > 0 else 0.0

    return results


def print_report(
    strategy_returns: pd.Series,
    strategy_cumulative: pd.Series,
    risk_free_rate: float = 0.04,
    benchmark_returns: pd.Series | None = None,
    benchmark_cumulative: pd.Series | None = None,
    strategy_name: str = "Strategy",
    benchmark_name: str = "SPY",
) -> None:
    """
    打印绩效报告到控制台。

    Args:
        strategy_returns: 策略日收益
        strategy_cumulative: 策略累计收益
        risk_free_rate: 无风险利率
        benchmark_returns: 基准日收益
        benchmark_cumulative: 基准累计收益
        strategy_name: 策略名称
        benchmark_name: 基准名称
    """
    metrics = evaluate(strategy_returns, strategy_cumulative, risk_free_rate, benchmark_returns)

    print("\n" + "=" * 60)
    print(f"  {strategy_name} 绩效报告")
    print("=" * 60)

    print(f"\n  {'指标':<25} {'值':>15}")
    print(f"  {'-'*25} {'-'*15}")

    for key, val in metrics.items():
        if "率" in key or "Ratio" in key or "Alpha" in key or "Beta" in key:
            print(f"  {key:<25} {val:>14.4f}")
        elif "回撤" in key or "收益" in key:
            print(f"  {key:<25} {val:>13.2%}")
        else:
            print(f"  {key:<25} {val:>15.4f}")

    if benchmark_cumulative is not None:
        final_strat = strategy_cumulative.iloc[-1]
        final_bench = benchmark_cumulative.iloc[-1]
        print(f"\n  {'策略累计收益':<25} {final_strat:>13.2%}")
        print(f"  {benchmark_name + ' 累计收益':<25} {final_bench:>13.2%}")

    print("=" * 60 + "\n")

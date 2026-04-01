"""诊断 v5 因子面板质量"""
import pandas as pd
import numpy as np

# Load fundamentals
f = pd.read_parquet('data/finnhub_fundamentals_annual.parquet')
print('=== Fundamentals raw data ===')
print(f'Shape: {f.shape}')
print()

# Check annual vs snapshot
annual_fields = ['grossMargin', 'operatingMargin', 'netMargin', 'pb', 'ps', 
                 'totalDebtToEquity', 'currentRatio', 'bookValue', 'eps']
snapshot_fields = [col for col in f['field'].unique() if col not in annual_fields]

print('Annual series fields (time-series):')
for field in annual_fields:
    subset = f[f['field'] == field]
    n_dates = subset.groupby('ticker')['date'].nunique().mean()
    print(f'  {field}: {len(subset)} records, ~{n_dates:.0f} unique dates per ticker')

print()
print('Snapshot fields (single point only):')
for field in snapshot_fields:
    subset = f[f['field'] == field]
    n_dates = subset.groupby('ticker')['date'].nunique().mean()
    print(f'  {field}: {len(subset)} records, ~{n_dates:.0f} unique dates per ticker')

# Simulate what happens when we pivot to panel
print()
print('=== Simulated Panel (gross_margin) ===')
gm = f[f['field'] == 'grossMargin'].copy()
gm_pivot = gm.pivot_table(index='date', columns='ticker', values='value', aggfunc='last')
gm_pivot = gm_pivot.sort_index()
print(f'Panel shape: {gm_pivot.shape}')
print(f'Date range: {gm_pivot.index[0]} ~ {gm_pivot.index[-1]}')
print(f'Unique dates: {len(gm_pivot.index)}')
print(f'Average non-NaN per date: {gm_pivot.notna().sum(axis=1).mean():.0f} / {gm_pivot.shape[1]}')

# Simulate reindexing to daily trading dates
# Create fake daily dates for 2021-2025
daily_dates = pd.bdate_range('2021-01-01', '2026-01-01')
gm_daily = gm_pivot.reindex(daily_dates).ffill()
print()
print('=== After reindex to daily + ffill ===')
print(f'Shape: {gm_daily.shape}')
# Check how much the cross-sectional ranking changes day to day
rank = gm_daily.rank(axis=1, pct=True)
rank_diff = rank.diff().abs().mean().mean()
print(f'Average daily rank change: {rank_diff:.6f} (0 = no change)')
print(f'Average monthly rank change: {rank.diff(21).abs().mean().mean():.6f}')
print(f'Average yearly rank change: {rank.diff(252).abs().mean().mean():.6f}')

# Compare with momentum
from data_loader import load_or_download, get_close_prices
fields = load_or_download('config.json')
close = get_close_prices(fields)
stock_close = close.drop(columns=['SPY'], errors='ignore')
from factors import momentum_12_1
mom = momentum_12_1(stock_close)
mom_rank = mom.rank(axis=1, pct=True)
mom_rank_diff = mom_rank.diff().abs().mean().mean()
print()
print(f'Momentum avg daily rank change: {mom_rank_diff:.6f} (reference)')
print()
print('CONCLUSION: Annual fundamentals change rank by ~{:.6f}/day vs momentum ~{:.6f}/day'.format(
    rank_diff, mom_rank_diff))

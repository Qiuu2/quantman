import pandas as pd

f = pd.read_parquet('data/finnhub_fundamentals_annual.parquet')

# Check roe data specifically
roe = f[f['field'].isin(['roeTTM', 'roaTTM'])]
print('=== ROE data ===')
print('Shape:', roe.shape)
print('Date distribution:')
print(roe.groupby('field')['date'].describe())
print()

# Check annual series fields - these should have time series
annual_fields = ['grossMargin', 'operatingMargin', 'netMargin', 'pb', 'ps', 'totalDebtToEquity', 'currentRatio', 'bookValue', 'eps']
for field in annual_fields:
    subset = f[f['field'] == field]
    dates = subset.groupby('ticker')['date'].count()
    print(f'{field}: {len(subset)} records, avg per ticker: {dates.mean():.1f}, max date per ticker: {dates.max()}')

# Check a specific ticker to see data structure
print()
print('=== AAPL grossMargin sample ===')
aapl_gm = f[(f['ticker'] == 'AAPL') & (f['field'] == 'grossMargin')]
aapl_gm = aapl_gm.sort_values('date')
print(aapl_gm.tail(10).to_string())

# Check EPS surprises 
print()
print('=== EPS Surprises ===')
eps = pd.read_parquet('data/finnhub_eps_surprises.parquet')
print(eps.head(5).to_string())
print()
print('Date column dtype:', eps['date'].dtype)
print('Sample dates:', eps['date'].head(10).tolist())

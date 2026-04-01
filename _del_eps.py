import pandas as pd
from pathlib import Path

# Delete old EPS cache (dates were NaT)
f = Path('data/finnhub_eps_surprises.parquet')
if f.exists():
    f.unlink()
    print('Deleted old EPS cache')

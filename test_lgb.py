"""Test LightGBM LambdaRank with different group sizes."""
import lightgbm as lgb
import numpy as np

print(f"LightGBM version: {lgb.__version__}")

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'max_position': 200,
    'verbose': -1,
}

for gsize in [29, 30, 31, 32, 33, 64]:
    np.random.seed(gsize + 100)
    X = np.random.randn(gsize, 5)
    y = np.arange(gsize, dtype=np.float64)
    d = lgb.Dataset(X, label=y, group=[gsize])
    try:
        m = lgb.train(params, d, num_boost_round=5)
        print(f"  group={gsize:>3}: OK (max_label={int(y.max())})")
    except Exception as e:
        print(f"  group={gsize:>3}: FAIL (max_label={int(y.max())}): {e}")

print("\nDone.")

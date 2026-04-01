"""Test: LightGBM lambdarank - use 1-based labels."""
import lightgbm as lgb
import numpy as np

print(f"LightGBM version: {lgb.__version__}")

params = {'objective': 'lambdarank', 'metric': 'ndcg', 'max_position': 200, 'verbose': -1}

for gsize in [29, 30, 31, 32, 33, 64]:
    np.random.seed(gsize + 200)
    X = np.random.randn(gsize, 5)
    # Use raw returns as labels (not ranks)
    y = np.random.randn(gsize).astype(np.float64)
    d = lgb.Dataset(X, label=y, group=[gsize])
    try:
        m = lgb.train(params, d, num_boost_round=5)
        print(f"  group={gsize:>3}: OK (relevance-based)")
    except Exception as e:
        print(f"  group={gsize:>3}: FAIL: {e}")

print("\n--- Now test with 1-based ranks ---")
for gsize in [29, 30, 31, 32, 33, 64]:
    np.random.seed(gsize + 300)
    X = np.random.randn(gsize, 5)
    y = np.arange(1, gsize + 1, dtype=np.float64)  # 1-based
    d = lgb.Dataset(X, label=y, group=[gsize])
    try:
        m = lgb.train(params, d, num_boost_round=5)
        print(f"  group={gsize:>3}: OK (1-based ranks)")
    except Exception as e:
        print(f"  group={gsize:>3}: FAIL: {e}")

print("\nDone.")

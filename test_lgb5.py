"""Test: LightGBM lambdarank label_gain parameter."""
import lightgbm as lgb
import numpy as np

print(f"LightGBM version: {lgb.__version__}")

# Use label_gain to map relevance levels to gains
# This should bypass the 31-label limit
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'max_position': 200,
    'verbose': -1,
    'label_gain': list(range(201)),  # Explicit mapping: relevance 0 -> gain 0, 1 -> 1, ..., 200 -> 200
}

for gsize in [31, 32, 64, 128]:
    np.random.seed(gsize + 400)
    X = np.random.randn(gsize, 5)
    y = np.arange(gsize, dtype=np.float64)  # 0-based ranks
    d = lgb.Dataset(X, label=y, group=[gsize])
    try:
        m = lgb.train(params, d, num_boost_round=5)
        print(f"  group={gsize:>3}: OK (label_gain)")
    except Exception as e:
        print(f"  group={gsize:>3}: FAIL: {e}")

print("\n--- Now test int32 labels ---")
params2 = {'objective': 'lambdarank', 'metric': 'ndcg', 'max_position': 200, 'verbose': -1}
for gsize in [31, 32, 64, 128]:
    np.random.seed(gsize + 500)
    X = np.random.randn(gsize, 5)
    y = np.arange(gsize, dtype=np.int32)  # int32 instead of float64
    d = lgb.Dataset(X, label=y, group=[gsize])
    try:
        m = lgb.train(params2, d, num_boost_round=5)
        print(f"  group={gsize:>3}: OK (int32)")
    except Exception as e:
        print(f"  group={gsize:>3}: FAIL: {e}")

print("\n--- Test int labels with label_gain ---")
params3 = {'objective': 'lambdarank', 'metric': 'ndcg', 'max_position': 200, 'verbose': -1, 'label_gain': list(range(201))}
for gsize in [31, 32, 64, 128]:
    np.random.seed(gsize + 600)
    X = np.random.randn(gsize, 5)
    y = np.arange(gsize, dtype=np.int32)
    d = lgb.Dataset(X, label=y, group=[gsize])
    try:
        m = lgb.train(params3, d, num_boost_round=5)
        print(f"  group={gsize:>3}: OK (int32 + label_gain)")
    except Exception as e:
        print(f"  group={gsize:>3}: FAIL: {e}")

print("\nDone.")

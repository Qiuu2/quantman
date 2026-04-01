"""Test: LightGBM lambdarank label mapping bug - group 32 in isolation."""
import lightgbm as lgb
import numpy as np

print(f"LightGBM version: {lgb.__version__}")
np.random.seed(132)
params = {'objective': 'lambdarank', 'metric': 'ndcg', 'max_position': 200, 'verbose': -1}
X = np.random.randn(32, 5)
y = np.arange(32, dtype=np.float64)
d = lgb.Dataset(X, label=y, group=[32])
try:
    m = lgb.train(params, d, num_boost_round=5)
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')

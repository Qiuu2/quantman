"""Test LightGBM LambdaRank - isolate the bug with per-process execution."""
import lightgbm as lgb
import numpy as np
import subprocess
import sys

print(f"LightGBM version: {lgb.__version__}")

def test_group(gsize):
    """Test a single group size in isolation."""
    code = f"""
import lightgbm as lgb
import numpy as np
np.random.seed({gsize + 100})
params = {{'objective': 'lambdarank', 'metric': 'ndcg', 'max_position': 200, 'verbose': -1}}
X = np.random.randn({gsize}, 5)
y = np.arange({gsize}, dtype=np.float64)
d = lgb.Dataset(X, label=y, group=[{gsize}])
try:
    m = lgb.train(params, d, num_boost_round=5)
    print('OK')
except Exception as e:
    print(f'FAIL: {{e}}')
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True
    )
    # stderr from LightGBM warnings go to stderr, actual print goes to stdout
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if 'OK' in stdout:
        return 'OK'
    else:
        # Extract the actual error from stderr
        for line in stderr.split('\n'):
            if 'Label' in line and 'not less' in line:
                return f'FAIL: {line}'
        return f'FAIL: {stdout} | {stderr[:200]}'

for gsize in [29, 30, 31, 32, 33, 64, 128]:
    result = test_group(gsize)
    print(f"  group={gsize:>3}: {result}")

print("\nDone.")

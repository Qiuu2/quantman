"""Test lambdarank with equal group sizes and label_gain."""
import numpy as np
import lightgbm as lgb

np.random.seed(42)

# Test 1: All groups same size
print("=== Test 1: Equal group sizes ===")
group_sizes = [40, 40, 40]
X_parts = []
y_parts = []

for gsize in group_sizes:
    X = np.random.randn(gsize, 5)
    raw = np.random.randn(gsize)
    sort_order = np.argsort(raw)
    reranked = np.empty(gsize, dtype=np.float64)
    reranked[sort_order] = np.arange(gsize, dtype=np.float64)
    X_parts.append(X)
    y_parts.append(reranked)
    print(f"Group size={gsize}, label range=[{reranked.min():.0f}, {reranked.max():.0f}]")

X_train = np.concatenate(X_parts, axis=0)
y_train = np.concatenate(y_parts, axis=0)

train_data = lgb.Dataset(X_train, label=y_train, group=group_sizes)
params = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "learning_rate": 0.05, "verbose": -1, "seed": 42,
    "eval_at": [20, 50, 100],
}
model = lgb.train(train_set=train_data, params=params, num_boost_round=10)
print("✅ Equal groups succeeded!\n")

# Test 2: Different group sizes (must use label_gain to handle variable max label)
print("=== Test 2: Different group sizes with label_gain ===")
group_sizes = [31, 40, 60, 75, 80]
max_group = max(group_sizes)
label_gain = list(range(max_group, 0, -1))  # [80, 79, ..., 1]

X_parts = []
y_parts = []
for gsize in group_sizes:
    X = np.random.randn(gsize, 5)
    raw = np.random.randn(gsize)
    sort_order = np.argsort(raw)
    reranked = np.empty(gsize, dtype=np.float64)
    reranked[sort_order] = np.arange(gsize, dtype=np.float64)
    X_parts.append(X)
    y_parts.append(reranked)
    print(f"Group size={gsize}, label range=[{reranked.min():.0f}, {reranked.max():.0f}]")

X_train = np.concatenate(X_parts, axis=0)
y_train = np.concatenate(y_parts, axis=0)

train_data = lgb.Dataset(X_train, label=y_train, group=group_sizes)
params2 = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "learning_rate": 0.05, "verbose": -1, "seed": 42,
    "eval_at": [20, 50, 100],
    "label_gain": label_gain,
}
model = lgb.train(train_set=train_data, params=params2, num_boost_round=10)
print("✅ Different group sizes with label_gain succeeded!\n")

# Test 3: Different group sizes WITHOUT label_gain (should fail)
print("=== Test 3: Different group sizes WITHOUT label_gain ===")
try:
    model = lgb.train(train_set=train_data, params=params, num_boost_round=10)
    print("✅ Unexpectedly succeeded!")
except lgb.basic.LightGBMError as e:
    print(f"❌ Failed as expected: {e}")

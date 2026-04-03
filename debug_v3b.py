"""检查 v3b 因子与同期/未来收益的相关性"""
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

from v6.data_loader import load_all_data
from v6.features import build_all_features

# 加载数据
data = load_all_data()
close_weekly = data['close_weekly']

# 构建特征
features_3d, feature_names = build_all_features(
    close_weekly=close_weekly,
    simfin_data=data['simfin'],
    universe=data['universe'],
    close_daily_for_pb=data['close_daily'],
)

# 计算同期收益（T-1 到 T）
contemp_returns = close_weekly.pct_change().values  # (n_weeks, n_stocks)

# 计算未来收益（T 到 T+3）
future_returns = (close_weekly.shift(-3) / close_weekly - 1).values

print("v3b 因子 (momentum_12_1 + roe) 与收益的相关性:")
print()

mom_idx = feature_names.index('momentum_12_1')
roe_idx = feature_names.index('roe')

mom_flat = features_3d[:, :, mom_idx].flatten()
roe_flat = features_3d[:, :, roe_idx].flatten()
contemp_flat = contemp_returns.flatten()
future_flat = future_returns.flatten()

# momentum_12_1 与同期收益
valid1 = ~np.isnan(mom_flat) & ~np.isnan(contemp_flat)
corr1 = np.corrcoef(mom_flat[valid1], contemp_flat[valid1])[0, 1]
print(f"momentum_12_1 vs 同期收益 (T-1→T): {corr1:.4f}")

# momentum_12_1 与未来收益
valid2 = ~np.isnan(mom_flat) & ~np.isnan(future_flat)
corr2 = np.corrcoef(mom_flat[valid2], future_flat[valid2])[0, 1]
print(f"momentum_12_1 vs 未来收益 (T→T+3): {corr2:.4f}")

# roe 与同期收益
valid3 = ~np.isnan(roe_flat) & ~np.isnan(contemp_flat)
corr3 = np.corrcoef(roe_flat[valid3], contemp_flat[valid3])[0, 1]
print(f"roe vs 同期收益 (T-1→T): {corr3:.4f}")

# roe 与未来收益
valid4 = ~np.isnan(roe_flat) & ~np.isnan(future_flat)
corr4 = np.corrcoef(roe_flat[valid4], future_flat[valid4])[0, 1]
print(f"roe vs 未来收益 (T→T+3): {corr4:.4f}")

print()
print("关键发现：")
print("- 动量因子与同期收益的相关性应该很高（动量=过去涨得好的）")
print("- 但动量与未来收益的相关性应该很低（反转效应或随机）")
print("- 这说明 v3b 可能在利用同期相关性，而非预测能力")

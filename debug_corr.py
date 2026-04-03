"""检查特征与未来收益的相关性"""
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

from v6.data_loader import load_all_data
from v6.features import build_all_features, build_labels_weekly
from v6.config import FORWARD_WEEKS

# 加载数据
data = load_all_data()
close_weekly = data['close_weekly']

# 构建特征和标签
features_3d, feature_names = build_all_features(
    close_weekly=close_weekly,
    simfin_data=data['simfin'],
    universe=data['universe'],
    close_daily_for_pb=data['close_daily'],
)
labels_2d = build_labels_weekly(close_weekly, forward_weeks=FORWARD_WEEKS)

print(f"features_3d shape: {features_3d.shape}")
print(f"labels_2d shape: {labels_2d.shape}")
print(f"feature_names: {feature_names}")

# 检查 momentum_12_1 与 labels 的相关性
mom_idx = feature_names.index('momentum_12_1')
mom_flat = features_3d[:, :, mom_idx].flatten()
labels_flat = labels_2d.flatten()

# 只保留有效值
valid = ~np.isnan(mom_flat) & ~np.isnan(labels_flat)
corr = np.corrcoef(mom_flat[valid], labels_flat[valid])[0, 1]
print(f'\nmomentum_12_1 vs future returns correlation: {corr:.4f}')

# 检查 roe 与 labels 的相关性
roe_idx = feature_names.index('roe')
roe_flat = features_3d[:, :, roe_idx].flatten()
valid2 = ~np.isnan(roe_flat) & ~np.isnan(labels_flat)
corr2 = np.corrcoef(roe_flat[valid2], labels_flat[valid2])[0, 1]
print(f'roe vs future returns correlation: {corr2:.4f}')

# 检查更多因子
print('\n所有因子与未来收益的相关性:')
for i, name in enumerate(feature_names):
    feat_flat = features_3d[:, :, i].flatten()
    valid_mask = ~np.isnan(feat_flat) & ~np.isnan(labels_flat)
    if valid_mask.sum() > 100:
        c = np.corrcoef(feat_flat[valid_mask], labels_flat[valid_mask])[0, 1]
        print(f'  {name:25s}: {c:+.4f}')

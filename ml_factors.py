"""
ml_factors.py - v4 LightGBM 因子合成模块

用机器学习代替人工线性加权来组合因子。

核心思路：
  1. 构建因子特征矩阵（动量、质量 + 技术面衍生因子）
  2. 标签 = 未来 N 日收益率的截面排名（谁涨得多谁好）
  3. Rolling 训练：用过去 2 年数据训练，预测下个月
  4. LightGBM 输出每只股票的"预期收益排名" → 做多前 20%，做空后 20%

相比 v3 的人工线性加权：
  - 能自动学习因子之间的非线性交互
  - 能处理市场环境切换（熊市 vs 牛市因子重要性不同）
  - 自动特征选择（没用的因子权重趋近 0）
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


# ============================================================================
# 特征工程：构建因子特征矩阵
# ============================================================================


def build_feature_matrix(
    close: pd.DataFrame,
    fundamentals: Optional[pd.DataFrame] = None,
    exclude_negative_ic: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    构建因子特征矩阵（向量化版本）。

    返回一个 3D numpy array: shape = (n_dates, n_stocks, n_features)
    以及对应的 feature_names。

    Args:
        close: 收盘价 DataFrame (index=Date, columns=Ticker)
        fundamentals: 基本面数据 (index=Ticker)，可选
        exclude_negative_ic: 是否排除在 2021-2026 区间 IC 为负的因子
    """
    logger.info("=" * 60)
    logger.info("  构建因子特征矩阵（向量化）")
    logger.info("=" * 60)

    dates = close.index
    tickers = close.columns.tolist()
    n_dates = len(dates)
    n_stocks = len(tickers)

    daily_ret = close.pct_change()
    features = []

    # --- 特征 1: 12-1 动量 --- IC=+0.029 ✅
    f = close.shift(21) / close.shift(252) - 1
    features.append(("momentum_12_1", f.values))

    # --- 特征 2: 3-1 短期动量 --- IC=-0.015 ❌ 排除
    if not exclude_negative_ic:
        f = close.shift(5) / close.shift(63) - 1
        features.append(("momentum_3_1", f.values))

    # --- 特征 3: 63 天波动率 --- IC=+0.032 ✅
    f = daily_ret.rolling(63).std() * np.sqrt(252)
    features.append(("volatility_63", f.values))

    # --- 特征 4: 20 日价格区间比 --- IC=+0.014 ✅
    rolling_max = close.rolling(20).max()
    rolling_min = close.rolling(20).min()
    f = (rolling_max - rolling_min) / close
    features.append(("price_range_20", f.values))

    # --- 特征 5: 5 日反转 --- IC=+0.005 (弱但正) ✅
    f = close.shift(5) / close - 1
    features.append(("reversal_5", f.values))

    # --- 特征 6: 动量加速度 --- IC=-0.035 ❌ 排除
    if not exclude_negative_ic:
        mom_recent = close.shift(21) / close.shift(126) - 1
        mom_far = close.shift(126) / close.shift(252) - 1
        f = mom_recent - mom_far
        features.append(("mom_acceleration", f.values))

    # --- 特征 7 & 8: 基本面 ---
    if fundamentals is not None:
        from fundamental_factors import compute_value_factor, compute_quality_factor

        quality_score = compute_quality_factor(fundamentals)
        quality_arr = np.full((n_dates, n_stocks), np.nan)
        for j, t in enumerate(tickers):
            if t in quality_score.index and not np.isnan(quality_score[t]):
                quality_arr[:, j] = quality_score[t]
        features.append(("quality", quality_arr))  # IC=+0.044 ✅

        # value factor: IC=-0.048 ❌ 排除
        if not exclude_negative_ic:
            value_score = compute_value_factor(fundamentals)
            value_arr = np.full((n_dates, n_stocks), np.nan)
            for j, t in enumerate(tickers):
                if t in value_score.index and not np.isnan(value_score[t]):
                    value_arr[:, j] = value_score[t]
            features.append(("value", value_arr))

    name_list = [name for name, _ in features]
    values = [val for _, val in features]

    # Stack into 3D array: (dates, stocks, features)
    features_3d = np.stack(values, axis=2)
    n_features = features_3d.shape[2]

    logger.info(f"  时间点: {n_dates}, 股票数: {n_stocks}, 特征数: {n_features}")
    logger.info(f"  特征列表: {name_list}")
    if exclude_negative_ic:
        logger.info(f"  [已排除负面IC因子: momentum_3_1, mom_acceleration, value]")

    return features_3d, name_list


# ============================================================================
# 标签构建
# ============================================================================


def build_labels(close: pd.DataFrame, forward_days: int = 21) -> np.ndarray:
    """
    构建训练标签：未来 N 日收益率（原始值）。

    Lambdarank 需要组内排名，排名在训练循环内按 sampled subset 完成，
    因为全局排名值可能超出单组 label 范围。
    这里返回原始收益率，供 train_and_predict 内部做局部 rank。
    """
    future_return = close.shift(-forward_days) / close - 1
    return future_return.values.copy()


# ============================================================================
# Rolling 训练 + 预测（向量化核心）
# ============================================================================


def train_and_predict(
    close: pd.DataFrame,
    features_3d: np.ndarray,
    feature_names: list[str],
    train_window: int = 504,
    forward_days: int = 21,
    rebalance_freq: int = 21,
    sample_ratio: float = 0.15,
    smooth_alpha: float = 0.6,
    lgb_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Rolling 窗口训练 + 预测（向量化版本）。

    核心优化：
    1. 用 numpy 广播 + masking 代替逐行循环构建训练数据
    2. 使用 lambdarank 排序学习目标（而非回归）
    3. 预测平滑：对上一期预测做 EMA 混合，降低换手率

    Args:
        close: 收盘价 DataFrame
        features_3d: shape (n_dates, n_stocks, n_features)
        feature_names: 特征名列表
        train_window: 训练窗口长度（交易日）
        forward_days: 标签的前向天数
        rebalance_freq: 调仓频率（交易日）
        sample_ratio: 每个截面随机采样的股票比例（降低噪音）
        smooth_alpha: 预测平滑系数 (0=全用上期, 1=全用本期)
        lgb_params: LightGBM 参数
    """
    logger.info("=" * 60)
    logger.info("  LightGBM Rolling 训练 + 预测 (LambdaRank)")
    logger.info("=" * 60)

    if lgb_params is None:
        lgb_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_at": [20, 50, 100],
        }

    n_dates, n_stocks, n_features = features_3d.shape
    dates = close.index
    tickers = close.columns.tolist()

    # 构建标签
    labels = build_labels(close, forward_days=forward_days)

    # 调仓日索引
    rebalance_indices = list(
        range(train_window, n_dates - forward_days, rebalance_freq)
    )

    if len(rebalance_indices) == 0:
        raise ValueError(
            f"数据不足：需要至少 {train_window + forward_days} 个交易日，"
            f"当前只有 {n_dates} 个"
        )

    logger.info(f"训练窗口: {train_window} 日 ({train_window/252:.1f} 年)")
    logger.info(f"前向天数: {forward_days} 日")
    logger.info(f"调仓次数: {len(rebalance_indices)}")
    logger.info(f"截面采样率: {sample_ratio:.0%} (每个截面随机采样)")
    logger.info(f"预测平滑系数: {smooth_alpha:.1%} (new={smooth_alpha:.0%}, old={1-smooth_alpha:.0%})")

    rng = np.random.RandomState(42)

    # 固定采样大小: lambdarank 要求所有 group 大小一致
    # 用保守值: 假设最少 200 只有效股票 × 15% = 30
    fixed_group_size = max(20, int(200 * sample_ratio))
    logger.info(f"  固定采样大小: {fixed_group_size} (所有 group 统一，满足 lambdarank 要求)")

    # 预测结果
    predictions = np.zeros((n_dates, n_stocks))

    # 特征重要性累计
    total_importance = np.zeros(n_features)
    n_models = 0

    for i, idx in enumerate(rebalance_indices):
        date = dates[idx]
        train_start = idx - train_window

        # ── 按截面组织训练集（lambdarank 需要 group）──
        X_parts = []
        y_parts = []
        group_sizes = []

        for t in range(train_start, idx):
            feat_row = features_3d[t]  # (n_stocks, n_features)
            label_row = labels[t]       # (n_stocks,) - raw future returns
            valid = ~np.any(np.isnan(feat_row), axis=1) & ~np.isnan(label_row)
            valid_idx = np.where(valid)[0]

            if len(valid_idx) < fixed_group_size:
                continue

            # 固定大小随机采样 (所有 group 大小一致，满足 lambdarank 要求)
            sampled = rng.choice(valid_idx, size=fixed_group_size, replace=False)

            # 局部排名: 在 sampled subset 内按收益率排序 (0=最差, n-1=最好)
            sampled_labels = label_row[sampled]
            sort_order = np.argsort(sampled_labels)
            reranked = np.empty(fixed_group_size, dtype=np.float64)
            reranked[sort_order] = np.arange(fixed_group_size, dtype=np.float64)

            X_parts.append(feat_row[sampled])
            y_parts.append(reranked)
            group_sizes.append(fixed_group_size)

        if len(X_parts) < 20:  # 至少 20 个截面
            continue

        X_train = np.concatenate(X_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)

        if len(X_train) < 1000:
            continue

        # ── 构建测试集（当前截面，完整截面）──
        X_test_raw = features_3d[idx]  # (n_stocks, n_features)
        valid_test = ~np.any(np.isnan(X_test_raw), axis=1)
        X_test = X_test_raw[valid_test]
        test_ticker_mask = valid_test

        if len(X_test) == 0:
            continue

        # ── 训练 LightGBM（lambdarank）──
        # group 信息告诉模型哪些样本属于同一个截面
        n_train = int(len(group_sizes) * 0.75)
        train_groups = group_sizes[:n_train]
        val_groups = group_sizes[n_train:]
        n_train_samples = sum(train_groups)
        n_val_samples = sum(val_groups)

        train_data = lgb.Dataset(
            X_train[:n_train_samples], label=y_train[:n_train_samples],
            group=train_groups,
        )
        val_data = lgb.Dataset(
            X_train[n_train_samples:], label=y_train[n_train_samples:],
            group=val_groups, reference=train_data,
        )

        model = lgb.train(
            train_set=train_data,
            valid_sets=[val_data],
            params=lgb_params,
            num_boost_round=200,
            callbacks=[
                lgb.early_stopping(stopping_rounds=15, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # ── 预测 + 时序平滑 ──
        y_pred = model.predict(X_test)

        # EMA 平滑：新预测与上一期预测混合，降低月度换手率
        # 公式：smoothed = alpha * new_pred + (1-alpha) * old_pred
        if smooth_alpha < 1.0 and i > 0:
            old_pred = predictions[idx - rebalance_freq, test_ticker_mask]
            has_old = old_pred != 0
            y_pred_smoothed = np.zeros_like(y_pred)
            y_pred_smoothed[has_old] = smooth_alpha * y_pred[has_old] + (1 - smooth_alpha) * old_pred[has_old]
            y_pred_smoothed[~has_old] = y_pred[~has_old]
            y_pred = y_pred_smoothed

        predictions[idx, test_ticker_mask] = y_pred

        # 累计特征重要性
        total_importance += model.feature_importance(importance_type="gain")
        n_models += 1

        if (i + 1) % 6 == 0 or i == len(rebalance_indices) - 1:
            best_iter = model.best_iteration
            logger.info(
                f"  [{i+1}/{len(rebalance_indices)}] {date.date()}: "
                f"train={n_train_samples:,} val={n_val_samples:,} "
                f"test={len(X_test)} best_iter={best_iter} "
                f"pred_mean={y_pred.mean():.3f}"
            )

    # ── 特征重要性排名 ──
    if n_models > 0:
        avg_importance = total_importance / n_models
        logger.info("\n特征重要性排名（平均 gain）:")
        ranked = sorted(
            zip(feature_names, avg_importance),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, imp in ranked:
            logger.info(f"  {name:<25} {imp:>10.1f}")

    # ── 转为 DataFrame + 前向填充 ──
    pred_df = pd.DataFrame(predictions, index=dates, columns=tickers)

    # 训练期之前置 0
    if len(rebalance_indices) > 0:
        first_date = dates[rebalance_indices[0]]
        pred_df.loc[pred_df.index < first_date] = 0.0

    # 前向填充
    pred_df = pred_df.ffill()

    valid_pct = (pred_df != 0).sum().sum() / pred_df.size
    logger.info(f"\n预测完成: 有效预测占比 {valid_pct:.1%}")
    logger.info("=" * 60)

    return pred_df

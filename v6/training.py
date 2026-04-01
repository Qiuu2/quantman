"""
v6/training.py - LightGBM LambdaRank 训练模块

核心改进（vs v4）：
1. 全截面不等 group：每天所有有效股票为一个 group，不再固定 30 只采样
2. Purged Walk-Forward：训练集和验证集间加 3 周 gap 避免数据泄露
3. 周频训练：已降采样到周频，消除日频自相关
4. EMA 平滑 alpha=0.35：更保守地混合新旧预测
5. 全量因子喂入：不做人工 IC 预筛，让模型自己选
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from v6.config import (
    TRAIN_WINDOW_WEEKS, GAP_WEEKS, VAL_WINDOW_WEEKS,
    FORWARD_WEEKS, SMOOTH_ALPHA, LGB_PARAMS,
    NUM_BOOST_ROUNDS, EARLY_STOPPING_ROUNDS,
)

logger = logging.getLogger(__name__)


def purged_walk_forward_train(
    features_3d: np.ndarray,
    labels_2d: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    feature_names: list[str],
    train_window_weeks: int = TRAIN_WINDOW_WEEKS,
    gap_weeks: int = GAP_WEEKS,
    val_window_weeks: int = VAL_WINDOW_WEEKS,
    forward_weeks: int = FORWARD_WEEKS,
    smooth_alpha: float = SMOOTH_ALPHA,
    lgb_params: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Purged Walk-Forward LambdaRank 训练。

    每个调仓周：
    - 训练集: T-train_window 到 T-val_window-gap（跳过最近 gap 周）
    - Gap: val_window+gap 到 val_window（避免数据泄露）
    - 验证集: T-val_window 到 T

    全截面 group：每个截面（周）所有有效股票为一个 group。

    Args:
        features_3d: (n_weeks, n_stocks, n_features)
        labels_2d: (n_weeks, n_stocks) 未来收益率
        dates: 周频日期
        tickers: 股票代码
        feature_names: 特征名列表
        train_window_weeks: 训练窗口（周）
        gap_weeks: gap 长度（周）
        val_window_weeks: 验证集大小（周）
        forward_weeks: 前向预测周数
        smooth_alpha: EMA 平滑系数
        lgb_params: LightGBM 参数

    Returns:
        tuple: (predictions_df, diagnostics_dict)
    """
    logger.info("=" * 60)
    logger.info("  Purged Walk-Forward LambdaRank 训练")
    logger.info("=" * 60)

    if lgb_params is None:
        lgb_params = LGB_PARAMS.copy()

    n_weeks, n_stocks, n_features = features_3d.shape
    logger.info(f"数据维度: {n_weeks} 周 × {n_stocks} 股 × {n_features} 因子")
    logger.info(f"训练窗口: {train_window_weeks} 周 ({train_window_weeks/52:.1f} 年)")
    logger.info(f"Gap: {gap_weeks} 周, 验证集: {val_window_weeks} 周")
    logger.info(f"EMA 平滑: alpha={smooth_alpha:.2f}")

    # 调仓周索引（每隔 forward_weeks 调仓一次）
    start_idx = train_window_weeks + val_window_weeks + gap_weeks
    rebalance_indices = list(range(start_idx, n_weeks - forward_weeks, forward_weeks))

    if len(rebalance_indices) == 0:
        raise ValueError(
            f"数据不足：需要至少 {start_idx + forward_weeks} 周，当前只有 {n_weeks} 周"
        )

    logger.info(f"调仓次数: {len(rebalance_indices)}")

    # 预测结果
    predictions = np.zeros((n_weeks, n_stocks))

    # 诊断数据
    total_importance = np.zeros(n_features)
    n_models = 0
    ndcg_train_history = []
    ndcg_val_history = []
    pred_change_history = []

    rng = np.random.RandomState(42)

    for i, idx in enumerate(rebalance_indices):
        date = dates[idx]

        # ── 切分训练/验证集（Purged Walk-Forward）──
        train_end = idx - val_window_weeks - gap_weeks
        train_start = train_end - train_window_weeks
        val_start = idx - val_window_weeks
        val_end = idx

        # 边界检查
        if train_start < 0 or val_start < train_end + gap_weeks:
            continue

        # ── 构建训练数据（全截面 group）──
        X_parts = []
        y_parts = []
        group_sizes = []

        for t in range(train_start, train_end):
            feat_row = features_3d[t]  # (n_stocks, n_features)
            label_row = labels_2d[t]   # (n_stocks,) raw future returns

            valid = ~np.any(np.isnan(feat_row), axis=1) & ~np.isnan(label_row)
            valid_idx = np.where(valid)[0]

            if len(valid_idx) < 10:
                continue

            sampled_returns = label_row[valid_idx]
            rank_order = np.argsort(-sampled_returns)
            ranks = np.empty(len(valid_idx), dtype=np.float64)
            ranks[rank_order] = np.arange(len(valid_idx), dtype=np.float64)

            X_parts.append(feat_row[valid_idx])
            y_parts.append(ranks)
            group_sizes.append(len(valid_idx))

        # ── 构建验证数据 ──
        X_val_parts = []
        y_val_parts = []
        val_group_sizes = []

        for t in range(val_start, val_end):
            feat_row = features_3d[t]
            label_row = labels_2d[t]

            valid = ~np.any(np.isnan(feat_row), axis=1) & ~np.isnan(label_row)
            valid_idx = np.where(valid)[0]

            if len(valid_idx) < 10:
                continue

            sampled_returns = label_row[valid_idx]
            rank_order = np.argsort(-sampled_returns)
            ranks = np.empty(len(valid_idx), dtype=np.float64)
            ranks[rank_order] = np.arange(len(valid_idx), dtype=np.float64)

            X_val_parts.append(feat_row[valid_idx])
            y_val_parts.append(ranks)
            val_group_sizes.append(len(valid_idx))

        if len(X_parts) < 10 or len(X_val_parts) < 2:
            continue

        X_train = np.concatenate(X_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)
        X_val = np.concatenate(X_val_parts, axis=0)
        y_val = np.concatenate(y_val_parts, axis=0)

        if len(X_train) < 1000 or len(X_val) < 100:
            continue

        # ── 训练 LightGBM ──
        train_data = lgb.Dataset(X_train, label=y_train, group=group_sizes)
        val_data = lgb.Dataset(
            X_val, label=y_val, group=val_group_sizes
        )

        model = lgb.train(
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            params=lgb_params,
            num_boost_round=NUM_BOOST_ROUNDS,
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        best_iter = model.best_iteration
        best_score = model.best_score.get("val", {}).get("ndcg", float("nan"))

        # 记录 NDCG
        train_ndcg = model.best_score.get("train", {}).get("ndcg", float("nan"))
        ndcg_train_history.append(train_ndcg)
        ndcg_val_history.append(best_score)

        # ── 预测 ──
        X_test = features_3d[idx]
        valid_test = ~np.any(np.isnan(X_test), axis=1)

        if valid_test.sum() == 0:
            continue

        y_pred = model.predict(X_test[valid_test])

        # ── EMA 平滑 ──
        if smooth_alpha < 1.0 and i > 0:
            prev_idx = rebalance_indices[i - 1]
            old_pred = predictions[prev_idx, valid_test]
            has_old = old_pred != 0

            y_pred_smoothed = np.zeros_like(y_pred)
            y_pred_smoothed[has_old] = (
                smooth_alpha * y_pred[has_old]
                + (1 - smooth_alpha) * old_pred[has_old]
            )
            y_pred_smoothed[~has_old] = y_pred[~has_old]
            y_pred = y_pred_smoothed

        predictions[idx, valid_test] = y_pred

        # 预测变化率
        if i > 0:
            prev_idx = rebalance_indices[i - 1]
            common_valid = valid_test & (predictions[prev_idx] != 0)
            if common_valid.sum() > 0:
                corr = np.corrcoef(predictions[prev_idx, common_valid], y_pred[common_valid[valid_test]])[0, 1]
                pred_change_history.append(corr)

        # 累计特征重要性
        total_importance += model.feature_importance(importance_type="gain")
        n_models += 1

        # 日志
        if (i + 1) % 10 == 0 or i == len(rebalance_indices) - 1:
            n_train = len(X_train)
            n_val = len(X_val)
            n_test = valid_test.sum()
            n_groups = len(group_sizes)
            avg_group = np.mean(group_sizes) if group_sizes else 0
            logger.info(
                f"  [{i+1}/{len(rebalance_indices)}] {date.date()}: "
                f"train={n_train:,} ({n_groups} groups, avg={avg_group:.0f}) "
                f"val={n_val:,} test={n_test} "
                f"best_iter={best_iter} val_ndcg={best_score:.4f}"
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
        for j, (name, imp) in enumerate(ranked):
            logger.info(f"  {j+1:>2}. {name:<25} {imp:>12.1f}")
    else:
        avg_importance = np.zeros(n_features)
        ranked = []

    # ── 前向填充预测 ──
    pred_df = pd.DataFrame(predictions, index=dates, columns=tickers)

    if rebalance_indices:
        first_date = dates[rebalance_indices[0]]
        pred_df.loc[pred_df.index < first_date] = 0.0

    pred_df = pred_df.ffill()

    valid_pct = (pred_df != 0).sum().sum() / pred_df.size
    logger.info(f"\n训练完成: {n_models} 个模型, 有效预测占比 {valid_pct:.1%}")

    # 预测稳定性
    if pred_change_history:
        mean_corr = np.mean(pred_change_history)
        logger.info(f"预测周间相关性: {mean_corr:.3f}")

    logger.info("=" * 60)

    # 诊断数据
    diagnostics = {
        "feature_importance": dict(ranked) if ranked else {},
        "avg_importance": dict(zip(feature_names, avg_importance)),
        "ndcg_train": ndcg_train_history,
        "ndcg_val": ndcg_val_history,
        "pred_correlation": pred_change_history,
        "n_models": n_models,
    }

    return pred_df, diagnostics

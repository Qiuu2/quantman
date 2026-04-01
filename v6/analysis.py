"""
v6/analysis.py - 诊断分析模块

生成训练过程的诊断图表：
1. IC 时序图（每个因子的信息系数随时间变化）
2. Feature Importance 排名
3. 训练集 vs 验证集 NDCG 曲线（过拟合诊断）
4. 预测稳定性分析（周间相关性）
5. 换手率分析
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from v6.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def plot_ic_timeseries(
    features_3d: np.ndarray,
    labels_2d: np.ndarray,
    feature_names: list[str],
    dates: pd.DatetimeIndex,
    save_dir: Path = REPORTS_DIR,
) -> Path:
    """
    计算并绘制每个因子的 IC 时序图。

    IC = Spearman 相关系数（因子值 vs 未来收益排名）
    """
    logger.info("绘制 IC 时序图...")
    n_weeks, n_stocks, n_features = features_3d.shape

    # 计算滚动 IC（20 周窗口）
    window = 20
    ic_data = {}

    for f_idx, fname in enumerate(feature_names):
        ics = []
        for t in range(window, n_weeks):
            feat = features_3d[t - window:t, :, f_idx]
            label = labels_2d[t - window:t, :]

            # 对每个截面计算 rank correlation
            corr_vals = []
            for k in range(window):
                valid = ~np.isnan(feat[k]) & ~np.isnan(label[k])
                if valid.sum() > 20:
                    corr = np.corrcoef(feat[k, valid], label[k, valid])[0, 1]
                    corr_vals.append(corr)

            if corr_vals:
                ics.append(np.mean(corr_vals))
            else:
                ics.append(np.nan)

        ic_data[fname] = ics

    # 绘图
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    fig.suptitle("Factor IC Time Series (20-Week Rolling Average)", fontsize=16, fontweight="bold")

    axes_flat = axes.flatten() if n_features > 1 else [axes]
    ic_dates = dates[window:]

    for f_idx, fname in enumerate(feature_names):
        ax = axes_flat[f_idx]
        ics = ic_data[fname]

        ax.plot(ic_dates[:len(ics)], ics, linewidth=0.8, alpha=0.8)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        mean_ic = np.nanmean(ics)
        ax.axhline(y=mean_ic, color="red", linestyle=":", alpha=0.6)

        ax.set_title(f"{fname} (mean={mean_ic:.3f})", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for f_idx in range(n_features, len(axes_flat)):
        axes_flat[f_idx].set_visible(False)

    plt.tight_layout()
    save_path = save_dir / "ic_timeseries.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"IC 时序图已保存: {save_path}")

    # 打印 IC 摘要
    logger.info("\n因子 IC 摘要:")
    for fname in feature_names:
        ics = ic_data[fname]
        mean_ic = np.nanmean(ics)
        std_ic = np.nanstd(ics)
        ir = mean_ic / std_ic if std_ic > 0 else 0
        pos_pct = (np.array(ics) > 0).mean() * 100 if ics else 0
        logger.info(f"  {fname:<25} IC={mean_ic:+.4f}  IR={ir:.2f}  正IC占比={pos_pct:.0f}%")

    return save_path


def plot_feature_importance(
    diagnostics: dict,
    save_dir: Path = REPORTS_DIR,
) -> Path:
    """
    绘制 Feature Importance 排名图。
    """
    logger.info("绘制特征重要性图...")

    importance = diagnostics.get("feature_importance", {})
    if not importance:
        logger.warning("无特征重要性数据")
        return Path("")

    # 排序
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Average Gain")
    ax.set_title("LightGBM Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # 在 bar 上标注值
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f}", va="center", fontsize=8)

    plt.tight_layout()
    save_path = save_dir / "feature_importance.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"特征重要性图已保存: {save_path}")

    return save_path


def plot_ndcg_curves(
    diagnostics: dict,
    save_dir: Path = REPORTS_DIR,
) -> Path:
    """
    绘制训练集 vs 验证集 NDCG 曲线（过拟合诊断）。
    """
    logger.info("绘制 NDCG 曲线...")

    train_ndcg = diagnostics.get("ndcg_train", [])
    val_ndcg = diagnostics.get("ndcg_val", [])

    if not train_ndcg or not val_ndcg:
        logger.warning("无 NDCG 数据")
        return Path("")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：NDCG 时序
    ax1 = axes[0]
    x = range(1, len(train_ndcg) + 1)
    ax1.plot(x, train_ndcg, label="Train NDCG", alpha=0.8, linewidth=0.8)
    ax1.plot(x, val_ndcg, label="Val NDCG", alpha=0.8, linewidth=0.8)
    ax1.set_xlabel("Training Round")
    ax1.set_ylabel("NDCG@100")
    ax1.set_title("NDCG over Training Rounds")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：Train-Val Gap（过拟合程度）
    ax2 = axes[1]
    gap = np.array(val_ndcg) - np.array(train_ndcg)
    ax2.bar(x, gap, alpha=0.6, color="red" if np.mean(gap) < -0.05 else "green")
    ax2.axhline(y=0, color="gray", linestyle="--")
    ax2.axhline(y=np.mean(gap), color="blue", linestyle=":", label=f"Mean Gap={np.mean(gap):.4f}")
    ax2.set_xlabel("Training Round")
    ax2.set_ylabel("Val - Train NDCG")
    ax2.set_title("Overfitting Diagnostic (Train-Val Gap)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "ndcg_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"NDCG 曲线已保存: {save_path}")

    # 诊断结论
    mean_gap = np.mean(gap)
    if mean_gap < -0.1:
        logger.warning(f"⚠️ 严重过拟合：平均 Train-Val NDCG Gap = {mean_gap:.4f}")
    elif mean_gap < -0.05:
        logger.warning(f"⚠️ 轻微过拟合：平均 Train-Val NDCG Gap = {mean_gap:.4f}")
    else:
        logger.info(f"✅ 过拟合程度可接受：平均 Train-Val NDCG Gap = {mean_gap:.4f}")

    return save_path


def plot_prediction_stability(
    diagnostics: dict,
    save_dir: Path = REPORTS_DIR,
) -> Path:
    """
    绘制预测稳定性分析（周间相关性）。
    """
    logger.info("绘制预测稳定性图...")

    corr = diagnostics.get("pred_correlation", [])
    if not corr:
        logger.warning("无预测相关性数据")
        return Path("")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(1, len(corr) + 1), corr, linewidth=0.8, alpha=0.8)
    ax.axhline(y=np.mean(corr), color="red", linestyle=":", label=f"Mean={np.mean(corr):.3f}")
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Threshold=0.5")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Correlation with Previous Prediction")
    ax.set_title("Prediction Stability (Week-over-Week Correlation)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "prediction_stability.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"预测稳定性图已保存: {save_path}")

    mean_corr = np.mean(corr)
    if mean_corr < 0.3:
        logger.warning(f"⚠️ 预测极度不稳定：平均周间相关性 = {mean_corr:.3f}（信号几乎随机翻转）")
    elif mean_corr < 0.6:
        logger.warning(f"⚠️ 预测不够稳定：平均周间相关性 = {mean_corr:.3f}")
    else:
        logger.info(f"✅ 预测稳定性良好：平均周间相关性 = {mean_corr:.3f}")

    return save_path


def analyze_turnover(
    weights: pd.DataFrame,
    save_dir: Path = REPORTS_DIR,
) -> dict:
    """
    分析组合换手率。
    """
    logger.info("分析换手率...")

    weight_changes = weights.diff()
    turnover = weight_changes.abs().sum(axis=1) / 2

    # 按月汇总
    monthly_turnover = turnover.resample("ME").sum()

    stats = {
        "mean_weekly": float(turnover.mean()),
        "median_weekly": float(turnover.median()),
        "std_weekly": float(turnover.std()),
        "max_weekly": float(turnover.max()),
        "mean_monthly": float(monthly_turnover.mean()),
        "total": float(turnover.sum()),
    }

    logger.info(f"  平均周换手率: {stats['mean_weekly']:.4f}")
    logger.info(f"  平均月换手率: {stats['mean_monthly']:.4f}")
    logger.info(f"  总换手率: {stats['total']:.2f}")

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    ax1.bar(turnover.index, turnover.values, width=5, alpha=0.6)
    ax1.axhline(y=stats["mean_weekly"], color="red", linestyle=":", label=f"Mean={stats['mean_weekly']:.4f}")
    ax1.set_ylabel("Weekly Turnover")
    ax1.set_title("Weekly Turnover", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    monthly_turnover.plot(kind="bar", ax=ax2, alpha=0.6, width=0.8)
    ax2.axhline(y=stats["mean_monthly"], color="red", linestyle=":", label=f"Mean={stats['mean_monthly']:.4f}")
    ax2.set_ylabel("Monthly Turnover")
    ax2.set_title("Monthly Turnover", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path = save_dir / "turnover_analysis.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"换手率图已保存: {save_path}")

    return stats


def run_full_diagnostics(
    features_3d: np.ndarray,
    labels_2d: np.ndarray,
    feature_names: list[str],
    dates: pd.DatetimeIndex,
    diagnostics: dict,
    weights: pd.DataFrame,
) -> None:
    """
    运行全部诊断并保存图表。
    """
    logger.info("=" * 60)
    logger.info("  运行诊断分析")
    logger.info("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. IC 时序图
    plot_ic_timeseries(features_3d, labels_2d, feature_names, dates)

    # 2. Feature Importance
    plot_feature_importance(diagnostics)

    # 3. NDCG 曲线
    plot_ndcg_curves(diagnostics)

    # 4. 预测稳定性
    plot_prediction_stability(diagnostics)

    # 5. 换手率分析
    analyze_turnover(weights)

    logger.info("=" * 60)
    logger.info("  诊断分析完成，图表保存在:", REPORTS_DIR)
    logger.info("=" * 60)

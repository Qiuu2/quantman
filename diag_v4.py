"""v4 LightGBM 诊断脚本：全面分析失败原因"""

import logging, warnings
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from data_loader import load_or_download, get_close_prices, load_config
from factors import momentum_12_1, combine_factors
from fundamental_factors import fetch_fundamental_data, build_fundamental_panel
from ml_factors import build_feature_matrix, build_labels, train_and_predict
from portfolio import construct_portfolio
from backtest import compute_returns, run_backtest


def max_consecutive(arr):
    max_count = count = 0
    for v in arr:
        if v:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


def main():
    config = load_config('config.json')
    fields = load_or_download('config.json')
    close = get_close_prices(fields)
    stock_close = close.drop(columns=['SPY'], errors='ignore')
    all_returns = compute_returns(close)

    fund = fetch_fundamental_data(stock_close.columns.tolist())
    features_3d, feature_names = build_feature_matrix(stock_close, fund)
    # Labels are now raw future returns, compute rank for IC analysis
    labels_raw = build_labels(stock_close, forward_days=21)
    # Compute cross-sectional rank for diagnostic purposes
    labels_rank = pd.DataFrame(labels_raw, index=stock_close.index, columns=stock_close.columns).rank(axis=1, pct=True).values
    labels = labels_raw  # Use raw returns for training/IC
    labels_ranked = labels_rank  # Use ranked for IC diagnostics
    n_dates, n_stocks, n_features = features_3d.shape

    # ========== Diagnostic 1: Label Quality ==========
    print('=' * 70)
    print('  DIAG 1: Label Quality (Future 21D Return)')
    print('=' * 70)
    print(f'Future 21D return mean:    {np.nanmean(labels):.4f}')
    print(f'Future 21D return median:  {np.nanmedian(labels):.4f}')
    print(f'Future 21D return std:     {np.nanstd(labels):.4f}')

    # Label autocorrelation (rank-based, month-to-month)
    valid_mask = ~np.isnan(labels)
    label_autocorr = []
    for t in range(1, n_dates - 21):
        row_prev = labels_ranked[t - 1]
        row_curr = labels_ranked[t]
        mask = valid_mask[t - 1] & valid_mask[t]
        if mask.sum() < 30:
            continue
        r = np.corrcoef(row_prev[mask], row_curr[mask])[0, 1]
        if not np.isnan(r) and not np.isinf(r):
            label_autocorr.append(r)
    print(f'\nLabel month-to-month autocorrelation: {np.mean(label_autocorr):.4f}')
    print(f'  (low = labels change a lot each month = hard to predict)')

    # ========== Diagnostic 2: Single Factor IC ==========
    print(f'\n{"=" * 70}')
    print('  DIAG 2: Single Factor IC (Feature vs Label)')
    print('=' * 70)
    ics = {name: [] for name in feature_names}

    for t in range(n_dates):
        for i, name in enumerate(feature_names):
            feat_col = features_3d[t, :, i]
            label_col = labels_ranked[t]  # Use ranked labels for IC
            mask = ~np.isnan(feat_col) & ~np.isnan(label_col)
            if mask.sum() < 30:
                continue
            r, _ = spearmanr(feat_col[mask], label_col[mask])
            if not np.isnan(r):
                ics[name].append(r)

    print(f'{"Factor":<25} {"IC":>8} {"IR":>8} {"IC>0%":>8} {"Flip%":>8} {"N":>6}')
    print('-' * 70)
    for name in feature_names:
        vals = np.array(ics[name])
        if len(vals) > 0:
            mean_ic = vals.mean()
            std_ic = vals.std()
            ir = mean_ic / std_ic if std_ic > 0 else 0
            ic_pos = (vals > 0).mean()
            flips = np.diff(np.sign(vals))
            flip_rate = (flips != 0).mean()
            print(f'  {name:<25} {mean_ic:>+8.4f} {ir:>+8.3f} {ic_pos:>7.0%} {flip_rate:>7.0%} {len(vals):>6}')

    # ========== Diagnostic 3: Prediction Quality ==========
    print(f'\n{"=" * 70}')
    print('  DIAG 3: Prediction Quality (Rolling Predict vs Actual)')
    print('=' * 70)

    ml_predictions = train_and_predict(
        close=stock_close,
        features_3d=features_3d,
        feature_names=feature_names,
        train_window=504,
        forward_days=21,
        rebalance_freq=21,
        sample_ratio=0.15,
    )

    pred_ic_pearson = []
    pred_ic_spearman = []
    for t in range(n_dates):
        pred_row = ml_predictions.iloc[t]
        label_row = labels_ranked[t]  # Use ranked labels for IC
        mask = (pred_row != 0) & ~np.isnan(label_row)
        if mask.sum() < 30:
            continue
        r_p = np.corrcoef(pred_row[mask], label_row[mask])[0, 1]
        if not np.isnan(r_p) and not np.isinf(r_p):
            pred_ic_pearson.append(r_p)
        r_s, _ = spearmanr(pred_row[mask], label_row[mask])
        if not np.isnan(r_s) and not np.isinf(r_s):
            pred_ic_spearman.append(r_s)

    pred_ic_p = np.array(pred_ic_pearson)
    pred_ic_s = np.array(pred_ic_spearman)
    if len(pred_ic_p) > 0:
        print(f'Prediction IC (Pearson):  mean={pred_ic_p.mean():+.4f}  std={pred_ic_p.std():.4f}  IR={pred_ic_p.mean()/pred_ic_p.std() if pred_ic_p.std()>0 else 0:+.3f}')
        print(f'Prediction IC (Spearman): mean={pred_ic_s.mean():+.4f}  std={pred_ic_s.std():.4f}  IR={pred_ic_s.mean()/pred_ic_s.std() if pred_ic_s.std()>0 else 0:+.3f}')
        print(f'Positive IC months: {(pred_ic_s > 0).mean():.0%}')
        print(f'Max consecutive positive IC: {max_consecutive(pred_ic_s > 0)}')
        print(f'Max consecutive negative IC: {max_consecutive(pred_ic_s < 0)}')

    # ========== Diagnostic 4: Yearly IC Breakdown ==========
    print(f'\n{"=" * 70}')
    print('  DIAG 4: Yearly IC Breakdown (When Does Model Fail?)')
    print('=' * 70)

    pred_dates = ml_predictions.index
    valid_mask2 = ml_predictions.ne(0).any(axis=1)
    monthly_ics = {}
    for t_idx in range(len(pred_dates)):
        if not valid_mask2.iloc[t_idx]:
            continue
        if t_idx + 21 >= n_dates:
            break
        pred_row = ml_predictions.iloc[t_idx]
        label_row = labels_ranked[t_idx]  # Use ranked labels for IC
        mask = (pred_row != 0) & ~np.isnan(label_row)
        if mask.sum() < 30:
            continue
        r_s, _ = spearmanr(pred_row[mask], label_row[mask])
        if not np.isnan(r_s):
            monthly_ics[pred_dates[t_idx]] = r_s

    for year in sorted(set(d.year for d in monthly_ics.keys())):
        year_ics = [v for d, v in monthly_ics.items() if d.year == year]
        pos = sum(1 for x in year_ics if x > 0)
        print(f'  {year}: IC_mean={np.mean(year_ics):+.4f}  IC_median={np.median(year_ics):+.4f}  pos_months={pos}/{len(year_ics)}')

    # ========== Diagnostic 5: Turnover Analysis ==========
    print(f'\n{"=" * 70}')
    print('  DIAG 5: Turnover & Transaction Cost Analysis')
    print('=' * 70)

    mom = momentum_12_1(stock_close, lookback=252, skip=21)
    quality_panel = build_fundamental_panel(fund, stock_close.index, 'quality')
    composite_v3b = combine_factors({'momentum': mom, 'quality': quality_panel}, {'momentum': 0.6, 'quality': 0.4})

    w3b = construct_portfolio(composite_v3b, top_pct=0.2, bottom_pct=0.2)
    w4 = construct_portfolio(ml_predictions, top_pct=0.2, bottom_pct=0.2)

    r3b = run_backtest(w3b, all_returns, 10, 100000)
    r4 = run_backtest(w4, all_returns, 10, 100000)

    print(f'v3b: turnover={float(r3b["turnover"].mean())*100:.1f}%  TC={float(r3b["total_tc"])*100:.2f}%  gross_ret={float(r3b["cumulative_returns"].iloc[-1])*100-100:.2f}%')
    print(f'v4:  turnover={float(r4["turnover"].mean())*100:.1f}%  TC={float(r4["total_tc"])*100:.2f}%  gross_ret={float(r4["cumulative_returns"].iloc[-1])*100-100:.2f}%')

    # Prediction stability
    pred_changes = ml_predictions.diff().abs()
    nonzero_mask = ml_predictions.shift(1) != 0
    print(f'\nPrediction change magnitude (non-zero periods):')
    print(f'  Mean:   {pred_changes[nonzero_mask].mean().mean():.4f}')
    print(f'  Std:    {pred_changes[nonzero_mask].std().mean():.4f}')
    print(f'  Max:    {pred_changes[nonzero_mask].max().max():.4f}')

    # Top/Bottom portfolio overlap month-to-month
    print(f'\nPortfolio overlap (month-to-month):')
    top_pct = 0.2
    n_top = int(n_stocks * top_pct)
    overlap_scores = []
    pred_dates_valid = [d for d in pred_dates if valid_mask2.loc[d]]
    for i in range(1, len(pred_dates_valid)):
        d_prev = pred_dates_valid[i - 1]
        d_curr = pred_dates_valid[i]
        top_prev = set(ml_predictions.loc[d_prev].nlargest(n_top).index)
        top_curr = set(ml_predictions.loc[d_curr].nlargest(n_top).index)
        overlap = len(top_prev & top_curr) / n_top
        overlap_scores.append(overlap)
    print(f'  Long overlap:  {np.mean(overlap_scores):.1%}')
    print(f'  (higher = more stable = lower turnover)')

    # ========== Diagnostic 6: Feature Importance ==========
    print(f'\n{"=" * 70}')
    print('  DIAG 6: Quick Feature Importance (Single Model)')
    print('=' * 70)
    import lightgbm as lgb

    # Train one model at midpoint
    mid_idx = len(pred_dates_valid) // 2
    train_start = pred_dates_valid[mid_idx] - pd.Timedelta(days=730)
    start_loc = stock_close.index.get_loc(train_start)
    end_loc = stock_close.index.get_loc(pred_dates_valid[mid_idx])

    X_parts = []
    y_parts = []
    group_sizes = []
    rng = np.random.RandomState(42)
    for t in range(start_loc, end_loc):
        feat_row = features_3d[t]
        label_row = labels[t]  # raw returns
        valid = ~np.any(np.isnan(feat_row), axis=1) & ~np.isnan(label_row)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 20:
            continue
        n_sample = max(20, int(len(valid_idx) * 0.15))
        sampled = rng.choice(valid_idx, size=n_sample, replace=False)
        # Re-rank within sampled subset for lambdarank
        sampled_labels = label_row[sampled]
        sort_order = np.argsort(sampled_labels)
        reranked = np.empty(n_sample, dtype=np.float64)
        reranked[sort_order] = np.arange(n_sample, dtype=np.float64)
        X_parts.append(feat_row[sampled])
        y_parts.append(reranked)
        group_sizes.append(n_sample)

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)

    n_train_groups = int(len(group_sizes) * 0.75)
    train_groups = group_sizes[:n_train_groups]
    n_train_samples = sum(train_groups)

    train_data = lgb.Dataset(
        X_train[:n_train_samples], label=y_train[:n_train_samples],
        group=train_groups,
    )
    params = {
        "objective": "lambdarank", "metric": "ndcg", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.05, "verbose": -1, "seed": 42,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "eval_at": [20, 50, 100],
    }
    model = lgb.train(train_set=train_data, params=params, num_boost_round=100)

    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
    total = sum(imp)
    for name, val in ranked:
        print(f'  {name:<25} gain={val:>10.1f}  share={val/total:>6.1%}')

    print(f'\nDone!')


if __name__ == '__main__':
    main()

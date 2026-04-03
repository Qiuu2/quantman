"""
v6/config.py - v6 专用配置

集中管理 API key、超参数、路径常量。
"""

from pathlib import Path

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SIMFIN_CACHE_DIR = DATA_DIR / "simfin"
REPORTS_DIR = PROJECT_ROOT / "reports" / "v6_analysis"

# ── API Keys ──
SIMFIN_API_KEY = "504dc423-489b-4c3e-99b6-9556f112bf45"

# ── 回测参数 ──
START_DATE = "2020-07-01"  # SimFin 免费版数据从 2020-05 开始
END_DATE = "2025-01-01"
BENCHMARK_TICKER = "SPY"
INITIAL_CAPITAL = 100000
TRANSACTION_COST_BPS = 10
RISK_FREE_RATE = 0.04

# ── 股票池 ──
UNIVERSE = "russell3000"
MIN_QUARTERS = 1  # SimFin 免费版数据有限，放宽到 1 个季度

# ── 周频参数 ──
FORWARD_WEEKS = 3  # ~21 交易日
TRAIN_WINDOW_WEEKS = 104  # 2 年
GAP_WEEKS = 3  # ~21 天
VAL_WINDOW_WEEKS = 6  # 验证集约 6 周

# ── LightGBM 参数 ──
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "num_leaves": 16,
    "max_depth": 6,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
    "min_child_samples": 50,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "eval_at": [20, 50, 100],
    "max_position": 200,
}

NUM_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 15

# ── 组合构建 ──
TOP_PCT = 0.20
BOTTOM_PCT = 0.20

# ── EMA 平滑 ──
SMOOTH_ALPHA = 0.70  # 新预测权重 70%，旧预测 30%

# ── yfinance 下载参数 ──
YF_BATCH_SIZE = 50
YF_SLEEP_SECONDS = 1.0
YF_MAX_RETRIES = 3

# ── 确保目录存在 ──
SIMFIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

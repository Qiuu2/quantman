# QuantMan - 量化交易研究平台

Baseline: 横截面动量策略 (12-1 Momentum)

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Architecture

```
data_loader.py  → 获取标普500成分股 5年日线数据
factors.py      → 计算 12-1 动量因子
portfolio.py    → 每月调仓：做多Top20%，做空Bottom20%
backtest.py     → 向量化回测引擎
metrics.py      → 绩效评估（夏普、回撤、IR等）
main.py         → 一键运行 baseline
```

## Improvement Roadmap

- v0 Baseline: 12-1 动量，等权，SP500
- v1: + 流动性过滤
- v2: + 多因子（价值 + 质量）
- v3: + ML 因子组合 (LightGBM)
- v4: + AI 情绪因子 (NLP)
- v5: + Alpaca 模拟盘对接
- v6: + 风险控制（波动率目标 + 动态仓位）

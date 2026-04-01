"""
QuantMan v6 - ML Factor Synthesis with SimFin Fundamentals

重构自 v4 LightGBM，修复全部技术缺陷：
- SimFin 季度财务时序数据（利润表、资产负债表、现金流量表）
- Russell 3000 扩展股票池
- 20 个因子（8 技术面 + 12 基本面）
- LambdaRank 全截面不等 group（取消固定采样）
- Purged Walk-Forward 训练（21 天 gap 避免数据泄露）
- 周频降采样（消除日频特征自相关）
- EMA 预测平滑（alpha=0.35）
"""

import sys
from pathlib import Path

# 将父目录加入 sys.path，以便 import 复用模块
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

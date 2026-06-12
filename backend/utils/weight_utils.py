"""
权重计算公共工具 — 统一权重归一化逻辑

项目规范5：权重归一化在11处各自实现，提取统一入口
项目规范1：归一化中的除法必须使用 safe_divide
"""

from typing import Dict, Union
import numpy as np
import pandas as pd
from backend.utils.safe_math import safe_divide
from backend.constants import FLOAT_ZERO_THRESHOLD


def normalize_weights(
    weights: Union[Dict[str, float], pd.Series],
    default_equal: bool = True,
) -> Union[Dict[str, float], pd.Series]:
    """
    归一化权重（使总和为1）

    Args:
        weights: 权重字典或Series
        default_equal: 总和为0时是否回退到等权；False时返回零权重

    Returns:
        归一化后的权重（同类型）
    """
    zero_threshold = FLOAT_ZERO_THRESHOLD
    if isinstance(weights, dict):
        total = sum(weights.values())
        if abs(total) < zero_threshold or not np.isfinite(total):
            if default_equal:
                n = len(weights)
                return {k: 1.0 / n if n > 0 else 0.0 for k in weights}
            else:
                return {k: 0.0 for k in weights}
        return {k: safe_divide(v, total, default=0.0) for k, v in weights.items()}
    else:
        # pd.Series
        total = weights.sum()
        if abs(total) < zero_threshold or not np.isfinite(total):
            if default_equal:
                n = len(weights)
                return pd.Series(1.0 / n if n > 0 else 0.0, index=weights.index)
            else:
                return pd.Series(0.0, index=weights.index)
        return safe_divide(weights, total, default=0.0)

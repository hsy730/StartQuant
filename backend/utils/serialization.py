"""
NumPy类型安全序列化工具

统一处理 NumPy/Pandas 类型到 JSON 安全类型的转换。
所有路由和服务应使用此模块，而非各自实现转换逻辑。
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union


def safe_numeric_value(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    将数值转换为 JSON 安全的 float 或 None

    NaN 和 Inf 统一转为 None（JSON 中为 null），
    前端可据此显示为 "N/A" 或 "--"，而非误导性的 0.0。

    Args:
        value: 待转换的数值
        default: 非数值类型的默认返回值，默认为 None

    Returns:
        float 或 None
    """
    if value is None:
        return None
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return default


def sanitize_dict(d: Any) -> Any:
    """
    递归地将字典中的 NumPy 类型转换为 JSON 安全类型

    - NaN/Inf → None
    - np.integer → int
    - np.floating → float
    - np.ndarray → list
    - pd.Timestamp → str
    - 递归处理嵌套 dict 和 list

    Args:
        d: 待转换的对象

    Returns:
        转换后的对象
    """
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(item) for item in d]
    elif isinstance(d, (np.integer,)):
        return int(d)
    elif isinstance(d, (np.floating,)):
        f = float(d)
        return None if (np.isnan(f) or np.isinf(f)) else f
    elif isinstance(d, float):
        return None if (np.isnan(d) or np.isinf(d)) else d
    elif isinstance(d, np.ndarray):
        return sanitize_dict(d.tolist())
    elif isinstance(d, pd.Timestamp):
        return str(d)
    elif isinstance(d, (pd.Timedelta,)):
        return None
    elif isinstance(d, (bool, int, str)):
        return d
    elif d is None:
        return None
    else:
        try:
            f = float(d)
            return None if (np.isnan(f) or np.isinf(f)) else f
        except (TypeError, ValueError):
            return None

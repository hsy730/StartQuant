"""
安全数学运算工具 — 统一除零/NaN保护

项目规则3：所有除法/标准差必须处理零值
所有需要除法保护的地方应使用此模块，禁止各自发明保护模式（如 +1e-10 hack）
"""
import numpy as np
import pandas as pd
from typing import Optional, Union


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    default: Optional[float] = None,
    min_threshold: float = 1e-10,
) -> Union[float, np.ndarray, pd.Series]:
    """
    安全除法 — 统一处理除零和NaN

    当分母为0、NaN或绝对值小于min_threshold时，返回default值。

    Args:
        numerator: 分子
        denominator: 分母
        default: 分母无效时的返回值（默认None，符合规则6）
        min_threshold: 分母绝对值的最小有效阈值（防止浮点噪声导致极大结果）

    Returns:
        安全除法结果，分母无效时返回default

    Examples:
        >>> safe_divide(0.05, 0.0)  # 除零 → None
        >>> safe_divide(0.05, 7e-18)  # 浮点噪声 → None
        >>> safe_divide(0.05, 0.1, default=0.0)  # 正常除法 → 0.5
        >>> safe_divide(0.05, np.nan)  # NaN分母 → None
    """
    if isinstance(denominator, (pd.Series, np.ndarray)):
        # 向量化处理
        if isinstance(denominator, pd.Series):
            invalid = (denominator.abs() < min_threshold) | denominator.isna()
        else:
            invalid = (np.abs(denominator) < min_threshold) | np.isnan(denominator)

        result = numerator / denominator
        if isinstance(result, pd.Series):
            result = result.mask(invalid, default)
        else:
            result[invalid] = default
        return result
    else:
        # 标量处理
        if denominator is None or (isinstance(denominator, float) and np.isnan(denominator)):
            return default
        if abs(denominator) < min_threshold:
            return default
        return numerator / denominator


def safe_ir(ic_mean: float, ic_std: float, default: Optional[float] = None) -> Optional[float]:
    """
    安全计算信息比率 IR = IC均值 / IC标准差

    统一处理IC标准差为0/NaN/极小值的情况，替代各处自行实现的IR计算。

    特殊处理：当IC完全稳定（std=0但mean≠0）时，IR应趋向极大值，
    而非返回default=0。因为IC完全稳定意味着因子预测能力极其稳定。

    Args:
        ic_mean: IC均值
        ic_std: IC标准差
        default: 标准差无效且IC均值为0时的返回值

    Returns:
        IR值，或default（标准差无效且均值为0时）
    """
    # 处理NaN输入
    if ic_mean is None or (isinstance(ic_mean, float) and np.isnan(ic_mean)):
        return default
    if ic_std is None or (isinstance(ic_std, float) and np.isnan(ic_std)):
        return default

    # IC完全稳定：std=0但mean≠0 → IR趋向极大值
    if abs(ic_std) < 1e-10:
        if abs(ic_mean) < 1e-10:
            return default  # 均值和标准差都为0，无法判断
        # 用sign(mean) * abs(mean) * 1e6 作为极大IR的近似
        return float(np.sign(ic_mean)) * abs(ic_mean) * 1e6

    return safe_divide(ic_mean, ic_std, default=default)

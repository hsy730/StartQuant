"""
安全数学运算工具 — 统一除零/NaN保护

项目规则3：所有除法/标准差必须处理零值
所有需要除法保护的地方应使用此模块，禁止各自发明保护模式（如 +1e-10 hack）
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


def safe_float(val, default=0.0):
    """
    安全转换为 float — 处理 None/NaN/Inf 崩溃（规则7.36）

    当值为 None、NaN 或 Inf 时返回 default，避免 float(None) TypeError 和
    dict.get(key, default) 在键存在但值为 None 时返回 None 的陷阱。

    Args:
        val: 待转换的值（任意类型）
        default: 转换失败时的返回值（默认 0.0）

    Returns:
        float: 转换结果，或 default

    Examples:
        >>> safe_float(None, default=0.0)      # → 0.0
        >>> safe_float(3.14)                    # → 3.14
        >>> safe_float(float('nan'), default=None)  # → None
        >>> safe_float(float('inf'), default=0.0)   # → 0.0
    """
    if val is None:
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    # 同时处理 numpy 和 Python 标量的 NaN/Inf
    if isinstance(f, (float, np.floating)) and (np.isnan(f) or np.isinf(f)):
        return default
    return f


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

        if isinstance(denominator, pd.Series):
            safe_denominator = denominator.copy()
            safe_denominator[invalid] = 1.0
            result = numerator / safe_denominator
            result = result.mask(invalid, default)
        else:
            # numpy数组：先替换无效分母为1（避免RuntimeWarning），再除法，最后覆盖无效位置
            safe_denominator = denominator.copy()
            safe_denominator[invalid] = 1.0
            result = numerator / safe_denominator
            result[invalid] = np.nan if default is None else default
        return result
    else:
        # 标量处理
        # 注意：np.float64(np.nan) 在 numpy>=2.0 中不是 float 子类，
        # 必须同时检查 np.floating 类型才能捕获 NaN
        if denominator is None:
            return default
        if isinstance(denominator, (float, np.floating)) and np.isnan(denominator):
            return default
        if abs(denominator) < min_threshold:
            if isinstance(numerator, pd.Series):
                return pd.Series(np.nan if default is None else default, index=numerator.index, dtype=float)
            elif isinstance(numerator, np.ndarray):
                return np.full_like(numerator, np.nan if default is None else default, dtype=float)
            return default
        return numerator / denominator


def safe_ir(ic_mean: float, ic_std: float, default: Optional[float] = None) -> Optional[float]:
    """
    安全计算信息比率 IR = IC均值 / IC标准差

    统一处理IC标准差为0/NaN/极小值的情况，替代各处自行实现的IR计算。

    当IC标准差为0或极小值时，IR不可计算（常数序列通常意味着数据问题），
    此时返回default值，而非极大值。

    Args:
        ic_mean: IC均值
        ic_std: IC标准差
        default: 标准差无效时的返回值

    Returns:
        IR值，或default（标准差无效时）
    """
    # 处理NaN输入（np.float64(np.nan) 在 numpy>=2.0 中不是 float 子类）
    if ic_mean is None or (isinstance(ic_mean, (float, np.floating)) and np.isnan(ic_mean)):
        return default
    if ic_std is None or (isinstance(ic_std, (float, np.floating)) and np.isnan(ic_std)):
        return default

    # 标准差为0或极小值时，IR不可计算
    if abs(ic_std) < 1e-10:
        return default

    return safe_divide(ic_mean, ic_std, default=default)


def safe_series_divide(
    numerator: Union[pd.Series, np.ndarray],
    denominator: Union[pd.Series, np.ndarray],
    fill_value: float = np.nan,
) -> Union[pd.Series, np.ndarray]:
    """
    安全Series除法 — 分母零值/NaN处填充fill_value而非产生inf

    适用于因子代码字符串中的除法保护模式（替代 .replace(0, np.nan) hack）。
    与 safe_divide 的区别：本函数对零值分母填充 fill_value（默认NaN），
    而 safe_divide 对零值分母填充 default（默认None）。

    因子计算场景中，NaN 是最合适的"无效值"标记（后续 dropna 可自动移除），
    而 None 无法放入 Series。因此因子代码字符串中的除法应统一使用本函数。

    Args:
        numerator: 分子（Series或ndarray）
        denominator: 分母（Series或ndarray）
        fill_value: 分母无效时的填充值（默认NaN，因子计算推荐）

    Returns:
        除法结果，分母零值/NaN处为fill_value

    Examples:
        >>> s1 = pd.Series([10, 20, 30])
        >>> s2 = pd.Series([2, 0, 5])
        >>> safe_series_divide(s1, s2)
        0    5.0
        1    NaN   # 0分母 → NaN
        2    6.0
        dtype: float64

        >>> safe_series_divide(s1, s2, fill_value=0.0)
        0    5.0
        1    0.0   # 0分母 → 0.0
        2    6.0
        dtype: float64
    """
    if isinstance(denominator, pd.Series):
        invalid = (denominator.abs() < 1e-10) | denominator.isna()
        safe_denom = denominator.copy()
        safe_denom[invalid] = 1.0
        result = numerator / safe_denom
        result[invalid] = fill_value
        return result
    elif isinstance(denominator, np.ndarray):
        invalid = (np.abs(denominator) < 1e-10) | np.isnan(denominator)
        safe_denom = denominator.copy()
        safe_denom[invalid] = 1.0
        result = numerator / safe_denom
        result[invalid] = fill_value
        return result
    else:
        # 标量分母
        if (
            denominator is None
            or (isinstance(denominator, (float, np.floating)) and np.isnan(denominator))
            or abs(denominator) < 1e-10
        ):
            # 保持与分子相同的类型：Series分子→Series全fill_value，ndarray分子→ndarray全fill_value
            if isinstance(numerator, pd.Series):
                return pd.Series(fill_value, index=numerator.index, dtype=float)
            elif isinstance(numerator, np.ndarray):
                return np.full_like(numerator, fill_value, dtype=float)
            return fill_value
        return numerator / denominator

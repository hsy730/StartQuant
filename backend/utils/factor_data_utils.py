"""
因子数据公共工具 — 消除 factor_data 遍历/查找的重复代码

项目规范5：相同逻辑出现 ≥ 2 次必须提取公共方法
"""
from typing import Dict, Tuple, Generator, Optional
import pandas as pd


def find_longest_stock(
    factor_data: Dict[str, pd.DataFrame],
    factor_name: Optional[str] = None,
) -> Tuple[str, pd.DataFrame]:
    """
    找到因子数据最长的股票

    Args:
        factor_data: {stock_code: DataFrame} 字典
        factor_name: 可选，只考虑包含此因子列的股票

    Returns:
        (stock_code, DataFrame) 元组
    """
    if not factor_data:
        raise ValueError("factor_data 不能为空")

    longest_code = None
    max_len = 0
    for code, df in factor_data.items():
        if factor_name and factor_name not in df.columns:
            continue
        n = len(df[factor_name].dropna()) if factor_name else len(df)
        if n > max_len:
            max_len = n
            longest_code = code

    if longest_code is None:
        # Fallback: 取第一个
        longest_code = next(iter(factor_data))
    return longest_code, factor_data[longest_code]


def iter_valid_stocks(
    factor_data: Dict[str, pd.DataFrame],
    factor_name: str,
    required_cols: list = None,
    min_length: int = 1,
) -> Generator[Tuple[str, pd.DataFrame], None, None]:
    """
    遍历有效的股票数据（统一列检查和空数据过滤）

    Args:
        factor_data: {stock_code: DataFrame} 字典
        factor_name: 因子名称
        required_cols: 必须包含的列（默认 ["close", factor_name]）
        min_length: 最小数据长度

    Yields:
        (stock_code, DataFrame) 元组
    """
    if required_cols is None:
        required_cols = ["close", factor_name]

    for code, df in factor_data.items():
        # 检查必需列
        if not all(col in df.columns for col in required_cols):
            continue
        # 检查因子列非空数据量
        valid_count = df[factor_name].notna().sum()
        if valid_count < min_length:
            continue
        yield code, df

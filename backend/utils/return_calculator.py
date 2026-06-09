"""
收益率计算公共工具 — 统一未来收益率计算方式

项目规范5：未来收益率计算在15处各自实现，提取统一入口
"""
import pandas as pd


def calculate_future_return(
    df: pd.DataFrame,
    price_column: str = "close",
    period: int = 1,
) -> pd.Series:
    """
    计算未来N期收益率（标准语义：t日决策，t+N日收益）

    等价于 df[price_column].pct_change(period).shift(-period)
    这是回测的标准语义 — 信号在先，收益在后，不是前视偏差。

    Args:
        df: 包含价格列的DataFrame
        price_column: 价格列名
        period: 向前看几期

    Returns:
        未来收益率Series
    """
    prices = df[price_column]

    # MultiIndex (date, asset) 下需按资产分组计算，否则 pct_change 会跨资产比较
    if isinstance(df.index, pd.MultiIndex):
        return prices.groupby(level=1).apply(
            lambda s: s.pct_change(period).shift(-period)
        ).droplevel(0)

    return prices.pct_change(period).shift(-period)

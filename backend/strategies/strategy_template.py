"""
策略开发模板 - 包含量化计算防坑规范的注释检查清单

使用方法：
1. 复制此文件并重命名为你的策略类名
2. 填充 TODO 部分
3. 开发过程中逐项检查下方的防坑规则
4. 提交前确认所有检查项已勾选
"""

import pandas as pd
from backend.strategies.base_strategy import BaseStrategy


class YourStrategyName(BaseStrategy):
    """
    TODO: 策略描述

    策略逻辑：
    - TODO: 描述信号生成逻辑
    - TODO: 描述权重计算逻辑

    参数说明：
    - TODO: 列出所有参数及其含义
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: 初始化策略参数
        self.param_example = kwargs.get("param_example", 10)

    # ============================================================
    # 防坑规则检查清单 - 信号生成阶段
    # ============================================================

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        防坑检查项（信号生成阶段）：
        [ ] 规则2 - 前视偏差检查：
            - 确认没有使用 shift(-1) / shift(1) 来获取未来价格
            - 确认没有使用 df.loc[t+1] 等未来数据
            - 确认滚动窗口（rolling）没有包含未来数据

        [ ] 规则3 - 除零保护检查：
            - 所有除法运算是否处理了分母为零的情况？
            - pct_change() + std() 组合是否考虑了价格不变的情况？
            - 滚动窗口计算是否考虑了窗口内数据恒定的情况？

        [ ] 规则4 - 开源库API检查：
            - 多返回值函数是否用命名变量接收？（禁止用索引取值）
            - 是否查文档确认了返回值顺序？

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            信号序列（1=买入, -1=卖出, 0=持仓不变）
        """
        signals = pd.Series(0, index=df.index)

        # TODO: 实现信号生成逻辑
        # 示例：简单动量信号（注意：这只是示例，实际策略需要更复杂的逻辑）
        # momentum = df["close"].pct_change(self.param_example)
        # signals = (momentum > 0).astype(int) * 2 - 1

        return signals.fillna(0)

    # ============================================================
    # 防坑规则检查清单 - 权重计算阶段
    # ============================================================

    def calculate_weights(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        计算持仓权重

        防坑检查项（权重计算阶段）：
        [ ] 规则1 - 量纲一致性检查：
            - 权重是否为比例值（0~1之间或-1~1之间）？
            - 权重变化量（diff）是否与收益率同量纲？
            - 不要在权重计算中引入绝对金额（如 initial_capital）

        [ ] 规则3 - 除零保护检查：
            - 权重归一化时是否处理了总和为零的情况？
            - 权重计算中是否有除法运算需要保护？

        [ ] 规则2 - 前视偏差检查：
            - 权重计算是否只使用了当前及历史数据？
            - 是否有隐式的前视（如使用未来均值/标准差）？

        Args:
            df: 包含OHLCV数据的DataFrame
            signals: generate_signals 生成的信号序列

        Returns:
            权重序列（比例值，如 0.5 表示50%仓位）
        """
        weights = signals.astype(float)

        # TODO: 实现权重计算逻辑
        # 示例：等权重（注意：这只是示例，实际策略需要更复杂的逻辑）
        # weights = signals * 0.5  # 半仓操作

        return weights

    # ============================================================
    # 防坑规则检查清单 - 回测验证阶段
    # ============================================================

    def backtest(self, df: pd.DataFrame) -> dict:
        """
        执行回测

        防坑检查项（回测验证阶段）：
        [ ] 规则1 - 量纲一致性检查（最重要）：
            - portfolio_returns 是否为比例收益率？
            - commission 是否为比例值（不是绝对金额）？
            - commission 与 portfolio_returns 相减时量纲是否一致？
            - 验证：commission = weight_change * commission_rate（不含 initial_capital）
            - 验证：equity = (1 + portfolio_returns).cumprod() * initial_capital（只在最后转换）

        [ ] 规则2 - 前视偏差检查：
            - next_return 是否使用 shift(-1) 获取次日收益？（这是正确的）
            - 确认信号生成没有使用未来数据
            - 确认权重计算没有使用未来数据

        [ ] 规则3 - 边界条件检查：
            - 空数据（df为空）是否处理了？
            - 最小数据量（如只有1行）是否处理了？
            - 价格不变（pct_change全为0）是否处理了？
            - 全NaN数据是否处理了？

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            回测结果字典
        """
        # 调用父类回测方法（已包含标准手续费计算逻辑）
        # 注意：如果你重写了 backtest 方法，必须确保手续费计算符合规则1
        return super().backtest(df)

    # ============================================================
    # 策略信息
    # ============================================================

    def get_name(self) -> str:
        """返回策略名称"""
        return "YourStrategyName"

    def get_description(self) -> str:
        """返回策略描述"""
        return "TODO: 填写策略描述"


# ============================================================
# 快速参考卡 - 防坑规则速查
# ============================================================

"""
【规则1：量纲一致性】比例与绝对金额不可混用

✅ 正确模式：
    commission = weight_change * commission_rate  # 比例
    net_return = portfolio_return - commission     # 比例 - 比例 = 比例
    equity = (1 + net_return).cumprod() * initial_capital  # 最后转换

❌ 错误模式：
    commission = weight_change * initial_capital * commission_rate  # 绝对金额！
    net_return = portfolio_return - commission  # 比例 - 金额 = 爆炸

【规则2：前视偏差判断】shift方向不等于前视偏差

✅ 合法：shift(-1) 用于获取未来收益（结果变量）
    df["next_return"] = df["close"].pct_change().shift(-1)

❌ 非法：shift(-1) 用于生成交易信号（决策变量）
    signal = df["close"].shift(-1) > df["close"]  # 用了明天价格！

【规则3：除零保护】所有除法/标准差必须处理零值

模式A - 替换为NaN：
    result = numerator / denominator.replace(0, np.nan)

模式B - 替换为极小值：
    result = numerator / denominator.clip(lower=1e-10)

模式C - 条件回退：
    if (denominator == 0).all():
        return fallback_value
    result = numerator / denominator

【规则4：开源库API返回值】必须查文档确认索引

✅ 正确：
    upper, middle, lower = talib.BBANDS(close)
    result = middle  # 命名变量

❌ 错误：
    result = talib.BBANDS(close)[2]  # 凭直觉猜索引
"""

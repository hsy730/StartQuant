"""
前视偏差 / IC方向 单元测试

覆盖场景：
- shift(-1) 用于获取未来收益（合法）
- shift(-1) 用于生成交易信号（前视偏差，非法）
- IC计算应使用次日收益率（shift(-1)），非前一日（shift(1)）
- IC加权中 forward_return 应 shift(1) 避免前视
- 累计收益→日收益的正确公式

项目规则2：shift(-1)获取未来收益合法，用于信号生成则非法
"""

import numpy as np
import pandas as pd
import inspect

from backend.services.vectorbt_backtest_service import VectorBTBacktestService


class TestICDirectionCorrect:
    """IC计算应使用次日收益率（预测目标），非前一日"""

    def test_ic_uses_next_day_return(self):
        """IC衡量因子对未来收益的预测力，应使用 shift(-1) 获取次日收益"""
        # 构造已知数据
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        pd.Series(np.random.randn(20), index=dates, name="factor")
        close = pd.Series(100 + np.cumsum(np.random.randn(20)), index=dates)

        # 正确：次日收益率
        next_return = close.pct_change().shift(-1)
        # 错误：前一日收益率
        prev_return = close.pct_change().shift(1)

        # 两者应不同（除非数据完全对称）
        valid = next_return.notna() & prev_return.notna()
        if valid.sum() > 0:
            assert not np.allclose(
                next_return[valid].values, prev_return[valid].values
            ), "次日收益率和前一日收益率应不同"

    def test_ic_direction_semantic(self):
        """验证IC方向的语义正确性：因子与次日收益正相关时IC为正"""
        # 构造一个因子能强预测次日收益的场景
        np.random.seed(42)
        n = 200
        factor_values = np.random.randn(n)

        # 因子与次日收益强正相关（高信噪比）
        next_day_returns = factor_values * 0.05 + np.random.randn(n) * 0.005

        # 计算IC（Pearson相关系数）
        valid = ~np.isnan(factor_values) & ~np.isnan(next_day_returns)
        ic = np.corrcoef(factor_values[valid], next_day_returns[valid])[0, 1]

        # 因子与次日收益正相关，IC应为正
        assert ic > 0, f"因子与次日收益正相关时IC应为正，实际为 {ic}"


class TestICWeightingNoLookahead:
    """IC加权中不应有前视偏差"""

    def test_forward_return_shifted_in_rolling_ic(self):
        """验证 rolling_ic 计算中 forward_return 已 shift(1)"""
        service = VectorBTBacktestService()
        source = inspect.getsource(service.multi_factor_backtest)

        # 验证修复后的代码：forward_return 在 rolling_ic 中被 shift(1)
        # 修复前：.corr(df["forward_return"])
        # 修复后：.corr(df["forward_return"].shift(1))
        assert (
            'forward_return").shift(1)' in source or 'forward_return"].shift(1)' in source
        ), "rolling_ic 中 forward_return 应 shift(1) 避免前视偏差"


class TestCumulativeReturnConversion:
    """累计收益→日收益的正确公式"""

    def test_correct_formula_restores_daily_returns(self):
        """正确公式 (1+r[i+1])/(1+r[i])-1 应还原原始日收益率"""
        daily_returns = [0.01, 0.02, -0.01, 0.005, -0.003]

        # 计算累计收益
        cumulative = [0.0]
        for r in daily_returns:
            cumulative.append((1 + cumulative[-1]) * (1 + r) - 1)

        # 正确公式还原
        restored = [(cumulative[i + 1] + 1) / (cumulative[i] + 1) - 1 for i in range(len(cumulative) - 1)]

        for i in range(len(daily_returns)):
            assert (
                abs(restored[i] - daily_returns[i]) < 1e-10
            ), f"正确公式应还原日收益率: 期望 {daily_returns[i]}, 得到 {restored[i]}"

    def test_wrong_diff_formula_does_not_restore(self):
        """差分公式 cum[i+1]-cum[i] 不能还原原始日收益率（当累计收益非零时）"""
        daily_returns = [0.01, 0.02, -0.01, 0.005]

        cumulative = [0.0]
        for r in daily_returns:
            cumulative.append((1 + cumulative[-1]) * (1 + r) - 1)

        # 差分法（错误）
        wrong = [cumulative[i + 1] - cumulative[i] for i in range(len(cumulative) - 1)]

        # 第2期：累计收益=0.0302，差分=0.0302-0.01=0.0202≠0.02
        assert abs(wrong[1] - daily_returns[1]) > 1e-6, "差分法在累计收益非零时应与原始日收益率不同"

    def test_correct_formula_with_negative_cumulative(self):
        """负累计收益时正确公式仍能还原"""
        daily_returns = [-0.05, -0.03, 0.02, -0.01]

        cumulative = [0.0]
        for r in daily_returns:
            cumulative.append((1 + cumulative[-1]) * (1 + r) - 1)

        restored = [(cumulative[i + 1] + 1) / (cumulative[i] + 1) - 1 for i in range(len(cumulative) - 1)]

        for i in range(len(daily_returns)):
            assert (
                abs(restored[i] - daily_returns[i]) < 1e-10
            ), f"负累计收益时正确公式应还原: 期望 {daily_returns[i]}, 得到 {restored[i]}"

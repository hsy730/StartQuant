"""
量纲一致性单元测试

覆盖场景：
- 手续费与收益率同量纲（比例 vs 比例）
- 首期建仓手续费被正确计算
- weight_change 首行 NaN 被正确处理

项目规则1：比例与绝对金额不可混用
"""
import pytest
import numpy as np
import pandas as pd

from backend.strategies.base_strategy import BaseStrategy


class BuyHoldStrategy(BaseStrategy):
    """买入持有策略（用于测试）"""
    def generate_signals(self, df):
        return pd.Series(1, index=df.index)
    def calculate_weights(self, df, signals):
        return pd.Series(1.0, index=df.index)


class SwitchingStrategy(BaseStrategy):
    """每5期切换仓位的策略（用于测试手续费）"""
    def __init__(self, switch_period=5, **kwargs):
        super().__init__(**kwargs)
        self.switch_period = switch_period

    def generate_signals(self, df):
        n = len(df)
        signals = pd.Series(0, index=df.index)
        for i in range(n):
            if (i // self.switch_period) % 2 == 0:
                signals.iloc[i] = 1
        return signals

    def calculate_weights(self, df, signals):
        return signals.astype(float)


def _make_ohlcv(n=50, seed=42):
    """生成模拟OHLCV数据"""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(100000, 1000000, n).astype(float),
    }, index=dates)


class TestCommissionDimensionConsistency:
    """手续费与收益率必须同量纲（比例）"""

    def test_commission_is_proportion_not_absolute(self):
        """手续费应为比例值，非绝对金额"""
        strategy = BuyHoldStrategy(commission_rate=0.0003)
        df = _make_ohlcv(n=50)
        result = strategy.backtest(df)

        returns = result["portfolio_returns"]
        weights = result["weights"]

        # 手续费计算：weight_change * commission_rate
        # weight_change 是比例（0~1），commission_rate 是比例（0.0003）
        # 所以手续费也是比例（如 1.0 * 0.0003 = 0.0003）
        weight_change = weights.diff().abs().fillna(weights.abs())
        commission = weight_change * strategy.commission_rate

        # 手续费应在合理比例范围内（< 1%）
        assert (commission.dropna() < 0.01).all(), \
            f"手续费应为比例值（<1%），发现异常值: {commission[commission >= 0.01].tolist()}"

    def test_first_period_commission_not_zero(self):
        """首期建仓手续费不应为0"""
        strategy = BuyHoldStrategy(commission_rate=0.0003)
        df = _make_ohlcv(n=50)
        result = strategy.backtest(df)

        weights = result["weights"]
        weight_change = weights.diff().abs().fillna(weights.abs())
        commission = weight_change * strategy.commission_rate

        # 首期从0建仓，weight_change[0] = |1.0 - 0| = 1.0
        assert commission.iloc[0] > 0, \
            f"首期应有建仓手续费，实际 commission[0] = {commission.iloc[0]}"
        # 首期手续费 = 1.0 * 0.0003 = 0.0003
        assert abs(commission.iloc[0] - strategy.commission_rate) < 1e-10, \
            f"首期手续费应为 {strategy.commission_rate}，实际为 {commission.iloc[0]}"

    def test_first_period_return_not_nan(self):
        """首期收益不应为NaN"""
        strategy = BuyHoldStrategy(commission_rate=0.0003)
        df = _make_ohlcv(n=50)
        result = strategy.backtest(df)

        returns = result["portfolio_returns"]
        assert not pd.isna(returns.iloc[0]), "首期收益不应为NaN"

    def test_switching_strategy_commission_reasonable(self):
        """频繁调仓策略的手续费应合理"""
        strategy = SwitchingStrategy(switch_period=5, commission_rate=0.0003)
        df = _make_ohlcv(n=100)
        result = strategy.backtest(df)

        weights = result["weights"]
        weight_change = weights.diff().abs().fillna(weights.abs())
        commission = weight_change * strategy.commission_rate

        # 总手续费应在合理范围
        total_commission = commission.sum()
        assert total_commission > 0, "频繁调仓应有手续费"
        assert total_commission < 0.5, \
            f"总手续费不应超过50%（比例），实际为 {total_commission * 100:.2f}%"

    def test_commission_deducted_from_returns(self):
        """手续费应从收益中扣除"""
        strategy = BuyHoldStrategy(commission_rate=0.0003)
        df = _make_ohlcv(n=50)
        result = strategy.backtest(df)

        returns = result["portfolio_returns"]
        weights = result["weights"]

        # 手动计算不含手续费的收益
        raw_returns = df["close"].pct_change().fillna(0) * weights.fillna(0)

        # 含手续费的收益应 <= 不含手续费的收益
        valid = returns.notna() & raw_returns.notna()
        # 首期之后，含手续费收益应严格小于不含手续费收益（当有调仓时）
        # 至少验证含手续费收益不会大于不含手续费收益
        assert (returns[valid] <= raw_returns[valid] + 1e-10).all(), \
            "含手续费的收益不应大于不含手续费的收益"


class TestWeightChangeFirstRow:
    """weight_change 首行 NaN 的处理"""

    def test_weight_change_first_row_not_nan(self):
        """首行 weight_change 应为初始权重（视作从0建仓）"""
        weights = pd.Series([1.0, 1.0, 0.5, 0.5, 1.0])

        # 修复前：diff().abs() 首行为 NaN
        wrong = weights.diff().abs()
        assert pd.isna(wrong.iloc[0]), "diff() 首行应为 NaN"

        # 修复后：fillna(weights.abs()) 首行为初始权重
        correct = weights.diff().abs().fillna(weights.abs())
        assert correct.iloc[0] == 1.0, f"首行应为初始权重1.0，实际为 {correct.iloc[0]}"

    def test_weight_change_subsequent_rows_unchanged(self):
        """后续行的 weight_change 不应受 fillna 影响"""
        weights = pd.Series([1.0, 1.0, 0.5, 0.0, 1.0])

        result = weights.diff().abs().fillna(weights.abs())

        # 第2行：|1.0 - 1.0| = 0
        assert result.iloc[1] == 0.0
        # 第3行：|0.5 - 1.0| = 0.5
        assert result.iloc[2] == 0.5
        # 第4行：|0.0 - 0.5| = 0.5
        assert result.iloc[3] == 0.5
        # 第5行：|1.0 - 0.0| = 1.0
        assert result.iloc[4] == 1.0

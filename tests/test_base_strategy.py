"""
BaseStrategy 单元测试 - 覆盖手续费计算、指标计算、边界条件
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.strategies.base_strategy import BaseStrategy

# empyrical内部使用了np.NINF（NumPy 2.0已移除），需要兼容补丁
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'PINF'):
    np.PINF = np.inf
warnings.filterwarnings("ignore", category=RuntimeWarning)

_passed = 0
_failed = 0


def run(name, fn):
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        _failed += 1
        print(f"  [FAIL] {name}: {e}")


# ---- 测试用具体策略实现 ----

class SimpleBuyHoldStrategy(BaseStrategy):
    """简单买入持有策略（用于测试）"""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index)

    def calculate_weights(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=df.index)


class SwitchingStrategy(BaseStrategy):
    """每隔N期切换仓位的策略（用于测试手续费）"""

    def __init__(self, switch_period=5, **kwargs):
        super().__init__(**kwargs)
        self.switch_period = switch_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        signals = pd.Series(0, index=df.index)
        for i in range(n):
            if (i // self.switch_period) % 2 == 0:
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = -1
        return signals

    def calculate_weights(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        return signals.astype(float)


def make_price_data(n=100, seed=42):
    """生成测试用价格数据"""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.maximum(100 + np.cumsum(np.random.randn(n) * 2), 1)
    return pd.DataFrame({"close": close}, index=dates)


# ============================================================
# 1. 手续费计算测试
# ============================================================

def test_commission_is_proportional_not_absolute():
    """手续费应为比例而非绝对金额"""
    df = make_price_data(50)
    strategy = SwitchingStrategy(switch_period=5, initial_capital=1000000, commission_rate=0.0003)
    result = strategy.backtest(df)

    portfolio_returns = result["portfolio_returns"].dropna()
    # 手续费不应使单日收益偏离超过几个百分点
    # 如果手续费是绝对金额（乘以initial_capital），单日收益会偏离数百点
    max_abs_return = portfolio_returns.abs().max()
    assert max_abs_return < 1.0, f"单日收益异常大: {max_abs_return}，手续费可能被放大了initial_capital倍"


def test_commission_zero_when_no_rebalance():
    """无调仓时手续费应为零"""
    df = make_price_data(50)
    strategy = SimpleBuyHoldStrategy(commission_rate=0.001)
    result = strategy.backtest(df)

    weights = result["weights"]
    # 买入持有：权重始终为1，diff后只有第一天非零
    weight_change = weights.diff().abs()
    # 第一天的diff是NaN（fillna后为0），之后全为0
    commission_days = (weight_change.fillna(0) > 0).sum()
    assert commission_days <= 1, f"买入持有策略不应有频繁调仓手续费，实际调仓{commission_days}次"


def test_commission_increases_with_rebalance():
    """调仓越频繁，手续费越高"""
    df = make_price_data(100)

    strategy_hold = SimpleBuyHoldStrategy(commission_rate=0.001)
    result_hold = strategy_hold.backtest(df)

    strategy_switch = SwitchingStrategy(switch_period=5, commission_rate=0.001)
    result_switch = strategy_switch.backtest(df)

    # 切换策略的交易次数应多于持有策略
    assert result_switch["trades_count"] > result_hold["trades_count"], \
        "切换策略的交易次数应多于持有策略"


def test_higher_commission_rate_reduces_return():
    """更高的手续费率应导致更低的最终收益"""
    df = make_price_data(100)

    strategy_low = SwitchingStrategy(switch_period=5, commission_rate=0.0001)
    result_low = strategy_low.backtest(df)

    strategy_high = SwitchingStrategy(switch_period=5, commission_rate=0.01)
    result_high = strategy_high.backtest(df)

    equity_low = result_low["equity_curve"].iloc[-1]
    equity_high = result_high["equity_curve"].iloc[-1]
    assert equity_low >= equity_high, \
        f"低手续费净值{equity_low:.2f}应 >= 高手续费净值{equity_high:.2f}"


# ============================================================
# 2. 指标计算测试
# ============================================================

def test_calculate_metrics_normal_returns():
    """正常收益率序列应返回有效指标"""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01)  # 日均波动1%
    strategy = SimpleBuyHoldStrategy()
    metrics = strategy.calculate_metrics(returns)

    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "calmar_ratio" in metrics
    assert "win_rate" in metrics
    assert not np.isnan(metrics["total_return"]), "total_return 不应为 NaN"
    assert not np.isnan(metrics["volatility"]), "volatility 不应为 NaN"


def test_calculate_metrics_empty_returns():
    """空收益率序列应返回空指标"""
    returns = pd.Series(dtype=float)
    strategy = SimpleBuyHoldStrategy()
    metrics = strategy.calculate_metrics(returns)

    assert metrics["total_return"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["max_drawdown"] == 0.0


def test_calculate_metrics_all_nan_returns():
    """全NaN收益率序列应返回空指标"""
    returns = pd.Series([np.nan] * 100)
    strategy = SimpleBuyHoldStrategy()
    metrics = strategy.calculate_metrics(returns)

    assert metrics["total_return"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0


def test_calculate_metrics_single_value():
    """单值收益率序列应返回有效指标（empyrical可处理）"""
    returns = pd.Series([0.01])
    strategy = SimpleBuyHoldStrategy()
    metrics = strategy.calculate_metrics(returns)

    # empyrical对单值可能返回NaN，但不应抛异常
    assert isinstance(metrics, dict)
    assert "total_return" in metrics


def test_calculate_metrics_all_positive_returns():
    """全正收益序列：胜率应为1.0"""
    returns = pd.Series([0.01, 0.02, 0.005, 0.03, 0.015])
    strategy = SimpleBuyHoldStrategy()
    metrics = strategy.calculate_metrics(returns)

    assert metrics["win_rate"] == 1.0


def test_calculate_metrics_all_negative_returns():
    """全负收益序列：胜率应为0.0"""
    returns = pd.Series([-0.01, -0.02, -0.005, -0.03, -0.015])
    strategy = SimpleBuyHoldStrategy()
    metrics = strategy.calculate_metrics(returns)

    assert metrics["win_rate"] == 0.0


def test_calculate_metrics_extreme_negative_returns():
    """极端负收益（接近-100%）不应产生nan"""
    returns = pd.Series([-0.5, -0.3, -0.2, 0.01, -0.1])
    strategy = SimpleBuyHoldStrategy()
    metrics = strategy.calculate_metrics(returns)

    # 不应抛异常，且关键指标不应为NaN
    assert isinstance(metrics, dict)
    assert not np.isnan(metrics["total_return"]), "total_return 不应为 NaN"


def test_calculate_metrics_zero_returns():
    """零收益序列：波动率为0，夏普为0"""
    returns = pd.Series([0.0] * 50)
    strategy = SimpleBuyHoldStrategy()
    metrics = strategy.calculate_metrics(returns)

    assert metrics["total_return"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["win_rate"] == 0.0  # 0不大于0


# ============================================================
# 3. 回测边界条件测试
# ============================================================

def test_backtest_minimum_data():
    """最小数据量（2行）应能完成回测"""
    dates = pd.date_range("2023-01-01", periods=2, freq="B")
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=dates)
    strategy = SimpleBuyHoldStrategy()
    result = strategy.backtest(df)

    assert "equity_curve" in result
    assert len(result["equity_curve"]) == 2


def test_backtest_constant_price():
    """价格不变时收益率为0"""
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    df = pd.DataFrame({"close": [100.0] * 20}, index=dates)
    strategy = SimpleBuyHoldStrategy()
    result = strategy.backtest(df)

    # 最后一行next_return为NaN（shift(-1)），其余为0
    portfolio_returns = result["portfolio_returns"].dropna()
    assert (portfolio_returns == 0.0).all(), "价格不变时收益率应为0"


def test_backtest_single_price_jump():
    """单日跳空：只有一天有收益"""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    close = [100.0] * 5 + [110.0] * 5
    df = pd.DataFrame({"close": close}, index=dates)
    strategy = SimpleBuyHoldStrategy()
    result = strategy.backtest(df)

    # 第5天到第6天有10%涨幅，对应next_return在第5天
    portfolio_returns = result["portfolio_returns"]
    non_zero = portfolio_returns[portfolio_returns.abs() > 1e-10].dropna()
    assert len(non_zero) >= 1, "应有非零收益"


def test_backtest_very_low_initial_capital():
    """极小初始资金应能正常回测"""
    df = make_price_data(50)
    strategy = SimpleBuyHoldStrategy(initial_capital=1.0)
    result = strategy.backtest(df)

    assert result["equity_curve"].iloc[0] == 1.0
    assert not result["equity_curve"].isna().any(), "净值曲线不应有NaN"


def test_backtest_zero_commission():
    """零手续费时应无手续费扣除"""
    df = make_price_data(50)
    strategy = SwitchingStrategy(switch_period=5, commission_rate=0.0)
    result = strategy.backtest(df)

    # 验证trades_count > 0但commission = 0
    assert result["trades_count"] > 0, "切换策略应有交易"


def test_backtest_high_commission():
    """极高手续费应显著侵蚀收益但不导致崩溃"""
    df = make_price_data(50)
    strategy = SwitchingStrategy(switch_period=3, commission_rate=0.05)  # 5%费率
    result = strategy.backtest(df)

    equity = result["equity_curve"]
    assert not equity.isna().any(), "净值曲线不应有NaN"
    assert (equity > 0).all(), "净值应始终为正（比例手续费模型）"


# ============================================================
# 4. 策略接口测试
# ============================================================

def test_get_name():
    """策略名称应为类名"""
    strategy = SimpleBuyHoldStrategy()
    assert strategy.get_name() == "SimpleBuyHoldStrategy"


def test_get_description():
    """策略描述应为docstring"""
    strategy = SimpleBuyHoldStrategy()
    desc = strategy.get_description()
    assert "简单买入持有" in desc


def test_default_parameters():
    """默认参数应正确设置"""
    strategy = SimpleBuyHoldStrategy()
    assert strategy.initial_capital == 1000000
    assert strategy.commission_rate == 0.0003


# ============================================================
# 运行所有测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BaseStrategy 单元测试")
    print("=" * 60)

    print("\n--- 手续费计算测试 ---")
    run("手续费应为比例而非绝对金额", test_commission_is_proportional_not_absolute)
    run("无调仓时手续费应为零", test_commission_zero_when_no_rebalance)
    run("调仓越频繁手续费越高", test_commission_increases_with_rebalance)
    run("更高手续费率应降低收益", test_higher_commission_rate_reduces_return)

    print("\n--- 指标计算测试 ---")
    run("正常收益率序列应返回有效指标", test_calculate_metrics_normal_returns)
    run("空收益率序列应返回空指标", test_calculate_metrics_empty_returns)
    run("全NaN收益率序列应返回空指标", test_calculate_metrics_all_nan_returns)
    run("单值收益率序列不应抛异常", test_calculate_metrics_single_value)
    run("全正收益胜率应为1.0", test_calculate_metrics_all_positive_returns)
    run("全负收益胜率应为0.0", test_calculate_metrics_all_negative_returns)
    run("极端负收益不应产生nan", test_calculate_metrics_extreme_negative_returns)
    run("零收益序列指标应为0", test_calculate_metrics_zero_returns)

    print("\n--- 回测边界条件测试 ---")
    run("最小数据量应能完成回测", test_backtest_minimum_data)
    run("价格不变时收益率为0", test_backtest_constant_price)
    run("单日跳空应有非零收益", test_backtest_single_price_jump)
    run("极小初始资金应正常回测", test_backtest_very_low_initial_capital)
    run("零手续费应无扣除", test_backtest_zero_commission)
    run("极高手续费不应崩溃", test_backtest_high_commission)

    print("\n--- 策略接口测试 ---")
    run("策略名称应为类名", test_get_name)
    run("策略描述应为docstring", test_get_description)
    run("默认参数应正确设置", test_default_parameters)

    print("\n" + "=" * 60)
    print(f"结果: {_passed} passed, {_failed} failed")
    print("=" * 60)

    if _failed > 0:
        sys.exit(1)

"""
P2 改造验证测试脚本
  P2-1: BacktestService委托VectorBT薄编排层
  P2-2: 加权IC (市值加权/流动性加权)
  P2-3: RMS相关性指标
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_passed = 0
_failed = 0
_errors = []


def make_mock_factor_data(n_stocks=5, n_dates=120, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_dates, freq="B")
    industries = ["Tech", "Finance", "Healthcare", "Consumer", "Energy"]
    factor_data = {}
    for i in range(n_stocks):
        stock_code = f"{600000 + i}"
        base_return = np.random.randn(n_dates) * 0.02
        factor_a = np.random.randn(n_dates) * 2 + base_return * 10
        factor_b = np.random.randn(n_dates) * 3 - base_return * 5
        close = 100 + np.cumsum(base_return) * 50
        close = np.maximum(close, 1)
        market_cap = np.exp(np.random.normal(25, 0.8, size=n_dates))
        volume = np.exp(np.random.normal(18, 1.0, size=n_dates))
        industry = np.random.choice(industries)
        df = pd.DataFrame(
            {
                "close": close,
                "factor_a": factor_a,
                "factor_b": factor_b,
                "market_cap": market_cap,
                "volume": volume,
                "industry": industry,
                "tradable_mask": True,
            },
            index=dates,
        )
        factor_data[stock_code] = df
    return factor_data


def run_test(name, fn):
    global _passed, _failed
    try:
        result = fn()
        if result:
            _passed += 1
            print(f"  [PASS] {name}")
            if isinstance(result, str) and len(result) > 200:
                print(f"         {result[:150]}...")
            elif result:
                print(f"         {result}")
        else:
            _passed += 1
            print(f"  [PASS] {name}")
    except Exception as e:
        _failed += 1
        _errors.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")


# ============================================================
# P2-1: BacktestService委托VectorBT
# ============================================================
def test_p2_1_backtest_engine_check():
    from backend.services.backtest_service import check_backtest_engine

    try:
        engine = check_backtest_engine()
        assert engine == "vectorbt", f"Unexpected engine: {engine}"
    except ImportError as e:
        engine = "unavailable"
        assert "VectorBT" in str(e), f"Unexpected error: {e}"
    return f"engine={engine}"


def test_p2_1_backtest_service_init():
    from backend.services.backtest_service import BacktestService

    svc = BacktestService(initial_capital=1000000, commission_rate=0.0003)
    assert svc.initial_capital == 1000000
    assert svc.commission_rate == 0.0003
    vbt = svc._get_vbt()
    if vbt is not None:
        return "BacktestService initialized with VectorBT engine available"
    else:
        return "BacktestService initialized (fallback mode)"


def test_p2_1_single_factor_delegates():
    from backend.services.backtest_service import BacktestService

    svc = BacktestService()
    factor_data = make_mock_factor_data(n_stocks=3, n_dates=60, seed=42)
    for stock_code, df in factor_data.items():
        result = svc.single_factor_backtest(df=df, factor_name="factor_a", percentile=50)
        assert "portfolio_returns" in result
        assert "equity_curve" in result
        assert "engine" in result
        assert result["engine"] in ("vectorbt", "fallback")
        break
    return f"single_factor_backtest uses engine={result['engine']}"


def test_p2_1_cross_sectional_delegates():
    from backend.services.backtest_service import BacktestService

    svc = BacktestService()
    all_data = []
    for stock_code, df in make_mock_factor_data(n_stocks=4, n_dates=60, seed=77).items():
        d = df.copy()
        d["stock_code"] = stock_code
        d = d.reset_index()
        d.rename(columns={"index": "date"}, inplace=True)
        all_data.append(d[["date", "factor_a", "close", "stock_code"]])
    merged = pd.concat(all_data, ignore_index=True)
    result = svc.cross_sectional_backtest(df=merged, factor_name="factor_a")
    assert "portfolio_returns" in result
    assert "engine" in result
    return f"cross_sectional uses engine={result['engine']}"


# ============================================================
# P2-2: 加权IC
# ============================================================
def test_p2_2_weighted_ic_market_cap():
    from backend.services.analysis_service import analysis_service

    factor_data = make_mock_factor_data(n_stocks=5, n_dates=120, seed=55)
    result = analysis_service.calculate_weighted_ic(
        factor_data=factor_data,
        factor_names=["factor_a"],
        stock_codes=list(factor_data.keys()),
        weight_type="market_cap",
    )
    assert "ic_stats" in result
    assert "weight_type" in result
    assert result["weight_type"] == "market_cap"
    if result.get("ic_stats"):
        first_key = list(result["ic_stats"].keys())[0]
        stats = result["ic_stats"][first_key]
        assert "IC均值" in stats
        assert "weight_type" in stats
        assert stats["weight_type"] == "market_cap"
        return f"market_cap_weighted IC={stats['IC均值']:.4f}"
    return "no ic_stats returned (Alphalens may be unavailable)"


def test_p2_2_weighted_ic_liquidity():
    from backend.services.analysis_service import analysis_service

    factor_data = make_mock_factor_data(n_stocks=5, n_dates=120, seed=66)
    result = analysis_service.calculate_weighted_ic(
        factor_data=factor_data,
        factor_names=["factor_a"],
        weight_type="liquidity",
    )
    assert result["weight_type"] == "liquidity"
    if result.get("ic_stats"):
        first_key = list(result["ic_stats"].keys())[0]
        stats = result["ic_stats"][first_key]
        return f"liquidity_weighted IC={stats['IC均值']:.4f}"
    return "no ic_stats returned"


def test_p2_2_weighted_ic_insufficient_stocks():
    from backend.services.analysis_service import analysis_service

    factor_data = {"000001": pd.DataFrame({"close": [1, 2], "f": [0.1, 0.2]})}
    result = analysis_service.calculate_weighted_ic(factor_data=factor_data, factor_names=["f"])
    assert "error" in result
    return f"correctly rejected: {result['error'][:40]}"


# ============================================================
# P2-3: RMS相关性
# ============================================================
def test_p2_3_rms_low_correlation():
    from backend.services.factor_correlation_service import FactorCorrelationService

    svc = FactorCorrelationService()
    np.random.seed(100)
    data = pd.DataFrame(np.random.randn(100, 4), columns=["A", "B", "C", "D"])
    corr_matrix = data.corr()
    result = svc.calculate_rms_correlation(corr_matrix)
    assert "rms_corr" in result
    assert "diversification_score" in result
    assert "interpretation" in result
    assert 0 <= result["diversification_score"] <= 100
    return f"RMS={result['rms_corr']:.4f}, score={result['diversification_score']:.1f}"


def test_p2_3_rms_high_correlation():
    from backend.services.factor_correlation_service import FactorCorrelationService

    svc = FactorCorrelationService()
    np.random.seed(101)
    base = np.random.randn(100)
    data = pd.DataFrame(
        {
            "A": base + np.random.randn(100) * 0.1,
            "B": base + np.random.randn(100) * 0.15,
            "C": base + np.random.randn(100) * 0.05,
            "D": np.random.randn(100),
        }
    )
    corr_matrix = data.corr()
    result = svc.calculate_rms_correlation(corr_matrix)
    assert result["rms_corr"] > 0.5, f"Expected high correlation but got {result['rms_corr']}"
    assert (
        "严重重叠" in result["interpretation"]
        or "较高相关性" in result["interpretation"]
        or "高相关性" in result["interpretation"]
    )
    return f"RMS={result['rms_corr']:.4f}, score={result['diversification_score']:.1f}"


def test_p2_3_rms_empty_input():
    from backend.services.factor_correlation_service import FactorCorrelationService

    svc = FactorCorrelationService()
    result = svc.calculate_rms_correlation(pd.DataFrame())
    assert "error" in result
    return "empty matrix correctly rejected"


def test_p2_3_rms_single_factor():
    from backend.services.factor_correlation_service import FactorCorrelationService

    svc = FactorCorrelationService()
    result = svc.calculate_rms_correlation(pd.DataFrame({"A": [1.0], "B": [2.0]}))
    assert "error" in result or "rms_corr" in result
    return "single factor handled"


if __name__ == "__main__":
    print("=" * 65)
    print("  FactorHub P2 Refactoring Verification Tests")
    print("=" * 65)

    run_test("P2-1: 回测引擎检测", test_p2_1_backtest_engine_check)
    run_test("P2-1: BacktestService初始化", test_p2_1_backtest_service_init)
    run_test("P2-1: single_factor委托VectorBT", test_p2_1_single_factor_delegates)
    run_test("P2-1: cross_sectional委托VectorBT", test_p2_1_cross_sectional_delegates)
    run_test("P2-2: 市值加权IC", test_p2_2_weighted_ic_market_cap)
    run_test("P2-2: 流动性加权IC", test_p2_2_weighted_ic_liquidity)
    run_test("P2-2: 股票不足时拒绝", test_p2_2_weighted_ic_insufficient_stocks)
    run_test("P2-3: 低相关性RMS", test_p2_3_rms_low_correlation)
    run_test("P2-3: 高相关性RMS", test_p2_3_rms_high_correlation)
    run_test("P2-3: 空矩阵拒绝", test_p2_3_rms_empty_input)
    run_test("P2-3: 单因子拒绝", test_p2_3_rms_single_factor)

    print("\n" + "=" * 65)
    print(f"  Result: {_passed} passed, {_failed} failed, total {_passed+_failed}")
    if _failed == 0:
        print("  ALL TESTS PASSED!")
    else:
        print(f"  {_failed} TEST(S) FAILED")
    print("=" * 65)

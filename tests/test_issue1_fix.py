"""
Issue1 修复验证: BacktestService委托VectorBT时传递use_tradable_mask参数
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_passed = 0
_failed = 0


def run(name, fn):
    global _passed, _failed
    try:
        r = fn()
        _passed += 1
        print(f"  [PASS] {name}")
        if r: print(f"         {r}")
    except Exception as e:
        _failed += 1
        print(f"  [FAIL] {name}: {e}")


def make_data(n=80, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.maximum(100 + np.cumsum(np.random.randn(n)*2), 1)
    factor = np.random.randn(n) * 3
    return pd.DataFrame({
        "close": close, "factor_x": factor,
        "tradable_mask": True,
        "is_limit_up": False, "is_limit_down": False,
    }, index=dates)


# Test 1: VectorBT路径接受use_tradable_mask且不报错
def test_vbt_accepts_mask():
    from backend.services.backtest_service import BacktestService
    svc = BacktestService(initial_capital=1e6)
    df = make_data(seed=42)

    r = svc.single_factor_backtest(df=df, factor_name="factor_x",
                                     percentile=50, use_tradable_mask=True)
    assert r["engine"] == "vectorbt"
    assert "mask_statistics" in r
    ms = r["mask_statistics"]
    assert ms["tradable_days"] == len(df)  # 全部可交易
    assert ms["mask_applied"] is True
    return f"engine={r['engine']}, mask_applied={ms['mask_applied']}, trades={r['trades_count']}"


# Test 2: VectorBT路径 use_tradable_mask=False时不应用mask
def test_vbt_no_mask():
    from backend.services.backtest_service import BacktestService
    svc = BacktestService(initial_capital=1e6)
    df = make_data(seed=43)

    r = svc.single_factor_backtest(df=df, factor_name="factor_x",
                                     percentile=50, use_tradable_mask=False)
    assert r["engine"] == "vectorbt"
    ms = r["mask_statistics"]
    assert ms["mask_applied"] is False
    return f"mask_applied={ms['mask_applied']}"


# Test 3: 有不可交易日时，VectorBT正确过滤
def test_vbt_mask_filters():
    from backend.services.backtest_service import BacktestService
    svc = BacktestService(initial_capital=1e6)
    df = make_data(n=100, seed=44)
    # 模拟20%不可交易（停牌/涨跌停）
    mask = pd.Series(True, index=df.index)
    non_trade_idx = df.index[::5]
    mask.loc[non_trade_idx] = False
    df["tradable_mask"] = mask

    r = svc.single_factor_backtest(df=df, factor_name="factor_x",
                                     percentile=50, use_tradable_mask=True)
    ms = r["mask_statistics"]
    assert ms["tradable_ratio"] < 1.0
    assert ms["total_days"] - ms["tradable_days"] == len(non_trade_idx)
    # 确保结果与fallback路径一致：都有mask_statistics
    assert "portfolio_returns" in r
    return f"ratio={ms['tradable_ratio']:.1%}, tradable={ms['tradable_days']}/{ms['total_days']}"


# Test 4: 参数签名一致性 — 两边都接受use_tradable_mask
def test_signature_consistency():
    import inspect
    from backend.services.backtest_service import BacktestService
    from backend.services.vectorbt_backtest_service import VectorBTBacktestService

    bs_sig = inspect.signature(BacktestService.single_factor_backtest)
    vbt_sig = inspect.signature(VectorBTBacktestService.single_factor_backtest)

    bs_params = list(bs_sig.parameters.keys())
    vbt_params = list(vbt_sig.parameters.keys())

    assert "use_tradable_mask" in bs_params, "BacktestService缺少use_tradable_mask"
    assert "use_tradable_mask" in vbt_params, "VectorBTBacktestService缺少use_tradable_mask"

    return f"BS params={bs_params}, VBT params={vbt_params}"


if __name__ == "__main__":
    print("=" * 60)
    print("  Issue1 Fix Verification: use_tradable_mask delegation")
    print("=" * 60)
    run("VBT接受use_tradable_mask=True", test_vbt_accepts_mask)
    run("VBT接受use_tradable_mask=False", test_vbt_no_mask)
    run("VBT正确过滤不可交易日", test_vbt_mask_filters)
    run("参数签名两边一致", test_signature_consistency)
    print("\n" + "=" * 60)
    print(f"  Result: {_passed}/{_passed+_failed} passed")
    print("=" * 60)
    sys.exit(0 if _failed == 0 else 1)

"""
API接口集成测试 - 验证P0/P1改造后的端到端功能
"""

import requests
import time

BASE = "http://localhost:8000"
passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        result = fn()
        if result:
            passed += 1
            print(f"  [PASS] {name}")
            if isinstance(result, str) and len(result) > 200:
                print(f"         {result[:150]}...")
            elif result:
                print(f"         {result}")
        else:
            passed += 1
            print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] {name}: {e}")


def test_health():
    r = requests.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200
    return f"status={r.json().get('status')}"


def test_factors_list():
    r = requests.get(f"{BASE}/api/factors/", timeout=15)
    assert r.status_code == 200
    data = r.json()
    assert data.get("success") is True
    total = data.get("total", 0)
    names = [f["name"] for f in data.get("data", [])[:3]]
    return f"total={total}, first3={names}"


def test_ic_api():
    body = {
        "factor_name": "distance_to_high_20",
        "stock_codes": ["000001", "000002", "600519"],
        "start_date": "2024-01-01",
        "end_date": "2024-06-30",
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/api/analysis/ic", json=body, timeout=120)
    elapsed = time.time() - t0
    assert r.status_code == 200
    data = r.json()
    assert data.get("success") is True

    ic_stats = data.get("data", {}).get("ic_stats", {})
    assert len(ic_stats) > 0, "ic_stats should not be empty"

    key = list(ic_stats.keys())[0]
    stat = ic_stats[key]

    parts = []
    if "IC均值" in stat:
        parts.append(f"IC={stat['IC均值']:.4f}")
    if "IR" in stat:
        parts.append(f"IR={stat['IR']:.4f}")
    if "IC类型" in stat:
        parts.append(f"type={stat['IC类型']}")
    if "t统计量" in stat:
        parts.append(f"t={stat['t统计量']:.2f}")
    if "p值" in stat:
        parts.append(f"p={stat['p值']:.4f}")
    parts.append(f"time={elapsed:.1f}s")

    return f"key={key}, {', '.join(parts)}"


def test_effectiveness_api():
    body = {
        "factor_name": "distance_to_high_20",
        "stock_codes": ["000001", "000002", "600519"],
        "start_date": "2024-03-01",
        "end_date": "2024-06-30",
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/api/analysis/effectiveness", json=body, timeout=120)
    elapsed = time.time() - t0
    assert r.status_code == 200
    data = r.json()
    assert data.get("success") is True

    eff_data = data.get("data", {})
    assert "scatter_plot" in eff_data
    assert "ic_time_series" in eff_data
    assert "decay_analysis" in eff_data
    assert "event_response" in eff_data

    ic_ts = eff_data.get("ic_time_series", {})
    source = ic_ts.get("source", "unknown")
    ic_mean = ic_ts.get("ic_mean")
    decay = eff_data.get("decay_analysis", {})
    decay_pts = decay.get("decay_curve", [])

    return f"IC_source={source}, IC_mean={ic_mean}, decay_points={len(decay_pts)}, time={elapsed:.1f}s"


def test_stability_api():
    body = {
        "factor_name": "distance_to_high_20",
        "stock_codes": ["000001", "600519"],
        "start_date": "2024-01-01",
        "end_date": "2024-06-30",
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/api/analysis/stability", json=body, timeout=60)
    elapsed = time.time() - t0
    assert r.status_code == 200
    data = r.json()
    assert data.get("success") is True

    stab_data = data.get("data", {})
    score = stab_data.get("overall_score")
    has_ic = "ic_analysis" in stab_data or "ic_validity" in stab_data

    return f"score={score}, has_ic_section={has_ic}, time={elapsed:.1f}s"


if __name__ == "__main__":
    print("=" * 65)
    print("  FactorHub API Integration Test - P0/P1 Refactoring Verification")
    print("=" * 65)

    run_test("Health check - 服务是否运行", test_health)
    run_test("GET /api/factors/ - 获取因子列表", test_factors_list)
    run_test("POST /api/analysis/ic - P0改造核心(AnalysisService IC/IR委托Alphalens)", test_ic_api)
    run_test("POST /api/analysis/effectiveness - P1改造(FactorEffectiveness委托Alphalens)", test_effectiveness_api)
    run_test("POST /api/analysis/stability - 稳定性分析（调用FactorStabilityService）", test_stability_api)

    print("\n" + "=" * 65)
    print(f"  Result: {passed} passed, {failed} failed, total {passed+failed}")
    if failed == 0:
        print("  ALL TESTS PASSED!")
    else:
        print(f"  {failed} TEST(S) FAILED")
    print("=" * 65)

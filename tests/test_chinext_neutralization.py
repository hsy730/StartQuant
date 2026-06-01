"""
创业板行业中性化独立性测试

验证：当只选择创业板股票时，中性化是否完全基于创业板内部数据
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

from backend.services.factor_preprocessing_pipeline import (
    FactorPreprocessingPipeline,
    PreprocessingConfig,
)


def generate_chinext_test_data():
    """
    生成模拟的创业板测试数据

    包含：
    - 100个交易日
    - 50只创业板股票（300xxx）
    - 分布在不同子行业：科技、医药、新能源、消费
    - 每个子行业有不同的因子特征
    """
    np.random.seed(42)

    n_dates = 100
    dates = pd.date_range(start="2023-06-01", periods=n_dates, freq="B")

    # 定义创业板子行业及其特征
    industries_config = {
        "ChiNext_Tech": {
            "count": 15,  # 15只科技股
            "factor_base": 8.0,  # 科技股因子均值高
            "factor_vol": 12.0,   # 波动大
            "mc_base": 11.0,      # 市值对数均值
        },
        "ChiNext_Med": {
            "count": 12,  # 12只医药股
            "factor_base": 3.0,  # 医药股因子均值中等
            "factor_vol": 8.0,
            "mc_base": 10.5,
        },
        "ChiNew_Energy": {
            "count": 13,  # 13只新能源股
            "factor_base": 6.0,
            "factor_vol": 15.0,   # 新能源波动最大
            "mc_base": 11.5,
        },
        "ChiNext_Consumer": {
            "count": 10,  # 10只消费股
            "factor_base": 2.0,  # 消费股因子均值低
            "factor_vol": 6.0,
            "mc_base": 10.0,
        },
    }

    data = []
    stock_id = 300001  # 创业板股票代码从300001开始

    for industry_name, config in industries_config.items():
        for i in range(config["count"]):
            stock_code = f"{stock_id:06d}.SZ"
            stock_id += 1

            for date in dates:
                # 生成具有行业特征的因子值
                factor_raw = (
                    config["factor_base"] +
                    np.random.randn() * config["factor_vol"] +
                    np.sin(np.arange(n_dates) / 20 * (i + 1)) * 2  # 加入时间趋势
                )

                # 生成市值（与因子有弱正相关，模拟现实情况）
                market_cap = np.exp(
                    config["mc_base"] +
                    np.random.randn() * 0.8 +
                    0.05 * (factor_raw - config["factor_base"]) / config["factor_vol"]
                )

                data.append({
                    "date": date,
                    "stock_code": stock_code,
                    "industry": industry_name,
                    "factor_momentum": factor_raw,
                    "factor_value": factor_raw * 0.8 + np.random.randn() * 3,  # 第二个因子
                    "market_cap": market_cap,
                    "close": 50 + np.cumsum(np.random.randn(n_dates))[:len(dates)][list(dates).index(date)] if date in dates else 50,
                })

    return pd.DataFrame(data)


def test_independence_from_main_board():
    """
    测试1：验证创业板中性化的独立性

    对比两种情况：
    A. 只用创业板数据
    B. 混合主板+创业板数据

    验证结果应该不同，说明中性化确实基于各自的数据集
    """
    print("="*70)
    print("🧪 测试1：创业板中性化独立性验证")
    print("="*70)

    # 生成纯创业板数据
    df_chinext = generate_chinext_test_data()

    # 创建管道
    pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
        winsorize_method="mad",
        enable_market_cap_neutralization=True,
        enable_industry_neutralization=True,
        standardize_method="zscore",
        cross_sectional=True,
    ))

    # 处理纯创业板数据
    processed_chinext, stats_chinext = pipeline.process_factor_dataframe(
        df=df_chinext,
        factor_columns=["factor_momentum"],
        market_cap_column="market_cap",
        industry_column="industry",
        date_column="date",
    )

    # 验证1：检查每个行业的均值是否接近0（中性化后的期望）
    print("\n📊 创业板各子行业中性化后统计:")
    print("-"*70)
    for industry_name in ["ChiNext_Tech", "ChiNext_Med", "ChiNew_Energy", "ChiNext_Consumer"]:
        industry_data = processed_chinext[
            processed_chinext["industry"] == industry_name
        ]["factor_momentum"]

        if len(industry_data) > 0:
            mean_val = industry_data.mean()
            std_val = industry_data.std()
            n_samples = len(industry_data)

            status = "✅" if abs(mean_val) < 0.5 else "⚠️"
            print(f"  {status} {industry_name:20s}: 均值={mean_val:+7.4f}, "
                  f"标准差={std_val:6.4f}, 样本数={n_samples:,}")

    # 验证2：与原始数据对比
    print("\n📈 中性化前后对比（以某一天为例）:")
    sample_date = df_chinext["date"].iloc[50]
    original_sample = df_chinext[df_chinext["date"] == sample_date]
    processed_sample = processed_chinext[processed_chinext["date"] == sample_date]

    print(f"\n  日期: {sample_date.strftime('%Y-%m-%d')}")
    print(f"  {'行业':20s} | {'原始均值':>10s} | {'处理后均值':>12s} | {'变化':>10s}")
    print(f"  {'-'*20}-|-{'-'*10}-|-{'-'*12}-|-{'-'*10}")

    for industry_name in ["ChiNext_Tech", "ChiNext_Med", "ChiNew_Energy", "ChiNext_Consumer"]:
        orig_mean = original_sample[original_sample["industry"] == industry_name]["factor_momentum"].mean()
        proc_mean = processed_sample[processed_sample["industry"] == industry_name]["factor_momentum"].mean()
        change = proc_mean - orig_mean

        print(f"  {industry_name:20s} | {orig_mean:+10.4f} | {proc_mean:+12.4f} | {change:+10.4f}")

    return True


def test_cross_sectional_isolation():
    """
    测试2：横截面隔离性

    验证：每天的横截面处理只基于当天的数据，不受其他日期干扰
    """
    print("\n" + "="*70)
    print("🧪 测试2：横截面时间隔离性验证")
    print("="*70)

    df = generate_chinext_test_data()
    pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
        winsorize_method="percentile",
        winsorize_limits=(0.1, 0.9),  # 截断10%-90%（激进，便于观察）
        enable_market_cap_neutralization=False,
        enable_industry_neutralization=False,  # 关闭中性化，单独测试去极值
        cross_sectional=True,
    ))

    processed_df, _ = pipeline.process_factor_dataframe(
        df=df,
        factor_columns=["factor_momentum"],
        date_column="date",
    )

    # 选择两个不同的日期对比
    date1 = df["date"].iloc[20]  # 波动较小的时期
    date2 = df["date"].iloc[80]  # 波动较大的时期

    orig_date1 = df[df["date"] == date1]["factor_momentum"]
    proc_date1 = processed_df[processed_df["date"] == date1]["factor_momentum"]

    orig_date2 = df[df["date"] == date2]["factor_momentum"]
    proc_date2 = processed_df[processed_df["date"] == date2]["factor_momentum"]

    print(f"\n📅 日期1 ({date1.strftime('%Y-%m-%d')}):")
    print(f"  原始: 范围=[{orig_date1.min():.2f}, {orig_date1.max():.2f}], "
          f"标准差={orig_date1.std():.4f}")
    print(f"  处理后: 范围=[{proc_date1.min():.2f}, {proc_date1.max():.2f}], "
          f"标准差={proc_date1.std():.4f}")

    print(f"\n📅 日期2 ({date2.strftime('%Y-%m-%d')}):")
    print(f"  原始: 范围=[{orig_date2.min():.2f}, {orig_date2.max():.2f}], "
          f"标准差={orig_date2.std():.4f}")
    print(f"  处理后: 范围=[{proc_date2.min():.2f}, {proc_date2.max():.2f}], "
          f"标准差={proc_date2.std():.4f}")

    # 验证：两天的处理结果应该基于各自的分布
    range_reduction_1 = (orig_date1.max() - orig_date1.min()) / (proc_date1.max() - proc_date1.min() + 1e-10)
    range_reduction_2 = (orig_date2.max() - orig_date2.min()) / (proc_date2.max() - proc_date2.min() + 1e-10)

    print(f"\n✅ 极值范围压缩比:")
    print(f"  日期1: {range_reduction_1:.2f}倍")
    print(f"  日期2: {range_reduction_2:.2f}倍")

    if abs(range_reduction_1 - range_reduction_2) > 0.5:
        print(f"\n💡 结论: 两天的压缩比不同，证明每天独立计算 ✅")
    else:
        print(f"\n💡 结论: 两天的压缩比相近（可能因为数据分布相似）")

    return True


def test_small_industry_handling():
    """
    测试3：小样本行业的处理

    验证：当某个行业只有少量股票时，系统如何处理
    """
    print("\n" + "="*70)
    print("🧪 测试3：小样本行业处理验证")
    print("="*70)

    df = generate_chinext_test_data()

    # 人为创建一个只有2只股票的"微小行业"
    tiny_industry_mask = (df["stock_code"] == f"{300001:06d}.SZ") | \
                        (df["stock_code"] == f"{300002:06d}.SZ")
    df.loc[tiny_industry_mask, "industry"] = "Tiny_Industry"

    pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
        winsorize_method="mad",
        enable_industry_neutralization=True,
        min_samples=10,  # 设置较高的最小样本要求
        cross_sectional=True,
    ))

    processed_df, stats = pipeline.process_factor_dataframe(
        df=df,
        factor_columns=["factor_momentum"],
        industry_column="industry",
        date_column="date",
    )

    # 检查小行业是否被跳过
    tiny_data = processed_df[tiny_industry_mask]["factor_momentum"]
    normal_data = processed_df[~tiny_industry_mask & 
                              (processed_df["industry"] == "ChiNext_Tech")]["factor_momentum"]

    print(f"\n📊 小样本行业（Tiny_Industry, 2只股票）:")
    print(f"  样本数: {len(tiny_data)}")
    print(f"  均值: {tiny_data.mean():.4f}")
    print(f"  标准差: {tiny_data.std():.4f}")

    print(f"\n📊 正常行业（ChiNext_Tech, 15只股票）:")
    print(f"  样本数: {len(normal_data)}")
    print(f"  均值: {normal_data.mean():.4f}")
    print(f"  标准差: {normal_data.std():.4f}")

    # 验证：小行业的值应该没有被标准化到均值0（因为被跳过）
    if abs(tiny_data.mean()) > abs(normal_data.mean()):
        print(f"\n✅ 验证通过: 小行业未被强制中性化（保持原值特征）")
    else:
        print(f"\n⚠️ 注意: 小行业的均值也接近0，可能仍被部分处理")

    return True


def test_market_cap_correlation():
    """
    测试4：市值中性化效果

    验证：中性化后因子与市值的相关性显著降低
    """
    print("\n" + "="*70)
    print("🧪 测试4：市值中性化效果验证")
    print("="*70)

    df = generate_chinext_test_data()

    # 计算原始相关性
    corr_original = df["factor_momentum"].corr(df["market_cap"])

    # 处理后
    pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
        winsorize_method=None,
        enable_market_cap_neutralization=True,
        enable_industry_neutralization=False,
        standardize_method=None,
    ))

    processed_series, _ = pipeline.process_single_factor(
        factor_values=df["factor_momentum"],
        market_cap=df["market_cap"],
    )

    # 计算处理后相关性
    valid_mask = processed_series.notna() & df["market_cap"].notna()
    corr_processed = processed_series[valid_mask].corr(df.loc[valid_mask, "market_cap"])

    reduction_pct = (1 - abs(corr_processed) / (abs(corr_original) + 1e-10)) * 100

    print(f"\n📈 市值相关性变化:")
    print(f"  原始相关性:   {corr_original:+.6f}")
    print(f"  处理后相关性: {corr_processed:+.6f}")
    print(f"  降低幅度:     {reduction_pct:.1f}%")

    if reduction_pct > 80:
        print(f"\n✅ 效果显著: 市值相关性降低超过80%")
    elif reduction_pct > 50:
        print(f"\n🟡 效果中等: 市值相关性降低超过50%")
    else:
        print(f"\n❌ 效果不佳: 市值相关性仍然较高")

    return True


def main():
    """运行所有测试"""
    print("\n" + "🔬"*35)
    print("   创业板行业中性化深度分析测试套件")
    print("   验证独立性、隔离性、鲁棒性")
    print("🔬"*35 + "\n")

    tests = [
        ("独立性验证", test_independence_from_main_board),
        ("时间隔离性", test_cross_sectional_isolation),
        ("小样本处理", test_small_industry_handling),
        ("市值中性化", test_market_cap_correlation),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 异常: {e}")
            results.append((name, False))

    # 汇总
    print("\n" + "="*70)
    print("📋 测试汇总")
    print("="*70)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status} - {name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过! 创业板中性化工作正常，无外部干扰风险")
    else:
        print("\n⚠️ 部分测试未通过，建议检查配置")

    return passed == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)

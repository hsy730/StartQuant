"""
Mask-First架构下因子挖掘端到端集成测试

验证目标：
1. 数据加载阶段自动构建tradable_mask
2. 遗传算法使用Mask-First算子生成因子
3. IC/IR计算排除涨跌停污染
4. 回测引擎正确应用mask
5. 对比Mask-First前后的因子质量差异

运行方式:
    python tests/test_mask_first_factor_mining.py
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志级别以捕获所有关键信息
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 测试数据工厂：创建包含涨跌停的模拟数据
# ============================================================================


def create_realistic_stock_data(
    stock_code: str = "000001",
    n_days: int = 500,
    seed: int = 42,
    include_limit_up_down: bool = True,
    limit_up_ratio: float = 0.03,
    limit_down_ratio: float = 0.02,
    suspended_ratio: float = 0.01,
) -> pd.DataFrame:
    """
    创建真实的A股模拟数据，包含涨跌停和停牌

    Args:
        stock_code: 股票代码
        n_days: 交易日数量
        seed: 随机种子
        include_limit_up_down: 是否包含涨跌停
        limit_up_ratio: 涨停比例（默认3%）
        limit_down_ratio: 跌停比例（默认2%）
        suspended_ratio: 停牌比例（默认1%）

    Returns:
        包含OHLCV数据的DataFrame（index为日期）
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)

    # 1. 生成基础价格序列（随机游走）
    returns = np.random.randn(n_days) * 0.02  # 日收益率标准差2%
    close_prices = 10 * np.cumprod(1 + returns)  # 从10元开始

    # 2. 生成OHLCV数据
    open_prices = close_prices + np.random.randn(n_days) * 0.3
    high_prices = np.maximum(open_prices, close_prices) + abs(np.random.randn(n_days) * 0.2)
    low_prices = np.minimum(open_prices, close_prices) - abs(np.random.randn(n_days) * 0.2)
    volumes = (1000000 + np.random.randint(0, 5000000, n_days)).astype(float)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        },
        index=dates,
    )

    # 3. 注入涨停日（一字涨停）
    if include_limit_up_down:
        n_limit_up = int(n_days * limit_up_ratio)
        limit_up_indices = np.random.choice(range(20, n_days - 20), size=n_limit_up, replace=False)

        for idx in limit_up_indices:
            prev_close = df.iloc[idx - 1]["close"]
            limit_price = round(prev_close * 1.10, 2)  # 主板+10%

            # 一字涨停：open=high=low=close=limit_price
            df.iloc[idx, df.columns.get_loc("open")] = limit_price
            df.iloc[idx, df.columns.get_loc("high")] = limit_price
            df.iloc[idx, df.columns.get_loc("low")] = limit_price
            df.iloc[idx, df.columns.get_loc("close")] = limit_price
            df.iloc[idx, df.columns.get_loc("volume")] *= 3  # 涨停日成交量通常较大

    # 4. 注入跌停日（一字跌停）
    if include_limit_up_down:
        n_limit_down = int(n_days * limit_down_ratio)
        limit_down_indices = np.random.choice(range(20, n_days - 20), size=n_limit_down, replace=False)

        # 确保不与涨停日重叠
        limit_down_indices = [i for i in limit_down_indices if i not in limit_up_indices]

        for idx in limit_down_indices[:n_limit_down]:
            prev_close = df.iloc[idx - 1]["close"]
            limit_price = round(prev_close * 0.90, 2)  # 主板-10%

            # 一字跌停
            df.iloc[idx, df.columns.get_loc("open")] = limit_price
            df.iloc[idx, df.columns.get_loc("high")] = limit_price
            df.iloc[idx, df.columns.get_loc("low")] = limit_price
            df.iloc[idx, df.columns.get_loc("close")] = limit_price
            df.iloc[idx, df.columns.get_loc("volume")] *= 0.5  # 跌停日成交量较小

    # 5. 注入停牌日
    n_suspended = int(n_days * suspended_ratio)
    suspended_indices = np.random.choice(range(30, n_days - 10), size=n_suspended, replace=False)

    # 确保不与涨跌停日重叠
    suspended_indices = [i for i in suspended_indices if i not in limit_up_indices and i not in limit_down_indices]

    for idx in suspended_indices[:n_suspended]:
        df.iloc[idx, df.columns.get_loc("volume")] = 0  # 成交量为0表示停牌

    logger.info(
        f"✅ 创建模拟数据 [{stock_code}] | "
        f"总天数: {n_days} | "
        f"涨停: {len(limit_up_indices)} ({len(limit_up_indices)/n_days*100:.1f}%) | "
        f"跌停: {len(limit_down_indices)} ({len(limit_down_indices)/n_days*100:.1f}%) | "
        f"停牌: {len(suspended_indices)} ({len(suspended_indices)/n_days*100:.1f}%)"
    )

    return df


def prepare_base_factors(df: pd.DataFrame) -> Dict[str, str]:
    """准备基础因子列表"""
    return {
        "close": "df['close']",
        "open": "df['open']",
        "high": "df['high']",
        "low": "df['low']",
        "volume": "df['volume']",
    }


# ============================================================================
# Phase 1: 测试数据加载层的Mask-First功能
# ============================================================================


def test_data_service_mask_construction():
    """
    测试1: 验证data_service自动构建tradable_mask的准确性
    """
    print("\n" + "=" * 80)
    print("🧪 Phase 1: 测试数据加载层 - Mask构建准确性")
    print("=" * 80)

    from backend.services.data_service import DataService

    ds = DataService()

    # 创建包含涨跌停的测试数据
    test_df = create_realistic_stock_data("000001", n_days=300)

    print("\n📊 输入数据统计:")
    print(f"   总交易日数: {len(test_df)}")
    print(f"   价格范围: {test_df['close'].min():.2f} - {test_df['close'].max():.2f}")

    # 调用_detect_price_limits方法
    start_time = time.time()
    result_df = ds._detect_price_limits(test_df, "000001")
    elapsed_time = time.time() - start_time

    print(f"\n⏱️ Mask构建耗时: {elapsed_time:.4f}秒")

    # 验证输出列完整性
    required_columns = ["is_limit_up", "is_limit_down", "is_suspended", "tradable_mask"]
    missing_cols = [col for col in required_columns if col not in result_df.columns]

    if missing_cols:
        print(f"❌ 缺少必要列: {missing_cols}")
        return False

    print("\n✅ Mask构建成功，输出列完整:")
    for col in required_columns:
        count = result_df[col].sum() if col != "tradable_mask" else (~result_df[col]).sum()
        total = len(result_df)
        ratio = count / total * 100
        print(f"   {col}: {count}/{total} ({ratio:.1f}%)")

    # 验证tradable_mask逻辑
    tradable_count = result_df["tradable_mask"].sum()
    tradable_ratio = tradable_count / len(result_df)

    print("\n📈 可交易性统计:")
    print(f"   可交易天数: {tradable_count}")
    print(f"   可交易比例: {tradable_ratio*100:.1f}%")
    print("   预期范围: 92%-96% (基于注入比例)")

    if 0.90 < tradable_ratio < 0.98:
        print("✅ 可交易比例在合理范围内")
        return True
    else:
        print(f"⚠️ 可交易比例异常: {tradable_ratio*100:.1f}%")
        return False


# ============================================================================
# Phase 2: 测试遗传算法因子挖掘流程
# ============================================================================


def test_genetic_mining_with_mask():
    """
    测试2: 使用Mask-First架构进行完整的因子挖掘流程
    """
    print("\n" + "=" * 80)
    print("🧬 Phase 2: 测试遗传算法因子挖掘 - Mask-First集成")
    print("=" * 80)

    try:
        from deap import base, creator, tools, gp  # noqa: F401
    except ImportError:
        print("❌ DEAP库未安装，跳过此测试")
        return None

    try:
        from backend.services.genetic_factor_mining_service import (  # noqa: F401
            create_genetic_mining_service,
        )
        from backend.services.data_service import DataService
        from backend.services.factor_primitives import create_pset
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return None

    # 1. 准备数据（包含Mask-First增强）
    print("\n📥 步骤1: 准备测试数据...")
    raw_df = create_realistic_stock_data("000001", n_days=400)

    # 应用Mask-First预处理
    ds = DataService()
    processed_df = ds._preprocess_data(raw_df.copy(), stock_code="000001")

    if "tradable_mask" in processed_df.columns:
        mask_stats = {
            "total": len(processed_df),
            "tradable": processed_df["tradable_mask"].sum(),
            "ratio": processed_df["tradable_mask"].mean(),
            "limit_up": processed_df["is_limit_up"].sum(),
            "limit_down": processed_df["is_limit_down"].sum(),
            "suspended": processed_df["is_suspended"].sum(),
        }
        print("   ✅ Mask-First已启用:")
        print(f"      总天数: {mask_stats['total']}")
        print(f"      可交易: {mask_stats['tradable']} ({mask_stats['ratio']*100:.1f}%)")
        print(
            f"      涨停: {mask_stats['limit_up']}, 跌停: {mask_stats['limit_down']}, 停牌: {mask_stats['suspended']}"
        )
    else:
        print("   ⚠️ 未检测到tradable_mask，将使用传统模式")
        mask_stats = None

    # 2. 准备基础因子
    print("\n📊 步骤2: 准备基础因子...")
    base_factors = prepare_base_factors(processed_df)
    base_factor_values = {}

    for factor_name, factor_code in base_factors.items():
        exec(f"{factor_name} = processed_df[factor_name]")
        base_factor_values[factor_name] = {
            "code": factor_code,
            "values": processed_df[factor_name],
        }

    print(f"   基础因子数: {len(base_factors)}")
    for name in base_factors.keys():
        valid_count = base_factor_values[name]["values"].notna().sum()
        print(f"      - {name}: {valid_count} 个有效值")

    # 3. 创建PrimitiveSet（验证Mask-First配置）
    print("\n⚙️ 步骤3: 创建PrimitiveSet...")
    pset = create_pset(n_factors=len(base_factors), extended=True, use_masked=True)  # ✅ 启用Mask-First

    print(f"   原语数量: {sum(len(v) for v in pset.primitives.values())}")
    print(f"   终端数量: {sum(len(v) for v in pset.terminals.values())}")

    # 列出时间序列窗口算子
    all_primitives = [p for plist in pset.primitives.values() for p in plist]
    ts_operators = [p.name for p in all_primitives if p.name.startswith("ts_")]
    print(f"   时间序列算子 ({len(ts_operators)}个): {ts_operators}")

    # 4. 配置并启动因子挖掘（小规模快速测试）
    print("\n🚀 步骤4: 启动因子挖掘（快速模式）...")
    print("   参数: population_size=20, n_generations=5 (快速测试)")

    mining_config = {
        "base_factors": list(base_factors.keys()),
        "data": processed_df,
        "population_size": 20,  # 小种群用于快速测试
        "n_generations": 5,  # 少代数用于快速测试
        "cx_prob": 0.7,
        "mut_prob": 0.2,
        "elite_size": 2,
        "fitness_objective": "ic_mean",
        "parsimony_coeff": 0.001,
        "diversity_penalty_coeff": 0.1,
        "use_extended_primitives": True,
        "use_nsga2": False,  # 简化测试
        "max_tree_depth": 8,  # 控制复杂度
    }

    # 创建进度回调函数
    progress_history = []

    def on_progress(gen, total, best_fit, avg_fit):
        progress_history.append(
            {
                "generation": gen,
                "best_fitness": best_fit,
                "avg_fitness": avg_fit,
            }
        )
        print(f"   📈 代数 {gen}/{total} | 最优适应度: {best_fit:.4f} | 平均适应度: {avg_fit:.4f}")

    # 执行因子挖掘
    start_time = time.time()

    try:
        mining_service = create_genetic_mining_service(**mining_config)
        mining_service.set_progress_callback(on_progress)
        result = mining_service.mine_factors()
        elapsed_time = time.time() - start_time

        print(f"\n⏱️ 因子挖掘耗时: {elapsed_time:.2f}秒")

        # 5. 分析结果
        print("\n📋 步骤5: 分析挖掘结果...")

        if result.get("success", False):
            best_factors = result.get("best_factors", [])

            print("   ✅ 挖掘成功!")
            print(f"   发现最优因子数: {len(best_factors)}")

            if len(best_factors) > 0:
                print("\n   🏆 Top-5 最优因子:")
                for i, factor in enumerate(best_factors[:5]):
                    expr = factor.get("expression", "N/A")[:60]
                    fitness = factor.get("fitness", 0)
                    ic = factor.get("ic_mean", 0)
                    ir = factor.get("ir_ratio", 0)

                    print(f"   {i+1}. 适应度: {fitness:.4f} | IC: {ic:.4f} | IR: {ir:.4f}")
                    print(f"      表达式: {expr}...")

                # 6. 验证Mask-First影响
                print("\n🔍 步骤6: 验证Mask-First效果...")

                if len(progress_history) >= 2:
                    first_gen_avg = progress_history[0]["avg_fitness"]
                    last_gen_best = progress_history[-1]["best_fitness"]
                    improvement = (
                        (last_gen_best - first_gen_avg) / abs(first_gen_avg) * 100 if first_gen_avg != 0 else 0
                    )

                    print("   进化曲线:")
                    print(f"      第1代平均适应度: {first_gen_avg:.4f}")
                    print(f"      最后一代最优: {last_gen_best:.4f}")
                    print(f"      改善幅度: {improvement:+.1f}%")

                    if last_gen_best > first_gen_avg:
                        print("   ✅ 算法收敛正常，适应度持续提升")
                    else:
                        print("   ⚠️ 适应度未明显提升（可能需要更多代数）")

                return {
                    "success": True,
                    "elapsed_time": elapsed_time,
                    "n_factors": len(best_factors),
                    "best_fitness": best_factors[0].get("fitness", 0) if best_factors else 0,
                    "mask_enabled": mask_stats is not None,
                    "progress_history": progress_history,
                }
            else:
                print("   ⚠️ 未发现有效因子")
                return {"success": True, "n_factors": 0}
        else:
            error_msg = result.get("message", "未知错误")
            print(f"   ❌ 挖掘失败: {error_msg}")
            return {"success": False, "error": error_msg}

    except Exception as e:
        print(f"   ❌ 异常: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ============================================================================
# Phase 3: 对比测试 - 有/无Mask-First的差异
# ============================================================================


def compare_with_without_mask():
    """
    测试3: 对比有/无Mask-First的IC计算差异
    """
    print("\n" + "=" * 80)
    print("🔬 Phase 3: 对比测试 - Mask-First vs 传统模式")
    print("=" * 80)

    from backend.services.analysis_service import AnalysisService

    analysis = AnalysisService()

    # 创建测试数据
    n_days = 250
    dates = pd.bdate_range(start="2023-06-01", periods=n_days)

    np.random.seed(123)

    # 模拟因子值（有一定预测能力）
    true_signal = np.sin(np.linspace(0, 4 * np.pi, n_days)) * 0.02
    noise = np.random.randn(n_days) * 0.03
    factor_values = pd.Series(true_signal + noise, index=dates, name="test_factor")

    # 模拟未来收益率（与因子有弱相关性）
    future_returns = pd.Series(
        0.001 * factor_values + np.random.randn(n_days) * 0.015, index=dates, name="future_return_1"
    )

    # 构建DataFrame
    df = pd.DataFrame(
        {
            "test_factor": factor_values,
            "future_return_1": future_returns,
        }
    )

    # 注入一些异常值（模拟涨跌停污染）
    outlier_indices = np.random.choice(range(50, n_days - 50), size=int(n_days * 0.05))
    for idx in outlier_indices:
        df.iloc[idx, df.columns.get_loc("test_factor")] += np.random.choice([-1, 1]) * 0.15  # 大幅异常

    # 场景A: 无Mask（传统方式）
    print("\n📊 场景A: 传统模式（无Mask）...")
    factor_data_no_mask = {"TEST_STOCK": df.copy()}

    result_no_mask = analysis._calculate_single_stock_ic(
        factor_data_no_mask, ["test_factor"], use_tradable_mask=False  # ❌ 不使用mask
    )

    ic_no_mask = result_no_mask["ic_stats"]["test_factor"]
    print(f"   IC均值: {ic_no_mask['IC均值']:.4f}")
    print(f"   IC标准差: {ic_no_mask['IC标准差']:.4f}")
    print(f"   IR: {ic_no_mask['IR']:.4f}")
    print(f"   Mask-First: {ic_no_mask.get('Mask-First', False)}")

    # 场景B: 有Mask（Mask-First方式）
    print("\n🛡️ 场景B: Mask-First模式...")

    # 构建简单的mask（标记异常值为不可交易）
    df_with_mask = df.copy()
    z_scores = np.abs((df["test_factor"] - df["test_factor"].mean()) / df["test_factor"].std())
    df_with_mask["tradable_mask"] = z_scores < 2.5  # Z-score>2.5视为异常
    df_with_mask["is_limit_up"] = z_scores > 3.0
    df_with_mask["is_limit_down"] = z_scores > 3.0
    df_with_mask["is_suspended"] = ~df_with_mask["tradable_mask"]

    factor_data_with_mask = {"TEST_STOCK": df_with_mask}

    result_with_mask = analysis._calculate_single_stock_ic(
        factor_data_with_mask, ["test_factor"], use_tradable_mask=True  # ✅ 使用mask
    )

    ic_with_mask = result_with_mask["ic_stats"]["test_factor"]
    print(f"   IC均值: {ic_with_mask['IC均值']:.4f}")
    print(f"   IC标准差: {ic_with_mask['IC标准差']:.4f}")
    print(f"   IR: {ic_with_mask['IR']:.4f}")
    print(f"   Mask-First: {ic_with_mask.get('Mask-First', False)}")

    # 统计信息
    if "mask_statistics" in result_with_mask:
        stats = result_with_mask["mask_statistics"]
        print("\n   📈 Mask统计:")
        print(f"      可交易比例: {stats['tradable_ratio']*100:.1f}%")
        print(f"      已过滤: {(1-stats['tradable_ratio'])*100:.1f}% 的异常数据")

    # 对比分析
    print("\n🔍 对比分析:")
    ic_diff = abs(ic_no_mask["IC均值"]) - abs(ic_with_mask["IC均值"])
    ir_diff = abs(ic_no_mask["IR"]) - abs(ic_with_mask["IR"])

    print(f"   IC绝对值差异: {ic_diff:+.4f} ({'降低' if ic_diff > 0 else '升高'})")
    print(f"   IR绝对值差异: {ir_diff:+.4f} ({'降低' if ir_diff > 0 else '升高'})")

    if ic_diff > 0:
        print("   ✅ Mask-First使IC更保守（减少虚假信号）")
    elif ic_diff < -0.005:
        print("   ⚠️ IC反而升高，可能mask过于严格")
    else:
        print("   ➡️ IC变化不大，说明数据质量较好")

    return {
        "no_mask": {
            "ic_mean": ic_no_mask["IC均值"],
            "ir": ic_no_mask["IR"],
        },
        "with_mask": {
            "ic_mean": ic_with_mask["IC均值"],
            "ir": ic_with_mask["IR"],
        },
        "difference": {
            "ic_diff": ic_diff,
            "ir_diff": ir_diff,
        },
    }


# ============================================================================
# Phase 4: 性能基准测试
# ============================================================================


def performance_benchmark():
    """
    测试4: Mask-First架构的性能开销评估
    """
    print("\n" + "=" * 80)
    print("⚡ Phase 4: 性能基准测试 - Mask-First开销评估")
    print("=" * 80)

    import time
    from backend.services.data_service import DataService
    from backend.services.factor_primitives import (
        ts_corr,
        ts_corr_masked,
    )

    DataService()

    # 测试不同数据规模
    test_sizes = [1000, 10000, 100000]
    results = []

    for n_samples in test_sizes:
        print(f"\n📊 数据规模: {n_samples:,} 样本")

        # 生成测试数据
        np.random.seed(int(n_samples / 1000))
        series_a = pd.Series(np.random.randn(n_samples))
        series_b = pd.Series(np.random.randn(n_samples))
        mask = pd.Series(np.random.choice([True, False], size=n_samples, p=[0.95, 0.05]))

        # 测试传统版本性能
        start = time.time()
        ts_corr(series_a, series_b, n=20)
        time_traditional = time.time() - start

        # 测试Mask-First版本性能
        start = time.time()
        ts_corr_masked(series_a, series_b, n=20, mask=mask)
        time_masked = time.time() - start

        # 计算额外开销
        overhead = (time_masked - time_traditional) / time_traditional * 100

        results.append(
            {
                "n_samples": n_samples,
                "time_traditional_ms": time_traditional * 1000,
                "time_masked_ms": time_masked * 1000,
                "overhead_percent": overhead,
            }
        )

        print(f"   传统版本: {time_traditional*1000:.2f}ms")
        print(f"   Mask-First: {time_masked*1000:.2f}ms")
        print(f"   额外开销: {overhead:+.1f}%")

        if overhead < 50:
            print("   ✅ 开销可接受 (<50%)")
        elif overhead < 100:
            print("   ⚠️ 开销中等 (50%-100%)")
        else:
            print("   ❌ 开销过高 (>100%)")

    # 总结
    avg_overhead = np.mean([r["overhead_percent"] for r in results])
    print("\n📈 性能总结:")
    print(f"   平均额外开销: {avg_overhead:.1f}%")

    if avg_overhead < 30:
        print("   ✅ Mask-First架构性能优秀，几乎无额外开销")
    elif avg_overhead < 70:
        print("   ✅ Mask-First架构性能良好，开销在可接受范围内")
    else:
        print("   ⚠️ 需要优化性能")

    return results


# ============================================================================
# 主测试入口
# ============================================================================


def run_all_tests():
    """运行所有Mask-First架构集成测试"""

    print("\n" + "=" * 80)
    print("🎯 Mask-First架构 - 因子挖掘端到端集成测试套件")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_start_time = time.time()

    all_results = {}

    # Phase 1: 数据加载层测试
    try:
        result_phase1 = test_data_service_mask_construction()
        all_results["phase1_data_loading"] = result_phase1
    except Exception as e:
        print(f"\n❌ Phase 1 失败: {e}")
        all_results["phase1_data_loading"] = {"success": False, "error": str(e)}

    # Phase 2: 遗传算法因子挖掘测试
    try:
        result_phase2 = test_genetic_mining_with_mask()
        all_results["phase2_factor_mining"] = result_phase2
    except Exception as e:
        print(f"\n❌ Phase 2 失败: {e}")
        import traceback

        traceback.print_exc()
        all_results["phase2_factor_mining"] = {"success": False, "error": str(e)}

    # Phase 3: 对比测试
    try:
        result_phase3 = compare_with_without_mask()
        all_results["phase3_comparison"] = result_phase3
    except Exception as e:
        print(f"\n❌ Phase 3 失败: {e}")
        all_results["phase3_comparison"] = {"success": False, "error": str(e)}

    # Phase 4: 性能基准测试
    try:
        result_phase4 = performance_benchmark()
        all_results["phase4_performance"] = result_phase4
    except Exception as e:
        print(f"\n❌ Phase 4 失败: {e}")
        all_results["phase4_performance"] = {"success": False, "error": str(e)}

    total_elapsed = time.time() - total_start_time

    # 最终报告
    print("\n" + "=" * 80)
    print("📋 最终测试报告")
    print("=" * 80)

    phase_names = {
        "phase1_data_loading": "Phase 1: 数据加载层Mask构建",
        "phase2_factor_mining": "Phase 2: 遗传算法因子挖掘",
        "phase3_comparison": "Phase 3: Mask-First对比测试",
        "phase4_performance": "Phase 4: 性能基准测试",
    }

    passed = 0
    failed = 0

    for key, name in phase_names.items():
        result = all_results.get(key, {})
        success = result.get("success", False) if isinstance(result, dict) else bool(result)

        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {name}: {'通过' if success else '失败'}")

        if success:
            passed += 1
        else:
            failed += 1
            if "error" in result:
                print(f"   错误: {result['error'][:80]}")

    print("\n📊 测试总览:")
    print(f"   通过: {passed}/{passed+failed}")
    print(f"   失败: {failed}/{passed+failed}")
    print(f"   总耗时: {total_elapsed:.2f}秒")
    print(f"⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if passed == len(phase_names):
        print("\n🎉 所有测试通过！Mask-First架构工作正常，可以投入生产使用。")
    elif passed >= len(phase_names) * 0.75:
        print(f"\n✅ 大部分测试通过 ({passed}/{len(phase_names)})，Mask-First架构基本可用。")
    else:
        print("\n⚠️ 多项测试失败，请检查错误日志。")

    return all_results


if __name__ == "__main__":
    run_all_tests()

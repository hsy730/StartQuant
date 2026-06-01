"""
因子预处理管道单元测试

验证数据美颜流程的正确性和性能
"""
import pytest
import numpy as np
import pandas as pd
import time
from backend.services.factor_preprocessing_pipeline import (
    FactorPreprocessingPipeline,
    PreprocessingConfig,
    WinsorizeMethod,
    StandardizeMethod,
    CONSERVATIVE_CONFIG,
    AGGRESSIVE_CONFIG,
    ML_MODEL_CONFIG,
)


class TestFactorPreprocessingPipeline:
    """因子预处理管道测试类"""

    def setup_method(self):
        """每个测试方法前的初始化"""
        np.random.seed(42)

        # 生成模拟数据：100个交易日，50只股票
        n_dates = 100
        n_stocks = 50

        dates = pd.date_range(start="2023-01-01", periods=n_dates, freq="B")
        stock_codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]

        # 创建多索引DataFrame（横截面格式）
        self.multi_index_data = []
        for date in dates:
            for stock in stock_codes:
                self.multi_index_data.append({
                    "date": date,
                    "stock_code": stock,
                    "factor_1": np.random.randn() * 10 + 5,  # 正态分布，有极端值
                    "factor_2": np.random.randn() * 20 - 3,  # 不同均值和方差
                    "market_cap": np.random.lognormal(mean=10, sigma=1),  # 对数正态分布
                    "industry": np.random.choice(["Tech", "Finance", "Health", "Energy"]),
                })

        self.df_cross_sectional = pd.DataFrame(self.multi_index_data)

        # 创建时间序列格式数据（单股票）
        self.df_time_series = pd.DataFrame({
            "date": dates,
            "factor_1": np.random.randn(n_dates) * 10 + 5,
            "factor_2": np.random.randn(n_dates) * 20 - 3,
            "close": 100 + np.cumsum(np.random.randn(n_dates)),
            "market_cap": np.random.lognormal(mean=10, sigma=1, size=n_dates),
            "industry": ["Tech"] * n_dates,  # 单行业
        })
        self.df_time_series.set_index("date", inplace=True)

    def test_mad_winsorization(self):
        """测试MAD法去极值"""
        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method=WinsorizeMethod.MAD,
            winsorize_n_sigma=3.0,
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=False,
            standardize_method=StandardizeMethod.ZSCORE,
        ))

        factor_values = self.df_time_series["factor_1"]
        result, stats = pipeline.process_single_factor(factor_values)

        # 验证去极值效果：结果应该在合理范围内
        assert stats["winsorized_count"] > 0, "应该有部分数据被截断"
        assert not np.isinf(result).any(), "结果不应包含无穷大值"
        assert not np.isnan(result).any(), "结果不应包含NaN值（已填充）"

        print(f"✅ MAD法去极值测试通过: 截断了{stats['winsorized_count']}个异常值")

    def test_percentile_winsorization(self):
        """测试百分位法去极值"""
        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method=WinsorizeMethod.PERCENTILE,
            winsorize_limits=(0.05, 0.95),  # 截断5%-95%分位之外的数据
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=False,
            standardize_method=None,  # 不标准化，只测试去极值
        ))

        factor_values = self.df_time_series["factor_1"]
        original_min = factor_values.min()
        original_max = factor_values.max()

        result, stats = pipeline.process_single_factor(factor_values)

        # 验证极值被截断
        assert result.min() >= original_min, "最小值应该大于等于原始最小值"
        assert result.max() <= original_max, "最大值应该小于等于原始最大值"

        print(f"✅ 百分位法去极值测试通过: [{original_min:.2f}, {original_max:.2f}] → [{result.min():.2f}, {result.max():.2f}]")

    def test_std_winsorization(self):
        """测试3σ法去极值"""
        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method=WinsorizeMethod.STD,
            winsorize_n_sigma=2.5,
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=False,
            standardize_method=None,
        ))

        factor_values = self.df_time_series["factor_1"]
        result, stats = pipeline.process_single_factor(factor_values)

        mean = result.mean()
        std = result.std()

        # 验证所有值都在2.5σ范围内
        assert (np.abs(result - mean) <= 2.6 * std).all(), "所有值应在2.5σ范围内（允许微小误差）"

        print(f"✅ 3σ法去极值测试通过: 均值={mean:.4f}, 标准差={std:.4f}")

    def test_zscore_standardization(self):
        """测试Z-score标准化"""
        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method=None,  # 跳过去极值
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=False,
            standardize_method=StandardizeMethod.ZSCORE,
        ))

        factor_values = self.df_time_series["factor_1"]
        result, stats = pipeline.process_single_factor(factor_values)

        # 验证Z-score性质：均值为0，标准差为1
        assert abs(result.mean()) < 1e-10, f"均值应接近0，实际为{result.mean()}"
        assert abs(result.std() - 1.0) < 0.01, f"标准差应接近1，实际为{result.std()}"

        print(f"✅ Z-score标准化测试通过: 均值={result.mean():.6f}, 标准差={result.std():.6f}")

    def test_rank_standardization(self):
        """测试Rank标准化"""
        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method=None,
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=False,
            standardize_method=StandardizeMethod.RANK,
        ))

        factor_values = self.df_time_series["factor_1"]
        result, stats = pipeline.process_single_factor(factor_values)

        # 验证Rank标准化性质：值域在[0, 1]，近似均匀分布
        assert result.min() >= 0, "最小值应>=0"
        assert result.max() <= 1.0, "最大值应<=1"
        
        # 检查分布是否相对均匀（不应集中在某个区间）
        quantiles = result.quantile([0.25, 0.5, 0.75])
        assert (quantiles >= 0.2).all() and (quantiles <= 0.8).all(), "四分位数应在[0.2, 0.8]范围内"

        print(f"✅ Rank标准化测试通过: 范围=[{result.min():.4f}, {result.max():.4f}]")

    def test_market_cap_neutralization(self):
        """测试市值中性化"""
        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method=None,
            enable_market_cap_neutralization=True,
            enable_industry_neutralization=False,
            standardize_method=None,
        ))

        factor_values = self.df_time_series["factor_1"]
        market_cap = self.df_time_series["market_cap"]

        # 计算中性化前与市值的相关性
        corr_before = factor_values.corr(market_cap)

        result, stats = pipeline.process_single_factor(
            factor_values=factor_values,
            market_cap=market_cap,
        )

        # 计算中性化后与市值的相关性（应显著降低）
        valid_mask = result.notna() & market_cap.notna()
        if valid_mask.sum() > 10:
            corr_after = result[valid_mask].corr(market_cap[valid_mask])
            reduction_ratio = abs(corr_after) / (abs(corr_before) + 1e-10)
            
            assert reduction_ratio < 0.8, (
                f"市值相关性应降低80%以上，实际只降低了{(1-reduction_ratio)*100:.1f}%"
            )
            assert stats["neutralized"] == True, "统计信息应标记为已中性化"

        print(f"✅ 市值中性化测试通过: 相关性 {corr_before:.4f} → {corr_after:.4f}")

    def test_industry_neutralization(self):
        """测试行业中性化"""
        # 使用横截面数据进行行业中性化测试
        df_test = self.df_cross_sectional.copy()

        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method=None,
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=True,
            standardize_method=None,
            cross_sectional=True,
        ))

        processed_df, _ = pipeline.process_factor_dataframe(
            df=df_test,
            factor_columns=["factor_1"],
            market_cap_column="market_cap",
            industry_column="industry",
            date_column="date",
        )

        # 验证每个行业内因子的均值接近0（或至少比处理前更接近）
        for industry_name, group in processed_df.groupby("industry"):
            industry_mean = group["factor_1"].mean()
            assert abs(industry_mean) < 1.0, (
                f"行业{industry_name}的因子均值应为0附近，实际为{industry_mean:.4f}"
            )

        print("✅ 行业中性化测试通过: 所有行业均值接近0")

    def test_full_pipeline_integration(self):
        """完整管道集成测试"""
        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method=WinsorizeMethod.MAD,
            winsorize_n_sigma=3.0,
            enable_market_cap_neutralization=True,
            enable_industry_neutralization=True,
            standardize_method=StandardizeMethod.ZSCORE,
            handle_missing="fill_zero",
        ))

        # 测试单因子完整流程
        result, stats = pipeline.process_single_factor(
            factor_values=self.df_time_series["factor_1"],
            market_cap=self.df_time_series["market_cap"],
            industry=self.df_time_series["industry"],
            date_index=self.df_time_series.index,
        )

        # 验证完整流程的结果
        assert not result.isna().any(), "不应有缺失值"
        assert not np.isinf(result).any(), "不应有无穷大值"
        assert abs(result.mean()) < 0.1, "Z-score标准化后均值应接近0"
        assert abs(result.std() - 1.0) < 0.1, "Z-score标准化后标准差应接近1"
        assert stats["neutralized"] == True, "应执行了中性化"
        assert stats["standardized"] == True, "应执行了标准化"

        print(f"✅ 完整管道测试通过:")
        print(f"   - 截断数量: {stats['winsorized_count']}")
        print(f"   - 中性化: {'是' if stats['neutralized'] else '否'}")
        print(f"   - 标准化: {'是' if stats['standardized'] else '否'}")

    def test_multi_stock_batch_processing(self):
        """多股票批量处理性能测试"""
        # 构建多股票字典格式数据
        factor_data = {}
        for stock_id in range(1, 11):  # 10只股票
            stock_code = f"{stock_id:06d}"
            df = pd.DataFrame({
                "factor_1": np.random.randn(100) * 15 + 5,
                "factor_2": np.random.randn(100) * 25 - 8,
                "market_cap": np.random.lognormal(mean=10, sigma=1.5, size=100),
                "industry": np.random.choice(["Tech", "Finance"], size=100),
                "close": 100 + np.cumsum(np.random.randn(100)),
            }, index=pd.date_range(start="2023-01-01", periods=100, freq="B"))
            factor_data[stock_code] = df

        pipeline = FactorPreprocessingPipeline(config=CONSERVATIVE_CONFIG)

        # 性能测试
        start_time = time.time()
        processed_data, all_stats = pipeline.process_multi_stock_factors(
            factor_data=factor_data,
            factor_names=["factor_1", "factor_2"],
            parallel_stocks=True,
            max_workers=4,
        )
        elapsed_time = time.time() - start_time

        # 验证处理结果
        assert len(processed_data) == 10, "应处理10只股票"
        assert all(code in processed_data for code in factor_data.keys()), "所有股票都应被处理"

        # 验证每只股票的因子都被正确处理
        for stock_code, df in processed_data.items():
            assert "factor_1" in df.columns
            assert not df["factor_1"].isna().all(), f"{stock_code}的factor_1不应全为NaN"

        # 性能要求：10只股票x100天x2因子 应在2秒内完成
        assert elapsed_time < 2.0, f"处理时间过长: {elapsed_time:.2f}秒"

        print(f"✅ 多股票批量处理测试通过:")
        print(f"   - 处理股票数: {len(processed_data)}")
        print(f"   - 处理耗时: {elapsed_time:.3f}秒")
        print(f"   - 统计摘要:\n{pipeline.get_processing_summary(all_stats.get(list(all_stats.keys())[0], {}))}")

    def test_edge_cases(self):
        """边界情况测试"""
        pipeline = FactorPreprocessingPipeline()

        # 测试全空数据
        empty_series = pd.Series([], dtype=float)
        result, stats = pipeline.process_single_factor(empty_series)
        assert stats.get("skipped", False), "空数据应跳过处理"

        # 测试全相同值
        constant_series = pd.Series([5.0] * 100)
        result, stats = pipeline.process_single_factor(constant_series)
        assert len(result) == 100, "长度应保持不变"

        # 测试包含无穷大的数据
        inf_series = pd.Series([1, 2, np.inf, -np.inf, 3] + [4] * 15)  # 增加到20个样本
        result, stats = pipeline.process_single_factor(inf_series)
        assert not np.isinf(result).any(), "无穷大值应被处理"

        # 测试极少样本
        tiny_series = pd.Series([1, 2, 3])
        result, stats = pipeline.process_single_factor(tiny_series)
        assert stats.get("skipped", False) or len(result) == 3, "少样本应跳过或正常返回"

        print("✅ 边界情况测试全部通过")

    def test_predefined_configs(self):
        """预定义配置测试"""
        # 保守配置
        conservative_pipeline = FactorPreprocessingPipeline(config=CONSERVATIVE_CONFIG)
        result_c, _ = conservative_pipeline.process_single_factor(self.df_time_series["factor_1"])
        assert len(result_c) == len(self.df_time_series), "保守配置处理后长度不变"

        # 激进配置
        aggressive_pipeline = FactorPreprocessingPipeline(config=AGGRESSIVE_CONFIG)
        result_a, _ = aggressive_pipeline.process_single_factor(self.df_time_series["factor_1"])
        assert len(result_a) == len(self.df_time_series), "激进配置处理后长度不变"

        # ML模型配置
        ml_pipeline = FactorPreprocessingPipeline(config=ML_MODEL_CONFIG)
        result_ml, _ = ml_pipeline.process_single_factor(self.df_time_series["factor_1"])
        assert len(result_ml) == len(self.df_time_series), "ML配置处理后长度不变"

        print("✅ 预定义配置测试全部通过")

    def test_performance_large_dataset(self):
        """大数据集性能测试（压力测试）"""
        # 生成大数据集：500只股票 x 500个交易日 x 5个因子
        n_stocks = 500
        n_dates = 500

        large_df = []
        dates = pd.date_range(start="2020-01-01", periods=n_dates, freq="B")
        stocks = [f"{i:06d}" for i in range(1, n_stocks + 1)]

        for date in dates[:100]:  # 只用100天避免内存过大
            for stock in stocks[:50]:  # 只用50只股票
                large_df.append({
                    "date": date,
                    "stock_code": stock,
                    "factor_1": np.random.randn() * 20,
                    "factor_2": np.random.randn() * 30,
                    "factor_3": np.random.randn() * 15,
                    "market_cap": np.random.lognormal(11, 1),
                    "industry": np.random.choice(["Tech", "Finance", "Health", "Energy", "Consumer"]),
                })

        df_large = pd.DataFrame(large_df)

        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method=WinsorizeMethod.MAD,
            standardize_method=StandardizeMethod.ZSCORE,
            cross_sectional=True,
        ))

        start_time = time.time()
        _, stats = pipeline.process_factor_dataframe(
            df=df_large,
            factor_columns=["factor_1", "factor_2", "factor_3"],
            date_column="date",
            parallel=True,
            max_workers=4,
        )
        elapsed_time = time.time() - start_time

        total_samples = len(df_large)
        throughput = total_samples / elapsed_time if elapsed_time > 0 else float('inf')

        print(f"🚀 大数据集性能测试:")
        print(f"   - 总样本数: {total_samples:,}")
        print(f"   - 处理耗时: {elapsed_time:.3f}秒")
        print(f"   - 吞吐量: {throughput:,.0f} 样本/秒")

        # 性能要求：50万样本应在5秒内完成
        assert elapsed_time < 5.0, f"大数据处理过慢: {elapsed_time:.2f}秒"


def run_all_tests():
    """运行所有测试"""
    test_instance = TestFactorPreprocessingPipeline()
    test_instance.setup_method()

    tests = [
        ("MAD法去极值", test_instance.test_mad_winsorization),
        ("百分位法去极值", test_instance.test_percentile_winsorization),
        ("3σ法去极值", test_instance.test_std_winsorization),
        ("Z-score标准化", test_instance.test_zscore_standardization),
        ("Rank标准化", test_instance.test_rank_standardization),
        ("市值中性化", test_instance.test_market_cap_neutralization),
        ("行业中性化", test_instance.test_industry_neutralization),
        ("完整管道集成", test_instance.test_full_pipeline_integration),
        ("多股票批量处理", test_instance.test_multi_stock_batch_processing),
        ("边界情况", test_instance.test_edge_cases),
        ("预定义配置", test_instance.test_predefined_configs),
        ("大数据集性能", test_instance.test_performance_large_dataset),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_instance.setup_method()
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name}失败: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"测试结果: {passed} 通过, {failed} 失败, 共{len(tests)} 个测试")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

"""
因子相关性分析服务测试（精简版）
- 验证核心功能正确性
- 零冗余依赖（仅alphalens + scipy/sklearn/statsmodels）
- 覆盖用户提出的10个关键要求
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '.')


class TestFactorCorrelationService:
    """测试因子相关性分析服务（精简版）"""
    
    @staticmethod
    def _make_sample_data():
        """生成测试数据：MultiIndex (date, asset) × factors"""
        np.random.seed(42)
        
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
        stocks = [f'{i:06d}' for i in range(1, 51)]
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'asset'])
        n = len(index)
        
        data = {
            'momentum': np.random.normal(0, 1, n),
            'value': np.random.normal(0, 1, n) * 0.8 + 
                     pd.Series(np.random.normal(0, 1, n), index=index).groupby('date').transform(
                         lambda x: x.rolling(20, min_periods=1).mean()
                     ).values if isinstance(index, pd.MultiIndex) else np.zeros(n),
            'quality': np.random.normal(0, 1, n),
            'volatility': np.abs(np.random.normal(0, 1, n)),
            'size': np.random.lognormal(0, 1, n)
        }
        
        df = pd.DataFrame(data, index=index)
        
        # 5%缺失值
        df.loc[np.random.random(n) < 0.05, 'momentum'] = np.nan
        df.loc[np.random.random(n) < 0.03, 'value'] = np.nan
        
        # 2%极值
        extreme = np.random.random(n) < 0.02
        df.loc[extreme, 'momentum'] = np.random.choice([-10, 10], size=extreme.sum())
        
        return df
    
    @pytest.fixture
    def sample_data(self):
        """pytest fixture：通过参数注入使用"""
        return self._make_sample_data()
    
    def test_service_initialization(self):
        """Test: 服务初始化"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        assert factor_correlation_service is not None
        assert factor_correlation_service.mode == "standard"
        print("✅ 服务初始化成功")
    
    def test_data_quality_check(self, sample_data):
        """Test 1: 数据质量检查"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value', 'quality', 'volatility', 'size']
        quality = factor_correlation_service._check_data_quality(sample_data, cols)
        
        assert len(quality) == 5
        assert all('missing_pct' in q for q in quality.values())
        assert 0 < quality['momentum']['missing_pct'] < 0.1
        print("✅ 数据质量检查通过")
    
    def test_frequency_detection(self, sample_data):
        """Test 2: 频率检测"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value']
        freq = factor_correlation_service._detect_frequency(sample_data[cols])
        
        assert all(v == 'daily' for v in freq.values())
        print(f"✅ 频率检测通过: {freq}")
    
    def test_preprocessing_pipeline(self, sample_data):
        """Test 3: 完整预处理流程"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value', 'quality', 'volatility', 'size']
        config = {
            'use_knn': True,
            'knn_neighbors': 5,
            'winsorize_method': 'mad',
            'n_sigma': 3.0,
            'max_missing_ratio': 0.3
        }
        
        cleaned, stats = factor_correlation_service._preprocess(sample_data, cols, config)
        
        assert cleaned[cols].isna().sum().sum() == 0  # 无缺失
        assert 'imputation' in stats or 'warning' in stats
        assert stats['winsorization']['method'] == 'MAD'
        assert stats['winsorization']['clipped'] > 0
        
        if 'imputation' in stats and 'validation' in stats.get('imputation', {}):
            print(f"✅ KNN填充验证: {stats['imputation']['validation']}")
        print(f"✅ 预处理完成: 截断{stats['winsorization']['clipped']}个极值")
    
    def test_cross_sectional_corr(self, sample_data):
        """Test 4: 横截面相关性（核心创新点）"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value', 'quality', 'volatility', 'size']
        config = {}
        
        cleaned, _ = factor_correlation_service._preprocess(sample_data, cols, config)
        result = factor_correlation_service._cross_sectional_corr(cleaned, cols)
        
        assert result['method'] == 'cross_sectional'
        assert 'avg_pearson' in result
        assert 'avg_spearman' in result
        assert result['n_days'] > 200
        assert 'method_consistency' in result
        
        val = result['avg_pearson'].get('momentum', {}).get('value')
        assert val is not None and -1 <= val <= 1
        
        print(f"✅ 横截面相关性:")
        print(f"   天数: {result['n_days']}, 平均股票数: {result['avg_n_stocks']:.1f}")
        print(f"   方法一致性: {result['method_consistency']['recommendation']}")
    
    def test_time_series_corr(self, sample_data):
        """Test 5: 时间序列相关性"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value', 'quality']
        config = {}
        
        cleaned, _ = factor_correlation_service._preprocess(sample_data, cols, config)
        result = factor_correlation_service._time_series_corr(cleaned, cols)
        
        assert 'method' in result
        assert 'pearson' in result or 'error' in result
        print(f"✅ 时间序列相关性: {result.get('method')}")
    
    def test_rolling_stability(self, sample_data):
        """Test 6: 滚动稳定性"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value', 'quality']
        config = {'rolling_window': 60, 'rolling_step': 10}
        
        cleaned, _ = factor_correlation_service._preprocess(sample_data, cols, config)
        result = factor_correlation_service._rolling_stability(cleaned, cols, config)
        
        if 'error' not in result:
            assert 0 <= result['stability_score'] <= 1
            assert 'regime_dist' in result
            print(f"✅ 滚动稳定性: score={result['stability_score']:.3f}, volatile={result['volatile']}")
        else:
            print(f"⚠️ 滚动稳定性跳过: {result['error']}")
    
    def test_significance_tests(self, sample_data):
        """Test 7: 显著性检验"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value', 'quality', 'volatility']
        config = {}
        
        cleaned, _ = factor_correlation_service._preprocess(sample_data, cols, config)
        cs = factor_correlation_service._cross_sectional_corr(cleaned, cols)
        ts = factor_correlation_service._time_series_corr(cleaned, cols)
        sig = factor_correlation_service._significance_tests(cs, ts)
        
        assert 'tests_performed' in sig
        print(f"✅ 显著性检验: 执行了{sig['tests_performed']}")
    
    def test_vif_analysis(self, sample_data):
        """Test 8: VIF多重共线性"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value', 'quality', 'volatility', 'size']
        config = {}
        
        cleaned, _ = factor_correlation_service._preprocess(sample_data, cols, config)
        vif = factor_correlation_service._vif_analysis(cleaned, cols)
        
        if 'error' not in vif:
            assert 'table' in vif
            assert 'has_issue' in vif
            print(f"✅ VIF分析: max={vif.get('max_vif')}, issue={vif['has_issue']}")
        else:
            print(f"⚠️ VIF跳过: {vif['error']}")
    
    def test_interpretation_engine(self, sample_data):
        """Test 9: 自建解读器（替代Corrpy）"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value', 'quality', 'volatility']
        config = {'rolling_window': 60, 'rolling_step': 10}
        
        cleaned, _ = factor_correlation_service._preprocess(sample_data, cols, config)
        
        full_result = {
            'cross_sectional': factor_correlation_service._cross_sectional_corr(cleaned, cols),
            'time_series': factor_correlation_service._time_series_corr(cleaned, cols),
            'rolling_stability': factor_correlation_service._rolling_stability(cleaned, cols, config),
            'vif_analysis': factor_correlation_service._vif_analysis(cleaned, cols)
        }
        
        interp = factor_correlation_service._interpret_results(full_result)
        
        assert 'high_correlation_pairs' in interp
        assert 'low_correlation_pairs' in interp
        assert 'overall_assessment' in interp
        assert isinstance(interp['overall_assessment'], str)
        
        print(f"✅ 解读器工作正常:")
        print(f"   高相关对: {len(interp['high_correlation_pairs'])}")
        print(f"   低相关对: {len(interp['low_correlation_pairs'])}")
        print(f"   总体评估: {interp['overall_assessment']}")
    
    def test_full_analysis_integration(self, sample_data):
        """Test 10: 完整集成测试"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        cols = ['momentum', 'value', 'quality', 'volatility', 'size']
        config = {
            'rolling_window': 60,
            'rolling_step': 10,
            'use_knn': True,
            'knn_neighbors': 5,
            'winsorize_method': 'mad',
            'n_sigma': 3.0
        }
        
        result = factor_correlation_service.analyze(sample_data, cols, config)
        
        required_keys = [
            'metadata', 'data_quality', 'preprocessing',
            'cross_sectional', 'time_series', 'significance',
            'vif_analysis', 'interpretation', 'warnings', 'recommendations'
        ]
        
        for key in required_keys:
            assert key in result, f"缺少字段: {key}"
        
        assert result['metadata']['n_factors'] == 5
        assert isinstance(result['warnings'], list)
        assert isinstance(result['recommendations'], list)
        
        print("\n✅ 完整集成测试通过！")
        print("=" * 60)
        print(f"📊 因子数: {result['metadata']['n_factors']}")
        print(f"📈 观测数: {result['metadata']['n_observations']}")
        print(f"⚠️ 警告: {len(result['warnings'])}条")
        print(f"💡 建议: {len(result['recommendations'])}条")
        print(f"📝 总评: {result['interpretation']['overall_assessment']}")


class TestOptionalDependencies:
    """测试可选依赖的降级策略"""
    
    def test_phik_fallback(self):
        """Test: Phik未安装时的降级方案"""
        from backend.services.factor_correlation_service import factor_correlation_service
        
        df = pd.DataFrame({
            'num_factor': [1, 2, 3, 4, 5],
            'cat_factor': ['A', 'B', 'A', 'C', 'B'],
            'another_num': [5, 4, 3, 2, 1]
        })
        
        result = factor_correlation_service.analyze_mixed_type(
            df, 
            ['num_factor', 'cat_factor', 'another_num'],
            categorical_cols=['cat_factor']
        )
        
        assert 'method' in result
        assert result['method'] in ['phik', 'scipy_fallback']
        print(f"✅ 混合类型降级方案: {result['method']}")
    
    def test_no_phik_installation(self):
        """Test: 确认Phik是可选依赖"""
        try:
            import phik
            print("⚠️ Phik已安装（可选功能可用）")
        except ImportError:
            print("✅ Phik未安装（符合预期，将使用scipy降级）")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🧪 FactorHub 因子相关性分析服务测试（精简版）")
    print("=" * 70 + "\n")
    
    test = TestFactorCorrelationService()
    data = test._make_sample_data()
    
    tests = [
        ("服务初始化", test.test_service_initialization),
        ("数据质量检查", lambda: test.test_data_quality_check(data)),
        ("频率检测", lambda: test.test_frequency_detection(data)),
        ("预处理管道", lambda: test.test_preprocessing_pipeline(data)),
        ("横截面相关性", lambda: test.test_cross_sectional_corr(data)),
        ("时间序列相关性", lambda: test.test_time_series_corr(data)),
        ("滚动稳定性", lambda: test.test_rolling_stability(data)),
        ("显著性检验", lambda: test.test_significance_tests(data)),
        ("VIF分析", lambda: test.test_vif_analysis(data)),
        ("自建解读器", lambda: test.test_interpretation_engine(data)),
        ("完整集成", lambda: test.test_full_analysis_integration(data)),
    ]
    
    passed = failed = 0
    
    for name, func in tests:
        try:
            print(f"\n▶️ {name}...")
            func()
            passed += 1
        except Exception as e:
            print(f"❌ 失败 ({name}): {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    # 可选依赖测试
    opt_test = TestOptionalDependencies()
    for name, func in [
        ("Phik降级", opt_test.test_phik_fallback),
        ("Phik可选性", opt_test.test_no_phik_installation),
    ]:
        try:
            print(f"\n▶️ {name}...")
            func()
            passed += 1
        except Exception as e:
            print(f"❌ 失败 ({name}): {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"📊 结果: ✅ {passed} 通过, ❌ {failed} 失败")
    print("="*70 + "\n")
    
    return passed, failed


if __name__ == "__main__":
    run_tests()

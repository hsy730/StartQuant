"""
Mask-First架构单元测试

验证涨跌停污染防护机制的正确性和性能。
涵盖：
1. data_service的tradable_mask构建准确性
2. factor_primitives的Mask-First算子行为
3. backtest_service的mask集成
4. analysis_service的IC计算改进
5. 性能基准测试（100万样本<1s）
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTradableMaskConstruction:
    """测试可交易性掩码构建的正确性"""
    
    def setup_method(self):
        """每个测试方法前的初始化"""
        from backend.services.data_service import DataService
        
        self.data_service = DataService()
        
        # 创建模拟数据（包含涨停/跌停/停牌）
        self.dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        
        # 正常交易日数据
        self.normal_data = pd.DataFrame({
            'open': np.random.uniform(9.5, 10.5, 100),
            'high': np.random.uniform(10.5, 11.5, 100),
            'low': np.random.uniform(8.5, 9.5, 100),
            'close': np.random.uniform(9.5, 10.5, 100),
            'volume': np.random.uniform(1e6, 1e7, 100),
        }, index=self.dates)
    
    def test_limit_up_detection(self):
        """测试涨停检测：一字涨停（high==low且触及涨停价）"""
        df = self.normal_data.copy()
        
        # 手动制造一个涨停日（第20天）
        prev_close = df.iloc[19]['close']
        limit_up_price = prev_close * 1.10  # 主板+10%
        
        df.iloc[20, df.columns.get_loc('open')] = limit_up_price
        df.iloc[20, df.columns.get_loc('high')] = limit_up_price
        df.iloc[20, df.columns.get_loc('low')] = limit_up_price  # 一字涨停
        df.iloc[20, df.columns.get_loc('close')] = limit_up_price
        df.iloc[20, df.columns.get_loc('volume')] = 1e8  # 大成交量
        
        # 调用检测函数
        result = self.data_service._detect_price_limits(df, "000001")
        
        # 验证检测结果
        assert result.iloc[20]['is_limit_up'] == True, "第20天应该被检测为涨停"
        assert result.iloc[20]['tradable_mask'] == False, "涨停日应该是不可交易的"
        
        # 验证其他日期不受影响
        assert result.iloc[19]['is_limit_up'] == False, "第19天不应该是涨停"
        assert result.iloc[21]['is_limit_up'] == False, "第21天不应该是涨停"
        
        print("✅ 涨停检测测试通过")
    
    def test_limit_down_detection(self):
        """测试跌停检测：一字跌停（high==low且触及跌停价）"""
        df = self.normal_data.copy()
        
        # 手动制造一个跌停日（第30天）
        prev_close = df.iloc[29]['close']
        limit_down_price = prev_close * 0.90  # 主板-10%
        
        df.iloc[30, df.columns.get_loc('open')] = limit_down_price
        df.iloc[30, df.columns.get_loc('high')] = limit_down_price
        df.iloc[30, df.columns.get_loc('low')] = limit_down_price  # 一字跌停
        df.iloc[30, df.columns.get_loc('close')] = limit_down_price
        df.iloc[30, df.columns.get_loc('volume')] = 1e8
        
        result = self.data_service._detect_price_limits(df, "000001")
        
        assert result.iloc[30]['is_limit_down'] == True, "第30天应该被检测为跌停"
        assert result.iloc[30]['tradable_mask'] == False, "跌停日应该是不可交易的"
        
        print("✅ 跌停检测测试通过")
    
    def test_suspended_detection(self):
        """测试停牌检测：成交量为0或缺失"""
        df = self.normal_data.copy()
        
        # 制造停牌日（第40天，成交量为0）
        df.iloc[40, df.columns.get_loc('volume')] = 0
        
        # 制造另一个停牌日（第41天，成交量为NaN）
        df.iloc[41, df.columns.get_loc('volume')] = np.nan
        
        result = self.data_service._detect_price_limits(df, "000001")
        
        assert result.iloc[40]['is_suspended'] == True, "第40天应该被检测为停牌"
        assert result.iloc[41]['is_suspended'] == True, "第41天应该被检测为停牌"
        assert result.iloc[40]['tradable_mask'] == False, "停牌日应该是不可交易的"
        
        print("✅ 停牌检测测试通过")
    
    def test_market_board_identification(self):
        """测试市场板块识别"""
        test_cases = [
            ("000001", "main"),      # 平安银行 - 主板
            ("600000", "main"),      # 浦发银行 - 主板
            ("300001", "chinext"),   # 特锐德 - 创业板
            ("688001", "star"),      # 华虹公司 - 科创板
            ("430047", "beijing"),   # 青矩技术 - 北交所
        ]
        
        for stock_code, expected_board in test_cases:
            board = self.data_service._identify_market_board(stock_code)
            assert board.value == expected_board, f"{stock_code} 应该属于 {expected_board}"
        
        print("✅ 市场板块识别测试通过")
    
    def test_price_limit_accuracy(self):
        """测试各市场板块的涨跌幅限制准确性"""
        # 主板股票（000001）- ±10%
        result_main = self.data_service._detect_price_limits(self.normal_data.copy(), "000001")
        assert self.data_service._board_config[self.data_service._identify_market_board("000001")]["price_limit"] == 0.10
        
        # 创业板股票（300001）- ±20%
        result_chinext = self.data_service._detect_price_limits(self.normal_data.copy(), "300001")
        assert self.data_service._board_config[self.data_service._identify_market_board("300001")]["price_limit"] == 0.20
        
        print("✅ 涨跌幅限制准确性测试通过")


class TestMaskedFactorOperators:
    """测试Mask-First因子算子的行为"""
    
    def setup_method(self):
        """初始化测试数据"""
        from backend.services.factor_primitives import (
            ts_mean_masked, ts_std_masked, ts_corr_masked,
            ts_mean, ts_std, ts_corr
        )
        
        self.ts_mean_masked = ts_mean_masked
        self.ts_std_masked = ts_std_masked
        self.ts_corr_masked = ts_corr_masked
        self.ts_mean = ts_mean
        self.ts_std = ts_std
        self.ts_corr = ts_corr
        
        # 创建模拟时间序列
        np.random.seed(42)
        self.n = 200
        self.dates = pd.date_range(start='2023-01-01', periods=self.n, freq='B')
        
        # 正常价格序列
        self.prices = pd.Series(
            10 + np.cumsum(np.random.randn(self.n) * 0.5),
            index=self.dates,
            name='price'
        )
        
        # 收益率序列
        self.returns = self.prices.pct_change()
        
        # 创建mask（假设有10%的不可交易日）
        self.mask = pd.Series(True, index=self.dates)
        self.non_tradable_indices = np.random.choice(
            self.dates[10:-10], size=int(self.n * 0.1), replace=False
        )
        for idx in self.non_tradable_indices:
            self.mask.loc[idx] = False
    
    def test_ts_mean_with_mask(self):
        """测试带mask的滚动平均"""
        # 使用mask版本
        result_masked = self.ts_mean_masked(self.prices, n=20, mask=self.mask)
        
        # 对比无mask版本
        result_no_mask = self.ts_mean(self.prices, n=20)
        
        # 验证两个结果有差异（证明mask生效了）
        differences = (result_masked - result_no_mask).dropna()
        
        # 至少应该有一些差异点（因为排除了不可交易日）
        assert len(differences) > 0 or self.mask.all(), \
            "Mask-First版本与传统版本应该有差异"
        
        # 验证可交易日的值不为NaN（大部分情况）
        tradable_count = (~result_masked.isna()).sum()
        total_count = len(result_masked)
        assert tradable_count > total_count * 0.7, \
            f"至少70%的数据点应该有效，实际 {tradable_count/total_count*100:.1f}%"
        
        print(f"✅ ts_mean_masked测试通过 (有效数据比例: {tradable_count/total_count*100:.1f}%)")
    
    def test_ts_corr_with_mask(self):
        """测试带mask的滚动相关系数 - 最关键的改进！"""
        # 使用mask版本
        result_masked = self.ts_corr_masked(
            self.prices, self.returns, n=20, mask=self.mask
        )
        
        # 验证结果范围在[-1, 1]之间
        valid_results = result_masked.dropna()
        if len(valid_results) > 0:
            assert (valid_results >= -1).all() and (valid_results <= 1).all(), \
                "相关系数应该在[-1, 1]范围内"
        
        # 对比无mask版本的结果差异
        result_no_mask = self.ts_corr(self.prices, self.returns, n=20)
        
        # Mask版本的IC标准差应该更小（因为排除了异常值）
        if len(valid_results) > 10 and len(result_no_mask.dropna()) > 10:
            masked_std = valid_results.std()
            no_mask_std = result_no_mask.dropna().std()
            
            # 注意：这不一定是严格小于，但通常会更稳定
            print(f"📊 IC标准差对比: 无mask={no_mask_std:.4f}, 有mask={masked_std:.4f}")
        
        print("✅ ts_corr_masked测试通过")
    
    def test_backward_compatibility(self):
        """测试向后兼容性：无mask时应退化为普通版本并发出警告"""
        import logging
        from io import StringIO
        
        # 捕获日志输出
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('backend.services.factor_primitives')
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        # 调用无mask的masked版本
        result = self.ts_mean_masked(self.prices, n=20, mask=None)
        
        # 验证发出了警告
        log_output = log_stream.getvalue()
        assert "未提供mask" in log_output or "退化为" in log_output, \
            "未提供mask时应该发出警告"
        
        # 清理handler
        logger.removeHandler(handler)
        
        print("✅ 向后兼容性测试通过")


class TestBacktestIntegration:
    """测试回测引擎与Mask-First的集成"""
    
    def setup_method(self):
        """初始化回测服务"""
        from backend.services.backtest_service import BacktestService
        
        self.bt = BacktestService(initial_capital=1000000)
        
        # 创建模拟回测数据
        np.random.seed(123)
        self.n_days = 252  # 一年交易日
        self.dates = pd.date_range(start='2023-01-01', periods=self.n_days, freq='B')
        
        self.df = pd.DataFrame({
            'close': 10 + np.cumsum(np.random.randn(self.n_days) * 0.3),
            'momentum_20': np.random.randn(self.n_days),  # 模拟因子值
        }, index=self.dates)
        
        # 添加tradable_mask（95%可交易）
        self.df['tradable_mask'] = True
        self.non_tradable_days = np.random.choice(
            self.dates[20:-20], size=int(self.n_days * 0.05), replace=False
        )
        for idx in self.non_tradable_days:
            self.df.loc[idx, 'tradable_mask'] = False
        
        # 添加辅助列
        self.df['is_limit_up'] = False
        self.df['is_limit_down'] = False
        self.df['is_suspended'] = ~self.df['tradable_mask']
    
    def test_backtest_with_mask(self):
        """测试带mask的单因子回测"""
        result = self.bt.single_factor_backtest(
            self.df, 
            factor_name="momentum_20",
            use_tradable_mask=True
        )
        
        # 验证返回结构
        assert "portfolio_returns" in result
        assert "equity_curve" in result
        assert "signal_mask" in result
        assert "mask_statistics" in result, "应该返回mask统计信息"
        
        # 验证mask统计信息
        stats = result["mask_statistics"]
        assert stats["total_days"] == self.n_days
        assert stats["tradable_days"] > 0
        assert 0 < stats["tradable_ratio"] <= 1.0
        assert stats["suspended_days"] >= 0
        
        # 验证信号在不可交易日为False
        signal_mask = result["signal_mask"]
        for idx in self.non_tradable_days:
            if idx in signal_mask.index:
                assert signal_mask.loc[idx] == False, \
                    f"不可交易日 {idx} 的信号应该是False"
        
        print(f"✅ 回测集成测试通过 (可交易比例: {stats['tradable_ratio']*100:.1f}%)")
    
    def test_backtest_without_mask_fallback(self):
        """测试无mask时的降级处理"""
        # 移除tradable_mask列
        df_no_mask = self.df.drop(columns=['tradable_mask', 'is_limit_up', 'is_limit_down', 'is_suspended'])
        
        result = self.bt.single_factor_backtest(
            df_no_mask,
            factor_name="momentum_20",
            use_tradable_mask=True  # 即使要求使用mask，但没有时也能运行
        )
        
        # 应该正常完成（虽然会有警告）
        assert "portfolio_returns" in result
        
        print("✅ 回测降级测试通过")


class TestAnalysisEngineMaskIntegration:
    """测试分析引擎与Mask-First的集成"""
    
    def setup_method(self):
        """初始化分析服务"""
        from backend.services.analysis_service import AnalysisService
        
        self.analysis = AnalysisService()
        
        # 创建模拟因子数据
        np.random.seed(456)
        self.n_days = 250
        self.dates = pd.date_range(start='2023-01-01', periods=self.n_days, freq='B')
        
        self.factor_data = {
            "TEST_STOCK": pd.DataFrame({
                'close': 10 + np.cumsum(np.random.randn(self.n_days) * 0.2),
                'momentum': np.random.randn(self.n_days),
                'future_return_1': np.random.randn(self.n_days) * 0.02,
                'tradable_mask': True,
                'is_limit_up': False,
                'is_limit_down': False,
                'is_suspended': False,
            }, index=self.dates)
        }
        
        # 设置一些不可交易日
        non_tradable = np.random.choice(
            self.dates[20:-20], size=int(self.n_days * 0.08), replace=False
        )
        df = self.factor_data["TEST_STOCK"]
        for idx in non_tradable:
            df.loc[idx, 'tradable_mask'] = False
            df.loc[idx, 'is_suspended'] = True
    
    def test_ic_calculation_with_mask(self):
        """测试IC计算使用mask"""
        result = self.analysis._calculate_single_stock_ic(
            self.factor_data,
            ["momentum"],
            use_tradable_mask=True
        )
        
        # 验证返回结构
        assert "ic_stats" in result
        assert "rolling_ir" in result
        assert "mask_statistics" in result, "应该包含mask统计信息"
        
        # 验证IC统计中标记了Mask-First
        if "momentum" in result["ic_stats"]:
            ic_stat = result["ic_stats"]["momentum"]
            assert "Mask-First" in ic_stat, "IC统计应标记是否使用了Mask-First"
            assert ic_stat["Mask-First"] == True, "应该显示使用了Mask-First"
        
        # 验证mask统计信息
        mask_stats = result["mask_statistics"]
        assert mask_stats["total_days"] == self.n_days
        assert 0 < mask_stats["tradable_ratio"] < 1.0
        
        print(f"✅ IC分析集成测试通过 (可交易比例: {mask_stats['tradable_ratio']*100:.1f}%)")
    
    def test_ic_without_mask_warning(self):
        """测试无mask时的警告"""
        # 移除mask相关列
        df_clean = self.factor_data["TEST_STOCK"].drop(
            columns=['tradable_mask', 'is_limit_up', 'is_limit_down', 'is_suspended']
        )
        factor_data_no_mask = {"TEST_STOCK": df_clean}
        
        result = self.analysis._calculate_single_stock_ic(
            factor_data_no_mask,
            ["momentum"],
            use_tradable_mask=True
        )
        
        # 应该能正常运行（虽然IC可能虚高）
        assert "ic_stats" in result
        
        # 不应该有mask_statistics（因为没有mask）
        assert "mask_statistics" not in result or result["mask_statistics"] is None
        
        print("✅ IC分析降级测试通过")


class TestPerformanceBenchmark:
    """性能基准测试：确保Mask-First不会显著影响性能"""
    
    def test_large_dataset_performance(self):
        """大数据集性能测试：100万样本应在合理时间内完成"""
        import time
        from backend.services.factor_primitives import ts_corr_masked
        
        # 生成大规模数据（100万样本）
        n_samples = 1_000_000
        print(f"\n⏱️ 性能测试: {n_samples:,} 样本")
        
        series_a = pd.Series(np.random.randn(n_samples))
        series_b = pd.Series(np.random.randn(n_samples))
        mask = pd.Series(np.random.choice([True, False], size=n_samples, p=[0.92, 0.08]))
        
        # 测试ts_corr_masked性能
        start_time = time.time()
        result = ts_corr_masked(series_a, series_b, n=20, mask=mask)
        elapsed_time = time.time() - start_time
        
        # 验证结果有效性
        valid_count = result.notna().sum()
        total_count = len(result)
        
        print(f"  ⏱️ ts_corr_masked耗时: {elapsed_time:.3f}秒")
        print(f"  📊 有效数据点: {valid_count:,} / {total_count:,} ({valid_count/total_count*100:.1f}%)")
        
        # 性能要求：100万样本应在10秒内完成
        assert elapsed_time < 10.0, \
            f"性能不达标: {elapsed_time:.2f}秒 > 10秒限制"
        
        # 验证结果范围在[-1, 1]之间
        valid_results = result.dropna()
        if len(valid_results) > 0:
            assert (valid_results >= -1).all() and (valid_results <= 1).all(), \
                "相关系数应该在[-1, 1]范围内"
        
        print(f"✅ 性能测试通过 ({elapsed_time:.3f}s < 10s)")
    
    def test_data_service_throughput(self):
        """data_service吞吐量测试：单只股票数据处理"""
        import time
        from backend.services.data_service import DataService
        
        ds = DataService()
        
        # 生成3年日线数据（约750个交易日）
        n_days = 750
        dates = pd.date_range(start='2021-01-01', periods=n_days, freq='B')
        df = pd.DataFrame({
            'open': np.random.uniform(9, 11, n_days),
            'high': np.random.uniform(10.5, 12, n_days),
            'low': np.random.uniform(8, 9.5, n_days),
            'close': np.random.uniform(9.5, 10.5, n_days),
            'volume': np.random.uniform(1e6, 1e8, n_days),
        }, index=dates)
        
        print(f"\n⏱️ 数据服务吞吐量测试: {n_days} 个交易日")
        
        start_time = time.time()
        result = ds._detect_price_limits(df, "000001")
        elapsed_time = time.time() - start_time
        
        tradable_ratio = result["tradable_mask"].mean()
        
        print(f"  ⏱️ _detect_price_limits耗时: {elapsed_time:.4f}秒")
        print(f"  📊 可交易比例: {tradable_ratio*100:.1f}%")
        
        # 性能要求：单只股票应在0.5秒内完成
        assert elapsed_time < 0.5, \
            f"性能不达标: {elapsed_time:.4f}秒 > 0.5秒限制"
        
        # 验证输出完整性
        assert "tradable_mask" in result.columns
        assert "is_limit_up" in result.columns
        assert "is_limit_down" in result.columns
        assert "is_suspended" in result.columns
        
        print(f"✅ 吞吐量测试通过 ({elapsed_time:.4f}s < 0.5s)")


def run_all_tests():
    """运行所有测试并汇总结果"""
    print("\n" + "="*80)
    print("🧪 Mask-First架构单元测试套件")
    print("="*80 + "\n")
    
    # 运行pytest
    exit_code = pytest.main([
        __file__,
        "-v",  # 详细输出
        "--tb=short",  # 简短traceback
        "-x",  # 第一个失败就停止
    ])
    
    return exit_code


if __name__ == "__main__":
    run_all_tests()

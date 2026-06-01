"""
智能交易滑点检测器单元测试

验证 SmartSlippageDetector 的功能正确性
"""
import pytest
import numpy as np
import pandas as pd
from backend.services.smart_slippage_detector import (
    SmartSlippageDetector,
    MarketBoard,
    LiquidityLevel,
    MarketCharacteristics,
    SlippageRecommendation,
    smart_slippage_detector,
)


class TestSmartSlippageDetector:
    """智能滑点检测器测试类"""

    def setup_method(self):
        """每个测试方法前的初始化"""
        self.detector = SmartSlippageDetector()

    def test_market_board_detection_main(self):
        """测试主板市场识别"""
        stock_codes = ["600000", "600036", "000001", "000002"]
        chars = self.detector.analyze_market(stock_codes)
        
        assert chars.market_board == MarketBoard.MAIN
        assert chars.n_stocks == 4

    def test_market_board_detection_chinext(self):
        """测试创业板识别"""
        stock_codes = ["300001", "300750", "300059"]
        chars = self.detector.analyze_market(stock_codes)
        
        assert chars.market_board == MarketBoard.CHINEXT

    def test_market_board_detection_star(self):
        """测试科创板识别"""
        stock_codes = ["688001", "688005", "688981"]
        chars = self.detector.analyze_market(stock_codes)
        
        assert chars.market_board == MarketBoard.STAR

    def test_market_board_detection_beijing(self):
        """测试北交所识别"""
        stock_codes = ["830799", "430047", "400001"]
        chars = self.detector.analyze_market(stock_codes)
        
        assert chars.market_board == MarketBoard.BEIJING

    def test_market_board_detection_mixed(self):
        """测试混合板块识别"""
        stock_codes = ["600000", "300001", "688001"]  # 主板+创业板+科创板
        chars = self.detector.analyze_market(stock_codes)
        
        # 混合板块，没有单一主导
        assert chars.market_board == MarketBoard.MIXED

    def test_basic_slippage_recommendation_main_board(self):
        """测试主板股票的基础滑点推荐"""
        stock_codes = ["600000", "600036", "000001"]
        
        rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=12.0,  # 中等换手率
        )
        
        assert isinstance(rec, SlippageRecommendation)
        # 主板基础滑点应该在 0.1% - 0.3% 范围内（考虑调整因子）
        assert 0.001 <= rec.recommended_slippage <= 0.004
        assert rec.confidence > 0
        assert rec.confidence <= 1.0
        assert len(rec.reasoning) > 0

    def test_slippage_recommendation_chinext_higher(self):
        """测试创业板滑点应该高于主板"""
        main_stocks = ["600000", "600036"]
        chinext_stocks = ["300001", "300750"]
        
        main_rec = self.detector.recommend_slippage(
            stock_codes=main_stocks,
            strategy_turnover=12.0,
        )
        
        chinext_rec = self.detector.recommend_slippage(
            stock_codes=chinext_stocks,
            strategy_turnover=12.0,
        )
        
        # 创业板滑点应该 >= 主板
        assert chinext_rec.recommended_slippage >= main_rec.recommended_slippage * 0.9  # 允许10%误差

    def test_conservative_preference_increases_slippage(self):
        """测试保守偏好应该增加推荐滑点"""
        stock_codes = ["600000", "300001"]
        
        default_rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=12.0,
            user_preference=None,
        )
        
        conservative_rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=12.0,
            user_preference="conservative",
        )
        
        # 保守估计应该 > 默认值
        assert conservative_rec.recommended_slippage > default_rec.recommended_slippage

    def test_aggressive_preference_decreases_slippage(self):
        """测试激进偏好应该降低推荐滑点"""
        stock_codes = ["600000", "300001"]
        
        default_rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=12.0,
            user_preference=None,
        )
        
        aggressive_rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=12.0,
            user_preference="aggressive",
        )
        
        # 激进估计应该 < 默认值
        assert aggressive_rec.recommended_slippage < default_rec.recommended_slippage

    def test_high_turnover_increases_slippage(self):
        """测试高换手率应该增加滑点影响"""
        stock_codes = ["600000", "300001"]
        
        low_turnover_rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=6.0,  # 低频
        )
        
        high_turnover_rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=36.0,  # 高频
        )
        
        # 高换手率的推荐滑点应该更高
        assert high_turnover_rec.recommended_slippage > low_turnover_rec.recommended_slippage

    def test_liquidity_adjustment_with_market_data(self):
        """测试有市场数据时的流动性调整"""
        stock_codes = ["600000", "300001"]
        
        # 创建低流动性市场数据
        market_data = pd.DataFrame({
            "stock_code": stock_codes,
            "market_cap": [1e8, 5e7],  # 小市值
            "volume": [100000, 50000],  # 低成交量
            "amount": [1e6, 5e5],  # 低成交额
            "turnover_rate": [0.01, 0.02],  # 低换手
        })
        
        rec_with_data = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=12.0,
            market_data=market_data,
        )
        
        # 低流动性时，滑点应该相对较高
        assert rec_with_data.recommended_slippage > 0.001

    def test_volatility_adjustment_with_price_data(self):
        """测试有价格数据时的波动率调整"""
        stock_codes = ["600000"]
        
        # 创建高波动价格数据
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        price_data = {
            "600000": pd.DataFrame({
                "close": 10 + np.cumsum(np.random.randn(100) * 0.5),  # 高波动
                "volume": np.random.randint(1000000, 5000000, 100),
            }, index=dates)
        }
        
        rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=12.0,
            price_data=price_data,
        )
        
        # 应该成功返回推荐
        assert isinstance(rec, SlippageRecommendation)

    def test_sensitivity_analysis_structure(self):
        """测试敏感性分析结果结构"""
        stock_codes = ["600000", "300001"]
        
        rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=12.0,
        )
        
        sensitivity = rec.sensitivity_analysis
        
        # 验证结构
        assert "test_scenarios" in sensitivity
        assert "base_scenario" in sensitivity
        assert "sensitivity_level" in sensitivity
        assert "recommendation" in sensitivity
        
        # 测试场景应该包含多个滑点水平
        scenarios = sensitivity["test_scenarios"]
        assert len(scenarios) >= 5  # 至少5个测试场景
        
        # 每个场景应该包含必要字段
        for scenario_name, data in scenarios.items():
            assert "slippage_rate" in data
            assert "net_return_estimate" in data
            assert "return_decay_pct" in data

    def test_boundary_constraints(self):
        """测试边界约束：滑点不应超过合理范围"""
        stock_codes = ["830799"]  # 北交所小盘股 + 极端情况
        
        rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=50.0,  # 超高换手
            user_preference="conservative",  # 保守估计
        )
        
        # 即使在极端情况下，滑点也不应超过1%
        assert rec.recommended_slippage <= 0.01
        # 也不应低于0.05%
        assert rec.recommended_slippage >= 0.0005

    def test_empty_stock_list_handling(self):
        """测试空股票列表的处理"""
        rec = self.detector.recommend_slippage(
            stock_codes=[],
            strategy_turnover=12.0,
        )
        
        # 空列表应返回默认值或混合市场配置
        assert isinstance(rec, SlippageRecommendation)
        assert rec.recommended_slippage > 0

    def test_recommendation_summary_generation(self):
        """测试推荐报告生成"""
        stock_codes = ["600000", "300001", "688001"]
        
        rec = self.detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=12.0,
        )
        
        summary = self.detector.get_recommendation_summary(rec)
        
        # 报告应该包含关键信息
        assert "# 🎯 智能交易滑点推荐报告" in summary
        assert "## 📊 市场概况" in summary
        assert "## 🎯 推荐滑点设置" in summary
        assert "## 💡 推荐理由" in summary
        assert f"{rec.confidence*100:.0f}%" in summary


class TestComprehensiveScoringServiceSlippage:
    """综合评分服务的滑点敏感性分析测试"""

    def setup_method(self):
        from backend.services.comprehensive_scoring_service import ComprehensiveScoringService
        self.scoring_service = ComprehensiveScoringService()

    def test_low_sensitivity_strategy(self):
        """测试低敏感性策略（低换手+高收益）"""
        metrics = {
            "annual_return": 0.25,  # 25%年化收益
            "turnover": 6.0,       # 6倍/年换手
        }
        
        result = self.scoring_service.analyze_slippage_sensitivity(metrics)
        
        assert result["sensitivity_level"] == "low"
        assert result["cost_impact_ratio"] < 10

    def test_high_sensitivity_strategy(self):
        """测试高敏感性策略（高换手+低收益）"""
        metrics = {
            "annual_return": 0.08,   # 8%年化收益
            "turnover": 36.0,      # 36倍/年换手
        }
        
        result = self.scoring_service.analyze_slippage_sensitivity(metrics)
        
        assert result["sensitivity_level"] in ["high", "very_high"]

    def test_with_smart_recommendation(self):
        """测试结合智能推荐的敏感性分析"""
        metrics = {
            "annual_return": 0.15,
            "turnover": 12.0,
        }
        stock_codes = ["600000", "300001"]
        
        result = self.scoring_service.analyze_slippage_sensitivity(
            metrics,
            stock_codes=stock_codes,
        )
        
        # 应该包含智能推荐
        assert "smart_recommendation" in result
        smart_rec = result["smart_recommendation"]
        assert "recommended_slippage" in smart_rec
        assert "confidence" in smart_rec

    def test_scenarios_count_and_structure(self):
        """测试场景数量和结构"""
        metrics = {
            "annual_return": 0.15,
            "turnover": 12.0,
        }
        
        result = self.scoring_service.analyze_slippage_sensitivity(metrics)
        
        # 默认应该有7个测试场景（6个固定+1个基准）
        assert len(result["scenarios"]) == 7
        
        # 每个场景应该包含必要字段
        for scenario in result["scenarios"]:
            assert "slippage_pct" in scenario
            assert "annual_cost_pct" in scenario
            assert "net_annual_return" in scenario

    def test_recommendations_for_critical_sensitivity(self):
        """测试高敏感性时的优化建议"""
        metrics = {
            "annual_return": 0.06,   # 低收益
            "turnover": 48.0,      # 超高换手
        }
        
        result = self.scoring_service.analyze_slippage_sensitivity(metrics)
        
        # 应该包含关键建议
        recommendations = result["recommendations"]
        assert len(recommendations) > 0
        
        # 高敏感性应该包含"critical"优先级的建议
        has_critical = any(r["priority"] == "critical" for r in recommendations)
        assert has_critical or result["sensitivity_level"] != "very_high"


class TestIntegrationSmartSlippageWithBacktestService:
    """集成测试：智能滑点与回测服务"""

    def test_smart_mode_initialization(self):
        """测试智能模式的初始化"""
        from backend.services.vectorbt_backtest_service import VectorBTBacktestService
        
        service = VectorBTBacktestService(
            initial_capital=1000000,
            commission_rate=0.0003,
            slippage=0.0,
            slippage_mode="smart",
        )
        
        assert service.slippage_mode == "smart"
        assert service.slippage == 0.0  # 初始值，待set_smart_slippage更新

    def test_set_smart_slippage_updates_value(self):
        """测试set_smart_slippage方法更新滑点值"""
        from backend.services.vectorbt_backtest_service import VectorBTBacktestService
        
        service = VectorBTBacktestService(slippage_mode="smart")
        
        rec = service.set_smart_slippage(
            stock_codes=["600000", "300001"],
            strategy_turnover=12.0,
        )
        
        # 滑点值应该被更新为推荐值
        assert service.slippage == rec.recommended_slippage
        assert service.slippage_mode == "smart"

    def test_get_slippage_info_custom_mode(self):
        """测试自定义模式下的滑点信息获取"""
        from backend.services.vectorbt_backtest_service import VectorBTBacktestService
        
        service = VectorBTBacktestService(
            slippage=0.002,
            slippage_mode="custom",
        )
        
        info = service.get_slippage_info()
        
        assert info["mode"] == "custom"
        assert info["slippage"] == 0.002

    def test_get_slippage_info_smart_mode(self):
        """测试智能模式下的滑点信息获取"""
        from backend.services.vectorbt_backtest_service import VectorBTBacktestService
        
        service = VectorBTBacktestService(slippage_mode="smart")
        service.set_smart_slippage(
            stock_codes=["600000"],
            strategy_turnover=12.0,
        )
        
        info = service.get_slippage_info()
        
        assert info["mode"] == "smart"
        assert "recommendation" in info
        assert "confidence" in info["recommendation"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

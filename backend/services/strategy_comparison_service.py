"""
策略对比服务 - 对比多个策略的表现
"""

import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats
from backend.services.strategy_registry import strategy_registry
from backend.utils.safe_math import safe_divide
from backend.constants import STATISTICAL_SIGNIFICANCE_ALPHA

logger = logging.getLogger(__name__)


class StrategyComparisonService:
    """策略对比服务"""

    def __init__(self):
        pass

    def compare_strategies(
        self,
        df: pd.DataFrame,
        strategy_names: List[str],
        strategy_params: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        """
        对比多个策略

        Args:
            df: 回测数据
            strategy_names: 策略名称列表
            strategy_params: 策略参数字典 {strategy_name: {param: value}}

        Returns:
            对比结果
        """
        if strategy_params is None:
            strategy_params = {}

        # Bug 3: 输入数据不可变，服务层入口必须 .copy()
        df = df.copy()

        results = {
            "strategies": {},
            "metrics_comparison": {},
            "equity_curves": {},
            "statistical_tests": {},
        }

        # 1. 运行所有策略
        for strategy_name in strategy_names:
            params = strategy_params.get(strategy_name, {})
            strategy = strategy_registry.get_strategy(strategy_name, **params)

            # 执行回测
            backtest_result = strategy.backtest(df)

            # 计算性能指标
            metrics = strategy.calculate_metrics(backtest_result["portfolio_returns"])

            results["strategies"][strategy_name] = {
                "backtest": backtest_result,
                "metrics": metrics,
            }

            # 保存净值曲线
            results["equity_curves"][strategy_name] = backtest_result["equity_curve"]

        # 2. 指标对比表
        results["metrics_comparison"] = self._create_metrics_table(results["strategies"])

        # 3. 统计显著性检验
        results["statistical_tests"] = self._perform_statistical_tests(results["strategies"], df)

        # 4. 排名
        results["ranking"] = self._rank_strategies(results["metrics_comparison"])

        return results

    def _create_metrics_table(self, strategies_results: Dict) -> pd.DataFrame:
        """
        创建指标对比表

        Args:
            strategies_results: 策略结果字典

        Returns:
            指标对比DataFrame
        """
        metrics_list = []

        for strategy_name, result in strategies_results.items():
            metrics = result["metrics"].copy()
            metrics["strategy"] = strategy_name
            metrics_list.append(metrics)

        df = pd.DataFrame(metrics_list)
        df = df.set_index("strategy")
        # 将NaN转为None，避免0.0掩盖数据缺失（符合规则6）
        df = df.where(df.notna(), None)

        return df

    def _perform_statistical_tests(self, strategies_results: Dict, df: pd.DataFrame) -> Dict:
        """
        执行统计显著性检验

        Args:
            strategies_results: 策略结果
            df: 数据

        Returns:
            检验结果
        """
        tests = {}
        strategy_names = list(strategies_results.keys())

        # 两两对比
        for i, strat1 in enumerate(strategy_names):
            for strat2 in strategy_names[i + 1 :]:  # noqa: E203
                returns1 = strategies_results[strat1]["backtest"]["portfolio_returns"]
                returns2 = strategies_results[strat2]["backtest"]["portfolio_returns"]

                # 对齐索引
                aligned_data = pd.DataFrame({"strategy1": returns1, "strategy2": returns2}).dropna()

                if len(aligned_data) < 30:  # 样本量太小
                    logger.warning(f"跳过 {strat1} vs {strat2} 对比：对齐后样本量不足 ({len(aligned_data)}<30)")
                    continue

                # T检验：均值是否显著不同
                t_stat, p_value = stats.ttest_ind(aligned_data["strategy1"], aligned_data["strategy2"], equal_var=False)

                # 配对T检验
                paired_t_stat, paired_p_value = stats.ttest_rel(aligned_data["strategy1"], aligned_data["strategy2"])

                # 相关系数
                correlation = aligned_data["strategy1"].corr(aligned_data["strategy2"])

                key = f"{strat1}_vs_{strat2}"
                tests[key] = {
                    "independent_t_test": {
                        "statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < STATISTICAL_SIGNIFICANCE_ALPHA,
                    },
                    "paired_t_test": {
                        "statistic": float(paired_t_stat),
                        "p_value": float(paired_p_value),
                        "significant": paired_p_value < STATISTICAL_SIGNIFICANCE_ALPHA,
                    },
                    "correlation": float(correlation) if pd.notna(correlation) else None,
                }

        return tests

    def _rank_strategies(self, metrics_df: pd.DataFrame) -> Dict:
        """
        对策略进行排名

        Args:
            metrics_df: 指标DataFrame

        Returns:
            排名结果
        """
        rankings = {}

        # 按不同指标排名（先检查列是否存在，避免KeyError）
        for col in ["annual_return", "sharpe_ratio", "max_drawdown"]:
            if col in metrics_df.columns:
                ascending = col != "max_drawdown"
                rankings[col] = metrics_df[col].rank(ascending=ascending).to_dict()

        # 综合得分（简单平均，仅基于已有指标排名）
        overall_scores = {}

        for strategy in metrics_df.index:
            rank_sum = 0
            rank_count = 0
            for col, col_ranking in rankings.items():
                if col != "overall" and strategy in col_ranking:
                    rank_val = col_ranking[strategy]
                    # Rule 7.41: 跳过NaN排名值（上游指标为None时rank()产生NaN）
                    if rank_val is not None and not (isinstance(rank_val, float) and np.isnan(rank_val)):
                        rank_sum += rank_val
                        rank_count += 1
            overall_scores[strategy] = safe_divide(rank_sum, rank_count, default=None)

        # 按综合得分排名（得分越低越好），排除 None 和 NaN
        valid_scores = {
            k: v for k, v in overall_scores.items() if v is not None and not (isinstance(v, float) and np.isnan(v))
        }
        sorted_overall = sorted(valid_scores.items(), key=lambda x: x[1])
        rankings["overall"] = {strategy: i + 1 for i, (strategy, _) in enumerate(sorted_overall)}

        return rankings

    def generate_comparison_report(self, comparison_result: Dict) -> str:
        """
        生成对比报告文本

        Args:
            comparison_result: 对比结果

        Returns:
            报告文本
        """
        report = "# 策略对比报告\n\n"

        # 1. 指标对比表
        report += "## 指标对比\n\n"
        metrics_df = comparison_result["metrics_comparison"]
        report += metrics_df.to_string()

        # 2. 排名
        report += "\n\n## 排名\n\n"
        overall_ranking = comparison_result["ranking"]["overall"]
        sorted_ranking = sorted(overall_ranking.items(), key=lambda x: x[1])

        for i, (strategy, rank) in enumerate(sorted_ranking, 1):
            report += f"{i}. {strategy}: 排名 {rank}\n"

        # 3. 统计检验
        report += "\n\n## 统计显著性检验\n\n"
        tests = comparison_result["statistical_tests"]

        for test_key, test_result in tests.items():
            report += f"### {test_key}\n"
            report += f"- 独立T检验 p值: {test_result['independent_t_test']['p_value']:.4f}"
            report += f" ({'显著' if test_result['independent_t_test']['significant'] else '不显著'})\n"
            report += f"- 配对T检验 p值: {test_result['paired_t_test']['p_value']:.4f}"
            report += f" ({'显著' if test_result['paired_t_test']['significant'] else '不显著'})\n"
            report += f"- 相关系数: {test_result['correlation']:.4f}\n\n"

        return report


# 全局策略对比服务实例
strategy_comparison_service = StrategyComparisonService()

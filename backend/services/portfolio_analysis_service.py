"""
组合分析服务 - 分析投资组合的暴露度和风险
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

import empyrical
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import HRPOpt


class PortfolioAnalysisService:
    """组合分析服务"""

    def __init__(self):
        pass

    def calculate_industry_exposure(
        self,
        positions: pd.DataFrame,
        industry_column: str = "industry",
        weight_column: str = "weight"
    ) -> Dict:
        """
        计算行业暴露度

        Args:
            positions: 持仓DataFrame，包含股票和权重
            industry_column: 行业列名
            weight_column: 权重列名

        Returns:
            行业暴露度字典
        """
        if industry_column not in positions.columns:
            return {"error": f"数据中缺少 {industry_column} 列"}

        if weight_column not in positions.columns:
            return {"error": f"数据中缺少 {weight_column} 列"}

        # 按行业汇总权重（同一行业多只股票时求和）
        industry_weights = positions.groupby(industry_column)[weight_column].sum()

        # 归一化
        total_weight = industry_weights.sum()
        if total_weight > 0:
            industry_exposure = industry_weights / total_weight
        else:
            industry_exposure = industry_weights

        # 转换为字典
        result = {
            "industry_exposure": industry_exposure.to_dict(),
            "max_exposure": float(industry_exposure.max()),
            "min_exposure": float(industry_exposure.min()),
            "concentration": float(industry_exposure.std()),
        }

        # 计算集中度（前3大行业占比）
        top3_exposure = industry_exposure.nlargest(3).sum()
        result["top3_concentration"] = float(top3_exposure)

        return result

    def calculate_factor_exposure(
        self,
        positions: pd.DataFrame,
        factor_data: Dict[str, pd.Series],
        weight_column: str = "weight",
    ) -> Dict:
        """
        计算因子暴露度

        Args:
            positions: 持仓DataFrame，包含股票和权重
            factor_data: 因子数据字典 {factor_name: factor_values}
            weight_column: 权重列名

        Returns:
            因子暴露度字典
        """
        factor_exposures = {}

        # 获取唯一的股票列表和对应的权重（假设每个股票只取第一条记录）
        if weight_column in positions.columns:
            stock_weights = positions.groupby("stock_code")[weight_column].first()
        else:
            return {"error": f"数据中缺少 {weight_column} 列"}

        for factor_name, factor_values in factor_data.items():
            try:
                # 因子值按股票代码对齐，计算加权平均暴露度
                if isinstance(factor_values, pd.Series):
                    aligned_factors = factor_values.reindex(stock_weights.index)
                    valid_mask = aligned_factors.notna() & stock_weights.notna()
                    if valid_mask.sum() > 0:
                        weighted_factor = (
                            stock_weights[valid_mask] * aligned_factors[valid_mask]
                        ).sum() / stock_weights[valid_mask].sum()
                    else:
                        weighted_factor = 0.0
                else:
                    # 标量因子值：所有股票相同，加权平均后仍是标量本身
                    weighted_factor = float(factor_values)

                factor_exposures[factor_name] = float(weighted_factor)

            except Exception as e:
                # 跳过计算失败的因子
                continue

        return {
            "factor_exposures": factor_exposures,
            "max_exposure": max([abs(v) for v in factor_exposures.values()]) if factor_exposures else 0.0,
        }

    def calculate_concentration(
        self,
        positions: pd.DataFrame,
        weight_column: str = "weight"
    ) -> Dict:
        """
        计算组合集中度

        Args:
            positions: 持仓DataFrame
            weight_column: 权重列名

        Returns:
            集中度指标
        """
        if weight_column not in positions.columns:
            return {"error": f"数据中缺少 {weight_column} 列"}

        weights = positions[weight_column].abs().dropna()

        if len(weights) == 0:
            return {
                "top10_concentration": 0.0,
                "herfindahl_index": 0.0,
                "gini_coefficient": 0.0,
            }

        # 1. 前十大持仓占比
        weights_sorted = weights.sort_values(ascending=False)
        top10_concentration = weights_sorted.head(10).sum() / weights.sum()

        # 2. Herfindahl指数（权重平方和）
        normalized_weights = weights / weights.sum()
        herfindahl_index = (normalized_weights ** 2).sum()

        # 3. 基尼系数
        gini_coefficient = self._calculate_gini(normalized_weights.values)

        return {
            "top10_concentration": float(top10_concentration),
            "herfindahl_index": float(herfindahl_index),
            "gini_coefficient": float(gini_coefficient),
        }

    def _calculate_gini(self, values: np.ndarray) -> float:
        """
        计算基尼系数

        Args:
            values: 权重值数组

        Returns:
            基尼系数
        """
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        if cumsum[-1] == 0:
            return 0.0
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        annual_trading_days: int = 252,
        risk_free_rate: float = 0.03,
    ) -> Dict:
        """
        计算组合风险指标

        Args:
            returns: 组合收益率序列
            benchmark_returns: 基准收益率序列（可选）
            annual_trading_days: 年化交易日数
            risk_free_rate: 无风险利率（年化）

        Returns:
            风险指标字典
        """
        returns_clean = returns.dropna()

        if len(returns_clean) == 0:
            return self._empty_risk_metrics()

        # 波动率
        volatility = returns_clean.std() * np.sqrt(annual_trading_days)

        # 下行波动率（Sortino标准：低于无风险利率的偏差）
        daily_rf = risk_free_rate / annual_trading_days
        downside_returns = returns_clean[returns_clean < daily_rf] - daily_rf
        downside_volatility = (
            downside_returns.std() * np.sqrt(annual_trading_days)
            if len(downside_returns) > 0
            else 0.0
        )

        # VaR (95%置信度)
        var_95 = returns_clean.quantile(0.05)

        # CVaR (条件VaR)
        cvar_95 = returns_clean[returns_clean <= var_95].mean()

        # 最大回撤
        cumulative = (1 + returns_clean).cumprod()
        peak = cumulative.cummax()
        drawdown = (peak - cumulative) / peak
        max_drawdown = drawdown.max()

        result = {
            "volatility": float(volatility),
            "downside_volatility": float(downside_volatility),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "max_drawdown": float(max_drawdown),
        }

        # 如果有基准，计算相对风险指标
        if benchmark_returns is not None:
            aligned_data = pd.DataFrame({
                "portfolio": returns_clean,
                "benchmark": benchmark_returns
            }).dropna()

            if len(aligned_data) > 0:
                # 跟踪误差
                excess_returns = aligned_data["portfolio"] - aligned_data["benchmark"]
                tracking_error = excess_returns.std() * np.sqrt(annual_trading_days)

                # Beta
                covariance = aligned_data["portfolio"].cov(aligned_data["benchmark"])
                benchmark_variance = aligned_data["benchmark"].var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else np.nan

                result["tracking_error"] = float(tracking_error)
                result["beta"] = float(beta)

        return result

    def _empty_risk_metrics(self) -> Dict:
        """返回空的风险指标"""
        return {
            "volatility": 0.0,
            "downside_volatility": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "max_drawdown": 0.0,
        }

    def analyze_portfolio_comprehensive(
        self,
        positions: pd.DataFrame,
        returns: pd.Series,
        factor_data: Optional[Dict[str, pd.Series]] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict:
        """
        综合分析投资组合

        Args:
            positions: 持仓数据
            returns: 收益率序列
            factor_data: 因子数据（可选）
            benchmark_returns: 基准收益率（可选）

        Returns:
            综合分析结果
        """
        result = {
            "industry_exposure": None,
            "factor_exposure": None,
            "concentration": None,
            "risk_metrics": None,
        }

        # 1. 行业暴露度
        if "industry" in positions.columns:
            result["industry_exposure"] = self.calculate_industry_exposure(positions)

        # 2. 因子暴露度
        if factor_data:
            result["factor_exposure"] = self.calculate_factor_exposure(
                positions, factor_data
            )

        # 3. 集中度
        result["concentration"] = self.calculate_concentration(positions)

        # 4. 风险指标
        result["risk_metrics"] = self.calculate_risk_metrics(
            returns, benchmark_returns
        )

        return result

    def optimize_weights(
        self,
        factor_returns: pd.DataFrame,
        method: str = "equal_weight",
        risk_free_rate: float = 0.03,
        **kwargs
    ) -> Dict:
        """
        优化因子权重

        Args:
            factor_returns: 因子收益率 DataFrame (columns=因子名, index=时间)
            method: 权重优化方法
                - "equal_weight": 等权重
                - "ic_weight": IC加权（基于因子历史表现）
                - "risk_parity": 风险平价
                - "max_sharpe": 最大夏普比率
                - "min_variance": 最小方差
            risk_free_rate: 无风险利率（年化）
            **kwargs: 其他参数

        Returns:
            优化结果字典，包含权重和统计信息
        """
        if factor_returns.empty:
            return {
                "weights": {},
                "method": method,
                "error": "因子收益率为空"
            }

        # 预处理因子收益率，处理 NaN 和 Inf 值
        factor_returns = factor_returns.replace([np.inf, -np.inf], np.nan)
        factor_returns = factor_returns.fillna(0.0)

        n_factors = len(factor_returns.columns)

        # 初始化权重
        weights = None
        extra_info = {}

        # 1. 等权重
        if method == "equal_weight":
            weights = pd.Series(1.0 / n_factors, index=factor_returns.columns)
            extra_info["note"] = "等权重分配"

        # 2. IC加权（基于因子均值和夏普比率）
        elif method == "ic_weight":
            # 计算每个因子的IC（均值和IR）
            factor_stats = {}
            for factor in factor_returns.columns:
                returns = factor_returns[factor].dropna()
                if len(returns) > 0:
                    mean_return = returns.mean()
                    std_return = returns.std()
                    ir = mean_return / std_return if std_return > 0 else 0

                    factor_stats[factor] = {
                        "mean": mean_return,
                        "std": std_return,
                        "ir": ir
                    }

            # 基于IR加权（IR为负的设为0）
            ir_values = pd.Series({
                factor: max(0, stats["ir"])
                for factor, stats in factor_stats.items()
            })

            if ir_values.sum() == 0:
                # 如果所有IR都<=0，回退到等权重
                weights = pd.Series(1.0 / n_factors, index=factor_returns.columns)
            else:
                weights = ir_values / ir_values.sum()

            extra_info["factor_stats"] = factor_stats

        # 3. 风险平价
        elif method == "risk_parity":
            # 计算每个因子的波动率
            volatilities = factor_returns.std()

            # 风险平价权重与波动率成反比
            inv_vol = 1.0 / volatilities
            weights = inv_vol / inv_vol.sum()

        # 4. 最大夏普比率（使用PyPortfolioOpt实现均值-方差优化）
        elif method == "max_sharpe":
            if n_factors >= 2:
                mu = expected_returns.mean_historical_return(factor_returns, frequency=252)
                S = risk_models.sample_cov(factor_returns, frequency=252)
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = pd.Series(ef.clean_weights(), index=factor_returns.columns)
                extra_info["optimization_status"] = "success"
            else:
                weights = pd.Series(1.0, index=factor_returns.columns)
                extra_info["optimization_status"] = "skipped: only one factor"

        # 5. 最小方差（使用PyPortfolioOpt实现最小方差优化）
        elif method == "min_variance":
            if n_factors >= 2:
                mu = expected_returns.mean_historical_return(factor_returns, frequency=252)
                S = risk_models.sample_cov(factor_returns, frequency=252)
                ef = EfficientFrontier(mu, S)
                ef.min_volatility()
                weights = pd.Series(ef.clean_weights(), index=factor_returns.columns)
                extra_info["optimization_status"] = "success"
            else:
                weights = pd.Series(1.0, index=factor_returns.columns)
                extra_info["optimization_status"] = "skipped: only one factor"

        else:
            return {
                "weights": {},
                "method": method,
                "error": f"不支持的权重优化方法: {method}"
            }

        # ========== 统一计算基于权重的组合指标 ==========

        # 计算加权期望收益（年化）
        mean_returns = factor_returns.mean()  # 每个因子的平均收益
        weighted_return = (weights * mean_returns).sum() * 252  # 年化

        # 计算加权波动率（年化）
        # 组合方差 = w' * Σ * w
        cov_matrix = factor_returns.cov() * 252  # 年化协方差矩阵
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        weighted_volatility = np.sqrt(portfolio_variance)

        # 计算夏普比率
        sharpe_ratio = (weighted_return - risk_free_rate) / weighted_volatility if weighted_volatility > 0 else 0

        # 构建返回结果
        result = {
            "weights": weights.to_dict(),
            "method": method,
            "expected_return": float(weighted_return),
            "expected_volatility": float(weighted_volatility),
            "sharpe_ratio": float(sharpe_ratio),
        }

        # 添加额外信息
        result.update(extra_info)

        return result

    def calculate_combined_factor_score(
        self,
        factor_data: Dict[str, pd.Series],
        weights: Dict[str, float],
        normalize: bool = True
    ) -> pd.Series:
        """
        根据权重计算综合因子得分

        Args:
            factor_data: 因子数据字典 {factor_name: factor_series}
            weights: 因子权重字典 {factor_name: weight}
            normalize: 是否标准化因子值

        Returns:
            综合因子得分序列
        """
        # 获取共同的索引
        common_index = None
        for factor_name, factor_series in factor_data.items():
            if common_index is None:
                common_index = factor_series.index
            else:
                common_index = common_index.intersection(factor_series.index)

        if common_index is None or len(common_index) == 0:
            return pd.Series(dtype=float)

        # 标准化因子值
        if normalize:
            normalized_factors = {}
            for factor_name, factor_series in factor_data.items():
                aligned_factor = factor_series.reindex(common_index)
                mean = aligned_factor.mean()
                std = aligned_factor.std()
                if std > 0:
                    normalized_factors[factor_name] = (aligned_factor - mean) / std
                else:
                    normalized_factors[factor_name] = aligned_factor - mean
        else:
            normalized_factors = {
                name: series.reindex(common_index)
                for name, series in factor_data.items()
            }

        # 计算加权得分
        combined_score = pd.Series(0.0, index=common_index)

        for factor_name, weight in weights.items():
            if factor_name in normalized_factors:
                combined_score += weight * normalized_factors[factor_name]

        # 处理特殊值（NaN, Inf），以避免 JSON 序列化错误
        combined_score = combined_score.replace([np.inf, -np.inf], np.nan)
        combined_score = combined_score.fillna(0.0)

        return combined_score

    def compare_weight_methods(
        self,
        factor_returns: pd.DataFrame,
        methods: List[str] = None,
        risk_free_rate: float = 0.03
    ) -> Dict:
        """
        比较不同权重优化方法的效果

        Args:
            factor_returns: 因子收益率
            methods: 要比较的方法列表（默认比较所有方法）
            risk_free_rate: 无风险利率

        Returns:
            比较结果字典，格式与前端期望匹配
        """
        if methods is None:
            methods = ["equal_weight", "ic_weight", "risk_parity", "max_sharpe"]

        results = {}

        for method in methods:
            optimization_result = self.optimize_weights(
                factor_returns,
                method=method,
                risk_free_rate=risk_free_rate
            )

            if "error" not in optimization_result:
                results[method] = {
                    "annual_return": optimization_result["expected_return"],
                    "volatility": optimization_result["expected_volatility"],
                    "sharpe_ratio": (
                        (optimization_result["expected_return"] - risk_free_rate) / optimization_result["expected_volatility"]
                        if optimization_result["expected_volatility"] > 0
                        else 0
                    ),
                }

        return results

    def _get_method_display_name(self, method: str) -> str:
        """获取方法的显示名称"""
        name_map = {
            "equal_weight": "等权重",
            "ic_weight": "IC加权",
            "risk_parity": "风险平价",
            "max_sharpe": "最大夏普",
            "min_variance": "最小方差",
        }
        return name_map.get(method, method)


# 全局组合分析服务实例
portfolio_analysis_service = PortfolioAnalysisService()

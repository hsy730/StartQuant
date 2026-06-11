"""
组合分析服务 - 分析投资组合的暴露度和风险
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import empyrical
import numpy as np

from pypfopt import EfficientFrontier, HRPOpt, risk_models, expected_returns

from backend.utils.safe_math import safe_divide
from backend.utils.weight_utils import normalize_weights
from backend.services.risk_metrics import calculate_relative_metrics

logger = logging.getLogger(__name__)


class PortfolioAnalysisService:
    """组合分析服务"""

    def __init__(self):
        pass

    def calculate_industry_exposure(
        self, positions: pd.DataFrame, industry_column: str = "industry", weight_column: str = "weight"
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
            industry_exposure = safe_divide(industry_weights, total_weight, default=0.0)
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
                        weight_sum = stock_weights[valid_mask].sum()
                        weighted_factor = safe_divide(
                            (stock_weights[valid_mask] * aligned_factors[valid_mask]).sum(),
                            weight_sum,
                            default=0.0,
                        )
                    else:
                        weighted_factor = 0.0
                else:
                    # 标量因子值：所有股票相同，加权平均后仍是标量本身
                    weighted_factor = float(factor_values)

                factor_exposures[factor_name] = float(weighted_factor)

            except Exception as e:
                # 跳过计算失败的因子
                logger.warning(f"因子 {factor_name} 曝露度计算失败: {e}")
                continue

        return {
            "factor_exposures": factor_exposures,
            "max_exposure": max([abs(v) for v in factor_exposures.values()]) if factor_exposures else 0.0,
        }

    def calculate_concentration(self, positions: pd.DataFrame, weight_column: str = "weight") -> Dict:
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
        top10_concentration = safe_divide(
            weights_sorted.head(10).sum(),
            weights.sum(),
            default=0.0,
        )

        # 2. Herfindahl指数（权重平方和）
        normalized_weights = normalize_weights(weights, default_equal=False)
        herfindahl_index = (normalized_weights**2).sum()

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
        if abs(cumsum[-1]) < 1e-10:
            return 0.0
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        annual_trading_days: int = 252,
        risk_free_rate: float = 0.03,
    ) -> Dict:
        """计算组合风险指标，委托 risk_metrics 统一入口"""
        from backend.services.risk_metrics import calculate_risk_metrics as calc_risk, _empty_metrics

        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return _empty_metrics()

        result = calc_risk(returns_clean, risk_free_rate, annual_trading_days)

        # 如果有基准，计算相对风险指标（委托risk_metrics统一入口，符合规则2）
        if benchmark_returns is not None:
            relative = calculate_relative_metrics(
                returns_clean, benchmark_returns, risk_free_rate=risk_free_rate, annual_trading_days=annual_trading_days
            )
            result["tracking_error"] = relative.get("tracking_error")
            result["beta"] = relative.get("beta")

        return result

    def _empty_risk_metrics(self) -> Dict:
        """返回空的风险指标"""
        from backend.services.risk_metrics import _empty_metrics

        return _empty_metrics()

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
            result["factor_exposure"] = self.calculate_factor_exposure(positions, factor_data)

        # 3. 集中度
        result["concentration"] = self.calculate_concentration(positions)

        # 4. 风险指标
        result["risk_metrics"] = self.calculate_risk_metrics(returns, benchmark_returns)

        return result

    def optimize_weights(
        self, factor_returns: pd.DataFrame, method: str = "equal_weight", risk_free_rate: float = 0.03, **kwargs
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
            return {"weights": {}, "method": method, "error": "因子收益率为空"}

        # 预处理因子收益率，处理 NaN 和 Inf 值
        # NaN 不填充为0.0（0.0在Z-score空间意味着"平均水平"，严重误导优化）
        # 先前向填充（保留时间序列连续性），再删除剩余NaN
        factor_returns = factor_returns.replace([np.inf, -np.inf], np.nan)
        factor_returns = factor_returns.ffill().dropna()

        n_factors = len(factor_returns.columns)

        # 初始化权重
        weights = None
        extra_info = {}

        # 1. 等权重
        if method == "equal_weight":
            weights = pd.Series(1.0 / n_factors, index=factor_returns.columns)
            extra_info["note"] = "等权重分配"

        # 2. IC加权 — 委托WeightOptimizer统一入口（规则2/7.13）
        elif method == "ic_weight":
            from backend.services.weight_optimizer_service import WeightOptimizer

            optimizer = WeightOptimizer()

            # 获取因子值数据（kwargs中传入）
            factor_values = kwargs.get("factor_values", None)
            factor_data_dict = kwargs.get("factor_data_dict", None)

            # 构建factor_values字典格式
            if factor_values is not None and isinstance(factor_values, pd.DataFrame):
                fv_dict = {col: factor_values[col] for col in factor_values.columns}
            else:
                fv_dict = {col: pd.Series(dtype=float) for col in factor_returns.columns}

            # IC加权需要stock_returns参数（因子值与股票收益的相关性）
            stock_returns = kwargs.get("stock_returns")
            if stock_returns is None:
                logger.warning("IC加权需要stock_returns参数，回退到等权")
                weights = pd.Series(1.0 / n_factors, index=factor_returns.columns)
                extra_info["note"] = "IC加权回退到等权：缺少stock_returns参数"
            else:
                # 使用统一入口计算IC加权权重
                result = optimizer.calculate_weights(
                    factor_values=fv_dict,
                    factor_names=list(factor_returns.columns),
                    method="ic_weight",
                    returns=stock_returns,
                    factor_data_dict=factor_data_dict,
                )
                weights = pd.Series(result["weights"])
                extra_info["ic_values"] = {k: v for k, v in result["weights"].items()}

        # 3. 风险平价 — 使用HRPOpt（层次风险平价，考虑因子间相关性）
        elif method == "risk_parity":
            if n_factors >= 2:
                try:
                    hrp = HRPOpt(factor_returns)
                    _raw_weights = hrp.optimize()  # noqa: F841
                    clean_weights = hrp.clean_weights()
                    weights = pd.Series(clean_weights, index=factor_returns.columns)
                    extra_info["optimization_status"] = "success"
                except Exception as e:
                    logger.warning(f"PyPortfolioOpt risk_parity失败: {e}，回退到等权重")
                    weights = pd.Series(1.0 / n_factors, index=factor_returns.columns)
                    extra_info["optimization_status"] = f"fallback: {str(e)}"
            else:
                weights = pd.Series(1.0, index=factor_returns.columns)
                extra_info["optimization_status"] = "skipped: only one factor"

        # 4. 最大夏普比率（使用PyPortfolioOpt实现均值-方差优化）
        elif method == "max_sharpe":
            if n_factors >= 2:
                try:
                    mu = expected_returns.mean_historical_return(factor_returns, returns_data=True, frequency=252)
                    S = risk_models.sample_cov(factor_returns, returns_data=True, frequency=252)
                    ef = EfficientFrontier(mu, S)
                    ef.max_sharpe(risk_free_rate=risk_free_rate)
                    weights = pd.Series(ef.clean_weights(), index=factor_returns.columns)
                    extra_info["optimization_status"] = "success"
                except Exception as e:
                    logger.warning(f"PyPortfolioOpt max_sharpe失败: {e}，回退到等权重")
                    weights = pd.Series(1.0 / n_factors, index=factor_returns.columns)
                    extra_info["optimization_status"] = f"fallback: {str(e)}"
            else:
                weights = pd.Series(1.0, index=factor_returns.columns)
                extra_info["optimization_status"] = "skipped: only one factor"

        # 5. 最小方差（使用PyPortfolioOpt实现最小方差优化）
        elif method == "min_variance":
            if n_factors >= 2:
                try:
                    mu = expected_returns.mean_historical_return(factor_returns, returns_data=True, frequency=252)
                    S = risk_models.sample_cov(factor_returns, returns_data=True, frequency=252)
                    ef = EfficientFrontier(mu, S)
                    ef.min_volatility()
                    weights = pd.Series(ef.clean_weights(), index=factor_returns.columns)
                    extra_info["optimization_status"] = "success"
                except Exception as e:
                    logger.warning(f"PyPortfolioOpt min_variance失败: {e}，回退到等权重")
                    weights = pd.Series(1.0 / n_factors, index=factor_returns.columns)
                    extra_info["optimization_status"] = f"fallback: {str(e)}"
            else:
                weights = pd.Series(1.0, index=factor_returns.columns)
                extra_info["optimization_status"] = "skipped: only one factor"

        else:
            return {"weights": {}, "method": method, "error": f"不支持的权重优化方法: {method}"}

        # ========== 统一计算基于权重的组合指标 ==========

        # 确保权重归一化（和为1），避免收益被错误缩放
        weight_sum = weights.sum()
        if abs(weight_sum) < 1e-10:
            return {"weights": {}, "method": method, "error": "权重总和为0，无法计算组合指标"}
        if abs(weight_sum - 1.0) > 1e-6:
            weights = normalize_weights(weights)

        # 计算加权期望收益（年化）— 使用几何复利（Rule 7.32）
        weighted_daily_returns = (factor_returns * weights).sum(axis=1)
        weighted_return = float(empyrical.annual_return(weighted_daily_returns, period="daily"))

        # 计算加权波动率（年化）
        # 组合方差 = w' * Σ * w
        cov_matrix = factor_returns.cov() * 252  # 年化协方差矩阵
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        # 数值精度保护：方差可能因浮点误差略小于0，截断到0
        portfolio_variance = max(0.0, portfolio_variance)
        weighted_volatility = np.sqrt(portfolio_variance)

        # 计算夏普比率（不可计算时返回None，符合规则6）
        sharpe_ratio = safe_divide(weighted_return - risk_free_rate, weighted_volatility, default=None)

        # 构建返回结果
        result = {
            "weights": weights.to_dict(),
            "method": method,
            "expected_return": float(weighted_return),
            "expected_volatility": float(weighted_volatility),
            "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio is not None else None,
        }

        # 添加额外信息
        result.update(extra_info)

        return result

    def calculate_combined_factor_score(
        self, factor_data: Dict[str, pd.Series], weights: Dict[str, float], normalize: bool = True
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
                if std > 1e-10:
                    normalized_factors[factor_name] = safe_divide(aligned_factor - mean, std, default=0.0)
                else:
                    normalized_factors[factor_name] = aligned_factor - mean
        else:
            normalized_factors = {name: series.reindex(common_index) for name, series in factor_data.items()}

        # 计算加权得分
        combined_score = pd.Series(0.0, index=common_index)

        for factor_name, weight in weights.items():
            if factor_name in normalized_factors:
                combined_score += weight * normalized_factors[factor_name]

        # 处理特殊值（Inf），以避免 JSON 序列化错误
        # 注意：NaN 保留不填充为0.0，因为在Z-score空间中0.0意味着"平均水平"，
        # 误导性很强。下游应自行处理NaN（如显示N/A）。
        combined_score = combined_score.replace([np.inf, -np.inf], np.nan)

        return combined_score

    def compare_weight_methods(
        self, factor_returns: pd.DataFrame, methods: List[str] = None, risk_free_rate: float = 0.03
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
            optimization_result = self.optimize_weights(factor_returns, method=method, risk_free_rate=risk_free_rate)

            if "error" not in optimization_result:
                results[method] = {
                    "annual_return": optimization_result["expected_return"],
                    "volatility": optimization_result["expected_volatility"],
                    "sharpe_ratio": safe_divide(
                        optimization_result["expected_return"] - risk_free_rate,
                        optimization_result["expected_volatility"],
                        default=None,
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

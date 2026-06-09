"""
权重优化服务 — 统一的投资组合权重计算逻辑

消除 portfolio.py 中 /optimize-weights 和 /compare-methods 的代码重复
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from backend.utils.safe_math import safe_divide

logger = logging.getLogger(__name__)


class WeightOptimizer:
    """投资组合权重优化器"""

    # 支持的权重计算方法
    METHODS = ["equal_weight", "ic_weight", "ir_weight", "max_sharpe", "min_variance", "risk_parity"]

    def calculate_weights(
        self,
        factor_values: dict,
        factor_names: list,
        method: str = "equal_weight",
        returns: Optional[pd.Series] = None,
    ) -> Dict:
        """
        计算投资组合权重

        Args:
            factor_values: 因子名称到因子值序列的映射
            factor_names: 因子名称列表
            method: 权重计算方法
            returns: 收益率序列（IC/IR加权需要）

        Returns:
            Dict: {"weights": {factor: weight}, "method": method}
        """
        if method not in self.METHODS:
            logger.warning(f"未知权重方法: {method}，回退到等权")
            method = "equal_weight"

        # 防御性复制：避免修改传入数据
        factor_values = {k: v.copy() for k, v in factor_values.items()}

        if method == "equal_weight":
            return self._equal_weight(factor_names)
        elif method == "ic_weight":
            return self._ic_weight(factor_values, factor_names, returns)
        elif method == "ir_weight":
            return self._ir_weight(factor_values, factor_names, returns)
        elif method == "max_sharpe":
            return self._max_sharpe(factor_values, factor_names, returns)
        elif method == "min_variance":
            return self._min_variance(factor_values, factor_names)
        elif method == "risk_parity":
            return self._risk_parity(factor_values, factor_names)

        return self._equal_weight(factor_names)

    def _equal_weight(self, factor_names: list) -> Dict:
        """等权重"""
        n = len(factor_names)
        weight = 1.0 / n if n > 0 else 0.0
        weights = {name: weight for name in factor_names}
        return {"weights": weights, "method": "equal_weight"}

    def _ic_weight(self, factor_values, factor_names, returns) -> Dict:
        """IC加权 — 因子IC绝对值归一化"""
        if returns is None or len(returns.dropna()) < 20:
            logger.debug("IC加权数据不足，回退到等权")
            return self._equal_weight(factor_names)

        ic_values = {}
        for factor_name in factor_names:
            values = factor_values.get(factor_name)
            if values is not None and len(values.dropna()) > 0:
                aligned_data = pd.DataFrame({
                    'factor': values,
                    'returns': returns
                }).dropna()

                if len(aligned_data) > 10:
                    ic = aligned_data['factor'].corr(aligned_data['returns'])
                    ic_values[factor_name] = abs(ic) if not np.isnan(ic) else 0.0
                else:
                    ic_values[factor_name] = 0.0
            else:
                ic_values[factor_name] = 0.0

        total_ic = sum(ic_values.values())
        if total_ic == 0:
            return self._equal_weight(factor_names)

        weights = {k: safe_divide(v, total_ic, default=1.0/len(factor_names)) for k, v in ic_values.items()}
        return {"weights": weights, "method": "ic_weight"}

    def _ir_weight(self, factor_values, factor_names, returns) -> Dict:
        """IR加权 — 因子信息比率绝对值归一化"""
        from backend.utils.safe_math import safe_ir
        if returns is None or len(returns.dropna()) < 20:
            logger.debug("IR加权数据不足，回退到等权")
            return self._equal_weight(factor_names)

        ir_values = {}
        for factor_name in factor_names:
            values = factor_values.get(factor_name)
            if values is not None and len(values.dropna()) > 0:
                aligned_data = pd.DataFrame({
                    'factor': values,
                    'returns': returns
                }).dropna()

                if len(aligned_data) > 20:
                    ic_series = aligned_data['factor'].rolling(
                        window=20, min_periods=10
                    ).corr(aligned_data['returns'])
                    ic_mean = ic_series.mean()
                    ic_std = ic_series.std()
                    ir = safe_ir(float(ic_mean), float(ic_std), default=0.0)
                    ir_values[factor_name] = abs(ir) if ir is not None else 0.0
                else:
                    ir_values[factor_name] = 0.0
            else:
                ir_values[factor_name] = 0.0

        total_ir = sum(ir_values.values())
        if total_ir == 0:
            return self._equal_weight(factor_names)

        weights = {k: safe_divide(v, total_ir, default=1.0/len(factor_names)) for k, v in ir_values.items()}
        return {"weights": weights, "method": "ir_weight"}

    def _max_sharpe(self, factor_values, factor_names, returns) -> Dict:
        """最大夏普比率 — 使用pyportfolioopt（规则0）

        注意：因子值不是价格，pct_change()无意义。
        正确做法是使用因子值的一阶差分(diff)作为因子收益代理，
        或要求调用方传入实际收益率。
        """
        try:
            from pypfopt import EfficientFrontier, risk_models, expected_returns
            # 构建因子收益矩阵
            # 因子值不是价格，不能对因子值求pct_change（如Z-score从-1到1，pct_change=-200%无意义）
            # 使用diff()（一阶差分）作为因子收益的代理指标
            factor_df = pd.DataFrame(factor_values)
            factor_returns = factor_df[factor_names].diff().dropna()
            if len(factor_returns) < 20:
                return self._equal_weight(factor_names)

            mu = expected_returns.mean_historical_return(factor_returns)
            S = risk_models.sample_cov(factor_returns)
            ef = EfficientFrontier(mu, S)
            _raw_weights = ef.max_sharpe()
            clean_weights = ef.clean_weights()

            weights = {k: v for k, v in clean_weights.items() if k in factor_names}
            return {"weights": weights, "method": "max_sharpe"}
        except Exception as e:
            logger.warning(f"PyPortfolioOpt max_sharpe失败: {e}，回退到等权重")
            return self._equal_weight(factor_names)

    def _min_variance(self, factor_values, factor_names) -> Dict:
        """最小方差 — 使用pyportfolioopt（规则0）

        注意：因子值不是价格，使用diff()而非pct_change()
        """
        try:
            from pypfopt import EfficientFrontier, risk_models
            # Align all series to common index before creating DataFrame
            aligned_values = {}
            common_index = None
            for name, series in factor_values.items():
                if isinstance(series, pd.Series):
                    if common_index is None:
                        common_index = series.index
                    else:
                        common_index = common_index.intersection(series.index)
            if common_index is not None and len(common_index) > 0:
                for name, series in factor_values.items():
                    if isinstance(series, pd.Series):
                        aligned_values[name] = series.reindex(common_index)
                    else:
                        aligned_values[name] = series
            else:
                aligned_values = factor_values
            factor_df = pd.DataFrame(aligned_values)
            factor_returns = factor_df[factor_names].diff().dropna()
            if len(factor_returns) < 20:
                return self._equal_weight(factor_names)

            S = risk_models.sample_cov(factor_returns)
            ef = EfficientFrontier(None, S)
            _raw_weights = ef.min_volatility()
            clean_weights = ef.clean_weights()

            weights = {k: v for k, v in clean_weights.items() if k in factor_names}
            return {"weights": weights, "method": "min_variance"}
        except Exception as e:
            logger.warning(f"PyPortfolioOpt min_variance失败: {e}，回退到等权重")
            return self._equal_weight(factor_names)

    def _risk_parity(self, factor_values, factor_names) -> Dict:
        """风险平价 — 使用pyportfolioopt HRPOpt（规则0）

        注意：因子值不是价格，使用diff()而非pct_change()
        """
        try:
            from pypfopt import HRPOpt
            factor_df = pd.DataFrame(factor_values)
            factor_returns = factor_df[factor_names].diff().dropna()
            if len(factor_returns) < 20:
                return self._equal_weight(factor_names)

            hrp = HRPOpt(factor_returns)
            _raw_weights = hrp.optimize()
            clean_weights = hrp.clean_weights()

            weights = {k: v for k, v in clean_weights.items() if k in factor_names}
            return {"weights": weights, "method": "risk_parity"}
        except Exception as e:
            logger.warning(f"PyPortfolioOpt risk_parity失败: {e}，回退到等权重")
            return self._equal_weight(factor_names)

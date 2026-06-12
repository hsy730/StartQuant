"""
权重优化服务 — 统一的投资组合权重计算逻辑

消除 portfolio.py 中 /optimize-weights 和 /compare-methods 的代码重复
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.stats import spearmanr

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
        factor_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict:
        """
        计算投资组合权重

        Args:
            factor_values: 因子名称到因子值序列的映射
            factor_names: 因子名称列表
            method: 权重计算方法
            returns: 收益率序列（IC/IR加权需要）
            factor_data_dict: alphalens格式的因子数据字典 {factor_name: factor_data}，
                传入时IC/IR加权委托alphalens计算横截面IC（推荐）

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
            return self._ic_weight(factor_values, factor_names, returns, factor_data_dict)
        elif method == "ir_weight":
            return self._ir_weight(factor_values, factor_names, returns, factor_data_dict)
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

    def _ic_weight(self, factor_values, factor_names, returns, factor_data_dict=None) -> Dict:
        """IC加权 — 优先使用alphalens横截面Spearman IC，回退到自实现（规则7.1/7.13）"""
        if returns is None or len(returns.dropna()) < 20:
            logger.debug("IC加权数据不足，回退到等权")
            return self._equal_weight(factor_names)

        ic_values = {}
        for factor_name in factor_names:
            values = factor_values.get(factor_name)
            if values is not None and len(values.dropna()) > 0:
                # 优先使用alphalens横截面IC（多股票场景）
                alphalens_ic = self._get_alphalens_ic(factor_name, factor_data_dict)
                if alphalens_ic is not None:
                    ic_values[factor_name] = abs(alphalens_ic)
                    continue

                # 回退：自实现IC计算
                aligned_data = pd.DataFrame({"factor": values, "returns": returns}).dropna()

                if len(aligned_data) > 10:
                    if isinstance(aligned_data.index, pd.MultiIndex):
                        daily_ics = []
                        for date, group in aligned_data.groupby(level=0):
                            if len(group) >= 5:
                                ic_val, _ = spearmanr(group["factor"], group["returns"])
                                if not np.isnan(ic_val):
                                    daily_ics.append(ic_val)
                        ic = float(np.mean(daily_ics)) if daily_ics else 0.0
                    else:
                        from backend.utils.ic_calculator import calculate_rolling_ic

                        rolling_ic = calculate_rolling_ic(
                            aligned_data["factor"], aligned_data["returns"], window=20, method="spearman"
                        )
                        ic = float(rolling_ic.dropna().mean()) if len(rolling_ic.dropna()) > 0 else 0.0

                    ic_values[factor_name] = abs(ic) if not np.isnan(ic) else 0.0
                else:
                    ic_values[factor_name] = 0.0
            else:
                ic_values[factor_name] = 0.0

        total_ic = sum(ic_values.values())
        if total_ic < 1e-10:
            return self._equal_weight(factor_names)

        weights = {k: safe_divide(v, total_ic, default=1.0 / len(factor_names)) for k, v in ic_values.items()}
        return {"weights": weights, "method": "ic_weight"}

    def _ir_weight(self, factor_values, factor_names, returns, factor_data_dict=None) -> Dict:
        """IR加权 — 优先使用alphalens横截面IC序列计算IR，回退到自实现"""
        from backend.utils.safe_math import safe_ir

        if returns is None or len(returns.dropna()) < 20:
            logger.debug("IR加权数据不足，回退到等权")
            return self._equal_weight(factor_names)

        ir_values = {}
        for factor_name in factor_names:
            values = factor_values.get(factor_name)
            if values is not None and len(values.dropna()) > 0:
                # 优先使用alphalens IC序列计算IR（多股票场景）
                alphalens_ir = self._get_alphalens_ir(factor_name, factor_data_dict)
                if alphalens_ir is not None:
                    ir_values[factor_name] = abs(alphalens_ir)
                    continue

                # 回退：自实现滚动Spearman IC → IR
                aligned_data = pd.DataFrame({"factor": values, "returns": returns}).dropna()

                if len(aligned_data) > 20:
                    from backend.utils.ic_calculator import calculate_rolling_ic

                    rolling_ic = calculate_rolling_ic(
                        aligned_data["factor"], aligned_data["returns"], window=20, method="spearman"
                    )
                    ic_mean = rolling_ic.mean()
                    ic_std = rolling_ic.std()
                    ir = safe_ir(float(ic_mean), float(ic_std), default=None)
                    if ir is not None:
                        ir_values[factor_name] = abs(ir)
                    elif abs(float(ic_mean)) > 1e-10:
                        # IR不可计算但IC_mean非零 → 因子极稳定，用IC绝对值作为权重代理
                        ir_values[factor_name] = abs(float(ic_mean)) * 10
                        logger.info(f"因子{factor_name} IR不可计算(IC_std≈0)，使用IC代理权重")
                    else:
                        ir_values[factor_name] = 0.0
                else:
                    ir_values[factor_name] = 0.0
            else:
                ir_values[factor_name] = 0.0

        total_ir = sum(ir_values.values())
        if total_ir < 1e-10:
            return self._equal_weight(factor_names)

        weights = {k: safe_divide(v, total_ir, default=1.0 / len(factor_names)) for k, v in ir_values.items()}
        return {"weights": weights, "method": "ir_weight"}

    @staticmethod
    def _get_alphalens_ic(factor_name: str, factor_data_dict: Optional[Dict] = None) -> Optional[float]:
        """从alphalens factor_data中获取横截面Spearman IC均值"""
        if factor_data_dict is None:
            return None
        factor_data = factor_data_dict.get(factor_name)
        if factor_data is None or not isinstance(factor_data, pd.DataFrame):
            return None
        if not isinstance(factor_data.index, pd.MultiIndex):
            return None
        num_assets = factor_data.index.get_level_values("asset").nunique()
        if num_assets < 2:
            return None
        try:
            import alphalens
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic_df = alphalens.performance.factor_information_coefficient(factor_data)
            ic_col = [c for c in ic_df.columns if "1" in str(c)]
            if ic_col:
                ic_series = ic_df[ic_col[0]].dropna()
            else:
                ic_series = ic_df.iloc[:, 0].dropna()
            if len(ic_series) > 0:
                return float(ic_series.mean())
        except Exception as e:
            logger.warning(f"alphalens IC计算失败(factor={factor_name}): {e}")
        return None

    @staticmethod
    def _get_alphalens_ir(factor_name: str, factor_data_dict: Optional[Dict] = None) -> Optional[float]:
        """从alphalens factor_data中获取横截面IC序列的IR"""
        if factor_data_dict is None:
            return None
        factor_data = factor_data_dict.get(factor_name)
        if factor_data is None or not isinstance(factor_data, pd.DataFrame):
            return None
        if not isinstance(factor_data.index, pd.MultiIndex):
            return None
        num_assets = factor_data.index.get_level_values("asset").nunique()
        if num_assets < 2:
            return None
        try:
            import alphalens
            import warnings
            from backend.utils.safe_math import safe_ir

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic_df = alphalens.performance.factor_information_coefficient(factor_data)
            ic_col = [c for c in ic_df.columns if "1" in str(c)]
            if ic_col:
                ic_series = ic_df[ic_col[0]].dropna()
            else:
                ic_series = ic_df.iloc[:, 0].dropna()
            if len(ic_series) >= 2:
                ic_mean = float(ic_series.mean())
                ic_std = float(ic_series.std())
                ir = safe_ir(ic_mean, ic_std, default=None)
                return ir
        except Exception as e:
            logger.warning(f"alphalens IR计算失败(factor={factor_name}): {e}")
        return None

    def _align_factor_indices(self, factor_values):
        """对齐所有因子Series的索引，避免不同起止日期导致NaN填充"""
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
        return aligned_values

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
            # 标准化后再diff，确保尺度不变性（避免大数值因子主导优化）
            aligned_values = self._align_factor_indices(factor_values)
            factor_df = pd.DataFrame(aligned_values)
            factor_standardized = factor_df[factor_names].apply(
                lambda x: (x - x.mean()) / x.std() if x.std() > 1e-10 else x - x.mean()
            )
            factor_returns = factor_standardized.diff().dropna()
            if len(factor_returns) < 20:
                return self._equal_weight(factor_names)

            mu = expected_returns.mean_historical_return(factor_returns, returns_data=True)
            S = risk_models.sample_cov(factor_returns, returns_data=True)
            ef = EfficientFrontier(mu, S)
            _raw_weights = ef.max_sharpe()  # noqa: F841
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

            # 标准化后再diff，确保尺度不变性（避免大数值因子主导优化）
            aligned_values = self._align_factor_indices(factor_values)
            factor_df = pd.DataFrame(aligned_values)
            factor_standardized = factor_df[factor_names].apply(
                lambda x: (x - x.mean()) / x.std() if x.std() > 1e-10 else x - x.mean()
            )
            factor_returns = factor_standardized.diff().dropna()
            if len(factor_returns) < 20:
                return self._equal_weight(factor_names)

            S = risk_models.sample_cov(factor_returns, returns_data=True)
            ef = EfficientFrontier(None, S)
            _raw_weights = ef.min_volatility()  # noqa: F841
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

            # 标准化后再diff，确保尺度不变性（避免大数值因子主导优化）
            aligned_values = self._align_factor_indices(factor_values)
            factor_df = pd.DataFrame(aligned_values)
            factor_standardized = factor_df[factor_names].apply(
                lambda x: (x - x.mean()) / x.std() if x.std() > 1e-10 else x - x.mean()
            )
            factor_returns = factor_standardized.diff().dropna()
            if len(factor_returns) < 20:
                return self._equal_weight(factor_names)

            hrp = HRPOpt(factor_returns)
            _raw_weights = hrp.optimize()  # noqa: F841
            clean_weights = hrp.clean_weights()

            weights = {k: v for k, v in clean_weights.items() if k in factor_names}
            return {"weights": weights, "method": "risk_parity"}
        except Exception as e:
            logger.warning(f"PyPortfolioOpt risk_parity失败: {e}，回退到等权重")
            return self._equal_weight(factor_names)

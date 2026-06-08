"""
因子收益分析服务 - 实现专业量化分析的核心功能

功能列表：
1. Quantile Returns（因子分组收益）- 按因子值分组计算各组收益
2. Cumulative Returns（累计收益曲线）- 分组累计收益走势
3. Long-Short Spread（多空利差）- 最高组与最低组的收益差
4. 因子自相关/换手率分析 - 完善版换手率计算

设计原则：
- 符合业界标准（JoinQuant/BigQuant/Alphalens）
- 使用pandas向量化操作，高性能
- 支持横截面和时间序列两种模式
- 提供详细的统计检验结果
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats as scipy_stats

from backend.utils.safe_math import safe_divide
from backend.services.risk_metrics import calculate_sharpe, calculate_risk_metrics

logger = logging.getLogger(__name__)


class QuantileMethod(str, Enum):
    """分位方法枚举"""
    EQUAL_WEIGHT = "equal_weight"      # 等权分组
    VALUE_WEIGHT = "value_weight"      # 值加权分组


@dataclass
class FactorReturnAnalysisConfig:
    """
    因子收益分析配置
    
    Attributes:
        n_quantiles: 分组数量（默认5组，符合业界标准）
        quantile_method: 分组方法
        forward_period: 前瞻收益期数（默认1期）
        weight_by_market_cap: 是否按市值加权
        min_samples_per_group: 每组最小样本量
        enable_bootstrap: 是否启用Bootstrap置信区间
        bootstrap_n: Bootstrap采样次数
    """
    n_quantiles: int = 5
    quantile_method: QuantileMethod = QuantileMethod.EQUAL_WEIGHT
    forward_period: int = 1
    weight_by_market_cap: bool = False
    min_samples_per_group: int = 5
    enable_bootstrap: bool = True
    bootstrap_n: int = 1000


class FactorReturnAnalysisService:
    """
    因子收益分析服务类
    
    提供专业的因子收益分析功能，包括：
    - 分组收益分析（Quantile Returns）
    - 累计收益曲线（Cumulative Returns）
    - 多空利差分析（Long-Short Spread）
    - 换手率分析（Turnover Analysis）
    
    所有方法均使用pandas向量化操作，确保高性能。
    """

    def __init__(self, config: Optional[FactorReturnAnalysisConfig] = None):
        """
        初始化服务
        
        Args:
            config: 分析配置，默认使用标准配置
        """
        self.config = config or FactorReturnAnalysisConfig()

    def calculate_quantile_returns(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        price_column: str = "close",
    ) -> Dict[str, Any]:
        """
        计算因子分组收益（Quantile Returns）
        
        这是因子分析中最核心的功能之一，用于验证因子的预测能力。
        将所有股票按因子值分成N组，计算每组的平均收益。
        如果因子有效，应该观察到单调性：高因子值组收益 > 低因子值组收益。
        
        Args:
            factor_data: {stock_code: DataFrame} 格式的因子数据
            factor_name: 因子名称
            price_column: 价格列名（用于计算收益）
            
        Returns:
            {
                "quantile_returns": [...],  # 各组平均收益
                "quantile_stats": [...],   # 各组统计信息
                "monotonicity_test": {...}, # 单调性检验
                "spread": {...},           # 多空利差
                "group_details": {...}     # 各组详细信息
            }
        """
        try:
            all_records = []

            for stock_code, df in factor_data.items():
                if factor_name not in df.columns or price_column not in df.columns:
                    continue

                df_copy = df.copy()

                if len(df_copy) < self.config.forward_period + 1:
                    continue

                df_copy["future_return"] = (
                    df_copy[price_column].shift(-self.config.forward_period) /
                    df_copy[price_column] - 1
                )

                df_copy["stock_code"] = stock_code
                # 保留日期索引信息，用于截面分位计算
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    df_copy["date"] = df_copy.index
                elif "date" in df_copy.columns:
                    df_copy["date"] = pd.to_datetime(df_copy["date"])
                else:
                    df_copy["date"] = pd.NaT

                valid_data = df_copy[
                    [factor_name, "future_return", "stock_code", "date"] +
                    (["market_cap"] if "market_cap" in df_copy.columns else [])
                ].dropna(subset=[factor_name, "future_return"])

                if len(valid_data) > 0:
                    all_records.append(valid_data)

            if not all_records:
                return {"error": "没有有效的数据用于计算分组收益"}

            merged_df = pd.concat(all_records, ignore_index=True)

            if len(merged_df) < self.config.n_quantiles * self.config.min_samples_per_group:
                return {
                    "error": f"数据不足，需要至少{self.config.n_quantiles * self.config.min_samples_per_group}个样本"
                }

            # 截面分位：按日期分组，每个截面内独立做qcut
            # 全局qcut无法回答"因子高的股票是否比因子低的收益更高"这个选股问题
            merged_df["quantile"] = np.nan
            if "date" in merged_df.columns and merged_df["date"].notna().any():
                for date, group in merged_df.groupby("date"):
                    if len(group) >= self.config.n_quantiles:
                        try:
                            merged_df.loc[group.index, "quantile"] = pd.qcut(
                                group[factor_name],
                                q=self.config.n_quantiles,
                                labels=False,
                                duplicates="drop"
                            )
                        except ValueError:
                            # 分位数相同时无法切分，跳过该截面
                            pass
            else:
                # 无日期信息时退化为全局分位（单股票场景）
                merged_df["quantile"] = pd.qcut(
                    merged_df[factor_name],
                    q=self.config.n_quantiles,
                    labels=False,
                    duplicates="drop"
                )

            # 排除无法分位的行
            merged_df = merged_df.dropna(subset=["quantile"])
            
            quantile_returns = []
            quantile_stats = []
            
            for q in range(self.config.n_quantiles):
                group_data = merged_df[merged_df["quantile"] == q]
                
                if self.config.weight_by_market_cap and "market_cap" in group_data.columns:
                    weights = group_data["market_cap"].fillna(group_data["market_cap"].median())
                    weight_sum = weights.sum()
                    if weight_sum == 0 or np.isnan(weight_sum):
                        avg_return = group_data["future_return"].mean()
                    else:
                        weights = safe_divide(weights, weight_sum, default=0.0)
                        avg_return = (group_data["future_return"] * weights).sum()
                else:
                    avg_return = group_data["future_return"].mean()
                
                std_return = group_data["future_return"].std()
                n_obs = len(group_data)
                
                t_stat, p_value = scipy_stats.ttest_1samp(
                    group_data["future_return"].dropna().values, 0
                )
                
                quantile_returns.append({
                    "group": f"Q{q+1}",
                    "avg_return": float(avg_return),
                    "std_return": float(std_return) if not np.isnan(std_return) else 0.0,
                    "n_observations": n_obs,
                    "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
                    "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
                    "is_significant": p_value < 0.05 if not np.isnan(p_value) else False,
                })
                
                quantile_stats.append({
                    "group": f"Q{q+1}",
                    "mean_factor": float(group_data[factor_name].mean()),
                    "min_factor": float(group_data[factor_name].min()),
                    "max_factor": float(group_data[factor_name].max()),
                    "n_stocks": group_data["stock_code"].nunique(),
                })
            
            spread_result = self._calculate_spread(quantile_returns)
            
            monotonicity_result = self._test_monotonicity(quantile_returns)
            
            if self.config.enable_bootstrap:
                bootstrap_result = self._bootstrap_quantile_returns(
                    merged_df, factor_name, "future_return"
                )
            else:
                bootstrap_result = None
            
            return {
                "success": True,
                "factor_name": factor_name,
                "n_quantiles": self.config.n_quantiles,
                "total_observations": len(merged_df),
                "n_stocks": merged_df["stock_code"].nunique(),
                "quantile_returns": quantile_returns,
                "quantile_stats": quantile_stats,
                "spread": spread_result,
                "monotonicity_test": monotonicity_result,
                "bootstrap_ci": bootstrap_result,
            }
            
        except Exception as e:
            logger.error(f"计算因子分组收益失败: {e}", exc_info=True)
            return {"error": str(e)}

    def calculate_cumulative_returns(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        price_column: str = "close",
        long_short: bool = True,
    ) -> Dict[str, Any]:
        """
        计算累计收益曲线（Cumulative Returns）
        
        基于因子分组构建多空组合，计算其累计收益曲线。
        这是最直观展示因子有效性的方式。
        
        Args:
            factor_data: {stock_code: DataFrame} 格式的因子数据
            factor_name: 因子名称
            price_column: 价格列名
            long_short: 是否计算多空组合（True=只显示多空，False=显示所有组）
            
        Returns:
            {
                "dates": [...],
                "cumulative_returns": {...},
                "summary_statistics": {...}
            }
        """
        try:
            date_returns = {}
            
            for stock_code, df in factor_data.items():
                if factor_name not in df.columns or price_column not in df.columns:
                    continue
                
                df_copy = df.copy()
                
                if len(df_copy) < self.config.forward_period + 1:
                    continue
                
                df_copy["future_return"] = (
                    df_copy[price_column].shift(-self.config.forward_period) / 
                    df_copy[price_column] - 1
                )
                
                valid_rows = df_copy[[factor_name, "future_return"]].dropna()
                if len(valid_rows) == 0:
                    continue

                for idx in valid_rows.index:
                    date_key = str(idx) if not isinstance(idx, pd.Timestamp) else idx.strftime("%Y-%m-%d")

                    if date_key not in date_returns:
                        date_returns[date_key] = []

                    date_returns[date_key].append({
                        "factor_value": valid_rows.loc[idx, factor_name],
                        "return": valid_rows.loc[idx, "future_return"],
                        "stock_code": stock_code,
                    })
            
            if not date_returns:
                return {"error": "没有有效的数据"}
            
            sorted_dates = sorted(date_returns.keys())
            
            group_cumreturns = {f"Q{i+1}": [] for i in range(self.config.n_quantiles)}
            long_short_returns = []
            all_dates = []
            
            cumulative_ls = 1.0
            
            for date_str in sorted_dates:
                observations = date_returns[date_str]
                
                if len(observations) < self.config.n_quantiles * self.config.min_samples_per_group:
                    if long_short:
                        long_short_returns.append(cumulative_ls - 1)
                    for i in range(self.config.n_quantiles):
                        group_cumreturns[f"Q{i+1}"].append(None)
                    all_dates.append(date_str)
                    continue
                
                obs_df = pd.DataFrame(observations)
                
                try:
                    obs_df["quantile"] = pd.qcut(
                        obs_df["factor_value"],
                        q=self.config.n_quantiles,
                        labels=False,
                        duplicates="drop",
                    )
                except ValueError:
                    if long_short:
                        long_short_returns.append(cumulative_ls - 1)
                    for i in range(self.config.n_quantiles):
                        group_cumreturns[f"Q{i+1}"].append(None)
                    all_dates.append(date_str)
                    continue
                
                group_returns = {}
                for q in range(self.config.n_quantiles):
                    group_data = obs_df[obs_df["quantile"] == q]
                    if len(group_data) > 0:
                        group_returns[q] = group_data["return"].mean()
                    else:
                        group_returns[q] = 0.0
                
                for q in range(self.config.n_quantiles):
                    group_cumreturns[f"Q{q+1}"].append(float(group_returns.get(q, 0.0)))
                
                if long_short:
                    long_return = group_returns.get(self.config.n_quantiles - 1, 0.0)
                    short_return = group_returns.get(0, 0.0)
                    ls_return = long_return - short_return
                    cumulative_ls *= (1 + ls_return)
                    long_short_returns.append(cumulative_ls - 1)
                
                all_dates.append(date_str)
            
            result = {
                "dates": all_dates,
                "n_periods": len(all_dates),
            }
            
            if long_short:
                result["long_short_cumulative"] = [float(r) if r is not None else None for r in long_short_returns]
                
                valid_returns = [r for r in long_short_returns if r is not None]
                if valid_returns:
                    final_return = valid_returns[-1]
                    max_drawdown = self._calculate_max_drawdown(valid_returns)
                    sharpe_ratio = self._calculate_sharpe_ratio(
                        pd.Series([(long_short_returns[i+1] + 1) / (long_short_returns[i] + 1) - 1
                                  for i in range(len(long_short_returns)-1) 
                                  if long_short_returns[i] is not None and long_short_returns[i+1] is not None])
                    )
                    
                    result["summary_statistics"] = {
                        "final_cumulative_return": float(final_return),
                        "max_drawdown": float(max_drawdown),
                        "sharpe_ratio": float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
                        "total_periods": len(valid_returns),
                    }
            
            if not long_short:
                result["group_cumulative"] = {}
                for q in range(self.config.n_quantiles):
                    result["group_cumulative"][f"Q{q+1}"] = [
                        float(r) if r is not None else None 
                        for r in group_cumreturns[f"Q{q+1}"]
                    ]
            
            return result
            
        except Exception as e:
            logger.error(f"计算累计收益失败: {e}", exc_info=True)
            return {"error": str(e)}

    def calculate_turnover_analysis(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        window: int = 20,
    ) -> Dict[str, Any]:
        """
        计算因子换手率分析（完善版）
        
        换手率衡量因子值的稳定性，低换手率意味着因子信号稳定，
        可以降低交易成本。这是专业因子分析的重要组成部分。
        
        Args:
            factor_data: {stock_code: DataFrame} 格式的因子数据
            factor_name: 因子名称
            window: 滚动窗口期
            
        Returns:
            {
                "turnover_stats": {...},
                "autocorrelation": {...},
                "stability_analysis": {...}
            }
        """
        try:
            all_factor_series = []
            
            for stock_code, df in factor_data.items():
                if factor_name in df.columns:
                    series = df[factor_name].dropna()
                    if len(series) > window:
                        all_factor_series.append(series)
            
            if not all_factor_series:
                return {"error": "没有有效的因子数据"}
            
            turnover_rates = []
            autocorrelations = []
            
            for series in all_factor_series:
                if len(series) < window + 1:
                    continue
                
                ranks = series.rolling(window=window, min_periods=1).rank(pct=True)
                rank_changes = ranks.diff().abs()
                turnover = rank_changes.mean()
                turnover_rates.append(turnover)
                
                auto_corr = series.autocorr(lag=1)
                if not np.isnan(auto_corr):
                    autocorrelations.append(auto_corr)
            
            if not turnover_rates:
                return {"error": "无法计算换手率"}
            
            avg_turnover = np.mean(turnover_rates)
            median_turnover = np.median(turnover_rates)
            std_turnover = np.std(turnover_rates)
            
            avg_autocorr = np.mean(autocorrelations) if autocorrelations else 0.0
            
            half_life = None
            if avg_autocorr > 0 and avg_autocorr < 1:
                half_life = -np.log(2) / np.log(avg_autocorr)
            
            stability_score = 1.0 - min(avg_turnover, 1.0)
            
            return {
                "success": True,
                "factor_name": factor_name,
                "n_stocks_analyzed": len(all_factor_series),
                "window": window,
                "turnover_stats": {
                    "mean_turnover": float(avg_turnover),
                    "median_turnover": float(median_turnover),
                    "std_turnover": float(std_turnover),
                    "min_turnover": float(np.min(turnover_rates)),
                    "max_turnover": float(np.max(turnover_rates)),
                    "interpretation": self._interpret_turnover(avg_turnover),
                },
                "autocorrelation": {
                    "mean_autocorrelation": float(avg_autocorr),
                    "n_stocks_with_valid_autocorr": len(autocorrelations),
                    "half_life": float(half_life) if half_life else None,
                    "interpretation": self._interpret_autocorrelation(avg_autocorr),
                },
                "stability_analysis": {
                    "stability_score": float(stability_score),
                    "is_stable": stability_score >= 0.6,
                    "recommendation": self._generate_stability_recommendation(stability_score, avg_turnover),
                },
            }
            
        except Exception as e:
            logger.error(f"换手率分析失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _calculate_spread(
        self,
        quantile_returns: List[Dict],
    ) -> Dict[str, Any]:
        """
        计算多空利差（Long-Short Spread）
        
        最高组收益减去最低组收益，这是衡量因子预测能力的核心指标。
        """
        if len(quantile_returns) < 2:
            return {"error": "分组不足"}
        
        top_group = quantile_returns[-1]
        bottom_group = quantile_returns[0]
        
        spread = top_group["avg_return"] - bottom_group["avg_return"]
        spread_std = np.sqrt(
            top_group["std_return"]**2 + bottom_group["std_return"]**2
        )
        
        n_top = top_group["n_observations"]
        n_bottom = bottom_group["n_observations"]
        
        t_stat = safe_divide(spread, spread_std * np.sqrt(safe_divide(1.0, n_top, default=0.0) + safe_divide(1.0, n_bottom, default=0.0)), default=0.0) if spread_std > 0 else 0
        
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n_top + n_bottom - 2))
        
        return {
            "long_short_spread": float(spread),
            "spread_std": float(spread_std),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": p_value < 0.05,
            "top_group_return": top_group["avg_return"],
            "bottom_group_return": bottom_group["avg_return"],
            "interpretation": self._interpret_spread(spread, p_value),
        }

    def _test_monotonicity(
        self,
        quantile_returns: List[Dict],
    ) -> Dict[str, Any]:
        """
        检验分组收益的单调性
        
        有效因子应该呈现单调性：Q1 < Q2 < Q3 < Q4 < Q5（或反向）
        """
        returns = [q["avg_return"] for q in quantile_returns]
        n = len(returns)
        
        increasing_count = sum(1 for i in range(n-1) if returns[i+1] > returns[i])
        decreasing_count = sum(1 for i in range(n-1) if returns[i+1] < returns[i])
        
        monotonicity_ratio = max(increasing_count, decreasing_count) / (n - 1) if n > 1 else 0
        
        spearman_corr, spearman_p = scipy_stats.spearmanr(range(n), returns)
        
        is_monotonic = monotonicity_ratio >= 0.8
        direction = "increasing" if increasing_count >= decreasing_count else "decreasing"
        
        return {
            "is_monotonic": is_monotonic,
            "direction": direction,
            "monotonicity_ratio": float(monotonicity_ratio),
            "spearman_correlation": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
            "spearman_p_value": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
            "n_increasing_pairs": increasing_count,
            "n_decreasing_pairs": decreasing_count,
            "interpretation": self._interpret_monotonicity(is_monotonic, direction, monotonicity_ratio),
        }

    def _bootstrap_quantile_returns(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
    ) -> Dict[str, Any]:
        """
        Bootstrap置信区间估计
        
        通过重抽样评估分组收益的稳健性
        """
        np.random.seed(42)
        
        bootstrapped_spreads = []
        bootstrapped_returns = {f"Q{i+1}": [] for i in range(self.config.n_quantiles)}
        
        for _ in range(self.config.bootstrap_n):
            sample_df = df.sample(n=len(df), replace=True)
            
            try:
                sample_df["quantile"] = pd.qcut(
                    sample_df[factor_col],
                    q=self.config.n_quantiles,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                continue
            
            for q in range(self.config.n_quantiles):
                group = sample_df[sample_df["quantile"] == q]
                if len(group) > 0:
                    bootstrapped_returns[f"Q{q+1}"].append(group[return_col].mean())
            
            if f"Q{self.config.n_quantiles}" in bootstrapped_returns and \
               "Q1" in bootstrapped_returns and \
               len(bootstrapped_returns[f"Q{self.config.n_quantiles}"]) > 0 and \
               len(bootstrapped_returns["Q1"]) > 0:
                spread = (bootstrapped_returns[f"Q{self.config.n_quantiles}"][-1] - 
                         bootstrapped_returns["Q1"][-1])
                bootstrapped_spreads.append(spread)
        
        result = {}
        
        for q in range(self.config.n_quantiles):
            key = f"Q{q+1}"
            if key in bootstrapped_returns and len(bootstrapped_returns[key]) > 0:
                values = sorted(bootstrapped_returns[key])
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                result[key] = {
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        
        if bootstrapped_spreads:
            spreads_sorted = sorted(bootstrapped_spreads)
            result["spread_ci"] = {
                "ci_lower": float(np.percentile(spreads_sorted, 2.5)),
                "ci_upper": float(np.percentile(spreads_sorted, 97.5)),
                "mean": float(np.mean(bootstrapped_spreads)),
                "std": float(np.std(bootstrapped_spreads)),
            }
        
        return result

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤（通过risk_metrics统一入口）"""
        if not returns:
            return 0.0

        # 将累计收益转换为日收益率
        daily_returns = pd.Series([(returns[i+1] + 1) / (returns[i] + 1) - 1
                                   for i in range(len(returns) - 1)])

        if len(daily_returns) >= 2:
            metrics = calculate_risk_metrics(daily_returns)
            dd = metrics.get("max_drawdown")
            return abs(float(dd)) if dd is not None else 0.0

        # 数据不足，手动计算
        wealth_index = pd.Series(returns).add(1).cumprod()
        peak = wealth_index.expanding().max()
        drawdown = safe_divide(wealth_index - peak, peak, default=0.0)
        return abs(float(drawdown.min())) if len(drawdown) > 0 else 0.0

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """计算夏普比率（年化，扣除无风险利率），委托risk_metrics统一入口"""
        sharpe = calculate_sharpe(returns, risk_free_rate=risk_free_rate)
        return sharpe if sharpe is not None else 0.0

    def _interpret_turnover(self, turnover: float) -> str:
        """解读换手率"""
        if turnover < 0.1:
            return "极低换手率，因子非常稳定，交易成本低"
        elif turnover < 0.2:
            return "较低换手率，因子较稳定"
        elif turnover < 0.4:
            return "中等换手率，需关注交易成本"
        elif turnover < 0.6:
            return "较高换手率，因子变动频繁"
        else:
            return "极高换手率，因子不稳定，交易成本高"

    def _interpret_autocorrelation(self, autocorr: float) -> str:
        """解读自相关系数"""
        if autocorr > 0.9:
            return "高度自相关，因子信号持续性很强"
        elif autocorr > 0.7:
            return "较强自相关，因子信号具有较好的持续性"
        elif autocorr > 0.5:
            return "中等自相关，因子信号有一定持续性"
        elif autocorr > 0.3:
            return "较弱自相关，因子信号持续性一般"
        else:
            return "低自相关，因子信号变化快，需要频繁调仓"

    def _interpret_spread(self, spread: float, p_value: float) -> str:
        """解读多空利差"""
        if p_value >= 0.05:
            return "多空利差不显著，因子预测能力较弱"
        elif spread > 0.02:
            return f"多空利差显著为正（{spread:.2%}），因子具有很强的预测能力"
        elif spread > 0.01:
            return f"多空利差显著为正（{spread:.2%}），因子具有一定的预测能力"
        elif spread > 0:
            return f"多空利差为正但较小（{spread:.2%}），考虑扣除交易成本后可能不经济"
        else:
            return f"多空利差为负（{spread:.2%}），因子方向可能错误或需要反转"

    def _interpret_monotonicity(
        self,
        is_monotonic: bool,
        direction: str,
        ratio: float
    ) -> str:
        """解读单调性"""
        if not is_monotonic:
            return f"分组收益单调性不明显（{ratio:.0%}），因子预测能力不稳定"
        
        direction_text = "递增" if direction == "increasing" else "递减"
        return f"分组收益呈{direction_text}趋势（{ratio:.0%}），因子具有良好的单调性"

    def _generate_stability_recommendation(
        self,
        stability_score: float,
        turnover: float
    ) -> str:
        """生成稳定性建议"""
        if stability_score >= 0.8:
            return "因子非常稳定，适合长期持有策略"
        elif stability_score >= 0.6:
            return "因子较为稳定，适合中等频率调仓"
        elif stability_score >= 0.4:
            return "因子稳定性一般，建议关注交易成本影响"
        else:
            return "因子稳定性较差，建议缩短调仓周期或优化因子构造"


# 全局默认实例
factor_return_analysis_service = FactorReturnAnalysisService()
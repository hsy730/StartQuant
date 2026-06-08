"""
因子贡献度分解服务
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from scipy.stats import pearsonr, ttest_1samp
import akshare as ak

from backend.services.risk_metrics import calculate_sharpe, calculate_volatility
from backend.utils.safe_math import safe_divide

logger = logging.getLogger(__name__)


class FactorAttributionService:
    """因子贡献度分解服务类"""

    def __init__(self):
        self._benchmark_cache = None

    def _get_benchmark_data(self, start_date=None, end_date=None) -> Optional[pd.DataFrame]:
        """
        获取上证指数作为基准数据

        Args:
            start_date: 开始日期 (datetime or str)
            end_date: 结束日期 (datetime or str)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            Index is set to date column
        """
        try:
            # 获取上证指数数据 (sh000001 是上证指数的代码)
            benchmark_df = ak.stock_zh_index_daily(symbol="sh000001")

            # 转换日期格式
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            benchmark_df.set_index('date', inplace=True)

            # 过滤日期范围
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                benchmark_df = benchmark_df[benchmark_df.index >= start_date]
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                benchmark_df = benchmark_df[benchmark_df.index <= end_date]

            return benchmark_df
        except Exception as e:
            logger.warning(f"获取基准指数数据失败: {str(e)}")
            return None

    def analyze_attribution(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        因子贡献度分解

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_name: 因子名称
            benchmark_data: 基准数据（DataFrame with 'close' column）

        Returns:
            {
                "factor_contribution": {...},  # 因子收益贡献
                "alpha_beta": {...},           # Alpha-Beta分解
                "return_decomposition": {...}  # 收益分解
            }
        """
        results = {}

        # 1. 因子收益贡献（基于因子暴露度与收益的关系）
        results["factor_contribution"] = self._calculate_contribution(
            factor_data, factor_name
        )

        # 2. Alpha-Beta分解（相对于基准的超额收益）
        # 如果没有提供基准数据，自动获取上证指数
        if benchmark_data is None:
            # 获取数据范围（从第一只股票的开始日期到最后日期）
            all_dates = []
            for df in factor_data.values():
                if len(df) > 0:
                    all_dates.append(df.index.min())
                    all_dates.append(df.index.max())
            if all_dates:
                benchmark_data = self._get_benchmark_data(
                    start_date=min(all_dates),
                    end_date=max(all_dates)
                )

        results["alpha_beta"] = self._decompose_alpha_beta(
            factor_data, factor_name, benchmark_data
        )

        # 3. 收益分解（按时间段和股票分解）
        results["return_decomposition"] = self._decompose_return(
            factor_data, factor_name
        )

        return results

    def _calculate_contribution(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> Dict[str, Any]:
        """
        计算因子收益贡献

        方法：因子暴露度与未来收益的相关性分析

        Returns:
            {
                "ic": float,                    # Information Coefficient
                "ic_pvalue": float,
                "high_exposure_return": float,  # 高暴露组收益
                "low_exposure_return": float,   # 低暴露组收益
                "long_short_return": float,     # 多空收益
                "contribution_ratio": float     # 因子贡献比例
            }
        """
        # 按日期计算横截面IC（业界标准方法）
        from scipy.stats import spearmanr

        # 构建日期-股票的因子值和未来收益面板
        factor_panel = {}
        return_panel = {}
        for stock_code, df in factor_data.items():
            if factor_name in df.columns and "close" in df.columns:
                df_copy = df.copy()
                df_copy["future_return"] = df_copy["close"].shift(-1) / df_copy["close"] - 1
                valid = df_copy[[factor_name, "future_return"]].dropna()
                valid = valid[~np.isinf(valid["future_return"])]
                for date, row in valid.iterrows():
                    if date not in factor_panel:
                        factor_panel[date] = []
                        return_panel[date] = []
                    factor_panel[date].append(row[factor_name])
                    return_panel[date].append(row["future_return"])

        # 每日横截面Rank IC
        daily_ics = []
        for date in sorted(factor_panel.keys()):
            fv = factor_panel[date]
            rv = return_panel[date]
            if len(fv) >= 5 and np.std(fv) > 1e-12:
                ic, _ = spearmanr(fv, rv)
                if not np.isnan(ic):
                    daily_ics.append(ic)

        if len(daily_ics) < 3:
            return {"error": "数据不足以计算贡献度"}

        ic = np.mean(daily_ics)
        _, ic_pvalue = ttest_1samp(daily_ics, 0) if len(daily_ics) > 1 else (0, 1.0)
        ic_pvalue = float(ic_pvalue)

        # 收集所有数据用于分组收益计算
        all_factor_values = []
        all_returns = []
        for stock_code, df in factor_data.items():
            if factor_name in df.columns and "close" in df.columns:
                df_copy = df.copy()
                df_copy["future_return"] = df_copy["close"].shift(-1) / df_copy["close"] - 1
                valid_data = df_copy[[factor_name, "future_return"]].dropna()
                valid_data = valid_data[~np.isinf(valid_data["future_return"])]
                all_factor_values.extend(valid_data[factor_name].tolist())
                all_returns.extend(valid_data["future_return"].tolist())

        # 计算因子暴露度分组收益
        factor_series = pd.Series(all_factor_values)
        return_series = pd.Series(all_returns)

        # 高暴露组（前30%）
        high_threshold = factor_series.quantile(0.7)
        high_mask = factor_series >= high_threshold
        high_return = float(return_series[high_mask].mean()) if high_mask.sum() > 0 else 0.0

        # 低暴露组（后30%）
        low_threshold = factor_series.quantile(0.3)
        low_mask = factor_series <= low_threshold
        low_return = float(return_series[low_mask].mean()) if low_mask.sum() > 0 else 0.0

        # 多空收益
        long_short_return = high_return - low_return

        # 因子贡献比例（IC解释的方差比例）
        contribution_ratio = ic ** 2 if not np.isnan(ic) else 0.0

        return {
            "ic": float(ic) if not np.isnan(ic) else 0.0,
            "ic_pvalue": float(ic_pvalue),
            "high_exposure_return": high_return,
            "low_exposure_return": low_return,
            "long_short_return": long_short_return,
            "contribution_ratio": float(contribution_ratio),
            "sample_size": len(all_factor_values)
        }

    def _decompose_alpha_beta(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Alpha-Beta分解

        方法：计算高因子暴露组合相对于基准的Alpha和Beta

        Returns:
            {
                "has_benchmark": bool,
                "alpha": float,          # 年化Alpha
                "beta": float,           # Beta
                "r_squared": float,      # 拟合度
                "interpretation": str
            }
        """
        # 基于因子暴露度构建高暴露组合收益（而非等权组合）
        # 对每个日期，根据因子值排序，取高暴露组（前30%）等权收益
        date_portfolio_returns = {}
        for stock_code, df in factor_data.items():
            if factor_name not in df.columns or "close" not in df.columns:
                continue
            df_copy = df.copy()
            df_copy["return"] = df_copy["close"].pct_change(1)
            valid = df_copy[[factor_name, "return"]].dropna()
            for date, row in valid.iterrows():
                if date not in date_portfolio_returns:
                    date_portfolio_returns[date] = {"factor": [], "return": []}
                date_portfolio_returns[date]["factor"].append(row[factor_name])
                date_portfolio_returns[date]["return"].append(row["return"])

        portfolio_daily_returns = {}
        for date in sorted(date_portfolio_returns.keys()):
            fv = pd.Series(date_portfolio_returns[date]["factor"])
            rv = pd.Series(date_portfolio_returns[date]["return"])
            if len(fv) < 5:
                continue
            top_threshold = fv.quantile(0.7)
            high_mask = fv >= top_threshold
            if high_mask.sum() > 0:
                portfolio_daily_returns[date] = rv[high_mask].mean()

        if len(portfolio_daily_returns) < 10:
            return {"error": "有效交易日不足，无法构建因子组合"}

        portfolio_returns = pd.Series(portfolio_daily_returns)
        common_index = portfolio_returns.index

        # 如果没有提供基准数据
        if benchmark_data is None:
            return {
                "has_benchmark": False,
                "message": "未提供基准数据（如市场指数），无法计算Alpha-Beta",
                "portfolio_return": {
                    "daily_mean": float(portfolio_returns.mean()),
                    "annual_return": float((1 + portfolio_returns.mean()) ** 252 - 1),
                    "volatility": calculate_volatility(portfolio_returns) or 0.0,
                    "sharpe": calculate_sharpe(portfolio_returns, risk_free_rate=0.03) or 0.0
                }
            }

        # 有基准数据的情况
        if "close" not in benchmark_data.columns:
            return {"error": "基准数据缺少close列"}

        # 对齐基准日期
        benchmark_aligned = benchmark_data.reindex(common_index)
        benchmark_returns = benchmark_aligned["close"].pct_change(1).dropna()

        # 再次对齐组合和基准
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned_data) < 10:
            return {"error": "对齐后数据不足"}

        # 线性回归：Portfolio Return = Alpha + Beta * Benchmark Return
        X = aligned_data[['benchmark']].values
        y = aligned_data['portfolio'].values

        # 计算Beta (协方差 / 基准方差)
        cov_matrix = np.cov(y, X.flatten())
        beta = safe_divide(float(cov_matrix[0, 1]), float(cov_matrix[1, 1]), default=1.0)

        # 计算Alpha (组合收益 - Beta * 基准收益)
        alpha = y.mean() - beta * X.mean()

        # 年化
        alpha_annual = alpha * 252

        # 计算R²
        y_pred = alpha + beta * X.flatten()
        ss_tot = np.sum((y - y.mean()) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1.0 - safe_divide(float(ss_res), float(ss_tot), default=0.0)

        interpretation = (
            f"相对于基准的年化Alpha: {alpha_annual:.4f}, "
            f"Beta: {beta:.4f}, "
            f"拟合度(R²): {r_squared:.4f}"
        )

        return {
            "has_benchmark": True,
            "alpha": float(alpha_annual),
            "beta": float(beta),
            "r_squared": float(r_squared),
            "daily_alpha": float(alpha),
            "interpretation": interpretation
        }

    def _decompose_return(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> Dict[str, Any]:
        """
        收益分解

        按股票和时间段分解收益

        Returns:
            {
                "overall_stats": {...},
                "return_by_stock": {...},
                "monthly_returns": [...]
            }
        """
        returns_by_stock = {}
        all_returns = []

        for stock_code, df in factor_data.items():
            if "close" in df.columns and len(df) > 1:
                returns = df["close"].pct_change(1).dropna()

                if len(returns) > 0:
                    avg_return = float(returns.mean())
                    cum_return = float((1 + returns).prod() - 1)
                    vol_annual = calculate_volatility(returns) or 0.0

                    returns_by_stock[stock_code] = {
                        "avg_daily_return": avg_return,
                        "annual_return": float((1 + avg_return) ** 252 - 1),
                        "cumulative_return": cum_return,
                        "volatility": vol_annual,
                        "daily_volatility": float(returns.std()),
                        "sharpe": calculate_sharpe(returns, risk_free_rate=0.03) or 0.0,
                        "win_rate": float((returns > 0).mean()),
                        "count": len(returns)
                    }

                    all_returns.extend(returns.tolist())

        if not all_returns:
            return {"error": "没有可用的收益数据"}

        all_returns_series = pd.Series(all_returns)
        overall_avg = float(all_returns_series.mean())
        overall_vol_annual = calculate_volatility(all_returns_series) or 0.0
        # 先计算每只股票的累计收益再取均值，而非跨股票连乘
        stock_cum_returns = [v["cumulative_return"] for v in returns_by_stock.values()]
        overall_cum = float(np.mean(stock_cum_returns)) if stock_cum_returns else 0.0

        return {
            "overall_stats": {
                "avg_daily_return": overall_avg,
                "annual_return": float((1 + overall_avg) ** 252 - 1),
                "cumulative_return": overall_cum,
                "volatility_annual": overall_vol_annual,
                "daily_volatility": float(all_returns_series.std()),
                "sharpe_ratio": calculate_sharpe(all_returns_series, risk_free_rate=0.03) or 0.0,
                "win_rate": float((all_returns_series > 0).mean())
            },
            "return_by_stock": returns_by_stock,
            "stock_count": len(returns_by_stock),
            "total_observations": len(all_returns)
        }


# 全局服务实例
factor_attribution_service = FactorAttributionService()

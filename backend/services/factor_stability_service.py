"""
因子稳定性分析服务
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from backend.utils.returns import calculate_future_returns
from backend.utils.safe_math import safe_divide, safe_ir
from backend.services.analysis_service import AnalysisService
from backend.services.data_service import data_service
from backend.services.factor_service import factor_service
from backend.repositories.factor_repository import FactorRepository
from backend.core.database import get_db_session

logger = logging.getLogger(__name__)


class FactorStabilityService:
    """因子稳定性分析服务类"""

    def __init__(self):
        pass

    def calculate_distribution_stability(
        self,
        factor_series: pd.Series,
        window: int = 252,
        method: str = "ks"
    ) -> Dict:
        """
        分布稳定性分析 - 使用KS检验比较不同窗口期的分布

        Args:
            factor_series: 因子值序列
            window: 窗口大小（交易日）
            method: 检验方法，"ks"（Kolmogorov-Smirnov）或 "ttest"

        Returns:
            稳定性分析结果
        """
        if len(factor_series) < window * 2:
            raise ValueError(f"数据长度不足，至少需要 {window * 2} 个数据点")

        results = {}
        p_values = []
        test_statistics = []

        # 分段数据
        n_windows = len(factor_series) // window
        segments = []
        for i in range(n_windows):
            start_idx = i * window
            end_idx = start_idx + window
            segments.append(factor_series.iloc[start_idx:end_idx].dropna())

        # 过滤掉数据量不足的段（至少需要3个数据点才能进行统计检验）
        segments = [s for s in segments if len(s) >= 3]

        # 两两比较
        comparisons = []
        for i in range(len(segments) - 1):
            for j in range(i + 1, len(segments)):
                segment1 = segments[i]
                segment2 = segments[j]

                try:
                    if method == "ks":
                        # Kolmogorov-Smirnov检验
                        statistic, p_value = stats.ks_2samp(segment1, segment2)
                    elif method == "ttest":
                        # t检验
                        statistic, p_value = stats.ttest_ind(segment1, segment2)
                    else:
                        raise ValueError(f"不支持的检验方法: {method}")

                    comparisons.append({
                        "segment1": i,
                        "segment2": j,
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                    })
                    p_values.append(p_value)
                    test_statistics.append(statistic)
                except Exception as e:
                    logger.debug(f"统计检验失败: {e}")
                    continue

        # 汇总统计
        if len(p_values) == 0:
            return {
                "method": method,
                "window": window,
                "n_comparisons": 0,
                "avg_p_value": None,
                "stable_ratio": None,
                "stability_score": 0.0,
                "comparisons": [],
            }
        avg_p_value = np.mean(p_values)
        stable_ratio = sum(1 for p in p_values if p > 0.05) / len(p_values)

        results = {
            "method": method,
            "window": window,
            "n_comparisons": len(comparisons),
            "avg_p_value": float(avg_p_value),
            "stable_ratio": float(stable_ratio),
            "stability_score": float(stable_ratio),  # 稳定性得分
            "comparisons": comparisons[:10],  # 只返回前10个比较
        }

        return results

    def calculate_time_series_stability(
        self,
        ic_series: pd.Series,
        maxlag: int = 10
    ) -> Dict:
        """
        时间序列稳定性分析 - 使用ADF检验判断平稳性

        Args:
            ic_series: IC值序列
            maxlag: ADF检验的最大滞后阶数

        Returns:
            平稳性分析结果
        """
        # 移除缺失值
        ic_clean = ic_series.dropna()

        if len(ic_clean) < 20:
            raise ValueError("IC序列长度不足，至少需要20个数据点")

        # ADF检验
        try:
            result = adfuller(ic_clean, maxlag=maxlag)

            adf_statistic = float(result[0])
            p_value = float(result[1])
            used_lag = int(result[2])
            n_obs = int(result[3])
            critical_values = result[4]

            # 判断平稳性（p < 0.05）
            is_stationary = p_value < 0.05

            return {
                "is_stationary": is_stationary,
                "adf_statistic": adf_statistic,
                "p_value": p_value,
                "used_lag": used_lag,
                "n_obs": n_obs,
                "critical_values": {
                    "1%": float(critical_values['1%']),
                    "5%": float(critical_values['5%']),
                    "10%": float(critical_values['10%']),
                },
                "interpretation": (
                    "序列平稳，拒绝存在单位根的原假设" if is_stationary
                    else "序列不平稳，存在单位根"
                ),
            }

        except Exception as e:
            return {
                "error": str(e),
                "is_stationary": None,
            }

    def calculate_coefficient_of_variation(
        self,
        ic_series: pd.Series,
    ) -> Dict:
        """
        计算变异系数 - 衡量离散程度

        Args:
            ic_series: IC值序列

        Returns:
            变异系数统计
        """
        ic_clean = ic_series.dropna()

        if len(ic_clean) == 0:
            return {"error": "没有有效数据"}

        mean = ic_clean.mean()
        std = ic_clean.std()

        cv = safe_divide(float(std), float(mean), default=None)

        return {
            "mean": float(mean),
            "std": float(std),
            "cv": float(cv),
            "interpretation": (
                "变异程度较低" if cv < 0.5 else
                "变异程度中等" if cv < 1.0 else
                "变异程度较高"
            ),
        }

    def calculate_rolling_stability(
        self,
        factor_data: pd.DataFrame,
        factor_name: str,
        return_col: str = "future_return",
        windows: List[int] = [20, 60, 120, 252]
    ) -> Dict:
        """
        滚动窗口稳定性分析 - 在不同窗口下计算IC

        Args:
            factor_data: 包含因子和收益率的数据框
            factor_name: 因子列名
            return_col: 收益率列名
            windows: 窗口大小列表

        Returns:
            各窗口的稳定性统计
        """
        results = {}

        for window in windows:
            if len(factor_data) < window * 2:
                continue

            # 计算滚动IC（向量化操作）
            if factor_name in factor_data.columns and return_col in factor_data.columns:
                rolling_ic_series = factor_data[factor_name].rolling(
                    window=window, min_periods=window
                ).corr(factor_data[return_col]).dropna()
                rolling_ic = rolling_ic_series.tolist()
            else:
                rolling_ic = []

            if rolling_ic:
                ic_series = pd.Series(rolling_ic)
                results[f"window_{window}"] = {
                    "window": window,
                    "mean_ic": float(ic_series.mean()),
                    "std_ic": float(ic_series.std()),
                    "ir": safe_ir(float(ic_series.mean()), float(ic_series.std()), default=None),
                    "cv": safe_divide(float(ic_series.std()), float(ic_series.mean()), default=None),
                }

        return results

    def calculate_market_regime_performance(
        self,
        factor_data: pd.DataFrame,
        factor_name: str,
        return_col: str = "future_return",
        price_col: str = "close",
        bull_threshold: float = 0.05,
        bear_threshold: float = -0.05
    ) -> Dict:
        """
        不同市场环境下的表现分析

        Args:
            factor_data: 因子数据
            factor_name: 因子列名
            return_col: 收益率列名
            price_col: 价格列名（用于判断市场环境）
            bull_threshold: 牛市阈值
            bear_threshold: 熊市阈值

        Returns:
            各市场环境下的IC统计
        """
        if price_col not in factor_data.columns:
            raise ValueError(f"数据框中缺少价格列: {price_col}")

        # 计算市场累计收益率
        factor_data = factor_data.copy()
        factor_data['market_return'] = factor_data[price_col].pct_change()

        # 划分市场环境（向量化操作）
        lookback = 20
        rolling_return = factor_data['market_return'].rolling(window=lookback, min_periods=lookback).sum()

        regime_labels = pd.cut(
            rolling_return,
            bins=[-float('inf'), bear_threshold, bull_threshold, float('inf')],
            labels=['bear', 'flat', 'bull']
        )

        # 将连续相同regime合并为区间（向量化替代逐行循环）
        regime_series = pd.Series(regime_labels, index=rolling_return.index)
        # 去除NaN
        regime_series = regime_series.dropna()
        if len(regime_series) == 0:
            return {}
        # 找到regime变化的边界点
        regime_change = regime_series != regime_series.shift(1)
        group_ids = regime_change.cumsum()
        # 对每个连续区间计算IC
        regime_performance = {}
        for group_id, group_indices in regime_series.groupby(group_ids).groups.items():
            regime = regime_series.loc[group_indices[0]]
            start_idx = factor_data.index.get_loc(group_indices[0])
            end_idx = factor_data.index.get_loc(group_indices[-1]) + 1

            regime_data = factor_data.iloc[start_idx:end_idx]

            if factor_name in regime_data.columns and return_col in regime_data.columns:
                ic = regime_data[factor_name].corr(regime_data[return_col])

                if regime not in regime_performance:
                    regime_performance[regime] = []

                if not np.isnan(ic):
                    regime_performance[regime].append(ic)

        # 汇总
        results = {}
        for regime, ics in regime_performance.items():
            results[regime] = {
                "mean_ic": float(np.mean(ics)),
                "std_ic": float(np.std(ics)),
                "n_periods": len(ics),
            }

        return results


    def comprehensive_stability_test(
        self,
        factor_name: str,
        stock_codes: List[str],
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        综合稳定性检验 - 整合多个维度评估因子稳定性

        Args:
            factor_name: 因子名称
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            综合稳定性分析报告，包含：
            - distribution_stability: 分布稳定性（KS检验）
            - time_series_stationarity: 时间序列平稳性（ADF检验）
            - coefficient_of_variation: 变异系数
            - rolling_window_analysis: 滚动窗口稳定性
            - market_regime_performance: 市场环境适应性
            - overall_score: 综合评分 (0-1)
            - recommendation: 推荐建议
        """
        logger = logging.getLogger(__name__)

        # 规则5：禁止就地修改传入的数据，必须先copy
        # 注意：防御性复制在循环内完成，确保后续处理不会修改原始数据

        try:
            logger.info(
                f"开始综合稳定性检验: 因子={factor_name}, "
                f"股票数={len(stock_codes)}, "
                f"时间范围={start_date} ~ {end_date}"
            )
            # 1. 参数校验
            if not factor_name or not isinstance(factor_name, str):
                raise ValueError("因子名称不能为空")
            
            if not stock_codes or len(stock_codes) == 0:
                raise ValueError("股票代码列表不能为空")
            
            if len(stock_codes) < 3:
                raise ValueError(f"稳定性检验至少需要3只股票，当前{len(stock_codes)}只")

            # 2. 获取因子定义
            db = get_db_session()
            repo = FactorRepository(db)
            factor = repo.get_by_name(factor_name)
            db.close()

            if not factor:
                raise ValueError(f"因子 '{factor_name}' 不存在")

            logger.info(f"获取到因子定义: {factor_name} (code={factor.code})")

            # 3. 获取所有股票的因子数据
            all_factor_data = []
            all_ic_series = []

            for stock_code in stock_codes:
                try:
                    data = data_service.get_stock_data(
                        stock_code, start_date, end_date
                    )

                    if data is None or len(data) < 60:
                        logger.warning(f"股票 {stock_code} 数据不足(需>60天)，跳过")
                        continue

                    # 计算因子值
                    factor_series = factor_service.calculator.calculate(
                        data, factor.code
                    )

                    if factor_series is None or len(factor_series.dropna()) < 30:
                        logger.warning(f"股票 {stock_code} 因子计算失败或有效值不足")
                        continue

                    # 计算未来收益率（用于IC计算）
                    df_with_returns = calculate_future_returns(data[['close']], periods=[20])
                    future_return = df_with_returns['future_return_20']
                    
                    # 构建该股票的分析数据框（立即复制，避免后续修改影响原始数据）
                    stock_df = pd.DataFrame({
                        'factor': factor_series,
                        'future_return': future_return,
                        'close': data['close']
                    }).dropna().copy()

                    if len(stock_df) < 30:
                        continue

                    all_factor_data.append({
                        'stock_code': stock_code,
                        'data': stock_df,
                        'factor_series': factor_series.dropna()
                    })

                    # 注意：不再计算单股票的时序IC序列
                    # 时序相关衡量的是"因子能否预测同一只股票未来收益"（择时），
                    # 而截面IC衡量的是"因子能否区分不同股票收益"（选股）。
                    # 截面IC需要在所有股票数据收集完毕后，按日期截面计算。

                except Exception as e:
                    logger.warning(f"处理股票 {stock_code} 时出错: {e}")
                    continue

            # 4. 计算截面IC序列（按日期截面，对多只股票的因子值和收益率计算Spearman相关）
            all_ic_series = []
            if len(all_factor_data) >= 3:
                # 向量化构建面板数据：concat一次，pivot一次（替代iterrows逐行遍历）
                panel_frames = []
                for item in all_factor_data:
                    stock_df = item['data'][['factor', 'future_return']].copy()
                    stock_df['stock_code'] = item['stock_code']
                    panel_frames.append(stock_df)
                panel_df = pd.concat(panel_frames)

                # 对每个日期截面计算Spearman IC（按日期分组向量化）
                from scipy.stats import spearmanr
                ic_dates = []
                ic_values = []
                for date, group in panel_df.groupby(panel_df.index):
                    if len(group) >= 3:  # 至少3只股票才有统计意义
                        ic, _ = spearmanr(group['factor'].values, group['future_return'].values)
                        if not np.isnan(ic):
                            ic_dates.append(date)
                            ic_values.append(ic)

                if len(ic_values) > 0:
                    all_ic_series = [pd.Series(ic_values, index=ic_dates)]

            # 5. 验证数据有效性
            if len(all_factor_data) == 0:
                raise ValueError(
                    "未能获取任何有效的因子数据。"
                    "请检查：1) 股票代码是否正确 2) 时间范围是否合理 "
                    "3) 因子定义是否有效"
                )

            if len(all_ic_series) == 0:
                raise ValueError("未能计算有效的截面IC序列，数据可能不足（需至少3只股票）")

            logger.info(
                f"成功获取 {len(all_factor_data)} 只股票的有效数据, "
                f"截面IC序列长度={len(all_ic_series[0]) if all_ic_series else 0}"
            )

            # 6. 截面IC序列（已经是按日期截面计算的Spearman IC）
            if not all_ic_series:
                raise ValueError("截面IC序列为空，无法进行稳定性分析")
            combined_ic = all_ic_series[0]

            # 横截面方式分析：在每个日期截面上对多只股票计算统计量
            # 构建横截面因子值面板（date × stock_code）
            cross_section_frames = []
            for item in all_factor_data:
                stock_df = item['data'][['factor']].copy()
                stock_df.columns = [item['stock_code']]
                cross_section_frames.append(stock_df)
            cross_section_panel = pd.concat(cross_section_frames, axis=1)

            # 横截面统计量的时间序列（均值和标准差）
            cs_mean = cross_section_panel.mean(axis=1)
            cs_std = cross_section_panel.std(axis=1)
            combined_factor = cs_mean.dropna()

            # 6. 执行多维度稳定性分析
            results = {}
            scores = []

            # 6.1 分布稳定性分析
            try:
                if len(combined_factor) >= 504:  # 至少2年数据(252*2)
                    dist_result = self.calculate_distribution_stability(
                        combined_factor, window=252, method="ks"
                    )
                    results["distribution_stability"] = dist_result
                    scores.append(dist_result.get("stability_score", 0.5))
                    logger.info(f"分布稳定性得分: {dist_result.get('stability_score', 0):.3f}")
                else:
                    results["distribution_stability"] = {
                        "warning": "数据不足504个点，跳过分布稳定性检验",
                        "data_points": len(combined_factor),
                        "required": 504
                    }
                    logger.warning("数据不足，跳过分布稳定性检验")
            except Exception as e:
                results["distribution_stability"] = {"error": str(e)}
                logger.error(f"分布稳定性检验失败: {e}")

            # 6.2 时间序列平稳性分析（ADF检验）
            try:
                ts_result = self.calculate_time_series_stability(
                    combined_ic, maxlag=10
                )
                results["time_series_stationarity"] = ts_result
                
                # 平稳性得分：p < 0.05 得高分
                if ts_result.get("is_stationary"):
                    scores.append(0.8)
                else:
                    scores.append(0.3)
                
                logger.info(
                    f"时间序列平稳性: p_value={ts_result.get('p_value', 1):.4f}, "
                    f"is_stationary={ts_result.get('is_stationary')}"
                )
            except Exception as e:
                results["time_series_stationarity"] = {"error": str(e)}
                logger.error(f"时间序列平稳性检验失败: {e}")

            # 6.3 变异系数分析
            try:
                cv_result = self.calculate_coefficient_of_variation(combined_ic)
                results["coefficient_of_variation"] = cv_result
                
                # CV得分：CV越小越稳定
                cv = cv_result.get("cv", float('inf'))
                if np.isnan(cv) or np.isinf(cv):
                    cv_score = 0.3
                elif cv < 0.5:
                    cv_score = 0.9
                elif cv < 1.0:
                    cv_score = 0.7
                elif cv < 2.0:
                    cv_score = 0.5
                else:
                    cv_score = 0.2
                scores.append(cv_score)
                
                logger.info(f"变异系数 CV={cv:.3f}, 得分={cv_score:.3f}")
            except Exception as e:
                results["coefficient_of_variation"] = {"error": str(e)}
                logger.error(f"变异系数计算失败: {e}")

            # 6.4 滚动窗口稳定性分析
            try:
                # 使用横截面均值的时序数据进行滚动分析
                rolling_data = pd.DataFrame({
                    'factor': cs_mean,
                    'future_return': pd.DataFrame({item['stock_code']: item['data']['future_return'] for item in all_factor_data}).mean(axis=1).reindex(cs_mean.index)
                    if len(all_factor_data) > 0 else pd.Series(dtype=float)
                }).dropna()
                rolling_result = self.calculate_rolling_stability(
                    rolling_data,
                    factor_name='factor',
                    return_col='future_return',
                    windows=[20, 60, 120]
                )
                results["rolling_window_analysis"] = rolling_result
                
                # 滚动稳定性得分：基于IR的均值
                ir_values = [
                    v.get('ir') for v in rolling_result.values() 
                    if v.get('ir') is not None and not np.isnan(v.get('ir'))
                ]
                if ir_values:
                    mean_ir = np.mean([abs(ir) for ir in ir_values])
                    if mean_ir > 0.5:
                        roll_score = 0.9
                    elif mean_ir > 0.3:
                        roll_score = 0.7
                    elif mean_ir > 0.1:
                        roll_score = 0.5
                    else:
                        roll_score = 0.3
                    scores.append(roll_score)
                
                logger.info(f"滚动窗口分析完成，包含 {len(rolling_result)} 个周期")
            except Exception as e:
                results["rolling_window_analysis"] = {"error": str(e)}
                logger.error(f"滚动窗口稳定性分析失败: {e}")

            # 6.5 市场环境适应性分析
            try:
                # 使用横截面均值构建市场环境分析数据
                market_data = pd.DataFrame({
                    'factor': cs_mean,
                    'future_return': pd.DataFrame({item['stock_code']: item['data']['future_return'] for item in all_factor_data}).mean(axis=1).reindex(cs_mean.index)
                    if len(all_factor_data) > 0 else pd.Series(dtype=float),
                    'close': pd.DataFrame({item['stock_code']: item['data']['close'] for item in all_factor_data}).mean(axis=1).reindex(cs_mean.index)
                    if len(all_factor_data) > 0 else pd.Series(dtype=float)
                }).dropna()
                regime_result = self.calculate_market_regime_performance(
                    market_data,
                    factor_name='factor',
                    return_col='future_return',
                    price_col='close'
                )
                results["market_regime_performance"] = regime_result
                
                # 市场环境适应性得分：各环境IC差异越小越好
                regime_ics = [v.get('mean_ic') for v in regime_result.values()]
                if len(regime_ics) >= 2:
                    ic_std = np.std(regime_ics)
                    if ic_std < 0.02:
                        regime_score = 0.9
                    elif ic_std < 0.05:
                        regime_score = 0.7
                    elif ic_std < 0.10:
                        regime_score = 0.5
                    else:
                        regime_score = 0.3
                    scores.append(regime_score)
                
                logger.info(f"市场环境分析完成，包含 {len(regime_result)} 个状态")
            except Exception as e:
                results["market_regime_performance"] = {"error": str(e)}
                logger.error(f"市场环境适应性分析失败: {e}")

            # 7. 计算综合评分和推荐建议
            overall_score = np.mean(scores) if scores else 0.0
            
            if overall_score >= 0.75:
                recommendation = "因子稳定性优秀，适合实盘应用"
                risk_level = "低"
            elif overall_score >= 0.55:
                recommendation = "因子稳定性良好，可考虑用于策略构建"
                risk_level = "中低"
            elif overall_score >= 0.40:
                recommendation = "因子稳定性一般，建议结合其他因子使用"
                risk_level = "中"
            else:
                recommendation = "因子稳定性较差，存在较高失效风险，谨慎使用"
                risk_level = "高"

            # 8. 构建最终返回结果
            final_result = {
                **results,
                "summary": {
                    "overall_score": round(float(overall_score), 4),
                    "risk_level": risk_level,
                    "recommendation": recommendation,
                    "analysis_dimensions": len([k for k in results.keys() if 'error' not in k]),
                    "total_stocks_analyzed": len(all_factor_data),
                    "valid_data_points": {
                        "factor_values": int(len(combined_factor)),
                        "ic_observations": int(len(combined_ic))
                    },
                    "warnings": self._generate_warnings(results)
                },
                "metadata": {
                    "factor_name": factor_name,
                    "stock_codes_count": len(stock_codes),
                    "effective_stocks": len(all_factor_data),
                    "date_range": f"{start_date} ~ {end_date}",
                    "analysis_timestamp": pd.Timestamp.now().isoformat()
                }
            }

            logger.info(
                f"综合稳定性检验完成: 总体得分={overall_score:.3f}, "
                f"风险等级={risk_level}, 建议={recommendation}"
            )

            return final_result

        except ValueError as ve:
            logger.error(f"参数校验失败: {ve}")
            raise ve
        except Exception as e:
            logger.error(f"综合稳定性检验异常: {e}", exc_info=True)
            raise RuntimeError(f"综合稳定性检验失败: {str(e)}") from e

    def _generate_warnings(self, results: Dict) -> List[str]:
        """生成警告信息"""
        warnings = []

        # 分布稳定性警告
        dist = results.get("distribution_stability", {})
        if dist.get("stable_ratio", 1.0) < 0.6:
            warnings.append("⚠️ 因子分布在不同时期差异显著，可能存在结构性变化")

        # 平稳性警告
        ts = results.get("time_series_stationarity", {})
        if not ts.get("is_stationary") and ts.get("p_value", 1) > 0.1:
            warnings.append("⚠️ IC序列不平稳，因子预测能力可能不稳定")

        # 变异系数警告
        cv = results.get("coefficient_of_variation", {})
        if cv.get("cv", 0) > 1.5:
            warnings.append("⚠️ IC变异程度较高，因子收益波动较大")

        # 市场环境警告
        regime = results.get("market_regime_performance", {})
        bull_ic = regime.get("bull", {}).get("mean_ic", 0)
        bear_ic = regime.get("bear", {}).get("mean_ic", 0)
        if abs(bull_ic - bear_ic) > 0.1:
            warnings.append("⚠️ 因子在牛熊市表现差异较大，需注意市场环境风险")

        return warnings


# 全局因子稳定性分析服务实例
factor_stability_service = FactorStabilityService()

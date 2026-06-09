"""
因子分析服务模块 - IC/IR统计、SHAP分析（含未来函数检测）
"""
import hashlib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import json

# 配置日志
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

from backend.core.settings import settings
from backend.core.database import get_db_session
from backend.repositories.factor_repository import AnalysisCacheRepository
from backend.models.factor import AnalysisCacheModel
from backend.services.factor_service import factor_service
from backend.services.alphalens_analysis_service import alphalens_analysis_service
from backend.services.data_service import data_service
from backend.services.factor_preprocessing_pipeline import (
    FactorPreprocessingPipeline,
    PreprocessingConfig,
    default_pipeline,
)
from backend.services.lookahead_bias_detector import (
    lookahead_bias_detector,
    BiasRiskLevel,
)
from backend.utils.safe_math import safe_ir, safe_divide
from backend.utils.weight_utils import normalize_weights


class AnalysisService:
    """因子分析服务类"""

    def __init__(self):
        self.results_cache = {}

    def _serialize_for_cache(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """将结果序列化为可JSON序列化的格式"""
        serialized = {
            "metadata": results["metadata"],
            "ic_ir": {},
            "shap": results.get("shap", {}),
            "alphalens": results.get("alphalens", {}),
            "lookahead_bias": results.get("lookahead_bias", {}),
        }

        # 序列化 factor_data（缓存命中时避免重新计算因子）
        if "factor_data" in results:
            serialized_factor_data = {}
            for stock_code, df in results["factor_data"].items():
                serialized_factor_data[stock_code] = df.to_dict(orient="list")
                # 保留索引信息以便反序列化恢复
                if hasattr(df.index, 'to_list'):
                    index_list = df.index.to_list()
                    serialized_factor_data[stock_code]["__index__"] = [
                        str(idx) if hasattr(idx, 'isoformat') else idx
                        for idx in index_list
                    ]
                    # 记录索引类型，避免整数索引被误解析为日期
                    serialized_factor_data[stock_code]["__index_type__"] = type(df.index).__name__
            serialized["factor_data"] = serialized_factor_data

        # 序列化 IC/IR 结果
        if "ic_ir" in results:
            ic_ir = results["ic_ir"]
            if "ic_stats" in ic_ir:
                # 需要将 IC序列 中的 Timestamp 键转换为字符串
                serialized_stats = {}
                for factor_name, stats in ic_ir["ic_stats"].items():
                    serialized_stats[factor_name] = {}
                    for key, value in stats.items():
                        if key == "IC序列" and isinstance(value, dict):
                            # 将 Timestamp 键转换为字符串
                            serialized_stats[factor_name][key] = {
                                str(k) if hasattr(k, 'isoformat') else k: v
                                for k, v in value.items()
                            }
                        else:
                            serialized_stats[factor_name][key] = value
                serialized["ic_ir"]["ic_stats"] = serialized_stats

            if "monthly_ic" in ic_ir:
                serialized["ic_ir"]["monthly_ic"] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in ic_ir["monthly_ic"].items()
                }

            if "rolling_ir" in ic_ir:
                serialized["ic_ir"]["rolling_ir"] = {
                    k: (
                        {str(idx): val for idx, val in v.to_dict().items()} if hasattr(v, 'to_dict')
                        else {str(idx): val for idx, val in dict(enumerate(v.tolist())).items()} if hasattr(v, 'tolist')
                        else v
                    )
                    for k, v in ic_ir["rolling_ir"].items()
                }

        return serialized

    def _deserialize_from_cache(self, cached_data: Dict[str, Any]) -> Dict[str, Any]:
        """从缓存数据反序列化"""
        # 从缓存中恢复 factor_data
        factor_data = {}
        if "factor_data" in cached_data:
            for stock_code, data in cached_data["factor_data"].items():
                data = dict(data)  # 复制，避免修改缓存原始数据
                index_data = data.pop("__index__", None)
                index_type = data.pop("__index_type__", None)
                df = pd.DataFrame(data)
                if index_data:
                    if index_type == "DatetimeIndex":
                        try:
                            df.index = pd.to_datetime(index_data)
                        except (ValueError, TypeError):
                            df.index = index_data
                    elif index_type == "RangeIndex":
                        # RangeIndex 自动生成，无需手动设置
                        pass
                    else:
                        # 其他类型（如 Int64Index）直接赋值
                        try:
                            df.index = [int(x) if str(x).isdigit() else x for x in index_data]
                        except (ValueError, TypeError):
                            df.index = index_data
                factor_data[stock_code] = df

        result = {
            "metadata": cached_data["metadata"],
            "factor_data": factor_data,
            "ic_ir": cached_data.get("ic_ir", {}),
            "shap": cached_data.get("shap", {}),
            "alphalens": cached_data.get("alphalens", {}),
        }

        # 将 monthly_ic 转回 DataFrame
        if "ic_ir" in result and "monthly_ic" in result["ic_ir"]:
            result["ic_ir"]["monthly_ic"] = {
                k: pd.DataFrame(v) if isinstance(v, dict) else v
                for k, v in result["ic_ir"]["monthly_ic"].items()
            }

        # 将 rolling_ir 转回 Series
        if "ic_ir" in result and "rolling_ir" in result["ic_ir"]:
            result["ic_ir"]["rolling_ir"] = {}
            for k, v in result["ic_ir"]["rolling_ir"].items():
                if isinstance(v, dict):
                    # 尝试解析字符串索引为 datetime
                    try:
                        result["ic_ir"]["rolling_ir"][k] = pd.Series(
                            list(v.values()),
                            index=pd.to_datetime(list(v.keys()))
                        )
                    except Exception as e:
                        # 如果失败，直接使用字符串索引
                        import logging
                        logging.getLogger(__name__).debug(f"日期解析失败，使用字符串索引: {e}")
                        result["ic_ir"]["rolling_ir"][k] = pd.Series(v)
                else:
                    result["ic_ir"]["rolling_ir"][k] = v

        return result

    def _generate_cache_key(
        self, stock_codes: List[str], factor_names: List[str], start_date: str, end_date: str,
        factor_version_hash: Optional[str] = None,
    ) -> str:
        """生成缓存键（包含因子版本哈希，因子更新时自动失效）"""
        parts = [
            ",".join(sorted(stock_codes)),
            ",".join(sorted(factor_names)),
            start_date,
            end_date,
        ]
        if factor_version_hash:
            parts.append(factor_version_hash)
        key_str = "_".join(parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:32]

    def _compute_factor_version_hash(self, factor_names: List[str]) -> str:
        """计算因子版本哈希（基于因子定义的code字段）"""
        try:
            from backend.repositories.factor_repository import FactorRepository
            from backend.core.database import get_db_session

            db = get_db_session()
            try:
                repo = FactorRepository(db)
                codes = []
                for name in factor_names:
                    factor = repo.get_by_name(name)
                    if factor and hasattr(factor, 'code') and factor.code:
                        codes.append(f"{name}:{factor.code}")

                if codes:
                    return hashlib.md5("|".join(sorted(codes)).encode()).hexdigest()[:16]
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"计算因子版本哈希失败: {e}")
        return ""

    def analyze(
        self,
        stock_codes: List[str],
        factor_names: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        rolling_window: int = 252,
    ) -> Dict[str, Any]:
        """
        执行完整的因子分析

        Args:
            stock_codes: 股票代码列表
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            rolling_window: 滚动窗口大小

        Returns:
            包含所有分析结果的字典
        """
        # 获取因子版本哈希（因子定义变更时缓存自动失效）
        factor_version_hash = self._compute_factor_version_hash(factor_names)

        cache_key = self._generate_cache_key(stock_codes, factor_names, start_date, end_date, factor_version_hash)

        # 缓存检查必须在数据获取之前，避免缓存命中时浪费昂贵的计算
        if use_cache:
            db = get_db_session()
            try:
                repo = AnalysisCacheRepository(db)
                cached = repo.get_by_key(cache_key)
                if cached:
                    return self._deserialize_from_cache(cached.result_data)
            finally:
                db.close()

        factor_data = factor_service.calculate_factors_for_stocks(
            stock_codes, factor_names, start_date, end_date, rolling_window
        )

        if not factor_data:
            raise ValueError("未能获取任何有效的因子数据")

        logger.info("开始执行因子数据美颜处理...")
        preprocessing_pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method="mad",
            enable_market_cap_neutralization=True,
            enable_industry_neutralization=True,
            standardize_method="zscore",
            cross_sectional=(len(stock_codes) > 1),
        ))

        factor_data, preprocessing_stats = preprocessing_pipeline.process_multi_stock_factors(
            factor_data=factor_data,
            factor_names=factor_names,
            parallel_stocks=(len(stock_codes) > 3),
        )

        logger.info(f"因子数据美颜完成，统计信息:\n{preprocessing_pipeline.get_processing_summary(preprocessing_stats)}")

        results = {
            "metadata": {
                "stock_codes": stock_codes,
                "factor_names": factor_names,
                "start_date": start_date,
                "end_date": end_date,
                "rolling_window": rolling_window,
                "analysis_time": datetime.now().isoformat(),
            },
            "factor_data": factor_data,
        }

        ic_ir_results = self.calculate_ic_ir(factor_data, factor_names, stock_codes)
        results["ic_ir"] = ic_ir_results

        # 未来函数检测（Look-ahead Bias Detection）
        # 需要先为factor_data添加future_return_1列
        factor_data_for_detection = {}
        for stock_code, df in factor_data.items():
            df_det = df.copy()
            if "future_return_1" not in df_det.columns and "close" in df_det.columns:
                df_det["future_return_1"] = df_det["close"].pct_change(1).shift(-1)
            factor_data_for_detection[stock_code] = df_det
        results["lookahead_bias"] = self._detect_lookahead_bias_for_all_factors(
            factor_data_for_detection, factor_names, stock_codes
        )

        if len(stock_codes) >= 2:
            skip_industry = len(stock_codes) > 30
            alphalens_results = self._run_alphalens_analysis(
                factor_data, factor_names, stock_codes, skip_industry=skip_industry
            )
            results["alphalens"] = alphalens_results

        if SHAP_AVAILABLE:
            shap_results = self.calculate_shap(factor_data, factor_names)
            results["shap"] = shap_results
        else:
            results["shap"] = {"error": "SHAP not available"}

        if use_cache:
            db = get_db_session()
            try:
                repo = AnalysisCacheRepository(db)
                serialized_results = self._serialize_for_cache(results)
                cache = AnalysisCacheModel(
                    cache_key=cache_key,
                    stock_codes=",".join(stock_codes),
                    factor_names=",".join(factor_names),
                    start_date=start_date,
                    end_date=end_date,
                    result_data=serialized_results,
                )
                repo.create(cache)
            finally:
                db.close()

        return results

    def _run_alphalens_analysis(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str],
        stock_codes: List[str],
        skip_industry: bool = False,
    ) -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_results = {}

        # 构建 pricing_df（所有因子共用，仅依赖close价格）
        all_dates = set()
        for df in factor_data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)

        pricing_df = pd.DataFrame(index=all_dates)
        for stock_code in stock_codes:
            df = factor_data[stock_code]
            if "close" in df.columns:
                pricing_df[stock_code] = df["close"]
        pricing_df = pricing_df.dropna(how="all")

        # 行业分类只获取一次（所有因子共用）
        groupby_dict = None
        if not skip_industry:
            try:
                groupby_dict = data_service.get_industry_classification(stock_codes)
                groupby_dict = {k: v for k, v in groupby_dict.items() if v}
                if not groupby_dict:
                    groupby_dict = None
            except Exception as e:
                logger.warning(f"获取行业分类失败: {e}")
                groupby_dict = None

        # 预提取每个因子的 factor_values_dict
        factor_values_maps = {}
        skip_factors = set()
        for factor_name in factor_names:
            fv_dict = {}
            for stock_code in stock_codes:
                df = factor_data[stock_code]
                if factor_name in df.columns:
                    fv_dict[stock_code] = df[factor_name].dropna()
            if len(fv_dict) < 2:
                all_results[factor_name] = {"error": "有效股票数不足2只，无法进行Alphalens分析"}
                skip_factors.add(factor_name)
            else:
                factor_values_maps[factor_name] = fv_dict

        # 并行执行各因子的 alphalens 分析
        def _analyze_one(name):
            try:
                result = alphalens_analysis_service.full_analysis(
                    factor_values_dict=factor_values_maps[name],
                    pricing_df=pricing_df,
                    groupby_dict=groupby_dict,
                )
                return name, result
            except Exception as e:
                logger.error(f"Alphalens分析因子 {name} 失败: {e}", exc_info=True)
                return name, {"error": str(e)}

        analyzable = [n for n in factor_names if n not in skip_factors]
        if analyzable:
            with ThreadPoolExecutor(max_workers=min(len(analyzable), 4)) as executor:
                futures = {executor.submit(_analyze_one, name): name for name in analyzable}
                for future in as_completed(futures):
                    name, result = future.result()
                    all_results[name] = result

        return all_results

    def calculate_ic_ir(
        self, factor_data: Dict[str, pd.DataFrame], factor_names: List[str],
        stock_codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        计算IC和IR统计（统一入口，自动路由到单股票或多股票模式）

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_names: 因子名称列表
            stock_codes: 股票代码列表（多股票模式需要）

        Returns:
            IC/IR统计结果
        """
        # 注意：不直接修改输入factor_data，避免副作用
        factor_data_copy = {}
        for stock_code, df in factor_data.items():
            df_copy = df.copy()
            df_copy["future_return_1"] = df_copy["close"].pct_change(1).shift(-1)
            df_copy["future_return_5"] = df_copy["close"].pct_change(5).shift(-5)
            for col in df_copy.columns:
                df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan)
            factor_data_copy[stock_code] = df_copy

        if len(factor_data_copy) == 1:
            result = self._calculate_single_stock_ic(factor_data_copy, factor_names)
            result["warning"] = "时序IC仅评估择时能力，不能回答选股问题。建议使用多只股票进行横截面IC分析。"
            return result

        # 多股票模式：使用Alphalens（业界金标准）
        try:
            result = self._calculate_multi_stock_ic_alphalens(factor_data_copy, factor_names, stock_codes)
            if result.get("ic_stats"):
                logger.info("Alphalens多股票IC计算成功")
                return result
            else:
                logger.warning("Alphalens返回空结果，尝试手动回退方法")
        except Exception as e:
            logger.warning(f"Alphalens多股票IC计算失败: {e}，尝试手动回退方法")

        # Alphalens失败时，回退到手动横截面IC计算
        try:
            result = self._calculate_multi_stock_ic_manual(factor_data_copy, factor_names, stock_codes)
            if result.get("ic_stats"):
                logger.info("手动横截面IC计算成功（Alphalens回退）")
                return result
        except Exception as e:
            logger.error(f"手动横截面IC计算也失败: {e}", exc_info=True)

        logger.error("Alphalens和手动方法均失败，无法进行横截面IC分析")
        return {
            "ic_stats": {},
            "monthly_ic": {},
            "rolling_ir": {},
            "error": "Alphalens和手动方法均失败",
        }

    def _calculate_multi_stock_ic_manual(
        self, factor_data: Dict[str, pd.DataFrame], factor_names: List[str],
        stock_codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """手动多股票横截面IC计算（当Alphalens不可用时的回退方案）

        使用Spearman秩相关（业界标准）+ 向量化groupby操作
        """
        from scipy import stats as scipy_stats

        codes = stock_codes or list(factor_data.keys())

        ic_series = {}
        for factor_name in factor_names:
            # 向量化构建横截面数据：拼接所有股票，用groupby计算每日IC
            all_rows = []
            for stock_code in codes:
                df = factor_data.get(stock_code)
                if df is None:
                    continue
                if factor_name not in df.columns or "future_return_1" not in df.columns:
                    continue

                stock_data = df[[factor_name, "future_return_1"]].copy()
                stock_data["stock_code"] = stock_code
                # 保留日期信息
                if isinstance(df.index, pd.DatetimeIndex):
                    stock_data["date"] = df.index
                elif "date" in df.columns:
                    stock_data["date"] = df["date"]
                else:
                    stock_data["date"] = df.index
                all_rows.append(stock_data)

            if not all_rows:
                continue

            merged = pd.concat(all_rows, ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"])

            # 清理无效值
            valid_mask = (
                merged[factor_name].notna() & merged["future_return_1"].notna()
                & ~np.isinf(merged[factor_name]) & ~np.isinf(merged["future_return_1"])
            )
            merged = merged[valid_mask]

            if len(merged) < 20:
                continue

            # 向量化：按日期分组计算Spearman秩相关IC
            def _spearman_ic(group):
                if len(group) < 2:
                    return np.nan
                return scipy_stats.spearmanr(group[factor_name], group["future_return_1"]).correlation

            daily_ic = merged.groupby("date").apply(_spearman_ic)
            daily_ic = daily_ic.dropna()

            if len(daily_ic) > 5:
                ic_series[factor_name] = daily_ic

        # 计算统计指标
        ic_stats = {}
        for factor_name, ic_s in ic_series.items():
            ic_mean = float(ic_s.mean())
            ic_std = float(ic_s.std()) if len(ic_s) > 1 else 0.0
            ir = safe_ir(ic_mean, ic_std, default=0.0)

            n = len(ic_s)
            t_stat = ic_mean / (ic_std / np.sqrt(n)) if n > 1 and ic_std > 0 else 0.0
            p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))) if n > 1 else 1.0

            ic_stats[factor_name] = {
                "IC均值": ic_mean,
                "IC标准差": ic_std,
                "IR": ir,
                "IC>0占比": float((ic_s > 0).mean()),
                "IC绝对值均值": float(abs(ic_s).mean()),
                "IC序列": ic_s.to_dict(),
                "IC类型": "横截面IC（手动计算）",
                "t统计量": t_stat,
                "p值": p_value,
            }

        return {
            "ic_stats": ic_stats,
            "monthly_ic": self._calculate_monthly_ic(ic_series),
            "rolling_ir": self._calculate_rolling_ir(ic_series, window=20),
            "warning": "使用手动横截面IC计算（Alphalens不可用或返回空结果）",
        }

    def _calculate_single_stock_ic(
        self, factor_data: Dict[str, pd.DataFrame], factor_names: List[str],
        use_tradable_mask: bool = True
    ) -> Dict[str, Any]:
        """单股票时序IC计算（Mask-First增强版）"""
        stock_code = list(factor_data.keys())[0]
        df = factor_data[stock_code]

        if use_tradable_mask and "tradable_mask" in df.columns:
            tradable_mask = df["tradable_mask"]
            logger.info(f"✅ IC分析: 使用Mask-First设计，可交易比例 {tradable_mask.mean():.1%}")
            mask_stats = {
                "total_days": len(df),
                "tradable_days": int(tradable_mask.sum()),
                "tradable_ratio": float(tradable_mask.mean()),
                "limit_up_days": int(df["is_limit_up"].sum()) if "is_limit_up" in df.columns else 0,
                "limit_down_days": int(df["is_limit_down"].sum()) if "is_limit_down" in df.columns else 0,
            }
        else:
            tradable_mask = None
            mask_stats = None
            if use_tradable_mask:
                logger.warning("⚠️ IC分析: 未找到tradable_mask列！IC可能虚高18%")

        ic_series = {}
        rank_ic_series = {}
        for factor_name in factor_names:
            if factor_name not in df.columns:
                continue
            factor_values = df[factor_name]
            return_values = df["future_return_1"]

            if tradable_mask is not None:
                valid_mask = (
                    factor_values.notna() & return_values.notna()
                    & ~np.isinf(factor_values) & ~np.isinf(return_values)
                )
                combined_mask = valid_mask & tradable_mask
                factor_clean = factor_values[combined_mask]
                return_clean = return_values[combined_mask]
                min_periods = max(2, int(60 * 0.6))
                rolling_ic = factor_clean.rolling(window=60, min_periods=min_periods).corr(return_clean)
                rolling_rank_ic = factor_clean.rank().rolling(window=60, min_periods=min_periods).corr(return_clean.rank())
            else:
                valid_mask = (
                    factor_values.notna() & return_values.notna()
                    & ~np.isinf(factor_values) & ~np.isinf(return_values)
                )
                factor_clean = factor_values[valid_mask]
                return_clean = return_values[valid_mask]
                if len(factor_clean) < 60:
                    continue
                rolling_ic = factor_clean.rolling(window=60).corr(return_clean)
                rolling_rank_ic = factor_clean.rank().rolling(window=60).corr(return_clean.rank())

            rolling_ic = rolling_ic.replace([np.inf, -np.inf], np.nan).dropna()
            rolling_rank_ic = rolling_rank_ic.replace([np.inf, -np.inf], np.nan).dropna()

            if len(rolling_ic) > 0:
                ic_series[factor_name] = rolling_ic
            if len(rolling_rank_ic) > 0:
                rank_ic_series[factor_name] = rolling_rank_ic

        ic_stats = {}
        for factor_name, ic_s in ic_series.items():
            ic_mean = ic_s.mean()
            ic_std = ic_s.std()
            ir = safe_ir(ic_mean, ic_std, default=0.0)
            stats = {
                "IC均值": ic_mean, "IC标准差": ic_std, "IR": ir,
                "IC>0占比": (ic_s > 0).mean(), "IC绝对值均值": abs(ic_s).mean(),
                "IC序列": ic_s.to_dict(), "IC类型": "时序IC（单股票）",
                "Mask-First": tradable_mask is not None,
            }
            if factor_name in rank_ic_series:
                rank_ic_s = rank_ic_series[factor_name]
                rank_ic_mean = rank_ic_s.mean()
                rank_ic_std = rank_ic_s.std()
                stats["Rank_IC均值"] = rank_ic_mean
                stats["Rank_IC标准差"] = rank_ic_std
                stats["Rank_IR"] = safe_ir(rank_ic_mean, rank_ic_std, default=0.0)
            ic_stats[factor_name] = stats

        result = {
            "ic_stats": ic_stats,
            "monthly_ic": self._calculate_monthly_ic(ic_series),
            "rolling_ir": self._calculate_rolling_ir(ic_series, window=20),
        }
        if mask_stats:
            result["mask_statistics"] = mask_stats
        return result

    def _calculate_multi_stock_ic_alphalens(
        self, factor_data: Dict[str, pd.DataFrame], factor_names: List[str],
        stock_codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """多股票横截面IC计算（委托Alphalens——业界金标准）"""
        codes = stock_codes or list(factor_data.keys())

        all_ic_stats = {}
        all_monthly_ic = {}
        all_rolling_ir = {}
        is_single_factor = len(factor_names) == 1

        try:
            groupby_dict = data_service.get_industry_classification(codes)
            groupby_dict = {k: v for k, v in groupby_dict.items() if v}
        except Exception as e:
            logger.warning(f"获取行业分类失败: {e}")
            groupby_dict = None

        for factor_name in factor_names:
            # 为当前因子构建独立的 pricing_df 和 factor_values_dict
            factor_values_dict = {}
            factor_dates = set()

            for stock_code in codes:
                df = factor_data.get(stock_code)
                if df is not None and factor_name in df.columns and "close" in df.columns:
                    series = df[factor_name].dropna()
                    if len(series) > 0:
                        factor_values_dict[stock_code] = series
                        factor_dates.update(series.index)

            if len(factor_values_dict) < 2:
                logger.warning(f"因子 {factor_name} 有效股票数不足2只，跳过")
                continue

            # 使用因子日期范围构建pricing_df（确保forward returns可计算）
            factor_dates_sorted = sorted(factor_dates)

            # 构建完整的pricing_df（包含因子日期+额外未来日期用于计算forward returns）
            all_price_dates = set()
            for stock_code in codes:
                df = factor_data.get(stock_code)
                if df is not None and "close" in df.columns:
                    all_price_dates.update(df.index)

            # 合并因子日期和价格日期
            combined_dates = sorted(factor_dates | all_price_dates)

            prices = pd.DataFrame(index=combined_dates)
            for stock_code in codes:
                df = factor_data.get(stock_code)
                if df is not None and "close" in df.columns:
                    prices[stock_code] = df["close"]
            prices = prices.dropna(how="all")

            if prices.empty:
                logger.warning(f"因子 {factor_name} 无法构建pricing数据")
                continue

            try:
                al_result = alphalens_analysis_service.full_analysis(
                    factor_values_dict=factor_values_dict,
                    pricing_df=prices,
                    groupby_dict=groupby_dict,
                    max_loss=1.0,
                )
            except Exception as e:
                logger.warning(f"因子 {factor_name} Alphalens分析失败: {e}")
                continue

            if "error" in al_result:
                logger.warning(f"因子 {factor_name} Alphalens分析失败: {al_result['error']}")
                continue

            ic_analysis = al_result.get("ic_analysis", {})
            for ic_type_key in ["spearman_ic", "pearson_ic"]:
                ic_data = ic_analysis.get(ic_type_key, {})
                for period_label, period_stats in ic_data.items():
                    if not isinstance(period_stats, dict) or "error" in period_stats:
                        continue
                    ic_series_data = period_stats.get("ic_series", {})
                    dates = ic_series_data.get("dates", [])
                    values = ic_series_data.get("values", [])
                    valid_values = [v for v in values if v is not None]
                    if not valid_values:
                        continue
                    ic_s = pd.Series(
                        [v if v is not None else np.nan for v in values],
                        index=pd.to_datetime(dates) if dates else range(len(values)),
                    ).dropna()

                    ic_type_name = period_stats.get("ic_type", ic_type_key)
                    if is_single_factor:
                        factor_key = f"{ic_type_key}_{period_label}"
                    else:
                        factor_key = f"{factor_name}_{ic_type_key}_{period_label}"

                    all_ic_stats[factor_key] = {
                        "IC均值": period_stats.get("mean_ic", 0),
                        "IC标准差": period_stats.get("std_ic", 0),
                        "IR": period_stats.get("ir", 0),
                        "IC>0占比": period_stats.get("ic_positive_ratio", 0),
                        "IC绝对值均值": abs(ic_s).mean() if len(ic_s) > 0 else 0,
                        "IC序列": ic_s.to_dict(),
                        "IC类型": f"横截面{ic_type_name}（Alphalens）",
                        "t统计量": period_stats.get("t_statistic", 0),
                        "p值": period_stats.get("p_value", 1),
                    }
                    if len(ic_s) > 0:
                        all_monthly_ic[factor_key] = self._calculate_monthly_ic({factor_key: ic_s})[factor_key]
                        all_rolling_ir[factor_key] = self._calculate_rolling_ir({factor_key: ic_s}, window=60)[factor_key]

        if not all_ic_stats:
            logger.warning("Alphalens未能返回有效IC数据，请检查因子数据质量")
            return {
                "ic_stats": {},
                "monthly_ic": {},
                "rolling_ir": {},
                "warning": "Alphalens分析未返回有效结果，请检查因子数据是否包含足够有效值",
            }

        return {
            "ic_stats": all_ic_stats,
            "monthly_ic": all_monthly_ic,
            "rolling_ir": all_rolling_ir,
        }

    def _calculate_monthly_ic(
        self, ic_series: Dict[str, pd.Series]
    ) -> Dict[str, pd.DataFrame]:
        """计算月度IC热力图数据"""
        monthly_ic = {}
        for factor_name, ic_s in ic_series.items():
            ic_df = pd.DataFrame({"ic": ic_s})
            ic_df["year"] = ic_df.index.year
            ic_df["month"] = ic_df.index.month
            pivot = ic_df.pivot_table(values="ic", index="year", columns="month", aggfunc="mean")
            monthly_ic[factor_name] = pivot
        return monthly_ic

    def _calculate_rolling_ir(
        self, ic_series: Dict[str, pd.Series], window: int = 60
    ) -> Dict[str, pd.Series]:
        rolling_ir = {}
        for factor_name, ic_s in ic_series.items():
            min_periods = max(1, window // 4)
            rolling_mean = ic_s.rolling(window=window, min_periods=min_periods).mean()
            rolling_std = ic_s.rolling(window=window, min_periods=min_periods).std()
            # 使用 safe_divide 统一处理除零，避免 *1e6 hack
            ir = safe_divide(rolling_mean, rolling_std, default=np.nan)
            rolling_ir[factor_name] = ir
        return rolling_ir

    def _detect_lookahead_bias_for_all_factors(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str],
        stock_codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        对所有因子执行未来函数检测（多股票模式优先使用横截面检测）

        Args:
            factor_data: {stock_code: DataFrame} 格式的因子数据
            factor_names: 因子名称列表
            stock_codes: 股票代码列表

        Returns:
            检测结果字典，包含每个因子的检测结果和汇总信息
        """
        all_bias_results = {}
        codes = stock_codes or list(factor_data.keys())
        has_multiple_stocks = len(factor_data) >= 2

        for factor_name in factor_names:
            try:
                if has_multiple_stocks and len(codes) >= 3:
                    # 多股票模式：使用横截面检测（更准确）
                    all_factor_rows = []
                    all_return_rows = []
                    for stock_code in codes:
                        df = factor_data.get(stock_code)
                        if df is None or factor_name not in df.columns:
                            continue
                        if "future_return_1" not in df.columns:
                            continue
                        # 向量化过滤替代 iterrows，避免逐行创建 Series 对象
                        valid_mask = df[factor_name].notna() & df["future_return_1"].notna()
                        valid_mask &= ~np.isinf(df[factor_name]) & ~np.isinf(df["future_return_1"])
                        if valid_mask.sum() >= 50:
                            valid_df = df.loc[valid_mask, [factor_name, "future_return_1"]].copy()
                            valid_df["date"] = valid_df.index
                            valid_df["stock_code"] = stock_code
                            valid_df = valid_df.rename(columns={"future_return_1": "return"})
                            all_factor_rows.append(valid_df)

                    if len(all_factor_rows) >= 50:
                        bias_df = pd.concat(all_factor_rows, ignore_index=True)
                        # factor_df仅包含date/stock_code/factor_value，避免与return_df的return列冲突
                        detection_result = lookahead_bias_detector.detect_cross_sectional(
                            factor_df=bias_df[["date", "stock_code", factor_name]].rename(columns={factor_name: "factor_value"}),
                            return_df=bias_df[["date", "stock_code", "return"]],
                            factor_name="factor_value",
                        )
                    else:
                        # 数据不足时回退到单股票检测
                        first_stock = codes[0]
                        df = factor_data.get(first_stock)
                        if df is not None and factor_name in df.columns and "future_return_1" in df.columns:
                            detection_result = lookahead_bias_detector.detect(
                                factor_values=df[factor_name].dropna(),
                                return_values=df["future_return_1"].dropna(),
                                factor_name=factor_name,
                            )
                        else:
                            detection_result = None
                else:
                    # 单股票模式：直接用序列检测
                    first_stock = codes[0] if codes else list(factor_data.keys())[0]
                    df = factor_data.get(first_stock)
                    if df is not None and factor_name in df.columns and "future_return_1" in df.columns:
                        detection_result = lookahead_bias_detector.detect(
                            factor_values=df[factor_name].dropna(),
                            return_values=df["future_return_1"].dropna(),
                            factor_name=factor_name,
                        )
                    else:
                        detection_result = None

                if detection_result is not None:
                    all_bias_results[factor_name] = {
                        "has_bias": detection_result.has_bias,
                        "risk_level": detection_result.risk_level.value,
                        "risk_score": detection_result.risk_score,
                        "summary": detection_result.summary,
                        "recommendations": detection_result.recommendations,
                        "n_checks": len(detection_result.checks),
                        "n_failed": sum(1 for c in detection_result.checks if not c.passed),
                    }
                    # 记录高风险因子日志
                    if detection_result.risk_level in (BiasRiskLevel.HIGH, BiasRiskLevel.CRITICAL):
                        logger.warning(
                            f"[未来函数检测] ⚠️ 因子 [{factor_name}] "
                            f"风险等级={detection_result.risk_level.value}, "
                            f"评分={detection_result.risk_score:.1f}"
                        )
                else:
                    all_bias_results[factor_name] = {
                        "has_bias": False,
                        "risk_level": "unknown",
                        "risk_score": 0,
                        "summary": f"因子 [{factor_name}] 数据不足，无法检测",
                        "recommendations": [],
                        "n_checks": 0,
                        "n_failed": 0,
                    }

            except Exception as e:
                logger.error(f"[未来函数检测] 因子 [{factor_name}] 检测失败: {e}", exc_info=True)
                all_bias_results[factor_name] = {
                    "has_bias": False,
                    "risk_level": "error",
                    "risk_score": 0,
                    "summary": f"检测异常: {str(e)}",
                    "recommendations": [],
                    "n_checks": 0,
                    "n_failed": 0,
                }

        # 汇总统计
        risk_levels = [r.get("risk_level", "unknown") for r in all_bias_results.values()]
        summary = {
            "per_factor": all_bias_results,
            "overall_risk": max(risk_levels, key=lambda x: ["safe", "low", "medium", "high", "critical", "error", "unknown"].index(x)) if risk_levels else "safe",
            "n_high_risk": sum(1 for r in all_bias_results.values() if r.get("risk_level") in ("high", "critical")),
            "n_total": len(all_bias_results),
        }

        logger.info(
            f"[未来函数检测] 全部完成: {summary['n_total']}个因子, "
            f"高风险={summary['n_high_risk']}, 综合等级={summary['overall_risk']}"
        )

        return summary

    # ==================== P2-2: 加权IC (市值加权/流动性加权) ====================

    def calculate_weighted_ic(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str],
        stock_codes: Optional[List[str]] = None,
        weight_type: str = "market_cap",
    ) -> Dict[str, Any]:
        """
        计算加权IC (JoinQuant/BQ标准扩展)

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_names: 因子名称列表
            stock_codes: 股票代码列表
            weight_type: 权重类型 "market_cap" / "liquidity" / "equal"

        Returns:
            {
                "ic_stats": {factor_key: {IC均值, IR, ...}},
                "weight_type": str,
                "monthly_ic": {...},
                "rolling_ir": {...}
            }
        """
        if len(factor_data) < 2:
            return {"error": "加权IC需要至少2只股票", "ic_stats": {}}

        codes = stock_codes or list(factor_data.keys())

        all_dates = set()
        for df in factor_data.values():
            if "close" in df.columns:
                all_dates.update(df.index)
        all_dates = sorted(all_dates)

        prices = None
        for stock_code in codes:
            df = factor_data[stock_code]
            if "close" in df.columns:
                if prices is None:
                    prices = pd.DataFrame(index=all_dates)
                prices[stock_code] = df["close"]
        if prices is None:
            return {"error": "无法构建pricing数据", "ic_stats": {}}
        prices = prices.dropna(how="all")

        weight_map = {}
        if weight_type == "market_cap":
            for stock_code in codes:
                df = factor_data.get(stock_code)
                if df is not None and "market_cap" in df.columns:
                    mc = df["market_cap"].dropna()
                    if len(mc) > 0:
                        # 使用截面市值（完整Series），而非时间平均市值
                        weight_map[stock_code] = mc
            # 截面市值无需全局归一化，_compute_weighted_ic_mean按日期逐日归一化
        elif weight_type == "liquidity":
            for stock_code in codes:
                df = factor_data.get(stock_code)
                if df is not None and "volume" in df.columns:
                    vol = df["volume"].dropna()
                    if len(vol) > 0:
                        weight_map[stock_code] = vol.mean()
                elif df is not None and "amount" in df.columns:
                    amt = df["amount"].dropna()
                    if len(amt) > 0:
                        weight_map[stock_code] = amt.mean()
            total_liq = sum(weight_map.values())
            if total_liq > 0:
                weight_map = normalize_weights(weight_map)

        all_ic_stats = {}
        all_monthly_ic = {}
        all_rolling_ir = {}

        for factor_name in factor_names:
            factor_values_dict = {}
            for stock_code in codes:
                df = factor_data[stock_code]
                if factor_name in df.columns and "close" in df.columns:
                    series = df[factor_name].dropna()
                    if len(series) > 0:
                        factor_values_dict[stock_code] = series

            if len(factor_values_dict) < 2:
                continue

            try:
                al_result = alphalens_analysis_service.full_analysis(
                    factor_values_dict=factor_values_dict,
                    pricing_df=prices,
                    periods=(1,),
                )
                ic_analysis = al_result.get("ic_analysis", {})
                for ic_type_key in ["spearman_ic", "pearson_ic"]:
                    ic_data = ic_analysis.get(ic_type_key, {})
                    for period_label, period_stats in ic_data.items():
                        if not isinstance(period_stats, dict) or "error" in period_stats:
                            continue
                        ic_series_data = period_stats.get("ic_series", {})
                        dates = ic_series_data.get("dates", [])
                        values = ic_series_data.get("values", [])
                        valid_values = [v for v in values if v is not None]
                        if not valid_values:
                            continue
                        ic_s = pd.Series(
                            [v if v is not None else np.nan for v in values],
                            index=pd.to_datetime(dates) if dates else range(len(values)),
                        ).dropna()

                        ic_type_name = period_stats.get("ic_type", ic_type_key)
                        factor_key = f"{factor_name}_{ic_type_key}_weighted_{weight_type}"

                        weighted_mean = self._compute_weighted_ic_mean(ic_s, factor_values_dict, weight_map, weight_type)
                        std_ic_val = period_stats.get("std_ic", ic_s.std())

                        all_ic_stats[factor_key] = {
                            "IC均值": float(weighted_mean),
                            "IC标准差": float(period_stats.get("std_ic", ic_s.std())),
                            "IR": safe_ir(float(weighted_mean), float(std_ic_val), default=None),
                            "IC>0占比": float((ic_s > 0).mean()),
                            "IC绝对值均值": float(abs(ic_s).mean()),
                            "IC序列": ic_s.to_dict(),
                            "IC类型": f"横截面{ic_type_name}（{weight_type}加权）",
                            "t统计量": float(period_stats.get("t_statistic", 0)),
                            "p值": float(period_stats.get("p_value", 1)),
                            "weight_type": weight_type,
                        }
                        if len(ic_s) > 0:
                            all_monthly_ic[factor_key] = self._calculate_monthly_ic({factor_key: ic_s})[factor_key]
                            all_rolling_ir[factor_key] = self._calculate_rolling_ir({factor_key: ic_s})[factor_key]
            except Exception as e:
                logger.warning(f"加权IC Alphalens失败({factor_name}): {e}")

        return {
            "ic_stats": all_ic_stats,
            "monthly_ic": all_monthly_ic,
            "rolling_ir": all_rolling_ir,
            "weight_type": weight_type,
        }

    @staticmethod
    def _compute_weighted_ic_mean(ic_s: pd.Series, factor_values_dict: dict, weight_map: dict, weight_type: str) -> float:
        """计算加权IC均值（按截面对齐权重，每日归一化）"""
        if not weight_map or weight_type == "equal":
            return float(ic_s.mean())

        date_weights = {}
        for date in ic_s.index:
            w_sum = 0.0
            count = 0
            for stock_code, w in weight_map.items():
                fv = factor_values_dict.get(stock_code)
                if fv is not None and date in fv.index:
                    if isinstance(w, pd.Series):
                        if date in w.index:
                            w_sum += abs(w.loc[date])
                            count += 1
                    elif isinstance(w, (int, float)):
                        w_sum += abs(w)
                        count += 1
            # 按日期归一化：使用总权重（而非平均权重），市值大的日期IC更可靠
            date_weights[date] = w_sum if count > 0 else 1.0

        weights_series = pd.Series(date_weights)
        total_weight = weights_series.sum()
        if total_weight == 0:
            return float(ic_s.mean())
        weights_normalized = normalize_weights(weights_series)

        aligned_weights = weights_normalized.reindex(ic_s.index, fill_value=1.0 / len(ic_s))

        weighted_vals = ic_s * aligned_weights
        return float(weighted_vals.sum())

    def calculate_shap(
        self, factor_data: Dict[str, pd.DataFrame], factor_names: List[str]
    ) -> Dict[str, Any]:
        """
        计算SHAP值分析

        Args:
            factor_data: 因子数据
            factor_names: 因子名称列表

        Returns:
            SHAP分析结果
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP library not installed"}

        logger.debug(f"[SHAP] Starting SHAP analysis")
        logger.debug(f"[SHAP] factor_names: {factor_names}")
        logger.debug(f"[SHAP] Number of stocks in factor_data: {len(factor_data)}")

        # 准备训练数据
        X_list = []
        y_list = []

        for stock_code, df in factor_data.items():
            df = df.copy()
            logger.debug(f"[SHAP] Processing stock: {stock_code}")
            logger.debug(f"[SHAP]   DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"[SHAP]   DataFrame shape: {df.shape}")

            if "future_return_5" not in df.columns:
                df["future_return_5"] = df["close"].pct_change(5).shift(-5)

            # 提取特征列
            feature_cols = [col for col in factor_names if col in df.columns]
            logger.debug(f"[SHAP]   Available feature_cols: {feature_cols}")

            if not feature_cols:
                logger.debug(f"[SHAP]   No feature columns found, skipping")
                continue

            X = df[feature_cols].dropna()
            y = df.loc[X.index, "future_return_5"]

            logger.debug(f"[SHAP]   X shape before NaN removal: {X.shape}")
            logger.debug(f"[SHAP]   X NaN count: {X.isna().sum().sum()}")

            # 移除NaN
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            logger.debug(f"[SHAP]   X_valid shape: {X_valid.shape}")

            if len(X_valid) > 0:
                X_list.append(X_valid)
                y_list.append(y_valid)
                logger.debug(f"[SHAP]   Added {len(X_valid)} valid samples")
            else:
                logger.debug(f"[SHAP]   No valid samples after NaN removal")

        logger.debug(f"[SHAP] Total X_list length: {len(X_list)}")

        if not X_list:
            logger.error("[SHAP] No valid data for SHAP analysis")
            return {"error": "No valid data for SHAP analysis"}

        # 合并所有数据（保留时间索引，避免未来信息泄漏）
        X_combined = pd.concat(X_list)
        y_combined = pd.concat(y_list)

        # 按时间索引排序，确保时间顺序正确，避免未来数据泄漏
        X_combined = X_combined.sort_index()
        y_combined = y_combined.sort_index()

        logger.debug(f"[SHAP] X_combined shape: {X_combined.shape}")
        logger.debug(f"[SHAP] X_combined columns: {X_combined.columns.tolist()}")

        # 按日期分割训练集和测试集（避免同日数据分属train/test导致泄漏）
        all_dates = sorted(X_combined.index.unique())
        if len(all_dates) < 2:
            logger.warning("[SHAP] 日期数不足2，无法分割训练/测试集")
            return {"error": "日期数不足，无法进行SHAP分析"}
        split_date = all_dates[int(len(all_dates) * 0.8)]
        X_train_raw = X_combined[X_combined.index < split_date]
        X_test_raw = X_combined[X_combined.index >= split_date]
        y_train = y_combined[y_combined.index < split_date]
        y_test = y_combined[y_combined.index >= split_date]

        # 标准化特征（仅在训练集上fit，避免数据泄漏）
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_combined.columns, index=X_train_raw.index)
        X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_combined.columns, index=X_test_raw.index)

        # 训练XGBoost模型
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=42,
        )

        logger.debug(f"[SHAP] Training with {len(X_train)} samples, testing with {len(X_test)} samples")

        model.fit(X_train, y_train)

        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 特征重要性
        feature_importance = pd.DataFrame({
            "feature": X_test.columns,
            "importance": np.abs(shap_values).mean(axis=0),
        }).sort_values("importance", ascending=False)

        logger.debug(f"[SHAP] SHAP analysis completed successfully")

        return {
            "feature_importance": feature_importance.to_dict("records"),
            "shap_values": shap_values.tolist(),
            "feature_names": X_test.columns.tolist(),
            "model_score": model.score(X_test, y_test),
        }

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        生成分析报告（Markdown格式）

        Args:
            analysis_results: 分析结果字典

        Returns:
            Markdown格式的报告文本
        """
        metadata = analysis_results["metadata"]
        ic_ir = analysis_results.get("ic_ir", {})
        shap_data = analysis_results.get("shap", {})
        bias_data = analysis_results.get("lookahead_bias", {})

        report = f"""# 因子分析报告

## 分析参数

- **股票代码**: {', '.join(metadata['stock_codes'])}
- **因子列表**: {', '.join(metadata['factor_names'])}
- **时间区间**: {metadata['start_date']} 至 {metadata['end_date']}
- **滚动窗口**: {metadata['rolling_window']}天
- **分析时间**: {metadata['analysis_time']}

---

## IC/IR 统计分析

"""

        if "ic_stats" in ic_ir:
            ic_stats = ic_ir["ic_stats"]
            report += "### 因子IC统计\n\n"
            report += "| 因子名称 | IC均值 | IC标准差 | IR | IC>0占比 | IC绝对值均值 |\n"
            report += "|---------|--------|----------|-----|---------|-------------|\n"

            for factor_name, stats in ic_stats.items():
                report += f"| {factor_name} | {stats['IC均值']:.4f} | {stats['IC标准差']:.4f} | {stats['IR']:.4f} | {stats['IC>0占比']:.2%} | {stats['IC绝对值均值']:.4f} |\n"

            report += "\n"

        if shap_data and "feature_importance" in shap_data:
            report += "---\n\n## SHAP 特征重要性分析\n\n"
            report += "### 全局特征重要性排序\n\n"
            report += "| 排名 | 特征名称 | 重要性 |\n"
            report += "|------|---------|--------|\n"

            for i, feat in enumerate(shap_data["feature_importance"], 1):
                report += f"| {i} | {feat['feature']} | {feat['importance']:.6f} |\n"

            report += f"\n**模型R²得分**: {shap_data.get('model_score', 0):.4f}\n\n"

        alphalens_data = analysis_results.get("alphalens", {})
        if alphalens_data:
            report += "---\n\n## Alphalens 因子分析\n\n"

            for factor_name, factor_result in alphalens_data.items():
                if isinstance(factor_result, dict) and "error" in factor_result:
                    report += f"### {factor_name}\n\n分析失败: {factor_result['error']}\n\n"
                    continue

                report += f"### {factor_name}\n\n"

                ic_analysis = factor_result.get("ic_analysis", {})
                if ic_analysis:
                    report += "#### IC分析\n\n"
                    for ic_type_key in ["spearman_ic", "pearson_ic"]:
                        ic_data = ic_analysis.get(ic_type_key, {})
                        if ic_data and "error" not in ic_data:
                            report += f"**{ic_type_key.replace('_', ' ').title()}**\n\n"
                            report += "| 周期 | IC均值 | IC标准差 | IR | IC>0占比 | t统计量 | p值 |\n"
                            report += "|------|--------|----------|-----|---------|---------|----|\n"
                            for period_label, period_stats in ic_data.items():
                                if isinstance(period_stats, dict) and "error" not in period_stats:
                                    report += (
                                        f"| {period_label} | "
                                        f"{period_stats.get('mean_ic', 0):.4f} | "
                                        f"{period_stats.get('std_ic', 0):.4f} | "
                                        f"{period_stats.get('ir', 0):.4f} | "
                                        f"{period_stats.get('ic_positive_ratio', 0):.2%} | "
                                        f"{period_stats.get('t_statistic', 0):.4f} | "
                                        f"{period_stats.get('p_value', 1):.4f} |\n"
                                    )
                            report += "\n"

                returns_analysis = factor_result.get("returns_analysis", {})
                if returns_analysis and "error" not in returns_analysis:
                    report += "#### 收益分析\n\n"
                    quantile_returns = returns_analysis.get("quantile_returns", {})
                    if quantile_returns:
                        report += "**分位数收益**\n\n"
                        for period_label, q_data in quantile_returns.items():
                            report += f"- {period_label}: "
                            q_strs = [f"Q{q}={v:.6f}" for q, v in q_data.items() if v is not None]
                            report += ", ".join(q_strs) + "\n"
                        report += "\n"

                    spread = returns_analysis.get("spread", {})
                    if spread and "error" not in spread:
                        report += "**Top-Bottom Spread**\n\n"
                        for period_label, spread_data in spread.items():
                            if isinstance(spread_data, dict) and "spread" in spread_data:
                                report += f"- {period_label}: {spread_data['spread']:.6f}\n"
                        report += "\n"

                    alpha_beta = returns_analysis.get("alpha_beta", {})
                    if alpha_beta:
                        report += "**Alpha/Beta**\n\n"
                        for period_label, ab_data in alpha_beta.items():
                            if isinstance(ab_data, dict):
                                alpha_val = ab_data.get("Ann. alpha", ab_data.get("alpha", "N/A"))
                                beta_val = ab_data.get("beta", "N/A")
                                report += f"- {period_label}: alpha={alpha_val}, beta={beta_val}\n"
                        report += "\n"

                turnover_analysis = factor_result.get("turnover_analysis", {})
                if turnover_analysis and "error" not in turnover_analysis:
                    report += "#### 换手率分析\n\n"
                    autocorr = turnover_analysis.get("factor_autocorrelation", {})
                    if autocorr:
                        for period_label, ac_data in autocorr.items():
                            if isinstance(ac_data, dict):
                                report += f"- {period_label}: 平均自相关={ac_data.get('mean_autocorrelation', 0):.4f}\n"
                        report += "\n"

        # 未来函数检测报告
        if bias_data and "per_factor" in bias_data:
            per_factor = bias_data.get("per_factor", {})
            overall_risk = bias_data.get("overall_risk", "safe")
            n_high_risk = bias_data.get("n_high_risk", 0)

            risk_emoji = {
                "safe": "✅", "low": "⚠️", "medium": "🔶",
                "high": "🔴", "critical": "💀", "error": "❓", "unknown": "➖",
            }
            report += "---\n\n## 未来函数（Look-ahead Bias）检测\n\n"
            report += f"**综合风险等级**: {risk_emoji.get(overall_risk, '')} `{overall_risk.upper()}`"
            if n_high_risk > 0:
                report += f" （{n_high_risk} 个高风险因子）"
            report += "\n\n"

            if per_factor:
                report += "| 因子名称 | 风险等级 | 评分 | 检测项通过/总数 | 摘要 |\n"
                report += "|---------|---------|------|-----------------|------|\n"
                for factor_name, bresult in per_factor.items():
                    level = bresult.get("risk_level", "unknown")
                    score = bresult.get("risk_score", 0)
                    n_checks = bresult.get("n_checks", 0)
                    n_failed = bresult.get("n_failed", 0)
                    summary = bresult.get("summary", "")[:60]
                    report += (
                        f"| {factor_name} | "
                        f"{risk_emoji.get(level, '')} {level} | "
                        f"{score:.0f} | "
                        f"{n_checks - n_failed}/{n_checks} | "
                        f"{summary} |\n"
                    )

            # 显示改进建议
            all_recommendations = set()
            for bresult in per_factor.values():
                for rec in bresult.get("recommendations", []):
                    all_recommendations.add(rec)
            if all_recommendations:
                report += "\n**改进建议**:\n"
                for rec in sorted(all_recommendations)[:5]:
                    report += f"- {rec}\n"

            report += "\n"

        report += "---\n\n*报告由 FactorFlow 自动生成*"

        return report

    def export_report(self, analysis_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        导出分析报告到文件

        Args:
            analysis_results: 分析结果
            output_path: 输出路径，默认保存到reports目录

        Returns:
            保存的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.REPORTS_DIR / f"factor_analysis_report_{timestamp}.md"

        report = self.generate_report(analysis_results)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        return str(output_path)


# 全局分析服务实例
analysis_service = AnalysisService()

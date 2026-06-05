"""
智能预处理参数检测器

根据数据特征自动选择最优的去极值/中性化/标准化参数
支持：
- 市场板块识别（主板/创业板/科创板/北交所）
- 波动性自适应
- 样本量感知
- 行业分布分析
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class MarketBoard(str, Enum):
    """市场板块枚举"""
    MAIN = "main"              # 主板 (60xxxx, 00xxxx)
    CHINEXT = "chinext"        # 创业板 (30xxxx)
    STAR = "star"              # 科创板 (68xxxx)
    BEIJING = "beijing"        # 北交所 (8xxxx, 4xxxx)
    MIXED = "mixed"            # 混合


@dataclass
class DataCharacteristics:
    """数据特征描述"""
    n_stocks: int
    n_dates: int
    total_samples: int
    
    market_board: MarketBoard
    avg_market_cap: float
    market_cap_std: float
    
    factor_volatility: float      # 因子横截面波动率（日均标准差）
    factor_skewness: float         # 因子偏度
    factor_kurtosis: float         # 因子峰度
    
    n_industries: int
    min_industry_size: int        # 最小行业样本量
    max_industry_size: int        # 最大行业样本量
    
    has_extreme_outliers: bool    # 是否存在极端异常值
    outlier_ratio: float          # 异常值比例
    
    is_fat_tailed: bool           # 是否肥尾分布
    time_varying_volatility: bool # 波动率是否时变


@dataclass
class PreprocessingRecommendation:
    """预处理参数推荐"""
    config_dict: Dict
    confidence: float              # 推荐置信度 (0-1)
    reasoning: str                # 推荐理由
    data_characteristics: DataCharacteristics
    warnings: List[str]           # 警告信息


class SmartPreprocessingDetector:
    """
    智能预处理参数检测器
    
    算法流程：
    1. 识别市场板块 → 确定基础参数范围
    2. 分析因子分布 → 选择去极值方法
    3. 检测行业结构 → 决定中性化策略
    4. 评估样本充足性 → 调整最小样本要求
    5. 输出推荐配置 + 置信度
    """

    def __init__(self):
        self._board_patterns = {
            MarketBoard.MAIN: {
                "code_pattern": r"^(60\d{4}|0\d{5})",
                "price_limit": 0.10,  # ±10%
                "volatility_factor": 1.0,
                "default_n_sigma": 3.0,
                "description": "主板市场",
            },
            MarketBoard.CHINEXT: {
                "code_pattern": r"^3\d{5}",
                "price_limit": 0.20,  # ±20%
                "volatility_factor": 1.5,  # 创业板波动大50%
                "default_n_sigma": 2.8,  # 更严格
                "description": "创业板市场",
            },
            MarketBoard.STAR: {
                "code_pattern": r"^68\d{4}",
                "price_limit": 0.20,
                "volatility_factor": 1.6,  # 科创板波动更大
                "default_n_sigma": 2.7,
                "description": "科创板市场",
            },
            MarketBoard.BEIJING: {
                "code_pattern": r"^(8\d{5}|4\d{5})",
                "price_limit": 0.30,  # ±30%
                "volatility_factor": 2.0,  # 北交所波动最大
                "default_n_sigma": 2.5,
                "description": "北交所市场",
            },
        }

    def analyze_data(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str],
        market_cap_column: str = "market_cap",
        industry_column: str = "industry",
    ) -> DataCharacteristics:
        """
        分析数据特征
        
        Args:
            factor_data: {stock_code: DataFrame} 格式的因子数据
            factor_names: 因子名称列表
            market_cap_column: 市值列名
            industry_column: 行业列名
            
        Returns:
            数据特征描述对象
        """
        # 合并所有股票数据用于全局分析
        all_dfs = []
        for stock_code, df in factor_data.items():
            if len(df) > 0:
                df_copy = df.copy()
                df_copy["stock_code"] = stock_code
                all_dfs.append(df_copy)

        if not all_dfs:
            return self._empty_characteristics()

        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # 1️⃣ 基础统计
        n_stocks = len(factor_data)
        n_dates = merged_df["date"].nunique() if "date" in merged_df.columns else 0
        total_samples = len(merged_df)

        # 2️⃣ 市场板块识别
        market_board = self._detect_market_board(list(factor_data.keys()))

        # 3️⃣ 市值统计
        avg_mc = 0
        mc_std = 0
        if market_cap_column in merged_df.columns:
            valid_mc = merged_df[market_cap_column].dropna()
            if len(valid_mc) > 0:
                avg_mc = valid_mc.mean()
                mc_std = valid_mc.std()

        # 4️⃣ 因子分布特征（使用第一个因子的统计）
        factor_vol = 0
        skewness = 0
        kurtosis = 0
        has_outliers = False
        outlier_ratio = 0
        is_fat_tail = False
        
        if factor_names and factor_names[0] in merged_df.columns:
            factor_col = merged_df[factor_names[0]].dropna()
            
            if len(factor_col) > 10:
                # 横截面波动率（每日标准差的均值）
                if "date" in merged_df.columns:
                    daily_std = factor_col.groupby(merged_df["date"]).std()
                    factor_vol = daily_std.mean()
                else:
                    factor_vol = factor_col.std()
                
                skewness = float(factor_col.skew())
                kurtosis = float(factor_col.kurtosis())
                
                # 异常值检测（MAD法）
                median = factor_col.median()
                mad = 1.4826 * np.median(np.abs(factor_col - median))
                if mad > 0:
                    outliers = np.abs(factor_col - median) > 3 * mad
                    has_outliers = outliers.any()
                    outlier_ratio = outliers.sum() / len(factor_col)
                
                # 肥尾检测（pandas kurtosis()返回超额峰度，正态分布=0，>0表示肥尾）
                is_fat_tail = kurtosis > 0

        # 5️⃣ 行业结构分析
        n_industries = 0
        min_ind_size = 0
        max_ind_size = 0
        
        if industry_column in merged_df.columns:
            industry_counts = merged_df[industry_column].value_counts()
            n_industries = len(industry_counts)
            if len(industry_counts) > 0:
                min_ind_size = industry_counts.min()
                max_ind_size = industry_counts.max()

        # 6️⃣ 时变波动性检测
        time_varying_vol = False
        if "date" in merged_df.columns and factor_names:
            first_factor = factor_names[0]
            if first_factor in merged_df.columns:
                rolling_vol = (
                    merged_df[first_factor]
                    .groupby(merged_df["date"])
                    .std()
                    .rolling(window=min(20, n_dates // 5))
                    .std()
                )
                if len(rolling_vol.dropna()) > 10:
                    vol_of_vol = rolling_vol.std() / (rolling_vol.mean() + 1e-10)
                    time_varying_vol = vol_of_vol > 0.3  # 波动率的变异系数>30%

        return DataCharacteristics(
            n_stocks=n_stocks,
            n_dates=n_dates,
            total_samples=total_samples,
            market_board=market_board,
            avg_market_cap=avg_mc,
            market_cap_std=mc_std,
            factor_volatility=factor_vol,
            factor_skewness=skewness,
            factor_kurtosis=kurtosis,
            n_industries=n_industries,
            min_industry_size=min_ind_size,
            max_industry_size=max_ind_size,
            has_extreme_outliers=has_outliers,
            outlier_ratio=outlier_ratio,
            is_fat_tailed=is_fat_tail,
            time_varying_volatility=time_varying_vol,
        )

    def _detect_market_board(self, stock_codes: List[str]) -> MarketBoard:
        """识别市场板块"""
        board_counts = {board: 0 for board in MarketBoard if board != MarketBoard.MIXED}

        for code in stock_codes:
            pure_code = code.replace(".SZ", "").replace(".SH", "")
            
            for board, pattern_info in self._board_patterns.items():
                if re.match(pattern_info["code_pattern"], pure_code):
                    board_counts[board] += 1
                    break

        # 找出占比最大的板块
        if sum(board_counts.values()) == 0:
            return MarketBoard.MIXED

        dominant_board = max(board_counts.items(), key=lambda x: x[1])
        
        # 如果单一板块占比>80%，认为是单一板块；否则是混合
        if dominant_board[1] / len(stock_codes) > 0.8:
            return dominant_board[0]
        else:
            return MarketBoard.MIXED

    def recommend_config(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str],
        user_preference: Optional[str] = None,  # "conservative"/"aggressive"/None
    ) -> PreprocessingRecommendation:
        """
        推荐最优预处理配置
        
        Args:
            factor_data: 因子数据字典
            factor_names: 因子名称列表
            user_preference: 用户偏好 ("conservative"保守/"aggressive"激进/None自动)
            
        Returns:
            推荐配置对象
        """
        # 分析数据特征
        chars = self.analyze_data(factor_data, factor_names)
        
        # 基于规则生成推荐配置
        config, confidence, reasoning, warnings = self._generate_recommendation(chars, user_preference)
        
        return PreprocessingRecommendation(
            config_dict=config,
            confidence=confidence,
            reasoning=reasoning,
            data_characteristics=chars,
            warnings=warnings,
        )

    def _generate_recommendation(
        self,
        chars: DataCharacteristics,
        preference: Optional[str],
    ) -> Tuple[Dict, float, str, List[str]]:
        """生成推荐配置的核心逻辑"""
        
        config = {}
        warnings = []
        confidence_scores = []  # 各项决策的置信度
        reasoning_parts = []

        # ==================== 1️⃣ 去极值方法选择 ====================
        if chars.is_fat_tailed or chars.has_extreme_outliers:
            # 肥尾或极端值 → 使用MAD法（更稳健）
            config["winsorize_method"] = "mad"
            confidence_scores.append(0.9)
            reasoning_parts.append("检测到肥尾分布/极端值，选用稳健的MAD法")
        elif chars.factor_skewness < -1 or chars.factor_skewness > 1:
            # 显著偏态 → 使用百分位法
            config["winsorize_method"] = "percentile"
            config["winsorize_limits"] = (0.02, 0.98)
            confidence_scores.append(0.85)
            reasoning_parts.append(f"检测到显著偏态(偏度={chars.factor_skewness:.2f})，选用百分位法")
        else:
            # 近似正态 → 使用3σ法或MAD
            config["winsorize_method"] = "mad"
            confidence_scores.append(0.8)
            reasoning_parts.append("分布接近正态，选用MAD法")

        # ==================== 2️⃣ 去极值强度（n_sigma）====================
        base_board_config = self._board_patterns.get(chars.market_board, {})
        volatility_factor = base_board_config.get("volatility_factor", 1.0)
        base_n_sigma = base_board_config.get("default_n_sigma", 3.0)

        # 根据波动性和用户偏好调整
        if preference == "conservative":
            n_sigma = base_n_sigma * 0.9  # 更严格
            confidence_scores.append(0.95)
            reasoning_parts.append("用户偏好保守策略，收紧去极值边界")
        elif preference == "aggressive":
            n_sigma = base_n_sigma * 1.15  # 更宽松
            confidence_scores.append(0.85)
            reasoning_parts.append("用户偏好激进策略，放宽去极值边界")
        else:
            # 自适应调整
            if chars.factor_volatility > 15:  # 高波动
                n_sigma = base_n_sigma * 0.92
                reasoning_parts.append(f"高波动环境({chars.factor_volatility:.1f})，适度收紧")
            elif chars.outlier_ratio > 0.05:  # 异常值多
                n_sigma = base_n_sigma * 0.95
                reasoning_parts.append(f"异常值比例较高({chars.outlier_ratio:.1%})，适度收紧")
            else:
                n_sigma = base_n_sigma
                reasoning_parts.append("使用该板块的标准参数")
            
            confidence_scores.append(0.88)

        config["winsorize_n_sigma"] = round(n_sigma, 2)

        # ==================== 3️⃣ 中性化策略 ====================
        # 市值中性化
        if chars.avg_market_cap > 0 and chars.market_cap_std / (chars.avg_market_cap + 1e-10) > 0.5:
            # 市值差异大 → 必须中性化
            config["enable_market_cap_neutralization"] = True
            confidence_scores.append(0.95)
            reasoning_parts.append(f"市值差异显著(CV={chars.market_cap_std/chars.avg_market_cap:.2f})，启用市值中性化")
        else:
            config["enable_market_cap_neutralization"] = True  # 默认开启
            confidence_scores.append(0.75)
            reasoning_parts.append("默认启用市值中性化（可手动关闭）")

        # 行业中性化
        if chars.n_industries >= 3 and chars.min_industry_size >= 10:
            # 行业数足够且每个行业样本充足
            config["enable_industry_neutralization"] = True
            confidence_scores.append(0.9)
            reasoning_parts.append(
                f"行业结构良好({chars.n_industries}个行业，最少{chars.min_industry_size}只/行业)，"
                f"启用行业中性化"
            )
        elif chars.n_industries >= 2 and chars.min_industry_size >= 5:
            # 行业数尚可但部分行业较小
            config["enable_industry_neutralization"] = True
            confidence_scores.append(0.7)
            reasoning_parts.append(
                f"部分行业样本较少(最少{chars.min_industry_size}只)，"
                f"行业中性化效果可能受限"
            )
            warnings.append(f"⚠️ 存在仅{chars.min_industry_size}只股票的小行业，中性化效果可能不稳定")
        else:
            # 行业数太少或样本严重不足
            config["enable_industry_neutralization"] = False
            confidence_scores.append(0.6)
            reasoning_parts.append("行业结构不足，跳过行业中性化")
            warnings.append("❌ 行业分类不足或样本量太少，不建议进行行业中性化")

        # ==================== 4️⃣ 标准化方法 ====================
        if chars.has_extreme_outliers or chars.is_fat_tailed:
            # 有异常值 → 使用Rank标准化（更抗异常）
            config["standardize_method"] = "rank"
            confidence_scores.append(0.85)
            reasoning_parts.append("存在异常值/肥尾，选用Rank标准化（更稳健）")
        else:
            config["standardize_method"] = "zscore"
            confidence_scores.append(0.9)
            reasoning_parts.append("分布正常，选用Z-score标准化")

        # ==================== 5️⃣ 其他参数 ====================
        # 缺失值处理
        if chars.outlier_ratio > 0.03:
            config["handle_missing"] = "fill_median"
            reasoning_parts.append("异常值较多，使用中位数填充缺失值")
        else:
            config["handle_missing"] = "fill_zero"
        
        # 最小样本量
        if chars.n_industries > 0:
            config["min_samples"] = max(10, min(chars.min_industry_size // 2, 20))
        else:
            config["min_samples"] = 15

        # 横截面模式
        config["cross_sectional"] = chars.n_stocks > 1

        # ==================== 综合评估 ====================
        overall_confidence = np.mean(confidence_scores)
        full_reasoning = "；".join(reasoning_parts)

        return config, overall_confidence, full_reasoning, warnings

    def _empty_characteristics(self) -> DataCharacteristics:
        return DataCharacteristics(
            n_stocks=0, n_dates=0, total_samples=0,
            market_board=MarketBoard.MIXED,
            avg_market_cap=0, market_cap_std=0,
            factor_volatility=0, factor_skewness=0, factor_kurtosis=0,
            n_industries=0, min_industry_size=0, max_industry_size=0,
            has_extreme_outliers=False, outlier_ratio=0,
            is_fat_tailed=False, time_varying_volatility=False,
        )

    def get_config_summary(self, recommendation: PreprocessingRecommendation) -> str:
        """生成人类可读的配置摘要"""
        chars = recommendation.data_characteristics
        config = recommendation.config_dict
        
        lines = [
            f"# 🤖 智能预处理参数推荐报告",
            f"",
            f"## 📊 数据概况",
            f"- **股票数量**: {chars.n_stocks} 只",
            f"- **时间跨度**: {chars.n_dates} 个交易日",
            f"- **总样本量**: {chars.total_samples:,}",
            f"- **市场板块**: {self._get_board_name(chars.market_board)}",
            f"- **平均市值**: {chars.avg_market_cap/1e8:.1f} 亿",
            f"- **因子波动率**: {chars.factor_volatility:.4f}",
            f"",
            f"## 🎯 推荐配置 (置信度: {recommendation.confidence*100:.0f}%)",
            f"",
            f"| 参数 | 推荐值 | 说明 |",
            f"|------|--------|------|",
            f"| 去极值方法 | **{config.get('winsorize_method', 'mad').upper()}** | {'稳健抗异常值' if config.get('winsorize_method') == 'mad' else '适应非正态'} |",
            f"| 去极值强度 | **{config.get('winsorize_n_sigma', 3.0):.2f}σ** | 基于{self._get_board_name(chars.market_board)}特性调整 |",
            f"| 市值中性化 | **{'✅ 启用' if config.get('enable_market_cap_neutralization') else '❌ 关闭'}** | {'市值差异大' if config.get('enable_market_cap_neutralization') else '-'} |",
            f"| 行业中性化 | **{'✅ 启用' if config.get('enable_industry_neutralization') else '❌ 关闭'}** | {f'{chars.n_industries}个行业' if config.get('enable_industry_neutralization') else '样本不足'} |",
            f"| 标准化方法 | **{config.get('standardize_method', 'zscore').upper()}** | {'抗异常值' if config.get('standardize_method') == 'rank' else '保持线性'} |",
            f"",
            f"## 💡 推荐理由",
            f"{recommendation.reasoning}",
        ]
        
        if recommendation.warnings:
            lines.extend([
                f"",
                f"## ⚠️ 注意事项",
            ])
            for warning in recommendation.warnings:
                lines.append(f"- {warning}")
        
        return "\n".join(lines)

    def _get_board_name(self, board: MarketBoard) -> str:
        names = {
            MarketBoard.MAIN: "主板",
            MarketBoard.CHINEXT: "创业板",
            MarketBoard.STAR: "科创板",
            MarketBoard.BEIJING: "北交所",
            MarketBoard.MIXED: "混合板块",
        }
        return names.get(board, "未知")


# 全局实例
smart_detector = SmartPreprocessingDetector()

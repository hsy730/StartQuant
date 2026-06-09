"""
因子相关性分析服务（精简版）
- 核心自研：频率对齐、横截面/时序相关性、滚动稳定性、显著性检验、VIF检查
- 开源复用：Alphalens（单因子分析）、scipy/sklearn/statsmodels（基础统计）
- 可选增强：Phik（混合类型，按需安装）

设计原则：
1. 零强制新依赖（仅alphalens-reloaded）
2. pandas/numpy/scipy 足够应对<1000因子的场景
3. 自建轻量级解读器（无需Corrpy）
4. Phik作为可选插件
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats as scipy_stats

from backend.utils.safe_math import safe_divide

logger = logging.getLogger(__name__)


try:
    import phik
    PHIK_AVAILABLE = True
except ImportError:
    PHIK_AVAILABLE = False


class FactorCorrelationService:
    """
    因子相关性分析服务
    
    特点：
    - 务实：只解决真实痛点，不过度工程化
    - 轻量：核心代码~400行，零冗余依赖
    - 正确：严格遵循量化研究最佳实践
    """
    
    def __init__(self):
        self.mode = "standard"
        logger.info(f"因子相关性服务初始化 (Alphalens: ✅, Phik: {'✅' if PHIK_AVAILABLE else '❌ (可选)'})")
    
    def analyze(
        self,
        factor_panel: pd.DataFrame,
        factor_cols: List[str],
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        完整的相关性分析
        
        Args:
            factor_panel: MultiIndex DataFrame (date, asset) × factors
            factor_cols: 要分析的因子列名列表
            config: 配置参数（可选）
        
        Returns:
            完整的分析结果字典
        """
        # 规则5：禁止就地修改传入的DataFrame，必须先copy
        factor_panel = factor_panel.copy()

        config = config or self._default_config()
        
        result = {
            'metadata': {
                'n_factors': len(factor_cols),
                'n_observations': len(factor_panel),
                'mode': self.mode,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'data_quality': self._check_data_quality(factor_panel, factor_cols),
            'preprocessing': {},
            'cross_sectional': {},
            'time_series': {},
            'rolling_stability': {},
            'significance': {},
            'vif_analysis': {},
            'interpretation': {},  # 自建解读器替代Corrpy
            'warnings': [],
            'recommendations': []
        }
        
        # Step 1: 数据预处理
        cleaned_df, prep_stats = self._preprocess(factor_panel, factor_cols, config)
        result['preprocessing'] = prep_stats
        
        # Step 2: 横截面相关性（每天→时间平均）- 核心创新
        result['cross_sectional'] = self._cross_sectional_corr(cleaned_df, factor_cols)
        
        # Step 3: 时间序列相关性（基于收益率）
        result['time_series'] = self._time_series_corr(cleaned_df, factor_cols)
        
        # Step 4: 滚动稳定性（可选，需要足够数据）
        if len(cleaned_df.index.get_level_values('date').unique()) > config['rolling_window'] * 2:
            result['rolling_stability'] = self._rolling_stability(cleaned_df, factor_cols, config)
        
        # Step 5: 显著性检验
        result['significance'] = self._significance_tests(result['cross_sectional'], result['time_series'])
        
        # Step 6: VIF多重共线性
        result['vif_analysis'] = self._vif_analysis(cleaned_df, factor_cols)
        
        # Step 7: 智能解读（自建规则引擎，无需Corrpy）
        result['interpretation'] = self._interpret_results(result)
        
        # Step 8: 生成建议
        result['warnings'], result['recommendations'] = self._generate_insights(result)
        
        return result
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'rolling_window': 120,
            'rolling_step': 20,
            'use_knn': True,
            'knn_neighbors': 5,
            'winsorize_method': 'mad',
            'n_sigma': 3.0,
            'max_missing_ratio': 0.3
        }
    
    def _preprocess(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        config: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        数据预处理（频率对齐 + 缺失值 + 极值）
        自研创新点：自动检测+验证
        """
        df = df.copy()
        stats = {'original_shape': df.shape}
        
        # 1. 频率检测和对齐
        freq_info = self._detect_frequency(df[factor_cols])
        if len(set(freq_info.values())) > 1:
            df = self._align_frequency(df, freq_info)
            stats['frequency_aligned'] = True
            stats['frequency_info'] = freq_info
        
        # 2. 缺失值处理
        missing = df[factor_cols].isna().mean()
        stats['missing_ratio'] = missing.to_dict()
        
        if missing.max() > 0:
            high_missing = missing[missing > config.get('max_missing_ratio', 0.3)]
            if len(high_missing) > 0:
                stats['warning'] = f"高缺失率因子: {list(high_missing.index)}"
            
            fill_cols = [c for c in factor_cols if missing[c] <= config.get('max_missing_ratio', 0.3)]
            
            if fill_cols and config.get('use_knn'):
                from sklearn.impute import KNNImputer
                
                orig_stats = df[fill_cols].describe()
                imputer = KNNImputer(n_neighbors=config.get('knn_neighbors', 5))
                df[fill_cols] = pd.DataFrame(
                    imputer.fit_transform(df[fill_cols]),
                    index=df.index,
                    columns=fill_cols
                )
                
                new_stats = df[fill_cols].describe()
                
                # 效果验证（关键！）
                validation = {}
                for col in fill_cols:
                    mean_chg = abs(safe_divide(
                        new_stats.loc['mean', col] - orig_stats.loc['mean', col],
                        orig_stats.loc['std', col],
                        default=0.0
                    ) * 100)
                    std_chg = abs(safe_divide(
                        new_stats.loc['std', col] - orig_stats.loc['std', col],
                        orig_stats.loc['std', col],
                        default=0.0
                    ) * 100)
                    validation[col] = {
                        'mean_change_pct': round(mean_chg, 2),
                        'std_change_pct': round(std_chg, 2),
                        'distortion_warning': mean_chg > 10 or std_chg > 15
                    }
                
                stats['imputation'] = {'method': 'KNN', 'validation': validation}
            else:
                df[fill_cols] = df[fill_cols].fillna(df[fill_cols].mean())
                stats['imputation'] = {'method': 'mean'}
        
        # 3. MAD法去极值（金融数据最佳实践）
        n_sigma = config.get('n_sigma', 3.0)
        total_clipped = 0
        
        for col in factor_cols:
            median = df[col].median()
            mad = (df[col] - median).abs().median() * 1.4826
            lower, upper = median - n_sigma * mad, median + n_sigma * mad
            n_clip = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower, upper)
            total_clipped += int(n_clip)
        
        stats['winsorization'] = {'method': 'MAD', 'n_sigma': n_sigma, 'clipped': total_clipped}
        stats['final_shape'] = df.shape
        
        return df, stats
    
    def _detect_frequency(self, df: pd.DataFrame) -> Dict[str, str]:
        """检测数据频率"""
        dates = df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index
        median_diff = pd.to_datetime(dates).diff().dropna().median()
        
        if median_diff <= pd.Timedelta(days=1):
            return {col: 'daily' for col in df.columns}
        elif median_diff <= pd.Timedelta(days=7):
            return {col: 'weekly' for col in df.columns}
        elif median_diff <= pd.Timedelta(days=31):
            return {col: 'monthly' for col in df.columns}
        else:
            return {col: 'quarterly' for col in df.columns}
    
    def _align_frequency(self, df, freq_info):
        """对齐到最低频率"""
        lowest = max(freq_info.values(), key=['daily', 'weekly', 'monthly', 'quarterly'].index)
        
        if not isinstance(df.index, pd.MultiIndex):
            return df
        
        dates = df.index.get_level_values(0)
        
        if lowest == 'monthly':
            month_ends = dates.groupby([dates.year, dates.month]).last()
            df = df.loc[df.index.isin(month_ends, level=0)]
        elif lowest == 'weekly':
            iso_cal = dates.isocalendar()
            week_ends = dates.groupby([iso_cal['year'], iso_cal['week']]).last()
            df = df.loc[df.index.isin(week_ends, level=0)]
        elif lowest == 'quarterly':
            quarter_ends = dates.groupby([dates.year, dates.quarter]).last()
            df = df.loc[df.index.isin(quarter_ends, level=0)]
        
        return df
    
    def _cross_sectional_corr(self, df, factor_cols) -> Dict:
        """
        横截面相关性计算（核心创新点）
        
        解决的问题：避免 df.corr() 混叠截面和时序信息
        正确做法：每天计算 → 时间平均
        """
        if not isinstance(df.index, pd.MultiIndex):
            return {
                'method': 'simple_fallback',
                'pearson': df[factor_cols].corr(method='pearson').to_dict(),
                'spearman': df[factor_cols].corr(method='spearman').to_dict(),
                'warning': '非MultiIndex格式，结果可能不准确'
            }
        
        daily_pearson = []
        daily_spearman = []
        method_diffs = []
        n_stocks_list = []
        
        for date in sorted(df.index.get_level_values('date').unique()):
            daily = df.xs(date, level='date')[factor_cols]
            if len(daily) >= 10:
                p = daily.corr(method='pearson')
                s = daily.corr(method='spearman')
                daily_pearson.append(p)
                daily_spearman.append(s)
                method_diffs.append((p - s).abs().max().max())
                n_stocks_list.append(len(daily))
        
        if not daily_pearson:
            return {'error': '有效天数不足'}
        
        avg_p = pd.concat(daily_pearson).groupby(level=0).mean()
        avg_s = pd.concat(daily_spearman).groupby(level=0).mean()
        
        return {
            'method': 'cross_sectional',
            'avg_pearson': avg_p.to_dict(),
            'avg_spearman': avg_s.to_dict(),
            'n_days': len(daily_pearson),
            'avg_n_stocks': float(np.mean(n_stocks_list)),
            'method_consistency': {
                'mean_diff': float(np.mean(method_diffs)),
                'recommendation': (
                    '存在非线性关系，优先参考Spearman' if np.mean(method_diffs) > 0.15 
                    else '线性假设成立'
                )
            }
        }
    
    def _time_series_corr(self, df, factor_cols) -> Dict:
        """时间序列相关性（基于因子收益率）"""
        returns = df[factor_cols].groupby(level='asset').diff().dropna()
        
        if len(returns) < 30:
            return {'error': '样本不足'}
        
        return {
            'method': 'time_series',
            'pearson': returns.corr(method='pearson').to_dict(),
            'spearman': returns.corr(method='spearman').to_dict(),
            'n_obs': len(returns)
        }
    
    def _rolling_stability(self, df, factor_cols, config) -> Dict:
        """滚动窗口稳定性分析"""
        window = config['rolling_window']
        step = config['rolling_step']
        
        if not isinstance(df.index, pd.MultiIndex):
            return {'error': '需要MultiIndex'}
        
        dates = sorted(df.index.get_level_values('date').unique())
        results = []
        
        for i in range(0, len(dates) - window + 1, step):
            win_dates = dates[i:i+window]
            sample = win_dates[::max(1, step//5)]
            
            corrs = []
            for d in sample:
                try:
                    daily = df.xs(d, level='date')[factor_cols]
                    if len(daily) >= 10:
                        c = daily.corr(method='spearman').values[
                            np.triu_indices_from(daily.corr(), k=1)
                        ]
                        corrs.extend(c.tolist())
                except (KeyError, ValueError):
                    continue
            
            if corrs:
                results.append({
                    'window_end': str(win_dates[-1].date()),
                    'mean_abs_corr': float(np.mean(np.abs(corrs))),
                    'regime': 'high' if np.mean(np.abs(corrs)) > 0.6 else ('low' if np.mean(np.abs(corrs)) < 0.25 else 'normal')
                })
        
        if not results:
            return {'error': '数据不足'}
        
        rdf = pd.DataFrame(results)
        stability = 1.0 - safe_divide(float(rdf['mean_abs_corr'].std()), float(rdf['mean_abs_corr'].mean()), default=0.0)
        
        return {
            'stability_score': round(float(stability), 4),
            'regime_dist': rdf['regime'].value_counts().to_dict(),
            'volatile': bool(stability < 0.6),
            'series': rdf.to_dict('records')
        }
    
    def _significance_tests(self, cs, ts=None) -> Dict:
        """显著性检验（横截面相关性Fisher z变换检验，ts参数预留时间序列检验扩展）"""
        tests = []
        
        if 'avg_pearson' in cs and 'n_days' in cs:
            n_days = cs['n_days']
            sig_pairs = []
            
            for f1, f2_dict in cs['avg_pearson'].items():
                for f2, val in f2_dict.items():
                    if f1 != f2 and isinstance(val, (int, float)):
                        if abs(val) < 1 and n_days > 3:
                            # Fisher z变换：对平均相关系数进行显著性检验
                            z_values = np.arctanh(pd.Series([val]))
                            z_mean = z_values.mean()
                            z_se = 1 / np.sqrt(n_days - 3)
                            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_mean) / z_se))
                            
                            if p_value < 0.05:
                                sig_pairs.append({
                                    'pair': f"{f1}-{f2}",
                                    'corr': round(val, 4),
                                    'p_value': round(p_value, 6)
                                })
            
            tests.append({'type': 't_test', 'significant': sig_pairs[:20]})
        
        return {'tests_performed': [t['type'] for t in tests], 'results': tests}
    
    def _vif_analysis(self, df, factor_cols) -> Dict:
        """VIF多重共线性检验"""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            clean = df[factor_cols].dropna()
            if len(clean) < 30:
                return {'error': '样本不足'}
            
            vif_data = []
            for i, col in enumerate(factor_cols):
                try:
                    v = variance_inflation_factor(clean[factor_cols].values, i)
                    vif_data.append({
                        'factor': col,
                        'vif': round(v, 2),
                        'level': ('severe' if v > 10 else ('high' if v > 5 else ('moderate' if v > 2.5 else 'low')))
                    })
                except (ValueError, np.linalg.LinAlgError):
                    vif_data.append({'factor': col, 'vif': np.nan, 'level': 'error'})
            
            vs = [x['vif'] for x in vif_data if not np.isnan(x.get('vif'))]
            
            return {
                'table': vif_data,
                'max_vif': max(vs) if vs else np.nan,
                'has_issue': max(vs) > 5 if vs else False,
                'warnings': (
                    [f"严重共线性(VIF>10): {[x['factor'] for x in vif_data if x['vif'] > 10]}"]
                    if any(x['vif'] > 10 for x in vif_data) else []
                ) + (
                    [f"高度共线性(5<VIF≤10): {[x['factor'] for x in vif_data if 5 < x['vif'] <= 10]}"]
                    if any(5 < x['vif'] <= 10 for x in vif_data) else []
                )
            }
        except ImportError:
            return {'error': 'statsmodels未安装'}

    # ==================== P2-3: RMS相关性指标 ====================

    def calculate_rms_correlation(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        计算RMS(均方根)相关性 — 多因子组合分散化评估指标

        BigQuant AlphaMiner 使用PCA分析因子解释方差比例，
        FactorHub 用 RMS 相关性作为等价且更直观的替代方案。

        Args:
            correlation_matrix: 因子间相关系数矩阵

        Returns:
            {
                "rms_corr": float,          # RMS(平均绝对相关性)
                "mean_abs_corr": float,     # 平均绝对相关性
                "max_abs_corr": float,      # 最大绝对相关性
                "interpretation": str,       # 可读解读
                "diversification_score": float  # 分散化评分 (0-100)
            }
        """
        if correlation_matrix is None or correlation_matrix.empty:
            return {"error": "相关矩阵为空"}

        n_factors = correlation_matrix.shape[0]
        if n_factors < 2:
            return {
                "error": f"相关矩阵仅包含{n_factors}个因子，计算RMS相关性至少需要2个因子",
                "rms_corr": 0.0,
                "mean_abs_corr": 0.0,
                "max_abs_corr": 0.0,
                "interpretation": "因子数量不足，无法评估分散化程度",
                "diversification_score": 100.0
            }

        vals = correlation_matrix.values
        upper_indices = np.triu_indices(n_factors, k=1)
        upper_vals = vals[upper_indices]

        mean_abs_corr = float(np.mean(np.abs(upper_vals)))
        rms_corr = float(np.sqrt(np.mean(np.square(upper_vals))))
        max_abs_corr = float(np.max(np.abs(upper_vals)))

        diversification_score = max(0.0, (1.0 - rms_corr) * 100)

        if rms_corr < 0.1:
            interpretation = f"极低相关性(RMS={rms_corr:.3f})，因子间高度独立，组合分散性优秀"
        elif rms_corr < 0.25:
            interpretation = f"低相关性(RMS={rms_corr:.3f})，因子间基本独立，组合分散性良好"
        elif rms_corr < 0.4:
            interpretation = f"中等相关性(RMS={rms_corr:.3f})，部分因子存在重叠，建议关注"
        elif rms_corr < 0.6:
            interpretation = f"较高相关性(RMS={rms_corr:.3f})，因子重叠明显，考虑去重或降维"
        else:
            interpretation = f"高相关性(RMS={rms_corr:.3f})，因子严重重叠，强烈建议去重"

        return {
            "rms_corr": rms_corr,
            "mean_abs_corr": mean_abs_corr,
            "max_abs_corr": max_abs_corr,
            "n_factor_pairs": len(upper_vals),
            "interpretation": interpretation,
            "diversification_score": diversification_score,
        }

    def _interpret_results(self, result: Dict) -> Dict:
        """
        智能解读（自建规则引擎 - 替代Corrpy）
        
        纯Python实现，零外部依赖，30行核心逻辑
        """
        interp = {
            'high_correlation_pairs': [],
            'low_correlation_pairs': [],
            'nonlinear_warnings': [],
            'overall_assessment': ''
        }
        
        # 解读横截面相关性
        cs = result.get('cross_sectional', {})
        if 'avg_spearman' in cs:
            for f1, f2_dict in cs['avg_spearman'].items():
                for f2, val in f2_dict.items():
                    if f1 < f2 and isinstance(val, (int, float)):
                        abs_val = abs(val)
                        
                        if abs_val > 0.7:
                            interp['high_correlation_pairs'].append({
                                'pair': f"{f1} vs {f2}",
                                'correlation': round(val, 4),
                                'strength': '极强' if abs_val > 0.9 else '强',
                                'action': '考虑正交化或移除'
                            })
                        elif abs_val < 0.2:
                            interp['low_correlation_pairs'].append({
                                'pair': f"{f1} vs {f2}",
                                'correlation': round(val, 4),
                                'note': '独立性良好'
                            })
            
            # 方法一致性检查
            consistency = cs.get('method_consistency', {})
            if consistency.get('mean_diff', 0) > 0.15:
                interp['nonlinear_warnings'].append(
                    f"Pearson与Spearman差异较大({consistency['mean_diff']:.3f})，可能存在非线性关系"
                )
        
        # 滚动稳定性解读
        rs = result.get('rolling_stability', {})
        if rs.get('volatile'):
            interp['nonlinear_warnings'].append(
                "因子相关性随时间剧烈波动，静态平均值可能具有误导性"
            )
        
        # VIF解读
        vif = result.get('vif_analysis', {})
        if vif.get('has_issue'):
            interp['nonlinear_warnings'].extend(vif.get('warnings', []))
        
        # 总体评估
        n_high = len(interp['high_correlation_pairs'])
        _n_low = len(interp['low_correlation_pairs'])
        
        if n_high == 0:
            interp['overall_assessment'] = '✅ 因子间独立性良好，适合组合使用'
        elif n_high <= 2:
            interp['overall_assessment'] = f'⚠️ 存在{n_high}对高相关因子，需关注组合权重'
        else:
            interp['overall_assessment'] = f'❌ 存在{n_high}对高相关因子，强烈建议正交化或筛选'
        
        return interp
    
    def _generate_insights(self, result: Dict) -> Tuple[List[str], List[str]]:
        """生成警告和建议"""
        warnings = []
        recommendations = []
        
        # 数据质量
        prep = result.get('preprocessing', {})
        if 'missing_ratio' in prep:
            for col, ratio in prep['missing_ratio'].items():
                if ratio > 0.15:
                    warnings.append(f"因子'{col}'缺失率{ratio:.1%}较高")
        
        # 方法一致性
        cs = result.get('cross_sectional', {}).get('method_consistency', {})
        if cs.get('mean_diff', 0) > 0.15:
            warnings.append("Pearson与Spearman差异大，可能存在非线性关系")
            recommendations.append("优先参考Spearman系数或考虑非线性组合")
        
        # 滚动稳定性
        rs = result.get('rolling_stability', {})
        if rs.get('volatile'):
            warnings.append("相关性随时间剧烈波动")
            recommendations.append("使用动态因子配置（如风险平价）")
        
        # VIF
        vif = result.get('vif_analysis', {})
        if vif.get('has_issue'):
            warnings.extend(vif.get('warnings', []))
            recommendations.append("对高VIF因子进行滚动窗口正交化")
        
        return warnings, recommendations
    
    def _check_data_quality(self, df, factor_cols) -> Dict:
        """快速数据质量检查"""
        quality = {}
        for col in factor_cols:
            s = df[col]
            quality[col] = {
                'missing_pct': round(s.isna().mean(), 4),
                'n_unique': s.nunique(),
                'extreme_pct': float(((s < s.quantile(0.01)) | (s > s.quantile(0.99))).mean()) if s.dtype != 'object' else None
            }
        return quality
    
    def analyze_mixed_type(self, df, factor_cols, categorical_cols=None):
        """
        混合类型相关性分析（需要phik，否则降级到scipy）
        
        Args:
            df: DataFrame
            factor_cols: 所有要分析的列
            categorical_cols: 明确的分类列（可选）
        
        Returns:
            分析结果
        """
        if categorical_cols is None:
            categorical_cols = [c for c in factor_cols if df[c].dtype == 'object' or df[c].nunique() < 10]
        
        interval_cols = [c for c in factor_cols if c not in categorical_cols]
        
        if not categorical_cols:
            return {'message': '无分类变量，使用标准相关性即可', 'use_standard': True}
        
        if PHIK_AVAILABLE:
            try:
                phi_k = df[factor_cols].phik_matrix(interval_cols=interval_cols)
                sig = df[factor_cols].significance_matrix(interval_cols=interval_cols)
                
                return {
                    'method': 'phik',
                    'phi_k_matrix': phi_k.to_dict(),
                    'significance': sig.to_dict(),
                    'categorical_cols': categorical_cols,
                    'interval_cols': interval_cols
                }
            except Exception as e:
                logger.warning(f"Phik计算失败: {e}，降级到scipy")
        
        # 降级方案：scipy实现
        logger.info("使用scipy降级方案处理混合类型数据")
        
        results = {}
        for cat_col in categorical_cols:
            for num_col in interval_cols:
                try:
                    groups = df.groupby(cat_col)[num_col]
                    f_stat, p_val = scipy_stats.f_oneway(*[g.values for _, g in groups])
                    results[f"{cat_col}_vs_{num_col}"] = {
                        'anova_f': float(f_stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    }
                except (ValueError, TypeError):
                    continue
        
        return {
            'method': 'scipy_fallback',
            'results': results,
            'note': 'Phik未安装，使用ANOVA作为替代'
        }


# 全局实例
factor_correlation_service = FactorCorrelationService()

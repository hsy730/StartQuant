/**
 * 因子相关性分析增强功能 - 前端调用示例
 * 
 * 对应后端API:
 * POST /api/analysis/correlation/enhanced - 完整相关性分析
 * POST /api/analysis/correlation/interpret  - 智能解读
 * POST /api/analysis/correlation/mixed-type - Phik混合类型分析
 */

import { useState } from 'react'

import request from './api'

// ==================== 类型定义 ====================

interface CorrelationAnalysisRequest {
  factor_names: string[];
  stock_codes: string[];
  start_date: string;
  end_date: string;
  config?: {
    rolling_window?: number;
    rolling_step?: number;
    use_knn?: boolean;
    knn_neighbors?: number;
    winsorize_method?: 'mad' | 'std';
    n_sigma?: number;
  };
}

interface CorrelationAnalysisResponse {
  success: boolean;
  data: {
    metadata: {
      n_factors: number;
      n_observations: number;
      mode: 'standard';
      timestamp: string;
    };
    data_quality: Record<string, any>;
    preprocessing: {
      original_shape?: [number, number];
      frequency_aligned?: boolean;
      frequency_info?: Record<string, string>;
      missing_ratio?: Record<string, number>;
      imputation?: {
        method: string;
        validation?: Record<string, any>;
      };
      winsorization?: {
        method: string;
        n_sigma: number;
        clipped: number;
      };
      final_shape?: [number, number];
      warning?: string;
    };
    cross_sectional: {
      method: string;
      avg_pearson: Record<string, any>;
      avg_spearman: Record<string, any>;
      n_days: number;
      avg_n_stocks: number;
      method_consistency: {
        mean_diff: number;
        recommendation: string;
      };
    };
    time_series: {
      method: string;
      pearson?: Record<string, any>;
      spearman?: Record<string, any>;
      n_obs?: number;
    };
    rolling_stability?: {
      stability_score: number;
      regime_dist: Record<string, number>;
      volatile: boolean;
      series: Array<{
        window_end: string;
        mean_abs_corr: number;
        regime: string;
      }>;
    };
    significance: {
      tests_performed: string[];
      results: Array<{
        type: string;
        significant: Array<{
          pair: string;
          corr: number;
          p_value: number;
        }>;
      }>;
    };
    vif_analysis: {
      table: Array<{
        factor: string;
        vif: number;
        level: string;
      }>;
      max_vif: number;
      has_issue: boolean;
      warnings: string[];
    };
    interpretation: {
      high_correlation_pairs: Array<{
        pair: string;
        correlation: number;
        strength: string;
        action: string;
      }>;
      low_correlation_pairs: Array<{
        pair: string;
        correlation: number;
        note: string;
      }>;
      nonlinear_warnings: string[];
      overall_assessment: string;
    };
    warnings: string[];
    recommendations: string[];
  };
  metadata: {
    factors_analyzed: number;
    stocks_analyzed: number;
    time_range: string;
    warnings_count: number;
    recommendations_count: number;
  };
}


// ==================== API 调用函数 ====================

export async function analyzeFactorCorrelation(
  requestData: CorrelationAnalysisRequest
): Promise<CorrelationAnalysisResponse> {
  try {
    const data = await request.post<CorrelationAnalysisResponse>(
      '/analysis/correlation/enhanced',
      requestData,
      { timeout: 30000 }
    )

    return data

  } catch (error) {
    console.error('因子相关性分析失败:', error)
    throw error
  }
}


// ==================== 使用示例 ====================

async function exampleUsage() {
  console.log('开始因子相关性分析...\n');
  
  try {
    const result = await analyzeFactorCorrelation({
      factor_names: ['momentum', 'value', 'quality', 'volatility', 'size'],
      stock_codes: [
        '000001', '000002', '000063', '000069', '000100',
        '600000', '600009', '600016', '600028', '600030'
      ],
      start_date: '2023-01-01',
      end_date: '2023-12-31',
      config: {
        rolling_window: 120,
        rolling_step: 20,
        use_knn: true,
        knn_neighbors: 5,
        winsorize_method: 'mad',
        n_sigma: 3.0
      }
    });
    
    if (result.success) {
      const { data, metadata } = result;
      
      console.log('分析完成!\n');
      console.log('基本信息:');
      console.log(`   分析因子数: ${metadata.factors_analyzed}`);
      console.log(`   股票数量: ${metadata.stocks_analyzed}`);
      console.log(`   时间范围: ${metadata.time_range}`);
      console.log(`   计算模式: ${data.metadata.mode}\n`);
      
      console.log('数据质量:');
      if (data.preprocessing.frequency_info) {
        console.log(`   频率对齐: ${JSON.stringify(data.preprocessing.frequency_info)}`);
      }
      if (data.preprocessing.missing_ratio) {
        console.log(`   缺失率: ${JSON.stringify(data.preprocessing.missing_ratio)}`);
      }
      console.log('');
      
      console.log('横截面相关性:');
      console.log(`   有效天数: ${data.cross_sectional.n_days}`);
      console.log(`   平均股票数: ${data.cross_sectional.avg_n_stocks.toFixed(1)}`);
      console.log(`   方法一致性差异: ${data.cross_sectional.method_consistency.mean_diff.toFixed(4)}`);
      console.log(`   建议: ${data.cross_sectional.method_consistency.recommendation}\n`);
      
      if (data.rolling_stability) {
        console.log('滚动稳定性:');
        console.log(`   稳定性评分: ${(data.rolling_stability.stability_score * 100).toFixed(1)}%`);
        console.log(`   波动警告: ${data.rolling_stability.volatile ? '是' : '否'}`);
        console.log(`   状态分布: ${JSON.stringify(data.rolling_stability.regime_dist)}\n`);
      }
      
      console.log('警告信息:');
      if (data.warnings.length > 0) {
        data.warnings.forEach((w, i) => console.log(`   ${i+1}. ${w}`));
      } else {
        console.log('   无警告');
      }
      console.log('');
      
      console.log('改进建议:');
      if (data.recommendations.length > 0) {
        data.recommendations.forEach((r, i) => console.log(`   ${i+1}. ${r}`));
      } else {
        console.log('   无特别建议');
      }
      console.log('');
      
      if (data.vif_analysis && !data.vif_analysis.error) {
        console.log(source: 'VIF多重共线性分析:');
        console.log(`   最大VIF: ${data.vif_analysis.max_vif?.toFixed(2)}`);
        console.log(`   存在共线性: ${data.vif_analysis.has_issue ? '是' : '否'}`);
        
        if (data.vif_analysis.warnings?.length > 0) {
          console.log('   警告:');
          data.vif_analysis.warnings.forEach(w => console.log(`     • ${w}`));
        }
        console.log('');
      }
      
    } else {
      console.error('分析失败');
    }
    
  } catch (error) {
    console.error('请求失败:', error);
  }
}


// ==================== React 组件示例 ====================

export function FactorCorrelationPanel() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CorrelationAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await analyzeFactorCorrelation({
        factor_names: ['momentum', 'value', 'quality'],
        stock_codes: ['000001', '000002', '600000'],
        start_date: '2023-01-01',
        end_date: '2023-12-31'
      });
      
      setResult(response);
      
    } catch (err: any) {
      setError(err.message || '分析失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="factor-correlation-panel">
      <h2>因子相关性分析</h2>
      
      <button 
        onClick={handleAnalyze} 
        disabled={loading}
      >
        {loading ? '分析中...' : '开始分析'}
      </button>
      
      {error && <div className="error">{error}</div>}
      
      {result?.success && (
        <div className="results">
          {/* TODO: 实现相关性热力图组件 */}
          {/* <CorrelationHeatmap data={result.data.cross_sectional.avg_spearman} /> */}
          
          {/* TODO: 实现滚动稳定性图表组件 */}
          {/* <RollingStabilityChart data={result.data.rolling_stability?.series} /> */}
          
          {/* TODO: 实现VIF条形图组件 */}
          {/* <VIFBarChart data={result.data.vif_analysis.table} /> */}
          
          {/* 警告和建议 */}
          <div className="alerts">
            <h3>分析结果</h3>
            <p>{result.data.interpretation.overall_assessment}</p>
            
            {result.data.warnings.length > 0 && (
              <div className="warnings">
                <h4>警告</h4>
                <ul>
                  {result.data.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {result.data.recommendations.length > 0 && (
              <div className="recommendations">
                <h4>建议</h4>
                <ul>
                  {result.data.recommendations.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}


export default {
  analyzeFactorCorrelation,
  FactorCorrelationPanel
};
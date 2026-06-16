/**
 * 因子相关性分析增强功能
 * 
 * 对应后端API:
 * POST /api/analysis/correlation/enhanced - 完整相关性分析
 * POST /api/analysis/correlation/interpret  - 智能解读
 * POST /api/analysis/correlation/mixed-type - Phik混合类型分析
 */

import request from './api'

// ==================== 类型定义 ====================

export interface CorrelationAnalysisRequest {
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

export interface CorrelationAnalysisResponse {
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
  const data = await request.post<CorrelationAnalysisResponse>(
    '/analysis/correlation/enhanced',
    requestData,
    { timeout: 30000 }
  ) as unknown as CorrelationAnalysisResponse
  return data
}

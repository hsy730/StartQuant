"""
Phik混合类型相关性服务（可选依赖）

使用场景：
- 因子包含行业分类、市值分层等离散变量
- 需要检测非线性依赖关系

安装方式（按需）:
    pip install factor-flow[advanced]
    或
    uv pip install phik

降级策略：
如果未安装phik，FactorCorrelationService会自动使用scipy的ANOVA作为替代
"""
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    import phik
    from phik import report as phik_report
    PHIK_AVAILABLE = True
except ImportError:
    PHIK_AVAILABLE = False


class PhikCorrelationService:
    """
    Phi_K相关性分析服务（可选插件）
    
    注意：这是高级功能，大多数情况下不需要。
    标准的数值因子分析请使用 FactorCorrelationService
    """
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return PHIK_AVAILABLE
    
    def analyze(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        categorical_cols: List[str] = None
    ) -> Dict[str, Any]:
        """执行Phi_K分析"""
        if not PHIK_AVAILABLE:
            return {
                'error': 'phik未安装',
                'install_hint': '运行: pip install phik 或 uv pip install "factor-flow[advanced]"'
            }
        
        try:
            if categorical_cols is None:
                categorical_cols = [c for c in factor_cols 
                                   if df[c].dtype == 'object' or df[c].nunique() < 10]
            
            interval_cols = [c for c in factor_cols if c not in categorical_cols]
            
            if not categorical_cols:
                return {'message': '无分类变量，无需使用Phik'}
            
            phi_k = df[factor_cols].phik_matrix(interval_cols=interval_cols)
            sig = df[factor_cols].significance_matrix(interval_cols=interval_cols)
            
            return {
                'method': 'phik',
                'phi_k_matrix': phi_k.to_dict(),
                'significance_matrix': sig.to_dict(),
                'categorical_cols': categorical_cols,
                'interval_cols': interval_cols
            }
            
        except Exception as e:
            logger.error(f"Phik分析失败: {e}")
            return {'error': str(e)}


# 全局实例（懒加载）
_phik_service = None

def get_phik_service() -> Optional[PhikCorrelationService]:
    """获取Phik服务实例（如果可用）"""
    global _phik_service
    if _phik_service is None and PHIK_AVAILABLE:
        _phik_service = PhikCorrelationService()
    return _phik_service

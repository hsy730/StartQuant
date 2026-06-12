"""
FactorHub 全局常量定义

集中管理所有魔法值，消除散落的硬编码。
所有服务/工具/API层应引用此文件中的常量。

修改日期: 2026-06-12 (代码审查第22轮后统一)
"""

import numpy as np

# ============================================================================
# 市场参数
# ============================================================================

# A股年交易日数（用于收益率/波动率年化）
ANNUAL_TRADING_DAYS = 252

# 无风险利率（年化，用于Sharpe/Sortino等风险调整指标）
RISK_FREE_RATE = 0.03

# 默认每笔交易股数
DEFAULT_SHARES_PER_TRADE = 100

# ============================================================================
# 浮点安全阈值
# ============================================================================

# 判定浮点数"近似为零"的阈值（规则7.6/7.40）
FLOAT_ZERO_THRESHOLD = 1e-10

# 极小正数（防止除零的下界）
EPSILON = 1e-10

# ============================================================================
# 因子验证阈值
# ============================================================================

# IC均值通过阈值 — 统一为0.02（原0.03与0.02混用）
IC_PASS_THRESHOLD = 0.02

# IR通过阈值
IR_PASS_THRESHOLD = 0.5

# IR上限封顶（防止极端IR爆炸）
IR_CAP = 5.0

# 换手率上限阈值
TURNOVER_THRESHOLD = 0.5

# 统计显著性水平（p < alpha 时拒绝原假设）
STATISTICAL_SIGNIFICANCE_ALPHA = 0.05
HIGHLY_SIGNIFICANT_ALPHA = 0.01

# VaR/CVaR置信水平截断
VAR_CONFIDENCE_CUTOFF = 0.05

# ============================================================================
# 滚动窗口参数
# ============================================================================

# 月度滚动IC窗口（约20个交易日）
ROLLING_IC_WINDOW = 20

# 季度滚动窗口
QUARTERLY_WINDOW = 60

# 半年度滚动窗口
SEMI_ANNUAL_WINDOW = 120

# 年度滚动窗口
ANNUAL_WINDOW = ANNUAL_TRADING_DAYS  # 252

# IC计算最小样本量
MIN_SAMPLE_SIZE_FOR_IC = 20

# 稳定性检验最少数据量（2年）
MIN_STABILITY_DAYS = ANNUAL_TRADING_DAYS * 2  # 504

# ============================================================================
# 预处理参数
# ============================================================================

# Winsorize百分位边界
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# 缺失率警告阈值
MISSING_RATE_WARNING_THRESHOLD = 0.3

# Z-Score clip范围
ZSCORE_CLIP_MIN = -3.0
ZSCORE_CLIP_MAX = 3.0

# ============================================================================
# 评分权重（comprehensive_scoring_service）
# ============================================================================

# 收益维度权重
SCORE_RETURN_WEIGHT = 0.3

# 风险维度权重
SCORE_RISK_WEIGHT = 0.3

# 效率维度权重
SCORE_EFFICIENCY_WEIGHT = 0.4

# 中性基线（评分回退值）
NEUTRAL_BASELINE = 0.5

# 评分满分
MAX_SCORE = 100.0

# ============================================================================
# 相关性阈值
# ============================================================================

# 高相关性判定阈值
HIGH_CORRELATION_THRESHOLD = 0.7

# 因子间最大允许相关性（唯一性惩罚起点）
MAX_FACTOR_CORRELATION = 0.7

# ============================================================================
# 遗传算法参数
# ============================================================================

# 交叉概率
GENETIC_CROSSOVER_PROB = 0.7

# 变异概率
GENETIC_MUTATION_PROB = 0.3

# 方程相似度判定阈值
EQUATION_SIMILARITY_THRESHOLD = 0.7

# ============================================================================
# 字典键名常量（消除中英文键名混用）
# ============================================================================


class Keys:
    """API响应和服务内部字典键名常量"""

    # IC/IR指标键名（统一使用英文）
    IC_MEAN = "ic_mean"
    IC_STD = "std_ic"
    IR = "ir"
    IC_POSITIVE_RATIO = "ic_positive_ratio"
    T_STATISTIC = "t_statistic"
    P_VALUE = "p_value"
    RANK_IC = "rank_ic"

    # 风险指标键名
    TOTAL_RETURN = "total_return"
    ANNUAL_RETURN = "annual_return"
    SHARPE_RATIO = "sharpe_ratio"
    VOLATILITY = "volatility"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    VAR_95 = "var_95"
    CVAR_95 = "cvar_95"

    # API响应结构键名
    SUCCESS = "success"
    ERROR = "error"
    MESSAGE = "message"
    DATA = "data"
    WARNING = "warning"

    # 因子挖掘结果键名
    BEST_FACTORS = "best_factors"
    FITNESS = "fitness"
    EXPRESSION = "expression"
    RANK = "rank"
    COMPLEXITY = "complexity"
    SCORE = "score"
    PASSED = "passed"
    OVERALL_SCORE = "overall_score"

    # 验证结果键名
    IC_VALIDATION = "ic_validation"
    IR_VALIDATION = "ir_validation"

    # 编排流水线键名
    HAS_BIAS = "has_bias"
    RISK_LEVEL = "risk_level"
    RISK_SCORE = "risk_score"
    STABILITY_SCORE = "stability_score"
    N_STOCKS = "n_stocks"


class ChineseKeys:
    """中文键名常量（analysis_service等历史接口使用）"""

    IC_MEAN = "IC均值"
    IC_STD = "IC标准差"
    IR = "IR"
    IC_POSITIVE_RATIO = "IC>0占比"
    T_STATISTIC = "t统计量"
    P_VALUE = "p值"
    IC_TYPE = "IC类型"
    IC_ABS_MEAN = "IC绝对值均值"
    IC_COUNT = "IC计数"
    RISK_ADJUSTED_IC = "风险调整IC"

# FactorHub 项目全局规则

## 📌 项目核心价值主张

> **"让专业用户拥有完全控制权，同时让新手用户也能获得业界最佳实践的结果"**

### 设计哲学

FactorHub 采用**混合模式（Smart Default + Optional Override）**设计所有需要参数配置的功能模块：

1. **智能默认**：系统根据数据特征自动选择最优参数，确保零配置即可获得可靠结果
2. **可选覆盖**：高级用户可以完全自定义每个参数，满足特殊研究需求
3. **透明可解释**：所有自动决策都有明确的理由和置信度评分

---

## 🔬 量化数据预处理规范（业界标准）

### 必须遵循的处理顺序

```
原始因子数据 → 缺失值处理 → 去极值 → 中性化(市值+行业) → 标准化 → 用于分析/回测
```

#### Step 1: 缺失值处理
- **默认策略**: 用0填充（适合因子值为比率/收益率的场景）
- **ML场景**: 用中位数填充（更稳健）
- **极端情况**: 超过30%缺失的因子应标记警告

#### Step 2: 去极值（Winsorization）

| 方法 | 适用场景 | 参数建议 | 稳健性 |
|------|---------|---------|--------|
| **MAD法** ⭐推荐 | 肥尾分布、存在异常值 | n_sigma=2.5-3.0 | ⭐⭐⭐⭐⭐ |
| 百分位法 | 非正态分布、偏态显著 | limits=(0.01, 0.99) | ⭐⭐⭐⭐ |
| 3σ标准差法 | 近似正态分布、快速计算 | n_sigma=3.0 | ⭐⭐⭐ |

**市场板块自适应规则**：
- 主板 (60xxxx, 00xxxx): `n_sigma = 3.0` （标准波动）
- 创业板 (30xxxx): `n_sigma = 2.8` （高波动，收紧20%）
- 科创板 (68xxxx): `n_sigma = 2.7` （更高波动，收紧25%）
- 北交所 (8xxxx, 4xxxx): `n_sigma = 2.5` （最高波动，收紧33%）

#### Step 3: 中性化（Neutralization）

##### 3a. 市值中性化（必须启用）
```python
# 线性回归残差法
log_market_cap = ln(market_cap)
residuals = factor_values - β * log_market_cap - α
```
- **触发条件**: 市值变异系数(CV) > 0.5 时强制启用
- **效果**: 消除规模效应，避免大市值股票主导结果

##### 3b. 行业中性化（条件启用）
```python
# 行业内Z-score标准化
factor_neutralized = (factor - industry_mean) / industry_std
```
- **启用条件**: 
  - 行业数 ≥ 3 且
  - 最小行业样本量 ≥ 10 只股票
- **禁用条件**: 
  - 行业分类不足或
  - 存在样本量 < 5 的微小行业

**重要原则**：中性化的基准严格基于传入的数据集，不会受到外部市场板块的干扰！

#### Step 4: 标准化（Standardization）

| 方法 | 输出范围 | 适用场景 |
|------|---------|---------|
| **Z-score** ⭐推荐 | (-∞, +∞), μ≈0, σ≈1 | 一般分析、线性模型 |
| Rank标准化 | [0, 1], 均匀分布 | 非参数方法、异常值多 |
| Median-MAD | (-∞, +∞), 抗异常值 | 肥尾分布、离群点 |

---

## 🏗️ 架构设计原则

### 0. 开源库优先原则（最重要）

> **"非核心竞争力的通用基础功能，必须使用成熟稳定的开源库，禁止手搓代码"**

量化系统的核心竞争力在于**业务逻辑和策略研究**，而非重新实现通用计算。手搓通用代码会导致：边界条件遗漏、长期维护负担、与业界标准不一致。

#### 必须使用开源库的功能领域

| 功能领域 | 推荐开源库 | 禁止自实现 |
|---------|-----------|-----------|
| 风险指标（Sharpe/Sortino/MaxDD/Calmar/VaR/CVaR） | `empyrical-reloaded` | ❌ 手动计算年化收益/夏普/索提诺 |
| 因子分析（IC/IR/分层收益/换手率） | `alphalens-reloaded` | ❌ 自实现横截面IC计算 |
| 投资组合优化（均值-方差/风险平价/最大夏普） | `pyportfolioopt` | ❌ 自实现权重优化 |
| 回测引擎 | `vectorbt` | ❌ 自建回测循环 |
| 技术指标（MA/MACD/RSI/布林带） | `TA-Lib` / `pandas-ta` | ❌ 手动实现技术指标公式 |
| 统计检验（t检验/正态性/相关性） | `scipy.stats` / `statsmodels` | ❌ 自实现统计检验 |
| 遗传编程/因子挖掘 | `DEAP` / `PySR` | ❌ 自实现进化算法 |
| 数据预处理（去极值/标准化） | `scipy.stats.mstats.winsorize` / `sklearn.preprocessing` | ❌ 自实现MAD/百分位截断的底层计算 |

#### 判断标准：什么时候用开源库 vs 自实现

```
是否属于项目核心竞争力（因子定义、策略逻辑、A股特有规则）？
├── 是 → 自实现（如涨跌停Mask-First设计、A股板块自适应参数）
└── 否 → 是否有成熟稳定的开源库？
    ├── 是 → 必须使用开源库（如empyrical计算Sharpe）
    └── 否 → 自实现，但必须：
        1. 在代码注释中说明"无合适开源库"
        2. 编写充分的单元测试覆盖边界条件
        3. 在代码审查时重点检查
```

#### 正确的封装模式

```python
# ✅ 正确：直接使用开源库，不做fallback
import empyrical

def calculate_sharpe(returns, risk_free_rate=0.03):
    """风险指标统一入口，底层委托empyrical"""
    return empyrical.sharpe_ratio(returns, risk_free=risk_free_rate)
```

```python
# ❌ 错误：try/except + 手动fallback（开源库是本地依赖，不存在不可用的情况）
try:
    import empyrical
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False

def calculate_sharpe(returns, risk_free_rate=0.03):
    if EMPYRICAL_AVAILABLE:
        return empyrical.sharpe_ratio(returns, risk_free=risk_free_rate)
    else:
        # 手动fallback → 维护负担，边界条件遗漏
        daily_rf = risk_free_rate / 252
        excess = returns - daily_rf
        return excess.mean() / excess.std() * np.sqrt(252)
```

```python
# ❌ 错误：直接手搓，不使用开源库
def calculate_sharpe(returns, risk_free_rate=0.03):
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    return excess.mean() / excess.std() * np.sqrt(252)  # 边界条件未处理
```

#### 代码审查强制检查项

- [ ] 新增的统计/数学/金融计算是否使用了对应的开源库？
- [ ] 如果自实现，是否在注释中说明了"无合适开源库"的理由？
- [ ] 自实现的代码是否有充分的边界条件测试（除零、NaN、空数据）？
- [ ] 风险指标计算是否通过 `risk_metrics.py` 统一入口？
- [ ] IC计算是否通过 `alphalens_analysis_service` 统一入口？

### 1. 代码复用优先级

```
公共工具类 → 服务层复用 → API层调用 → 前端展示
```

**示例**：
- ✅ 创建统一的 `FactorPreprocessingPipeline` 类
- ❌ 在 AnalysisService 和 BacktestService 中分别实现去极值逻辑

### 2. 性能要求

- **单因子 x 单股票 x 100天**: < 1ms
- **多因子(5) x 多股票(50) x 250天**: < 2秒
- **大数据集(100万样本)**: 吞吐量 > 3000 样本/秒

**实现手段**：
- 使用 pandas 向量化操作（禁止 Python 循环）
- 合理使用并行处理（ThreadPoolExecutor）
- 避免不必要的 DataFrame 复制

### 3. 错误处理策略

```python
# 分级错误处理
try:
    # 正常业务逻辑
    result = pipeline.process(data)
except InsufficientDataError as e:
    # 数据不足 → 返回部分结果 + 警告
    logger.warning(f"样本不足: {e}")
    return partial_result_with_warning(e)
except InvalidParameterError as e:
    # 参数错误 → 抛出明确错误信息
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    # 未预期错误 → 记录日志 + 返回友好提示
    logger.error(f"预处理失败: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="内部服务错误")
```

---

## 🎨 UI/UX 设计规范

### 配置面板设计模式

对于需要复杂配置的功能（如数据预处理），必须采用以下UI模式：

#### 模式切换器
```
┌─────────────────────────────────────┐
│  ○ 智能(推荐)  ● 自定义             │  ← Radio.Group
└─────────────────────────────────────┘
```

#### 智能模式下：
1. 显示"一键生成"按钮
2. 展示检测到的数据特征（只读）
3. 显示推荐配置及置信度
4. 显示推荐理由（人类可读）
5. 提供预设模板快捷选择

#### 自定义模式下：
1. 显示所有可调参数
2. 每个参数提供：
   - 清晰的标签和单位
   - Tooltip 解释
   - 合理的默认值
   - 滑块或下拉框（根据参数类型）
   - 实时验证反馈

### 信息层次结构

```
Card Title (图标 + 文字 + Tooltip)
├── Mode Selector (智能/自定义)
│   ├── [Smart Mode Panel]
│   │   ├── Action Button (触发智能检测)
│   │   ├── Confidence Badge (置信度)
│   │   ├── Data Characteristics (数据特征摘要)
│   │   ├── Reasoning Text (推荐理由)
│   │   └── Preset Selector (预设模板)
│   └── [Custom Mode Panel]
│       ├── Parameter Group 1 (去极值)
│       │   ├── Method Select
│       │   └── Intensity Slider
│       ├── Parameter Group 2 (中性化)
│       │   └── Toggle Switches
│       └── Parameter Group 3 (标准化)
│           └── Method Radio
└── Validation Warnings (如果有)
```

---

## 📊 数据流规范

### 因子分析标准流程

```
用户输入(stock_codes, factor_names, dates)
    ↓
[API Layer] validation + logging
    ↓
[Service Layer] 
    ├─ FactorService.calculate_factors()     ← 获取原始因子
    ├─ SmartDetector.analyze_data()           ← 分析特征（可选）
    ├─ PreprocessingPipeline.process()        ← 执行美颜
    └─ AnalysisService.calculate_ic_ir()      ← 计算指标
    ↓
[Response] results + preprocessing_stats + warnings
```

### 日志记录规范

```python
# 关键节点必须记录日志
logger.info(f"开始因子分析: {len(stock_codes)}只股票, {factor_names}")
logger.info(f"智能推荐配置(置信度{confidence*100:.0f}%): {reasoning}")
logger.info(f"预处理完成: 截断{stats['winsorized_count']}个异常值")
logger.warning(f"行业{ind}只有{n}只股票，中性化可能不稳定")
```

---

## 🧪 测试规范

### 单元测试覆盖要求

| 功能模块 | 最少测试用例 | 必须覆盖的场景 |
|---------|------------|---------------|
| 去极值方法 | 3+ | MAD/Percentile/STD 各一个 |
| 中性化 | 2+ | 有市值/无市值，有行业/无行业 |
| 标准化 | 3+ | Z-score/Rank/Median-MAD |
| 完整管道 | 1+ | 三步流程集成 |
| 性能测试 | 1+ | 大数据集吞吐量 |
| 边界情况 | 3+ | 空数据/常数/无穷大 |

### 测试命名规范

```python
def test_[功能]_[场景]_[预期结果]:
    """
    示例:
    test_mad_winsorization_with_outliers_should_clip_extreme_values
    test_market_cap_neutralization_should_reduce_correlation_by_80_percent
    """
```

---

## 🔐 安全性规范

### 敏感数据处理

- ❌ **禁止**: 在日志中打印原始因子值、用户密码、API Key
- ✅ **允许**: 打印统计量（均值、标准差）、脱敏后的股票代码前缀
- ✅ **必须**: 对用户输入进行校验（SQL注入、XSS防护）

### 参数校验

```python
# 所有外部输入必须校验
def validate_preprocessing_config(config: Dict):
    if config.get("winsorize_n_sigma", 3.0) < 1.5:
        raise ValueError("去极值强度过低，可能导致过度截断")
    if config.get("winsorize_n_sigma", 3.0) > 6.0:
        raise ValueError("去极值强度过高，可能丢失有效信号")
```

---

## 📈 版本兼容性

### 向后兼容原则

新增可选参数时：
- ✅ 设置合理的默认值（与旧行为一致）
- ✅ 在文档中标注"新增于 v.x.x"
- ✅ 提供迁移指南（如有破坏性变更）

修改核心算法时：
- ✅ 保留旧版本作为 fallback（通过配置开关）
- ✅ 对比新旧结果的差异报告
- ✅ 至少保留2个大版本的兼容期

---

## 🚀 性能优化 Checklist

在提交性能敏感的代码前，必须检查：

- [ ] 是否使用了 pandas/numpy 向量化操作？
- [ ] 是否避免了 Python 循环（for/while）？
- [ ] 是否合理使用了并行处理？
- [ ] 是否有不必要的 DataFrame copy？
- [ ] 内存占用是否合理（大数据集是否分块处理）？
- [ ] 是否有性能基准测试（benchmark）？

---

## 📚 参考资源

### 业界权威资料
1. 西部证券《2024年多因子Alpha挖掘框架》
2. BigQuant 量化平台《因子清洗与预处理》
3. MLFactor《Machine Learning for Factor Investing》
4. Grinold & Kahn《Active Portfolio Management》

### 开源项目参考
1. Alphalens (Quantopian) - 因子分析工具
2. PyPortfolioOpt - 投资组合优化
3. VectorBT - 回测引擎

---

## ✅ 代码审查 Checklist

在提交涉及数据预处理的PR时， reviewer必须确认：

- [ ] 是否遵循了"去极值→中性化→标准化"的正确顺序？
- [ ] 是否使用了统一的 `FactorPreprocessingPipeline`？
- [ ] 是否支持智能默认 + 手动覆盖两种模式？
- [ ] 是否有充分的单元测试覆盖？
- [ ] 是否有性能测试通过？
- [ ] 日志记录是否充分？
- [ ] 错误处理是否完善？

---

**最后更新**: 2026-06-06  
**维护者**: FactorHub Core Team  
**适用版本**: v1.0.0+

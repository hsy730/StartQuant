# FactorHub vs BigQuant vs JoinQuant 全维度对比分析

> **更新日期**: 2026-06-02
> **维护者**: FactorHub Core Team
> **适用版本**: v1.0.0+

---

## 目录

- [一、平台定位与架构差异](#一平台定位与架构差异)
- [二、因子计算流程差异](#二因子计算流程差异)
- [三、数据预处理流程差异](#三数据预处理流程差异)
- [四、回测系统差异](#四回测系统差异)
- [五、因子分析方法差异](#五因子分析方法差异)
- [六、因子挖掘系统差异](#六因子挖掘系统差异)
- [七、数据层差异](#七数据层差异)
- [八、前端与交互差异](#八前端与交互差异)
- [九、核心差异总结矩阵](#九核心差异总结矩阵)
- [十、FactorHub 差异化竞争力与改进方向](#十factorhub-差异化竞争力与改进方向)

---

## 一、平台定位与架构差异

| 维度 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **定位** | 开源全栈因子分析平台 | 商业AI量化云平台 | 商业量化投研云平台 |
| **部署模式** | 本地部署（单机） | 云原生 + 本地SDK代理 | SaaS云端 + 本地客户端 |
| **架构风格** | 单体 FastAPI + React SPA | 微服务 + 云原生（Docker/K8s/Spark） | 微服务 + 分布式（Celery/RabbitMQ） |
| **开源程度** | ✅ 全栈开源 | ❌ 商业闭源，SDK部分开放 | ❌ 商业闭源，jqdatasdk 开源 |
| **核心卖点** | Smart Default + Mask-First | AI驱动 + StockRanker + DAI | 事件驱动回测 + 实盘交易 + 社区生态 |
| **目标用户** | 个人量化研究者 | 机构 + 个人（AI方向） | 个人 + 高校 + 券商 |

### 架构图对比

**FactorHub**:
```
┌──────────────────────────────────────────────────┐
│            前端 (React + Ant Design + ECharts)    │
└──────────────────────┬───────────────────────────┘
                       │ REST API (JSON)
┌──────────────────────┴───────────────────────────┐
│            API 层 (FastAPI Routers)                │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────┐
│            服务层 (38个 Business Services)          │
│  FactorService | AnalysisService | BacktestService │
│  FactorPreprocessingPipeline | SmartDetector       │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────┐
│            数据层 (AKShare + SQLite + Cache)        │
└──────────────────────────────────────────────────┘
```

**BigQuant**:
```
┌──────────────────────────────────────────────────┐
│     用户交互层 (Web UI / Notebook / Open API)      │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────┐
│            应用服务层 (微服务群)                     │
│  因子实验室 | 策略引擎 | AI模型中心 | 回测引擎       │
│  实盘交易网关 | 市场系统 | 组合管理                  │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────┐
│            数据管理层                                │
│  行情数据库 | 财务数据库 | 因子库 | Redis缓存        │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────┐
│            基础设施层                                │
│  云计算(阿里云) | Docker/K8s | Spark/Dask           │
│  Kafka/RabbitMQ | CI/CD & 监控                     │
└──────────────────────────────────────────────────┘
```

**JoinQuant**:
```
┌──────────────────────────────────────────────────┐
│     Web 前端层 (Vue.js + Element UI + Monaco)      │
└──────────────────────┬───────────────────────────┘
                       │ HTTPS / WebSocket
┌──────────────────────┴───────────────────────────┐
│     API 网关层 (Nginx + Tornado)                    │
│     认证鉴权 | 限流 | 路由分发                       │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────┐
│     业务逻辑层 (Python 微服务群)                     │
│  策略引擎 | 回测引擎 | 因子分析 | 数据服务            │
│  模拟交易 | 实盘交易 | 风控服务 | 调度服务            │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────┐
│     数据存储层                                       │
│  MongoDB(行情) | Redis(缓存) | MySQL(用户/策略)     │
│  InfluxDB(时序) | ElasticSearch(日志/搜索)          │
└──────────────────────────────────────────────────┘
```

### 关键差异解读

- **FactorHub** 是唯一完全开源、可本地部署的平台，适合对数据隐私有要求、希望深度定制的用户
- **BigQuant** 的核心优势在于 **DAI 数据引擎 + FAI 分布式算力 + BigTrader C++ 回测内核** 三角架构，以及 StockRanker 排序学习模型
- **JoinQuant** 的核心优势在于 **完整的实盘交易链路**（模拟→实盘→券商对接）和 **15万+社区生态**

---

## 二、因子计算流程差异

### 2.1 因子定义方式

| 维度 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **表达式语法** | 麦语言 + TALib + eval/exec | SQL（DAI查询） | Python API（get_factor_values） |
| **自定义函数** | 18个麦语言函数 + 14个TALib函数 | SQL内置函数（m_avg, m_lag等） | Python自由编程 |
| **因子编译** | FormulaCompilerService 可视化公式树→代码 | SQL直接执行 | 无编译，直接Python |
| **因子存储** | SQLite + YAML配置 | 云端DataSource + 分区索引 | 云端策略代码 |
| **版本管理** | ✅ FactorVersionService 完整版本控制 | ✅ DataSource版本管理 | ❌ 无显式版本管理 |

**关键差异**：

- BigQuant 用 **SQL** 表达因子逻辑（`SELECT date, instrument, -1 * m_avg(turn,60) AS factor FROM cn_stock_bar1d`），这是其 DAI 引擎的核心设计——因子计算下推到数据库层执行，避免大量数据传输
- FactorHub 用 **eval/exec** 动态执行因子代码，灵活性最高但安全性需额外关注
- JoinQuant 用 **Python API** 编程，学习曲线最平缓

### 2.2 因子计算引擎

| 维度 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **计算模式** | 批量向量化（pandas） | SQL引擎 + FAI分布式（Ray） | 逐Bar推送（事件驱动） |
| **并行策略** | ThreadPoolExecutor（10 workers） | Ray分布式集群 | Celery分布式任务队列 |
| **预置因子数** | 60+（12类） | 数百（含Alpha101/191） | 数百（含Alpha101 + 技术指标 + 基本面） |
| **Mask-First** | ✅ 原生支持（涨跌停/停牌过滤） | ✅ 平台内置 | ⚠️ 部分支持（需手动处理） |

### 2.3 因子定义语法示例

**FactorHub**（麦语言 + TALib）:
```python
# 表达式模式
code = "close / SMA(close, timeperiod=20)"

# 函数模式
def calculate_factor(df):
    return df["close"].rolling(20).mean() / df["close"].rolling(60).mean()
```

**BigQuant**（SQL）:
```sql
SELECT date, instrument, -1 * m_avg(turn, 60) AS factor
FROM cn_stock_bar1d
ORDER BY date, instrument
```

**JoinQuant**（Python API）:
```python
def initialize(context):
    g.factor_data = {}

def handle_data(context, data):
    stocks = get_index_stocks('000300.XSHG')
    for stock in stocks:
        hist = attribute_history(stock, 20, '1d', ['close', 'volume'])
        g.factor_data[stock] = hist['close'].mean()
```

---

## 三、数据预处理流程差异

这是三者差异**最小**的领域，因为 FactorHub 明确参考了 JoinQuant/BigQuant 的业界标准。

### 3.1 标准流程对比

```
三者一致的标准流程：
原始因子数据 → 缺失值处理 → 去极值 → 中性化(市值+行业) → 标准化
```

FactorHub 在 `factor_preprocessing_pipeline.py` 和 `factor_neutralization_service.py` 中明确标注了 "JoinQuant/BigQuant标准"。

### 3.2 去极值方法对比

| 方法 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **MAD法** | ✅ 推荐，n_sigma=2.5-3.0 | ✅ 推荐 | ✅ 推荐 |
| **百分位法** | ✅ limits=(0.01,0.99) | ✅ | ✅ |
| **3σ法** | ✅ n_sigma=3.0 | ✅ | ✅ |
| **市场板块自适应** | ✅ **独有**（主板3.0/创业板2.8/科创板2.7/北交所2.5） | ❌ 无公开文档 | ❌ 无公开文档 |
| **智能参数推荐** | ✅ **独有**（SmartPreprocessingDetector + 置信度评分） | ⚠️ AlphaMiner内置data_process=True | ❌ 手动配置 |

**FactorHub 的创新点**：

1. **市场板块自适应规则**：根据股票代码前缀自动识别板块，调整 n_sigma 参数。这是对 BigQuant/JoinQuant 标准的**增量改进**，更精细地处理 A 股不同板块的波动特性

   | 市场板块 | 代码规则 | 涨跌幅限制 | n_sigma 推荐 |
   |---------|---------|-----------|-------------|
   | 主板 | 60xxxx, 00xxxx | +/-10% | 3.0 |
   | 创业板 | 30xxxx | +/-20% | 2.8 |
   | 科创板 | 68xxxx | +/-20% | 2.7 |
   | 北交所 | 8xxxx, 4xxxx | +/-30% | 2.5 |

2. **SmartPreprocessingDetector**：自动分析数据分布特征（偏度、峰度、肥尾检测），输出推荐配置 + 置信度评分 + 人类可读推荐理由。BigQuant 的 AlphaMiner 有 `data_process=True` 开关，但没有公开的智能推荐机制

### 3.3 中性化方法对比

| 方法 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **市值中性化** | ✅ `factor ~ log(market_cap)` 回归残差 | ✅ 一致 | ✅ 一致 |
| **行业中性化** | ✅ `factor ~ industry_dummies` 回归残差 | ✅ 一致 | ✅ 一致 |
| **联合中性化** | ✅ `factor ~ industry + log(cap)` 一次回归 | ✅ 一致 | ✅ 一致 |
| **中性化基准** | ✅ 严格基于传入数据集 | ✅ 一致 | ✅ 一致 |

**结论**：中性化方法三者完全一致，均采用 **线性回归残差法**，且都采用 **一次回归联合中性化**（而非分步），数学上最严谨。

核心实现代码（三者一致）:

```python
# 市值中性化
log_market_cap = np.log(market_cap)
model = LinearRegression()
model.fit(log_market_cap.reshape(-1,1), factor_values)
residuals = factor_values - model.predict(log_market_cap)

# 行业中性化（回归残差法，非简单减均值）
industry_dummies = pd.get_dummies(industries, drop_first=True)
model = LinearRegression()
model.fit(industry_dummies, factor_values)
residuals = factor_values - model.predict(industry_dummies)

# 联合中性化（市值+行业一次回归）
X = np.hstack([industry_dummies, log_market_cap.reshape(-1,1)])
model = LinearRegression()
model.fit(X, factor_values)
residuals = factor_values - model.predict(X)
```

### 3.4 标准化方法对比

| 方法 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **Z-score** | ✅ 推荐 | ✅ 推荐 | ✅ 推荐 |
| **Rank** | ✅ | ✅ | ✅ |
| **Median-MAD** | ✅ | ❌ 无公开文档 | ❌ 无公开文档 |

### 3.5 预定义配置对比

| 配置 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **保守配置** | ✅ CONSERVATIVE_CONFIG: MAD + 更严格边界 + 双中性化 | ❌ | ❌ |
| **激进配置** | ✅ AGGRESSIVE_CONFIG: 百分位 + 更宽松边界 + 无中性化 | ❌ | ❌ |
| **ML配置** | ✅ ML_MODEL_CONFIG: MAD + 中位数填充 + 双中性化 | ❌ | ❌ |

---

## 四、回测系统差异

这是三者差异**最大**的领域。

### 4.1 回测引擎架构

| 维度 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **引擎** | VectorBT（向量化） | BigTrader（C++内核） | Zipline-reloaded（事件驱动） |
| **执行模式** | 批量向量化 | 事件驱动 + 撮合模拟 | 事件驱动（handle_data逐Bar） |
| **撮合模型** | 简化（无逐笔撮合） | **VWAP加权撮合** | **精确撮合（含涨跌停判断）** |
| **回测速度** | ⚡ 极快（向量化） | ⚡ 快（C++内核） | 🐢 较慢（逐Bar遍历） |
| **真实性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **降级策略** | ✅ VectorBT不可用→自建fallback | N/A（商业平台） | N/A |

**关键差异解读**：

- **FactorHub** 选择 VectorBT 的核心原因是**性能**——向量化回测比事件驱动快 10-100 倍，适合快速迭代因子研究。但代价是撮合精度较低，不适合高频策略验证
- **BigQuant** 的 BigTrader 是 **C++ 内核**，兼具性能和精度，支持 VWAP 加权撮合、滑点建模、手续费精确计算，是三者中回测最真实的
- **JoinQuant** 继承 Zipline 的 **事件驱动** 架构，最接近真实交易流程，支持分钟级回测、融资融券、期货等复杂场景

### 4.2 回测模式对比

| 模式 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **单因子分层** | ✅ | ✅ | ✅ |
| **多因子组合** | ✅ 等权/IC加权/IR加权 | ✅ StockRanker排序 | ✅ 自由编程 |
| **横截面选股** | ✅ Top N 选股 | ✅ | ✅ |
| **分钟级回测** | ❌ | ✅ | ✅ |
| **模拟交易** | ❌ | ✅ | ✅ |
| **实盘交易** | ❌ | ✅ | ✅（对接大同证券） |
| **策略对比** | ✅ StrategyComparisonService | ✅ | ✅ |

### 4.3 交易成本模型

| 维度 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **佣金** | 万三（买卖双边） | 可配置 | 万2.5-万三 |
| **印花税** | 千一（仅卖出） | 可配置 | 千一（仅卖出） |
| **滑点** | ✅ 智能滑点检测（**独有**） | ✅ 可配置 | ✅ 可配置 |
| **涨跌停限制** | ✅ Mask-First | ✅ 内置 | ✅ 内置 |

**FactorHub 独有**：SmartSlippageDetector 根据股票流动性、市值、策略换手率自动推荐滑点参数，支持保守/激进偏好模式。BigQuant 和 JoinQuant 都需要手动设置滑点。

### 4.4 回测防作弊机制

| 机制 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **数据对齐** | ✅ 因子值t期，收益t+1期 | ✅ | ✅ |
| **信号延迟** | ✅ 可配置 | ✅ | ✅ 当日信号次日执行 |
| **停牌处理** | ✅ Mask-First tradable_mask | ✅ | ✅ 自动跳过 |
| **涨跌停限制** | ✅ Mask-First | ✅ | ✅ 涨停不可买/跌停不可卖 |
| **未来函数检测** | ❌ | ✅ 内置 | ✅ 内置 |

### 4.5 绩效指标对比

| 指标 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **总收益率/年化收益率** | ✅ | ✅ | ✅ |
| **夏普比率** | ✅ | ✅ | ✅ |
| **索提诺比率** | ✅ | ✅ | ✅ |
| **最大回撤** | ✅ | ✅ | ✅ |
| **卡玛比率** | ✅ | ✅ | ✅ |
| **胜率** | ✅ | ✅ | ✅ |
| **VaR/CVaR** | ✅ 95%置信度 | ✅ | ⚠️ |
| **换手率** | ✅ | ✅ | ✅ |
| **Alpha/Beta** | ✅ | ✅ | ✅ |
| **信息比率** | ✅ | ✅ | ✅ |

---

## 五、因子分析方法差异

### 5.1 IC/IR 分析

| 维度 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **时序IC** | ✅ 滚动窗口 | ✅ | ✅ |
| **横截面IC** | ✅ Alphalens金标准 + fallback | ✅ 自研 | ✅ 自研 |
| **加权IC** | ✅ 市值/流动性/等权 | ✅ | ✅ |
| **Rank IC** | ✅ | ✅ | ✅ |
| **IC统计** | IC均值/标准差/IR/t-stat/p-value | IC均值/ICIR/夏普/换手率 | 类似 |

### 5.2 因子分析工具链

| 工具 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **Alphalens** | ✅ 原生集成 + fallback | ❌ 自研 M.factorlens | ❌ 自研 |
| **SHAP解释** | ✅ **独有**（XGBoost + SHAP） | ❌ | ❌ |
| **因子稳定性** | ✅ FactorStabilityService | ⚠️ AlphaMiner内置 | ⚠️ 需手动 |
| **因子有效性** | ✅ FactorEffectivenessService | ✅ | ✅ |
| **因子相关性** | ✅ 增强（横截面/时序/滚动/VIF/RMS/Phik） | ✅ | ✅ 基础 |
| **因子暴露度** | ✅ FactorExposureService | ✅ | ✅ |
| **因子贡献度** | ✅ FactorAttributionService | ✅ | ✅ 归因分析 |
| **因子监测** | ✅ FactorMonitoringService 时间序列动态 | ❌ | ❌ |
| **综合评分** | ✅ ComprehensiveScoringService IC35%/IR30%/稳定性20%/换手率15% | ✅ | ❌ |

**FactorHub 的优势**：因子分析模块是三者中**最全面**的，7个专项分析服务 + 综合评分，覆盖了因子从"出生"到"退役"的全生命周期。BigQuant 和 JoinQuant 的因子分析更偏向于"验证有效性"，而 FactorHub 提供了持续监测和衰减预警能力。

### 5.3 Alphalens 集成对比

**FactorHub** 的 AlphalensAnalysisService 封装了 alphalens-reloaded 库：

| 分析维度 | 方法 | 输出 |
|---------|------|------|
| 因子分位收益 | 按因子值分5组，计算各组收益 | 分位收益图 |
| 因子IC分析 | Spearman Rank IC + Pearson IC | IC时间序列、IC直方图 |
| 因子换手率 | 分位组合换手率 | 换手率统计 |
| 事件研究 | 因子事件收益 | 事件收益图 |
| Alpha/Beta | CAPM回归 | Alpha/Beta值 |

**BigQuant** 的 M.factorlens.v4 是自研的因子分析模块，返回：
- `_result`: IC均值、ICIR、夏普比率、换手率
- `_detail`: 分组回测、IC分析详细数据

**JoinQuant** 使用自研因子分析工具，功能类似但集成度不如 FactorHub 的 Alphalens 封装。

### 5.4 因子综合评分体系

FactorHub 独有的多维度综合评分：

| 维度 | 权重 | 说明 |
|------|------|------|
| IC | 35% | 信息系数 |
| IR | 30% | 信息比率 |
| 稳定性 | 20% | 因子稳定性得分 |
| 换手率 | 15% | 交易成本评估 |

BigQuant 和 JoinQuant 均无公开的综合评分体系。

---

## 六、因子挖掘系统差异

### 6.1 挖掘算法对比

| 维度 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **遗传规划** | ✅ DEAP（8-Phase优化） | ✅ gplearn深度定制 | ❌ |
| **符号回归** | ✅ **独有**（PySR） | ❌ | ❌ |
| **双算法并行** | ✅ **独有**（DEAP+PySR） | ❌ | ❌ |
| **AlphaMiner** | ❌ | ✅ SQL因子分析框架 | ❌ |
| **StockRanker** | ❌ | ✅ GBDT排序学习 | ❌ |
| **ML模型** | ✅ XGBoost + SHAP | ✅ XGBoost/LGBM/TF/PyTorch | ❌ |

### 6.2 遗传规划实现细节对比

| 维度 | FactorHub (DEAP) | BigQuant (gplearn) |
|------|-------------------|---------------------|
| **基础框架** | DEAP 1.3+ | gplearn（深度修改） |
| **原语集** | 25个（9算术+10时序窗口+3时序相关+2成对+2激活） | 扩充函数集（时序/截面/统计） |
| **Mask-First算子** | ✅ 所有时序算子有Mask版本 | ❌ 无公开文档 |
| **过拟合控制** | ✅ 交叉验证 + 简约性压力 | ✅ 奥卡姆惩罚 |
| **多目标优化** | ✅ NSGA-II（IC最大化 + 复杂度最小化） | ❌ 单目标 |
| **多样性保护** | ✅ 去重 + 相似度惩罚 | ⚠️ 部分支持 |
| **因子值缓存** | ✅ Phase 4 | ❌ 无公开文档 |
| **中性化集成** | ✅ 挖掘过程中可启用 | ✅ 适应度计算时中性化 |

### 6.3 DEAP 8-Phase 优化策略

FactorHub 的 GeneticFactorMiningService 采用 8-Phase 优化：

1. **Phase 1**: 精英策略 + 适应度目标路由
2. **Phase 2**: 简约性压力（防膨胀）
3. **Phase 3**: 多样性保护（去重 + 相似度惩罚）
4. **Phase 4**: 因子值缓存
5. **Phase 5**: 向量化滚动 IC
6. **Phase 6**: 交叉验证过拟合控制
7. **Phase 7**: 扩展基元集（9 → 约25个，含时序窗口操作）
8. **Phase 8**: 前端更新

### 6.4 DEAP GP 原语集

FactorHub 的 FactorPrimitives 定义了25个原语：

| 类别 | 原语 | 数量 |
|------|------|------|
| **基础算术** | add, sub, mul, div, neg, abs, log, sqrt, rank | 9 |
| **时序窗口** | ts_mean_5/10/20, ts_std_5/10/20, ts_delay_1/5, ts_delta_1/5 | 10 |
| **时序相关** | ts_corr_5/10/20（Mask-First版本） | 3 |
| **成对操作** | max, min | 2 |
| **激活函数** | sigmoid, tanh | 2 |

所有时序算子都有 Mask-First 版本，自动过滤涨跌停日。

### 6.5 FactorHub 的独有优势

1. **PySR 符号回归**：基于 Julia 后端的高性能符号回归，与 DEAP 互补——DEAP 擅长探索大搜索空间，PySR 擅长发现简洁可解释的数学公式
2. **双算法并行**：DualMiningService 同时运行 DEAP 和 PySR，合并最优结果
3. **NSGA-II 多目标优化**：同时优化 IC 和复杂度，避免 gplearn 单目标优化的过拟合风险

### 6.6 BigQuant 的独有优势

1. **AlphaMiner**：完整的 SQL 因子分析框架，支持一键式因子验证（分组回测 + IC分析 + 绩效统计）
2. **StockRanker**：GBDT 排序学习模型，能同时吸纳全市场 3000 只股票数据进行排序预测
3. **DAI + FAI**：SQL 计算下推 + Ray 分布式调度，处理亿级数据的能力远超单机方案

---

## 七、数据层差异

| 维度 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **数据源** | AKShare（免费开源） | 自有数据源（DAI） | 自有数据源（JQData） |
| **数据覆盖** | A股日线 + 申万行业 | A股/期货/期权/基金/债券/宏观 | A股/基金/期货/宏观 |
| **分钟数据** | ❌ | ✅ | ✅ |
| **Tick数据** | ❌ | ✅ | ✅ |
| **传输协议** | HTTP（akshare API） | **Apache Arrow Flight**（二进制零拷贝） | RPC + 专属压缩 |
| **缓存机制** | 双层（内存 + SQLite） | 云端分布式缓存 | Redis缓存 |
| **数据质量** | ⚠️ 依赖akshare质量 | ✅ 40万+策略验证 | ✅ 百亿级基金实战验证 |
| **费用** | 免费 | 付费（有免费额度） | 付费（有免费额度） |

### 数据源详细对比

| 数据维度 | FactorHub (AKShare) | BigQuant (DAI) | JoinQuant (JQData) |
|---------|---------------------|----------------|---------------------|
| 日线行情 | ✅ | ✅ | ✅ |
| 分钟行情 | ❌ | ✅ | ✅ |
| 财务数据 | ✅ | ✅ | ✅ |
| 行业分类 | ✅ 申万 | ✅ 申万/中信 | ✅ 申万/中信 |
| 因子数据 | 自研计算 | 内置因子库 | jqfactor 内置 |
| 指数成分 | ✅ | ✅ | ✅ |
| 期货/期权 | ❌ | ✅ | ✅ |
| 宏观数据 | ⚠️ 部分 | ✅ | ✅ |

**关键差异**：

- BigQuant 的 **Arrow Flight** 协议是其核心技术优势——二进制零拷贝传输，支持亿级数据本地化操作，体验如同操作本地文件
- FactorHub 的 AKShare 数据源虽然免费，但在数据完整性、更新及时性、分钟级数据覆盖上与商业数据源有差距
- BigQuant SDK 的 **Local-Cloud 一致性协议**：本地BigTrader C++内核与云端同一代码库，DAI模块屏蔽历史与实时的物理差异，确保回测与线上表现一致

---

## 八、前端与交互差异

| 维度 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **UI框架** | React 19 + Ant Design 6 | React/Vue + Ant Design | Vue 2 + Element UI |
| **可视化** | ECharts 6 | ECharts / Plotly | Matplotlib（Jupyter内） |
| **研究环境** | Web UI 配置面板 | Jupyter Notebook（云端IDE） | Jupyter Notebook（云端IDE） |
| **智能模式** | ✅ Smart/Custom 切换 | ⚠️ data_process=True 开关 | ❌ 手动配置 |
| **代码编辑** | react-simple-code-editor | Monaco Editor（VSCode内核） | Monaco Editor |
| **实时进度** | ✅ 因子挖掘进度条 | ✅ | ✅ |

### FactorHub 的 UI 创新点

PreprocessingConfigPanel 实现了 **Smart/Custom 双模式切换**，这是 BigQuant 和 JoinQuant 都没有的交互模式：

**Smart 模式**：
```
┌─────────────────────────────────────┐
│  ● 智能(推荐)  ○ 自定义             │
├─────────────────────────────────────┤
│  [一键生成] 按钮                     │
│  置信度 Badge: 85%                  │
│  数据特征: 偏度1.2, 峰度5.3, 肥尾   │
│  推荐理由: 检测到创业板股票...       │
│  预设模板: [保守] [标准] [激进]      │
└─────────────────────────────────────┘
```

**Custom 模式**：
```
┌─────────────────────────────────────┐
│  ○ 智能(推荐)  ● 自定义             │
├─────────────────────────────────────┤
│  去极值方法: [MAD ▼]                │
│  去极值强度: ───●──── 3.0           │
│  市值中性化: [✓]                     │
│  行业中性化: [✓]                     │
│  标准化方法: ○ Z-score ● Rank       │
│  缺失值处理: [0填充 ▼]              │
└─────────────────────────────────────┘
```

BigQuant 和 JoinQuant 的因子预处理更偏向"代码级"控制，需要用户自己写 Python 代码配置参数。

---

## 九、核心差异总结矩阵

| 能力维度 | FactorHub | BigQuant | JoinQuant |
|---------|-----------|----------|-----------|
| **开源** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ |
| **部署灵活性** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **数据预处理** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **回测精度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **回测速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **因子分析深度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **因子挖掘** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **AI/ML能力** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **实盘交易** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **数据覆盖** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **智能推荐** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **社区生态** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 十、FactorHub 差异化竞争力与改进方向

### ✅ FactorHub 的独有优势（BigQuant/JoinQuant 不具备）

| # | 优势 | 说明 |
|---|------|------|
| 1 | **市场板块自适应去极值** | 根据主板/创业板/科创板/北交所自动调整 n_sigma |
| 2 | **SmartPreprocessingDetector** | 数据特征自动分析 + 参数推荐 + 置信度评分 |
| 3 | **SmartSlippageDetector** | 基于股票流动性自动推荐滑点 |
| 4 | **Mask-First 全链路** | 从数据加载到因子计算到IC分析，涨跌停过滤贯穿始终 |
| 5 | **PySR 符号回归挖掘** | 与 DEAP 互补的第二种挖掘范式 |
| 6 | **双算法并行挖掘** | DEAP + PySR 同时运行 |
| 7 | **NSGA-II 多目标优化** | IC + 复杂度同时优化 |
| 8 | **因子全生命周期管理** | 版本控制 + 持续监测 + 衰减预警 + 综合评分 |
| 9 | **Smart/Custom 双模式 UI** | 新手零配置 + 专家全控制 |
| 10 | **全栈开源** | 唯一可完全本地部署、深度定制的方案 |

### ⚠️ FactorHub 相对 BigQuant/JoinQuant 的不足

| 不足 | BigQuant 方案 | JoinQuant 方案 | 改进建议 |
|------|-------------|---------------|---------|
| **无分钟级数据** | DAI 支持分钟/Tick | get_bars 支持分钟线 | 接入更多数据源或支持自定义数据导入 |
| **无实盘交易** | BigTrader C++内核 | 对接大同证券实盘 | 可考虑对接开源交易网关 |
| **撮合精度低** | VWAP加权撮合 | 精确撮合含涨跌停 | 可在 VectorBT 基础上增加撮合模拟层 |
| **单机算力有限** | FAI Ray分布式 | Celery分布式 | 可引入 Dask/Ray 实现分布式计算 |
| **无 Alpha101/191 因子库** | 内置数百因子 | 内置Alpha101 | 可批量实现 WorldQuant Alpha101 |
| **无排序学习模型** | StockRanker GBDT | 无 | 可集成 LightGBM Ranker |
| **无云端IDE** | JupyterHub | Jupyter Notebook | 可嵌入 JupyterLab |
| **社区生态薄弱** | 15万+用户 | 15万+用户 | 需长期建设 |

### 技术栈对比总览

| 类别 | FactorHub | BigQuant | JoinQuant |
|------|-----------|----------|-----------|
| **Web框架** | FastAPI | Tornado + 微服务 | Tornado + Nginx |
| **前端** | React 19 + Ant Design 6 | React/Vue + Ant Design | Vue 2 + Element UI |
| **回测引擎** | VectorBT | BigTrader (C++) | Zipline-reloaded |
| **因子分析** | Alphalens-reloaded | M.factorlens (自研) | 自研 |
| **遗传规划** | DEAP | gplearn (深度定制) | ❌ |
| **符号回归** | PySR | ❌ | ❌ |
| **ML框架** | XGBoost + SHAP | XGBoost/LGBM/TF/PyTorch | ❌ |
| **数据源** | AKShare | DAI (Arrow Flight) | JQData (RPC) |
| **数据库** | SQLite | 分布式存储 | MongoDB + MySQL |
| **分布式** | ThreadPoolExecutor | Ray (FAI) | Celery + RabbitMQ |
| **可视化** | ECharts 6 | ECharts / Plotly | Matplotlib |
| **部署** | 本地单机 | 云原生 (K8s) | SaaS + 本地客户端 |

---

**最后更新**: 2026-06-02
**维护者**: FactorHub Core Team
**适用版本**: v1.0.0+

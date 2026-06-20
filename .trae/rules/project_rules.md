# FactorHub 项目全局规则

> 设计哲学：**Smart Default + Optional Override**。系统智能推荐最优参数，高级用户可完全自定义。

---

## 量化数据预处理顺序

```
原始数据 → 缺失值处理 → 去极值 → 中性化(市值+行业) → 标准化 → 分析/回测
```

| 步骤 | 方法 | 参数/规则 |
|------|------|----------|
| 缺失值 | 默认填0；ML场景填中位数 | >30%缺失标记警告 |
| 去极值 | **MAD** ⭐推荐 | 主板 n_sigma=3.0；创业板 2.8；科创板 2.7；北交所 2.5 |
| 市值中性化 | 线性回归残差法 | CV>0.5 时强制启用 |
| 行业中性化 | 行业内Z-score | 行业≥3 且 最小行业≥10只才启用 |
| 标准化 | **Z-score** ⭐推荐 | 异常值多用 Rank；肥尾用 Median-MAD |

**中性化基准严格基于传入数据集**，不受外部市场板块干扰。

---

## 架构设计原则

### 0. 开源库优先（最重要）

> 非核心竞争力的通用功能，**必须使用成熟开源库，禁止手搓**。

| 功能 | 库 | 禁止 |
|------|-----|------|
| 风险指标 | `empyrical-reloaded` | 手动计算 Sharpe/Sortino |
| 因子分析 | `alphalens-reloaded` | 自实现横截面 IC |
| 组合优化 | `pyportfolioopt` | 自实现权重优化 |
| 回测 | `vectorbt` | 自建回测循环 |
| 技术指标 | `TA-Lib` / `pandas-ta` | 手动实现指标公式 |
| 统计检验 | `scipy.stats` | 自实现统计检验 |
| 因子挖掘 | `DEAP` / `PySR` | 自实现进化算法 |

**判断标准**：是否属于核心竞争力（因子定义、策略逻辑、A股特有规则）？→ 否 → 必须用开源库。

**封装模式**：直接调用，不做 `try/except ImportError + fallback`。开源库是本地依赖，不存在不可用的情况。

### 1. 代码复用优先级

```
公共工具类 → 服务层复用 → API层调用 → 前端展示
```

### 2. 性能要求

- 单因子×单股票×100天: < 1ms
- 多因子(5)×多股票(50)×250天: < 2秒
- 大数据集(100万样本): 吞吐量 > 3000 样本/秒

**手段**：pandas 向量化（禁 Python 循环）、ThreadPoolExecutor 并行、避免不必要 copy。

### 3. 错误处理

```python
try:
    result = pipeline.process(data)
except ValueError as e:        # 参数错误 → 400
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:         # 未预期 → 500 + 日志
    logger.error(f"失败: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="内部服务错误")
```

### 4. Bug 排查方法论（后端 Python）

> **优先复现，再读源码**。禁止未复现就主观推断问题根因。

| 优先级 | 手段 | 适用场景 |
|-------|------|---------|
| 1 | **分析故障日志** | 生产/测试环境报错，日志中有 traceback |
| 2 | **单元测试复现** | 纯逻辑错误、金融计算异常、边界条件 |
| 3 | **API 接口测试复现** | 端到端链路问题、跨层交互异常、进度/状态异常 |
| 4 | **源码分析** | 仅在前三步已复现并定位范围后，才深入源码 |

**原则**：
- 先拿到 ** traceback + 输入参数 + 返回结果**，再下结论。
- 能写测试脚本复现的，必须写测试脚本；不能复现的 bug 不做修复。
- 禁止根据源码"觉得会出错"就直接改代码，必须有运行证据。

---

## 量化计算防坑（project_rules 独有规则）

> 以下规则在 `code_smell_prevention.md` 中**无对应条目**，需特别注意。

### 规则A：量纲一致性 — 比例与绝对金额不可混用

手续费必须与收益率同量纲（比例）。只在最终输出净值时乘以 `initial_capital`。

```python
# ✅ 比例手续费
commission = weight_change * commission_rate  # 0.1 * 0.0003 = 0.003%
net_return = portfolio_returns - commission

# ❌ 绝对金额混用
commission = weight_change * capital * rate   # = 30元
net_return = portfolio_returns - commission    # 0.01 - 30 = 净值爆炸
```

### 规则B：前视偏差 — shift方向 ≠ 前视偏差

| 场景 | 是否前视 |
|------|---------|
| `shift(-1)` 获取**未来收益**（结果变量） | ✅ 合法 |
| `shift(-1)` 生成**交易信号**（决策变量） | ❌ 前视偏差 |

区分关键：**信号生成**是否用了未来数据。

### 规则C：开源库多返回值必须用命名变量

```python
# ❌ 凭直觉猜索引
result = talib.BBANDS(close)[2]

# ✅ 命名变量接收
upper, middle, lower = talib.BBANDS(close)
result = middle
```

### 规则D：`safe_divide` 与 `safe_ir` 规范

详见 `code_smell_prevention.md` 规范1 + 规则7.x。此处不再重复。

---

## UI/UX 配置面板模式

复杂配置功能必须采用 **智能/自定义双模式**：

```
Card Title
├── Radio: ○ 智能(推荐)  ● 自定义
│   ├── [智能模式] → 一键生成 + 数据特征(只读) + 置信度 + 推荐理由 + 预设模板
│   └── [自定义模式] → 全部参数可调 + Tooltip + 默认值 + 实时验证
└── Validation Warnings
```

---

## 数据流规范

```
用户输入 → API Layer (校验+日志) → Service Layer
    ├─ FactorService.calculate_factors()      ← 原始因子
    ├─ SmartDetector.analyze_data()            ← 特征分析(可选)
    ├─ PreprocessingPipeline.process()         ← 预处理
    └─ AnalysisService.calculate_ic_ir()       ← 指标计算
→ Response: results + preprocessing_stats + warnings
```

---

## 测试规范

| 模块 | 最少用例 | 必覆盖场景 |
|------|---------|-----------|
| 去极值 | 3+ | MAD/Percentile/STD |
| 中性化 | 2+ | 有/无市值，有/无行业 |
| 标准化 | 3+ | Z-score/Rank/Median-MAD |
| 完整管道 | 1+ | 四步集成 |
| 性能 | 1+ | 大数据集吞吐量 |
| 边界 | 3+ | 空数据/常数/无穷大 |

**命名**：`test_[功能]_[场景]_[预期]`

---

## 安全与性能

### 安全
- ❌ 日志中禁止打印原始因子值、密码、API Key
- ✅ 必须校验用户输入（SQL注入、XSS）

### 性能 Checklist
- [ ] pandas/numpy 向量化，无 Python 循环
- [ ] 合理并行，无不必要 DataFrame copy
- [ ] 大数据集分块处理，有 benchmark

---

## 版本兼容

- 新增可选参数：默认值与旧行为一致，标注"新增于 v.x.x"
- 修改核心算法：保留旧版本 fallback + 差异报告 + 兼容2个大版本

---

## PR 审查 Checklist

### 数据预处理 PR
- [ ] 顺序正确：缺失值→去极值→中性化→标准化
- [ ] 使用统一 `FactorPreprocessingPipeline`
- [ ] 支持智能默认 + 手动覆盖
- [ ] 单元测试 + 性能测试通过

### 量化计算 PR
- [ ] 量纲一致（比例 vs 绝对金额）
- [ ] shift 区分"获取未来收益"和"信号前视"
- [ ] 开源库多返回值用命名变量
- [ ] 新增统计/数学计算使用对应开源库
- [ ] 遵守 `code_smell_prevention.md` 全部规范

---

**最后更新**: 2026-06-16
**维护者**: FactorHub Core Team
**适用版本**: v1.0.0+

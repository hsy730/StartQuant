# 操作日志

## 2026-11-16 | 创建 | 安全除法
- 依据 backend/utils/safe_math.py 与代码规范提炼除零保护的统一入口与语义边界

## 2026-11-16 | 创建 | 因子预处理四步顺序
- 依据 project_rules.md 与 factor_preprocessing_pipeline.py 提炼四步顺序及板块参数

## 2026-11-16 | 创建 | 横截面SpearmanIC
- 依据 ic_calculator.py 提炼逐日横截面 Spearman 口径与统一入口

## 2026-11-16 | 创建 | FactorHub
- 依据项目根目录与规则文件提炼平台技术栈、启动方式与分层结构

## 2026-11-16 | 创建 | FactorPreprocessingPipeline
- 依据 factor_preprocessing_pipeline.py 提炼默认配置与对外方法

## 2026-11-16 | 创建 | WeightOptimizer
- 依据 weight_optimizer_service.py 提炼六种权重方法入口与回退语义

## 2026-11-16 | 创建 | 因子研究链路
- 汇总预处理、IC/IR、权重三个环节的单点实现关系

## 2026-11-16 | 更新 | 内容目录
- 首次写入 7 个页面后同步 index.md 三类目录

## 2026-11-16 | 创建 | 前视偏差判定
- 依据 lookahead_bias_detector.py 与 project_rules.md 提炼 shift 语义与泄漏阈值

## 2026-11-16 | 创建 | 评分归一与截断
- 依据 comprehensive_scoring_service.py 提炼维度映射公式与总分截断

## 2026-11-16 | 创建 | 量纲一致性
- 依据 project_rules.md 规则A 与回测服务默认成本参数提炼量纲约束

## 2026-11-16 | 创建 | LookaheadBiasDetector
- 依据 lookahead_bias_detector.py 提炼五档风险等级与默认阈值

## 2026-11-16 | 创建 | ComprehensiveScoringService
- 依据 comprehensive_scoring_service.py 提炼三个评分入口与默认权重

## 2026-11-16 | 创建 | VectorBTBacktestService
- 依据 vectorbt_backtest_service.py 提炼回测入口与分块、滑点机制

## 2026-11-16 | 创建 | 因子评估与回测链路
- 汇总体检、评分、回测三个环节与因子研究链路的衔接关系

## 2026-11-16 | 更新 | 内容目录
- 第二批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 单一真相源
- 依据代码规范与 risk_metrics、ic_calculator 等统一入口提炼

## 2026-11-16 | 创建 | 智能默认与手动覆盖
- 依据 project_rules.md 与 smart_preprocessing_detector.py 提炼推荐流程

## 2026-11-16 | 创建 | 零波动率边界处理
- 依据 risk_metrics.py 提炼零标准差下的指标降级规则

## 2026-11-16 | 创建 | RiskMetrics
- 依据 risk_metrics.py 与 constants.py 提炼 10 项指标与相对指标入口

## 2026-11-16 | 创建 | MiningAlgorithmRegistry
- 依据 mining_algorithm_registry.py 提炼 5 个内置算法与工厂接口

## 2026-11-16 | 创建 | SmartPreprocessingDetector
- 依据 smart_preprocessing_detector.py 提炼特征分析与推荐输出结构

## 2026-11-16 | 创建 | 因子挖掘与入库链路
- 汇总推荐、挖掘、体检、评分、入库的先后依赖

## 2026-11-16 | 更新 | 内容目录
- 第三批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 版本快照与回滚
- 依据 factor_version_service.py 与 factor_version.py 提炼版本留痕规则

## 2026-11-16 | 创建 | 公式白名单沙箱
- 依据 formula_compiler_service.py 提炼函数与属性白名单、校验顺序

## 2026-11-16 | 创建 | 时序标准化避免前视
- 依据编译器的 zscore 生成逻辑提炼滚动窗口约束

## 2026-11-16 | 创建 | FactorVersionService
- 依据 factor_version_service.py 提炼六个方法、版本号规则与回滚流程

## 2026-11-16 | 创建 | FormulaCompilerService
- 依据 formula_compiler_service.py 提炼编译、解析、校验三个入口

## 2026-11-16 | 创建 | 因子定义与版本演进链路
- 汇总编译、校验、留痕、回滚的先后依赖与主表关系

## 2026-11-16 | 更新 | 内容目录
- 第四批 6 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 可交易性掩码优先
- 依据 data_service.py 的 Mask-First 设计提炼掩码条件与生成时机

## 2026-11-16 | 创建 | 缓存键与TTL失效
- 依据 cache_service.py 与 data_service.py 提炼键设计与过期语义

## 2026-11-16 | 创建 | 智能自定义双模式面板
- 依据 PreprocessingConfigPanel.tsx 与 project_rules.md 提炼交互规范

## 2026-11-16 | 创建 | DataService
- 依据 data_service.py 提炼取数、复权、板块限幅与掩码配置

## 2026-11-16 | 创建 | CacheService
- 依据 cache_service.py 提炼元数据/数据分离与 7 个对外方法

## 2026-11-16 | 创建 | PreprocessingConfigPanel
- 依据面板组件提炼模式切换、置信度着色与警告展示

## 2026-11-16 | 创建 | 数据加载与预处理链路
- 汇总取数、缓存、掩码、参数配置到预处理执行的关系

## 2026-11-16 | 更新 | 内容目录
- 第五批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 横截面分位归因
- 依据 factor_attribution_service.py 提炼逐日分位与未来收益口径

## 2026-11-16 | 创建 | 稳定性判定窗口
- 依据 factor_stability_service.py 提炼双窗口要求与多窗口滚动

## 2026-11-16 | 创建 | 组合集中度与暴露
- 依据 portfolio_analysis_service.py 提炼三维度分析与风险委托

## 2026-11-16 | 创建 | FactorAttributionService
- 依据 factor_attribution_service.py 提炼贡献拆解与显著性判定

## 2026-11-16 | 创建 | FactorStabilityService
- 依据 factor_stability_service.py 提炼四个稳定性方法

## 2026-11-16 | 创建 | PortfolioAnalysisService
- 依据 portfolio_analysis_service.py 提炼暴露、集中度与权重委托

## 2026-11-16 | 创建 | 组合构建与归因链路
- 汇总稳定性、权重、暴露、归因、风险的先后关系

## 2026-11-16 | 更新 | 内容目录
- 第六批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 多Sheet导出约定
- 依据 export_service.py 提炼五 Sheet 顺序与职责边界

## 2026-11-16 | 创建 | 配置覆盖优先级
- 依据 preprocessing_api.py 提炼推荐、预设、校验三端点与覆盖关系

## 2026-11-16 | 创建 | 挖掘检查点续跑
- 依据挖掘任务与检查点模型提炼逐代持久化与恢复条件

## 2026-11-16 | 创建 | ExportService
- 依据 export_service.py 提炼回测与对比导出方法

## 2026-11-16 | 创建 | PreprocessingConfigAPI
- 依据 preprocessing_api.py 提炼请求模型与三个端点

## 2026-11-16 | 创建 | 挖掘任务与检查点模型
- 依据 mining_task.py 与 mining_checkpoint.py 提炼两张表字段

## 2026-11-16 | 创建 | 挖掘任务执行与恢复链路
- 汇总参数、算法、检查点、入库的任务生命周期

## 2026-11-16 | 更新 | 内容目录
- 第七批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 策略信号与权重分离
- 依据 base_strategy.py 提炼信号/权重两段式与基类回测

## 2026-11-16 | 创建 | Alpha101表达式因子库
- 依据 alpha101_factors.py 提炼三元组结构与分类规则

## 2026-11-16 | 创建 | 一键验证流水线
- 依据 factor_orchestrator_service.py 提炼阶段顺序与三档结论

## 2026-11-16 | 创建 | StrategyRegistry
- 依据 strategy_registry.py 提炼四个内置模板与取用接口

## 2026-11-16 | 创建 | Alpha101FactorLibrary
- 依据 alpha101_factors.py 提炼五分类与表达式示例

## 2026-11-16 | 创建 | FactorOrchestrator
- 依据编排器提炼配置项、七个阶段方法与输出结构

## 2026-11-16 | 创建 | 策略模板与回测链路
- 汇总选策略、信号、权重、回测、指标到导出

## 2026-11-16 | 更新 | 内容目录
- 第八批 7 个页面写入后同步 index.md

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

## 2026-11-16 | 创建 | 相关性口径分离
- 依据 factor_correlation_service.py 提炼横截面与时序两种相关口径

## 2026-11-16 | 创建 | 多重共线性VIF筛查
- 依据 factor_correlation_service.py 与 constants.py 提炼 VIF 分档与 RMS 指标

## 2026-11-16 | 创建 | 因子暴露转移矩阵
- 依据 factor_monitoring_service.py 提炼分位迁移、断点与周期性检测

## 2026-11-16 | 创建 | FactorValidationService
- 依据 factor_validation_service.py 提炼六项检查与评分扣减规则

## 2026-11-16 | 创建 | AlphalensAnalysisService
- 依据 alphalens_analysis_service.py 提炼数据准备与四类分析入口

## 2026-11-16 | 创建 | FactorCorrelationService
- 依据 factor_correlation_service.py 提炼八步流程与默认配置

## 2026-11-16 | 创建 | 因子体检与相关性诊断链路
- 汇总三个诊断服务的职责分工与到评估环节的衔接

## 2026-11-16 | 更新 | 内容目录
- 第九批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 回归残差中性化
- 依据 factor_neutralization_service.py 提炼三种中性化与样本门槛

## 2026-11-16 | 创建 | 横截面分层与单调性
- 依据 factor_return_analysis_service.py 提炼逐日分组、利差与单调判定

## 2026-11-16 | 创建 | 加权IC与相关性折减
- 依据 weighted_ic_service.py 提炼五种权重与高相关降权规则

## 2026-11-16 | 创建 | FactorNeutralizationService
- 依据 factor_neutralization_service.py 提炼三个入口与常量

## 2026-11-16 | 创建 | FactorReturnAnalysisService
- 依据 factor_return_analysis_service.py 提炼配置与四个分析方法

## 2026-11-16 | 创建 | WeightedICService
- 依据 weighted_ic_service.py 提炼配置、权重与重要性排序

## 2026-11-16 | 创建 | 因子中性化与收益验证链路
- 汇总中性化、分层、加权 IC 到评估回测的顺序与边界

## 2026-11-16 | 更新 | 内容目录
- 第十批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 因子衰减与IC自相关
- 依据 statistics_service.py 提炼多周期 IC 衰减、自相关与滚动稳定性口径

## 2026-11-16 | 创建 | 策略对比两两显著性检验
- 依据 strategy_comparison_service.py 提炼双检验、样本门槛与秩均值排名

## 2026-11-16 | 创建 | 滑点自适应推荐
- 依据 smart_slippage_detector.py 提炼板块基准、流动性、波动与换手惩罚

## 2026-11-16 | 创建 | StatisticsService
- 依据 statistics_service.py 提炼 14 个统计方法与常数IC守卫

## 2026-11-16 | 创建 | TearSheetService
- 依据 tear_sheet_service.py 提炼评分权重、公式与等级划分

## 2026-11-16 | 创建 | StrategyComparisonService
- 依据 strategy_comparison_service.py 提炼对比流程与缺失值处理

## 2026-11-16 | 创建 | 因子报告与策略对比链路
- 汇总报告汇总、独立统计、策略对比到导出的职责分工

## 2026-11-16 | 更新 | 内容目录
- 第十一批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 分组排序学习
- 依据 stock_ranker_service.py 提炼排序目标、分组与组内排名口径

## 2026-11-16 | 创建 | 训练期统计量防泄漏
- 依据训练切分与填充逻辑提炼组边界对齐与训练集中位数复用

## 2026-11-16 | 创建 | 因子质量分级评分
- 依据 factor_summary_service.py 提炼四项权重与七档评级

## 2026-11-16 | 创建 | StockRankerService
- 依据 stock_ranker_service.py 提炼训练配置、LRU 缓存与解释入口

## 2026-11-16 | 创建 | FactorSummaryService
- 依据 factor_summary_service.py 提炼摘要结构与质量分计算

## 2026-11-16 | 创建 | VisualizationService
- 依据 visualization_service.py 提炼八个绘图方法与降级策略

## 2026-11-16 | 创建 | 排序模型训练与预测链路
- 汇总预处理、防泄漏、排序训练、回测与解释的先后关系

## 2026-11-16 | 更新 | 内容目录
- 第十二批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 适应度目标路由与Z-Score归一化
- 依据 base_mining_service.py 提炼四种适应度目标、先验冷启动与 σ 下界保护

## 2026-11-16 | 创建 | 交叉验证过拟合惩罚
- 依据 base_mining_service.py 的分段 IC 惩罚提炼公式、采样与超时边界

## 2026-11-16 | 创建 | 树模型预筛选两阶段管道
- 依据 tree_prescreen 与 dual 服务提炼阶段划分、Top-K 与回退逻辑

## 2026-11-16 | 创建 | BaseMiningService
- 依据 base_mining_service.py 提炼构造默认值、共享缓存与生命周期方法

## 2026-11-16 | 创建 | DualMiningService
- 依据 dual_mining_service.py 提炼五种算法分派、参数分组与回退行为

## 2026-11-16 | 创建 | ModelRegistry
- 依据 model_registry.py 提炼框架/阶段枚举、版本命名与持久化规则

## 2026-11-16 | 创建 | 挖掘算法调度与模型持久化链路
- 汇总注册表、调度器、共享基类与模型版本中心的职责边界

## 2026-11-16 | 更新 | 内容目录
- 第十三批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 挖掘结果统一验证口径
- 依据 factor_analyzer.py 提炼预计算优先、IC/IR 绝对值与分数换算口径

## 2026-11-16 | 创建 | 组合因子生成与预筛
- 依据 factor_generator_service.py 提炼四类生成器、深度上限与预筛门限

## 2026-11-16 | 创建 | 混合类型Phi_K相关性
- 依据 phik_correlation_service.py 提炼分类列判定、双矩阵输出与降级行为

## 2026-11-16 | 创建 | FactorAnalyzer
- 依据 factor_analyzer.py 提炼逐条验证、超时保护与查重入库

## 2026-11-16 | 创建 | FactorGeneratorService
- 依据 factor_generator_service.py 提炼三类能力表、生成方法与预筛默认值

## 2026-11-16 | 创建 | FactorImportService
- 依据 factor_import_service.py 提炼双导入入口、默认列名与返回结构

## 2026-11-16 | 创建 | 因子生成与统一验证入库链路
- 汇总生成、导入、统一验证、版本留痕与质量评级的先后关系

## 2026-11-16 | 更新 | 内容目录
- 第十四批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 统一响应信封与错误映射
- 依据 main.py、路由与 api.ts 提炼 success/data 信封与状态码映射

## 2026-11-16 | 创建 | 挖掘任务内存表与TTL回收
- 依据 mining.py 提炼任务表、TTL、条数上限与关闭清理

## 2026-11-16 | 创建 | 按耗时分级设置请求超时
- 依据 api.ts 提炼默认 300 秒与各接口超时分档

## 2026-11-16 | 创建 | FactorFlowAPI应用
- 依据 main.py 提炼生命周期、CORS、路由前缀、SPA 回退与异常处理

## 2026-11-16 | 创建 | ApiClient
- 依据 api.ts 提炼 axios 配置、拦截器与七类方法覆盖

## 2026-11-16 | 创建 | MiningTaskRouter
- 依据 mining.py 提炼五种算法、默认参数、端点与内存常量

## 2026-11-16 | 创建 | 挖掘任务提交与轮询链路
- 汇总提交、状态轮询、结果获取、刷新接管与重启恢复

## 2026-11-16 | 更新 | 内容目录
- 第十五批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 因子版本哈希缓存失效
- 依据 analysis_service.py 提炼缓存键构造与因子代码哈希失效

## 2026-11-16 | 创建 | 持仓时长与换手口径
- 依据 position_analysis_service.py 提炼换手除二与连续区间口径

## 2026-11-16 | 创建 | 代表样本选取最长有效序列
- 依据 factor_data_utils.py 与暴露度服务提炼代表样本规则

## 2026-11-16 | 创建 | FactorCalculator
- 依据 factor_service.py 提炼 AST 校验、TALib 与麦语言函数

## 2026-11-16 | 创建 | FactorService
- 依据 factor_service.py 提炼预置加载、统计、门控与并行计算

## 2026-11-16 | 创建 | AnalysisService
- 依据 analysis_service.py 提炼固定预处理、alphalens、检测与缓存

## 2026-11-16 | 创建 | 因子计算与缓存分析链路
- 汇总代码解析、并行计算、缓存键与固定预处理的先后

## 2026-11-16 | 更新 | 内容目录
- 第十六批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 数据库会话生命周期管理
- 依据 database.py 提炼 get_db、with_db 与废弃接口的会话管理规则

## 2026-11-16 | 创建 | 向量化滚动IC
- 依据 ic_calculator.py 提炼滑动窗口排名与向量化 Pearson 实现

## 2026-11-16 | 创建 | 仓储层提交与回滚边界
- 依据 factor_repository.py 提炼写操作提交、回滚与批量删除语义

## 2026-11-16 | 创建 | FactorRepository
- 依据 factor_repository.py 提炼查询过滤、硬删除与预置保护

## 2026-11-16 | 创建 | AnalysisCacheRepository
- 依据 factor_repository.py 提炼缓存查询、写入与按天清理

## 2026-11-16 | 创建 | 因子与缓存数据模型
- 依据 models/factor.py 提炼 factors 与 analysis_cache 两表字段

## 2026-11-16 | 创建 | 因子持久化与缓存数据链路
- 汇总会话、仓储、表结构与事务边界的串联关系

## 2026-11-16 | 更新 | 内容目录
- 第十七批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | JSON安全序列化
- 依据 serialization.py 提炼 NaN/Inf 归 null 与递归类型转换规则

## 2026-11-16 | 创建 | 配置默认值与目录自举
- 依据 settings.py 提炼默认值、目录创建与样本库复制时机

## 2026-11-16 | 创建 | 面板按资产分组计算
- 依据 return_calculator.py 提炼 MultiIndex 分组与未来收益语义

## 2026-11-16 | 创建 | Settings
- 依据 settings.py 提炼路径派生、默认参数与初始化副作用

## 2026-11-16 | 创建 | SerializationUtils
- 依据 serialization.py 提炼两个对外函数与类型映射

## 2026-11-16 | 创建 | ReturnCalculator
- 依据 return_calculator.py 提炼唯一函数与面板分组方式

## 2026-11-16 | 创建 | 配置与序列化基础设施链路
- 汇总配置自举、响应清洗与收益口径三者的分工

## 2026-11-16 | 更新 | 内容目录
- 第十八批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 当前版本唯一性
- 依据 factor_version_repository.py 提炼 is_current 单标记的维护顺序

## 2026-11-16 | 创建 | 重启时中断任务回收
- 依据 mining_task_repository.py 提炼启动时批量标注 aborted 的规则

## 2026-11-16 | 创建 | 生成因子去重与保存标记
- 依据 generated_factor_repository.py 提炼表达式去重与双状态字段

## 2026-11-16 | 创建 | GeneratedFactorRepository
- 依据生成因子仓储提炼查询过滤、保存标记与字典输出

## 2026-11-16 | 创建 | MiningTaskRepository
- 依据挖掘任务仓储提炼状态流转、分页查询与摘要输出

## 2026-11-16 | 创建 | FactorVersionRepository
- 依据版本仓储提炼创建、当前版本切换与批量删除

## 2026-11-16 | 创建 | 生成因子、任务与版本持久化链路
- 汇总三类仓储的去重、状态与单当前标记职责

## 2026-11-16 | 更新 | 内容目录
- 第十九批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 日志即时刷盘与幂等初始化
- 依据 logging_config.py 与 main.py 提炼刷盘时机与重复挂载保护

## 2026-11-16 | 创建 | 先子后父删除关联记录
- 依据 backtest_repository.py 提炼两级删除顺序与事务回滚

## 2026-11-16 | 创建 | JSON字段入库前转字典
- 依据 backtest_repository.py 提炼净值与分层收益的落库前转换

## 2026-11-16 | 创建 | BacktestRepository
- 依据回测仓储提炼保存、分页、关联删除与统计方法

## 2026-11-16 | 创建 | CacheRepository
- 依据缓存仓储提炼元数据 CRUD、访问更新与过期统计

## 2026-11-16 | 创建 | 回测与交易记录模型
- 依据 backtest.py 与 cache_metadata.py 提炼三张表字段与行为方法

## 2026-11-16 | 创建 | 回测结果与缓存元数据持久化链路
- 汇总回测落库、关联删除与缓存元数据统计的分工

## 2026-11-16 | 更新 | 内容目录
- 第二十批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | GFlowNet表达式状态机
- 依据 gflownet 服务提炼节点状态、动作空间与轨迹平衡损失

## 2026-11-16 | 创建 | 符号回归复杂度与样本门槛
- 依据 pysr 服务提炼尺寸深度约束、算子默认值与最小样本

## 2026-11-16 | 创建 | 表达式相似度去重与惩罚
- 依据 gflownet 服务与 factor_primitives 提炼双阈值规则

## 2026-11-16 | 创建 | GeneticFactorMiningService
- 依据遗传规划服务提炼默认参数、工具集与选择策略

## 2026-11-16 | 创建 | PySRFactorMiningService
- 依据符号回归服务提炼参数默认值、评估路径与排序

## 2026-11-16 | 创建 | GFlowNetMiningService
- 依据 GFlowNet 服务提炼训练循环、HoF 与返回结构

## 2026-11-16 | 创建 | 三种表达式挖掘实现对比
- 汇总三种算法的搜索方式与共享验证口径

## 2026-11-16 | 更新 | 内容目录
- 第二十一批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 异常值检测双口径与处理方法
- 依据 data_preprocessing_service.py 提炼 std/IQR 判定与四种处理方式

## 2026-11-16 | 创建 | 回测薄编排层
- 依据 backtest_service.py 提炼全量委托与结果 engine 标记

## 2026-11-16 | 创建 | 隐因子序列建模
- 依据 deep_factor_mining_service.py 提炼窗口、网络与训练参数

## 2026-11-16 | 创建 | DeepFactorMiningService
- 依据深度挖掘服务提炼流程、返回结构与模型持久化

## 2026-11-16 | 创建 | TreePrescreenMiningService
- 依据树预筛服务提炼模型选择、阈值与下游算法透传

## 2026-11-16 | 创建 | DataPreprocessingService
- 依据预处理服务提炼六个公开能力与默认行为

## 2026-11-16 | 创建 | 树预筛与深度挖掘链路
- 汇总两阶段预筛、隐因子训练与共享验证口径

## 2026-11-16 | 更新 | 内容目录
- 第二十二批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 事件响应阈值分组
- 依据有效性服务提炼横截面与全局分位阈值的分组口径

## 2026-11-16 | 创建 | 时间序列分位与变异系数
- 依据暴露服务提炼分位计算、滚动窗口与变异系数口径

## 2026-11-16 | 创建 | 滚动均值断点与傅里叶周期
- 依据监测服务提炼断点阈值与 FFT 周期识别方法

## 2026-11-16 | 创建 | FactorEffectivenessService
- 依据有效性服务提炼入口、alphalens 优先级与四类结果

## 2026-11-16 | 创建 | FactorExposureService
- 依据暴露服务提炼四个入口与分布统计口径

## 2026-11-16 | 创建 | FactorMonitoringService
- 依据监测服务提炼滚动带、断点、周期与迁移矩阵

## 2026-11-16 | 创建 | 因子有效性、暴露与动态监测分工
- 汇总三个服务分别回答的预测力、位置与结构变化问题

## 2026-11-16 | 更新 | 内容目录
- 第二十三批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | IC显著性单样本t检验
- 依据增强分析服务提炼逐日 IC 序列的检验前置条件与输出

## 2026-11-16 | 创建 | 单票集中度占位实现
- 依据持仓服务提炼单标的固定返回值与简化说明

## 2026-11-16 | 创建 | 滑点成本敏感性分级
- 依据滑点检测器提炼成本公式、分档阈值与建议条件

## 2026-11-16 | 创建 | EnhancedAnalysisService
- 依据增强分析服务提炼两个入口与委托口径

## 2026-11-16 | 创建 | PositionAnalysisService
- 依据持仓服务提炼三个入口与统计字段

## 2026-11-16 | 创建 | SmartSlippageDetector
- 依据滑点检测器提炼输入形式、偏好档位与报告输出

## 2026-11-16 | 创建 | 组合持仓与交易成本评估链路
- 汇总持仓统计、集中度占位与滑点敏感性的分工

## 2026-11-16 | 更新 | 内容目录
- 第二十四批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 算子加权复杂度
- 依据 factor_primitives.py 的权重表提炼四档算子代价与终端权重

## 2026-11-16 | 创建 | Zobrist哈希同构去重
- 依据 factor_primitives.py 提炼交换律排序、位置种子与重复子树替换规则

## 2026-11-16 | 创建 | 魔法值集中管控
- 依据 constants.py 提炼市场参数、阈值、窗口与键名常量

## 2026-11-16 | 创建 | FactorPrimitives
- 依据 factor_primitives.py 提炼算子、原语集三态构造与表达式工具

## 2026-11-16 | 创建 | 全局常量模块
- 依据 constants.py 提炼常量分组与 Keys/ChineseKeys 双键名体系

## 2026-11-16 | 创建 | AnalysisRouter
- 依据 analysis.py 提炼 19 个端点、代码解析函数与依赖服务

## 2026-11-16 | 创建 | 表达式去重与简化链路
- 汇总结构指纹、代数规范键与两类惩罚的分工

## 2026-11-16 | 更新 | 内容目录
- 第二十五批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 暂存池与正式库两阶段提升
- 依据 factors.py 提升端点提炼三项准入校验与来源关联

## 2026-11-16 | 创建 | 表达式安全校验与函数包装
- 依据 factors.py 白名单校验与 promote 模板提炼包装规则

## 2026-11-16 | 创建 | 权重方法横向对比
- 依据 portfolio.py compare-methods 提炼归一加权与 IC 口径

## 2026-11-16 | 创建 | FactorRouter
- 依据 factors.py 提炼 15 个端点、复制命名与暂存池操作

## 2026-11-16 | 创建 | DataRouter
- 依据 data.py 提炼 6 个端点与 8 个预设股票池

## 2026-11-16 | 创建 | OrchestratorRouter
- 依据 orchestrator.py 提炼验证编排、排序模型与注册中心端点

## 2026-11-16 | 创建 | 因子从暂存到正式与模型晋升链路
- 汇总因子与模型两类资产的两段式晋升路径

## 2026-11-16 | 更新 | 内容目录
- 第二十六批 7 个页面写入后同步 index.md

## 2026-11-16 | 创建 | 按分位选股的对比口径
- 依据 backtest.py 对比接口提炼逐日分组、选股与年化规则

## 2026-11-16 | 创建 | 回测频率与周期解析
- 依据 backtest.py 单回测入口提炼日线/分钟线切换与归一规则

## 2026-11-16 | 创建 | 可选依赖缺失前置返回503
- 依据 check_vectorbt_available 门禁提炼状态码与提示语义

## 2026-11-16 | 创建 | BacktestRouter
- 依据 backtest.py 提炼 4 个端点、默认参数与滑点模式

## 2026-11-16 | 创建 | FactorManagementPage
- 依据 FactorManagement.tsx 提炼列表、表单校验与复制简化操作

## 2026-11-16 | 创建 | FactorDetailPage
- 依据 FactorDetail.tsx 提炼取数、IC 与八类分析面板调用

## 2026-11-16 | 创建 | 回测发起与结果落库链路
- 汇总依赖探测、取数计算、回测与历史查询的职责分工

## 2026-11-16 | 更新 | 内容目录
- 第二十七批 7 个页面写入后同步 index.md

## 2026-11-16 | 更新 | 实体页全集
- 80 个实体页统一收敛为「是什么 + 关键事实」，删除端点、方法与参数明细罗列

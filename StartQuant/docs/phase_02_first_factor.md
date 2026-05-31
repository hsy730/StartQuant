# 阶段 2：构建第一个因子 —— 跑通从数据到分析的全流程

## 目标
从零开始构建一个自定义因子，完整经历"数据处理 → 因子构建 → 有效性检验 → 报告输出"的全流程。

## 预计时长
2-3 天

## 输出物
1 个自定义因子的完整分析报告（含因子逻辑、IC分析、分层测试、风险提示）

---

## 步骤详解

### Step 1：因子构思与文献调研（0.5天）

1. **确定因子方向**
   从以下方向中选择一个作为切入点：
   - 改进现有经典因子（如改进 EP，加入盈利质量）
   - 组合多个基础因子（如价值+质量复合因子）
   - 基于量价行为的原创因子（如开盘跳空、尾盘异动）
   - 基于另类数据的因子（如分析师预期、北向资金）

2. **文献与研报参考**
   - 搜索相关券商研报（如"中金因子深度系列"、"华泰金工因子研究"）
   - 参考学术论文（如 Fama-French 五因子、Quality Minus Junk）
   - 记录核心逻辑和构建方法

3. **明确因子假设**
   - 因子预测的逻辑基础是什么？
   - 预期与收益的关系方向（正相关/负相关）？
   - 因子适用的市场环境？

---

### Step 2：单因子构建三步曲（聚宽平台）

在聚宽平台上，构建一个单因子通常分为三步：

#### 2.1 因子定义：创建一个因子类

通过继承 `Factor` 类来定义一个因子，核心是实现 `calc` 方法。

```python
from jqfactor import Factor

class MAS(Factor):
    name = 'mas'              # 因子名称，不能重复
    max_window = 5            # 需要获取过去5天的数据
    dependencies = ['close']  # 依赖的基础数据：收盘价

    def calc(self, data):
        # data 是一个字典，包含过去5天的收盘价
        # 返回每只股票的5日均值
        return data['close'][-5:].mean()
```

**关键参数说明**：

| 参数 | 含义 | 示例 |
|-----|------|------|
| `max_window` | 回头看多远的窗口 | 算年线写 250，算5日均线写 5 |
| `name` | 因子唯一名称 | 不能与其他因子重复 |
| `dependencies` | 计算需要的基础数据 | `close`, `open`, `volume`, `market_cap` 等 |

**calc 函数可以获取的数据**：

- `self._current_date`：当前逻辑时间（T日收盘后），**不会用到未来数据**
- `data`：依赖的基础数据字典

**calc 函数返回值**：

- 必须是一个 `pandas.Series`
- 索引（Index）是股票代码
- 值（Value）是算出来的因子值

#### 2.2 计算因子值：调用 calc_factors

```python
from jqfactor import calc_factors

# 定义股票池
stocks = ['000001.XSHE', '600000.XSHG']

# 计算因子值
factors_data = calc_factors(
    securities=stocks,           # 股票列表
    factors=[MAS()],             # 因子对象列表
    start_date='2023-01-01',     # 开始日期
    end_date='2023-01-31',       # 结束日期
    use_real_price=False         # 使用后复权价格
)

# 查看结果
print(factors_data['mas'].head())
```

**运行结果**：行是日期，列是股票代码，值是算出来的因子值。

#### 2.3 数据清洗（给数据"美颜"）

算出来的原始因子值通常不能直接用的原因：

- **极值干扰**：某个股票突然涨了100倍，会把平均值拉偏
- **行业偏差**：银行股的PE天生比科技股低，直接比不公平
- **量纲不同**：有的因子是0.5，有的是10000，没法一起比较

**标准处理流程（顺序不能乱）**：

```python
from jqfactor import winsorize, neutralize, standardlize

# 步骤1: 去极值（Winsorize）
# 将超过 97.5% 和低于 2.5% 分位数的极端值拉回来
clean_data = winsorize(original_data, qrange=[0.025, 0.975], axis=1)

# 步骤2: 中性化（Neutralize）
# 消除行业、市值等已知偏差，让因子更纯粹
neutral_data = neutralize(clean_data, how=['jq_l1', 'market_cap'], axis=1)

# 步骤3: 标准化（Standardize）
# 统一量纲，方便多因子合成
final_data = standardlize(neutral_data, axis=1)
```

**处理顺序：先去极值 → 再中性化 → 最后标准化**（业界标准做法）

---

### Step 3：完整实战代码（聚宽单因子生产流水线）

```python
# 导入库
import pandas as pd
from jqfactor import Factor, calc_factors, winsorize, neutralize, standardlize

# 1. 定义因子 - 包含完整数据处理流程
class MyProcessedFactor(Factor):
    name = 'my_processed_factor'
    max_window = 1
    dependencies = ['close', 'open']

    def calc(self, data):
        # 获取当日数据
        close_data = data['close'].iloc[-1]
        open_data = data['open'].iloc[-1]

        # 计算原始因子值（日内收益率）
        raw_factor = close_data / open_data - 1

        # 在因子计算阶段直接完成所有数据处理
        processed_factor = self._process_factor_data(raw_factor, self._current_date)

        return processed_factor

    def _process_factor_data(self, factor_series, current_date):
        """
        因子数据处理管道：去极值 -> 中性化 -> 标准化
        """
        # 步骤1: 去极值（分位数法）
        winsorized = winsorize(factor_series, qrange=[0.025, 0.975])

        # 步骤2: 中性化（去除行业和市值影响）
        neutralized = neutralize(
            winsorized,
            how=['jq_l1'],
            date=current_date
        )

        # 步骤3: 标准化（Z-score）
        standardized = standardlize(neutralized)

        return standardized

# 2. 准备参数
stocks = get_index_stocks('000300.XSHG')  # 拿沪深300的票
start = '2025-10-01'
end = '2025-11-26'

# 3. 计算因子值
raw_data = calc_factors(
    securities=stocks,              # 股票列表
    factors=[MyProcessedFactor()],  # 因子列表
    start_date=start,               # 开始日期
    end_date=end,                   # 结束日期
    use_real_price=True,            # 使用后复权价格
    skip_paused=False               # 包含停牌数据
)

raw_data
```

---

### Step 4：自定义因子示例（聚宽 Factor 类实现）

以下提供几个不同方向的自定义因子示例，均使用聚宽 `Factor` 类实现。

#### 示例一：改进EP因子（加入盈利质量）

```python
from jqfactor import Factor, calc_factors, winsorize, neutralize, standardlize

class EnhancedEP(Factor):
    """
    改进EP因子 = EP × 盈利质量
    盈利质量 = 经营现金流 / 净利润
    逻辑：不仅便宜，还要现金流好的公司
    """
    name = 'enhanced_ep'
    max_window = 1
    dependencies = ['pe_ratio', 'operating_cash_flow', 'net_profit']

    def calc(self, data):
        # 获取数据
        pe = data['pe_ratio'].iloc[-1]
        cf = data['operating_cash_flow'].iloc[-1]
        profit = data['net_profit'].iloc[-1]

        # 计算EP
        ep = 1.0 / pe

        # 计算盈利质量（现金流/利润）
        quality = cf / profit.abs()
        quality = quality.replace([np.inf, -np.inf], np.nan).fillna(0)
        quality = quality.clip(-10, 10)  # 限制极端值

        # 合成因子
        factor = ep * quality

        # 数据处理
        factor = winsorize(factor, qrange=[0.025, 0.975])
        factor = neutralize(factor, how=['jq_l1'], date=self._current_date)
        factor = standardlize(factor)

        return factor
```

#### 示例二：量价复合因子（开盘跳空）

```python
from jqfactor import Factor, calc_factors, winsorize, neutralize, standardlize

class GapFactor(Factor):
    """
    开盘跳空因子 = (开盘价 - 昨日收盘价) / 昨日收盘价
    逻辑：正向跳空可能反映利好信息，负向跳空反映利空
    """
    name = 'gap_factor'
    max_window = 2  # 需要昨天和今天的数据
    dependencies = ['open', 'close']

    def calc(self, data):
        # 获取数据
        open_today = data['open'].iloc[-1]
        close_yesterday = data['close'].iloc[-2]

        # 计算跳空幅度
        gap = (open_today - close_yesterday) / close_yesterday

        # 数据处理
        gap = winsorize(gap, qrange=[0.025, 0.975])
        gap = neutralize(gap, how=['jq_l1', 'market_cap'], date=self._current_date)
        gap = standardlize(gap)

        return gap
```

#### 示例三：波动率因子（20日波动率）

```python
from jqfactor import Factor, calc_factors, winsorize, neutralize, standardlize
import numpy as np

class Volatility20D(Factor):
    """
    20日波动率因子
    逻辑：低波动股票长期表现优于高波动股票（低波动异象）
    预期方向：负（低波动 → 高收益）
    """
    name = 'volatility_20d'
    max_window = 21
    dependencies = ['close']

    def calc(self, data):
        # 获取20日收盘价
        close = data['close']

        # 计算日收益率
        returns = close.pct_change().dropna()

        # 计算20日波动率（年化）
        vol = returns.std() * np.sqrt(252)

        # 取负值（低波动得分高）
        factor = -vol

        # 数据处理
        factor = winsorize(factor, qrange=[0.025, 0.975])
        factor = neutralize(factor, how=['jq_l1', 'market_cap'], date=self._current_date)
        factor = standardlize(factor)

        return factor
```

---

### Step 5：因子有效性检验（1天）

#### 5.1 IC 检验

```python
def calculate_ic(factor_df, forward_return='next_month_return'):
    """计算月度IC序列"""
    ic_list = []
    for date in factor_df['trade_date'].unique():
        截面 = factor_df[factor_df['trade_date'] == date]
        ic = 截面['factor'].corr(截面[forward_return], method='spearman')
        ic_list.append({'date': date, 'ic': ic})
    return pd.DataFrame(ic_list)
```

**输出指标**：
- IC 均值、IC 标准差
- ICIR（信息比率）
- IC 胜率（IC > 0 的月份占比）
- IC 最大回撤（连续失效期）

#### 5.2 分层回测

```python
def 分层回测(factor_df, n_groups=5):
    """
    每月按因子分n组，计算各组次月收益
    """
    results = []
    for date in factor_df['trade_date'].unique():
        截面 = factor_df[factor_df['trade_date'] == date].copy()
        截面['group'] = pd.qcut(截面['factor'], n_groups, labels=False) + 1

        for g in range(1, n_groups + 1):
            group_return = 截面[截面['group'] == g]['next_month_return'].mean()
            results.append({
                'date': date,
                'group': g,
                'return': group_return
            })
    return pd.DataFrame(results)
```

**输出指标**：
- 各组年化收益
- 各组夏普比率
- 多空对冲收益（Group5 - Group1）
- 分层单调性检验

#### 5.3 可视化输出

必须生成的图表：
1. **IC 时间序列图** —— 观察IC稳定性
2. **IC 分布直方图** —— 观察IC分布形态
3. **分组收益柱状图** —— 检验单调性
4. **多空累计收益曲线** —— 观察因子持续盈利能力
5. **分组净值曲线** —— 观察各组走势分化

---

### Step 6：撰写因子分析报告（0.5天）

#### 报告模板

```markdown
# 因子分析报告：[因子名称]

## 一、因子概述
- 因子名称：
- 因子类别（估值/质量/动量/波动/成长/其他）：
- 构建逻辑：
- 预期方向（正/负相关）：

## 二、因子构建方法
- 计算公式：
- 数据来源：
- 预处理步骤：
- 参数说明：

## 三、IC 分析
- IC 均值：
- IC 标准差：
- ICIR：
- IC 胜率：
- 最大连续失效月数：
- IC 分析结论：

## 四、分层回测结果
- 回测区间：
- 股票池：
- 分组数量：
- 各组年化收益：
- 多空年化收益：
- 多空夏普比率：
- 多空最大回撤：
- 分层单调性评价：

## 五、风险提示
- 因子失效场景：
- 极端市场环境表现：
- 与其他因子的相关性：
- 改进方向：

## 六、附录
- 图表清单
- 代码片段
- 参考文献
```

---

## 避坑指南（新人必看）

### 1. dependencies 怎么填？

聚宽数据库里有的都能填：
- 行情数据：`close` (收盘价), `open` (开盘价), `high` (最高价), `low` (最低价), `volume` (成交量)
- 估值数据：`pe_ratio` (市盈率), `pb_ratio` (市净率), `market_cap` (市值)
- 财务数据：`roe` (净资产收益率), `net_profit` (净利润) 等

### 2. skip_paused 参数

在 `calc_factors` 里有个参数叫 `skip_paused`：
- 如果是**价量因子**，建议设为 `True`
- 不然停牌期间价格没变，算出来的波动率是0，会误导模型

### 3. 清洗顺序

**先去极值，再中性化，最后标准化**。这个顺序不要乱，这是业界的标准做法。

### 4. max_window 设置

- `max_window` 决定了 `calc` 函数中 `data` 里有多少天的数据
- 算5日均线写 5，算20日动量写 21，算年线写 250
- 宁大勿小，写小了会报错

### 5. 返回值必须是 pandas.Series

`calc` 函数的返回值必须是一个 `pandas.Series`，索引是股票代码。

### 6. 避免未来函数

- `self._current_date` 是 T日收盘后的时间
- `data` 里的数据都是 T日及之前的数据
- **千万不要用 T+1 的数据来计算 T日的因子值**

---

## 检查清单

- [ ] 因子方向与假设已明确
- [ ] 数据清洗完成，无缺失/异常
- [ ] Factor 类编写完成（name, max_window, dependencies, calc）
- [ ] calc_factors 调用成功，返回正确格式
- [ ] 因子预处理（去极值→中性化→标准化）完成
- [ ] IC 分析完成，统计量已记录
- [ ] 分层回测完成，5组收益已计算
- [ ] 5张核心图表已生成
- [ ] 因子分析报告撰写完成

---

## 常见问题

**Q1：因子构建后IC为负怎么办？**
> 检查因子方向假设是否正确。如果逻辑是"高因子值对应高收益"但IC为负，尝试取反（-1 × 因子值）再测试。

**Q2：分层不单调怎么办？**
> 1. 检查因子与收益是否非线性，尝试分位数变换
> 2. 检查是否存在极端值干扰，加强去极值
> 3. 考虑因子是否需与其他因子组合使用

**Q3：IC很高但回测收益一般？**
> 可能是换手率过高导致交易成本侵蚀收益，或因子集中在小市值股票（流动性差）。需加入换手率分析和市值分布分析。

**Q4：Factor 类和传统 API 方式有什么区别？**
> Factor 类是聚宽推荐的新方式，自动处理数据获取和窗口管理，代码更简洁。传统 API 方式更灵活，适合复杂逻辑。

**Q5：中性化一定要做吗？**
> 不是必须的。如果因子本身已经比较稳定，可以跳过。但对于估值类因子（如EP、BP），建议做行业中性化，因为不同行业的估值水平差异很大。

# 阶段 1：因子看板游览 —— 建立因子"直觉"

## 目标
通过浏览和观察现有高IC均值因子，建立对因子形态、分布和预测能力的直观认知，为后续自定义因子开发奠定基础。

## 预计时长
1-2 天

## 输出物
3 个高IC均值因子的观察记录（包含因子逻辑、IC表现、分层特征、直观感受）

---

## 步骤详解

### Step 1：环境准备与数据接入（0.5天）

1. **启动量化研究环境**
   - 打开 Jupyter Notebook / JupyterLab
   - 加载常用库：`pandas`, `numpy`, `matplotlib`, `seaborn`
   - 聚宽平台用户额外加载：`from jqfactor import Factor, calc_factors, winsorize, neutralize, standardlize`
   - 确认数据接口可用（股票日线、财务数据、行业分类等）

2. **数据范围设定**
   - 股票池：全A股（剔除ST、停牌、上市不足60日）
   - 时间范围：近 3-5 年（如 2020-01-01 至 2024-12-31）
   - 频率：月度（每月末截面数据）

3. **工具函数准备**
   - 准备 IC 计算函数（Spearman 秩相关系数）
   - 准备分层回测函数（按因子值分5组/10组）
   - 准备可视化模板（IC时间序列、分组收益、累计收益图）

---

### Step 2：什么是因子？

在量化投资中，**因子**可以理解为一种"特征"或"指标"，用来衡量股票的某种属性：

- **市盈率（PE）**：衡量公司估值高低
- **动量因子**：反映股票过去一段时间的涨跌趋势
- **市值因子**：反映公司规模大小

因子的目标很简单：**找到对股票未来收益有预测能力的特征，构建投资组合**。

简单来说，因子就是从一大堆数据里提炼出的一个数值，用来给股票打分。**分高买入，分低卖出**。

---

### Step 3：单因子构建三步曲（聚宽平台）

在聚宽平台上，构建一个单因子通常分为三步：

#### 3.1 因子定义：创建一个因子类

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

#### 3.2 计算因子值：调用 calc_factors

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

#### 3.3 数据清洗（给数据"美颜"）

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

### Step 4：完整实战代码（聚宽单因子生产流水线）

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

### Step 5：选取高IC均值因子进行观察（1天）

从以下经典因子类别中，挑选 **3 个** 高IC均值因子进行深度观察：

| 因子类别 | 示例因子 | 预期IC方向 |
|---------|---------|-----------|
| 估值类 | EP（市盈率倒数）、BP（市净率倒数）、SP | 正 |
| 质量类 | ROE、ROA、毛利率 | 正 |
| 动量类 | 20日收益率、60日收益率 | 正/负 |
| 波动类 | 20日波动率、20日振幅 | 负 |
| 流动性 | 20日换手率、20日成交额 | 负 |
| 成长类 | 营收增长率、净利润增长率 | 正 |

**建议组合**：
- 1 个估值因子（如 EP）
- 1 个质量/成长因子（如 ROE）
- 1 个量价因子（如 20日动量或波动率）

---

### Step 6：高IC因子推荐与聚宽实现

以下推荐 **3 个经过验证的高IC因子**，并附上在 **聚宽（JoinQuant）** 平台的完整实现代码。

#### 因子一：EP（市盈率倒数，Earnings-to-Price）

| 属性 | 说明 |
|-----|------|
| 因子逻辑 | 市盈率倒数，反映单位市值的盈利能力。EP越高，股票越"便宜" |
| 预期IC方向 | 正（高EP → 高收益） |
| 历史IC均值 | 约 0.04 ~ 0.06 |
| 适用市场 | 震荡市、价值风格占优时表现更佳 |

**聚宽 Factor 类实现**：

```python
from jqfactor import Factor, calc_factors, winsorize, neutralize, standardlize

class EP(Factor):
    """
    EP因子（市盈率倒数）
    使用聚宽 Factor 类实现，自动处理数据获取
    """
    name = 'ep'
    max_window = 1
    dependencies = ['pe_ratio']  # 依赖聚宽内置的市盈率数据

    def calc(self, data):
        # 获取当日PE
        pe = data['pe_ratio'].iloc[-1]

        # EP = 1 / PE
        ep = 1.0 / pe

        # 去极值
        ep = winsorize(ep, qrange=[0.025, 0.975])

        # 中性化（去除行业影响）
        ep = neutralize(ep, how=['jq_l1'], date=self._current_date)

        # 标准化
        ep = standardlize(ep)

        return ep

# 使用 calc_factors 计算
stocks = get_index_stocks('000300.XSHG')
ep_data = calc_factors(
    securities=stocks,
    factors=[EP()],
    start_date='2023-01-01',
    end_date='2023-12-31',
    use_real_price=True
)
print(ep_data['ep'].head())
```

**传统 API 实现（备用）**：

```python
from jqdata import *

def get_ep_factor(stock_list, date):
    """使用传统 API 计算EP因子"""
    pe = get_valuation(stock_list, end_date=date, fields=['pe_ratio'])
    pe = pe.set_index('code')['pe_ratio']
    ep = 1.0 / pe
    ep = ep.replace([np.inf, -np.inf], np.nan).dropna()
    ep = ep.clip(ep.quantile(0.01), ep.quantile(0.99))
    return (ep - ep.mean()) / ep.std()
```

---

#### 因子二：ROE（净资产收益率）

| 属性 | 说明 |
|-----|------|
| 因子逻辑 | 净利润/净资产，衡量股东权益的回报效率 |
| 预期IC方向 | 正（高ROE → 高收益） |
| 历史IC均值 | 约 0.03 ~ 0.05 |
| 适用市场 | 牛市、质量风格占优时表现更佳 |

**聚宽 Factor 类实现**：

```python
from jqfactor import Factor, calc_factors, winsorize, neutralize, standardlize

class ROE(Factor):
    """
    ROE因子（净资产收益率）
    使用聚宽财务数据 indicator.roe
    """
    name = 'roe'
    max_window = 1
    dependencies = ['roe']  # 依赖聚宽内置的ROE数据

    def calc(self, data):
        # 获取当日ROE
        roe = data['roe'].iloc[-1]

        # 去极值
        roe = winsorize(roe, qrange=[0.025, 0.975])

        # 中性化
        roe = neutralize(roe, how=['jq_l1'], date=self._current_date)

        # 标准化
        roe = standardlize(roe)

        return roe

# 计算
roe_data = calc_factors(
    securities=get_index_stocks('000300.XSHG'),
    factors=[ROE()],
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

---

#### 因子三：Momentum_20D（20日动量）

| 属性 | 说明 |
|-----|------|
| 因子逻辑 | 过去20个交易日收益率，反映短期趋势 |
| 预期IC方向 | 正（高收益 → 未来高收益） |
| 历史IC均值 | 约 0.02 ~ 0.04 |
| 适用市场 | 趋势明确的市场 |
| 注意 | A股长期存在反转效应，20日动量属于短期动量 |

**聚宽 Factor 类实现**：

```python
from jqfactor import Factor, calc_factors, winsorize, neutralize, standardlize

class Momentum20D(Factor):
    """
    20日动量因子
    使用收盘价计算20日收益率
    """
    name = 'momentum_20d'
    max_window = 21           # 需要21天数据算20日收益
    dependencies = ['close']  # 依赖收盘价

    def calc(self, data):
        # 获取收盘价序列
        close = data['close']

        # 计算20日动量 = (今日收盘价 / 20日前收盘价) - 1
        momentum = close.iloc[-1] / close.iloc[0] - 1

        # 去极值
        momentum = winsorize(momentum, qrange=[0.025, 0.975])

        # 中性化（去除市值和行业影响）
        momentum = neutralize(
            momentum,
            how=['jq_l1', 'market_cap'],
            date=self._current_date
        )

        # 标准化
        momentum = standardlize(momentum)

        return momentum

# 计算
mom_data = calc_factors(
    securities=get_index_stocks('000300.XSHG'),
    factors=[Momentum20D()],
    start_date='2023-01-01',
    end_date='2023-12-31',
    use_real_price=True
)
```

---

### Step 7：批量IC分析（研究模块代码）

在聚宽**研究模块**中运行以下代码，批量计算三个因子的IC：

```python
import numpy as np
import pandas as pd
from jqdata import *
from jqfactor import Factor, calc_factors, winsorize, neutralize, standardlize
import matplotlib.pyplot as plt

# ========== 定义三个因子类 ==========

class EP(Factor):
    name = 'ep'
    max_window = 1
    dependencies = ['pe_ratio']

    def calc(self, data):
        ep = 1.0 / data['pe_ratio'].iloc[-1]
        ep = winsorize(ep, qrange=[0.025, 0.975])
        ep = neutralize(ep, how=['jq_l1'], date=self._current_date)
        return standardlize(ep)

class ROE(Factor):
    name = 'roe'
    max_window = 1
    dependencies = ['roe']

    def calc(self, data):
        roe = data['roe'].iloc[-1]
        roe = winsorize(roe, qrange=[0.025, 0.975])
        roe = neutralize(roe, how=['jq_l1'], date=self._current_date)
        return standardlize(roe)

class Momentum20D(Factor):
    name = 'momentum_20d'
    max_window = 21
    dependencies = ['close']

    def calc(self, data):
        mom = data['close'].iloc[-1] / data['close'].iloc[0] - 1
        mom = winsorize(mom, qrange=[0.025, 0.975])
        mom = neutralize(mom, how=['jq_l1', 'market_cap'], date=self._current_date)
        return standardlize(mom)

# ========== 计算月度IC ==========

def calc_monthly_ic(factor_class, factor_name, month_ends, stock_pool='000300.XSHG'):
    """计算某个因子的月度IC序列"""
    ic_list = []

    for i in range(len(month_ends) - 1):
        date = month_ends[i]
        next_date = month_ends[i + 1]
        date_str = date.strftime('%Y-%m-%d')

        # 获取股票池
        stocks = get_index_stocks(stock_pool)

        # 基础过滤：剔除ST
        st_info = get_extras('is_st', stocks, end_date=date_str, count=1)
        stocks = [s for s in stocks if not st_info[s][0]]

        try:
            # 计算因子值
            factor_data = calc_factors(
                securities=stocks,
                factors=[factor_class()],
                start_date=date_str,
                end_date=date_str,
                use_real_price=True
            )
            factor_values = factor_data[factor_name].iloc[0]

            # 获取次月收益
            next_returns = {}
            for stock in factor_values.index:
                try:
                    p = get_price(stock, start_date=date, end_date=next_date,
                                  frequency='daily', fields=['close'], skip_paused=False)
                    if len(p) >= 2:
                        next_returns[stock] = p['close'].iloc[-1] / p['close'].iloc[0] - 1
                except:
                    continue

            # 计算IC
            ret_series = pd.Series(next_returns)
            common = factor_values.index.intersection(ret_series.index)
            if len(common) > 20:
                ic = factor_values[common].corr(ret_series[common], method='spearman')
                ic_list.append({'date': date_str, 'ic': ic})
        except:
            continue

    ic_df = pd.DataFrame(ic_list)
    if len(ic_df) == 0:
        return None

    ic_df['date'] = pd.to_datetime(ic_df['date'])
    ic_df = ic_df.set_index('date')

    # 统计输出
    print(f"\n===== {factor_name.upper()} 因子IC分析 =====")
    print(f"IC均值: {ic_df['ic'].mean():.4f}")
    print(f"IC标准差: {ic_df['ic'].std():.4f}")
    print(f"ICIR: {ic_df['ic'].mean() / ic_df['ic'].std():.4f}")
    print(f"IC胜率: {(ic_df['ic'] > 0).mean():.2%}")

    return ic_df

# ========== 执行分析 ==========

# 获取每月最后一个交易日
date_list = get_trade_days(start_date='2020-01-01', end_date='2024-12-31')
month_ends = []
for i in range(len(date_list) - 1):
    if date_list[i].month != date_list[i+1].month:
        month_ends.append(date_list[i])
month_ends.append(date_list[-1])

# 计算三个因子的IC
ic_ep = calc_monthly_ic(EP, 'ep', month_ends)
ic_roe = calc_monthly_ic(ROE, 'roe', month_ends)
ic_mom = calc_monthly_ic(Momentum20D, 'momentum_20d', month_ends)

# 可视化
plt.figure(figsize=(14, 8))
if ic_ep is not None:
    plt.plot(ic_ep.index, ic_ep['ic'], label='EP', alpha=0.7)
if ic_roe is not None:
    plt.plot(ic_roe.index, ic_roe['ic'], label='ROE', alpha=0.7)
if ic_mom is not None:
    plt.plot(ic_mom.index, ic_mom['ic'], label='Momentum_20D', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.title('月度IC时间序列对比')
plt.xlabel('日期')
plt.ylabel('IC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Step 8：三个因子预期表现对比

| 因子 | EP | ROE | Momentum_20D |
|-----|-----|-----|-------------|
| **因子类别** | 估值 | 质量 | 动量 |
| **核心逻辑** | 便宜的好公司 | 盈利能力强的公司 | 趋势延续 |
| **预期IC均值** | ~0.05 | ~0.04 | ~0.03 |
| **IC稳定性** | 中等 | 较高 | 较低 |
| **失效场景** | 成长风格极端占优 | 盈利陷阱 | 震荡市、反转行情 |
| **最佳市场** | 价值风格、震荡市 | 牛市、质量风格 | 趋势市 |
| **聚宽dependencies** | `pe_ratio` | `roe` | `close` |

---

### Step 9：逐个因子深度观察（每个因子约2-3小时）

#### 9.1 因子构建与截面分布

1. **构建因子值**
   ```python
   # 使用 calc_factors 获取因子值
   factor_df = calc_factors(securities=stocks, factors=[EP()], ...)
   ```

2. **截面分布观察**
   - 绘制某月因子值的直方图 + KDE 曲线
   - 观察：是否近似正态？有无极端 outliers？
   - 记录：偏度（Skewness）、峰度（Kurtosis）

3. **行业/市值中性化对比**
   - 对比中性化前后的分布变化
   - 观察：中性化是否消除了行业/市值偏差？

#### 9.2 IC 分析

1. **计算月度IC序列**
   ```python
   # 每月末计算因子值与次月收益的 Spearman 相关系数
   ic_series = monthly_ic(factor_df, forward_return='next_month_return')
   ```

2. **IC 统计量**
   - IC 均值（Mean IC）
   - IC 标准差（Std IC）
   - ICIR = Mean IC / Std IC
   - IC > 0 的占比（胜率）

3. **IC 时间序列可视化**
   - 绘制 IC 逐月变化图
   - 标注 IC 显著为正/负的月份
   - 观察：是否存在持续性？是否有明显的失效期？

#### 9.3 分层测试

1. **分组方法**
   - 每月末按因子值将股票分为 5 组（或 10 组）
   - 组 1：因子值最低（空头组）
   - 组 5：因子值最高（多头组）

2. **分层收益观察**
   - 计算每组次月的平均收益
   - 绘制分组收益柱状图（是否单调？）
   - 绘制多空对冲（Group5 - Group1）累计收益曲线

3. **关键观察点**
   - 单调性：Group1 < Group2 < Group3 < Group4 < Group5？
   - 多空收益：是否稳定向上？最大回撤多少？
   - 多头超额：相对基准的累计超额收益如何？

#### 9.4 记录观察笔记

为每个因子填写以下观察记录模板：

```
因子名称：
因子逻辑：
IC 均值：
ICIR：
IC 胜率：
分布特征：
分层单调性：
多空累计收益：
失效期观察：
直观感受（3句话）：
```

---

### Step 10：跨因子对比与总结（0.5天）

1. **制作对比表格**

| 指标 | 因子A | 因子B | 因子C |
|-----|------|------|------|
| IC均值 | | | |
| ICIR | | | |
| IC胜率 | | | |
| 分层单调性 | | | |
| 多空夏普 | | | |
| 最大回撤 | | | |

2. **提炼关键洞察**
   - 哪类因子IC最稳定？
   - 哪类因子在牛市/熊市表现不同？
   - 因子失效的常见场景是什么？

3. **形成因子直觉**
   - 一个好的因子应该具备哪些特征？
   - 因子与收益的关系是线性的还是非线性的？
   - 为什么有些因子IC高但实盘效果差？

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

---

## 检查清单

- [ ] 环境搭建完成，数据接口可用
- [ ] 理解 Factor 类的三个核心参数（name, max_window, dependencies）
- [ ] 掌握 calc_factors 的使用方法
- [ ] 掌握数据清洗三步曲（去极值→中性化→标准化）
- [ ] 3 个高IC因子已选定并构建
- [ ] 每个因子完成：分布观察、IC分析、分层测试
- [ ] 3 份观察记录填写完整
- [ ] 跨因子对比表格完成
- [ ] 形成个人对"好因子"的初步判断标准

---

## 常见问题

**Q1：IC 均值多少算"高"？**
> 一般而言，|IC均值| > 0.03 可认为有一定预测能力，> 0.05 属于较强因子。但需结合ICIR和胜率综合判断。

**Q2：为什么我的因子IC很高但分层不单调？**
> 可能是因子与收益呈非线性关系，或存在极端值影响。尝试对因子进行分位数变换（rank）或去除极值（winsorize）。

**Q3：需要看多少年的数据？**
> 至少覆盖一个完整的牛熊周期（3-5年），才能观察因子在不同市场环境下的稳定性。

**Q4：Factor 类和传统 API 方式有什么区别？**
> Factor 类是聚宽推荐的新方式，自动处理数据获取和窗口管理，代码更简洁。传统 API 方式更灵活，适合复杂逻辑。

**Q5：中性化一定要做吗？**
> 不是必须的。如果因子本身已经比较稳定，可以跳过。但对于估值类因子（如EP、BP），建议做行业中性化，因为不同行业的估值水平差异很大。

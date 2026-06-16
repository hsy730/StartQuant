# FactorHub 代码坏味道防范规范

> 基于 2026-06-07 ~ 06-10 代码审查中发现的系统性坏味道提炼。
> 违反即埋雷。

---

## 项目概览

### 简介
FactorHub（又名 FactorFlow）是一款量化因子研究平台，支持因子管理、智能挖掘、多维度分析、回测验证与组合优化。

### 技术栈
| 层 | 技术 |
|---|------|
| 后端 | Python 3.12 + FastAPI + SQLAlchemy + SQLite |
| 前端 | React 18 + Ant Design 5 + TypeScript + Vite |
| 计算 | pandas + numpy + empyrical + scipy + vectorbt |
| 日志 | Python logging + RotatingFileHandler |

### 核心模块
| 目录 | 职责 |
|------|------|
| `backend/api/routers/` | REST API 路由（factors / analysis / mining / portfolio / backtest / data / preprocessing） |
| `backend/services/` | 业务服务层（40+ 个服务，含因子分析、挖掘、回测、风险指标等） |
| `backend/models/` | SQLAlchemy ORM 模型（factors / generated_factors / mining_tasks 等） |
| `backend/repositories/` | 数据访问层 |
| `backend/utils/` | 工具函数（`safe_math.py` / `ic_calculator.py` / `serialization.py` 等） |
| `backend/strategies/` | 策略模板（等权 / 市值 / 动量 / 均值回归） |
| `frontend/react-antd/` | React 前端（因子管理、挖掘、分析、回测页面） |

### 主要业务流程
1. **因子管理**：因子库 CRUD、表达式/代码双模式、版本控制
2. **因子挖掘**：遗传算法、GFlowNet、PySR、深度学习、双轨策略等多算法
3. **因子分析**：IC/IR、衰减、暴露、归因、稳定性、前视偏差检测、分位数收益
4. **回测**：vectorbt 回测、组合分析、TearSheet、滑点检测
5. **组合优化**：权重优化（IC/IR/等权/最大夏普/最小方差/风险平价）

### 启动方式
```bash
# 仅后端 API
python start_api.py          # → http://localhost:8000

# 后端 + 前端（完整服务）
python start_all.py          # 自动打开浏览器
```

### 日志
- **存储位置**：`logs/factorhub.log`（项目根目录下）
- **格式**：`2024-01-01 12:00:00 [INFO] module.name: message`
- **轮转策略**：单文件 10MB，保留 5 个备份
- **级别**：DEBUG（文件）/ INFO（控制台）/ WARNING（第三方库降噪）

---

## 6条基础规范（必遵守）

| # | 规范 | 正确做法 | 错误做法 |
|---|------|---------|---------|
| 1 | **安全除法** | `safe_divide(a, b, default=None)` | `a / b`、 `a / (b + 1e-10)`、 `a / b if b != 0 else 0.0` |
| 2 | **单一真相源** | 风险指标→`risk_metrics.py`；权重→`WeightOptimizer`；IC→`ic_calculator` | 各服务自行实现、绕过统一入口 |
| 3 | **输入不可变** | 服务层入口 `df = df.copy()` | 直接修改传入参数 |
| 4 | **日志纪律** | `logger.warning(f"xxx: {e}")` | `print()`、`except: pass` |
| 5 | **代码复用** | 相同逻辑≥2次→提取公共方法 | 复制粘贴 |
| 6 | **禁止全局副作用** | `with warnings.catch_warnings():` 局部抑制 | 模块级 `warnings.filterwarnings("ignore")` |

**基础规范检查清单**
- [ ] 所有 `/` 用 `safe_divide`，IR 用 `safe_ir`
- [ ] 新增指标走统一入口（`risk_metrics`、`WeightOptimizer`、`ic_calculator`）
- [ ] 服务层入口 `.copy()` 传入数据
- [ ] 无 `print()`、无 `except.*: pass`
- [ ] 相同逻辑提取公共方法
- [ ] warnings 抑制用上下文管理器

---

## 金融计算语义（43条速查表）

### IC / 相关
| # | 规则 | 正确 | 错误 |
|---|------|------|------|
| 7.1 | IC 必须横截面 Spearman | 逐日期 `spearmanr(group["f"], group["r"])` 再取均值 | 池化全量数据、用 Pearson |
| 7.12 | IC/IR 验证必须用 Spearman | `spearmanr(factor, return)` | `.corr()`（默认 Pearson） |
| 7.13 | IC 加权必须横截面 | MultiIndex 按 `level=0` groupby 逐日算 IC | 池化 `spearmanr(all_f, all_r)` |
| 7.14 | IC 计算器默认 Spearman | `method="spearman"` | `method="pearson"` |
| 7.30 | 滚动 IC 用统一入口 | `calculate_rolling_ic(..., method="spearman")` | 各服务自行实现 |
| 7.31 | API 路由层也用 Spearman | 调用 `calculate_rolling_ic` 或 `spearmanr` | `.rolling().corr()`（Pearson） |
| 7.4 | `spearmanr` 前清理 NaN | `valid = x.notna() & y.notna(); spearmanr(x[valid], y[valid])` | 直接传入含 NaN 数据 |
| 7.15 | IC_std≈0 且 mean≠0 → t=inf | `if abs(ic_mean) > 1e-10: t_stat = float('inf')` | `t_stat = 0.0, p_value = 1.0` |

### IR
| # | 规则 | 正确 | 错误 |
|---|------|------|------|
| 7.10 | IR 不可计算返回 None | `safe_ir(ic_mean, ic_std, default=None)` | `default=0.0`（好因子被拒） |
| 7.11 | 滚动 IR default=None | `safe_divide(mean, std, default=None)` | `default=np.nan` |

### 风险 / 收益
| # | 规则 | 正确 | 错误 |
|---|------|------|------|
| 7.5 | 组合方差 sqrt 前截断 | `np.sqrt(max(0.0, variance))` | `np.sqrt(variance)`（可能负数→NaN） |
| 7.6 | 零标准差用 `< 1e-10` | `if np.std(returns) < 1e-10:` | `if np.std(returns) == 0:` |
| 7.27 | 零标准差仍返回可计算指标 | `total_return`、`annual_return`、`win_rate` 正常返回 | 全部返回 None |
| 7.32 | 年化用几何复利 | `empyrical.annual_return(returns)` | `(1 + mean) ** 252 - 1` |
| 7.21 | 多期 Sharpe 调整频率 | 先转日频等价再算 | 直接乘 sqrt(252) |
| 7.7 | 因子收益率 NaN 禁止填 0 | `dropna()` 或 `ffill().dropna()` | `fillna(0.0)` |
| 7.34 | t 统计量 se 不过 safe_divide | `float(mean_ic) / float(se)`（se 合法可极小） | `safe_divide(mean_ic, se, default=0.0)`（吞掉大 t） |
| 7.16 | Welch's t 检验无交叉项 | `se = sqrt(std_top²/n_top + std_bottom²/n_bottom)` | `spread_std * sqrt(1/n_top + 1/n_bottom)` |
| 7.29 | Alpha/Beta 走统一入口 | `calculate_relative_metrics()` | 手动 `np.cov` |

### 评分 / 截断
| # | 规则 | 正确 | 错误 |
|---|------|------|------|
| 7.23 | 评分截断 [0, 100] | `max(0, min(score, 100))` | 负分或超 100 |
| 7.38 | 总分也截断 [0, 100] | 每维度和总分都截断 | 只截断单维度 |
| 7.35 | CV 分母用 abs(mean) | `safe_divide(std, abs(mean))` | `safe_divide(std, mean)`（负 CV） |
| 7.8 | default 符合语义 | 衰减率 `default=float('inf')`；不可计算 `default=None` | `default=0.0` 语义不匹配 |
| 7.39 | 收益为零衰减率=inf | `safe_divide(cost, return, default=float('inf'))` | `if return > 0 else 0` |
| 7.18 | R² 在 ss_tot=0 时=None | `if ss_tot < 1e-10: r_squared = None` | `1.0 - safe_divide(..., default=0.0)` → R²=1.0 |

### 数据 / 索引
| # | 规则 | 正确 | 错误 |
|---|------|------|------|
| 7.2 | 换手率横截面分位数 | `pd.cut(rank(pct=True), bins=5)` | 时序 `rolling().rank(pct=True)` |
| 7.9 | MultiIndex groupby 用 transform | `groupby(level=1).transform(...)` | `apply(...).droplevel(0)` |
| 7.17 | 因子归因横截面分位数 | 逐日期 `group["factor"].quantile(0.7)` | 全局 `all_factor.quantile(0.7)` |
| 7.22 | Duplicate DatetimeIndex 用 .iloc | `reset_index(drop=True); .iloc[group_idx.index]` | `.loc[duplicate_dates]`（样本膨胀） |
| 7.28 | Bootstrap 按日期聚类 | `np.random.choice(unique_dates, replace=True)` | `df.sample(replace=True)`（破坏截面结构） |
| 7.19 | pyportfolioopt 输入尺度无关 | 标准化后再 `diff()` | 原始因子值直接 `diff()` |

### 累计 / 复利
| # | 规则 | 正确 | 错误 |
|---|------|------|------|
| 7.20 | 累计收益必须复利 | `cum *= (1 + ret); append(cum - 1)` | 存储期间收益但标签"累计" |
| 7.3 | Fisher z 方法一致 | `z_val = mean(daily_z); z_se = std(daily_z)/sqrt(n)` | `z_val = arctanh(avg_corr)` + `z_se = std(daily_z)/sqrt(n)` 混用 |

### None 安全 / API
| # | 规则 | 正确 | 错误 |
|---|------|------|------|
| 7.36 | 禁止 `float(None)` | `_safe_float(val, default=0.0)` 先检查 None | `float(stats.get("IR", 0))` |
| 7.26 | 禁止 `or 0.0` | 让 None 传播到 `sanitize_dict` | `calculate_sharpe(...) or 0.0` |
| 7.41 | `dict.get` 值为 None 时不生效 | `if val is not None else default` | `dict.get("ir", 0)`（键存在值为 None 时返回 None） |
| 7.43 | f-string 格式化前检查 None | `f"{v:.4f}" if v is not None else "N/A"` | `f"{None:.4f}"` → TypeError |
| 7.24 | API 调用核对签名 | 匹配方法签名和返回值类型 | 传不存在的参数被 broad except 吞掉 |
| 7.42 | 循环变量不泄漏到循环外 | 从聚合结果列表中取均值 | 循环结束后用最后一次迭代的值 |

### 公式编译 / 特定场景
| # | 规则 | 正确 | 错误 |
|---|------|------|------|
| 7.37 | 公式编译除法用 safe_series_divide | `safe_series_divide(left, right)` | `left / right` |
| 7.25 | 前视偏差检测器用 Spearman | `spearmanr(f, r)` | `.corr()`（Pearson） |
| 7.33 | IC 加权用股票收益率 | `calculate_weights(..., returns=stock_returns)` | `returns=factor_returns.mean(axis=1)` |
| 7.40 | 浮点 `== 0` 改 `< 1e-10` | `if std < 1e-10:` | `if std == 0:` |

---

## PR 审查速查（7大类）

| 类别 | 检查点 |
|------|--------|
| **除法安全** | `/` → `safe_divide`；IR → `safe_ir`；无 `+1e-10` hack；default 语义正确 |
| **统一入口** | 新指标走 `risk_metrics.py`；权重走 `WeightOptimizer`；不可计算→None |
| **数据安全** | 服务层入口 `.copy()`；不直接修改传入 DataFrame |
| **日志规范** | 无 `print()`；无 `except: pass`；异常有日志 |
| **代码复用** | 相同逻辑≥2次→提取公共方法 |
| **全局副作用** | warnings 局部化；无模块级 `filterwarnings` |
| **金融语义** | IC=横截面Spearman；IR不可计算=None；评分[0,100]；零std仍返回可计算指标；NaN不填0；年化几何复利；f-string前检查None |

---

**最后更新**: 2026-06-16
**维护者**: FactorHub Core Team
**适用版本**: v1.0.0+

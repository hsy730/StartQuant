---
标题: FactorAttributionService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_attribution_service.py]
---

## 是什么

`backend/services/factor_attribution_service.py` 中的因子归因服务，把收益拆成因子贡献与 alpha/beta 两部分。

## 关键事实

- 单一归因入口，内部拆为贡献度计算、alpha/beta 分解与收益分解三段。
- 未来收益由收盘价前移构造，贡献度基于横截面 Spearman IC 计算。
- 分组阈值取逐日分位数，输出高暴露、低暴露与多空收益及贡献比例。
- 日 IC 样本过少时返回数据不足提示，多于一日时给出 IC 显著性检验。
- 基准序列由内部取数方法提供，归因口径见 [[横截面分位归因]]，IC 口径见 [[横截面SpearmanIC]]。

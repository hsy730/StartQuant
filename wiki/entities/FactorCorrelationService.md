---
标题: FactorCorrelationService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_correlation_service.py, backend/constants.py]
---

## 是什么

`backend/services/factor_correlation_service.py` 中的因子相关性分析服务，围绕因子面板给出横截面与时序相关性、滚动稳定性、显著性与共线性结论。

## 关键事实

- 输入约定为日期乘资产的 MultiIndex 面板，入口先复制，不就地修改调用方数据。
- 处理顺序固定：先做数据质量检查与预处理，再算相关性与稳定性，最后做显著性与共线性判断并给出解读。
- 预处理会统一各因子频率并对齐，缺失率过高的列剔除，插补后还要校验分布是否被扭曲。
- 相关性口径见 [[相关性口径分离]]，共线性筛查口径见 [[多重共线性VIF筛查]]。
- 除主分析外还提供均方根相关与混合类型列的相关性入口。

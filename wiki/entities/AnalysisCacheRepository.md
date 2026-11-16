---
标题: AnalysisCacheRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/factor_repository.py]
---

## 是什么

`backend/repositories/factor_repository.py` 中的分析结果缓存表数据访问类，与因子仓储同文件但职责独立。

## 关键事实

- 只围绕分析缓存表做单条记录的读写与过期清理，不承载业务计算。
- 以唯一缓存键作为查询与定位依据。
- 写操作在方法内自行提交，未实现异常回滚。
- 过期清理按记录创建时间与给定天数比较，返回删除行数。
- 缓存键的构造与失效规则见 [[缓存键与TTL失效]]。

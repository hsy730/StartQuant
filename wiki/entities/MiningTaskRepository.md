---
标题: MiningTaskRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/mining_task_repository.py]
---

## 是什么

`backend/repositories/mining_task_repository.py` 中的挖掘任务表数据访问类，负责任务状态、进度与结果的持久化。

## 关键事实

- 同时支持按业务任务号与主键查询。
- 可查活跃任务（待执行与运行中）与分页历史记录，并返回总数。
- 状态更新只写入模型已存在的属性，异常时回滚。
- 进度与完成写入会从结果中提取最优与平均适应度等信息。
- 启动时可批量回收遗留的中断任务，见 [[重启时中断任务回收]]。

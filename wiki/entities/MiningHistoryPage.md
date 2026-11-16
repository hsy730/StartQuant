---
标题: MiningHistoryPage
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/pages/MiningHistory.tsx]
---

## 是什么

`frontend/react-antd/src/pages/MiningHistory.tsx` 中的挖掘历史页面，以表格分页展示历史任务，并提供详情查看、删除与因子入库。

## 关键事实

- 列表、详情与删除分别调用 `getMiningHistory`、`getMiningHistoryDetail`、`deleteMiningHistory`。
- 状态用统一的配置表映射颜色、图标与中文标签，未知状态原样显示。
- 进度条按状态着色：完成显示成功，失败与中止显示异常，其余显示进行中。
- 详情中可直接把候选因子保存到自定义因子库。
- 接口封装来自 [[ApiClient]]，数据来源为 [[MiningTaskRepository]]。

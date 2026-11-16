---
标题: FactorManagementPage
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/pages/FactorManagement.tsx]
---

## 是什么

`frontend/react-antd/src/pages/FactorManagement.tsx` 中的因子管理页面，以表格为主体承载因子的浏览、创建、编辑、删除、复制与简化。

## 关键事实

- 数据来源为 `api.getFactors`，表格使用 antd 的列配置与分页配置类型。
- 新建与编辑通过表单提交到 `api.createFactor` 与 `api.updateFactor`。
- 提交前调用 `api.validateFactor` 校验公式，校验不通过不落库。
- 删除走 `api.deleteFactor`，复制走 `api.copyFactor`，均需确认后执行。
- 简化操作先调 `api.simplifyFactor`，再把简化结果回写到因子。
- 页面跳转使用 `useNavigate`，点击行进入因子详情。
- 样式由独立的 `FactorManagement.css` 承载。
- 接口封装来自 [[ApiClient]]。

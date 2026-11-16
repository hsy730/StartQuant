---
标题: FactorManagementPage
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/pages/FactorManagement.tsx]
---

## 是什么

`frontend/react-antd/src/pages/FactorManagement.tsx` 中的因子管理页面，以表格为主体承载因子的浏览与维护操作。

## 关键事实

- 列表数据来自后端因子接口，分页展示，点击行进入因子详情。
- 新建与编辑共用表单，提交前必须先通过公式校验，未通过不落库。
- 删除与复制等破坏性或派生操作需二次确认后执行。
- 简化操作先取回简化结果，再回写到因子。
- 样式独立成文件，接口封装来自 [[ApiClient]]。

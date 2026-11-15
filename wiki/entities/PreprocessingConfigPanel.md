---
标题: PreprocessingConfigPanel
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/components/PreprocessingConfigPanel.tsx]
---

## 是什么

`frontend/react-antd/src/components/PreprocessingConfigPanel.tsx` 是因子预处理参数配置面板，提供智能推荐与自定义两种模式。

## 关键事实

- 组件导出 `PreprocessingConfig` 接口（第 41 行）与面板组件，内部定义 `SmartRecommendation`（含 confidence、warnings）与 `PreprocessingConfigPanelProps`。
- 模式切换用 Radio：`智能(推荐)` 配 `RobotOutlined` 图标，`自定义` 配 `SettingOutlined` 图标。
- 智能模式通过后端接口获取推荐，成功后提示"智能推荐配置已生成"，失败时记录 `获取智能推荐失败` 日志。
- 置信度着色阈值：`>= 0.85` 绿色 `#52c41a`、`>= 0.7` 橙色 `#faad14`，并在徽标上显示百分比。
- 智能模式展示推荐理由与 `warnings` 列表，自定义模式展示实时校验结果中的 `warnings`。
- 交互规范见 [[智能自定义双模式面板]]，推荐算法来源见 [[SmartPreprocessingDetector]]。

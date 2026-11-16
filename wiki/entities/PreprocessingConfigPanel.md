---
标题: PreprocessingConfigPanel
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/components/PreprocessingConfigPanel.tsx]
---

## 是什么

`frontend/react-antd/src/components/PreprocessingConfigPanel.tsx` 中的因子预处理参数配置面板，提供智能推荐与自定义两种模式。

## 关键事实

- 组件同时导出配置接口与面板组件，内部定义推荐结果与 props 类型。
- 模式切换用单选控件呈现「智能(推荐)」与「自定义」两个选项。
- 智能模式调用后端推荐接口，成功后提示配置已生成，失败时记录日志。
- 置信度按阈值分级着色并在徽标上显示百分比，智能模式展示推荐理由与告警。
- 交互规范见 [[智能自定义双模式面板]]，推荐算法来源见 [[SmartPreprocessingDetector]]。

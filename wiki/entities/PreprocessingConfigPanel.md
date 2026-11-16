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
- 模式切换用单选控件，默认落在智能(推荐)；切换后控件区域整体替换而非叠加。
- 智能模式调用推荐接口并展示四类信息：数据特征、置信度徽标、推荐理由与告警；置信度 0.85 以上绿色、0.7 以上橙色、其余红色。
- 自定义模式为每个参数提供说明、默认值与实时校验，校验告警就地展示。
- 两种模式共用同一份配置结构，后端只认最终参数；推荐算法来源见 [[SmartPreprocessingDetector]]。

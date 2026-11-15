---
标题: FormulaCompilerService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/formula_compiler_service.py]
---

## 是什么

`backend/services/formula_compiler_service.py` 中的公式编译器，把可视化公式树或表达式转换成可执行、可校验的因子代码。

## 关键事实

- 类定义在该文件第 12 行，核心入口 `compile_formula(formula_tree)` 递归调用 `_compile_node` 生成代码字符串。
- `AVAILABLE_ELEMENTS` 定义四类因子元素：price（6 个行情字段）、indicators（9 个技术指标）、operators（9 个运算符）、statistics（7 个统计函数）。
- `parse_expression(expression)` 用 `ast` 把文本表达式解析成公式树，`simplify_formula` 做结果化简，`get_available_elements` 供前端渲染元素库。
- `validate_formula(formula_code)` 返回 `(是否通过, 错误消息)`，同时支持表达式与 `def` 函数两种形式。
- 除法节点编译为 `safe_series_divide`，`zscore` 节点编译为 20 日滚动标准化。
- 安全约束见 [[公式白名单沙箱]]，时序窗口约束见 [[时序标准化避免前视]]。

import React, { useState, useEffect, useCallback } from 'react'
import {
  Card,
  Select,
  Slider,
  Switch,
  Radio,
  Space,
  Tag,
  Tooltip,
  Button,
  Alert,
  Spin,
  Typography,
  Row,
  Col,
  Divider,
  Collapse,
  InputNumber,
  message
} from 'antd'
import {
  QuestionCircleOutlined,
  RobotOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  ThunderboltOutlined,
  ShieldOutlined,
  RocketOutlined,
  StockOutlined,
  ExperimentOutlined
} from '@ant-design/icons'
import { api } from '@/services/api'
import './PreprocessingConfigPanel.css'

const { Text, Paragraph } = Typography
const { Panel } = Collapse

export interface PreprocessingConfig {
  mode: 'smart' | 'custom'
  preset?: string
  winsorize_method: 'mad' | 'percentile' | 'std'
  winsorize_n_sigma?: number
  winsorize_limits?: [number, number]
  enable_market_cap_neutralization: boolean
  enable_industry_neutralization: boolean
  standardize_method: 'zscore' | 'rank'
  handle_missing: 'fill_zero' | 'fill_median' | 'drop'
}

interface SmartRecommendation {
  recommended_config: PreprocessingConfig
  final_config: PreprocessingConfig
  confidence: number
  reasoning: string
  warnings: string[]
  data_characteristics: {
    market_board: string
    n_stocks: number
    n_dates: number
    factor_volatility: number
    is_fat_tailed: boolean
    outlier_ratio: number
    n_industries: number
    min_industry_size: number
  }
  report: string
  presets: Record<string, {
    name: string
    description: string
    icon: string
    config?: PreprocessingConfig
    suitable_for: string[]
  }>
}

interface PreprocessingConfigPanelProps {
  value?: PreprocessingConfig
  onChange?: (config: PreprocessingConfig) => void
  stockCodes: string[]
  factorNames: string[]
  startDate?: string
  endDate?: string
  size?: 'default' | 'small' | 'large'
  disabled?: boolean
}

const DEFAULT_CONFIG: PreprocessingConfig = {
  mode: 'smart',
  winsorize_method: 'mad',
  winsorize_n_sigma: 3.0,
  enable_market_cap_neutralization: true,
  enable_industry_neutralization: true,
  standardize_method: 'zscore',
  handle_missing: 'fill_zero'
}

const PRESET_ICONS: Record<string, React.ReactNode> = {
  auto: <RobotOutlined />,
  shield: <ShieldOutlined />,
  rocket: <RocketOutlined />,
  machine_learning: <ExperimentOutlined />,
  stock: <StockOutlined />
}

export const PreprocessingConfigPanel: React.FC<PreprocessingConfigPanelProps> = ({
  value = DEFAULT_CONFIG,
  onChange,
  stockCodes = [],
  factorNames = [],
  startDate,
  endDate,
  size = 'default',
  disabled = false
}) => {
  const [recommendation, setRecommendation] = useState<SmartRecommendation | null>(null)
  const [loading, setLoading] = useState(false)
  const [validating, setValidating] = useState(false)
  const [validationResult, setValidationResult] = useState<{
    is_valid: boolean
    warnings: string[]
    suggestions: string[]
    risk_level: 'low' | 'medium' | 'high'
  } | null>(null)

  const fetchRecommendation = useCallback(async () => {
    if (stockCodes.length === 0 || factorNames.length === 0) {
      message.warning('请先选择股票和因子')
      return
    }

    setLoading(true)
    try {
      const result = await api.recommendPreprocessing({
        stock_codes: stockCodes,
        factor_names: factorNames,
        start_date: startDate || '2024-01-01',
        end_date: endDate || new Date().toISOString().split('T')[0],
        mode: 'smart'
      })

      if (result.success) {
        setRecommendation(result.data)

        if (result.data?.final_config && onChange) {
          onChange({
            ...value,
            ...result.data.final_config,
            mode: value.mode
          })
        }

        message.success('智能推荐配置已生成')
      } else {
        message.error(result.detail || '获取推荐配置失败')
      }
    } catch (error) {
      console.error('获取智能推荐失败:', error)
      message.error('网络错误，请稍后重试')
    } finally {
      setLoading(false)
    }
  }, [stockCodes, factorNames, startDate, endDate, value, onChange])

  const validateConfig = useCallback(async () => {
    setValidating(true)
    try {
      const result = await api.validatePreprocessing(value)

      if (result.success) {
        setValidationResult(result.data)
      }
    } catch (error) {
      console.error('验证失败:', error)
    } finally {
      setValidating(false)
    }
  }, [value])

  useEffect(() => {
    if (value.mode === 'smart' && stockCodes.length > 0 && factorNames.length > 0 && !recommendation) {
      fetchRecommendation()
    }
  }, [value.mode, stockCodes, factorNames, recommendation, fetchRecommendation])

  const updateConfig = (updates: Partial<PreprocessingConfig>) => {
    if (onChange && !disabled) {
      onChange({ ...value, ...updates })
      if (value.mode === 'custom') {
        validateConfig()
      }
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.85) return '#52c41a'
    if (confidence >= 0.7) return '#faad14'
    return '#ff4d4f'
  }

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'low': return '#52c41a'
      case 'medium': return '#faad14'
      case 'high': return '#ff4d4f'
      default: return '#d9d9d9'
    }
  }

  const getBoardDisplayName = (board: string) => {
    const names: Record<string, string> = {
      main: '主板',
      chinext: '创业板',
      star: '科创板',
      beijing: '北交所',
      mixed: '混合板块'
    }
    return names[board] || board
  }

  return (
    <Card
      className={`preprocessing-config-panel preprocessing-config-panel-${size}`}
      title={
        <Space>
          <ThunderboltOutlined style={{ color: '#1890ff' }} />
          <span>数据预处理（美颜）</span>
          <Tooltip title="根据数据特征智能优化去极值/中性化/标准化参数，确保因子分析结果的可靠性">
            <QuestionCircleOutlined style={{ color: '#1890ff', fontSize: '14px' }} />
          </Tooltip>
        </Space>
      }
      size={size}
      extra={
        <Tooltip title="查看预处理规范文档">
          <Button type="text" size="small" icon={<InfoCircleOutlined />} />
        </Tooltip>
      }
    >
      <div className="preprocessing-content">
        {/* 模式选择器 */}
        <div className="mode-selector">
          <Text strong style={{ marginBottom: 8, display: 'block' }}>处理模式：</Text>
          <Radio.Group
            value={value.mode}
            onChange={(e) => updateConfig({ mode: e.target.value })}
            disabled={disabled}
            optionType="button"
            buttonStyle="solid"
          >
            <Radio.Button value="smart">
              <RobotOutlined /> 智能(推荐)
            </Radio.Button>
            <Radio.Button value="custom">
              <SettingOutlined /> 自定义
            </Radio.Button>
          </Radio.Group>
        </div>

        <Divider style={{ margin: '12px 0' }} />

        {value.mode === 'smart' ? (
          /* ========== 智能模式面板 ========== */
          <div className="smart-mode-panel">
            {/* 一键生成按钮 */}
            <div className="action-bar">
              <Button
                type="primary"
                icon={<RobotOutlined />}
                loading={loading}
                onClick={fetchRecommendation}
                disabled={disabled || stockCodes.length === 0}
                block
                size={size}
              >
                🤖 分析数据并生成最优配置
              </Button>

              {stockCodes.length === 0 && (
                <Alert
                  message="请先选择股票代码"
                  type="info"
                  showIcon
                  size="small"
                  style={{ marginTop: 8 }}
                />
              )}
            </div>

            {recommendation && (
              <>
                {/* 置信度和数据特征 */}
                <div className="recommendation-header">
                  <Row gutter={[16, 8]} align="middle">
                    <Col>
                      <div className="confidence-badge">
                        <Text strong>置信度:</Text>
                        <Text
                          strong
                          style={{
                            color: getConfidenceColor(recommendation.confidence),
                            fontSize: size === 'small' ? 16 : 20,
                            marginLeft: 4
                          }}
                        >
                          {(recommendation.confidence * 100).toFixed(0)}%
                        </Text>
                      </div>
                    </Col>
                    <Col>
                      <Tag
                        icon={<StockOutlined />}
                        color={recommendation.data_characteristics.market_board !== 'mixed' ? 'blue' : 'orange'}
                      >
                        {getBoardDisplayName(recommendation.data_characteristics.market_board)}
                      </Tag>
                    </Col>
                    <Col>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {recommendation.data_characteristics.n_stocks}只股票 ×{' '}
                        {recommendation.data_characteristics.n_dates}天
                      </Text>
                    </Col>
                  </Row>
                </div>

                {/* 推荐理由 */}
                <div className="reasoning-section">
                  <Alert
                    message={
                      <Space direction="vertical" size={0}>
                        <Text strong>💡 智能推荐理由：</Text>
                        <Text>{recommendation.reasoning}</Text>
                      </Space>
                    }
                    type="info"
                    showIcon
                    icon={<CheckCircleOutlined />}
                    closable
                  />
                </div>

                {/* 警告信息 */}
                {recommendation.warnings.length > 0 && (
                  <div className="warnings-section">
                    {recommendation.warnings.map((warning, index) => (
                      <Alert
                        key={index}
                        message={warning}
                        type="warning"
                        showIcon
                        icon={<WarningOutlined />}
                        closable
                        style={{ marginBottom: 8 }}
                      />
                    ))}
                  </div>
                )}

                {/* 数据特征摘要 */}
                <Collapse ghost size="small" style={{ marginTop: 12 }}>
                  <Panel header="📊 数据特征详情" key="characteristics">
                    <Row gutter={[16, 12]}>
                      <Col span={12}>
                        <Text type="secondary">市场板块：</Text>
                        <Text strong>
                          {getBoardDisplayName(recommendation.data_characteristics.market_board)}
                        </Text>
                      </Col>
                      <Col span={12}>
                        <Text type="secondary">因子波动率：</Text>
                        <Text strong>
                          {recommendation.data_characteristics.factor_volatility.toFixed(4)}
                        </Text>
                      </Col>
                      <Col span={12}>
                        <Text type="secondary">肥尾分布：</Text>
                        <Tag color={recommendation.data_characteristics.is_fat_tailed ? 'warning' : 'success'}>
                          {recommendation.data_characteristics.is_fat_tailed ? '是' : '否'}
                        </Tag>
                      </Col>
                      <Col span={12}>
                        <Text type="secondary">异常值比例：</Text>
                        <Text strong>
                          {(recommendation.data_characteristics.outlier_ratio * 100).toFixed(2)}%
                        </Text>
                      </Col>
                      <Col span={12}>
                        <Text type="secondary">行业数量：</Text>
                        <Text strong>{recommendation.data_characteristics.n_industries}</Text>
                      </Col>
                      <Col span={12}>
                        <Text type="secondary">最小行业样本：</Text>
                        <Text strong>
                          {recommendation.data_characteristics.min_industry_size}只
                        </Text>
                      </Col>
                    </Row>
                  </Panel>
                </Collapse>

                {/* 预设模板快捷选择 */}
                <div className="preset-selector" style={{ marginTop: 16 }}>
                  <Text strong style={{ marginBottom: 8, display: 'block' }}>
                    或选择预设模板：
                  </Text>
                  <Select
                    placeholder="选择预设模板..."
                    onChange={(presetName) => {
                      const preset = recommendation.presets[presetName]
                      if (preset?.config && onChange) {
                        onChange({
                          ...value,
                          ...preset.config,
                          preset: presetName
                        })
                        message.success(`已应用"${preset.name}"配置`)
                      }
                    }}
                    disabled={disabled}
                    style={{ width: '100%' }}
                    size={size}
                  >
                    {Object.entries(recommendation.presets || {}).map(([key, val]) => (
                      <Select.Option key={key} value={key}>
                        <Space>
                          {PRESET_ICONS[val.icon] || <SettingOutlined />}
                          <span>{val.name}</span>
                        </Space>
                      </Select.Option>
                    ))}
                  </Select>
                </div>

                {/* 当前生效的配置预览 */}
                <div className="config-preview" style={{ marginTop: 16 }}>
                  <Divider orientation="left" plain>
                    <Text type="secondary">当前配置预览</Text>
                  </Divider>
                  <Row gutter={[16, 8]}>
                    <Col span={8}>
                      <Text type="secondary">去极值：</Text>
                      <br />
                      <Tag color="blue">
                        {value.winsorize_method.toUpperCase()}
                        {' '}
                        {value.winsorize_n_sigma?.toFixed(1)}σ
                      </Tag>
                    </Col>
                    <Col span={8}>
                      <Text type="secondary">中性化：</Text>
                      <br />
                      <Space>
                        {value.enable_market_cap_neutralization && <Tag color="green">市值</Tag>}
                        {value.enable_industry_neutralization && <Tag color="green">行业</Tag>}
                        {!value.enable_market_cap_neutralization &&
                         !value.enable_industry_neutralization && (
                           <Tag color="default">无</Tag>
                         )}
                      </Space>
                    </Col>
                    <Col span={8}>
                      <Text type="secondary">标准化：</Text>
                      <br />
                      <Tag color="purple">{value.standardize_method.toUpperCase()}</Tag>
                    </Col>
                  </Row>
                </div>
              </>
            )}

            {!recommendation && !loading && (
              <div className="empty-state">
                <Spin tip="等待分析数据..." />
              </div>
            )}
          </div>
        ) : (
          /* ========== 自定义模式面板 ========== */
          <div className="custom-mode-panel">
            <Collapse defaultActiveKey={['winsorize', 'neutralize', 'standardize']} ghost>
              {/* 去极值参数组 */}
              <Panel
                header={
                  <Space>
                    <ThunderboltOutlined />
                    <Text strong>去极值处理</Text>
                    <Tooltip title="去除因子中的极端异常值，防止个别股票扭曲整体结果">
                      <QuestionCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </Space>
                }
                key="winsorize"
              >
                <Row gutter={[24, 16]}>
                  <Col span={24}>
                    <Text strong style={{ display: 'block', marginBottom: 8 }}>去极值方法</Text>
                    <Select
                      value={value.winsorize_method}
                      onChange={(v) => updateConfig({ winsorize_method: v })}
                      disabled={disabled}
                      style={{ width: '100%' }}
                      size={size}
                    >
                      <Select.Option value="mad">
                        <Space>
                          <strong>MAD法</strong>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            （稳健，适合肥尾分布）
                          </Text>
                        </Space>
                      </Select.Option>
                      <Select.Option value="percentile">
                        <Space>
                          <strong>百分位法</strong>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            （常用，截断固定比例）
                          </Text>
                        </Space>
                      </Select.Option>
                      <Select.Option value="std">
                        <Space>
                          <strong>3σ标准差法</strong>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            （快速，假设正态分布）
                          </Text>
                        </Space>
                      </Select.Option>
                    </Select>
                  </Col>

                  {value.winsorize_method === 'mad' || value.winsorize_method === 'std' ? (
                    <Col span={24}>
                      <Text strong style={{ display: 'block', marginBottom: 8 }}>
                        去极值强度：
                        <Tooltip title="值越小越严格，截断更多异常值；建议范围：2.0-5.0">
                          <QuestionCircleOutlined style={{ color: '#1890ff', marginLeft: 4 }} />
                        </Tooltip>
                        <Text type="secondary" style={{ fontWeight: 'normal', marginLeft: 8 }}>
                          {value.winsorize_n_sigma?.toFixed(1)}σ
                        </Text>
                      </Text>
                      <Slider
                        min={2.0}
                        max={5.0}
                        step={0.1}
                        value={value.winsorize_n_sigma || 3.0}
                        onChange={(v) => updateConfig({ winsorize_n_sigma: v })}
                        disabled={disabled}
                        marks={{
                          2.0: '严格',
                          2.8: '创业板',
                          3.0: '标准',
                          4.0: '宽松',
                          5.0: '很宽松'
                        }}
                      />
                    </Col>
                  ) : null}

                  {value.winsorize_method === 'percentile' ? (
                    <Col span={24}>
                      <Text strong style={{ display: 'block', marginBottom: 8 }}>
                        截断边界：
                        <Tooltip title="截断头尾的百分比，如(0.01, 0.99)表示保留1%-99%分位的数据">
                          <QuestionCircleOutlined style={{ color: '#1890ff', marginLeft: 4 }} />
                        </Tooltip>
                      </Text>
                      <Space.Compact style={{ width: '100%' }}>
                        <InputNumber
                          min={0}
                          max={0.49}
                          step={0.005}
                          value={value.winsorize_limits?.[0]}
                          onChange={(v) =>
                            updateConfig({
                              winsorize_limits: [v || 0.01, value.winsorize_limits?.[1] || 0.99]
                            })
                          }
                          disabled={disabled}
                          addonBefore="下限"
                          style={{ width: '50%' }}
                          size={size}
                        />
                        <InputNumber
                          min={0.51}
                          max={1}
                          step={0.005}
                          value={value.winsorize_limits?.[1]}
                          onChange={(v) =>
                            updateConfig({
                              winsorize_limits: [value.winsorize_limits?.[0] || 0.01, v || 0.99]
                            })
                          }
                          disabled={disabled}
                          addonAfter="上限"
                          style={{ width: '50%' }}
                          size={size}
                        />
                      </Space.Compact>
                    </Col>
                  ) : null}
                </Row>
              </Panel>

              {/* 中性化参数组 */}
              <Panel
                header={
                  <Space>
                    <ExperimentOutlined />
                    <Text strong>中性化处理</Text>
                    <Tooltip title="消除市值和行业对因子的影响，分离纯Alpha信号">
                      <QuestionCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </Space>
                }
                key="neutralize"
              >
                <Row gutter={[24, 16]}>
                  <Col span={24}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Text strong>市值中性化</Text>
                        <Switch
                          checkedChildren="启用"
                          unCheckedChildren="关闭"
                          checked={value.enable_market_cap_neutralization}
                          onChange={(v) => updateConfig({ enable_market_cap_neutralization: v })}
                          disabled={disabled}
                        />
                      </div>
                      <Paragraph type="secondary" style={{ margin: 0, fontSize: 12 }}>
                        去除规模效应，避免大市值股票主导因子表现
                      </Paragraph>
                    </Space>
                  </Col>

                  <Col span={24}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Text strong>行业中性化</Text>
                        <Switch
                          checkedChildren="启用"
                          unCheckedChildren="关闭"
                          checked={value.enable_industry_neutralization}
                          onChange={(v) => updateConfig({ enable_industry_neutralization: v })}
                          disabled={disabled}
                        />
                      </div>
                      <Paragraph type="secondary" style={{ margin: 0, fontSize: 12 }}>
                        在行业内进行标准化，消除板块偏好
                      </Paragraph>
                    </Space>
                  </Col>

                  {!value.enable_market_cap_neutralization && !value.enable_industry_neutralization ? (
                    <Col span={24}>
                      <Alert
                        message="建议至少启用一种中性化以消除已知风险因子"
                        type="info"
                        showIcon
                        size="small"
                      />
                    </Col>
                  ) : null}
                </Row>
              </Panel>

              {/* 标准化参数组 */}
              <Panel
                header={
                  <Space>
                    <StockOutlined />
                    <Text strong>标准化处理</Text>
                    <Tooltip title="将不同量纲的因子转换为可比较的标准形式">
                      <QuestionCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </Space>
                }
                key="standardize"
              >
                <Row gutter={[24, 16]}>
                  <Col span={24}>
                    <Text strong style={{ display: 'block', marginBottom: 8 }}>标准化方法</Text>
                    <Radio.Group
                      value={value.standardize_method}
                      onChange={(e) => updateConfig({ standardize_method: e.target.value })}
                      disabled={disabled}
                      optionType="button"
                      buttonStyle="solid"
                    >
                      <Radio.Button value="zscore">
                        Z-score
                        <Tooltip title="线性变换，保持可解释性">
                          <QuestionCircleOutlined style={{ marginLeft: 4, fontSize: 12 }} />
                        </Tooltip>
                      </Radio.Button>
                      <Radio.Button value="rank">
                        Rank
                        <Tooltip title="均匀分布，抗异常值">
                          <QuestionCircleOutlined style={{ marginLeft: 4, fontSize: 12 }} />
                        </Tooltip>
                      </Radio.Button>
                    </Radio.Group>
                  </Col>

                  <Col span={24}>
                    <Text strong style={{ display: 'block', marginBottom: 8 }}>缺失值处理</Text>
                    <Select
                      value={value.handle_missing}
                      onChange={(v) => updateConfig({ handle_missing: v as any })}
                      disabled={disabled}
                      style={{ width: '100%' }}
                      size={size}
                    >
                      <Select.Option value="fill_zero">填充为 0</Select.Option>
                      <Select.Option value="fill_median">填充为中位数（更稳健）</Select.Option>
                      <Select.Option value="drop">删除缺失样本</Select.Option>
                    </Select>
                  </Col>
                </Row>
              </Panel>
            </Collapse>

            {/* 实时验证反馈 */}
            {validationResult && (
              <div className="validation-result" style={{ marginTop: 16 }}>
                <Divider orientation="left" plain>
                  <Text type="secondary">配置验证</Text>
                </Divider>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>风险等级：</Text>
                    <Tag
                      color={getRiskLevelColor(validationResult.risk_level)}
                      style={{ marginLeft: 8 }}
                    >
                      {validationResult.risk_level === 'low'
                        ? '低风险'
                        : validationResult.risk_level === 'medium'
                        ? '中等风险'
                        : '高风险'}
                    </Tag>
                  </div>

                  {validationResult.warnings.length > 0 && (
                    <div>
                      <Text strong style={{ color: '#faad14' }}>警告：</Text>
                      <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                        {validationResult.warnings.map((w, i) => (
                          <li key={i}>
                            <Text type="warning">{w}</Text>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {validationResult.suggestions.length > 0 && (
                    <div>
                      <Text strong style={{ color: '#1890ff' }}>建议：</Text>
                      <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                        {validationResult.suggestions.map((s, i) => (
                          <li key={i}>
                            <Text type="secondary">{s}</Text>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </Space>
              </div>
            )}

            {/* 手动触发验证按钮 */}
            <div style={{ textAlign: 'center', marginTop: 16 }}>
              <Button
                icon={<CheckCircleOutlined />}
                loading={validating}
                onClick={validateConfig}
                disabled={disabled}
                size={size}
              >
                验证配置合理性
              </Button>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}

// 导出默认配置常量
export { DEFAULT_CONFIG }

export default PreprocessingConfigPanel

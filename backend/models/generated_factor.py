"""
生成的因子数据模型
"""
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

from backend.core.database import Base


class GeneratedFactorModel(Base):
    """生成的因子模型"""

    __tablename__ = "generated_factors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 因子表达式
    expression: Mapped[str] = mapped_column(String(500), nullable=False, index=True)

    # 生成方法
    generation_method: Mapped[str] = mapped_column(String(50), nullable=False)

    # 验证结果
    ic_value: Mapped[float] = mapped_column(Float, nullable=True)
    ir_value: Mapped[float] = mapped_column(Float, nullable=True)
    turnover_value: Mapped[float] = mapped_column(Float, nullable=True)
    stability_score: Mapped[float] = mapped_column(Float, nullable=True)
    validation_score: Mapped[float] = mapped_column(Float, nullable=True)

    # 是否通过验证
    is_valid: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # 是否已保存到因子库
    is_saved: Mapped[bool] = mapped_column(Boolean, default=False)

    # 因子名称（如果已保存）
    factor_name: Mapped[str] = mapped_column(String(100), nullable=True)

    # 其他元数据
    complexity: Mapped[str] = mapped_column(String(20), nullable=True)
    extra_info: Mapped[str] = mapped_column(Text, nullable=True)

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<GeneratedFactor(id={self.id}, expression='{self.expression[:30]}...', is_valid={self.is_valid})>"

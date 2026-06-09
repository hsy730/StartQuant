"""
挖掘任务数据模型
"""
from sqlalchemy import Integer, String, Text, Float, JSON, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

from backend.core.database import Base


class MiningTaskModel(Base):
    """挖掘任务模型 - 持久化挖掘任务状态和结果"""
    __tablename__ = "mining_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 任务唯一标识
    task_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, index=True, comment="任务UUID")

    # 任务状态: pending / running / completed / failed / cancelled
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True, comment="任务状态")

    # 算法类型
    algorithm: Mapped[str] = mapped_column(String(30), nullable=False, default="genetic", comment="挖掘算法")

    # 输入参数
    stock_codes: Mapped[str] = mapped_column(Text, nullable=True, comment="股票代码列表(JSON)")
    base_factors: Mapped[str] = mapped_column(Text, nullable=True, comment="基础因子列表(JSON)")
    start_date: Mapped[str] = mapped_column(String(20), nullable=True, comment="数据开始日期")
    end_date: Mapped[str] = mapped_column(String(20), nullable=True, comment="数据结束日期")
    freq: Mapped[str] = mapped_column(String(10), nullable=True, default="D", comment="数据频率")
    config: Mapped[dict] = mapped_column(JSON, nullable=True, comment="完整挖掘配置(JSON)")

    # 进度信息
    progress: Mapped[int] = mapped_column(Integer, default=0, comment="进度百分比")
    current_generation: Mapped[int] = mapped_column(Integer, default=0, comment="当前迭代代数")
    total_generations: Mapped[int] = mapped_column(Integer, default=0, comment="总迭代代数")
    best_fitness: Mapped[float] = mapped_column(Float, nullable=True, comment="最优适应度")
    avg_fitness: Mapped[float] = mapped_column(Float, nullable=True, comment="平均适应度")
    fitness_history: Mapped[dict] = mapped_column(JSON, nullable=True, comment="适应度历史(JSON)")

    # 结果
    result: Mapped[dict] = mapped_column(JSON, nullable=True, comment="挖掘结果(JSON)")
    process_info: Mapped[dict] = mapped_column(JSON, nullable=True, comment="挖掘过程信息(JSON)")
    error: Mapped[str] = mapped_column(Text, nullable=True, comment="错误信息")

    # 时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(), comment="创建时间")
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=True, comment="开始执行时间")
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True, comment="完成时间")

    def __repr__(self):
        return f"<MiningTask(id={self.id}, task_id={self.task_id[:8]}..., status={self.status}, algorithm={self.algorithm})>"

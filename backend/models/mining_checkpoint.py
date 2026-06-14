"""
挖掘检查点数据模型 - 支持断点续跑

每代进化结束后保存种群和精英信息到磁盘，
当任务中断后可从最近的检查点恢复进化。
"""

from sqlalchemy import Integer, String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

from backend.core.database import Base


class MiningCheckpointModel(Base):
    """挖掘检查点模型 - 存储进化中间状态"""

    __tablename__ = "mining_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 关联的挖掘任务
    task_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True, comment="关联的挖掘任务UUID"
    )

    # 进化状态
    generation: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="已完成的代数"
    )
    total_generations: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="总代数"
    )

    # 种群序列化：[{tree_str, fitness_values}, ...]
    population_json: Mapped[str] = mapped_column(
        Text, nullable=False, comment="种群序列化(JSON)"
    )

    # 精英序列化：[{tree_str, fitness_values}, ...]
    hof_json: Mapped[str] = mapped_column(
        Text, nullable=True, comment="Hall-of-Fame序列化(JSON)"
    )

    # Z-Score 归一化状态
    zscore_stats_json: Mapped[str] = mapped_column(
        Text, nullable=True, comment="Z-Score统计量(JSON)"
    )

    # 适应度历史
    fitness_history_json: Mapped[str] = mapped_column(
        Text, nullable=True, comment="适应度历史(JSON)"
    )

    # 时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(), comment="创建时间"
    )

    def __repr__(self):
        return (
            f"<MiningCheckpoint(id={self.id}, task_id={self.task_id[:8]}..., "
            f"gen={self.generation}/{self.total_generations})>"
        )

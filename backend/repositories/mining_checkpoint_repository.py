"""
挖掘检查点数据访问层
"""

import json
import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from backend.models.mining_checkpoint import MiningCheckpointModel

logger = logging.getLogger(__name__)


class MiningCheckpointRepository:
    """挖掘检查点数据访问类"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, checkpoint: MiningCheckpointModel) -> MiningCheckpointModel:
        """创建检查点记录"""
        try:
            self.db.add(checkpoint)
            self.db.commit()
            self.db.refresh(checkpoint)
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建检查点失败: {e}")
            raise
        return checkpoint

    def get_latest(self, task_id: str) -> Optional[MiningCheckpointModel]:
        """获取指定任务的最新检查点"""
        return self.db.scalar(
            select(MiningCheckpointModel)
            .where(MiningCheckpointModel.task_id == task_id)
            .order_by(desc(MiningCheckpointModel.generation))
            .limit(1)
        )

    def get_by_task_id(self, task_id: str) -> List[MiningCheckpointModel]:
        """获取指定任务的所有检查点"""
        return list(
            self.db.scalars(
                select(MiningCheckpointModel)
                .where(MiningCheckpointModel.task_id == task_id)
                .order_by(MiningCheckpointModel.generation)
            ).all()
        )

    def delete_by_task_id(self, task_id: str) -> int:
        """删除指定任务的所有检查点"""
        checkpoints = self.get_by_task_id(task_id)
        count = 0
        for cp in checkpoints:
            self.db.delete(cp)
            count += 1
        self.db.commit()
        return count

    def cleanup_old_checkpoints(self, task_id: str, keep_last: int = 3) -> int:
        """清理旧检查点，只保留最近N个"""
        checkpoints = self.get_by_task_id(task_id)
        if len(checkpoints) <= keep_last:
            return 0
        to_delete = checkpoints[:-keep_last]
        for cp in to_delete:
            self.db.delete(cp)
        self.db.commit()
        return len(to_delete)

    @staticmethod
    def serialize_population(population) -> str:
        """将DEAP种群序列化为JSON字符串"""
        individuals = []
        for ind in population:
            individuals.append({
                "tree_str": str(ind),
                "fitness_values": list(ind.fitness.values) if ind.fitness.valid else None,
            })
        return json.dumps(individuals, ensure_ascii=False)

    @staticmethod
    def deserialize_population(population_json: str):
        """从JSON字符串反序列化种群信息（返回list of dicts）"""
        return json.loads(population_json)

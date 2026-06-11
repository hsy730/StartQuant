"""
挖掘任务数据访问层
"""

import json
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.models.mining_task import MiningTaskModel


class MiningTaskRepository:
    """挖掘任务数据访问类"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, task: MiningTaskModel) -> MiningTaskModel:
        """创建挖掘任务记录"""
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task

    def get_by_task_id(self, task_id: str) -> Optional[MiningTaskModel]:
        """根据task_id获取挖掘任务"""
        return self.db.scalar(select(MiningTaskModel).where(MiningTaskModel.task_id == task_id))

    def get_by_id(self, id: int) -> Optional[MiningTaskModel]:
        """根据主键ID获取挖掘任务"""
        return self.db.get(MiningTaskModel, id)

    def get_active_tasks(self) -> List[MiningTaskModel]:
        """获取所有活跃任务（pending/running）"""
        return list(
            self.db.scalars(
                select(MiningTaskModel)
                .where(MiningTaskModel.status.in_(["pending", "running"]))
                .order_by(MiningTaskModel.created_at.desc())
            ).all()
        )

    def get_history(self, limit: int = 50, offset: int = 0) -> List[MiningTaskModel]:
        """获取挖掘历史记录（分页）"""
        return list(
            self.db.scalars(
                select(MiningTaskModel).order_by(MiningTaskModel.created_at.desc()).limit(limit).offset(offset)
            ).all()
        )

    def get_history_count(self) -> int:
        """获取挖掘历史总数"""
        from sqlalchemy import func

        return self.db.scalar(select(func.count(MiningTaskModel.id)))

    def update_status(self, task_id: str, **kwargs) -> Optional[MiningTaskModel]:
        """更新任务状态"""
        task = self.get_by_task_id(task_id)
        if not task:
            return None
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        self.db.commit()
        self.db.refresh(task)
        return task

    def update_progress(
        self,
        task_id: str,
        progress: int,
        current_generation: int,
        total_generations: int,
        best_fitness: float,
        avg_fitness: float,
        fitness_history: dict = None,
    ) -> Optional[MiningTaskModel]:
        """更新挖掘进度"""
        task = self.get_by_task_id(task_id)
        if not task:
            return None
        task.status = "running"
        task.progress = progress
        task.current_generation = current_generation
        task.total_generations = total_generations
        task.best_fitness = best_fitness
        task.avg_fitness = avg_fitness
        if fitness_history is not None:
            task.fitness_history = fitness_history
        self.db.commit()
        self.db.refresh(task)
        return task

    def complete_task(self, task_id: str, result: dict, fitness_history: dict = None) -> Optional[MiningTaskModel]:
        """标记任务完成"""
        from datetime import datetime

        task = self.get_by_task_id(task_id)
        if not task:
            return None
        task.status = "completed"
        task.progress = 100
        task.result = result
        task.completed_at = datetime.now()
        if fitness_history:
            task.fitness_history = fitness_history
        if result:
            task.best_fitness = result.get("best_fitness", task.best_fitness)
            task.avg_fitness = result.get("avg_fitness", task.avg_fitness)
        self.db.commit()
        self.db.refresh(task)
        return task

    def fail_task(self, task_id: str, error: str) -> Optional[MiningTaskModel]:
        """标记任务失败"""
        from datetime import datetime

        task = self.get_by_task_id(task_id)
        if not task:
            return None
        task.status = "failed"
        task.error = error
        task.completed_at = datetime.now()
        self.db.commit()
        self.db.refresh(task)
        return task

    def delete(self, id: int) -> bool:
        """删除挖掘任务记录"""
        task = self.get_by_id(id)
        if not task:
            return False
        self.db.delete(task)
        self.db.commit()
        return True

    def to_dict(self, task: MiningTaskModel) -> Dict:
        """将挖掘任务模型转换为字典"""
        stock_codes = []
        if task.stock_codes:
            try:
                stock_codes = json.loads(task.stock_codes)
            except (json.JSONDecodeError, TypeError):
                stock_codes = []

        base_factors = []
        if task.base_factors:
            try:
                base_factors = json.loads(task.base_factors)
            except (json.JSONDecodeError, TypeError):
                base_factors = []

        return {
            "id": task.id,
            "task_id": task.task_id,
            "status": task.status,
            "algorithm": task.algorithm,
            "stock_codes": stock_codes,
            "base_factors": base_factors,
            "start_date": task.start_date,
            "end_date": task.end_date,
            "freq": task.freq,
            "progress": task.progress,
            "current_generation": task.current_generation,
            "total_generations": task.total_generations,
            "best_fitness": task.best_fitness,
            "avg_fitness": task.avg_fitness,
            "fitness_history": task.fitness_history,
            "result": task.result,
            "process_info": task.process_info,
            "error": task.error,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }

    def to_summary_dict(self, task: MiningTaskModel) -> Dict:
        """将挖掘任务模型转换为摘要字典（用于历史列表）"""
        stock_codes = []
        if task.stock_codes:
            try:
                stock_codes = json.loads(task.stock_codes)
            except (json.JSONDecodeError, TypeError):
                stock_codes = []

        factor_count = 0
        if task.result and isinstance(task.result, dict):
            factors = task.result.get("factors", [])
            factor_count = len(factors)

        return {
            "id": task.id,
            "task_id": task.task_id,
            "status": task.status,
            "algorithm": task.algorithm,
            "stock_codes": stock_codes,
            "start_date": task.start_date,
            "end_date": task.end_date,
            "progress": task.progress,
            "best_fitness": task.best_fitness,
            "factor_count": factor_count,
            "error": task.error,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }

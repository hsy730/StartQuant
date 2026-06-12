"""
因子数据访问层
"""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, delete, func

from backend.models.factor import FactorModel, AnalysisCacheModel

logger = logging.getLogger(__name__)


class FactorRepository:
    """因子数据访问类"""

    def __init__(self, db: Session):
        self.db = db

    def get_all(
        self, source: Optional[str] = None, active_only: bool = False
    ) -> List[FactorModel]:
        """获取所有因子"""
        query = select(FactorModel)
        if source:
            query = query.where(FactorModel.source == source)
        if active_only:
            query = query.where(FactorModel.is_active == 1)
        query = query.order_by(FactorModel.category, FactorModel.name)
        return list(self.db.scalars(query).all())

    def get_by_id(self, factor_id: int) -> Optional[FactorModel]:
        """根据ID获取因子"""
        return self.db.get(FactorModel, factor_id)

    def get_by_name(
        self, name: str, include_inactive: bool = False
    ) -> Optional[FactorModel]:
        """根据名称获取因子

        Args:
            name: 因子名称
            include_inactive: 是否包含已删除的因子（is_active=0）
        """
        query = select(FactorModel).where(FactorModel.name == name)
        if not include_inactive:
            query = query.where(FactorModel.is_active == 1)
        return self.db.scalar(query)

    def get_active_by_name(self, name: str) -> Optional[FactorModel]:
        """根据名称获取活跃因子（仅返回 is_active=1 的记录）"""
        return self.db.scalar(
            select(FactorModel)
            .where(FactorModel.name == name)
            .where(FactorModel.is_active == 1)
        )

    def create(self, factor: FactorModel) -> FactorModel:
        """创建因子"""
        try:
            self.db.add(factor)
            self.db.commit()
            self.db.refresh(factor)
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建因子失败: {e}")
            raise
        return factor

    def update(self, factor: FactorModel) -> FactorModel:
        """更新因子"""
        try:
            self.db.commit()
            self.db.refresh(factor)
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新因子失败: {e}")
            raise
        return factor

    def delete(self, factor_id: int) -> bool:
        """删除因子（硬删除，从数据库中完全移除，仅限用户自定义因子）"""

        factor = self.get_by_id(factor_id)
        if not factor:
            return False
        if factor.source == "preset":
            raise ValueError("预置因子不能删除")

        try:
            self.db.delete(factor)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除因子失败: {e}")
            raise

        return True

    def get_preset_count(self) -> int:
        """获取预置因子数量（仅统计启用的）"""
        return (
            self.db.scalar(
                select(func.count(FactorModel.id))
                .where(FactorModel.source == "preset")
                .where(FactorModel.is_active == 1)
            )
            or 0
        )

    def get_user_count(self) -> int:
        """获取用户自定义因子数量（仅统计启用的）"""
        return (
            self.db.scalar(
                select(func.count(FactorModel.id))
                .where(FactorModel.source == "user")
                .where(FactorModel.is_active == 1)
            )
            or 0
        )


class AnalysisCacheRepository:
    """分析结果缓存数据访问类"""

    def __init__(self, db: Session):
        self.db = db

    def get_by_key(self, cache_key: str) -> Optional[AnalysisCacheModel]:
        """根据缓存键获取缓存"""
        return self.db.scalar(
            select(AnalysisCacheModel).where(AnalysisCacheModel.cache_key == cache_key)
        )

    def create(self, cache: AnalysisCacheModel) -> AnalysisCacheModel:
        """创建缓存"""
        self.db.add(cache)
        self.db.commit()
        self.db.refresh(cache)
        return cache

    def update(self, cache: AnalysisCacheModel) -> AnalysisCacheModel:
        """更新缓存"""
        self.db.commit()
        self.db.refresh(cache)
        return cache

    def delete(self, cache_id: int) -> bool:
        """删除缓存"""
        cache = self.db.get(AnalysisCacheModel, cache_id)
        if not cache:
            return False
        self.db.delete(cache)
        self.db.commit()
        return True

    def delete_old_cache(self, days: int = 7) -> int:
        """删除旧缓存"""
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days)
        stmt = delete(AnalysisCacheModel).where(
            AnalysisCacheModel.created_at < cutoff_date
        )
        result = self.db.execute(stmt)
        self.db.commit()
        return result.rowcount

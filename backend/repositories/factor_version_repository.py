"""
因子版本数据仓储
"""

from typing import List, Optional
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.models.factor_version import FactorVersionModel


class FactorVersionRepository:
    """因子版本仓储类"""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        factor_id: int,
        version_code: str,
        code: str,
        description: Optional[str] = None,
        change_reason: Optional[str] = None,
    ) -> FactorVersionModel:
        """创建新版本"""
        version = FactorVersionModel(
            factor_id=factor_id,
            version_code=version_code,
            code=code,
            description=description,
            change_reason=change_reason,
            is_current=True,
        )

        # 将该因子的其他版本标记为非当前
        self._set_current_false(factor_id)

        self.db.add(version)
        self.db.commit()
        self.db.refresh(version)
        return version

    def get_by_id(self, version_id: int) -> Optional[FactorVersionModel]:
        """根据ID获取版本"""
        return self.db.get(FactorVersionModel, version_id)

    def get_by_factor_id(self, factor_id: int) -> List[FactorVersionModel]:
        """获取因子的所有版本"""
        stmt = (
            select(FactorVersionModel)
            .where(FactorVersionModel.factor_id == factor_id)
            .order_by(FactorVersionModel.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def get_current_version(self, factor_id: int) -> Optional[FactorVersionModel]:
        """获取因子的当前版本"""
        stmt = select(FactorVersionModel).where(
            FactorVersionModel.factor_id == factor_id,
            FactorVersionModel.is_current,
        )
        return self.db.scalar(stmt)

    def get_by_version_code(self, factor_id: int, version_code: str) -> Optional[FactorVersionModel]:
        """根据版本号获取版本"""
        stmt = select(FactorVersionModel).where(
            FactorVersionModel.factor_id == factor_id,
            FactorVersionModel.version_code == version_code,
        )
        return self.db.scalar(stmt)

    def delete(self, version_id: int) -> bool:
        """删除版本"""
        version = self.get_by_id(version_id)
        if not version:
            return False

        self.db.delete(version)
        self.db.commit()
        return True

    def delete_by_factor_id(self, factor_id: int) -> int:
        """删除因子的所有版本"""
        from sqlalchemy import delete

        stmt = delete(FactorVersionModel).where(FactorVersionModel.factor_id == factor_id)
        result = self.db.execute(stmt)
        self.db.commit()
        return result.rowcount

    def set_current(self, version_id: int) -> bool:
        """设置某个版本为当前版本"""
        version = self.get_by_id(version_id)
        if not version:
            return False

        # 将该因子的其他版本标记为非当前
        self._set_current_false(version.factor_id)

        # 设置该版本为当前
        version.is_current = True
        self.db.commit()
        return True

    def _set_current_false(self, factor_id: int) -> None:
        """将因子的所有版本设置为非当前"""
        stmt = update(FactorVersionModel).where(FactorVersionModel.factor_id == factor_id).values(is_current=False)
        self.db.execute(stmt)

    def get_version_count(self, factor_id: int) -> int:
        """获取因子的版本数量"""
        from sqlalchemy import func

        return (
            self.db.scalar(select(func.count(FactorVersionModel.id)).where(FactorVersionModel.factor_id == factor_id))
            or 0
        )

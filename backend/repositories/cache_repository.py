"""
缓存元数据仓储
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from backend.models.cache_metadata import CacheMetadataModel


class CacheRepository:
    """缓存元数据仓储类"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, cache_key: str, file_path: str, ttl: Optional[int], size: int) -> CacheMetadataModel:
        """创建缓存元数据记录"""
        metadata = CacheMetadataModel(
            cache_key=cache_key,
            file_path=file_path,
            created_at=datetime.now(),
            ttl=ttl,
            size=size,
            access_count=0,
            last_accessed=None,
            expired=False,
        )
        self.db.add(metadata)
        self.db.commit()
        self.db.refresh(metadata)
        return metadata

    def get_by_key(self, cache_key: str) -> Optional[CacheMetadataModel]:
        """根据缓存键获取元数据"""
        stmt = select(CacheMetadataModel).where(CacheMetadataModel.cache_key == cache_key)
        result = self.db.execute(stmt).scalar_one_or_none()
        return result

    def update_access(self, metadata: CacheMetadataModel) -> None:
        """更新访问信息"""
        metadata.mark_as_accessed()
        self.db.commit()

    def mark_as_expired(self, metadata: CacheMetadataModel) -> None:
        """标记为已过期"""
        metadata.expired = True
        self.db.commit()

    def delete(self, metadata: CacheMetadataModel) -> None:
        """删除缓存元数据记录"""
        self.db.delete(metadata)
        self.db.commit()

    def delete_by_key(self, cache_key: str) -> bool:
        """根据缓存键删除"""
        metadata = self.get_by_key(cache_key)
        if metadata:
            self.delete(metadata)
            return True
        return False

    def get_all_expired(self) -> List[CacheMetadataModel]:
        """获取所有过期的缓存元数据"""
        stmt = select(CacheMetadataModel)
        all_metadata = self.db.execute(stmt).scalars().all()

        expired_list = []
        for metadata in all_metadata:
            if metadata.is_expired():
                expired_list.append(metadata)

        return expired_list

    def get_all(self) -> List[CacheMetadataModel]:
        """获取所有缓存元数据"""
        stmt = select(CacheMetadataModel).order_by(CacheMetadataModel.created_at.desc())
        return list(self.db.execute(stmt).scalars().all())

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        all_metadata = self.get_all()

        total_count = len(all_metadata)
        total_size = sum(m.size for m in all_metadata)
        total_access_count = sum(m.access_count for m in all_metadata)

        expired_count = sum(1 for m in all_metadata if m.is_expired())

        # 计算命中率（假设每次 get 都是一次访问尝试）
        # 这里简化计算：访问总数 / (访问总数 + 过期数)
        # 实际命中率需要在 CacheService 中跟踪
        _hit_count = total_access_count  # noqa: F841
        _miss_count = expired_count  # 简化假设  # noqa: F841

        return {
            "total_count": total_count,
            "total_size": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "total_access_count": total_access_count,
            "expired_count": expired_count,
            "active_count": total_count - expired_count,
        }

    def clear_all(self) -> int:
        """清空所有缓存元数据"""
        stmt = delete(CacheMetadataModel)
        result = self.db.execute(stmt)
        self.db.commit()
        return result.rowcount

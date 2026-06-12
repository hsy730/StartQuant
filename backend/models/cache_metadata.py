"""
缓存元数据模型
"""

from datetime import datetime
from sqlalchemy import String, Integer, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class CacheMetadataModel(Base):
    """缓存元数据模型 - 跟踪缓存文件的使用情况和过期时间"""

    __tablename__ = "cache_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cache_key: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    file_path: Mapped[str] = mapped_column(
        String(500), nullable=False
    )  # 缓存文件的完整路径
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)  # 创建时间
    ttl: Mapped[int] = mapped_column(
        Integer, nullable=True
    )  # 生存时间（秒），None表示永不过期
    size: Mapped[int] = mapped_column(Integer, nullable=False)  # 缓存文件大小（字节）
    access_count: Mapped[int] = mapped_column(Integer, default=0)  # 访问次数
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime, nullable=True
    )  # 最后访问时间
    expired: Mapped[bool] = mapped_column(Boolean, default=False)  # 是否已过期

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "cache_key": self.cache_key,
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ttl": self.ttl,
            "size": self.size,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
            if self.last_accessed
            else None,
            "expired": self.expired,
            "is_expired": self.is_expired(),
        }

    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        if self.expired:
            return True
        if self.ttl is None:
            return False
        # 检查是否超过 TTL
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl

    def mark_as_accessed(self) -> None:
        """标记为已访问"""
        self.access_count += 1
        self.last_accessed = datetime.now()

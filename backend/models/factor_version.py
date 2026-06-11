"""
因子版本数据模型
"""

from datetime import datetime
from sqlalchemy import String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class FactorVersionModel(Base):
    """因子版本模型 - 记录因子的历史版本"""

    __tablename__ = "factor_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factor_id: Mapped[int] = mapped_column(Integer, ForeignKey("factors.id"), nullable=False, index=True)
    version_code: Mapped[str] = mapped_column(String(50), nullable=False)  # 版本号，如 v1.0, v1.1
    code: Mapped[str] = mapped_column(Text, nullable=False)  # 因子计算代码
    description: Mapped[str] = mapped_column(Text, nullable=True)  # 因子描述
    change_reason: Mapped[str] = mapped_column(Text, nullable=True)  # 变更原因
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    is_current: Mapped[bool] = mapped_column(Integer, default=False)  # 是否为当前版本

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "factor_id": self.factor_id,
            "version_code": self.version_code,
            "code": self.code,
            "description": self.description,
            "change_reason": self.change_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_current": bool(self.is_current),
        }

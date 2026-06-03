"""
数据库连接管理模块
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator

from backend.core.settings import settings


# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# 创建 Session 工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """数据库模型基类"""
    pass


def init_db() -> None:
    """初始化数据库，创建所有表"""
    from backend.models.factor import FactorModel, AnalysisCacheModel
    from backend.models.backtest import BacktestResultModel, TradeRecordModel
    from backend.models.cache_metadata import CacheMetadataModel
    from backend.models.factor_version import FactorVersionModel
    from backend.models.generated_factor import GeneratedFactorModel

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """获取数据库会话的上下文管理器"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """获取数据库会话（非上下文管理器方式）"""
    return SessionLocal()

"""
数据库连接管理模块

提供两种DB会话获取方式：
1. get_db() — 上下文管理器（推荐），自动关闭session
2. with_db() — 装饰器，为服务方法自动注入db session

禁止直接使用 get_db_session()（已废弃），因其无异常保护，易导致session泄漏。
"""

import logging
from functools import wraps
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from contextlib import contextmanager
from typing import Generator, Callable, TypeVar, Any

from backend.core.settings import settings

logger = logging.getLogger(__name__)

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

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话的上下文管理器（推荐方式）

    用法:
        with get_db() as db:
            repo = SomeRepository(db)
            result = repo.get_by_id(1)

    异常时session也会自动关闭，杜绝泄漏。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


F = TypeVar("F", bound=Callable)


def with_db(func: F = None, *, db_param: str = "db") -> Any:
    """
    装饰器：自动为服务方法注入数据库session，方法结束后自动关闭

    替代在方法内部手动调用 get_db_session() + db.close() 的模式，
    确保异常路径下session也能正确关闭。

    用法:
        @with_db
        def get_factor(self, factor_id: int, db: Session = None):
            repo = FactorRepository(db)
            return repo.get_by_id(factor_id)

        # 自定义db参数名
        @with_db(db_param="session")
        def get_factor(self, factor_id: int, session: Session = None):
            repo = FactorRepository(session)
            return repo.get_by_id(factor_id)

    Args:
        func: 被装饰的函数
        db_param: 方法中接收session的参数名（默认"db"）
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # 如果调用方已显式传入db，直接使用（允许覆盖）
            if db_param in kwargs and kwargs[db_param] is not None:
                return fn(*args, **kwargs)

            with get_db() as db:
                kwargs[db_param] = db
                return fn(*args, **kwargs)

        return wrapper

    if func is not None:
        # 无参调用: @with_db
        return decorator(func)
    # 有参调用: @with_db(db_param="session")
    return decorator


def get_db_session() -> Session:
    """
    获取数据库会话（已废弃 — 使用 get_db() 或 @with_db 替代）

    此函数返回的session无自动关闭保护，异常时易泄漏。
    保留仅为向后兼容，新代码禁止使用。
    """
    import warnings

    warnings.warn(
        "get_db_session() is deprecated, use get_db() context manager or @with_db decorator instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return SessionLocal()

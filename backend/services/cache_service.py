"""
智能缓存服务 - 管理缓存文件的元数据、TTL、统计和清理
"""
import os
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict

from sqlalchemy.orm import Session

from backend.core.settings import settings
from backend.core.database import get_db_session
from backend.repositories.cache_repository import CacheRepository

logger = logging.getLogger(__name__)


class CacheService:
    """智能缓存服务类"""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        初始化缓存服务

        Args:
            cache_dir: 缓存目录，默认使用 settings.AKSHARE_CACHE_DIR
        """
        self.cache_dir = cache_dir or settings.AKSHARE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 统计信息
        self._hits = 0  # 缓存命中次数
        self._misses = 0  # 缓存未命中次数

    def _get_db(self) -> Session:
        """获取数据库会话"""
        return get_db_session()

    def _generate_cache_key(self, *args) -> str:
        """
        生成缓存键

        Args:
            *args: 用于生成键的参数

        Returns:
            MD5哈希值作为缓存键
        """
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(
        self,
        cache_key: str,
        default: Any = None,
        update_access: bool = True,
    ) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            cache_key: 缓存键
            default: 默认值，缓存不存在时返回
            update_access: 是否更新访问统计

        Returns:
            缓存的数据，如果不存在或已过期则返回默认值
        """
        db = self._get_db()
        try:
            repo = CacheRepository(db)
            metadata = repo.get_by_key(cache_key)

            # 检查缓存是否存在
            cache_path = self._get_cache_path(cache_key)
            if not cache_path.exists():
                self._misses += 1
                return default

            # 检查元数据
            if metadata:
                # 检查是否过期
                if metadata.is_expired():
                    self._misses += 1
                    # 标记为过期
                    repo.mark_as_expired(metadata)
                    return default

                # 更新访问统计
                if update_access:
                    repo.update_access(metadata)

            # 加载缓存数据
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                self._hits += 1
                return data
            except Exception as e:
                logger.debug(f"缓存文件损坏: {e}")
                # 缓存文件损坏，删除
                self._misses += 1
                if cache_path.exists():
                    os.remove(cache_path)
                if metadata:
                    repo.delete(metadata)
                return default

        finally:
            db.close()

    def set(
        self,
        cache_key: str,
        data: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        保存数据到缓存

        Args:
            cache_key: 缓存键
            data: 要缓存的数据
            ttl: 生存时间（秒），None 表示永不过期

        Returns:
            是否成功保存
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            # 保存数据到文件
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            # 获取文件大小
            file_size = os.path.getsize(cache_path)

            # 保存或更新元数据
            db = self._get_db()
            try:
                repo = CacheRepository(db)
                existing_metadata = repo.get_by_key(cache_key)

                if existing_metadata:
                    # 更新现有元数据
                    existing_metadata.created_at = datetime.now()
                    existing_metadata.ttl = ttl
                    existing_metadata.size = file_size
                    existing_metadata.expired = False
                else:
                    # 创建新元数据
                    repo.create(
                        cache_key=cache_key,
                        file_path=str(cache_path),
                        ttl=ttl,
                        size=file_size,
                    )

                return True

            finally:
                db.close()

        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
            return False

    def delete(self, cache_key: str) -> bool:
        """
        删除缓存

        Args:
            cache_key: 缓存键

        Returns:
            是否成功删除
        """
        cache_path = self._get_cache_path(cache_key)

        # 删除文件
        if cache_path.exists():
            try:
                os.remove(cache_path)
            except Exception as e:
                logger.debug(f"删除缓存文件失败: {e}")

        # 删除元数据
        db = self._get_db()
        try:
            repo = CacheRepository(db)
            return repo.delete_by_key(cache_key)
        finally:
            db.close()

    def cleanup_expired(self) -> int:
        """
        清理所有过期的缓存

        Returns:
            清理的缓存数量
        """
        db = self._get_db()
        try:
            repo = CacheRepository(db)
            expired_metadata_list = repo.get_all_expired()

            cleaned_count = 0
            for metadata in expired_metadata_list:
                # 删除文件
                file_path = Path(metadata.file_path)
                if file_path.exists():
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.debug(f"删除过期缓存文件失败: {e}")

                # 删除元数据
                repo.delete(metadata)
                cleaned_count += 1

            return cleaned_count

        finally:
            db.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            包含统计信息的字典
        """
        db = self._get_db()
        try:
            repo = CacheRepository(db)
            stats = repo.get_stats()

            # 添加命中率统计
            total_attempts = self._hits + self._misses
            hit_rate = self._hits / total_attempts if total_attempts > 0 else 0.0

            stats.update({
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            })

            return stats

        finally:
            db.close()

    def clear_all(self) -> int:
        """
        清空所有缓存（元数据和文件）

        Returns:
            清理的缓存数量
        """
        db = self._get_db()
        try:
            repo = CacheRepository(db)
            all_metadata = repo.get_all()

            # 删除所有文件
            for metadata in all_metadata:
                file_path = Path(metadata.file_path)
                if file_path.exists():
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.debug(f"清理缓存文件失败: {e}")

            # 清空元数据表
            return repo.clear_all()

        finally:
            db.close()

    def exists(self, cache_key: str) -> bool:
        """
        检查缓存是否存在且未过期

        Args:
            cache_key: 缓存键

        Returns:
            缓存是否存在且有效
        """
        data = self.get(cache_key, default=None, update_access=False)
        return data is not None


# 全局缓存服务实例
cache_service = CacheService()

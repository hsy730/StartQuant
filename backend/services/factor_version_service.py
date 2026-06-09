"""
因子版本服务 - 因子版本管理
"""
from typing import List, Optional, Dict

from backend.core.database import get_db
from backend.repositories.factor_version_repository import FactorVersionRepository
from backend.repositories.factor_repository import FactorRepository


class FactorVersionService:
    """因子版本服务类"""

    def __init__(self):
        pass

    def create_version(
        self,
        factor_id: int,
        code: str,
        description: Optional[str] = None,
        change_reason: Optional[str] = None,
        auto_increment: bool = True,
    ) -> Dict:
        """
        创建因子新版本

        Args:
            factor_id: 因子ID
            code: 因子代码
            description: 因子描述
            change_reason: 变更原因
            auto_increment: 是否自动生成版本号

        Returns:
            创建的版本信息
        """
        with get_db() as db:
            version_repo = FactorVersionRepository(db)
            factor_repo = FactorRepository(db)

            # 获取因子信息
            factor = factor_repo.get_by_id(factor_id)
            if not factor:
                raise ValueError(f"因子 ID {factor_id} 不存在")

            # 生成版本号
            if auto_increment:
                version_code = self._generate_version_code(factor_id)
            else:
                version_code = "v1.0"

            # 创建版本
            version = version_repo.create(
                factor_id=factor_id,
                version_code=version_code,
                code=code,
                description=description or factor.description,
                change_reason=change_reason or "手动创建版本",
            )

            return version.to_dict()

    def rollback_to_version(self, factor_id: int, version_code: str) -> bool:
        """
        回滚因子到指定版本

        Args:
            factor_id: 因子ID
            version_code: 目标版本号

        Returns:
            是否回滚成功
        """
        with get_db() as db:
            version_repo = FactorVersionRepository(db)
            factor_repo = FactorRepository(db)

            # 获取目标版本
            target_version = version_repo.get_by_version_code(factor_id, version_code)
            if not target_version:
                raise ValueError(f"版本 {version_code} 不存在")

            # 获取因子
            factor = factor_repo.get_by_id(factor_id)
            if not factor:
                raise ValueError(f"因子 ID {factor_id} 不存在")

            # 创建当前版本的备份（在回滚前）
            self.create_version(
                factor_id=factor_id,
                code=factor.code,
                description=factor.description,
                change_reason=f"回滚前自动备份（回滚到 {version_code}）",
                auto_increment=True,
            )

            # 回滚因子代码和描述
            factor.code = target_version.code
            factor.description = target_version.description
            factor_repo.update(factor)

            # 设置目标版本为当前版本
            version_repo.set_current(target_version.id)

            return True

    def get_version_history(self, factor_id: int) -> List[Dict]:
        """
        获取因子的版本历史

        Args:
            factor_id: 因子ID

        Returns:
            版本历史列表
        """
        with get_db() as db:
            version_repo = FactorVersionRepository(db)
            versions = version_repo.get_by_factor_id(factor_id)
            return [v.to_dict() for v in versions]

    def get_current_version_info(self, factor_id: int) -> Optional[Dict]:
        """
        获取因子当前版本信息

        Args:
            factor_id: 因子ID

        Returns:
            当前版本信息
        """
        with get_db() as db:
            version_repo = FactorVersionRepository(db)
            version = version_repo.get_current_version(factor_id)
            return version.to_dict() if version else None

    def delete_version(self, version_id: int) -> bool:
        """
        删除指定版本

        Args:
            version_id: 版本ID

        Returns:
            是否删除成功
        """
        with get_db() as db:
            version_repo = FactorVersionRepository(db)

            # 不能删除当前版本
            version = version_repo.get_by_id(version_id)
            if version and version.is_current:
                raise ValueError("不能删除当前版本")

            return version_repo.delete(version_id)

    def compare_versions(self, factor_id: int, version_code1: str, version_code2: str) -> Dict:
        """
        比较两个版本的差异

        Args:
            factor_id: 因子ID
            version_code1: 版本1
            version_code2: 版本2

        Returns:
            比较结果
        """
        with get_db() as db:
            version_repo = FactorVersionRepository(db)

            version1 = version_repo.get_by_version_code(factor_id, version_code1)
            version2 = version_repo.get_by_version_code(factor_id, version_code2)

            if not version1 or not version2:
                raise ValueError("版本不存在")

            return {
                "version1": {
                    "code": version_code1,
                    "code_content": version1.code,
                    "description": version1.description,
                },
                "version2": {
                    "code": version_code2,
                    "code_content": version2.code,
                    "description": version2.description,
                },
                "code_changed": version1.code != version2.code,
                "description_changed": version1.description != version2.description,
            }

    def _generate_version_code(self, factor_id: int) -> str:
        """自动生成版本号"""
        with get_db() as db:
            version_repo = FactorVersionRepository(db)
            count = version_repo.get_version_count(factor_id)
            major = count // 10 + 1
            minor = count % 10
            return f"v{major}.{minor}"


# 全局因子版本服务实例
factor_version_service = FactorVersionService()

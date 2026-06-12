"""
ML 模型注册中心（Model Registry）— ML 模型的版本管理和生命周期管理

提供统一的模型存储、加载、查询接口。
支持两种后端：
  1. 文件系统（默认，无需额外依赖）
  2. 数据库（通过 SQLAlchemy，适合生产环境）

设计原则：
- 框架无关：可存储 XGBoost、LightGBM、PyTorch、sklearn 等任何可序列化的模型
- 版本管理：每次 save 创建新版本，支持版本回滚
- 元数据丰富：记录训练参数、特征列表、评估指标等
- 类型安全：强类型约束，避免运行时错误

使用方式：
    registry = ModelRegistry()

    # 保存模型
    model_id = registry.save(xgb_model, metadata={"features": [...], "params": {...}})

    # 加载模型
    model = registry.load(model_id)

    # 列出模型
    models = registry.list_models(tags=["production"])

    # 版本管理
    versions = registry.get_version_history(model_base_name="stock_ranker")
"""

import logging
import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelFramework(str, Enum):
    """支持的 ML 框架"""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    GENERIC = "generic"


class ModelStage(str, Enum):
    """模型生命周期阶段"""

    DEVELOPMENT = "development"  # 开发中
    STAGING = "staging"  # 预发布
    PRODUCTION = "production"  # 生产环境
    ARCHIVED = "archived"  # 已归档


@dataclass
class ModelMetadata:
    """模型元数据"""

    model_id: str
    model_name: str
    framework: ModelFramework
    version: int
    stage: ModelStage = ModelStage.DEVELOPMENT
    created_at: str = ""
    updated_at: str = ""
    created_by: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    feature_cols: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    file_size_bytes: int = 0
    checksum_sha256: str = ""

    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "framework": self.framework.value,
            "version": self.version,
            "stage": self.stage.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
            "params": self.params,
            "feature_cols": self.feature_cols,
            "metrics": self.metrics,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelMetadata":
        return cls(
            model_id=data.get("model_id", ""),
            model_name=data.get("model_name", ""),
            framework=ModelFramework(data.get("framework", "generic")),
            version=data.get("version", 1),
            stage=ModelStage(data.get("stage", "development")),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            created_by=data.get("created_by", "system"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            params=data.get("params", {}),
            feature_cols=data.get("feature_cols", []),
            metrics=data.get("metrics", {}),
            file_size_bytes=data.get("file_size_bytes", 0),
            checksum_sha256=data.get("checksum_sha256", ""),
        )


class ModelRegistry:
    """
    ML 模型注册中心

    提供模型的 CRUD 操作、版本管理、生命周期管理。
    默认使用文件系统作为后端存储。
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        use_database: bool = False,
    ):
        """
        初始化模型注册中心

        Args:
            base_path: 模型存储根目录（默认: 项目目录下的 models/registry）
            use_database: 是否使用数据库后端（需要 SQLAlchemy 配置）
        """
        if base_path is None:
            base_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "models",
                "registry",
            )
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.use_database = use_database

        # 内存索引（加速查询）
        self._index: Dict[str, ModelMetadata] = {}
        self._rebuild_index()

    def save(
        self,
        model: Any,
        metadata: Dict[str, Any],
        framework: str = "generic",
        model_name: Optional[str] = None,
        stage: str = "development",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        保存模型并返回 model_id

        Args:
            model: 模型对象（需支持 save_model / save / joblib.dump 等）
            metadata: 元数据字典
            framework: 框架标识 (xgboost/lightgbm/pytorch/sklearn/generic)
            model_name: 模型名称（用于版本管理）
            stage: 生命周期阶段
            description: 模型描述
            tags: 标签列表

        Returns:
            model_id: 唯一模型标识符
        """
        now = datetime.now()
        _model_name = model_name or metadata.get("model_name", "unnamed_model")

        # 生成 model_id 和版本号
        latest_version = self._get_latest_version(_model_name)
        new_version = latest_version + 1
        model_id = f"{_model_name}_v{new_version}"

        # 创建模型专属目录
        model_dir = self.base_path / _model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 序列化模型文件
        model_file_path = model_dir / f"v{new_version}.bin"
        meta_file_path = model_dir / f"v{new_version}_metadata.json"

        fw = ModelFramework(framework.lower())

        # 根据框架选择序列化方式
        file_size = self._serialize_model(model, str(model_file_path), fw)
        checksum = self._checksum(str(model_file_path))

        # 构建 ModelMetadata
        meta = ModelMetadata(
            model_id=model_id,
            model_name=_model_name,
            framework=fw,
            version=new_version,
            stage=ModelStage(stage),
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            description=description,
            tags=tags or [],
            params=metadata.get("params", {}),
            feature_cols=metadata.get("feature_cols", []),
            metrics=metadata.get("metrics", {}),
            file_size_bytes=file_size,
            checksum_sha256=checksum,
        )

        # 写入元数据
        with open(meta_file_path, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        # 更新索引
        self._index[model_id] = meta

        logger.info(
            f"[ModelRegistry] 模型已保存: {model_id} "
            f"(framework={fw.value}, size={file_size} bytes, version=v{new_version})"
        )

        return model_id

    def load(self, model_id: str) -> Any:
        """
        加载模型

        Args:
            model_id: 模型 ID

        Returns:
            反序列化的模型对象
        """
        meta = self._get_metadata(model_id)
        if meta is None:
            raise FileNotFoundError(f"模型不存在: {model_id}")

        model_file_path = self.base_path / meta.model_name / f"v{meta.version}.bin"
        if not model_file_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file_path}")

        return self._deserialize_model(str(model_file_path), meta.framework)

    def get_metadata(self, model_id: str) -> Dict[str, Any]:
        """获取模型元数据"""
        meta = self._get_metadata(model_id)
        return meta.to_dict() if meta else {}

    def delete(self, model_id: str) -> bool:
        """
        删除模型及其元数据

        Args:
            model_id: 模型 ID

        Returns:
            是否成功删除
        """
        meta = self._get_metadata(model_id)
        if meta is None:
            return False

        model_dir = self.base_path / meta.model_name
        version_prefix = f"v{meta.version}"

        deleted = False
        for child in model_dir.iterdir():
            if child.name.startswith(version_prefix):
                child.unlink()
                deleted = True

        # 清理索引
        if model_id in self._index:
            del self._index[model_id]

        logger.info(f"[ModelRegistry] 模型已删除: {model_id}")
        return deleted

    def list_models(
        self,
        model_name: Optional[str] = None,
        framework: Optional[str] = None,
        stage: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        列出符合条件的模型

        Args:
            model_name: 按模型名称过滤
            framework: 按框架过滤
            stage: 按生命周期阶段过滤
            tags: 按标签过滤（AND 逻辑）
            limit: 返回上限
            offset: 偏移量

        Returns:
            元数据列表
        """
        results = []
        for meta in self._index.values():
            if model_name and meta.model_name != model_name:
                continue
            if framework and meta.framework.value != framework:
                continue
            if stage and meta.stage.value != stage:
                continue
            if tags:
                if not all(t in meta.tags for t in tags):
                    continue
            results.append(meta.to_dict())

        # 按 created_at 降序
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results[offset : offset + limit]  # noqa: E203

    def get_version_history(self, model_name: str) -> List[Dict[str, Any]]:
        """
        获取指定模型的版本历史

        Args:
            model_name: 模型名称

        Returns:
            按版本号升序排列的元数据列表
        """
        versions = [
            m.to_dict() for m in self._index.values() if m.model_name == model_name
        ]
        versions.sort(key=lambda x: x.get("version", 0))
        return versions

    def promote(
        self,
        model_id: str,
        target_stage: str,
    ) -> bool:
        """
        将模型提升到新的生命周期阶段

        Args:
            model_id: 模型 ID
            target_stage: 目标阶段 (development/staging/production/archived)

        Returns:
            是否成功
        """
        meta = self._get_metadata(model_id)
        if meta is None:
            return False

        # 如果目标是 production，先将同名的其他 production 模型降级
        if target_stage == ModelStage.PRODUCTION.value:
            for m in self._index.values():
                if (
                    m.model_name == meta.model_name
                    and m.stage == ModelStage.PRODUCTION
                    and m.model_id != model_id
                ):
                    m.stage = ModelStage.STAGING
                    self._persist_metadata(m)

        meta.stage = ModelStage(target_stage)
        meta.updated_at = datetime.now().isoformat()
        self._persist_metadata(meta)

        logger.info(f"[ModelRegistry] 模型 {model_id} 已提升至 {target_stage}")
        return True

    def compare_models(
        self,
        model_ids: List[str],
    ) -> Dict[str, Any]:
        """
        对比多个模型的指标

        Args:
            model_ids: 要对比的模型 ID 列表

        Returns:
            对比结果
        """
        metas = []
        for mid in model_ids:
            meta = self._get_metadata(mid)
            if meta:
                metas.append(meta.to_dict())

        if not metas:
            return {"error": "未找到任何有效模型"}

        # 提取所有指标键
        all_metric_keys = set()
        for m in metas:
            all_metric_keys.update(m.get("metrics", {}).keys())

        comparison = {
            "models": metas,
            "metrics_comparison": {},
        }

        for key in sorted(all_metric_keys):
            values = [(m["model_id"], m.get("metrics", {}).get(key)) for m in metas]
            values = [(mid, v) for mid, v in values if v is not None]
            if values:
                best = max(values, key=lambda x: x[1])
                worst = min(values, key=lambda x: x[1])
                comparison["metrics_comparison"][key] = {
                    "values": {mid: v for mid, v in values},
                    "best_model": best[0],
                    "worst_model": worst[0],
                    "range": abs(best[1] - worst[1]),
                }

        return comparison

    def get_statistics(self) -> Dict[str, Any]:
        """获取注册中心的统计信息"""
        total = len(self._index)
        by_framework = {}
        by_stage = {}
        total_size = 0

        for meta in self._index.values():
            by_framework[meta.framework.value] = (
                by_framework.get(meta.framework.value, 0) + 1
            )
            by_stage[meta.stage.value] = by_stage.get(meta.stage.value, 0) + 1
            total_size += meta.file_size_bytes

        return {
            "total_models": total,
            "by_framework": by_framework,
            "by_stage": by_stage,
            "total_storage_bytes": total_size,
            "storage_human_readable": self._humanize_size(total_size),
            "unique_model_names": len(set(m.model_name for m in self._index.values())),
        }

    # ==================== 内部方法 ====================

    def _get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """从内存索引获取元数据"""
        return self._index.get(model_id)

    def _get_latest_version(self, model_name: str) -> int:
        """获取指定模型的最新版本号"""
        versions = [
            m.version for m in self._index.values() if m.model_name == model_name
        ]
        return max(versions) if versions else 0

    def _persist_metadata(self, meta: ModelMetadata):
        """持久化元数据到磁盘"""
        model_dir = self.base_path / meta.model_name
        meta_file = model_dir / f"v{meta.version}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    def _rebuild_index(self):
        """从磁盘重建内存索引"""
        self._index.clear()
        if not self.base_path.exists():
            return

        for model_dir in self.base_path.iterdir():
            if not model_dir.is_dir():
                continue
            for meta_file in model_dir.glob("*_metadata.json"):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    meta = ModelMetadata.from_dict(data)
                    self._index[meta.model_id] = meta
                except Exception as e:
                    logger.debug(f"加载元数据失败 {meta_file}: {e}")

        logger.info(f"[ModelRegistry] 索引重建完成: {len(self._index)} 个模型")

    @staticmethod
    def _serialize_model(model: Any, path: str, framework: ModelFramework) -> int:
        """根据框架序列化模型"""
        if framework == ModelFramework.XGBOOST:
            model.save_model(path)
        elif framework == ModelFramework.LIGHTGBM:
            model.save_model(path)
        elif framework == ModelFramework.SKLEARN:
            import joblib

            joblib.dump(model, path)
        elif framework == ModelFramework.PYTORCH:
            import torch

            torch.save(model.state_dict(), path)
        else:
            # Generic: 尝试 pickle
            import pickle

            with open(path, "wb") as f:
                pickle.dump(model, f)

        return os.path.getsize(path)

    @staticmethod
    def _deserialize_model(path: str, framework: ModelFramework) -> Any:
        """根据框架反序列化模型"""
        if framework == ModelFramework.XGBOOST:
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(path)
            return model
        elif framework == ModelFramework.LIGHTGBM:
            import lightgbm as lgb

            return lgb.Booster(model_file=path)
        elif framework == ModelFramework.SKLEARN:
            import joblib

            return joblib.load(path)
        elif framework == ModelFramework.PYTORCH:
            import torch

            return torch.load(path, weights_only=False)
        else:
            import pickle

            with open(path, "rb") as f:
                return pickle.load(f)

    @staticmethod
    def _checksum(path: str) -> str:
        """计算 SHA256 校验和"""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _humanize_size(size_bytes: int) -> str:
        """人类可读的文件大小"""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


# ==================== 全局实例 ====================

model_registry = ModelRegistry()

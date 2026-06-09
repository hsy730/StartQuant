"""
生成因子数据访问层
"""
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.models.generated_factor import GeneratedFactorModel


class GeneratedFactorRepository:
    """生成因子数据访问类"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, factor: GeneratedFactorModel) -> GeneratedFactorModel:
        """创建生成因子记录"""
        self.db.add(factor)
        self.db.commit()
        self.db.refresh(factor)
        return factor

    def get_by_id(self, id: int) -> Optional[GeneratedFactorModel]:
        """根据ID获取生成因子"""
        return self.db.get(GeneratedFactorModel, id)

    def get_by_expression(self, expression: str) -> Optional[GeneratedFactorModel]:
        """根据表达式获取生成因子（用于去重）"""
        return self.db.scalar(
            select(GeneratedFactorModel).where(GeneratedFactorModel.expression == expression)
        )

    def get_all_valid(self) -> List[GeneratedFactorModel]:
        """获取所有验证通过的因子"""
        return list(
            self.db.scalars(
                select(GeneratedFactorModel)
                .where(GeneratedFactorModel.is_valid == True)
                .order_by(GeneratedFactorModel.created_at.desc())
            ).all()
        )

    def get_all_unsaved(self) -> List[GeneratedFactorModel]:
        """获取所有未保存到因子库的因子"""
        return list(
            self.db.scalars(
                select(GeneratedFactorModel)
                .where(GeneratedFactorModel.is_saved == False)
                .order_by(GeneratedFactorModel.created_at.desc())
            ).all()
        )

    def get_all(self) -> List[GeneratedFactorModel]:
        """获取所有生成因子记录"""
        return list(
            self.db.scalars(
                select(GeneratedFactorModel).order_by(GeneratedFactorModel.created_at.desc())
            ).all()
        )

    def mark_saved(self, id: int, factor_name: str) -> bool:
        """标记为已保存到因子库

        Args:
            id: 因子ID
            factor_name: 保存到因子库时使用的名称

        Returns:
            是否标记成功
        """
        factor = self.get_by_id(id)
        if not factor:
            return False
        factor.is_saved = True
        factor.factor_name = factor_name
        self.db.commit()
        self.db.refresh(factor)
        return True

    def delete(self, id: int) -> bool:
        """删除生成因子记录

        Args:
            id: 因子ID

        Returns:
            是否删除成功
        """
        factor = self.get_by_id(id)
        if not factor:
            return False
        self.db.delete(factor)
        self.db.commit()
        return True

    def to_dict(self, factor: GeneratedFactorModel) -> Dict:
        """将生成因子模型转换为字典"""
        return {
            "id": factor.id,
            "expression": factor.expression,
            "generation_method": factor.generation_method,
            "ic_value": factor.ic_value,
            "ir_value": factor.ir_value,
            "turnover_value": factor.turnover_value,
            "stability_score": factor.stability_score,
            "validation_score": factor.validation_score,
            "is_valid": factor.is_valid,
            "is_saved": factor.is_saved,
            "factor_name": factor.factor_name,
            "complexity": factor.complexity,
            "extra_info": factor.extra_info,
            "created_at": factor.created_at.isoformat() if factor.created_at else None,
        }

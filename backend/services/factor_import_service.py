"""
因子导入服务 - 从CSV文件导入因子
"""
import logging

logger = logging.getLogger(__name__)
import pandas as pd
from typing import Dict
from pathlib import Path

from backend.core.database import get_db
from backend.repositories.factor_repository import FactorRepository
from backend.models.factor import FactorModel


class FactorImportService:
    """因子导入服务类"""

    def __init__(self):
        pass

    def import_from_csv(
        self,
        csv_file_path: str,
        factor_name: str,
        description: str = "",
        category: str = "导入",
        date_column: str = "date",
        factor_column: str = "factor_value",
    ) -> Dict:
        """
        从CSV文件导入因子

        Args:
            csv_file_path: CSV文件路径
            factor_name: 因子名称
            description: 因子描述
            category: 因子分类
            date_column: 日期列名
            factor_column: 因子值列名

        Returns:
            导入结果信息
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)

            # 验证必需列
            if date_column not in df.columns:
                raise ValueError(f"CSV文件中缺少日期列: {date_column}")
            if factor_column not in df.columns:
                raise ValueError(f"CSV文件中缺少因子值列: {factor_column}")

            # 验证数据
            if len(df) == 0:
                raise ValueError("CSV文件为空")

            # 生成因子代码（表达式形式）
            factor_code = self._generate_import_code(date_column, factor_column)

            # 保存到数据库
            with get_db() as db:
                repo = FactorRepository(db)

                # 检查因子是否已存在
                existing = repo.get_by_name(factor_name)
                if existing:
                    raise ValueError(f"因子 '{factor_name}' 已存在")

                # 创建因子
                factor = FactorModel(
                    name=factor_name,
                    code=factor_code,
                    description=description or f"从CSV文件导入: {Path(csv_file_path).name}",
                    source="user",
                    category=category,
                    is_active=1,
                )

                repo.create(factor)

                return {
                    "success": True,
                    "factor_id": factor.id,
                    "factor_name": factor.name,
                    "row_count": len(df),
                    "date_range": f"{df[date_column].min()} 至 {df[date_column].max()}",
                    "message": f"成功导入因子 '{factor_name}'，共 {len(df)} 条数据",
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"导入失败: {e}",
            }

    def import_from_dataframe(
        self,
        df: pd.DataFrame,
        factor_name: str,
        description: str = "",
        category: str = "导入",
        date_column: str = "date",
        factor_column: str = "factor_value",
    ) -> Dict:
        """
        从DataFrame导入因子

        Args:
            df: 包含日期和因子值的DataFrame
            factor_name: 因子名称
            description: 因子描述
            category: 因子分类
            date_column: 日期列名
            factor_column: 因子值列名

        Returns:
            导入结果信息
        """
        try:
            # 验证必需列
            if date_column not in df.columns:
                raise ValueError(f"DataFrame中缺少日期列: {date_column}")
            if factor_column not in df.columns:
                raise ValueError(f"DataFrame中缺少因子值列: {factor_column}")

            # 验证数据
            if len(df) == 0:
                raise ValueError("DataFrame为空")

            # 生成因子代码
            factor_code = self._generate_import_code(date_column, factor_column)

            # 保存到数据库
            with get_db() as db:
                repo = FactorRepository(db)

                # 检查因子是否已存在
                existing = repo.get_by_name(factor_name)
                if existing:
                    raise ValueError(f"因子 '{factor_name}' 已存在")

                # 创建因子
                factor = FactorModel(
                    name=factor_name,
                    code=factor_code,
                    description=description or "从DataFrame导入",
                    source="user",
                    category=category,
                    is_active=1,
                )

                repo.create(factor)

                return {
                    "success": True,
                    "factor_id": factor.id,
                    "factor_name": factor.name,
                    "row_count": len(df),
                    "message": f"成功导入因子 '{factor_name}'，共 {len(df)} 条数据",
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"导入失败: {e}",
            }

    def validate_csv_format(
        self,
        csv_file_path: str,
        date_column: str = "date",
        factor_column: str = "factor_value",
    ) -> Dict:
        """
        验证CSV文件格式

        Args:
            csv_file_path: CSV文件路径
            date_column: 期望的日期列名
            factor_column: 期望的因子值列名

        Returns:
            验证结果
        """
        try:
            df = pd.read_csv(csv_file_path)

            result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "info": {},
            }

            # 检查列
            if date_column not in df.columns:
                result["valid"] = False
                result["errors"].append(f"缺少日期列: {date_column}")

            if factor_column not in df.columns:
                result["valid"] = False
                result["errors"].append(f"缺少因子值列: {factor_column}")

            # 检查数据
            if len(df) == 0:
                result["valid"] = False
                result["errors"].append("CSV文件为空")

            # 检查空值
            if date_column in df.columns:
                null_count = df[date_column].isnull().sum()
                if null_count > 0:
                    result["warnings"].append(f"日期列有 {null_count} 个空值")

            if factor_column in df.columns:
                null_count = df[factor_column].isnull().sum()
                if null_count > 0:
                    result["warnings"].append(f"因子值列有 {null_count} 个空值")

                # 检查是否为数值类型
                try:
                    pd.to_numeric(df[factor_column], errors="coerce")
                except Exception as e:
                    logger.warning(f"因子值列包含非数值数据: {e}")
                    result["warnings"].append("因子值列包含非数值数据")

            # 添加基本信息
            result["info"] = {
                "row_count": len(df),
                "columns": list(df.columns),
            }

            return result

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"文件读取失败: {e}"],
                "warnings": [],
                "info": {},
            }

    def _generate_import_code(self, date_column: str, factor_column: str) -> str:
        """
        生成导入因子的代码

        Args:
            date_column: 日期列名
            factor_column: 因子值列名

        Returns:
            因子计算代码
        """
        # 返回一个函数形式的代码
        return f'''def calculate_factor(df):
    """
    从导入的CSV数据中读取因子值
    日期列: {date_column}
    因子值列: {factor_column}
    """
    import pandas as pd

    # 确保索引是日期
    if "{date_column}" in df.columns:
        df = df.set_index("{date_column}")
        df.index = pd.to_datetime(df.index)

    # 返回因子值列
    if "{factor_column}" in df.columns:
        return pd.to_numeric(df["{factor_column}"], errors="coerce")
    else:
        raise ValueError(f"列 '{factor_column}' 不存在")
'''

    def get_import_template(self) -> pd.DataFrame:
        """
        获取导入模板

        Returns:
            示例DataFrame
        """
        import pandas as pd
        from datetime import datetime, timedelta

        # 创建示例数据
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        factor_values = [0.5 + i * 0.1 for i in range(10)]

        df = pd.DataFrame({
            "date": dates,
            "factor_value": factor_values,
        })

        return df


# 全局因子导入服务实例
factor_import_service = FactorImportService()

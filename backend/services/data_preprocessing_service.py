"""
数据预处理服务 - 异常值检测、数据清洗、增量更新
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataPreprocessingService:
    """数据预处理服务类"""

    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        n_sigma: float = 3.0,
        method: str = "std",
    ) -> pd.Series:
        """
        检测异常值

        Args:
            df: 数据框
            column: 要检测的列名
            n_sigma: 标准差倍数（默认3倍）
            method: 检测方法，"std"（标准差）或 "iqr"（四分位距）

        Returns:
            布尔序列，True表示该点是异常值
        """
        if column not in df.columns:
            raise ValueError(f"列 '{column}' 不存在于数据框中")

        if method == "std":
            # 3σ原则检测
            mean = df[column].mean()
            std = df[column].std()
            lower_bound = mean - n_sigma * std
            upper_bound = mean + n_sigma * std
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            return outliers

        elif method == "iqr":
            # 四分位距检测
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            return outliers

        else:
            raise ValueError(f"不支持的检测方法: {method}")

    def handle_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "clip",
        n_sigma: float = 3.0,
        detection_method: str = "std",
    ) -> pd.DataFrame:
        """
        处理异常值

        Args:
            df: 数据框
            column: 要处理的列名
            method: 处理方法
                - "clip": 截断到边界值
                - "remove": 删除异常值所在的行
                - "replace": 替换为均值
                - "replace_median": 替换为中位数
            n_sigma: 标准差倍数
            detection_method: 异常值检测方法

        Returns:
            处理后的数据框
        """
        df = df.copy()

        # 检测异常值
        outliers = self.detect_outliers(df, column, n_sigma, detection_method)

        if method == "clip":
            # 截断到边界值（根据检测方法选择对应的边界计算方式）
            if detection_method == "iqr":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            else:  # std
                mean = df[column].mean()
                std = df[column].std()
                lower_bound = mean - n_sigma * std
                upper_bound = mean + n_sigma * std
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound

        elif method == "remove":
            # 删除异常值所在的行
            df = df[~outliers].copy()

        elif method == "replace":
            # 替换为均值（排除异常值后计算均值）
            mean_value = df.loc[~outliers, column].mean() if (~outliers).any() else df[column].mean()
            df.loc[outliers, column] = mean_value

        elif method == "replace_median":
            # 替换为中位数
            median_value = df[column].median()
            df.loc[outliers, column] = median_value

        else:
            raise ValueError(f"不支持的处理方法: {method}")

        return df

    def incremental_update(
        self,
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
        on: str = "date",
        how: str = "outer",
    ) -> pd.DataFrame:
        """
        增量更新数据

        Args:
            existing_df: 现有的数据框
            new_df: 新增的数据框
            on: 合并的键（通常是日期列）
            how: 合并方式，"outer"（并集）或 "inner"（交集）

        Returns:
            合并后的数据框
        """
        existing_df = existing_df.copy()
        new_df = new_df.copy()

        # 确保索引是日期类型
        if not isinstance(existing_df.index, pd.DatetimeIndex):
            if on in existing_df.columns:
                existing_df = existing_df.set_index(on)
                existing_df.index = pd.to_datetime(existing_df.index)

        if not isinstance(new_df.index, pd.DatetimeIndex):
            if on in new_df.columns:
                new_df = new_df.set_index(on)
                new_df.index = pd.to_datetime(new_df.index)

        # 找出新数据中的新增日期
        existing_dates = set(existing_df.index)
        new_dates = set(new_df.index)

        added_dates = new_dates - existing_dates

        if len(added_dates) == 0:
            # 没有新数据，返回原数据
            return existing_df.sort_index()

        # 合并数据，新数据覆盖旧数据
        combined_df = pd.concat([existing_df, new_df], axis=0)

        # 去重（保留最新的）
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

        return combined_df.sort_index()

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        required_columns: list = None,
    ) -> Tuple[bool, str]:
        """
        验证数据质量

        Args:
            df: 数据框
            required_columns: 必需的列名列表

        Returns:
            (是否通过验证, 错误消息)
        """
        # 检查必需列
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return False, f"缺少必需的列: {missing_columns}"

        # 检查空值
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            null_info = null_counts[null_counts > 0].to_dict()
            return False, f"存在空值: {null_info}"

        # 检查无穷值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                return False, f"列 '{col}' 包含无穷值"

        # 检查数据框是否为空
        if len(df) == 0:
            return False, "数据框为空"

        return True, "数据质量验证通过"

    def standardize_columns(
        self,
        df: pd.DataFrame,
        column_mapping: dict = None,
    ) -> pd.DataFrame:
        """
        标准化列名

        Args:
            df: 数据框
            column_mapping: 列名映射字典，如果为None则使用默认映射

        Returns:
            标准化后的数据框
        """
        df = df.copy()

        # 默认的列名映射（中文到英文）
        default_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }

        mapping = column_mapping or default_mapping

        # 重命名列
        df = df.rename(columns=mapping)

        # 确保日期列是datetime类型
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        # 确保数值列是正确的类型
        numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_index()

    def fill_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
    ) -> pd.DataFrame:
        """
        填充缺失值

        Args:
            df: 数据框
            method: 填充方法
                - "ffill": 前向填充
                - "bfill": 后向填充
                - "interpolate": 线性插值
                - "mean": 均值填充

        Returns:
            填充后的数据框
        """
        df = df.copy()

        if method == "ffill":
            # 使用新API替代已弃用的fillna(method="ffill")
            df = df.ffill()
        elif method == "bfill":
            # 使用新API替代已弃用的fillna(method="bfill")
            df = df.bfill()
        elif method == "interpolate":
            df = df.interpolate(method="linear")
        elif method == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # 避免使用inplace参数（pandas 2.0+不推荐）
                df[col] = df[col].fillna(df[col].mean())
        else:
            raise ValueError(f"不支持的填充方法: {method}")

        return df

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[list] = None,
        keep: str = "last",
    ) -> pd.DataFrame:
        """
        删除重复行

        Args:
            df: 数据框
            subset: 用于识别重复的列名列表，None表示使用所有列
            keep: 保留哪个重复值，"first" 或 "last"

        Returns:
            删除重复后的数据框
        """
        return df.drop_duplicates(subset=subset, keep=keep)

    def detect_and_handle_anomalies(
        self,
        df: pd.DataFrame,
        price_columns: list = None,
        n_sigma: float = 3.0,
        handle_method: str = "clip",
    ) -> Tuple[pd.DataFrame, dict]:
        """
        检测并处理价格数据的异常值

        Args:
            df: 数据框
            price_columns: 价格列名列表，默认为 ["open", "high", "low", "close"]
            n_sigma: 标准差倍数
            handle_method: 处理方法

        Returns:
            (处理后的数据框, 异常值统计信息)
        """
        if price_columns is None:
            price_columns = ["open", "high", "low", "close"]

        df = df.copy()
        stats = {
            "total_outliers": 0,
            "outliers_by_column": {},
        }

        for col in price_columns:
            if col in df.columns:
                outliers = self.detect_outliers(df, col, n_sigma)
                outlier_count = outliers.sum()
                stats["outliers_by_column"][col] = int(outlier_count)
                stats["total_outliers"] += outlier_count

                if outlier_count > 0:
                    df = self.handle_outliers(
                        df,
                        col,
                        method=handle_method,
                        n_sigma=n_sigma,
                    )

        return df, stats


# 全局数据预处理服务实例
data_preprocessing_service = DataPreprocessingService()

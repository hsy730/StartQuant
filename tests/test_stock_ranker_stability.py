"""
StockRankerService 计算逻辑稳定性 — 单元测试

重点覆盖：
1. _split_groups 分组分割算法（含组边界对齐）
2. 特征缺失值填充逻辑
3. 排名方法正确性
4. 训练期重叠检测
5. 边界情况与容错
"""

import pytest
import numpy as np
import pandas as pd


class TestSplitGroups:
    """_split_groups 分组分割算法测试"""

    @staticmethod
    def _split_groups(groups, total_rows, split_idx):
        """直接调用静态方法"""
        from backend.services.stock_ranker_service import StockRankerService

        return StockRankerService._split_groups(groups, total_rows, split_idx)

    def test_exact_boundary_split(self):
        """split_idx 恰好在组边界上 → 无需调整"""
        groups = [100, 100, 100, 100, 100]
        train_g, valid_g, adj_idx = self._split_groups(groups, 500, 300)
        assert train_g == [100, 100, 100]
        assert valid_g == [100, 100]
        assert adj_idx == 300
        assert sum(train_g) == adj_idx
        assert sum(valid_g) == 500 - adj_idx

    def test_straddle_group_assigned_to_train(self):
        """split_idx 落在组内部 → 整组归训练集，adjusted_split_idx 对齐到组末尾"""
        groups = [100, 100, 100, 100, 100]
        train_g, valid_g, adj_idx = self._split_groups(groups, 500, 350)
        # 第4组(索引3)跨越350：cumulative=300, cumulative+g=400 > 350
        assert train_g == [100, 100, 100, 100]
        assert valid_g == [100]
        assert adj_idx == 400  # 调整到第4组末尾
        assert sum(train_g) == adj_idx
        assert sum(valid_g) == 500 - adj_idx

    def test_split_at_very_beginning(self):
        """split_idx=0 → 所有组归验证集（但容错保证至少1个训练组）"""
        groups = [50, 50, 50]
        train_g, valid_g, adj_idx = self._split_groups(groups, 150, 0)
        # 容错：至少1个训练组
        assert len(train_g) >= 1
        assert sum(train_g) == adj_idx

    def test_split_at_very_end(self):
        """split_idx=total_rows → 所有组归训练集（但容错保证至少1个验证组）"""
        groups = [50, 50, 50]
        train_g, valid_g, adj_idx = self._split_groups(groups, 150, 150)
        # 容错：至少1个验证组
        assert len(valid_g) >= 1

    def test_unequal_group_sizes(self):
        """不等大小的组"""
        groups = [30, 70, 50, 100, 80]
        train_g, valid_g, adj_idx = self._split_groups(groups, 330, 150)
        # cumulative: 0→30→100→150→250→330
        # split_idx=150 恰好在第3组末尾
        assert train_g == [30, 70, 50]
        assert valid_g == [100, 80]
        assert adj_idx == 150
        assert sum(train_g) == adj_idx
        assert sum(valid_g) == 330 - adj_idx

    def test_unequal_groups_straddle(self):
        """不等大小的组，且跨越分割点"""
        groups = [30, 70, 50, 100, 80]
        train_g, valid_g, adj_idx = self._split_groups(groups, 330, 120)
        # cumulative: 0→30→100→150
        # 第3组: cumulative=100, cumulative+g=150 > 120, 100 < 120 → 跨越 → train
        assert train_g == [30, 70, 50]
        assert valid_g == [100, 80]
        assert adj_idx == 150  # 调整到第3组末尾
        assert sum(train_g) == adj_idx
        assert sum(valid_g) == 330 - adj_idx

    def test_single_group(self):
        """只有1个组"""
        groups = [200]
        train_g, valid_g, adj_idx = self._split_groups(groups, 200, 100)
        # 唯一的组跨越分割点 → 归训练集
        assert train_g == [200]
        assert adj_idx == 200
        # 只有1个组时无法保证有验证组

    def test_two_groups(self):
        """2个组"""
        groups = [80, 120]
        train_g, valid_g, adj_idx = self._split_groups(groups, 200, 80)
        # split_idx=80 恰好在第1组末尾
        assert train_g == [80]
        assert valid_g == [120]
        assert adj_idx == 80

    def test_empty_groups(self):
        """空组列表"""
        train_g, valid_g, adj_idx = self._split_groups([], 0, 0)
        assert train_g == []
        assert valid_g == []
        assert adj_idx == 0

    def test_group_sum_equals_dmatrix_rows(self):
        """核心不变量：train_groups 总和 == adjusted_split_idx，valid_groups 总和 == total_rows - adjusted_split_idx"""
        for groups, split_idx in [
            ([100] * 5, 250),
            ([100] * 5, 350),
            ([50, 150, 80, 120], 200),
            ([50, 150, 80, 120], 50),
            ([50, 150, 80, 120], 300),
            ([10] * 20, 100),
            ([10] * 20, 95),  # 跨越
        ]:
            total = sum(groups)
            train_g, valid_g, adj_idx = self._split_groups(groups, total, split_idx)
            assert sum(train_g) == adj_idx, (
                f"train_groups sum {sum(train_g)} != adjusted_split_idx {adj_idx} "
                f"for groups={groups}, split_idx={split_idx}"
            )
            assert sum(valid_g) == total - adj_idx, (
                f"valid_groups sum {sum(valid_g)} != {total - adj_idx} " f"for groups={groups}, split_idx={split_idx}"
            )


class TestFeatureMissingValueFilling:
    """特征缺失值填充逻辑测试"""

    def test_feature_nan_filled_with_median(self):
        """特征列 NaN 应被中位数填充，而非删除整行"""
        pytest.importorskip("xgboost")
        from backend.services.stock_ranker_service import StockRankerService, RankTrainingConfig

        ranker = StockRankerService(
            default_config=RankTrainingConfig(
                n_estimators=3,
                max_depth=2,
                early_stopping_rounds=2,
            )
        )

        np.random.seed(42)
        n = 300
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n, freq="B"),
                "stock_code": [f"{i % 10:06d}" for i in range(n)],
                "feature_1": np.random.randn(n),
                "feature_2": np.random.randn(n),
                "forward_return_5d": np.random.randn(n) * 0.02,
            }
        )

        # 在 feature_1 中引入 20% NaN
        nan_mask = np.random.rand(n) < 0.2
        df.loc[nan_mask, "feature_1"] = np.nan

        original_len = len(df)
        result = ranker.train(
            feature_df=df,
            label_col="forward_return_5d",
            date_col="date",
            group_col="date",
            config=RankTrainingConfig(
                objective="reg:squarederror",
                n_estimators=3,
                max_depth=2,
            ),
            enable_bias_check=False,
        )

        # 训练应成功，且样本数应接近原始数据量（仅 label NaN 的行被删除）
        assert result.status.value == "ready"
        assert result.n_samples >= original_len * 0.9  # 不应丢失太多数据

    def test_label_nan_rows_dropped(self):
        """标签列 NaN 的行应被删除"""
        pytest.importorskip("xgboost")
        from backend.services.stock_ranker_service import StockRankerService, RankTrainingConfig

        ranker = StockRankerService(
            default_config=RankTrainingConfig(
                n_estimators=3,
                max_depth=2,
                early_stopping_rounds=2,
            )
        )

        np.random.seed(42)
        n = 300
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n, freq="B"),
                "stock_code": [f"{i % 10:06d}" for i in range(n)],
                "feature_1": np.random.randn(n),
                "forward_return_5d": np.random.randn(n) * 0.02,
            }
        )

        # 在 label 中引入 10% NaN
        nan_mask = np.random.rand(n) < 0.1
        df.loc[nan_mask, "forward_return_5d"] = np.nan

        result = ranker.train(
            feature_df=df,
            label_col="forward_return_5d",
            date_col="date",
            group_col="date",
            config=RankTrainingConfig(
                objective="reg:squarederror",
                n_estimators=3,
                max_depth=2,
            ),
            enable_bias_check=False,
        )

        # label NaN 的行应被删除
        expected_max = n - nan_mask.sum()
        assert result.n_samples <= expected_max + 5  # 允许少量误差


class TestRankMethod:
    """排名方法正确性测试"""

    def test_rank_position_uses_first_method(self):
        """排名应使用 method='first'，并列时按出现顺序分配唯一整数排名"""
        pytest.importorskip("xgboost")
        from backend.services.stock_ranker_service import StockRankerService, RankTrainingConfig

        ranker = StockRankerService(
            default_config=RankTrainingConfig(
                n_estimators=3,
                max_depth=2,
                early_stopping_rounds=2,
            )
        )

        np.random.seed(42)
        n = 300
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n, freq="B"),
                "stock_code": [f"{i % 10:06d}" for i in range(n)],
                "feature_1": np.random.randn(n),
                "forward_return_5d": np.random.randn(n) * 0.02,
            }
        )

        result = ranker.train(
            feature_df=df,
            label_col="forward_return_5d",
            date_col="date",
            group_col="date",
            config=RankTrainingConfig(
                objective="reg:squarederror",
                n_estimators=3,
                max_depth=2,
            ),
            enable_bias_check=False,
        )

        today_df = df[df["date"] == df["date"].max()].copy()
        prediction = ranker.predict(model_id=result.model_id, features=today_df, top_n=5)

        # rank_position 应全部为整数（无小数）
        positions = prediction.predictions["rank_position"].values
        assert all(
            p == int(p) for p in positions
        ), f"rank_position 包含非整数值: {positions[positions != positions.astype(int)]}"


class TestTrainPeriodOverlapDetection:
    """训练期重叠检测测试"""

    def test_overlap_detection_with_training_period(self):
        """回测数据与训练期重叠时应产生警告日志"""
        pytest.importorskip("xgboost")
        from backend.services.stock_ranker_service import StockRankerService, RankTrainingConfig
        from unittest.mock import patch

        ranker = StockRankerService(
            default_config=RankTrainingConfig(
                n_estimators=3,
                max_depth=2,
                early_stopping_rounds=2,
            )
        )

        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "date": dates,
                "stock_code": [f"{i % 10:06d}" for i in range(n)],
                "feature_1": np.random.randn(n),
                "close": 10 + np.cumsum(np.random.randn(n) * 0.5),
                "forward_return_5d": np.random.randn(n) * 0.02,
            }
        )

        result = ranker.train(
            feature_df=df,
            label_col="forward_return_5d",
            date_col="date",
            group_col="date",
            config=RankTrainingConfig(
                objective="reg:squarederror",
                n_estimators=3,
                max_depth=2,
            ),
            enable_bias_check=False,
        )

        # 验证 train_period 已保存到 metadata
        metadata = ranker._get_model_metadata(result.model_id)
        assert metadata.get("train_period") is not None
        assert "~" in metadata["train_period"]

        # 使用训练期数据做回测，验证重叠警告被触发
        import backend.services.stock_ranker_service as srv_module

        with patch.object(srv_module.logger, "warning") as mock_warn:
            try:
                ranker.predict_and_backtest(
                    model_id=result.model_id,
                    feature_history=df,
                    date_col="date",
                    stock_col="stock_code",
                    price_col="close",
                    top_n=3,
                )
            except Exception:
                pass  # 回测可能因数据不足失败，不影响警告检测
            # 应至少有一条重叠警告
            warn_calls = [str(c) for c in mock_warn.call_args_list]
            overlap_warns = [c for c in warn_calls if "重叠" in c or "in-sample" in c]
            assert len(overlap_warns) > 0, f"未检测到训练期重叠警告，实际调用: {warn_calls}"


class TestEdgeCases:
    """边界情况测试"""

    def test_split_groups_all_same_size(self):
        """所有组大小相同"""
        from backend.services.stock_ranker_service import StockRankerService

        groups = [50] * 10
        for split_ratio in [0.1, 0.2, 0.5, 0.8, 0.9]:
            split_idx = int(sum(groups) * split_ratio)
            train_g, valid_g, adj_idx = StockRankerService._split_groups(groups, sum(groups), split_idx)
            assert sum(train_g) == adj_idx
            assert sum(valid_g) == sum(groups) - adj_idx

    def test_split_groups_large_first_group(self):
        """第一个组特别大"""
        from backend.services.stock_ranker_service import StockRankerService

        groups = [500, 50, 50]
        train_g, valid_g, adj_idx = StockRankerService._split_groups(groups, 600, 100)
        # 第1组跨越 → 归训练集
        assert train_g == [500]
        assert adj_idx == 500
        assert sum(train_g) == adj_idx

    def test_split_groups_large_last_group(self):
        """最后一个组特别大"""
        from backend.services.stock_ranker_service import StockRankerService

        groups = [50, 50, 500]
        train_g, valid_g, adj_idx = StockRankerService._split_groups(groups, 600, 100)
        # 第3组在验证集
        assert train_g == [50, 50]
        assert valid_g == [500]
        assert adj_idx == 100

    def test_predict_with_tied_scores(self):
        """预测时存在并列分数，排名应为整数"""
        pytest.importorskip("xgboost")
        from backend.services.stock_ranker_service import StockRankerService, RankTrainingConfig

        ranker = StockRankerService(
            default_config=RankTrainingConfig(
                n_estimators=3,
                max_depth=2,
                early_stopping_rounds=2,
            )
        )

        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n, freq="B"),
                "stock_code": [f"{i % 10:06d}" for i in range(n)],
                "feature_1": np.random.randn(n),
                "forward_return_5d": np.random.randn(n) * 0.02,
            }
        )

        result = ranker.train(
            feature_df=df,
            label_col="forward_return_5d",
            date_col="date",
            group_col="date",
            config=RankTrainingConfig(
                objective="reg:squarederror",
                n_estimators=3,
                max_depth=2,
            ),
            enable_bias_check=False,
        )

        # 构造含并列分数的预测数据
        today_df = df[df["date"] == df["date"].max()].copy()
        # 强制部分分数相同
        today_df["feature_1"] = 0.0  # 所有特征相同 → 分数可能相同
        prediction = ranker.predict(model_id=result.model_id, features=today_df, top_n=5)

        # 即使分数相同，rank_position 也应为整数
        positions = prediction.predictions["rank_position"].values
        assert all(p == int(p) for p in positions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

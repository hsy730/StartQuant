"""
Mining算法注册表参数传递回归测试

防护Bug: unified_factory未传递gflownet/deep_implicit/tree_prescreen的算法特定参数，
导致服务使用默认值（如gflownet_n_iterations=50而非用户设定的5）。
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services.mining_algorithm_registry import create_algorithm  # noqa: E402


class DummyRequest:
    """模拟API请求对象"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestUnifiedFactoryParameterPassing:
    """统一工厂参数传递测试"""

    def test_gflownet_params_passed_to_service(self):
        """gflownet特定参数应正确传递到服务"""
        req = DummyRequest(
            algorithm="gflownet",
            n_generations=3,
            population_size=10,
            gflownet_n_trajectories=50,
            gflownet_n_iterations=5,
            gflownet_hidden_dim=64,
            gflownet_learning_rate=1e-3,
            gflownet_max_expression_depth=3,
            gflownet_temperature=1.0,
            gflownet_reward_scale=10.0,
            gflownet_buffer_size=100,
        )

        mock_data = MagicMock()
        mock_fs = MagicMock()
        mock_fs.calculator = MagicMock()

        # 创建服务时传入的参数应被正确设置
        service = create_algorithm(
            "gflownet",
            task_id="test",
            request=req,
            data=mock_data,
            base_factor_codes=["RSI", "SMA", "MACD"],
            factor_service=mock_fs,
            stock_codes=["000001.SZ"],
            logger=MagicMock(),
        )

        assert service is not None
        # 验证gflownet参数被正确传递（通过_dual_mining_service._gflownet_params间接验证）
        assert service._gflownet_params["n_iterations"] == 5
        assert service._gflownet_params["n_trajectories"] == 50
        assert service._gflownet_params["hidden_dim"] == 64
        assert service._gflownet_params["learning_rate"] == 1e-3
        assert service._gflownet_params["max_expression_depth"] == 3
        assert service._gflownet_params["temperature"] == 1.0
        assert service._gflownet_params["reward_scale"] == 10.0
        assert service._gflownet_params["buffer_size"] == 100

    def test_deep_implicit_params_passed_to_service(self):
        """deep_implicit特定参数应正确传递到服务"""
        req = DummyRequest(
            algorithm="deep_implicit",
            n_generations=3,
            population_size=10,
            deep_d_model=32,
            deep_n_heads=2,
            deep_n_layers=1,
            deep_d_ff=64,
            deep_n_latent_factors=2,
            deep_dropout=0.1,
            deep_seq_length=10,
            deep_learning_rate=1e-4,
            deep_n_epochs=3,
            deep_batch_size=8,
            deep_weight_decay=1e-5,
            deep_early_stopping_patience=2,
        )

        mock_data = MagicMock()
        mock_fs = MagicMock()
        mock_fs.calculator = MagicMock()

        service = create_algorithm(
            "deep_implicit",
            task_id="test",
            request=req,
            data=mock_data,
            base_factor_codes=["RSI", "SMA", "MACD"],
            factor_service=mock_fs,
            stock_codes=["000001.SZ"],
            logger=MagicMock(),
        )

        assert service is not None
        assert service._deep_factor_params["d_model"] == 32
        assert service._deep_factor_params["n_heads"] == 2
        assert service._deep_factor_params["n_layers"] == 1
        assert service._deep_factor_params["d_ff"] == 64
        assert service._deep_factor_params["n_latent_factors"] == 2
        assert service._deep_factor_params["dropout"] == 0.1
        assert service._deep_factor_params["seq_length"] == 10
        assert service._deep_factor_params["learning_rate"] == 1e-4
        assert service._deep_factor_params["n_epochs"] == 3
        assert service._deep_factor_params["batch_size"] == 8
        assert service._deep_factor_params["weight_decay"] == 1e-5
        assert service._deep_factor_params["early_stopping_patience"] == 2

    def test_tree_prescreen_params_passed_to_service(self):
        """tree_prescreen特定参数应正确传递到服务"""
        req = DummyRequest(
            algorithm="tree_prescreen",
            n_generations=3,
            population_size=10,
            tree_model_type="auto",
            top_k=5,
            importance_threshold=0.01,
            tree_n_estimators=10,
            tree_max_depth=3,
            downstream_algorithm="genetic",
        )

        mock_data = MagicMock()
        mock_fs = MagicMock()
        mock_fs.calculator = MagicMock()

        service = create_algorithm(
            "tree_prescreen",
            task_id="test",
            request=req,
            data=mock_data,
            base_factor_codes=["RSI", "SMA", "MACD"],
            factor_service=mock_fs,
            stock_codes=["000001.SZ"],
            logger=MagicMock(),
        )

        assert service is not None
        assert service._tree_prescreen_params["tree_model_type"] == "auto"
        assert service._tree_prescreen_params["top_k"] == 5
        assert service._tree_prescreen_params["importance_threshold"] == 0.01
        assert service._tree_prescreen_params["tree_n_estimators"] == 10
        assert service._tree_prescreen_params["tree_max_depth"] == 3
        assert service._tree_prescreen_params["downstream_algorithm"] == "genetic"

    def test_default_params_not_override_user_values(self):
        """用户传入的参数不应被默认值覆盖"""
        req = DummyRequest(
            algorithm="deep_implicit",
            n_generations=3,
            population_size=10,
            deep_n_epochs=3,  # 用户明确设定为3
        )

        mock_data = MagicMock()
        mock_fs = MagicMock()
        mock_fs.calculator = MagicMock()

        service = create_algorithm(
            "deep_implicit",
            task_id="test",
            request=req,
            data=mock_data,
            base_factor_codes=["RSI"],
            factor_service=mock_fs,
            stock_codes=["000001.SZ"],
            logger=MagicMock(),
        )

        assert service is not None
        # 用户传入的3应被使用，而不是默认值50
        assert service._deep_factor_params["n_epochs"] == 3
        assert service._deep_factor_params["n_epochs"] != 50

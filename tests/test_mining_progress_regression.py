"""
因子挖掘进度防倒退回归测试

防护Bug: _update_progress在_set_phase预设50%后，后续回调可能将进度设得更低，
导致用户看到进度从50%跳回6%的异常现象。
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 导入被测函数
from backend.api.routers.mining import _update_progress  # noqa: E402


class TestMiningProgressRegression:
    """挖掘进度防倒退测试"""

    @pytest.fixture(autouse=True)
    def setup_task_store(self):
        """每个测试前重置任务存储"""
        from backend.api.routers.mining import mining_tasks

        self.task_id = "test_progress_task"
        mining_tasks[self.task_id] = {
            "progress": 0,
            "current_generation": 0,
            "total_generations": 5,
            "best_fitness": 0.0,
            "avg_fitness": 0.0,
        }
        yield
        # 清理
        if self.task_id in mining_tasks:
            del mining_tasks[self.task_id]

    def test_progress_does_not_regress(self):
        """进度不应倒退：新计算值低于当前值时应保持当前值"""
        from backend.api.routers.mining import mining_tasks

        # 模拟_set_phase设置了50%进度
        mining_tasks[self.task_id]["progress"] = 50

        # 后续回调计算出的新进度为20%
        _update_progress(
            self.task_id,
            gen=1,
            total_gen=5,
            best_fitness=0.1,
            avg_fitness=0.05,
            algorithm="gflownet",
            logger=MagicMock(),
        )

        # 进度应保持50%，不应倒退到20%
        assert mining_tasks[self.task_id]["progress"] == 50, (
            f"进度不应倒退，但得到 {mining_tasks[self.task_id]['progress']}"
        )

    def test_progress_can_advance(self):
        """进度应能正常前进"""
        from backend.api.routers.mining import mining_tasks

        mining_tasks[self.task_id]["progress"] = 20

        _update_progress(
            self.task_id,
            gen=3,
            total_gen=5,
            best_fitness=0.3,
            avg_fitness=0.15,
            algorithm="gflownet",
            logger=MagicMock(),
        )

        # 进度应从20%前进到60%
        assert mining_tasks[self.task_id]["progress"] == 60, (
            f"进度应前进到60%，但得到 {mining_tasks[self.task_id]['progress']}"
        )

    def test_progress_capped_at_99(self):
        """进度不应超过99%（验证阶段不超过99%）"""
        from backend.api.routers.mining import mining_tasks

        mining_tasks[self.task_id]["progress"] = 50

        _update_progress(
            self.task_id,
            gen=10,
            total_gen=5,
            best_fitness=0.5,
            avg_fitness=0.25,
            algorithm="genetic",
            logger=MagicMock(),
        )

        # gen=10 > total_gen=5，属于验证阶段，进度不应超过99%
        assert mining_tasks[self.task_id]["progress"] <= 99, (
            f"验证阶段进度不应超过99%，但得到 {mining_tasks[self.task_id]['progress']}"
        )

    def test_initial_progress_from_zero(self):
        """从0开始的进度应正常增长"""
        from backend.api.routers.mining import mining_tasks

        mining_tasks[self.task_id]["progress"] = 0

        _update_progress(
            self.task_id,
            gen=2,
            total_gen=5,
            best_fitness=0.2,
            avg_fitness=0.1,
            algorithm="deep_implicit",
            logger=MagicMock(),
        )

        assert mining_tasks[self.task_id]["progress"] == 40, (
            f"进度应增长到40%，但得到 {mining_tasks[self.task_id]['progress']}"
        )

    def test_multiple_updates_no_regression(self):
        """多次更新序列中不应出现倒退"""
        from backend.api.routers.mining import mining_tasks

        progress_history = []

        updates = [
            (0, 5, 0.0, 0.0),   # 0%
            (1, 5, 0.1, 0.05),  # 20%
            (2, 5, 0.2, 0.1),   # 40%
            (3, 5, 0.15, 0.08), # 60% (best下降但gen增加)
            (4, 5, 0.25, 0.12), # 80%
        ]

        for gen, total, best, avg in updates:
            _update_progress(
                self.task_id,
                gen=gen,
                total_gen=total,
                best_fitness=best,
                avg_fitness=avg,
                algorithm="gflownet",
                logger=MagicMock(),
            )
            current = mining_tasks[self.task_id]["progress"]
            progress_history.append(current)

        # 验证进度序列单调不减
        for i in range(1, len(progress_history)):
            assert progress_history[i] >= progress_history[i - 1], (
                f"进度序列不应倒退: {progress_history[i-1]}% -> {progress_history[i]}% "
                f"at step {i}"
            )

    def test_progress_never_negative(self):
        """进度不应为负数"""
        from backend.api.routers.mining import mining_tasks

        mining_tasks[self.task_id]["progress"] = 10

        _update_progress(
            self.task_id,
            gen=0,
            total_gen=5,
            best_fitness=-0.1,
            avg_fitness=-0.05,
            algorithm="pysr",
            logger=MagicMock(),
        )

        assert mining_tasks[self.task_id]["progress"] >= 0, (
            f"进度不应为负数，但得到 {mining_tasks[self.task_id]['progress']}"
        )

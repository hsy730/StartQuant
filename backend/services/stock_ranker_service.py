"""
StockRanker 排序学习服务 — 替代 BigQuant StockRanker 的 GBDT Learning-to-Rank 实现

核心能力：
1. 基于 XGBoost Ranking 目标的排序模型训练（pairwise / listwise）
2. 全市场多股票特征吸收（支持 3000+ 股票同时训练）
3. 排序分数预测 → 组合权重自动映射
4. 模型持久化和版本管理（依赖 ModelRegistry）
5. 训练 → 预测 → 回测闭环

与 BigQuant StockRanker 对比：
  StockRanker: 封闭式 GBDT 排序学习，内置特征工程
  FactorHub: 开放式 XGBoost/LightGBM Ranking，用户自定义特征，
             支持 SHAP 解释 + 未来函数检测 + 完整回测链路

设计原则：
- Smart Default: 默认参数即可获得合理结果
- 可解释: 内置 SHAP 特征重要性分析
- 安全: 自动进行未来函数检测（训练前检查特征）
- 高性能: 支持增量训练和批量预测
"""
import logging
import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost 未安装，StockRanker 排序学习功能不可用。请执行: pip install xgboost")


class RankObjective(str, Enum):
    """排序目标类型"""
    PAIRWISE = "rank:pairwise"      # 成对排序损失
    NDCG = "rank:ndcg"              # NDCG 优化（listwise）
    MAP = "rank:map"                # Mean Average Precision
    REGRESSION = "reg:squarederror" # 回退为回归（非排序）


class ModelStatus(str, Enum):
    """模型状态"""
    TRAINING = "training"
    READY = "ready"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class RankTrainingConfig:
    """排序模型训练配置"""
    objective: str = RankObjective.NDCG.value
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    n_estimators: int = 200
    early_stopping_rounds: int = 20
    eval_metric: str = "ndcg@5"

    # 排序特定参数
    lambdarank_pair_method: str = "mean"  # mean / max / sum
    ndcg_truncation: int = 5              # NDCG@k 的 k 值

    # 正则化
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0


@dataclass
class RankPredictionResult:
    """排序预测结果"""
    predictions: pd.DataFrame  # columns: [date, stock_code, rank_score, rank_position]
    top_n_stocks: pd.DataFrame  # Top-N 推荐股票
    metrics: Dict[str, float]  # 预测质量指标
    generated_at: str


@dataclass
class RankTrainingResult:
    """排序模型训练结果"""
    model_id: str
    status: ModelStatus
    config: RankTrainingConfig
    training_metrics: Dict[str, Any]
    feature_importance: Dict[str, float]
    shap_summary: Optional[Dict] = None
    duration_seconds: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    train_period: str = ""


class StockRankerService:
    """
    GBDT 排序学习服务（替代 BigQuant StockRanker）

    使用方式：
        ranker = StockRankerService()

        # 训练
        training_result = ranker.train(
            feature_df=df,     # MultiIndex DataFrame: (date, asset) → features
            label_col="return_5d",
            date_col="date",
            group_col="date",   # 按 date 分组做 listwise ranking
        )

        # 预测
        prediction = ranker.predict(
            model_id=training_result.model_id,
            features_today=today_features,
        )

        # 预测 + 回测闭环
        backtest = ranker.predict_and_backtest(
            model_id=training_result.model_id,
            feature_history=recent_features,
        )
    """

    DEFAULT_CONFIG = RankTrainingConfig()

    def __init__(
        self,
        model_registry=None,
        default_config: Optional[RankTrainingConfig] = None,
    ):
        """
        初始化排序服务

        Args:
            model_registry: 模型注册中心实例（可选，默认用文件存储）
            default_config: 默认训练配置
        """
        if not XGB_AVAILABLE:
            raise ImportError(
                "XGBoost 未安装。请执行: pip install xgboost"
            )

        self.model_registry = model_registry
        self.default_config = default_config or RankTrainingConfig()
        self._loaded_models: Dict[str, xgb.Booster] = {}  # 内存缓存

    def train(
        self,
        feature_df: pd.DataFrame,
        label_col: str = "forward_return_5d",
        date_col: str = "date",
        group_col: str = "date",
        feature_cols: Optional[List[str]] = None,
        config: Optional[RankTrainingConfig] = None,
        validation_split: float = 0.2,
        model_name: str = "stock_ranker",
        tags: Optional[List[str]] = None,
        enable_bias_check: bool = True,
    ) -> RankTrainingResult:
        """
        训练排序模型

        Args:
            feature_df: 特征数据 DataFrame（必须包含 date_col 和 label_col）
            label_col: 标签列名（收益率）
            date_col: 日期列名
            group_col: 排序分组列名（通常按日期分组）
            feature_cols: 使用的特征列名列表（None=自动排除非数值列）
            config: 训练配置
            validation_split: 验证集比例
            model_name: 模型名称
            tags: 模型标签
            enable_bias_check: 是否启用特征的未来函数检测

        Returns:
            RankTrainingResult: 训练结果
        """
        t0 = time.time()
        cfg = config or self.default_config

        logger.info(
            f"[StockRanker] 开始训练: 模型={model_name}, 样本={len(feature_df)}, "
            f"目标={cfg.objective}, 评估指标={cfg.eval_metric}"
        )

        # ---- 数据准备 ----
        df = feature_df.copy()
        if date_col not in df.columns:
            raise ValueError(f"日期列 '{date_col}' 不存在于数据中")
        if label_col not in df.columns:
            raise ValueError(f"标签列 '{label_col}' 不存在于数据中")

        # 自动识别特征列
        if feature_cols is None:
            exclude_cols = {date_col, label_col, "stock_code", "asset", "index"}
            feature_cols = [
                c for c in df.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
            ]

        if len(feature_cols) == 0:
            raise ValueError("没有可用的数值特征列")

        # 清洗数据
        df = df[[date_col, label_col] + feature_cols].dropna(subset=[label_col] + feature_cols)
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.dropna(subset=[label_col])

        if len(df) < 100:
            raise ValueError(f"清洗后样本数({len(df)})不足100，无法训练排序模型")

        # ---- 未来函数检测（对每个特征列）----
        bias_warnings = []
        if enable_bias_check:
            from backend.services.lookahead_bias_detector import lookahead_bias_detector
            for feat in feature_cols:
                try:
                    check = lookahead_bias_detector.detect(
                        factor_values=df[feat],
                        return_values=df[label_col],
                        factor_name=feat,
                    )
                    if check.risk_level.value in ("high", "critical"):
                        bias_warnings.append(
                            f"特征 [{feat}] 存在未来函数风险({check.risk_level.value})，建议移除"
                        )
                except Exception:
                    pass

        # ---- 构造 DMatrix ----
        # 对于排序任务，需要按 group_col 分组
        df = df.sort_values(date_col)
        X = df[feature_cols].values
        y = df[label_col].values

        # 按日期分组（每组的长度用于 XGBoost ranking）
        groups = df.groupby(group_col).size().values.tolist()

        # 时间分割：后 validation_split 作为验证集
        split_idx = int(len(df) * (1 - validation_split))
        dtrain = xgb.DMatrix(X[:split_idx], label=y[:split_idx], feature_names=feature_cols)
        dvalid = xgb.DMatrix(X[split_idx:], label=y[split_idx:], feature_names=feature_cols)

        # 将 groups 按行数分割到 train/valid
        train_groups, valid_groups = self._split_groups(groups, len(df), split_idx)
        dtrain.set_group(train_groups)
        dvalid.set_group(valid_groups)

        # ---- 训练 ----
        # 根据目标函数自动选择评估指标
        if cfg.objective.startswith("rank:"):
            eval_metric = cfg.eval_metric
        else:
            eval_metric = "rmse"

        params = {
            "objective": cfg.objective,
            "learning_rate": cfg.learning_rate,
            "max_depth": cfg.max_depth,
            "min_child_weight": cfg.min_child_weight,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_alpha": cfg.reg_alpha,
            "reg_lambda": cfg.reg_lambda,
            "eval_metric": eval_metric,
            "verbosity": 1,
            "seed": 42,
        }

        eval_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=cfg.n_estimators,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=cfg.early_stopping_rounds,
            evals_result=eval_result,
            verbose_eval=False,
        )

        # ---- 特征重要性 ----
        importance = model.get_score(importance_type="gain")
        importance_sorted = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        # ---- 保存模型 ----
        model_id = self._save_model(model, model_name, cfg, tags, feature_cols)

        duration = time.time() - t0

        logger.info(
            f"[StockRanker] 训练完成: model_id={model_id}, "
            f"best_iteration={model.best_iteration}, "
            f"best_score={eval_result['valid'][eval_metric][-1] if eval_result.get('valid') else 'N/A'}, "
            f"耗时={duration:.1f}s"
        )

        return RankTrainingResult(
            model_id=model_id,
            status=ModelStatus.READY,
            config=cfg,
            training_metrics={
                "best_iteration": int(model.best_iteration),
                "train_score": float(eval_result["train"][eval_metric][-1]) if eval_result.get("train") else 0,
                "valid_score": float(eval_result["valid"][eval_metric][-1]) if eval_result.get("valid") else 0,
                "eval_history": {
                    k: {metric: [float(v) for v in vals] for metric, vals in metrics.items() if isinstance(metrics, dict)}
                    if isinstance(metrics, dict) else [float(v) for v in metrics]
                    for k, metrics in eval_result.items()
                } if eval_result else {},
            },
            feature_importance=importance_sorted,
            shap_summary=self._compute_shap_summary(model, dtrain, feature_cols) if len(feature_cols) <= 50 else None,
            duration_seconds=duration,
            n_samples=len(df),
            n_features=len(feature_cols),
            train_period=f"{df[date_col].min()} ~ {df[date_col].max()}",
        )

    def predict(
        self,
        model_id: str,
        features: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        top_n: int = 50,
    ) -> RankPredictionResult:
        """
        使用训练好的排序模型进行预测

        Args:
            model_id: 模型 ID
            features: 当日/当期特征数据
            feature_cols: 特征列名（必须与训练时一致）
            top_n: 返回 Top-N 股票

        Returns:
            RankPredictionResult: 预测结果
        """
        model = self._load_model(model_id)
        metadata = self._get_model_metadata(model_id)

        trained_features = metadata.get("feature_cols", [])
        use_cols = feature_cols or trained_features

        # 确保特征列一致
        missing = set(use_cols) - set(features.columns)
        if missing:
            raise ValueError(f"缺少特征列: {missing}")

        X = features[use_cols].fillna(0).values
        dmatrix = xgb.DMatrix(X, feature_names=use_cols)

        scores = model.predict(dmatrix)

        # 构建结果 DataFrame
        result_df = features.copy()
        result_df["rank_score"] = scores
        result_df["rank_position"] = pd.Series(scores).rank(ascending=False).values
        result_df = result_df.sort_values("rank_score", ascending=False)

        top_n_df = result_df.head(top_n).copy()

        # 预测质量指标
        metrics = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "top_score": float(np.max(scores)),
            "bottom_score": float(np.min(scores)),
            "score_range": float(np.max(scores) - np.min(scores)),
            "n_predicted": len(features),
            "effective_ratio": float((scores > scores.mean()).sum()) / len(scores),
        }

        return RankPredictionResult(
            predictions=result_df,
            top_n_stocks=top_n_df,
            metrics=metrics,
            generated_at=datetime.now().isoformat(),
        )

    def predict_and_backtest(
        self,
        model_id: str,
        feature_history: pd.DataFrame,
        date_col: str = "date",
        stock_col: str = "stock_code",
        price_col: str = "close",
        feature_cols: Optional[List[str]] = None,
        top_n: int = 50,
        rebalance_freq: str = "M",
    ) -> Dict[str, Any]:
        """
        预测 + 回测闭环（排序模型的杀手级功能）

        流程：
        1. 按时间顺序逐期调用 predict
        2. 根据 Top-N 排序结果构建组合权重
        3. 将权重送入 VectorBT 回测引擎
        4. 返回完整的回测统计

        Args:
            model_id: 模型 ID
            feature_history: 历史特征数据（含日期、股票代码、价格等）
            date_col: 日期列名
            stock_col: 股票代码列名
            price_col: 价格列名
            feature_cols: 特征列名
            top_n: 每期选股数量
            rebalance_freq: 再平衡频率 (D/W/M/Q)

        Returns:
            回测结果字典
        """
        model = self._load_model(model_id)
        metadata = self._get_model_metadata(model_id)
        use_cols = feature_cols or metadata.get("feature_cols", [])

        logger.info(
            f"[StockRanker] 开始预测+回测: model_id={model_id}, "
            f"历史数据={len(feature_history)}行, top_n={top_n}"
        )

        # 按日期滚动预测
        dates = sorted(feature_history[date_col].unique())
        portfolio_weights_list = []

        for i, date in enumerate(dates):
            day_data = feature_history[feature_history[date_col] == date]

            if len(day_data) < top_n or len(day_data) < 5:
                continue

            try:
                X_day = day_data[use_cols].fillna(0).values
                dm = xgb.DMatrix(X_day, feature_names=use_cols)
                scores = model.predict(dm)

                day_result = day_data.copy()
                day_result["rank_score"] = scores
                day_result = day_result.sort_values("rank_score", ascending=False)
                top_stocks = day_result.head(top_n)

                # 等权分配
                weight = 1.0 / max(len(top_stocks), 1)
                for _, row in top_stocks.iterrows():
                    portfolio_weights_list.append({
                        date_col: date,
                        stock_col: row.get(stock_col, row.get("asset", "")),
                        "weight": weight,
                        "rank_score": float(row["rank_score"]),
                    })

            except Exception as e:
                logger.debug(f"日期 {date} 预测失败: {e}")
                continue

        if not portfolio_weights_list:
            return {"error": "无法在任何日期上完成预测", "success": False}

        weights_df = pd.DataFrame(portfolio_weights_list)

        # 送入回测引擎
        try:
            from backend.services.vectorbt_backtest_service import vectorbt_backtest_service

            backtest_result = vectorbt_backtest_service.run_vectorbt_backtest_from_weights(
                weights_df=weights_df,
                price_data=feature_history[[date_col, stock_col, price_col]].rename(
                    columns={stock_col: "ticker", price_col: "close"}
                ) if stock_col != "ticker" else feature_history[[date_col, stock_col, price_col]],
                rebalance_freq=rebalance_freq,
            )
            backtest_result["success"] = True
            backtest_result["model_id"] = model_id
            backtest_result["n_predictions"] = len(portfolio_weights_list)
            backtest_result["n_dates_covered"] = len(weights_df[date_col].unique())
            return backtest_result

        except Exception as e:
            logger.error(f"[StockRanker] 回测引擎执行失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"回测引擎失败: {str(e)}",
                "portfolio_weights": weights_df.to_dict(orient="records")[:100],
            }

    def explain_model(
        self,
        model_id: str,
        feature_sample: Optional[pd.DataFrame] = None,
        max_display: int = 20,
    ) -> Dict[str, Any]:
        """
        模型解释（SHAP + 特征重要性）

        Args:
            model_id: 模型 ID
            feature_sample: 用于 SHAP 分析的样本（None 则跳过 SHAP）
            max_display: 最大显示特征数

        Returns:
            解释结果字典
        """
        model = self._load_model(model_id)
        metadata = self._get_model_metadata(model_id)
        feature_cols = metadata.get("feature_cols", [])

        # 特征重要性
        importance = model.get_score(importance_type="gain")
        importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:max_display]

        result = {
            "model_id": model_id,
            "feature_importance_gain": [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in importance_sorted
            ],
            "feature_importance_split": [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in sorted(
                    model.get_score(importance_type="weight").items(),
                    key=lambda x: x[1], reverse=True
                )[:max_display]
            ],
            "n_features": len(feature_cols),
            "metadata": metadata,
        }

        # SHAP 分析
        if feature_sample is not None:
            try:
                shap_summary = self._compute_shap_summary(
                    model,
                    xgb.DMatrix(
                        feature_sample[feature_cols].fillna(0).values,
                        feature_names=feature_cols,
                    ),
                    feature_cols,
                )
                result["shap_summary"] = shap_summary
            except Exception as e:
                logger.warning(f"SHAP 分析失败: {e}")
                result["shap_warning"] = str(e)

        return result

    def list_models(self, tags: Optional[List[str]] = None) -> List[Dict]:
        """列出所有已保存的排序模型"""
        if self.model_registry:
            return self.model_registry.list_models(tags=tags)
        return self._file_based_list_models()

    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        if self.model_registry:
            return self.model_registry.delete(model_id)
        return self._file_based_delete(model_id)

    # ==================== 模型持久化 ====================

    def _save_model(
        self,
        model: xgb.Booster,
        model_name: str,
        config: RankTrainingConfig,
        tags: Optional[List[str]],
        feature_cols: List[str],
    ) -> str:
        """保存模型到注册中心或文件系统"""
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "config": {
                "objective": config.objective,
                "learning_rate": config.learning_rate,
                "max_depth": config.max_depth,
                "n_estimators": config.n_estimators,
            },
            "feature_cols": feature_cols,
            "tags": tags or [],
            "framework": "xgboost_ranking",
            "version": "1.0.0",
        }

        if self.model_registry:
            model_id = self.model_registry.save(model, metadata, framework="xgboost")
        else:
            self._file_based_save(model, model_id, metadata)

        # 缓存到内存
        self._loaded_models[model_id] = model

        return model_id

    def _load_model(self, model_id: str) -> xgb.Booster:
        """加载模型（优先从内存缓存）"""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        if self.model_registry:
            model = self.model_registry.load(model_id)
        else:
            model = self._file_based_load(model_id)

        if model is not None:
            self._loaded_models[model_id] = model
        return model

    def _get_model_metadata(self, model_id: str) -> Dict:
        """获取模型元信息"""
        if self.model_registry:
            return self.model_registry.get_metadata(model_id)
        return self._file_based_get_metadata(model_id)

    # ==================== 文件系统存储（fallback）====================

    def _model_dir(self) -> str:
        base = os.path.join(os.path.dirname(__file__), "..", "..", "models", "stock_ranker")
        os.makedirs(base, exist_ok=True)
        return base

    def _file_based_save(self, model: xgb.Booster, model_id: str, metadata: Dict):
        model_dir = self._model_dir()
        path = os.path.join(model_dir, f"{model_id}.json")
        meta_path = os.path.join(model_dir, f"{model_id}_metadata.json")
        model.save_model(path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"[StockRanker] 模型已保存: {path}")

    def _file_based_load(self, model_id: str) -> Optional[xgb.Booster]:
        path = os.path.join(self._model_dir(), f"{model_id}.json")
        if not os.path.exists(path):
            return None
        model = xgb.Booster()
        model.load_model(path)
        return model

    def _file_based_get_metadata(self, model_id: str) -> Dict:
        path = os.path.join(self._model_dir(), f"{model_id}_metadata.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _file_based_list_models(self) -> List[Dict]:
        model_dir = self._model_dir()
        models = []
        for fname in os.listdir(model_dir):
            if fname.endswith("_metadata.json"):
                with open(os.path.join(model_dir, fname), "r", encoding="utf-8") as f:
                    models.append(json.load(f))
        return sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)

    def _file_based_delete(self, model_id: str) -> bool:
        model_dir = self._model_dir()
        deleted = False
        for suffix in (".json", "_metadata.json"):
            path = os.path.join(model_dir, f"{model_id}{suffix}")
            if os.path.exists(path):
                os.remove(path)
                deleted = True
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
        return deleted

    # ==================== SHAP 辅助 ====================

    @staticmethod
    def _split_groups(groups: List[int], total_rows: int, split_idx: int) -> Tuple[List[int], List[int]]:
        """
        将分组列表按行数分割为训练集和验证集

        Args:
            groups: 每个分组的样本数列表
            split_idx: 训练/验证分割的行索引
            total_rows: 总行数

        Returns:
            (train_groups, valid_groups) 分组后的列表
        """
        train_groups = []
        valid_groups = []
        cumulative = 0

        for g in groups:
            if cumulative + g <= split_idx:
                # 整个组属于训练集
                train_groups.append(g)
            elif cumulative >= split_idx:
                # 整个组属于验证集
                valid_groups.append(g)
            else:
                # 组被分割：整个组归训练集（Learning-to-Rank要求同一日期组完整）
                train_groups.append(g)
            cumulative += g

        # 确保总和正确（容错）
        if sum(train_groups) == 0:
            train_groups = [split_idx]
        if sum(valid_groups) == 0:
            valid_groups = [total_rows - split_idx]

        return train_groups, valid_groups

    @staticmethod
    def _compute_shap_summary(
        model: xgb.Booster,
        dmatrix: xgb.DMatrix,
        feature_names: List[str],
    ) -> Optional[Dict]:
        """计算 SHAP 值摘要"""
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(dmatrix)

            # 全局特征重要性（按 |mean(SHAP)| 排序）
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = {
                feat: float(mean_abs_shap[i])
                for i, feat in enumerate(feature_names)
            }
            shap_importance_sorted = dict(
                sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
            )

            return {
                "shap_importance": shap_importance_sorted,
                "shap_shape": list(shap_values.shape),
                "base_value": float(explainer.expected_value) if hasattr(explainer, "expected_value") else None,
            }
        except ImportError:
            logger.debug("SHAP 库未安装，跳过 SHAP 分析")
            return None
        except Exception as e:
            logger.debug(f"SHAP 计算失败: {e}")
            return None


# ==================== 全局实例 ====================

try:
    stock_ranker_service = StockRankerService()
except ImportError:
    stock_ranker_service = None
    logger.warning("StockRanker 服务初始化失败（缺少 XGBoost），相关 API 将返回 503")

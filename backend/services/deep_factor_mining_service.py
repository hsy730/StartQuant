"""
深度隐式因子挖掘服务 - 基于 Transformer 架构的动态时变隐因子学习

与公式化因子挖掘（DEAP GP / PySR）不同，深度隐式因子走的是"隐因子"赛道：
- 不追求可解释性，纯以收益率预测为目标
- Transformer 自注意力机制捕获时序动态依赖
- 输出 K 个时变隐因子，每个因子在不同时间步有不同取值
- 适用于高频交易、纯收益导向场景

参考: NeurIF (CIKM 2025) — Neural Implicit Factor 框架

架构:
    时序特征 → 位置编码 → Transformer Encoder → 隐因子表征 → 收益率预测头

设计原则:
- Smart Default: 默认参数即可获得合理结果
- 与现有挖掘服务接口一致: set_stock_pool / set_progress_callback / mine_factors
- 模型持久化: 依赖 ModelRegistry (framework="pytorch")
- 安全: 训练前数据清洗，NaN 处理，特征标准化
"""
import logging
import os
import json
import time
import math
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    DEEP_FACTOR_AVAILABLE = True
except ImportError:
    DEEP_FACTOR_AVAILABLE = False
    logger.warning("PyTorch 未安装，深度隐式因子挖掘功能不可用。请运行: pip install torch")

from backend.services.factor_validation_service import factor_validation_service
from backend.services.data_service import data_service


# ======================================================================
# PyTorch 模型定义（仅在 PyTorch 可用时加载）
# ======================================================================

if DEEP_FACTOR_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """正弦余弦位置编码，为 Transformer 注入时序信息"""

        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, d_model)
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)

    class TransformerFactorModel(nn.Module):
        """Transformer 隐因子模型

        输入: (batch, seq_len, n_features) 的时序特征矩阵
        输出:
            - return_pred: (batch,) 收益率预测
            - latent_factors: (batch, seq_len, n_latent_factors) 隐因子表征
        """

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 3,
            d_ff: int = 256,
            n_latent_factors: int = 5,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.n_features = n_features
            self.d_model = d_model
            self.n_latent_factors = n_latent_factors

            # 输入投影: n_features → d_model
            self.input_proj = nn.Linear(n_features, d_model)

            # 位置编码
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )

            # 隐因子提取头: d_model → n_latent_factors
            self.factor_head = nn.Linear(d_model, n_latent_factors)

            # 收益率预测头: 取最后一个时间步的隐因子，映射到标量
            self.prediction_head = nn.Sequential(
                nn.Linear(n_latent_factors, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

            self._init_weights()

        def _init_weights(self):
            """Xavier 均匀初始化"""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (batch, seq_len, n_features)

            Returns:
                return_pred: (batch,) 收益率预测
                latent_factors: (batch, seq_len, n_latent_factors) 隐因子表征
            """
            # 投影 + 位置编码
            h = self.input_proj(x)  # (batch, seq_len, d_model)
            h = self.pos_encoder(h)

            # Transformer 编码
            h = self.transformer_encoder(h)  # (batch, seq_len, d_model)

            # 提取隐因子
            latent_factors = self.factor_head(h)  # (batch, seq_len, n_latent_factors)

            # 使用最后一个时间步的隐因子做预测
            last_factor = latent_factors[:, -1, :]  # (batch, n_latent_factors)
            return_pred = self.prediction_head(last_factor).squeeze(-1)  # (batch,)

            return return_pred, latent_factors

    class TimeSeriesDataset(Dataset):
        """时序滑动窗口数据集"""

        def __init__(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            seq_length: int,
        ):
            """
            Args:
                features: (N, n_features) 特征矩阵
                targets: (N,) 目标收益率
                seq_length: 滑动窗口长度
            """
            self.features = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(targets)
            self.seq_length = seq_length

        def __len__(self) -> int:
            return len(self.features) - self.seq_length

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.features[idx: idx + self.seq_length]  # (seq_len, n_features)
            y = self.targets[idx + self.seq_length]  # 标量
            return x, y


# ======================================================================
# 服务主类
# ======================================================================

class DeepFactorMiningService:
    """深度隐式因子挖掘服务（基于 Transformer）

    与 GeneticFactorMiningService / PySRFactorMiningService 接口一致，
    可互换使用或并行调用。

    核心流程:
        1. 预计算基础因子值
        2. 构建时序滑动窗口数据集
        3. 训练 Transformer 模型（MSE + 稀疏惩罚）
        4. 提取隐因子表征
        5. 逐因子验证（IC / IR / 换手率等）
        6. 返回结构化结果
    """

    def __init__(
        self,
        base_factors: List[str],
        data: pd.DataFrame,
        return_column: str = "return",
        factor_calculator=None,
        max_eval_stocks: int = 50,
        # ---- Transformer 模型参数 ----
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        n_latent_factors: int = 5,
        dropout: float = 0.1,
        # ---- 训练参数 ----
        seq_length: int = 20,
        learning_rate: float = 1e-4,
        n_epochs: int = 50,
        batch_size: int = 32,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 5,
        sparsity_coeff: float = 1e-4,
    ):
        if not DEEP_FACTOR_AVAILABLE:
            raise ImportError("PyTorch 未安装，请运行: pip install torch")

        self.base_factor_codes = base_factors
        self.data = data
        self.return_column = return_column
        self.factor_calculator = factor_calculator
        self.max_eval_stocks = max_eval_stocks

        # 模型参数
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.n_latent_factors = n_latent_factors
        self.dropout = dropout

        # 训练参数
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.sparsity_coeff = sparsity_coeff

        self.return_values = data[return_column] if return_column in data.columns else None

        # 股票池
        self.stock_codes: List[str] = []
        self.stock_pool_data: Dict[str, pd.DataFrame] = {}
        self.stock_pool_return_values: Dict[str, pd.Series] = {}
        self.stock_pool_base_factor_values: Dict[str, dict] = {}
        self._sampled_stock_codes: List[str] = []

        # 预计算基础因子
        self.base_factor_values: Dict[str, dict] = {}
        self._precompute_base_factors()

        # 训练设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"深度因子挖掘使用设备: {self.device}")

        # 训练后的模型和数据
        self._trained_model = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

        self.progress_callback = None
        self._cancel_flag = False

    # ------------------------------------------------------------------
    # 股票池设置
    # ------------------------------------------------------------------

    def set_stock_pool(self, stock_codes: List[str], start_date: str, end_date: str):
        """设置股票池，用于横截面 IC 评估"""
        self.stock_codes = stock_codes
        self.stock_pool_data = data_service.get_multiple_stocks_data(stock_codes, start_date, end_date)

        for code, df in self.stock_pool_data.items():
            if "close" in df.columns:
                df["return"] = df["close"].pct_change()
            self.stock_pool_return_values[code] = (
                df[self.return_column] if self.return_column in df.columns else None
            )

            if self.factor_calculator is None:
                from backend.services.factor_service import factor_service
                self.factor_calculator = factor_service.calculator

            stock_base_factors = {}
            for i, factor_code in enumerate(self.base_factor_codes):
                try:
                    fv = self.factor_calculator.calculate(df, factor_code)
                    if fv is not None and len(fv.dropna()) > 0:
                        var_name = f"factor_{i}"
                        stock_base_factors[var_name] = {
                            "code": factor_code,
                            "values": fv,
                        }
                except Exception as e:
                    logger.warning(f"Stock {code} factor {factor_code} 计算出错: {e}")
            self.stock_pool_base_factor_values[code] = stock_base_factors

        self._refresh_stock_sample()
        logger.info(
            f"深度因子股票池已设置: {len(self.stock_pool_data)} 只股票, "
            f"评估样本={len(self._sampled_stock_codes)}"
        )

    def _refresh_stock_sample(self):
        available = list(self.stock_pool_base_factor_values.keys())
        if len(available) <= self.max_eval_stocks:
            self._sampled_stock_codes = available
        else:
            import random
            self._sampled_stock_codes = random.sample(available, self.max_eval_stocks)

    def set_progress_callback(self, callback):
        """设置进度回调函数

        Args:
            callback: 签名为 callback(epoch, total_epochs, train_loss, val_loss)
        """
        self.progress_callback = callback

    def request_cancel(self):
        """请求取消挖掘任务"""
        self._cancel_flag = True
        logger.info("收到取消请求，将在当前epoch结束后停止")

    # ------------------------------------------------------------------
    # 基础因子预计算
    # ------------------------------------------------------------------

    def _precompute_base_factors(self):
        """预计算所有基础因子值"""
        if self.factor_calculator is None:
            from backend.services.factor_service import factor_service
            self.factor_calculator = factor_service.calculator

        logger.info(f"深度因子预计算 {len(self.base_factor_codes)} 个基础因子...")

        for i, factor_code in enumerate(self.base_factor_codes):
            try:
                fv = self.factor_calculator.calculate(self.data, factor_code)
                if fv is not None and len(fv.dropna()) > 0:
                    var_name = f"factor_{i}"
                    self.base_factor_values[var_name] = {
                        "code": factor_code,
                        "values": fv,
                    }
                    logger.info(
                        f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: "
                        f"{len(fv.dropna())} 个有效值"
                    )
                else:
                    logger.warning(
                        f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: 计算失败或无有效值"
                    )
            except Exception as e:
                logger.warning(
                    f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: 计算出错 - {e}"
                )

        logger.info(f"深度因子成功预计算 {len(self.base_factor_values)} 个基础因子")

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------

    def _build_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """构建特征矩阵和目标向量

        Returns:
            (X, y, feature_names) — X: (N, n_features), y: (N,)
        """
        feature_names = []
        series_list = []

        for var_name in sorted(self.base_factor_values.keys()):
            info = self.base_factor_values[var_name]
            feature_names.append(var_name)
            series_list.append(info["values"])

        if not series_list:
            raise ValueError("没有有效的基础因子来构建特征矩阵")

        combined = pd.DataFrame({name: s for name, s in zip(feature_names, series_list)})

        if self.return_values is not None:
            combined["__target__"] = self.return_values
        else:
            raise ValueError("没有收益率数据可用作目标")

        combined = combined.dropna()
        if len(combined) < self.seq_length + 10:
            raise ValueError(
                f"有效数据点不足 ({len(combined)})，至少需要 {self.seq_length + 10} 个"
            )

        X = combined[feature_names].values
        y = combined["__target__"].values

        return X, y, feature_names

    @staticmethod
    def _normalize_features(
        X_train: np.ndarray, X_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Z-score 标准化（基于训练集统计量）

        Returns:
            (X_train_norm, X_val_norm, mean, std)
        """
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0  # 避免除零

        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std

        return X_train_norm, X_val_norm, mean, std

    def _prepare_data(
        self,
    ) -> Tuple["DataLoader", "DataLoader", List[str], np.ndarray, np.ndarray]:
        """构建训练/验证 DataLoader

        Returns:
            (train_loader, val_loader, feature_names, feature_mean, feature_std)
        """
        X, y, feature_names = self._build_feature_matrix()

        # 前向填充 + 零填充处理 NaN
        df_features = pd.DataFrame(X)
        df_features = df_features.ffill().fillna(0)
        X = df_features.values

        # 时间分割: 后 20% 作为验证集
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 标准化
        X_train_norm, X_val_norm, feature_mean, feature_std = self._normalize_features(
            X_train, X_val
        )
        self._feature_mean = feature_mean
        self._feature_std = feature_std

        # 构建 Dataset
        train_dataset = TimeSeriesDataset(X_train_norm, y_train, self.seq_length)
        val_dataset = TimeSeriesDataset(X_val_norm, y_val, self.seq_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        logger.info(
            f"数据准备完成: 训练样本={len(train_dataset)}, 验证样本={len(val_dataset)}, "
            f"特征维度={len(feature_names)}"
        )

        return train_loader, val_loader, feature_names, feature_mean, feature_std

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def _train_model(
        self,
        train_loader: "DataLoader",
        val_loader: "DataLoader",
        n_features: int,
    ) -> Tuple["TransformerFactorModel", Dict[str, List[float]]]:
        """训练 Transformer 因子模型

        Returns:
            (trained_model, history) — history 包含 train_loss / val_loss 曲线
        """
        model = TransformerFactorModel(
            n_features=n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            n_latent_factors=self.n_latent_factors,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        logger.info(
            f"开始训练: d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, n_latent_factors={self.n_latent_factors}, "
            f"n_epochs={self.n_epochs}, device={self.device}"
        )

        for epoch in range(1, self.n_epochs + 1):
            # 取消检查
            if self._cancel_flag:
                logger.info(f"深度因子训练在第 {epoch} 个epoch被用户取消")
                break

            # ---- 训练阶段 ----
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred, latent = model(batch_x)

                # MSE 损失
                mse_loss = criterion(pred, batch_y)

                # 稀疏惩罚: 鼓励仅使用少数隐因子（组稀疏正则化）
                # 对每个隐因子维度计算L2范数，再用L1惩罚，鼓励整个因子维度趋零
                # latent shape: [batch, seq_len, n_latent_factors]
                factor_norms = torch.norm(latent, p=2, dim=(0, 1))  # [n_latent_factors]
                sparsity_loss = self.sparsity_coeff * torch.sum(factor_norms)

                loss = mse_loss + sparsity_loss
                loss.backward()
                optimizer.step()

                train_losses.append(mse_loss.item())

            # ---- 验证阶段 ----
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    pred, _ = model(batch_x)
                    val_loss = criterion(pred, batch_y)
                    val_losses.append(val_loss.item())

            avg_train_loss = float(np.mean(train_losses))
            avg_val_loss = float(np.mean(val_losses))
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # 进度回调
            if self.progress_callback:
                self.progress_callback(epoch, self.n_epochs, avg_train_loss, avg_val_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{self.n_epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )

            if patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping: 验证损失连续 {self.early_stopping_patience} 轮未改善, "
                    f"best_val_loss={best_val_loss:.6f}"
                )
                break

        # 恢复最优模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(self.device)

        logger.info(
            f"训练完成: best_val_loss={best_val_loss:.6f}, "
            f"实际训练轮数={len(history['train_loss'])}"
        )

        return model, history

    # ------------------------------------------------------------------
    # 隐因子提取
    # ------------------------------------------------------------------

    def _extract_latent_factors(
        self,
        model: "TransformerFactorModel",
        feature_names: List[str],
    ) -> Dict[str, pd.Series]:
        """从训练好的模型中提取隐因子时序

        对完整数据集做前向推理，提取每个时间步的隐因子表征。

        Returns:
            {factor_name: pd.Series} — 每个隐因子的时序值
        """
        model.eval()

        X, y, _ = self._build_feature_matrix()

        # 前向填充 + 零填充
        df_features = pd.DataFrame(X)
        df_features = df_features.ffill().fillna(0)
        X = df_features.values

        # 标准化（使用训练时保存的统计量）
        if self._feature_mean is not None and self._feature_std is not None:
            X_norm = (X - self._feature_mean) / self._feature_std
        else:
            X_norm = X

        # 构建滑动窗口
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        all_factors = []

        with torch.no_grad():
            for start in range(0, len(X_tensor) - self.seq_length + 1):
                window = X_tensor[start: start + self.seq_length].unsqueeze(0)
                _, latent = model(window)
                # 取最后一个时间步的隐因子
                last_step_factors = latent[0, -1, :].cpu().numpy()
                all_factors.append(last_step_factors)

        if not all_factors:
            logger.warning("未能提取任何隐因子")
            return {}

        factor_array = np.array(all_factors)  # (N_valid, n_latent_factors)

        # 构建索引: 对齐到原始数据的时间索引
        combined = pd.DataFrame(
            {name: self.base_factor_values[name]["values"]
             for name in sorted(self.base_factor_values.keys())}
        )
        if self.return_values is not None:
            combined["__target__"] = self.return_values
        combined = combined.dropna()
        valid_index = combined.index[self.seq_length - 1: self.seq_length - 1 + len(factor_array)]

        factor_series = {}
        for k in range(self.n_latent_factors):
            factor_name = f"DeepFactor_{k + 1}"
            factor_series[factor_name] = pd.Series(
                factor_array[:, k],
                index=valid_index[:len(factor_array)],
                name=factor_name,
            )

        logger.info(f"成功提取 {len(factor_series)} 个隐因子")
        return factor_series

    # ------------------------------------------------------------------
    # 因子验证
    # ------------------------------------------------------------------

    def _validate_factors(
        self,
        factor_series: Dict[str, pd.Series],
    ) -> List[Dict]:
        """逐因子验证，返回排序后的因子信息列表"""
        validated_factors = []

        for factor_name, fv in factor_series.items():
            if fv is None or len(fv.dropna()) < 10:
                logger.warning(f"{factor_name} 有效值不足，跳过验证")
                continue

            validation = {}
            fitness = 0.0

            if self.return_values is not None:
                try:
                    # 对齐因子值和收益率
                    aligned_ret = self.return_values.reindex(fv.index).dropna()
                    aligned_fv = fv.reindex(aligned_ret.index).dropna()

                    if len(aligned_fv) >= 10 and len(aligned_ret) >= 10:
                        validation = factor_validation_service.validate_factor(
                            factor_values=aligned_fv,
                            return_values=aligned_ret,
                            existing_factors=None,
                        )
                        fitness = validation.get("score", 0) / 100.0
                except Exception as e:
                    logger.warning(f"{factor_name} 验证失败: {e}")

            # 构建架构描述
            arch_desc = (
                f"Transformer(d_model={self.d_model},n_heads={self.n_heads},"
                f"n_layers={self.n_layers})_factor_{factor_name.split('_')[-1]}"
            )

            factor_info = {
                "expression": arch_desc,
                "fitness": fitness,
                "validation": validation,
                "source": "deep_implicit",
                "factor_type": "implicit",
                "model_architecture": (
                    f"TransformerEncoder(d_model={self.d_model}, n_heads={self.n_heads}, "
                    f"n_layers={self.n_layers}, d_ff={self.d_ff}, "
                    f"n_latent_factors={self.n_latent_factors}, dropout={self.dropout})"
                ),
            }
            validated_factors.append(factor_info)

        # 按 fitness 降序排序
        validated_factors.sort(key=lambda f: f.get("fitness", 0), reverse=True)

        for i, fi in enumerate(validated_factors):
            fi["rank"] = i + 1

        return validated_factors

    # ------------------------------------------------------------------
    # 模型持久化
    # ------------------------------------------------------------------

    def _save_model(
        self,
        model: "TransformerFactorModel",
        feature_names: List[str],
        training_history: Dict[str, List[float]],
    ) -> Optional[str]:
        """保存训练好的模型到 ModelRegistry 或文件系统"""
        model_name = "deep_factor_transformer"
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "framework": "pytorch",
            "version": "1.0.0",
            "params": {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "d_ff": self.d_ff,
                "n_latent_factors": self.n_latent_factors,
                "dropout": self.dropout,
                "seq_length": self.seq_length,
                "learning_rate": self.learning_rate,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "weight_decay": self.weight_decay,
                "sparsity_coeff": self.sparsity_coeff,
            },
            "feature_cols": feature_names,
            "training_history": training_history,
            "tags": ["deep_implicit", "transformer"],
        }

        # 尝试使用 ModelRegistry
        try:
            from backend.services.model_registry import model_registry
            model_id = model_registry.save(
                model, metadata, framework="pytorch", model_name=model_name
            )
            logger.info(f"模型已保存到 ModelRegistry: {model_id}")
            return model_id
        except Exception as e:
            logger.debug(f"ModelRegistry 保存失败，回退到文件系统: {e}")

        # 文件系统回退
        return self._file_based_save(model, model_id, metadata)

    def _file_based_save(
        self,
        model: "TransformerFactorModel",
        model_id: str,
        metadata: Dict,
    ) -> str:
        """文件系统存储回退"""
        model_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "deep_factor"
        )
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{model_id}.pt")
        meta_path = os.path.join(model_dir, f"{model_id}_metadata.json")

        torch.save(model.state_dict(), model_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"模型已保存到文件系统: {model_path}")
        return model_id

    # ------------------------------------------------------------------
    # 挖掘入口
    # ------------------------------------------------------------------

    def mine_factors(self) -> Dict:
        """执行深度隐式因子挖掘

        完整流程:
            1. 数据准备（特征矩阵 + 滑动窗口）
            2. Transformer 模型训练
            3. 隐因子提取
            4. 逐因子验证
            5. 模型持久化
            6. 返回结构化结果

        Returns
        -------
        dict with keys: ``success``, ``best_factors``, ``fitness_history``,
        ``training_history``, ``model_info``
        """
        if not DEEP_FACTOR_AVAILABLE:
            return {
                "success": False,
                "message": "PyTorch 未安装",
                "best_factors": [],
            }

        logger.info("开始深度隐式因子挖掘...")
        logger.info(
            f"模型参数: d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, d_ff={self.d_ff}, "
            f"n_latent_factors={self.n_latent_factors}, dropout={self.dropout}"
        )
        logger.info(
            f"训练参数: seq_length={self.seq_length}, lr={self.learning_rate}, "
            f"epochs={self.n_epochs}, batch_size={self.batch_size}, "
            f"weight_decay={self.weight_decay}, "
            f"early_stopping_patience={self.early_stopping_patience}, "
            f"sparsity_coeff={self.sparsity_coeff}"
        )

        t0 = time.time()

        try:
            # ---- Step 1: 数据准备 ----
            train_loader, val_loader, feature_names, _, _ = self._prepare_data()
            n_features = len(feature_names)

            # ---- Step 2: 模型训练 ----
            model, training_history = self._train_model(train_loader, val_loader, n_features)
            self._trained_model = model

            # ---- Step 3: 隐因子提取 ----
            factor_series = self._extract_latent_factors(model, feature_names)

            if not factor_series:
                logger.warning("未能提取有效隐因子")
                return {
                    "success": True,
                    "best_factors": [],
                    "fitness_history": {"best": [], "average": []},
                    "training_history": training_history,
                    "model_info": {
                        "architecture": "Transformer",
                        "n_parameters": sum(p.numel() for p in model.parameters()),
                        "training_epochs": len(training_history["train_loss"]),
                        "best_val_loss": min(training_history["val_loss"]) if training_history["val_loss"] else float("inf"),
                    },
                }

            # ---- Step 4: 逐因子验证 ----
            best_factors = self._validate_factors(factor_series)

            # ---- Step 5: 模型持久化 ----
            model_id = None
            try:
                model_id = self._save_model(model, feature_names, training_history)
            except Exception as e:
                logger.warning(f"模型保存失败（不影响因子结果）: {e}")

            # ---- Step 6: 构建结果 ----
            n_params = sum(p.numel() for p in model.parameters())
            best_val_loss = min(training_history["val_loss"]) if training_history["val_loss"] else float("inf")
            actual_epochs = len(training_history["train_loss"])

            # 为每个因子添加复杂度（模型参数量）
            for fi in best_factors:
                fi["complexity"] = float(n_params)

            # fitness_history: 兼容前端展示（用 val_loss 的负值作为 fitness 代理）
            fitness_history = {
                "best": [-l for l in training_history["val_loss"]],
                "average": [-l for l in training_history["train_loss"]],
            }

            arch_str = (
                f"TransformerEncoder(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_layers={self.n_layers}, d_ff={self.d_ff}, "
                f"n_latent_factors={self.n_latent_factors})"
            )

            duration = time.time() - t0
            logger.info(
                f"深度因子挖掘完成: 发现 {len(best_factors)} 个隐因子, "
                f"模型参数量={n_params}, 训练轮数={actual_epochs}, "
                f"best_val_loss={best_val_loss:.6f}, 耗时={duration:.1f}s"
            )

            return {
                "success": True,
                "best_factors": best_factors,
                "fitness_history": fitness_history,
                "training_history": training_history,
                "model_info": {
                    "architecture": arch_str,
                    "n_parameters": n_params,
                    "training_epochs": actual_epochs,
                    "best_val_loss": best_val_loss,
                    "model_id": model_id,
                    "device": str(self.device),
                },
            }

        except Exception as e:
            logger.error(f"深度因子挖掘失败: {e}", exc_info=True)
            return {
                "success": False,
                "message": str(e),
                "best_factors": [],
                "fitness_history": {"best": [], "average": []},
                "training_history": {"train_loss": [], "val_loss": []},
                "model_info": {},
            }


# ======================================================================
# 工厂函数
# ======================================================================

def create_deep_factor_mining_service(
    base_factors: List[str],
    data: pd.DataFrame,
    factor_calculator=None,
    **kwargs,
) -> DeepFactorMiningService:
    """创建配置好的 :class:`DeepFactorMiningService` 实例

    可通过关键字参数覆盖的配置项:

    * ``d_model`` – Transformer 模型维度 (默认 64)
    * ``n_heads`` – 注意力头数 (默认 4)
    * ``n_layers`` – Transformer 层数 (默认 3)
    * ``d_ff`` – 前馈网络维度 (默认 256)
    * ``n_latent_factors`` – 隐因子数量 (默认 5)
    * ``dropout`` – Dropout 率 (默认 0.1)
    * ``seq_length`` – 输入序列长度 (默认 20)
    * ``learning_rate`` – 学习率 (默认 1e-4)
    * ``n_epochs`` – 训练轮数 (默认 50)
    * ``batch_size`` – 批大小 (默认 32)
    * ``weight_decay`` – L2 正则化系数 (默认 1e-5)
    * ``early_stopping_patience`` – 早停耐心 (默认 5)
    * ``sparsity_coeff`` – 隐因子稀疏惩罚系数 (默认 1e-4)
    * ``max_eval_stocks`` – 最大评估股票数 (默认 50)
    """
    return DeepFactorMiningService(
        base_factors=base_factors,
        data=data,
        factor_calculator=factor_calculator,
        **kwargs,
    )

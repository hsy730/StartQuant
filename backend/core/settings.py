"""
应用配置模块
"""

import logging
import shutil
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# 性能档位预设：只决定各项并发默认值，显式写出的同名配置优先。
# low = 低性能笔记本，mid = 台式机（多核）。
_PERFORMANCE_PRESETS = {
    "low": {
        "DATA_FETCH_MAX_WORKERS": 4,
        "INDUSTRY_FETCH_MAX_WORKERS": 4,
        "FACTOR_CALC_MAX_WORKERS": 4,
        "ANALYSIS_MAX_WORKERS": 2,
        "VALIDATION_MAX_WORKERS": 2,
        "PREPROCESSING_MAX_WORKERS": 2,
        "PREPROCESSING_STOCK_MAX_WORKERS": 4,
        "MINING_PRECOMPUTE_MAX_WORKERS": 2,
        "MINING_STOCK_POOL_MAX_WORKERS": 2,
    },
    "mid": {
        "DATA_FETCH_MAX_WORKERS": 8,
        "INDUSTRY_FETCH_MAX_WORKERS": 8,
        "FACTOR_CALC_MAX_WORKERS": 8,
        "ANALYSIS_MAX_WORKERS": 8,
        "VALIDATION_MAX_WORKERS": 8,
        "PREPROCESSING_MAX_WORKERS": 8,
        "PREPROCESSING_STOCK_MAX_WORKERS": 16,
        "MINING_PRECOMPUTE_MAX_WORKERS": 0,  # 0 = CPU 核数
        "MINING_STOCK_POOL_MAX_WORKERS": 0,  # 0 = min(CPU, 8)
    },
}


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 项目路径
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = DATA_DIR / "cache"
    DB_DIR: Path = DATA_DIR / "db"
    CONFIG_DIR: Path = BASE_DIR / "config"
    REPORTS_DIR: Path = DATA_DIR / "reports"

    # 数据库配置
    DATABASE_URL: str = f"sqlite:///{DB_DIR}/factorflow.db"

    # akshare 配置
    AKSHARE_CACHE_ENABLED: bool = True
    AKSHARE_CACHE_DIR: Path = CACHE_DIR / "akshare"

    # 缓存配置
    CACHE_DEFAULT_TTL: int = 7 * 24 * 60 * 60  # 默认TTL: 7天（秒）
    CACHE_AUTO_CLEANUP: bool = True  # 是否自动清理过期缓存
    CACHE_CLEANUP_INTERVAL: int = 24 * 60 * 60  # 清理间隔: 24小时（秒）

    # 数据预处理配置
    DATA_OUTLIER_DETECTION: bool = True  # 是否启用异常值检测
    DATA_OUTLIER_N_SIGMA: float = 3.0  # 异常值检测的σ倍数
    DATA_OUTLIER_METHOD: str = "clip"  # 异常值处理方法: clip/remove/replace
    DATA_FILL_MISSING: bool = True  # 是否填充缺失值
    DATA_FILL_METHOD: str = "ffill"  # 缺失值填充方法: ffill/bfill/interpolate

    # ---- 并发配置（按机器性能调节，可在 .env 覆盖）----
    # 先选性能档位，档位只提供下面各项的默认值；单独写出的项优先于档位。
    # low: 低性能笔记本（保守，避免内存/线程争抢）
    # mid: 台式机（多核，放开取数与分析并发）
    PERFORMANCE_PROFILE: str = "low"

    # 均为线程池上限（默认值等于 low 档位）；标 0 的项表示按 CPU 核数自动推导。
    # 注意：akshare 内部用 py_mini_racer 执行 JS 签名，并非线程安全。
    # 实测 15 线程冷缓存并发取数在 160 只股票上会随机硬崩溃（exit 0x80000003），
    # 机器核数少或需稳定批量拉数时把 DATA_FETCH_MAX_WORKERS 调到 1~4。
    DATA_FETCH_MAX_WORKERS: int = 4  # 多股行情并行取数
    INDUSTRY_FETCH_MAX_WORKERS: int = 4  # 行业成分股并行取数
    FACTOR_CALC_MAX_WORKERS: int = 4  # 多股票批量因子计算
    ANALYSIS_MAX_WORKERS: int = 2  # 多因子 Alphalens 分析
    VALIDATION_MAX_WORKERS: int = 2  # 批量表达式验证
    PREPROCESSING_MAX_WORKERS: int = 2  # 单表多因子并行预处理
    PREPROCESSING_STOCK_MAX_WORKERS: int = 4  # 多股票并行预处理
    MINING_PRECOMPUTE_MAX_WORKERS: int = 2  # 基础因子预计算（0=CPU 核数）
    MINING_STOCK_POOL_MAX_WORKERS: int = 2  # 股票池因子预计算（0=min(CPU, 8)）

    # 分析配置
    DEFAULT_START_DATE: str = "2020-01-01"
    DEFAULT_END_DATE: str = "2024-12-31"

    # 日志配置
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_LEVEL: str = "INFO"  # DEBUG/INFO/WARNING/ERROR
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024  # 单文件最大10MB
    LOG_FILE_BACKUP_COUNT: int = 5  # 保留5个备份

    # 滚动窗口配置
    ROLLING_WINDOW: int = 252  # 一年交易日

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._apply_performance_profile()
        # 创建必要的目录
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.AKSHARE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # 检测数据库文件是否存在，若不存在则从样本文件复制
        db_file = self.DB_DIR / "factorflow.db"
        sample_db_file = self.DB_DIR / "factorflow.sample.db"
        if not db_file.exists() and sample_db_file.exists():
            logger.info(
                f"Copying sample database file: factorflow.sample.db -> {db_file}"
            )
            shutil.copy2(sample_db_file, db_file)

    def _apply_performance_profile(self) -> None:
        """用性能档位填充并发项；显式配置（.env/环境变量/构造参数）优先。"""
        preset = _PERFORMANCE_PRESETS.get(self.PERFORMANCE_PROFILE.lower())
        if preset is None:
            logger.warning(
                f"未知的 PERFORMANCE_PROFILE={self.PERFORMANCE_PROFILE!r}，"
                f"可选 {sorted(_PERFORMANCE_PRESETS)}；沿用各项默认值"
            )
            return
        for name, value in preset.items():
            if name not in self.model_fields_set:
                setattr(self, name, value)


# 全局配置实例
settings = Settings()

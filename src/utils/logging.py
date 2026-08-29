"""统一日志（T01）。

职责：
    - get_logger：带控制台 handler 的命名 logger（避免重复 handler 堆叠）；
    - setup_run_logger：为 runs/<run_id>/ 附加文件 handler（run 级审计留痕，
      如盲法揭盲操作、人工修正、meta 覆盖告知都写日志）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["get_logger", "setup_run_logger", "LOG_FORMAT"]

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """获取带控制台输出的命名 logger（幂等：重复调用不堆叠 handler）。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "_ff_std_console", False)
        for h in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=_DATEFMT))
        handler._ff_std_console = True  # type: ignore[attr-defined]  # 标记自管理 handler
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def setup_run_logger(run_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """为单个 run 目录建立文件日志（runs/<run_id>/analysis.log）。

    Returns:
        名为 "run.<run_dir名>" 的 logger，同时输出到控制台与文件。
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "analysis.log"
    logger = get_logger(f"run.{run_dir.name}", level=level)
    if not any(
        isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=_DATEFMT))
        fh._ff_std_file = True  # type: ignore[attr-defined]
        logger.addHandler(fh)
    logger.info("run 日志初始化: %s", log_path)
    return logger

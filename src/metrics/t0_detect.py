"""metrics/t0_detect.py · t0 自动检测（FR-08）。

PRD §4.1.1 FR-08：t0（投喂起点）应支持自动检测，且与人工打点偏差 > 1 s
时给出告警。

实现约定：
    - 自动检测口径 = "第一颗饲料出现在画面"（即 t0_definition = pellet_in_frame
      的客观可复现代理）。对 feeder_start / pellet_released 这两种"画面外事件"，
      帧内观测只能以"首颗入画"近似，报告中须如实披露这一近似。
    - 输入 samples 为 **相对当前 t0** 的 (t_s, n_pellets) 有序序列；返回值即
      "自动检测 t0 相对人工 t0 的偏移"，可直接用于偏差判定。
    - 纯函数、无副作用、可单测。

任务编号：审计整改（FR-08 PARTIAL）。
"""
from __future__ import annotations

from typing import Sequence

__all__ = ["detect_first_pellet_s", "T0_DEVIATION_ALERT_S"]

# 自动检测与人工打点偏差超过该阈值（秒）即告警（FR-08）
T0_DEVIATION_ALERT_S: float = 1.0


def detect_first_pellet_s(
    samples: Sequence[tuple[float, int | None]],
) -> float | None:
    """检测第一颗饲料出现的相对时刻。

    Args:
        samples: 按 t_s 升序的 (t_s_relative_to_t0, n_pellets) 序列。

    Returns:
        首颗 n_pellets > 0 的 t_s（相对当前 t0 的偏移）；若全程无颗粒则为 None。
    """
    for t_s, n in samples:
        if n is not None and n > 0:
            return float(t_s)
    return None


def deviation_alert(t0_auto_rel: float | None) -> str | None:
    """若 |自动检测偏移| 超过阈值，返回告警文本；否则 None。"""
    if t0_auto_rel is None:
        return None
    if abs(t0_auto_rel) > T0_DEVIATION_ALERT_S:
        return (
            f"t0 自动检测与人工打点偏差 {t0_auto_rel:+.1f}s "
            f"> {T0_DEVIATION_ALERT_S:.0f}s：请复核人工打点的准确性"
            f"（自动检测口径=首颗饲料入画）"
        )
    return None

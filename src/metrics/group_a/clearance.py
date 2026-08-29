"""A8–A10 · T50 / T90 / T100 清空时间（T04）。

职责（docs/04 §3.1 A8–A10）：
    - T = min{t : N_p(t) ≤ p × N₀}，相邻观测间线性插值求亚采样精度
      穿越点（合法插值：定位事件时刻，不制造观测点）；
    - 未穿越 → status='censored'，value=None，quality.window_s=窗长
      （输出 ">窗长"，绝不输出 0/NaN）；
    - T100 阈值 = ε = max(2 颗, 0.02×N₀)，任何时候都不得单独作为
      "吃完时间"对外，必须与 Q_pelletloss 并列展示。

硬规则（docs/04 §4.5）：
    - Q_pelletloss > 0.15 或 pellet_type='sinking' → T90/T100 降级 +
      contains_non_feeding_loss flag；
    - pellet_type=UNKNOWN → 只加 sedimentation_risk_unassessed 标记。

任务编号：T04。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.config import Thresholds
from src.core.metric_value import MetricValue
from src.metrics.group_a.pellet_curve import PelletSeries

__all__ = ["crossing_time", "crossing_threshold", "clearance_metric"]


def crossing_threshold(p: float, n0: float, thresholds: Thresholds) -> float:
    """穿越阈值：T50/T90 为 p×N₀；T100 为 ε=max(2, 0.02×N₀)。"""
    if p is None:
        raise ValueError("p 不可为 None")
    return p * float(n0)


def _interpolate_crossing(
    t: np.ndarray, n: np.ndarray, thr: float
) -> float | None:
    """在相邻观测之间线性插值定位精确穿越时刻；未穿越返回 None。"""
    below = n <= thr
    if not np.any(below):
        return None
    i = int(np.argmax(below))  # 第一个低于阈值的采样点
    if i == 0:
        return float(t[0])
    t0, t1 = float(t[i - 1]), float(t[i])
    n0v, n1 = float(n[i - 1]), float(n[i])
    if n0v == n1:
        return t1
    # 线性插值：N(t0) > thr ≥ N(t1)
    frac = (n0v - thr) / (n0v - n1)
    return t0 + frac * (t1 - t0)


def crossing_time(
    series: PelletSeries,
    n0: float,
    p: float,
    window_s: float,
    thresholds: Thresholds | None = None,
    use_epsilon: bool = False,
) -> float | None:
    """裸穿越时刻（数值正确性内核；MetricValue 包装见 clearance_metric）。

    Args:
        use_epsilon: True = T100 口径（阈值取 ε=max(2, 0.02×N₀)）。
    """
    th = thresholds if thresholds is not None else Thresholds()
    if not series.available or n0 is None or series.t.size < 2:
        return None
    idx = series.valid_mask()
    t, n = series.t[idx], series.n[idx]
    if t.size < 2:
        return None
    thr = (
        th.t100_epsilon(n0) if use_epsilon else crossing_threshold(p, n0, th)
    )
    return _interpolate_crossing(t, n, thr)


def clearance_metric(
    metric_id: str,
    t_cross: float | None,
    window_s: float,
    q_pelletloss: float | None = None,
    pellet_type: str | None = None,
    n_frames_used: int = 0,
    extra_flags: tuple[str, ...] = (),
    thresholds: Thresholds | None = None,
    local_slope: float | None = None,
) -> MetricValue:
    """T50/T90/T100 的 MetricValue 包装（censored 语义唯一入口）。

    censored 输出：
        value=None、status='censored'、quality={window_s, censored=True}、
        reason 含 ">窗长" 字样——绝不为 0/NaN。
    """
    th = thresholds if thresholds is not None else Thresholds()
    flags: list[str] = list(extra_flags)
    quality: dict[str, Any] = {
        "Q_pelletloss": q_pelletloss,
        "n_frames_used": n_frames_used,
    }
    if local_slope is not None:
        quality["local_slope"] = local_slope  # 穿越点局部斜率（越陡越可靠）

    degrade = (
        q_pelletloss is not None and q_pelletloss > th.pelletloss_degrade
    ) or (pellet_type in ("sinking", "slow-sinking"))

    if t_cross is None:
        quality["window_s"] = float(window_s)
        quality["censored"] = True
        flags.append("censored")
        return MetricValue(
            metric_id=metric_id,
            value=None,
            unit="s",
            status="censored",
            reason=(
                f"观察窗 {window_s:.0f}s 内未穿越阈值（>{window_s:.0f}s），"
                "右删失：不输出 0/NaN 冒充"
            ),
            flags=tuple(flags),
            quality=quality,
            unit_scale="none",
        )

    quality["censored"] = False
    if degrade:
        flags.append("contains_non_feeding_loss")
        reason = (
            "非摄食损失不可忽略（Q_pelletloss 超阈或沉性料）：降级为参考值，"
            "必须与 A14 非摄食损失并列展示"
        )
        return MetricValue(
            metric_id=metric_id,
            value=float(t_cross),
            unit="s",
            status="degraded",
            reason=reason,
            flags=tuple(flags),
            quality=quality,
            unit_scale="none",
        )
    if pellet_type == "unknown" or pellet_type is None:
        # 未确认的元数据只追加 ⚠（§4.0 元规则），不降级
        flags.append("sedimentation_risk_unassessed")
    return MetricValue(
        metric_id=metric_id,
        value=float(t_cross),
        unit="s",
        status="ok",
        reason=None,
        flags=tuple(flags),
        quality=quality,
        unit_scale="none",
    )

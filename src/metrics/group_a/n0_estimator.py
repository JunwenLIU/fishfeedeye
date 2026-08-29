"""A1 · 初始颗粒数 N₀ 双轨估计（T04）。

职责（docs/04 §3.1 A1）：
    - N0_det = 早期窗口 [t0, t0+W] 内 N(t) 的平滑最大值（3 点移动中位数
      后取 max），严禁单帧 argmax（防峰值帧漏检）；
    - N0_meta = feed_mass_g × 1000 / pellet_mass_mg（RunMeta 缺项 → None，
      cross_validated=False）；
    - Q_n0gap = |N0_det − N0_meta| / N0_meta，> 0.20 → 降级 +
      denominator_suspect（N₀ 系统性低估是"人为放大组间差异"的第一入口）。

失效条件：
    - 早期窗口有效帧 < 3 → 不可用；
    - RunMeta 缺项 → N0_meta=None，标记 cross_validated=False。

任务编号：T04。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.config import Thresholds
from src.core.frame_context import RunMeta
from src.core.metric_value import MetricValue
from src.metrics.group_a.pellet_curve import PelletSeries, moving_median

__all__ = ["estimate_n0", "n0_gap"]


def early_window_s(
    thresholds: Thresholds, video_duration_s: float | None
) -> float:
    """W = min(n0_early_window_s, 0.1 × 视频时长)；时长未知 = 固定窗口。"""
    w = float(thresholds.n0_early_window_s)
    if video_duration_s is not None and video_duration_s > 0:
        w = min(w, 0.1 * float(video_duration_s))
    return w


def n0_gap(n0_det: float, n0_meta: float | None) -> float | None:
    """Q_n0gap = |N0_det − N0_meta| / N0_meta（meta 缺失 → None，不猜测）。"""
    if n0_meta is None or n0_meta <= 0:
        return None
    return abs(float(n0_det) - float(n0_meta)) / float(n0_meta)


def estimate_n0(
    series: PelletSeries,
    meta: RunMeta | None = None,
    thresholds: Thresholds | None = None,
    video_duration_s: float | None = None,
) -> MetricValue:
    """A1 双轨 N₀ 估计（输出经 MetricValue 强制校验）。"""
    th = thresholds if thresholds is not None else Thresholds()
    meta = meta if meta is not None else RunMeta()
    quality: dict[str, Any] = {}

    # ---- N0_meta（元数据轨）----
    if meta.feed_mass_g is not None and meta.pellet_mass_mg is not None:
        n0_meta: float | None = meta.feed_mass_g * 1000.0 / meta.pellet_mass_mg
        quality["cross_validated"] = True
    else:
        n0_meta = None
        quality["cross_validated"] = False

    # ---- N0_det（检测轨）----
    if not series.available:
        return MetricValue(
            metric_id="A1_N0",
            value=None,
            unit="颗",
            status="unavailable",
            reason=series.reason or "颗粒序列不可用",
            flags=(),
            quality=quality,
            unit_scale="none",
        )

    w = early_window_s(th, video_duration_s)
    idx = (series.t >= 0.0) & (series.t <= w)
    n_win = series.n[idx]
    conf_win = series.conf_med[idx]
    low_win = series.low_conf[idx]
    quality["W_used"] = w
    quality["n_frames"] = int(n_win.size)

    if n_win.size < 3:
        return MetricValue(
            metric_id="A1_N0",
            value=None,
            unit="颗",
            status="unavailable",
            reason=f"早期窗口 [{0:.0f}s, {w:.0f}s] 有效帧不足（n={n_win.size} < 3）",
            flags=(),
            quality=quality,
            unit_scale="none",
        )

    # 平滑最大值：3 点移动中位数后取 max（严禁单帧 argmax）
    ok = ~low_win
    if np.any(ok):
        smoothed = moving_median(n_win[ok], 3)
        n0_det = float(np.max(smoothed))
        peak_i = int(np.argmax(smoothed))
        peak_conf = (
            float(np.nanmax(conf_win[ok])) if np.any(~np.isnan(conf_win[ok])) else None
        )
        quality["t_at_peak"] = float(series.t[idx][ok][peak_i])
    else:
        n0_det = float(np.max(n_win))
        peak_conf = None
    quality["peak_conf"] = peak_conf
    quality["N0_meta"] = n0_meta

    gap = n0_gap(n0_det, n0_meta)
    quality["Q_n0gap"] = gap
    if n0_meta is not None:
        quality["N0_meta"] = float(n0_meta)

    flags: list[str] = []
    status = "ok"
    reason: str | None = None
    if gap is not None and gap > th.n0_gap_warn:
        status = "degraded"
        flags.append("denominator_suspect")
        reason = (
            f"Q_n0gap = {gap:.1%} > {th.n0_gap_warn:.0%}："
            "N₀ 检测值与投喂量口径偏差过大（可能漏检或元数据口径不符），"
            "降级为参考值；所有以 N₀ 为分母的指标标记 denominator_suspect"
        )

    return MetricValue(
        metric_id="A1_N0",
        value=n0_det,
        unit="颗",
        status=status,
        reason=reason,
        flags=tuple(flags),
        quality=quality,
        unit_scale="none",
    )

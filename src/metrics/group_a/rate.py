"""A5–A7 · 消耗速率指标（T04）。

职责（docs/04 §3.1 A5/A6/A7）：
    - A5 v(t) = −dN_p/dt（颗/s，原生不规则采样；先单调化再求导，
      回补帧不得产生负速率）；
    - A6 v_max：搜索窗口 [t0, T50]（限定早期避免尾部噪声）；T50 未定义
      → 窗口退化为 [t0, 0.5×窗长] 并标注（method=window_fallback）；
    - A7 v̄₅₀ = 0.5×N₀/T50；T50 右删失 → 不可用（不得用窗长代替）。

纪律：
    - 所有积分/微分在原生序列上计算（基于真实 dt），禁止重采样；
    - N_p 非单调上升段占比 > 0.10 → counting_unstable，v(t) 仅供定性；
    - 窗口内有效点 < 5 → 该指标不输出（不得用单点噪声冒充速率）。
"""
from __future__ import annotations

import numpy as np

from src.core.config import Thresholds
from src.core.metric_value import MetricValue

__all__ = ["instantaneous_rate", "rate_metrics"]


def instantaneous_rate(t: np.ndarray, n: np.ndarray) -> np.ndarray:
    """A5 瞬时消耗速率 v(t) = −dN_p/dt（原生 dt 差分，先单调化）。

    Args:
        t: 原生采样时刻（秒，升序，可不等间隔）。
        n: 有效计数序列（调用方须传入单调化前的平滑序列；本函数内部
            先做累积最小值单调化，契约 A5 算法约束）。

    Returns:
        v 数组（颗/s），与 t 等长；首点无差分 → NaN。
    """
    t = np.asarray(t, dtype=float)
    n = np.asarray(n, dtype=float)
    v = np.full(t.shape, np.nan, dtype=float)
    if t.size < 2:
        return v
    mono = np.minimum.accumulate(n)
    dt = np.diff(t)
    ok = dt > 1e-9
    v[1:][ok] = -np.diff(mono)[ok] / dt[ok]
    return v


def rate_metrics(
    t: np.ndarray,
    n_smooth: np.ndarray,
    n0: float | None,
    t50: float | None,
    thresholds: Thresholds,
    window_s: float,
    quality: dict,
    base_flags: tuple[str, ...] = (),
    unstable_frac: float = 0.0,
) -> dict[str, MetricValue]:
    """计算 A6_v_max 与 A7_v50（A5 v(t) 为时序，由 aggregator 落入时序表）。

    Args:
        t: 原生采样时刻。
        n_smooth: 平滑后的有效计数序列。
        n0: 主口径 N₀；None → 双指标 unavailable。
        t50: T50（秒）；None（右删失）→ v_max 窗口退化、v̄₅₀ 不可用。
        thresholds: 阈值。
        window_s: 实际观察窗长度。
        quality: MetricValue 构造所需的 Q_* 引用。
        base_flags: 继承 flag（denominator_suspect / counting_unstable）。
        unstable_frac: 非单调上升段占比（>0.10 → counting_unstable）。
    """
    out: dict[str, MetricValue] = {}
    q = dict(quality)
    q["n_frames_used"] = int(np.sum(~np.isnan(np.asarray(n_smooth, dtype=float))))
    counting_unstable = unstable_frac > 0.10
    flags = tuple(base_flags) + (("counting_unstable",) if counting_unstable else ())

    # ---- A6 v_max ----
    if n0 is None or n0 <= 0 or t.size < 2:
        out["A6_v_max"] = MetricValue(
            metric_id="A6_v_max", value=None, unit="颗/s", status="unavailable",
            reason="N₀ 不可用或有效点不足，峰值速率无从定义",
            flags=flags, quality=dict(q), unit_scale="none",
        )
    else:
        v = instantaneous_rate(t, n_smooth)
        if t50 is not None and t50 > 0:
            win_hi = float(t50)
            method = "window=[t0,T50]"
        else:
            win_hi = 0.5 * float(window_s)
            method = "window_fallback=[t0,0.5×窗长]（T50 右删失）"
        sel = (t >= 0) & (t <= win_hi + 1e-9) & ~np.isnan(v)
        n_valid = int(np.sum(sel))
        q6 = dict(q)
        q6["window_s"] = win_hi
        q6["method"] = method
        q6["t_at_max"] = None
        if n_valid < 5:
            out["A6_v_max"] = MetricValue(
                metric_id="A6_v_max", value=None, unit="颗/s", status="unavailable",
                reason=f"搜索窗口 [{0:.0f}, {win_hi:.1f}]s 内有效速率点 {n_valid} < 5，"
                       "不得用单点噪声冒充峰值速率",
                flags=flags, quality=q6, unit_scale="none",
            )
        else:
            v_sel = v[sel]
            t_sel = t[sel]
            imax = int(np.argmax(v_sel))
            q6["t_at_max"] = float(t_sel[imax])
            status = "degraded" if counting_unstable else "ok"
            reason = None
            if counting_unstable:
                reason = (f"N_p 非单调上升段占比 {unstable_frac:.1%} > 10% "
                          "（counting_unstable）：速率仅供定性")
            out["A6_v_max"] = MetricValue(
                metric_id="A6_v_max", value=float(v_sel[imax]), unit="颗/s",
                status=status, reason=reason, flags=flags, quality=q6,
                unit_scale="none",
            )

    # ---- A7 v̄₅₀ ----
    if n0 is None or n0 <= 0:
        out["A7_v50"] = MetricValue(
            metric_id="A7_v50", value=None, unit="颗/s", status="unavailable",
            reason="N₀ 不可用，v̄₅₀ 无从定义", flags=flags,
            quality=dict(q), unit_scale="none",
        )
    elif t50 is None:
        out["A7_v50"] = MetricValue(
            metric_id="A7_v50", value=None, unit="颗/s", status="unavailable",
            reason="T50 右删失：契约禁止用窗长代替（观察窗内未消耗到半量）",
            flags=tuple(dict.fromkeys(flags + ("censored",))), quality=dict(q),
            unit_scale="none",
        )
    else:
        out["A7_v50"] = MetricValue(
            metric_id="A7_v50", value=0.5 * float(n0) / float(t50), unit="颗/s",
            status="degraded" if counting_unstable else "ok",
            reason=("counting_unstable：速率仅供定性" if counting_unstable else None),
            flags=flags, quality=dict(q), unit_scale="none",
        )
    return out

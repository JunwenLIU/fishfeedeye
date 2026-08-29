"""B2 · 投喂区鱼数 / 占比 / 相对偏好（T04，探索性）。

职责（docs/04 §3.2 B2-5/B2-6/B2-7 + docs/06 T04）：
    - B2-5 N_fz：投喂区（ROI.feeding_zone）内鱼体质心计数时序与试验窗均值；
    - B2-6 P_fz = mean(N_fz) / n_fish_total × 100；
      n_fish_total 缺失 → 只输出 N_fz 绝对值（no_denominator）；
    - B2-7 RP = mean(N_fz, 试验窗) / 基线期 n_fz_mean：
      Q_baseline=False → 强制关闭（定义即相对基线）；
      n_fz_mean < 1 → 不可用（除零风险 + 比值无意义）；
      基线变异系数 > 0.5 → 降级（unstable_baseline）。

数据来源：obs.extra['fish']（FishDetections）——指标层只消费契约，不 import
检测器；无鱼体检测数据 → 全组 unavailable（户外斜拍默认降级的根因）。

纪律：户外场景 B2 默认整体 degraded（outdoor_exploratory），
主结论只建立在 A 组颗粒曲线上（docs/06 §7.8）。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.core.frame_context import BaselineStats, FishDetections, FrameObservation, RunMeta
from src.core.metric_value import MetricValue
from src.core.roi import ROI, point_in_polygon
from src.metrics.group_a.pellet_curve import TimeSeries

__all__ = ["zone_metrics"]


def _as_obs_list(
    observations: Sequence[FrameObservation] | FrameObservation,
) -> list[FrameObservation]:
    """归一化入参为观测列表（容错：也接受单个 FrameObservation）。

    为什么需要：单帧调用是本层最常见的手误（写 `zone_metrics(obs, ...)`
    漏了方括号），原实现会在 `sorted(...)` 处抛
    "FrameObservation object is not iterable"——一个与业务语义无关的
    TypeError，排查成本高。此处显式容错，行为与传入 `[obs]` 完全一致。
    """
    if isinstance(observations, FrameObservation):
        return [observations]
    return list(observations)


def _n_fz_series(
    observations: Sequence[FrameObservation] | FrameObservation,
    feed_zone: np.ndarray | None,
) -> TimeSeries | None:
    """N_fz(t) 时序（有鱼体数据的帧；无任何数据 → None）。"""
    rows: list[tuple[float, float]] = []
    for obs in sorted(_as_obs_list(observations), key=lambda o: o.t_s):
        fish = obs.extra.get("fish")
        if not isinstance(fish, FishDetections) or fish.n_det() == 0:
            if isinstance(fish, FishDetections):
                rows.append((float(obs.t_s), 0.0))  # 零值是有效观测
            continue
        if feed_zone is None:
            continue
        cents = fish.centroids()
        n_in = int(
            sum(1 for c in cents if point_in_polygon((float(c[0]), float(c[1])), feed_zone))
        )
        rows.append((float(obs.t_s), float(n_in)))
    if not rows or feed_zone is None:
        return None
    t = np.array([r[0] for r in rows], dtype=float)
    v = np.array([r[1] for r in rows], dtype=float)
    return TimeSeries(metric_id="N_fz", t=t, values=v, unit="尾")


def zone_metrics(
    observations: Sequence[FrameObservation] | FrameObservation,
    roi: ROI | None,
    meta: RunMeta,
    baseline: BaselineStats | None,
    quality: dict[str, Any],
    outdoor: bool = False,
) -> tuple[dict[str, MetricValue], TimeSeries | None]:
    """计算 B2-5/B2-6/B2-7。Returns: (metrics, N_fz 时序)。

    Args:
        observations: 帧观测序列（也容错接受单个 FrameObservation）。
        outdoor: 户外斜拍（用户确认）→ B2 标量 degraded + outdoor_exploratory；
            室内（默认 False）正常输出（降级交给 capability 门控）。
    """
    out: dict[str, MetricValue] = {}
    feed_zone = roi.feeding_zone if roi is not None else None
    series = _n_fz_series(observations, feed_zone)
    if outdoor:
        ex_status, ex_flags, ex_reason = (
            "degraded", ("outdoor_exploratory",),
            "户外斜拍鱼体检测不可行，仅探索性输出（outdoor_exploratory）",
        )
    else:
        ex_status, ex_flags, ex_reason = "ok", (), None

    if series is None:
        reason = (
            "鱼体检测数据不可用（obs.extra['fish'] 缺失）：户外斜拍场景"
            "鱼体检测默认不可行，B2 组仅探索性（A 组颗粒曲线为主结论）"
        )
        if feed_zone is None:
            reason = "ROI.feeding_zone 未定义，投喂区指标不可用（" + reason + "）"
        for mid, unit in (("B2-5_N_fz_mean", "尾"), ("B2-6_P_fz", "%"), ("B2-7_RP", "-")):
            out[mid] = MetricValue(
                metric_id=mid, value=None, unit=unit, status="unavailable",
                reason=reason, quality=dict(quality),
            )
        return out, None

    t, v = series.valid_t_values()
    trial_mask = t >= -1e-9
    n_trial = int(np.count_nonzero(trial_mask))
    mean_n_fz = float(np.mean(v[trial_mask])) if n_trial > 0 else None
    q = dict(quality)

    # ---- B2-5 N_fz 均值 ----
    if mean_n_fz is None:
        out["B2-5_N_fz_mean"] = MetricValue(
            metric_id="B2-5_N_fz_mean", value=None, unit="尾", status="unavailable",
            reason="试验窗内无鱼体观测，N_fz 均值不可得", quality=q,
        )
    else:
        out["B2-5_N_fz_mean"] = MetricValue(
            metric_id="B2-5_N_fz_mean", value=mean_n_fz, unit="尾",
            status=ex_status, reason=ex_reason, flags=ex_flags,
            quality=dict(quality, n_frames_used=n_trial),
        )

    # ---- B2-6 P_fz ----
    if mean_n_fz is None:
        out["B2-6_P_fz"] = MetricValue(
            metric_id="B2-6_P_fz", value=None, unit="%", status="unavailable",
            reason="试验窗内无鱼体观测，P_fz 不可得", quality=q,
        )
    elif meta.n_fish_total is None or meta.n_fish_total <= 0:
        out["B2-6_P_fz"] = MetricValue(
            metric_id="B2-6_P_fz", value=None, unit="%", status="unavailable",
            reason="RunMeta.n_fish_total 缺失：占比分母不可得（no_denominator），"
                   "只输出 N_fz 绝对值，不猜测总数",
            flags=("no_denominator",), quality=q,
        )
    else:
        out["B2-6_P_fz"] = MetricValue(
            metric_id="B2-6_P_fz", value=mean_n_fz / float(meta.n_fish_total) * 100.0,
            unit="%", status=ex_status, reason=ex_reason, flags=ex_flags,
            quality=dict(quality, n_frames_used=n_trial, n_fish_total=meta.n_fish_total),
        )

    # ---- B2-7 RP ----
    if mean_n_fz is None:
        out["B2-7_RP"] = MetricValue(
            metric_id="B2-7_RP", value=None, unit="-", status="unavailable",
            reason="试验窗内无鱼体观测，RP 不可得", quality=q,
        )
    elif baseline is None or not baseline.ok:
        out["B2-7_RP"] = MetricValue(
            metric_id="B2-7_RP", value=None, unit="-", status="unavailable",
            reason="基线缺失（Q_baseline=False）：RP 的定义即相对基线，无基线"
                   "即无意义，强制关闭（docs/04 硬规则 4）",
            quality=q,
        )
    elif baseline.n_fz_mean < 1.0:
        out["B2-7_RP"] = MetricValue(
            metric_id="B2-7_RP", value=None, unit="-", status="unavailable",
            reason=f"基线期投喂区鱼数 n_fz_mean={baseline.n_fz_mean:.2f} < 1："
                   "除零风险且基线无鱼时比值无意义",
            quality=q,
        )
    else:
        rp = mean_n_fz / float(baseline.n_fz_mean)
        q7 = dict(quality)
        cv = baseline.n_fz_std / baseline.n_fz_mean if baseline.n_fz_mean > 0 else None
        q7.update({"baseline_n_fz_mean": baseline.n_fz_mean,
                   "baseline_cv": cv, "n_frames_used": n_trial})
        flags: tuple[str, ...] = ex_flags
        status = ex_status
        reason = ex_reason
        if cv is not None and cv > 0.5:
            flags = flags + ("unstable_baseline",)
            status = "degraded"
            reason = (reason or "") + ("；" if reason else "") + (
                "基线变异系数 > 0.5（unstable_baseline）"
            )
        out["B2-7_RP"] = MetricValue(
            metric_id="B2-7_RP", value=rp, unit="-", status=status,
            reason=reason, flags=flags, quality=q7,
        )
    return out, series

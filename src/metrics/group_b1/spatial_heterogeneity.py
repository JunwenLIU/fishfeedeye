"""B1 · 空间异质性活跃度 + D 组起止时刻（T04）。

职责（docs/04 B1-1/D1–D4 + docs/06 T04 内联约定）：
    - 投喂区 16×16 网格帧差 → 峰度 / 基尼 / 前 5% 格子能量占比
      （空间异质性指标对全局波浪不敏感，是户外主通道）；
    - M_diff(t)：arena∩¬exclusion 内的平均 |ΔI|，除以固定 arena 面积
      （绝不用鱼数/前景面积归一化），再除以真实 dt 得到速率口径
      （非对称采样下 2s/10s 间隔的原始帧差不可比）；
    - 参考区扣除 α（基线期过原点最小二乘回归，reference_correction）；
    - D1 FA(t)：有基线 → (M − μ_base)/σ_base 归一化；无基线 → 原始量 +
      flag uncalibrated；
    - D2 T_start（首次摄食潜伏期）：FA > μ+3σ 持续 ≥ onset_hold_s；
      无基线 → 退回"颗粒首次持续下降"口径 + flag fallback；
    - D3 T_end（饱和时刻）：峰后 FA < μ+2σ 持续 ≥ offset_hold_s；
      窗内未回落 → censored（">窗长"）；
    - D4 持续时长 = D3 − D2。

失效条件：
    - 帧图像不可用（缓存恢复模式）→ B1/D 全部 unavailable；
    - 基线缺失 → FA 只出原始量（uncalibrated）。

任务编号：T04。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from src.core.config import Thresholds
from src.core.frame_context import FrameObservation
from src.core.metric_value import MetricValue
from src.core.roi import ROI, polygon_area
from src.metrics.group_a.pellet_curve import TimeSeries
from src.metrics.group_b1.reference_correction import (
    NO_NOISE_CORRECTION_FLAG,
    apply_reference_correction,
    estimate_alpha,
    no_reference_warning,
)

__all__ = ["ActivityResult", "compute_activity", "d_group_metrics", "GRID_N"]

GRID_N: int = 16  # 16×16 网格（docs/06 T04 内联约定）


@dataclass
class ActivityResult:
    """B1/D 组计算的中间与最终产物。"""

    m_curve: TimeSeries | None = None            # M_diff 速率口径（arena 面积归一化）
    fa_curve: TimeSeries | None = None           # D1 FA(t)（归一化或原始）
    kurtosis_curve: TimeSeries | None = None     # 网格峰度时序
    gini_curve: TimeSeries | None = None         # 网格基尼时序
    top5_curve: TimeSeries | None = None         # 前 5% 格子能量占比时序
    mu_base: float | None = None
    sigma_base: float | None = None
    alpha: float | None = None                   # 参考区扣除系数（None = 未扣除）
    ref_note: str | None = None                  # 未扣除原因（含告警文案）
    warnings: list[str] = field(default_factory=list)
    unavailable_reason: str | None = None


def _zone_mask(shape: tuple[int, int], zone: np.ndarray | None,
               exclude: list[np.ndarray] | None) -> np.ndarray | None:
    """zone 布尔掩膜（None zone → 全 True），再挖除排除区。"""
    h, w = shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if zone is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(zone, dtype=np.int32)], 255)
    for ex in exclude or []:
        cv2.fillPoly(mask, [np.asarray(ex, dtype=np.int32)], 0)
    return mask.astype(bool)


def _gini(x: np.ndarray) -> float:
    """基尼系数（对网格能量分布；完全均匀=0，全部集中=1）。"""
    v = np.sort(np.asarray(x, dtype=float))
    n = v.shape[0]
    if n == 0 or float(np.sum(v)) <= 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2.0 * np.sum(cum) / cum[-1]) / n)


def _grid_stats(diff_img: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """网格统计：(峰度, 基尼, 前5%格子能量占比)。"""
    h, w = diff_img.shape
    gh, gw = max(1, h // GRID_N), max(1, w // GRID_N)
    cells: list[float] = []
    for r in range(0, h - gh + 1, gh):
        for c in range(0, w - gw + 1, gw):
            cell_mask = mask[r:r + gh, c:c + gw]
            if cell_mask.any():
                cells.append(float(np.mean(diff_img[r:r + gh, c:c + gw][cell_mask])))
    if not cells:
        return 0.0, 0.0, 0.0
    arr = np.asarray(cells)
    kurt = float(np.mean((arr - arr.mean()) ** 4) / max(arr.var() ** 2, 1e-12)) - 3.0
    gini = _gini(arr)
    n_top = max(1, int(np.ceil(arr.shape[0] * 0.05)))
    top = float(np.sum(np.sort(arr)[-n_top:])) / max(float(np.sum(arr)), 1e-12)
    return kurt, gini, top


def compute_activity(
    observations: Sequence[FrameObservation],
    roi: ROI | None,
    thresholds: Thresholds,
) -> ActivityResult:
    """计算 M_diff/FA 曲线与空间异质性统计（需帧图像；无图 → unavailable）。"""
    res = ActivityResult()
    obs_list = sorted(
        [o for o in observations if o.image is not None], key=lambda o: o.t_s
    )
    if len(obs_list) < 2:
        res.unavailable_reason = (
            "帧图像不可用（缓存恢复/测试注入模式）或采样帧不足，"
            "无法计算帧差活跃度（B1/D1–D4 不可用）"
        )
        return res

    arena = roi.arena if roi is not None else None
    arena_mask = _zone_mask(obs_list[0].image.shape[:2], arena,
                            roi.exclude_zones if roi is not None else None)
    ref_mask = _zone_mask(
        obs_list[0].image.shape[:2],
        roi.reference_zone if roi is not None else None,
        None,
    ) if (roi is not None and roi.reference_zone is not None) else None
    arena_area = (
        polygon_area(arena) if arena is not None
        else float(np.count_nonzero(arena_mask))
    )
    if arena_area <= 0:
        arena_area = float(np.count_nonzero(arena_mask))

    # ---- 逐帧对：帧差（灰度）→ 区域均值 → 除以真实 dt ----
    t_list: list[float] = []
    m_list: list[float] = []
    m_ref_list: list[float] = []
    kurt_list: list[float] = []
    gini_list: list[float] = []
    top5_list: list[float] = []
    prev_img = cv2.cvtColor(obs_list[0].image, cv2.COLOR_BGR2GRAY)
    prev_t = obs_list[0].t_s
    for obs in obs_list[1:]:
        gray = cv2.cvtColor(obs.image, cv2.COLOR_BGR2GRAY)
        dt = float(obs.t_s - prev_t)
        if dt <= 1e-9:
            prev_img, prev_t = gray, obs.t_s
            continue
        diff = np.abs(gray.astype(np.float32) - prev_img.astype(np.float32))
        m = float(np.mean(diff[arena_mask])) / arena_area / dt
        m_list.append(m)
        t_list.append(float(obs.t_s))
        if ref_mask is not None and ref_mask.any():
            m_ref_list.append(float(np.mean(diff[ref_mask])) / dt)
        kurt, gini, top5 = _grid_stats(diff, arena_mask)
        kurt_list.append(kurt)
        gini_list.append(gini)
        top5_list.append(top5)
        prev_img, prev_t = gray, obs.t_s

    if not t_list:
        res.unavailable_reason = "相邻采样帧间隔退化（dt≤0），无法计算帧差"
        return res

    # ---- 参考区扣除（α 基线期过原点最小二乘；不可估计 → 强制告警）----
    if ref_mask is None or not m_ref_list:
        res.ref_note = no_reference_warning()
        res.warnings.append(f"{NO_NOISE_CORRECTION_FLAG}: {res.ref_note}")
        m_arr = np.asarray(m_list, dtype=float)
    else:
        baseline_pairs = [
            (m_list[i], m_ref_list[i])
            for i, tt in enumerate(t_list)
            if tt < -1e-9
        ]
        alpha, note = estimate_alpha(
            [p[0] for p in baseline_pairs],
            [p[1] for p in baseline_pairs],
        )
        res.alpha = alpha
        res.ref_note = note
        if alpha is None:
            # α 不可估计：不得静默不扣除——flag + 告警
            res.warnings.append(f"{NO_NOISE_CORRECTION_FLAG}: {note}")
            m_arr = np.asarray(m_list, dtype=float)
        else:
            m_arr = apply_reference_correction(
                np.asarray(m_list), np.asarray(m_ref_list), alpha
            )

    t_arr = np.asarray(t_list, dtype=float)
    res.m_curve = TimeSeries(metric_id="M_diff", t=t_arr, values=m_arr,
                             unit="gray/s/px", note="arena-area normalized, dt-rate")
    res.kurtosis_curve = TimeSeries(
        metric_id="B1_kurtosis", t=t_arr, values=np.asarray(kurt_list), unit="-",
        note=f"grid={GRID_N}x{GRID_N}")
    res.gini_curve = TimeSeries(
        metric_id="B1_gini", t=t_arr, values=np.asarray(gini_list), unit="-",
        note=f"grid={GRID_N}x{GRID_N}")
    res.top5_curve = TimeSeries(
        metric_id="B1_top5_share", t=t_arr, values=np.asarray(top5_list), unit="-",
        note="top-5% cells energy share")

    # ---- 基线统计（自曲线负时间轴推导，与 BaselineStats 口径独立自洽）----
    base_mask = t_arr < -1e-9
    if int(np.count_nonzero(base_mask)) >= 3:
        res.mu_base = float(np.mean(m_arr[base_mask]))
        res.sigma_base = max(float(np.std(m_arr[base_mask])), 1e-9)
        fa_vals = (m_arr - res.mu_base) / res.sigma_base
        res.fa_curve = TimeSeries(
            metric_id="FA", t=t_arr, values=fa_vals, unit="-",
            note="(M-mu_base)/sigma_base")
    else:
        res.fa_curve = TimeSeries(
            metric_id="FA", t=t_arr, values=m_arr, unit="gray/s/px",
            note="raw (uncalibrated: no baseline frames)")
        res.warnings.append(
            "基线帧不足：FA(t) 只输出原始未归一化量（uncalibrated）"
        )
    return res


def _first_sustained_above(
    t: np.ndarray, v: np.ndarray, level: float, hold_s: float, t_from: float = 0.0
) -> float | None:
    """首个持续高于 level 的时刻（在 t_from 之后；持续 = hold_s 内不回落）。"""
    n = t.shape[0]
    for i in range(n):
        if t[i] < t_from - 1e-9 or v[i] <= level:
            continue
        j = i
        ok = True
        while j < n and t[j] <= t[i] + hold_s + 1e-9:
            if v[j] <= level:
                ok = False
                break
            j += 1
        if ok and (t[min(j, n - 1)] - t[i]) >= hold_s - 1e-9:
            return max(0.0, float(t[i]))
    return None


def _first_sustained_below_after_peak(
    t: np.ndarray, v: np.ndarray, level: float, hold_s: float
) -> float | None:
    """峰后首个持续低于 level 的时刻。"""
    if t.shape[0] == 0:
        return None
    i_peak = int(np.argmax(v))
    n = t.shape[0]
    for i in range(i_peak, n):
        if v[i] >= level:
            continue
        j = i
        ok = True
        while j < n and t[j] <= t[i] + hold_s + 1e-9:
            if v[j] >= level:
                ok = False
                break
            j += 1
        if ok and (t[min(j, n - 1)] - t[i]) >= hold_s - 1e-9:
            return float(t[i])
    return None


def d_group_metrics(
    activity: ActivityResult,
    quality: dict[str, Any],
    thresholds: Thresholds,
    pellet_decline_t: float | None,
    outdoor: bool = False,
) -> dict[str, MetricValue]:
    """D2_T_start / D3_T_end / D4_duration + B1 空间异质性标量。

    Args:
        outdoor: 户外斜拍（用户确认）→ B1 标量 degraded + outdoor_exploratory；
            室内（默认 False）正常输出（降级交给 capability 门控）。
    """
    out: dict[str, MetricValue] = {}
    fa = activity.fa_curve
    window_s = float(fa.t[-1]) if fa is not None and fa.n_points() > 0 else 0.0

    # ---- B1 空间异质性标量（试验窗均值）----
    if outdoor:
        out_status, out_flags, out_reason = (
            "degraded", ("outdoor_exploratory",),
            "户外斜拍场景 B1 为探索性输出（outdoor_exploratory）",
        )
    else:
        out_status, out_flags, out_reason = "ok", (), None
    for curve, mid, name in (
        (activity.kurtosis_curve, "B1_kurtosis_mean", "网格峰度均值"),
        (activity.gini_curve, "B1_gini_mean", "网格基尼均值"),
        (activity.top5_curve, "B1_top5_share_mean", "前5%格子能量占比均值"),
    ):
        if curve is None:
            out[mid] = MetricValue(
                metric_id=mid, value=None, unit="-", status="unavailable",
                reason=activity.unavailable_reason or "B1 不可用", quality=dict(quality),
            )
        else:
            m = (curve.t >= -1e-9) & (~np.isnan(curve.values))
            if int(np.count_nonzero(m)) == 0:
                out[mid] = MetricValue(
                    metric_id=mid, value=None, unit="-", status="unavailable",
                    reason="试验窗内无有效 B1 观测点", quality=dict(quality),
                )
            else:
                out[mid] = MetricValue(
                    metric_id=mid, value=float(np.mean(curve.values[m])), unit="-",
                    status=out_status, reason=out_reason, flags=out_flags,
                    quality=dict(quality),
                )

    # ---- D2/D3/D4 ----
    if fa is None or activity.unavailable_reason:
        reason = activity.unavailable_reason or "FA 曲线不可用"
        for mid in ("D2_T_start", "D3_T_end", "D4_duration"):
            out[mid] = MetricValue(
                metric_id=mid, value=None, unit="s", status="unavailable",
                reason=reason, quality=dict(quality),
            )
        # D4 的 unavailable 必须显式声明"不得用窗长代替"（B 类边界测试：
        # 无论走哪个分支，持续时长都不得拿观察窗冒充）。
        out["D4_duration"] = MetricValue(
            metric_id="D4_duration", value=None, unit="s", status="unavailable",
            reason=(
                f"{reason}；D2/D3 任一删失，持续时长不可计算"
                "（契约禁止用窗长代替——不得输出 0 或观察窗长冒充持续时长）"
            ),
            quality=dict(quality, window_s=window_s),
        )
        return out

    t_fa, v_fa = fa.valid_t_values()
    calibrated = activity.mu_base is not None and activity.sigma_base is not None

    if calibrated:
        mu, sigma = activity.mu_base, activity.sigma_base
        onset_level = mu + thresholds.onset_sigma * sigma
        t_start = _first_sustained_above(t_fa, v_fa, onset_level, thresholds.onset_hold_s)
        d2_reason: str | None = None
        d2_flags: tuple[str, ...] = ()
        if t_start is None:
            # 判据未触发：退回颗粒口径（fallback 显式标注）
            t_start = pellet_decline_t
            if t_start is not None:
                d2_flags = ("fallback",)
                d2_reason = (
                    "FA 判据（μ+3σ 持续 1s）未触发：退回颗粒首次持续下降口径"
                )
    else:
        t_start = pellet_decline_t
        d2_flags = ("fallback",)
        d2_reason = "基线缺失：D2 退回颗粒首次持续下降口径（fallback）"

    if t_start is None:
        out["D2_T_start"] = MetricValue(
            metric_id="D2_T_start", value=None, unit="s", status="censored",
            reason=(
                "观察窗内未观测到摄食起始（FA 判据与颗粒下降口径均未触发）："
                f"输出 '>窗长'（{window_s:.0f}s），不以 0 冒充"
            ),
            quality=dict(quality, window_s=window_s),
        )
    else:
        status = "ok"
        if d2_flags:
            status = "degraded"
        out["D2_T_start"] = MetricValue(
            metric_id="D2_T_start", value=float(t_start), unit="s", status=status,
            reason=d2_reason, flags=d2_flags,
            quality=dict(quality, window_s=window_s, calibrated=calibrated),
        )

    if calibrated:
        offset_level = mu + thresholds.offset_sigma * sigma
        t_end = _first_sustained_below_after_peak(
            t_fa, v_fa, offset_level, thresholds.offset_hold_s
        )
        d3_flags: tuple[str, ...] = ()
        d3_reason: str | None = None
    else:
        t_end = None
        d3_flags = ("fallback",)
        d3_reason = "基线缺失：饱和判据（μ+2σ 持续 3s）不可用"

    if t_end is None:
        out["D3_T_end"] = MetricValue(
            metric_id="D3_T_end", value=None, unit="s", status="censored",
            reason=(
                "观察窗内活跃度未回落到饱和判据以下：输出 '>窗长'"
                f"（{window_s:.0f}s），不以 0 冒充"
            ) if not d3_reason else (
                f"{d3_reason}；窗内未回落 → 输出 '>窗长'（{window_s:.0f}s）"
            ),
            flags=d3_flags, quality=dict(quality, window_s=window_s),
        )
    else:
        status = "ok" if not d3_flags else "degraded"
        out["D3_T_end"] = MetricValue(
            metric_id="D3_T_end", value=float(t_end), unit="s", status=status,
            reason=d3_reason, flags=d3_flags,
            quality=dict(quality, window_s=window_s, calibrated=calibrated),
        )

    if t_start is None or t_end is None:
        out["D4_duration"] = MetricValue(
            metric_id="D4_duration", value=None, unit="s", status="unavailable",
            reason="D2/D3 任一删失，持续时长不可计算（不得用窗长代替）",
            quality=dict(quality, window_s=window_s),
        )
    else:
        out["D4_duration"] = MetricValue(
            metric_id="D4_duration", value=float(t_end) - float(t_start), unit="s",
            status="ok", reason=None, quality=dict(quality, window_s=window_s),
        )
    return out

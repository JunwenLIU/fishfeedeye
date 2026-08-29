"""aggregator.py · 指标汇总与导出（T04 业务核心，收口版）。

职责（docs/06 §6 T04 + docs/04 §6.3 + team-lead 派单）：
    - 消费 FrameObservation 时序 → A 组（颗粒曲线族）/ B1（活跃度 + 空间
      异质性）/ B2（投喂区）/ D（起止时刻）标量与时序，全部经 MetricValue；
    - 13+ 个 Q_* 质量信号（quality.py）→ 降级矩阵（capability.py）→
      把追加 flag / 降级 / 关闭落到每个 MetricValue（只降级不升级，
      censored/unavailable 语义优先）；
    - **B1 与 B2 物理隔离**：groups 为独立 dict（'A'/'B1'/'B2'/'D'），
      严禁合成单一标量（活跃度与聚集度口径不可比）；
    - 人工修正双轨并列：带 extra['manual_count'] 的帧 → 第二遍全量计算，
      指标 metric_id 加 "_manual" 后缀 + manual_corrected flag，
      **原始值不覆盖**（审计双轨，docs/06 §7.6 重算不覆盖）；
    - run 目录入口：compute_metrics_from_run_dir(run_dir) 一次调用读取
      run_config.yaml + cache/detections.jsonl + corrections.jsonl，
      输出 JSON 可序列化 MetricsReport（供 T05 UI 直接消费）；
    - 落盘 write_run_outputs：metrics_summary.csv（17 列冻结）/
      quality_signals.csv（含 threshold/passed）/ flag_glossary.csv /
      metrics_timeseries.csv（原生不规则时间戳 + dt_s）/ cache/quality.json /
      summary.json。

T04 收口（本轮修复的历史并发写入分裂）：
    - FLAG_GLOSSARY 只保留 capability.py 一份（4 列口径），本模块改为
      导入并再导出，杜绝两份术语表；
    - ManualSummary / 修正留痕读写一律委托 corrections.py，本模块不再
      自带第二套 ManualSummary；
    - quality_signals.csv 行生成统一走 quality.to_quality_rows（含 Q_glare
      行与 threshold/passed 两列），删除本模块的重复实现；
    - 新增 MetricsSummary（17 列冻结 schema 的表格容器，__init__ 导出）；
    - 新增 build_baseline_stats（基线期统计唯一构造器；RP/归一化活跃度
      的硬输入，基线不可得 → 返回 None，绝不造 0）。

17 列 schema（docs/04 §6.3，空值 = 空字符串，绝不为 0）：
    metric_id, metric_name_zh, group, value, unit, unit_scale, status,
    reason, flags, n_frames_used, window_s, window_truncated, method,
    inferentially_valid, blocking_flags, blocking_flag_count,
    metrics_spec_version

任务编号：T04。
"""
from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.core.config import RunConfig, Thresholds
from src.core.frame_context import (
    BaselineStats,
    FishDetections,
    FrameObservation,
    RunMeta,
    Tracks,
)
from src.core.metric_value import MetricValue
from src.core.roi import ROI, point_in_polygon
from src.metrics.capability import (
    FLAG_GLOSSARY,
    CapabilityReport,
    apply_capability,
)
from src.metrics.corrections import (
    ManualSummary,
    apply_manual_counts,
    load_corrections_jsonl,
    summarize_corrections,
)
from src.metrics.group_a.clearance import clearance_metric, crossing_time
from src.metrics.group_a.n0_estimator import estimate_n0
from src.metrics.group_a.nonfeeding_loss import (
    nonfeeding_loss_metric,
    tally_vanish,
)
from src.metrics.group_a.pellet_curve import (
    PelletSeries,
    TimeSeries,
    extract_pellet_series,
    rebound_fraction,
    smooth_series,
)
from src.metrics.group_a.rate import instantaneous_rate, rate_metrics
from src.metrics.group_a.residual import residual_metrics
from src.metrics.group_b1.spatial_heterogeneity import (
    ActivityResult,
    compute_activity,
    d_group_metrics,
)
from src.metrics.group_b2.zone_metrics import zone_metrics
from src.metrics.quality import Q_KEYS, QualitySignals, to_quality_rows

__all__ = [
    "MetricsAggregator",
    "Aggregator",
    "MetricsReport",
    "MetricsSummary",
    "ManualSummary",
    "build_baseline_stats",
    "compute_metrics_from_run_dir",
    "compute_run_metrics",
    "load_corrections",
    "write_run_outputs",
    "quality_rows",
    "summary_row",
    "SUMMARY_COLUMNS",
    "METRIC_NAMES_ZH",
    "METRIC_GROUPS",
    "FLAG_GLOSSARY",
    "MANUAL_SUFFIX",
]

# ----------------------------------------------------------------------
# 常量表（17 列 / 中文指标名 / 组归属）
# ----------------------------------------------------------------------
SUMMARY_COLUMNS: tuple[str, ...] = (
    "metric_id", "metric_name_zh", "group", "value", "unit", "unit_scale",
    "status", "reason", "flags", "n_frames_used", "window_s",
    "window_truncated", "method", "inferentially_valid", "blocking_flags",
    "blocking_flag_count", "metrics_spec_version",
)

METRIC_NAMES_ZH: dict[str, str] = {
    "A1_N0": "初始颗粒数 N₀",
    "A6_v_max": "峰值消耗速率",
    "A7_v50": "半量平均消耗速率",
    "A8_T50": "50% 清空时间",
    "A9_T90": "90% 清空时间",
    "A10_T100": "100% 清空时间",
    "A11_RR": "残留率",
    "A12_k": "指数衰减速率常数",
    "A12_tau": "衰减半衰期",
    "A13_AUC60": "60s 累计消耗积分",
    "A14_NonFeedingLoss": "非摄食损失率",
    "B1_frame_diff": "帧差能量均值（归一化）",
    "B1_kurtosis_mean": "空间异质性·峰度均值",
    "B1_gini_mean": "空间异质性·基尼均值",
    "B1_top5_share_mean": "前5%格子能量占比",
    "B2-5_N_fz_mean": "投喂区平均鱼数",
    "B2-6_P_fz": "投喂区鱼数占比",
    "B2-7_RP": "投喂区相对偏好系数",
    "D2_T_start": "首次摄食潜伏期",
    "D3_T_end": "饱和时刻",
    "D4_duration": "摄食持续时长",
}

METRIC_GROUPS: dict[str, str] = {
    "A1_N0": "A", "A6_v_max": "A", "A7_v50": "A", "A8_T50": "A",
    "A9_T90": "A", "A10_T100": "A", "A11_RR": "A", "A12_k": "A",
    "A12_tau": "A", "A13_AUC60": "A", "A14_NonFeedingLoss": "A",
    "B1_frame_diff": "B1",
    "B1_kurtosis_mean": "B1", "B1_gini_mean": "B1",
    "B1_top5_share_mean": "B1",
    "B2-5_N_fz_mean": "B2", "B2-6_P_fz": "B2", "B2-7_RP": "B2",
    "D2_T_start": "D", "D3_T_end": "D", "D4_duration": "D",
}

# inferentially_valid（docs/04 §6.1.4）：跨组统计推断可用性，正交于优先级。
# 判据：门控是否中立（外部条件 → yes；依赖被测变量 → conditional；
# 缺失具信息性 → no）。
_INFERENTIALLY_VALID: dict[str, str] = {
    "B1_frame_diff": "yes",            # 门控 Q_interf/Q_motion = 外部环境量
    "B1_kurtosis_mean": "yes",
    "B1_gini_mean": "yes",
    "B1_top5_share_mean": "yes",
    "B2-5_N_fz_mean": "conditional",   # 依赖鱼数计数（Q_overlap 非中立）
    "B2-6_P_fz": "conditional",
    "B2-7_RP": "conditional",
    "D2_T_start": "conditional",       # 依赖基线（Q_baseline 外部，但判据含 FA）
    "D3_T_end": "conditional",
    "D4_duration": "conditional",
}

MANUAL_SUFFIX = "_manual"


# ----------------------------------------------------------------------
# 报告结构
# ----------------------------------------------------------------------
@dataclass
class MetricsReport:
    """aggregate / compute_metrics_from_run_dir 的输出（内存态）。

    Attributes:
        run_id: run 标识（留档用；可为 None）。
        config: 运行配置（metrics_spec_version / thresholds 锚点）。
        groups: 组名 → {metric_id: MetricValue}，'A'/'B1'/'B2'/'D'
            物理隔离（B1 活跃度与 B2 聚集度严禁合成单一标量）。
        metrics: 全组展平（自动口径 + "_manual" 人工修正口径）。
        manual_groups: 人工修正并列口径（无修正帧时为 None，不输出空壳）。
        manual_summary: 修正程度留档（无修正时为 None）。
        timeseries: 时序列表（A2_Np / A5_v / M_diff / FA / B1 网格 / N_fz）。
        quality_signals: 13+ Q_* 全键（None = 未测得，绝不为 0/False 冒充）。
        quality_table: quality_signals.csv 行（signal/value/threshold/passed）。
        capability: 降级矩阵应用结果。
        window_s: 实际观察窗长（试验期最大 t）。
        window_truncated: 视频短于标称观察窗。
        warnings / notes: 告警与备注。
    """

    run_id: str | None
    config: RunConfig
    groups: dict[str, dict[str, MetricValue]] = field(default_factory=dict)
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    manual_groups: dict[str, dict[str, MetricValue]] | None = None
    manual_summary: ManualSummary | None = None
    timeseries: list[TimeSeries] = field(default_factory=list)
    quality_signals: dict[str, Any] = field(default_factory=dict)
    quality_table: list[dict[str, Any]] = field(default_factory=list)
    capability: CapabilityReport | None = None
    window_s: float = 0.0
    window_truncated: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def by_group(self, group: str) -> dict[str, MetricValue]:
        """取单个组的指标（B1/B2 物理隔离视图）。"""
        return dict(self.groups.get(group, {}))

    # ------------------------------------------------------------------
    def summary_rows(self) -> list[dict[str, Any]]:
        """17 列冻结 schema 行（空值为空字符串，绝不为 0）。"""
        return [
            summary_row(mid, mv, self.config.metrics_spec_version)
            for mid, mv in (
                (m, self.metrics[m]) for m in _metric_order(self.metrics)
            )
        ]

    # ------------------------------------------------------------------
    def to_summary(self) -> "MetricsSummary":
        """17 列冻结表格容器（T05 UI / 导出层消费）。"""
        return MetricsSummary(
            run_id=self.run_id,
            metrics_spec_version=self.config.metrics_spec_version,
            columns=list(SUMMARY_COLUMNS),
            rows=self.summary_rows(),
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """summary.json 主体（JSON 可序列化，供 T05 UI 直接消费）。"""
        return {
            "run_id": self.run_id,
            "metrics_spec_version": self.config.metrics_spec_version,
            "run_config": self.config.to_dict(),
            "window_s": self.window_s,
            "window_truncated": self.window_truncated,
            "groups": {
                g: {mid: mv.to_dict() for mid, mv in d.items()}
                for g, d in self.groups.items()
            },
            "metrics": {mid: mv.to_dict() for mid, mv in self.metrics.items()},
            "manual_groups": (
                None if self.manual_groups is None else
                {g: {mid: mv.to_dict() for mid, mv in d.items()}
                 for g, d in self.manual_groups.items()}
            ),
            "manual_summary": (
                None if self.manual_summary is None
                else self.manual_summary.to_dict()
            ),
            "timeseries": [ts.to_dict() for ts in self.timeseries],
            "quality_signals": dict(self.quality_signals),
            "quality_table": list(self.quality_table),
            "capability": (
                None if self.capability is None else self.capability.to_dict()
            ),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


@dataclass
class MetricsSummary:
    """17 列冻结 schema 的指标汇总表（docs/04 §6.1）。

    为什么单独成类：metrics_summary.csv 是**跨 run 比较与统计的输入**，
    必须有独立的、可序列化的表格载体（不能只藏在 MetricsReport 方法里）；
    T05 的 compare / 导出 / UI 三处都要消费同一份行序与列序。

    Attributes:
        run_id: run 标识。
        metrics_spec_version: 决定"什么叫干净"的清单版本（跨 run 必校项）。
        columns: 17 列列名（冻结，顺序即 CSV 列序）。
        rows: 行列表（每行为 17 键 dict；空值为空字符串，绝不为 0）。
    """

    run_id: str | None = None
    metrics_spec_version: str = "ms-v1"
    columns: list[str] = field(default_factory=lambda: list(SUMMARY_COLUMNS))
    rows: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def from_report(cls, report: "MetricsReport") -> "MetricsSummary":
        """从 MetricsReport 构造（列序与行序唯一来源）。"""
        return report.to_summary()

    # ------------------------------------------------------------------
    def metric_ids(self) -> list[str]:
        """行序中的指标 ID 列表。"""
        return [str(r.get("metric_id", "")) for r in self.rows]

    def get(self, metric_id: str) -> dict[str, Any] | None:
        """按 metric_id 取行（该 ID 未输出 → None，不造空行冒充）。"""
        for r in self.rows:
            if r.get("metric_id") == metric_id:
                return r
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "metrics_spec_version": self.metrics_spec_version,
            "columns": list(self.columns),
            "rows": list(self.rows),
        }


def summary_row(
    mid: str, mv: MetricValue, metrics_spec_version: str
) -> dict[str, Any]:
    """单个 MetricValue → 17 列冻结 schema 行（空值纪律的唯一实现）。

    空值纪律（docs/04 §6.1.2 ③）：
        - unavailable/censored → value 空、blocking_flag_count 空（绝不 0）；
        - 无内容 → 空字符串，绝不写 "none"/"-"/"OK"。
    """
    base_id = mid[: -len(MANUAL_SUFFIX)] if mid.endswith(MANUAL_SUFFIX) else mid
    bf = mv.blocking_flags()
    status = mv.status
    if status in ("unavailable", "censored"):
        bf_count: int | None = None  # 空，绝不为 0（空值纪律）
    else:
        bf_count = len(bf)  # degraded 恒 ≥ 1（status:degraded 伪 flag）
    n_used = (
        mv.quality.get("n_frames_used")
        if mv.quality.get("n_frames_used") is not None
        else mv.quality.get("n_points")
    )
    return {
        "metric_id": mid,
        "metric_name_zh": METRIC_NAMES_ZH.get(base_id, ""),
        "group": METRIC_GROUPS.get(base_id, _group_of(mid)),
        "value": _fmt_value(mv.value),
        "unit": mv.unit or "",
        "unit_scale": mv.unit_scale,
        "status": status,
        "reason": mv.reason or "",
        "flags": ";".join(mv.flags),
        "n_frames_used": _num_or_empty(n_used),
        "window_s": _num_or_empty(mv.quality.get("window_s")),
        "window_truncated": _bool_or_empty(mv.quality.get("window_truncated")),
        "method": _method_of(mv),
        "inferentially_valid": _inferentially_valid(base_id, mv),
        "blocking_flags": ";".join(bf),
        "blocking_flag_count": _num_or_empty(bf_count),
        "metrics_spec_version": metrics_spec_version,
    }


def _inferentially_valid(base_id: str, mv: MetricValue) -> str:
    """跨组统计推断可用性（docs/04 §6.1.4）。

    status 非 ok 的指标本身不可用于统计推断；A 组默认 conditional
    （需 Q_n0gap 通过 + A14 校正），B1 帧差/异质性为 yes（门控中立）。
    """
    if mv.status != "ok" and mv.status != "degraded":
        return "no"
    return _INFERENTIALLY_VALID.get(base_id, "conditional")


# ----------------------------------------------------------------------
# 基线统计构造（RP / 归一化活跃度的硬输入）
# ----------------------------------------------------------------------
def build_baseline_stats(
    observations: Sequence[FrameObservation],
    roi: ROI | None = None,
) -> BaselineStats | None:
    """从基线期（t < 0）观测构造 BaselineStats（唯一构造器）。

    docs/04 §1：基线期 = [t0 − baseline_window, t0)，最短 30 s；
    基线缺失（帧数 < 2 或时长 < 30 s）→ 返回 **None**（BaselineStats
    存在即代表 Q_baseline=True），绝不返回"看起来能用的"空壳统计。

    Returns:
        BaselineStats | None。None = 基线不可用（RP 与归一化活跃度随后
        被强制关闭，绝对值口径保留）。
    """
    base_obs = [o for o in observations if o.t_s < -1e-9]
    if len(base_obs) < 2:
        return None
    duration_s = float(max(o.t_s for o in base_obs) - min(o.t_s for o in base_obs))
    if duration_s < 30.0:
        return None

    feed_zone = roi.feeding_zone if roi is not None else None
    fish_counts: list[float] = []
    n_fz_counts: list[float] = []
    activity: list[float] = []

    prev_gray: np.ndarray | None = None
    prev_t: float | None = None
    for obs in sorted(base_obs, key=lambda o: o.t_s):
        fish = obs.extra.get("fish")
        if isinstance(fish, FishDetections):
            fish_counts.append(float(fish.n_det()))
            if feed_zone is not None and fish.n_det() > 0:
                cents = fish.centroids()
                n_fz_counts.append(
                    float(
                        sum(
                            1
                            for c in cents
                            if point_in_polygon(
                                (float(c[0]), float(c[1])), feed_zone
                            )
                        )
                    )
                )
        if obs.image is not None:
            import cv2

            gray = cv2.cvtColor(obs.image, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and prev_t is not None:
                dt = float(obs.t_s - prev_t)
                if dt > 1e-9:
                    activity.append(
                        float(
                            np.mean(
                                np.abs(
                                    gray.astype(np.float32)
                                    - prev_gray.astype(np.float32)
                                )
                            )
                        )
                        / dt
                    )
            prev_gray, prev_t = gray, obs.t_s

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _std(xs: list[float]) -> float:
        return float(np.std(xs)) if len(xs) >= 2 else 0.0

    annd_mean = _mean(n_fz_counts)  # 无个体级输入时的保守占位（见下 note）
    return BaselineStats(
        n_frames=len(base_obs),
        duration_s=duration_s,
        fish_count_mean=_mean(fish_counts),
        fish_count_std=_std(fish_counts),
        n_fz_mean=_mean(n_fz_counts),
        n_fz_std=_std(n_fz_counts),
        activity_mean=_mean(activity),
        activity_std=_std(activity),
        flow_mean=0.0,
        flow_std=0.0,
        annd_mean=annd_mean,
        annd_std=0.0,
        pellet_count_mean=0.0,
    )


# ----------------------------------------------------------------------
# 内部：单口径（自动或人工修正）全量计算
# ----------------------------------------------------------------------
@dataclass
class _CoreResult:
    """_compute_core 的内部产物。"""

    metrics: dict[str, MetricValue] = field(default_factory=dict)
    timeseries: list[TimeSeries] = field(default_factory=list)
    n0: float | None = None
    n0_meta: float | None = None
    t50: float | None = None
    q_n0gap: float | None = None
    q_pelletloss: float | None = None
    q_censored: bool = False
    fa_dynamic_range: float | None = None
    activity: ActivityResult | None = None
    unstable_frac: float = 0.0


def _pellet_first_decline(
    series: PelletSeries, n0: float | None, hold: int = 2
) -> float | None:
    """D2 回退口径：颗粒首次持续下降时刻（hold+1 个采样点均低于阈值）。"""
    if not series.available or n0 is None or n0 <= 0:
        return None
    mask = series.trial_mask()
    t, n = series.t[mask], series.n[mask]
    if t.size < hold + 1:
        return None
    thr = n0 - max(1.0, 0.02 * n0)
    for i in range(n.size - hold):
        if all(n[i + k] <= thr for k in range(hold + 1)):
            j = i
            while j > 0 and n[j - 1] >= n[j]:
                j -= 1  # 回溯到下降起点
            return float(t[j])
    return None


def _fa_dynamic_range(activity: ActivityResult) -> float | None:
    """FA 试验窗峰值（z 分数口径；烈度自动分档判据，None = UNKNOWN 档）。"""
    fa = activity.fa_curve
    if fa is None or activity.mu_base is None:
        return None
    t, v = fa.valid_t_values()
    trial = t >= -1e-9
    if not np.any(trial):
        return None
    peak = float(np.max(v[trial]))
    return peak if np.isfinite(peak) else None


def _compute_core(
    observations: Sequence[FrameObservation],
    th: Thresholds,
    meta: RunMeta,
    roi: ROI | None,
    baseline: BaselineStats | None,
    px_per_mm: float | None,
    link_result: Any,
    window_s: float,
    truncated: bool,
    video_duration_s: float | None,
    pellet_type: str | None,
) -> _CoreResult:
    """单口径全量计算（自动轨与人工修正轨共用；唯一计算路径）。"""
    res = _CoreResult()
    q_base: dict[str, Any] = {
        "Q_calib": px_per_mm is not None,
        "Q_baseline": baseline is not None and baseline.ok,
    }
    obs_sorted = sorted(observations, key=lambda o: o.t_s)
    t_max_s = float(obs_sorted[-1].t_s) if obs_sorted else None

    # ---- A2 序列（提取 → 平滑）----
    series = extract_pellet_series(obs_sorted, roi, th)
    if not series.available:
        res.metrics["A1_N0"] = MetricValue(
            metric_id="A1_N0", value=None, unit="颗", status="unavailable",
            reason=series.reason or "颗粒序列不可用", quality=dict(q_base),
        )
    smoothed = smooth_series(series, th.smooth_window)

    # ---- A1 N₀ 双轨 ----
    if series.available:
        n0_mv = estimate_n0(
            series, meta=meta, thresholds=th, video_duration_s=video_duration_s
        )
        res.metrics["A1_N0"] = n0_mv
        res.n0 = n0_mv.value
        res.n0_meta = n0_mv.quality.get("N0_meta")
        res.q_n0gap = n0_mv.quality.get("Q_n0gap")
    q_base["Q_n0gap"] = res.q_n0gap
    n0 = res.n0

    base_flags: tuple[str, ...] = ()
    n0_mv = res.metrics.get("A1_N0")
    if n0_mv is not None and "denominator_suspect" in n0_mv.flags:
        base_flags = ("denominator_suspect",)

    # 计数稳定性（回补帧占比；>10% → counting_unstable）
    trial_mask = smoothed.trial_mask()
    unstable_frac = (
        rebound_fraction(smoothed.n[trial_mask])
        if smoothed.available and int(np.count_nonzero(trial_mask)) >= 2 else 0.0
    )
    res.unstable_frac = unstable_frac
    if unstable_frac > 0.10:
        base_flags = tuple(dict.fromkeys(base_flags + ("counting_unstable",)))

    # ---- A14 非摄食损失（先于清空时间：Q_pelletloss 是其硬输入）----
    tally = tally_vanish(link_result, obs_sorted, t_max_s, th)
    bottom_band_defined = roi is not None and roi.bottom_band is not None
    a14_mv, q_pelletloss = nonfeeding_loss_metric(
        tally, n0 if n0 is not None else 0.0,
        bottom_band_defined=bottom_band_defined, thresholds=th,
    )
    res.metrics["A14_NonFeedingLoss"] = a14_mv
    res.q_pelletloss = q_pelletloss
    q_base["Q_pelletloss"] = q_pelletloss

    # ---- A8/A9/A10 清空时间 ----
    t50_raw: float | None = None
    for mid, p, use_eps in (
        ("A8_T50", 0.5, False),
        ("A9_T90", 0.1, False),
        ("A10_T100", 0.0, True),
    ):
        t_cross = (
            crossing_time(smoothed, n0, p, window_s, th, use_epsilon=use_eps)
            if n0 is not None else None
        )
        if mid == "A8_T50":
            t50_raw = t_cross
        extra = base_flags
        if mid == "A10_T100":
            # T100 恒加并列 flag（docs/04 §3.1：任何时候不得单独对外）
            extra = tuple(dict.fromkeys(base_flags + ("must_pair_with_pelletloss",)))
        res.metrics[mid] = clearance_metric(
            mid, t_cross, window_s,
            q_pelletloss=q_pelletloss,
            pellet_type=pellet_type,
            n_frames_used=smoothed.n_frames_used(),
            extra_flags=extra,
            thresholds=th,
        )
        if res.metrics[mid].status == "censored":
            res.q_censored = True
    res.t50 = t50_raw

    # ---- A5/A6/A7 速率 ----
    if smoothed.available:
        valid = smoothed.valid_mask()
        t_fit = smoothed.t[valid]
        n_fit = smoothed.n[valid]
    else:
        t_fit = np.zeros(0)
        n_fit = np.zeros(0)
    rates = rate_metrics(
        t_fit, n_fit, n0, res.t50, th, window_s,
        quality=dict(q_base), base_flags=base_flags,
        unstable_frac=unstable_frac,
    )
    res.metrics.update(rates)
    # A5 v(t) 时序（原生 dt 差分；时序表输出，不进标量表）
    if t_fit.size >= 2:
        v_series = instantaneous_rate(t_fit, n_fit)
        res.timeseries.append(TimeSeries(
            metric_id="A5_v", t=t_fit, values=v_series, unit="颗/s",
            note="v=-dN/dt; monotone-cummin; native dt",
        ))

    # ---- A11/A12/A13 残留/拟合/积分 ----
    resid = residual_metrics(
        t_fit, n_fit.copy(), n0, th, window_s, truncated,
        quality=dict(q_base), base_flags=base_flags,
        pellet_type=pellet_type or "floating",
        q_pelletloss=q_pelletloss,
    )
    res.metrics.update(resid)

    # ---- A2 N_p(t) 时序（平滑轨；基线期不进曲线）----
    if smoothed.available and int(np.count_nonzero(trial_mask)) > 0:
        res.timeseries.insert(0, TimeSeries(
            metric_id="A2_Np", t=smoothed.t[trial_mask],
            values=smoothed.n[trial_mask], unit="颗",
            note=f"source={smoothed.source};smooth=median-{th.smooth_window}",
        ))

    # ---- B1 + D 组（需帧图像；缓存恢复模式 → unavailable）----
    activity = compute_activity(obs_sorted, roi, th)
    res.activity = activity
    res.fa_dynamic_range = _fa_dynamic_range(activity)
    pellet_decline_t = _pellet_first_decline(smoothed, n0)
    bd_metrics = d_group_metrics(activity, dict(q_base), th, pellet_decline_t)
    for mid, mv in bd_metrics.items():
        if mid.startswith("D") and mv.status == "censored":
            res.q_censored = True
        res.metrics[mid] = mv

    # ---- B2 组（投喂区；探索性）----
    b2_metrics, n_fz_series = zone_metrics(
        obs_sorted, roi, meta, baseline, dict(q_base)
    )
    res.metrics.update(b2_metrics)
    if n_fz_series is not None:
        res.timeseries.append(n_fz_series)

    # ---- 时序补充（M_diff / FA / B1 网格）----
    for curve in (
        activity.m_curve, activity.fa_curve,
        activity.kurtosis_curve, activity.gini_curve, activity.top5_curve,
    ):
        if curve is not None:
            res.timeseries.append(curve)
    return res


# ----------------------------------------------------------------------
# B1 帧差标量（B1-1 主口径）
# ----------------------------------------------------------------------
def _b1_frame_diff(activity: ActivityResult, signals: dict[str, Any]) -> MetricValue:
    """B1 帧差能量均值（试验窗；固定 arena 面积归一化 + 真实 dt 速率）。"""
    if activity.m_curve is None:
        return MetricValue(
            metric_id="B1_frame_diff", value=None, unit="gray/s/px",
            status="unavailable",
            reason=activity.unavailable_reason or "帧差曲线不可用",
            quality=dict(signals),
        )
    t, v = activity.m_curve.valid_t_values()
    trial = t >= -1e-9
    n_trial = int(np.count_nonzero(trial))
    if n_trial == 0:
        return MetricValue(
            metric_id="B1_frame_diff", value=None, unit="gray/s/px",
            status="unavailable",
            reason="试验窗内无帧差观测点（基线期图像之外的帧不可得）",
            quality=dict(signals),
        )
    flags: list[str] = []
    status = "ok"
    reason: str | None = None
    if activity.mu_base is None:
        flags.append("uncalibrated")
        status = "degraded"
        reason = "基线缺失：只输出未归一化绝对量（uncalibrated）"
    if activity.alpha is None:
        # 参考区未定义或 α 不可估计：不得静默不扣除
        flags.append("no_noise_correction")
        if status == "ok":
            status = "degraded"
            reason = "未做参考区噪声扣除（no_noise_correction）"
    q = dict(signals)
    q.update(
        {
            "n_frames_used": n_trial,
            "method": "frame_diff;arena_area_norm;dt_rate",
            "integral_method": "trapz",
        }
    )
    return MetricValue(
        metric_id="B1_frame_diff",
        value=float(np.mean(v[trial])),
        unit="gray/s/px",
        status=status,
        reason=reason,
        flags=tuple(flags),
        quality=q,
    )


# ----------------------------------------------------------------------
# 降级矩阵应用（capability → MetricValue）
# ----------------------------------------------------------------------
_CAPABILITY_ID_ALIASES: dict[str, tuple[str, ...]] = {
    "B1_frame_diff": ("B1_frame_diff",),
    "B1_spatial_heterogeneity": (
        "B1_kurtosis_mean", "B1_gini_mean", "B1_top5_share_mean",
    ),
    "B1_flow_mm_s": (),
    "B2-5_N_fz_mean": ("B2-5_N_fz_mean",),
    "B2-6_P_fz": ("B2-6_P_fz",),
    "B2-7_RP": ("B2-7_RP",),
    "B2-1_ANND": (),
    "B2-2_MDC": (),
    "B2-4_FIFFB": (),
    "A4_P_end": (),
    "D2_T_start": ("D2_T_start",),
}


def _apply_capability_to_metrics(
    metrics: dict[str, MetricValue],
    groups: dict[str, dict[str, MetricValue]],
    capability: CapabilityReport,
) -> tuple[dict[str, MetricValue], dict[str, dict[str, MetricValue]]]:
    """把 CapabilityReport 的 metric_flags / degrade_reasons 落到 MetricValue。

    - flags：合并进现有 flags（去重；MetricValue 不可变 → 重建）；
    - degrade_reasons：仅当当前 status=='ok' 时降为 degraded（censored /
      unavailable 的语义优先，绝不被降级覆盖——删失≠失败）；
    - 关闭清单（disabled_with_reason / closed_groups）保留在 report 中
      显式列出，不静默删除指标行（B2-7_RP 等关闭条件在指标内部已同步
      落实为 unavailable + reason，双保险）。
    """
    def _targets(src_id: str) -> list[str]:
        if src_id in metrics:
            return [src_id]
        return [t for t in _CAPABILITY_ID_ALIASES.get(src_id, ()) if t in metrics]

    for src_id, flags in capability.metric_flags.items():
        for tgt in _targets(src_id):
            mv = metrics[tgt]
            if flags and not set(flags) <= set(mv.flags):
                metrics[tgt] = _with_flags(mv, tuple(flags))
    for src_id, reason in capability.degrade_reasons.items():
        for tgt in _targets(src_id):
            mv = metrics[tgt]
            if mv.status == "ok":
                metrics[tgt] = MetricValue(
                    metric_id=mv.metric_id,
                    value=mv.value,
                    unit=mv.unit,
                    status="degraded",
                    reason=reason,
                    flags=mv.flags,
                    quality=dict(mv.quality),
                    unit_scale=mv.unit_scale,
                )
    # 同步回 groups（物理隔离的 B1/B2 视图保持一致）
    for g, m in groups.items():
        for mid in list(m.keys()):
            if mid in metrics:
                m[mid] = metrics[mid]
    return metrics, groups


# ----------------------------------------------------------------------
# 聚合器
# ----------------------------------------------------------------------
class MetricsAggregator:
    """聚合入口（classDiagram Aggregator：aggregate(frames, tracks, signals)）。"""

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config if config is not None else RunConfig()
        self.th = self.config.thresholds

    # ------------------------------------------------------------------
    def aggregate(
        self,
        observations: Sequence[FrameObservation],
        meta: RunMeta | None = None,
        roi: ROI | None = None,
        baseline: BaselineStats | None = None,
        tracks: Tracks | None = None,
        link_result: Any | None = None,
        px_per_mm: float | None = None,
        video_duration_s: float | None = None,
        outdoor: bool = False,
        extra_quality: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> MetricsReport:
        """执行全量指标计算（A + B1 + B2 + D + quality + capability + 双轨）。

        Args:
            observations: 帧观测（image 可为 None：缓存恢复模式，B1 不可用；
                含 extra['manual_count'] 的帧触发人工修正双轨）。
            meta: 用户元数据（pellet_type / feed_mass 等参与计算与门控）。
            roi: 区域定义。
            baseline: 基线期统计（None → Q_baseline=False，相对指标关闭）。
            tracks: 轻量关联轨迹（Q_track 消费）。
            link_result: PelletLinker 输出（A14 消费）。
            px_per_mm: 标定尺度（Q_calib）。
            video_duration_s: 视频时长（N₀ 早期窗口收缩判据）。
            outdoor: 户外斜拍（用户确认位；B2/C 只降级不关闭）。
            extra_quality: 上游注入的额外质量信号（如 Q_dualtrack_gap）。
        """
        meta = meta if meta is not None else RunMeta()
        obs_list = list(observations)
        obs_sorted = sorted(obs_list, key=lambda o: o.t_s)
        warnings: list[str] = []
        notes: list[str] = []
        th = self.th

        # ---- 观察窗（试验期最大 t；视频短于标称窗 → truncated）----
        trial_ts = [o.t_s for o in obs_sorted if o.t_s >= 0.0]
        window_s = float(max(trial_ts)) if trial_ts else 0.0
        truncated = bool(window_s < th.observation_window_s - 1e-9)
        if truncated:
            notes.append(
                f"观察窗截断：实际 {window_s:.0f}s < 标称 "
                f"{th.observation_window_s}s（window_truncated=True）"
            )
        if video_duration_s is None and obs_sorted:
            video_duration_s = (
                float(obs_sorted[-1].t_s - obs_sorted[0].t_s) if obs_sorted else None
            )
        pellet_type = (
            meta.pellet_type if meta.pellet_type is not None else self.config.pellet_type
        )
        px = px_per_mm if px_per_mm is not None else self.config.px_per_mm_ref

        # ---- 自动口径（剥离 manual_count，原始观测不被修改）----
        auto_obs = _strip_manual(obs_sorted)
        core = _compute_core(
            auto_obs, th, meta, roi, baseline, px, link_result,
            window_s, truncated, video_duration_s, pellet_type,
        )
        if core.metrics["A1_N0"].status != "unavailable" and core.n0 is None:
            notes.append("N₀ 不可用：所有以 N₀ 为分母的指标已标 unavailable")

        # ---- 13 Q_*（quality.compute + A1/A14/A8-A10 注入）----
        signals = QualitySignals().compute(
            auto_obs,
            tracks=tracks,
            link_result=link_result,
            baseline=baseline,
            meta=meta,
            n0_det=core.n0,
            n0_meta=core.n0_meta,
            px_per_mm=px,
            thresholds=th,
        )
        if core.q_pelletloss is not None:
            signals["Q_pelletloss"] = core.q_pelletloss
        if core.q_n0gap is not None:
            signals["Q_n0gap"] = core.q_n0gap
        signals["Q_censored"] = bool(core.q_censored)
        if extra_quality:
            signals.update(extra_quality)
        for key in Q_KEYS:
            signals.setdefault(key, None)  # "None 也要有行" 纪律兜底

        # ---- B1-1 帧差标量 ----
        core.metrics["B1_frame_diff"] = _b1_frame_diff(core.activity, signals)

        # ---- capability 降级矩阵 ----
        capability = apply_capability(
            signals, th, meta=meta,
            fa_dynamic_range=core.fa_dynamic_range, outdoor=outdoor,
        )

        # ---- 组装（B1 与 B2 物理隔离）----
        groups: dict[str, dict[str, MetricValue]] = {
            "A": {}, "B1": {}, "B2": {}, "D": {},
        }
        metrics: dict[str, MetricValue] = {}
        for mid, mv in core.metrics.items():
            g = METRIC_GROUPS.get(mid, _group_of(mid))
            groups.setdefault(g, {})[mid] = mv
            metrics[mid] = mv
        metrics, groups = _apply_capability_to_metrics(metrics, groups, capability)

        # ---- 告警汇总（覆盖写 warnings，每次重算全量重生成）----
        if core.unstable_frac > 0.10:
            warnings.append(
                f"counting_unstable：N_p 非单调上升段占比 "
                f"{core.unstable_frac:.1%} > 10%，速率指标仅供定性"
            )
        if signals.get("Q_n0gap") is not None and signals["Q_n0gap"] > th.n0_gap_warn:
            warnings.append(
                f"Q_n0gap = {signals['Q_n0gap']:.1%} > {th.n0_gap_warn:.0%}："
                "N₀ 双轨偏差超阈（可能漏检或口径不符），归一化颗粒指标"
                "标记 denominator_suspect"
            )
        t50_mv = core.metrics.get("A8_T50")
        if t50_mv is not None and t50_mv.status == "censored":
            warnings.append(
                f"T50 右删失：>{window_s:.0f}s（输出下界，绝不为 0）"
            )
        a14_mv = core.metrics.get("A14_NonFeedingLoss")
        if a14_mv is not None and a14_mv.status == "unavailable":
            notes.append(
                "A14 不可用：任何清空时间类指标都不得声明已校正沉降损失"
            )
        if pellet_type in (None, "unknown"):
            warnings.append(
                "pellet_type 未确认：T 系列标记 sedimentation_risk_unassessed"
                "（未确认元数据只加标记，不关闭）"
            )
        warnings.extend(core.activity.warnings)

        # ---- 人工修正双轨（并列重算，不覆盖原值）----
        manual_groups: dict[str, dict[str, MetricValue]] | None = None
        manual_summary: ManualSummary | None = None
        corrected = [o for o in obs_sorted if o.extra.get("manual_count") is not None]
        if corrected:
            core_m = _compute_core(
                obs_sorted, th, meta, roi, baseline, px, link_result,
                window_s, truncated, video_duration_s, pellet_type,
            )
            core_m.metrics["B1_frame_diff"] = _b1_frame_diff(core_m.activity, signals)
            manual_groups = {"A": {}, "B1": {}, "B2": {}, "D": {}}
            for mid, mv in core_m.metrics.items():
                mid_m = f"{mid}{MANUAL_SUFFIX}"
                mv_m = _with_flags(
                    dataclasses.replace(mv, metric_id=mid_m),
                    ("manual_corrected",),
                )
                g = METRIC_GROUPS.get(mid, _group_of(mid))
                manual_groups.setdefault(g, {})[mid_m] = mv_m
            manual_summary = summarize_corrections(
                obs_sorted,
                [
                    {
                        "frame_idx": int(o.frame_idx),
                        "t_s": float(o.t_s),
                        "count": int(o.extra["manual_count"]),
                        "operator": str(o.extra.get("manual_count_operator") or "user"),
                    }
                    for o in corrected
                ],
                source="corrections_jsonl",
            )
            for g, m in manual_groups.items():
                for mid_m, mv_m in m.items():
                    metrics[mid_m] = mv_m
            notes.append(
                f"人工修正 {manual_summary.corrected_frames} 帧（占 "
                f"{manual_summary.corrected_fraction:.1%}）：_manual 口径与"
                "原始口径并列，原始值不被覆盖"
            )

        return MetricsReport(
            run_id=run_id,
            config=self.config,
            groups=groups,
            metrics=metrics,
            manual_groups=manual_groups,
            manual_summary=manual_summary,
            timeseries=core.timeseries,
            quality_signals=signals,
            quality_table=to_quality_rows(signals, th),
            capability=capability,
            window_s=window_s,
            window_truncated=truncated,
            warnings=warnings,
            notes=notes,
        )


# ----------------------------------------------------------------------
# 修正注入与剥离（统一委托 corrections.py）
# ----------------------------------------------------------------------
def load_corrections(run_dir: str | Path) -> list[dict[str, Any]]:
    """读 run 目录下 corrections.jsonl（委托 corrections.load_corrections_jsonl）。"""
    return load_corrections_jsonl(Path(run_dir) / "corrections.jsonl")


def _strip_manual(
    observations: Sequence[FrameObservation],
) -> list[FrameObservation]:
    """剥离 manual_count 标记（自动口径的输入净化；原始观测不被修改）。"""
    out: list[FrameObservation] = []
    for o in observations:
        if o.extra.get("manual_count") is None:
            out.append(o)
        else:
            extra = {
                k: v for k, v in o.extra.items()
                if not k.startswith("manual_count")
            }
            out.append(dataclasses.replace(o, extra=extra))
    return out


def apply_manual_corrections(
    observations: Sequence[FrameObservation],
    corrections: Sequence[dict[str, Any]],
) -> list[FrameObservation]:
    """把 corrections.jsonl 记录注入观测**副本**（不修改输入序列）。

    纪律（docs/06 §7.12）：任何人工修正只留痕、不静默；原始观测对象
    不被就地改写，故返回新列表（inplace=False 委托 corrections 层）。
    """
    return apply_manual_counts(observations, corrections, inplace=False)


# ----------------------------------------------------------------------
# 便捷入口
# ----------------------------------------------------------------------
def compute_run_metrics(
    run_result: Any,
    roi: ROI | None = None,
    baseline: BaselineStats | None = None,
    tracks: Tracks | None = None,
    outdoor: bool = False,
) -> MetricsReport:
    """从 Orchestrator.RunResult 聚合指标（含 corrections 消费）。

    消费顺序（幂等）：
        1. observations 中已有 extra['manual_count']（recompute_metrics
           注入）→ 直接生效；
        2. run_dir/corrections.jsonl 存在而某帧尚无 manual_count → 注入
           副本（对应"load_cache 后直接算指标"的路径，留痕文件不修改）。
    """
    obs_list = list(run_result.observations)
    corrections = load_corrections(run_result.run_dir)
    if corrections:
        obs_list = apply_manual_counts(obs_list, corrections, inplace=False)

    extra_quality: dict[str, Any] = {}
    qs = getattr(run_result, "quality_signals", None) or {}
    if qs.get("Q_dualtrack_gap_max") is not None:
        extra_quality["Q_dualtrack_gap"] = qs["Q_dualtrack_gap_max"]

    if baseline is None:
        baseline = build_baseline_stats(obs_list, roi)

    agg = MetricsAggregator(config=run_result.config)
    return agg.aggregate(
        obs_list,
        meta=run_result.meta,
        roi=roi,
        baseline=baseline,
        tracks=tracks,
        link_result=run_result.link_result,
        px_per_mm=getattr(run_result.config, "px_per_mm_ref", None),
        outdoor=outdoor,
        extra_quality=extra_quality or None,
        run_id=run_result.run_id,
    )


def _meta_from_json(path: Path) -> RunMeta | None:
    """从 run 目录 meta.json 恢复 RunMeta（T05 写入；缺失 → None）。"""
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        return None
    valid = {
        "species", "n_fish_total", "body_length_mm", "body_weight_g",
        "feed_mass_g", "pellet_mass_mg", "pellet_type", "water_temp_c",
        "pond_id", "group_label_encrypted", "blind_code",
    }
    return RunMeta(**{k: v for k, v in d.items() if k in valid})


def compute_metrics_from_run_dir(
    run_dir: str | Path,
    roi: ROI | None = None,
    baseline: BaselineStats | None = None,
    tracks: Tracks | None = None,
    outdoor: bool = False,
    meta: RunMeta | None = None,
) -> MetricsReport:
    """run 目录 → 完整指标报告（T05 UI 消费的单次调用入口）。

    读取：
        - run_config.yaml（可复现锚点；缺失 → 内置默认 + 告警）；
        - cache/detections.jsonl（逐帧观测，含缓存态 vanish_class）；
        - corrections.jsonl（人工修正留痕 → manual_count 注入 → 双轨）；
        - meta.json（可选，RunMeta 留档）。

    Returns:
        MetricsReport（to_dict() JSON 可序列化，含全部 A/B1/B2/D 指标、
        13 个 Q_*、capability 降级结果、人工修正双轨并列）。
    """
    from src.pipeline.orchestrator import Orchestrator  # 局部导入（避免环）

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")

    missing_cfg = False
    config: RunConfig
    cfg_path = run_dir / "run_config.yaml"
    if cfg_path.exists():
        config = RunConfig.from_yaml(cfg_path)
    else:
        config = RunConfig()
        missing_cfg = True

    observations = Orchestrator._read_cache(run_dir / "cache" / "detections.jsonl")
    if not observations:
        raise ValueError(
            f"cache/detections.jsonl 无可读观测帧: {run_dir}"
        )
    corrections = load_corrections(run_dir)
    if corrections:
        observations = apply_manual_counts(observations, corrections, inplace=False)
    if meta is None:
        meta = _meta_from_json(run_dir / "meta.json")
    if baseline is None:
        baseline = build_baseline_stats(observations, roi)

    agg = MetricsAggregator(config=config)
    report = agg.aggregate(
        observations,
        meta=meta,
        roi=roi,
        baseline=baseline,
        tracks=tracks,
        link_result=None,  # 缓存态：A14 走帧级 vanish_class 回退
        px_per_mm=config.px_per_mm_ref,
        outdoor=outdoor,
        run_id=run_dir.name,
    )
    if missing_cfg:
        report.warnings.append(
            "run_config.yaml 缺失：使用内置默认配置"
            "（跨 run 比较将被 compare 拦截）"
        )
    if corrections:
        report.notes.append(
            f"从 corrections.jsonl 注入 {len(corrections)} 条人工修正（双轨重算）"
        )
    return report


# ----------------------------------------------------------------------
# quality_signals.csv 行（统一委托 quality.to_quality_rows）
# ----------------------------------------------------------------------
def quality_rows(
    signals: dict[str, Any], thresholds: Thresholds | None = None
) -> list[dict[str, Any]]:
    """quality_signals.csv 行（signal/value/threshold/passed，含 Q_glare 行）。

    T04 收口：本函数改为 quality.to_quality_rows 的别名（此前两份实现
    口径不同：本模块的旧实现漏了 Q_glare 行与 Q_interf 的方向判定）。
    """
    return to_quality_rows(signals, thresholds)


# ----------------------------------------------------------------------
# 落盘（五件套 + summary.json）
# ----------------------------------------------------------------------
def write_run_outputs(report: MetricsReport, run_dir: str | Path) -> list[Path]:
    """落盘 metrics_summary.csv / quality_signals.csv / flag_glossary.csv /
    metrics_timeseries.csv / cache/quality.json / summary.json。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ---- metrics_summary.csv（17 列冻结）----
    p = run_dir / "metrics_summary.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SUMMARY_COLUMNS))
        w.writeheader()
        for row in report.summary_rows():
            w.writerow(row)
    written.append(p)

    # ---- quality_signals.csv（含 threshold / passed）----
    p = run_dir / "quality_signals.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["signal", "value", "threshold", "passed"])
        w.writeheader()
        for row in report.quality_table:
            w.writerow(
                {
                    "signal": row["signal"],
                    "value": _fmt_value(row["value"]),
                    "threshold": _fmt_value(row["threshold"]),
                    "passed": _bool_or_empty(row["passed"]),
                }
            )
    written.append(p)

    # ---- flag_glossary.csv（术语表与 summary 并列同目录）----
    p = run_dir / "flag_glossary.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["flag_token", "中文名", "含义", "建议动作"])
        w.writeheader()
        for token, (zh, meaning, action) in FLAG_GLOSSARY.items():
            w.writerow(
                {
                    "flag_token": token,
                    "中文名": zh,
                    "含义": meaning,
                    "建议动作": action,
                }
            )
    written.append(p)

    # ---- metrics_timeseries.csv（原生不规则时间戳 + dt_s + low_conf 对）----
    p = run_dir / "metrics_timeseries.csv"
    cols = _timeseries_columns(report.timeseries)
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in _timeseries_rows(report.timeseries):
            w.writerow(row)
    written.append(p)

    # ---- cache/quality.json ----
    cache = run_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / "quality.json"
    p.write_text(
        json.dumps(_jsonify(report.quality_signals), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    written.append(p)

    # ---- summary.json ----
    p = run_dir / "summary.json"
    p.write_text(
        json.dumps(_jsonify(report.to_dict()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    written.append(p)
    return written


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------
def _metric_order(metrics: dict[str, MetricValue]) -> list[str]:
    """输出顺序：A → B1 → B2 → D（组内按 METRIC_NAMES_ZH 声明序，
    _manual 口径紧随对应原始指标）。"""
    order = [m for m in METRIC_NAMES_ZH if m in metrics]
    extra = [m for m in metrics if m not in order]
    # _manual 指标按其原始 ID 在声明序中的位置插入
    extra_sorted: list[str] = []
    for m in METRIC_NAMES_ZH:
        key = f"{m}{MANUAL_SUFFIX}"
        if key in metrics:
            extra_sorted.append(key)
    leftover = sorted(m for m in extra if m not in extra_sorted)
    return order + extra_sorted + leftover


def _group_of(metric_id: str) -> str:
    if metric_id.startswith("B1"):
        return "B1"
    if metric_id.startswith("B2"):
        return "B2"
    if metric_id.startswith("A"):
        return "A"
    if metric_id.startswith("D"):
        return "D"
    return "?"


def _fmt_value(v: Any) -> Any:
    """CSV 值格式化：None/NaN → ''（空字符串，绝不为 0）。"""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{float(v):.6g}"
    return v


def _num_or_empty(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return ""
    return _fmt_value(v)


def _bool_or_empty(v: Any) -> Any:
    if v is None:
        return ""
    return "true" if bool(v) else "false"


def _method_of(mv: MetricValue) -> str:
    parts: list[str] = []
    q = mv.quality
    if q.get("method"):
        parts.append(str(q["method"]))
    if q.get("model"):
        parts.append(f"fit={q['model']}")
    if q.get("integral_method"):
        parts.append(f"integral={q['integral_method']}")
    return ";".join(parts)


def _with_flags(mv: MetricValue, extra: tuple[str, ...]) -> MetricValue:
    """返回追加了 flags 的拷贝（MetricValue 不可变）。"""
    merged = tuple(dict.fromkeys(mv.flags + extra))
    return MetricValue(
        metric_id=mv.metric_id,
        value=mv.value,
        unit=mv.unit,
        status=mv.status,
        reason=mv.reason,
        flags=merged,
        quality=dict(mv.quality),
        unit_scale=mv.unit_scale,
    )


def _timeseries_columns(series_list: list[TimeSeries]) -> list[str]:
    cols = ["t_seconds", "dt_s", "in_baseline"]
    for ts in series_list:
        cols.append(ts.metric_id)
        cols.append(f"{ts.metric_id}_low_conf")
    return cols


def _timeseries_rows(series_list: list[TimeSeries]) -> list[dict[str, Any]]:
    """宽表行：各列按时间戳对齐（缺失点为空字符串，绝不补 0）。"""
    all_t: dict[float, None] = {}
    for ts in series_list:
        for t in ts.t:
            all_t[float(t)] = None
    t_list = sorted(all_t)
    rows: list[dict[str, Any]] = []
    prev_t: float | None = None
    for t in t_list:
        row: dict[str, Any] = {
            "t_seconds": f"{t:.6g}",
            "dt_s": "" if prev_t is None else f"{t - prev_t:.6g}",
            "in_baseline": "true" if t < 0 else "false",
        }
        for ts in series_list:
            hit = np.where(np.isclose(ts.t, t))[0]
            if hit.size:
                i = int(hit[0])
                v = ts.values[i]
                row[ts.metric_id] = "" if np.isnan(v) else f"{float(v):.6g}"
                row[f"{ts.metric_id}_low_conf"] = "false"
            else:
                row[ts.metric_id] = ""
                row[f"{ts.metric_id}_low_conf"] = ""
        rows.append(row)
        prev_t = t
    return rows


def _jsonify(obj: Any) -> Any:
    """递归转 JSON 安全类型（numpy 标量 → Python 标量）。"""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ----------------------------------------------------------------------
# 命名兼容别名（classDiagram 用 Aggregator 命名）
# ----------------------------------------------------------------------
Aggregator = MetricsAggregator

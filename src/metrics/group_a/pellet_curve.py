"""A2–A4 · 颗粒曲线 N_p(t) / C(t) / P(t)（T04）。

职责（docs/04 §3.1 A2/A3/A4）：
    - 从 FrameObservation 提取剩余颗粒数序列（原生不规则时间戳，
      唯一可信源；人工修正 manual_count 优先消费）；
    - 平滑（移动中位数，默认 5 采样点）与单调化（累积最小值，
      防回补帧产生负速率）；
    - N_p > pellet_saturation（计数饱和）的帧标 low_conf，不参与拟合。

纪律：
    - 序列不可用 = available=False + reason（无检测数据），绝不用 0 冒充；
    - 可用轨的实测 0 颗是有效零值观测（≠ None）。

任务编号：T04。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.core.config import Thresholds
from src.core.frame_context import FrameObservation
from src.core.roi import ROI

__all__ = [
    "PelletSeries",
    "TimeSeries",
    "extract_pellet_series",
    "moving_median",
    "monotonize",
    "smooth_series",
    "rebound_fraction",
]

@dataclass
class TimeSeries:
    """通用指标时序（A2/A3/A5/D1/B1 曲线的统一载体，非 MetricValue 标量轨）。

    原生不规则时间戳是唯一可信源（契约 §3 采样约定：基线 2s / 早期 1s /
    尾段 10s 的非对称采样），绝不重采样到等间隔。

    Attributes:
        metric_id: 曲线编号，如 "A2_N_p"、"FA"、"M_diff"。
        t: (K,) 时间戳（秒，相对 t0，基线期为负）。
        values: (K,) 数值（可含 NaN = 该点无有效观测）。
        unit: 数值单位。
        note: 口径说明（归一化方式 / 面积口径 / fallback 等）。
    """

    metric_id: str
    t: np.ndarray
    values: np.ndarray
    unit: str = "-"
    note: str | None = None

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        if self.t.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"TimeSeries {self.metric_id!r} 的 t({self.t.shape[0]}) 与 "
                f"values({self.values.shape[0]}) 长度不一致"
            )

    def n_points(self) -> int:
        """总采样点数（含 NaN 点）。"""
        return int(self.t.shape[0])

    def valid_t_values(self) -> tuple[np.ndarray, np.ndarray]:
        """(t, values) 有效点（剔除 NaN）。"""
        ok = ~np.isnan(self.values)
        return self.t[ok], self.values[ok]

    def to_rows(self) -> list[dict]:
        """导出为行列表（timeseries.csv 消费；NaN → None 不冒充 0）。"""
        return [
            {
                "metric_id": self.metric_id,
                "t_s": float(self.t[i]),
                "value": None if np.isnan(self.values[i]) else float(self.values[i]),
            }
            for i in range(self.t.shape[0])
        ]

    def to_dict(self) -> dict:
        """JSON 可序列化结构（summary.json / T05 UI 直接消费；NaN → None）。"""
        return {
            "metric_id": self.metric_id,
            "unit": self.unit,
            "note": self.note,
            "t": [float(x) for x in self.t],
            "values": [
                None if np.isnan(v) else float(v) for v in self.values
            ],
        }


@dataclass
class PelletSeries:
    """剩余颗粒数序列（A2 输出，原生不规则采样）。

    Attributes:
        t: (K,) 相对 t0 的时间戳（秒，基线期为负）。
        n: (K,) 剩余颗粒数（颗）。人工修正帧为修正值。
        conf_med: (K,) 帧内检测置信度中位数（无检测框的帧为 nan）。
        low_conf: (K,) bool，计数饱和（N > pellet_saturation）等不可信帧。
        frame_idx: (K,) 源帧号。
        source: 主来源轨（'det' / 'area' / 'manual' / 'mixed'）。
        available: 序列是否可用（无任何检测数据 → False）。
        reason: available=False 时的原因。
    """

    t: np.ndarray
    n: np.ndarray
    conf_med: np.ndarray
    low_conf: np.ndarray
    frame_idx: np.ndarray
    source: str = "mixed"
    available: bool = True
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def trial_mask(self) -> np.ndarray:
        """试验期掩码（t ≥ 0，含 t0 帧）。"""
        return self.t >= 0.0

    def valid_mask(self) -> np.ndarray:
        """参与拟合/统计的有效点（试验期且非 low_conf）。"""
        return self.trial_mask() & (~self.low_conf)

    def n_frames_used(self) -> int:
        """有效观测点数（n_frames_used 列来源）。"""
        return int(np.count_nonzero(self.valid_mask())) if self.available else 0


# ----------------------------------------------------------------------
# 提取与变换
# ----------------------------------------------------------------------
def extract_pellet_series(
    observations: Sequence[FrameObservation],
    roi: ROI | None = None,
    thresholds: Thresholds | None = None,
    use_manual: bool = True,
) -> PelletSeries:
    """从帧观测提取 A2 序列（人工修正 manual_count 优先）。

    Args:
        observations: 采样帧观测（pellets 可为 None = 该帧无检测数据）。
        roi: 提供 pellet_zone 时按质心过滤（计数区口径）。
        thresholds: pellet_saturation 饱和阈值。
        use_manual: True = 人工修正帧取 extra['manual_count']（修正轨）；
            False = 忽略修正、始终用检测计数（原始轨，*_corrected 双轨
            的对照基准——重算不覆盖原值的前提）。
    """
    th = thresholds if thresholds is not None else Thresholds()
    rows: list[tuple[float, float, float, bool, int, str]] = []
    has_any_detection = False
    for obs in observations:
        pel = obs.pellets
        if pel is None:
            # 该帧无检测数据：跳过（不造 0 冒充观测）
            continue
        has_any_detection = True
        src = str(obs.extra.get("pellet_source") or "det")
        manual = obs.extra.get("manual_count") if use_manual else None
        if manual is not None:
            n = float(manual)
            src = "manual"
            conf_med = float(np.median(pel.conf)) if pel.n_det() > 0 else np.nan
        else:
            if pel.n_det() == 0:
                n = 0.0  # 有效零值观测
                conf_med = np.nan
            else:
                centroids = pel.centroids()
                if roi is not None and roi.pellet_zone is not None:
                    keep = np.array(
                        [roi.contains((c[0], c[1]), "pellet_zone") for c in centroids],
                        dtype=bool,
                    )
                else:
                    keep = np.ones(pel.n_det(), dtype=bool)
                n = float(np.count_nonzero(keep))
                conf_med = (
                    float(np.median(pel.conf[keep])) if np.any(keep) else np.nan
                )
        low = bool(n > th.pellet_saturation)
        rows.append((obs.t_s, n, conf_med, low, obs.frame_idx, src))

    if not has_any_detection:
        return PelletSeries(
            t=np.zeros(0), n=np.zeros(0), conf_med=np.zeros(0),
            low_conf=np.zeros(0, dtype=bool), frame_idx=np.zeros(0, dtype=int),
            available=False,
            reason="无颗粒检测数据（未配置检测器或检测轨不可用）",
        )

    rows.sort(key=lambda r: r[0])
    arr = np.array([(r[0], r[1], r[2], r[4]) for r in rows], dtype=float)
    srcs = {r[5] for r in rows}
    source = srcs.pop() if len(srcs) == 1 else "mixed"
    return PelletSeries(
        t=arr[:, 0],
        n=arr[:, 1],
        conf_med=arr[:, 2],
        low_conf=np.array([r[3] for r in rows], dtype=bool),
        frame_idx=arr[:, 3].astype(int),
        source=source,
        available=True,
        warnings=[],
    )


def moving_median(x: np.ndarray, w: int = 3) -> np.ndarray:
    """滑动中位数（边界用边缘值镜像填充；w=1 原样返回）。

    与滑动均值不同，中位数对单帧漏检尖峰鲁棒（A1 峰值估计的关键）。
    """
    x = np.asarray(x, dtype=float)
    w = int(w)
    if w <= 1 or x.size == 0:
        return x.copy()
    pad = w // 2
    padded = np.concatenate(
        [np.full(pad, x[0] if x.size else 0.0), x, np.full(pad, x[-1] if x.size else 0.0)]
    )
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = np.median(padded[i : i + w])
    return out


def _smooth_segments(t: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    """切出「有效且同相位」的连续段（返回索引数组列表）。

    为什么必须分段（docs/04 §3.1 A2 + §1 时间基准约定）：
        t0 处存在**真实阶跃**——基线期（t < 0，投喂前）颗粒数恒为 0，
        试验期起始为 N₀。若对整条时间轴一把平滑，投喂前的零值会被卷进
        早期试验窗的滑动窗口，**系统性压低 N₀、T50 与 v_max 的估计**
        ——而早期窗口恰恰是 A1/A6/A8 主结论所在。

    分段条件：有效点（~low_conf）且相邻索引在 **同一相位**（同属基线期
    或同属试验期）且索引连续。
    """
    t = np.asarray(t, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    segments: list[list[int]] = []
    current: list[int] = []
    prev_idx: int | None = None
    prev_phase: bool | None = None
    for i in range(t.shape[0]):
        if not bool(valid[i]):
            prev_idx = None
            prev_phase = None
            continue
        phase = bool(t[i] >= 0.0)
        contiguous = (
            prev_idx is not None
            and i == prev_idx + 1
            and prev_phase is not None
            and phase == prev_phase
        )
        if current and not contiguous:
            segments.append(current)
            current = []
        current.append(i)
        prev_idx, prev_phase = i, phase
    if current:
        segments.append(current)
    return [np.asarray(seg, dtype=int) for seg in segments]


def smooth_series(series: PelletSeries, w: int | None = None) -> PelletSeries:
    """对 n 做移动中位数平滑（A2 契约：平滑窗口默认 5 采样点）。

    low_conf 帧不参与平滑（其原值保留并继续标记不可信）。
    **分段平滑**：基线期与试验期之间不跨 t0 混合（见 _smooth_segments）。
    """
    w_use = 5 if w is None else int(w)
    n = series.n.copy()
    if series.available and n.size > 0:
        for seg in _smooth_segments(series.t, ~series.low_conf):
            if seg.size:
                n[seg] = moving_median(n[seg], w_use)
    return PelletSeries(
        t=series.t, n=n, conf_med=series.conf_med, low_conf=series.low_conf,
        frame_idx=series.frame_idx, source=series.source,
        available=series.available, reason=series.reason,
        warnings=list(series.warnings),
    )


def monotonize(n: np.ndarray) -> np.ndarray:
    """强制非增（累积最小值；A5 前置，防回补帧产生负速率）。"""
    return np.minimum.accumulate(np.asarray(n, dtype=float))


def rebound_fraction(n: np.ndarray) -> float:
    """非单调上升段占比（n[i+1] > n[i] 的步数占比）——counting_unstable 判据。"""
    n = np.asarray(n, dtype=float)
    if n.size < 2:
        return 0.0
    return float(np.count_nonzero(np.diff(n) > 0)) / float(n.size - 1)

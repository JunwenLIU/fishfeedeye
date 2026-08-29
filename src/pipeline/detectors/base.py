"""PelletDetector 抽象接口 + 颗粒过滤 + 双轨偏差（T03）。

职责（docs/06 §6 T03 内联约定 + classDiagram）：
    - 统一接口 detect(frame) -> PelletDetections，三种实现可插拔：
        YoloPelletDetector（L1 微调权重主通道）
        YoloEDetector（开放词汇冷启动，无需训练）
        AreaIntegralCounter（面积积分轨，密集粘连时段补充）
    - 颗粒过滤（test_filters.py 消费）：尺寸 / 置信度 / ROI 区域过滤；
    - 双轨偏差：|n_det − n_area| / max(n_det, n_area) > 20% → 质量信号
      Q_dualtrack_gap（供告警）。

纪律：
    - 各实现 detect() 返回同构 PelletDetections（xyxy/conf 行数一致，
      centroid/area 可派生），保证接口一致性可测；
    - 轨不可用（如 A_single 未标定、模型缺失）时通过 available() /
      unavailable_reason 显式暴露，调用方（orchestrator）负责在
      FrameObservation.extra 里打状态标记——**绝不返回 0 冒充观测值**。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.frame_context import PelletDetections
from src.core.roi import ROI

__all__ = [
    "DetectStats",
    "PelletDetector",
    "filter_pellets",
    "compute_dualtrack_gap",
    "DUALTRACK_GAP_WARN",
]


# 双轨偏差告警阈值（docs/06 T03：>20% 记录 Q_dualtrack_gap）
DUALTRACK_GAP_WARN: float = 0.20


@dataclass
class DetectStats:
    """单帧检测统计（写入 FrameObservation.extra，含来源轨与可用状态）。"""

    n: int                                  # 本轨颗粒计数估计
    source: str                             # 'det' | 'area'
    available: bool                         # 本轨是否可用
    reason: str | None = None               # 不可用原因（必填 when not available）
    fg_area_px: float | None = None         # 面积积分轨前景面积
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": self.n,
            "source": self.source,
            "available": self.available,
            "reason": self.reason,
        }
        if self.fg_area_px is not None:
            out["fg_area_px"] = self.fg_area_px
        out.update(self.extra)
        return out


class PelletDetector(ABC):
    """颗粒检测器抽象基类（可插拔策略，classDiagram PelletDetector）。"""

    def __init__(self) -> None:
        self.last_stats: DetectStats | None = None

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        t_s: float | None = None,
        roi: ROI | None = None,
    ) -> PelletDetections:
        """检测一帧 → 同构 PelletDetections。

        Args:
            frame: BGR 帧。
            t_s: 该帧相对 t0 的时间戳（面积积分轨按时间取 A_single(t)）。
            roi: 区域定义（检测轨做区域过滤；面积积分轨限定计数区）。
        """

    @abstractmethod
    def name(self) -> str:
        """检测器名称（缓存/报告留档）。"""

    @abstractmethod
    def available(self) -> bool:
        """本轨当前是否可用（权重/标定齐备）。"""

    def unavailable_reason(self) -> str | None:
        """不可用原因（可用时 None）。"""
        return None


# ----------------------------------------------------------------------
# 颗粒过滤（三轨共用；tests/test_filters.py 消费）
# ----------------------------------------------------------------------
def filter_pellets(
    detections: PelletDetections,
    min_area_px: float | None = None,
    max_area_px: float | None = None,
    min_conf: float | None = None,
    roi: ROI | None = None,
    zone: str = "pellet_zone",
) -> PelletDetections:
    """按尺寸 / 置信度 / ROI 区域过滤颗粒检测框。

    Args:
        detections: 输入检测。
        min_area_px / max_area_px: 框面积（w×h 像素²）上下限；None = 不过滤。
        min_conf: 置信度下限。
        roi: 区域定义；zone 指定过滤区域名（'pellet_zone' / 'arena' 等），
            质心在区域外（或在排除区）的框剔除。roi=None = 不过滤。
    """
    n = detections.n_det()
    if n == 0:
        return detections
    keep = np.ones(n, dtype=bool)
    boxes = detections.xyxy
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if min_area_px is not None:
        keep &= area >= float(min_area_px)
    if max_area_px is not None:
        keep &= area <= float(max_area_px)
    if min_conf is not None:
        keep &= detections.conf >= float(min_conf)
    if roi is not None:
        cents = detections.centroids()
        from src.core.roi import point_in_polygon  # 局部导入避免顶层循环

        poly = getattr(roi, zone, None)
        for i in range(n):
            if not keep[i]:
                continue
            if poly is not None and not point_in_polygon(cents[i], poly):
                keep[i] = False
            if keep[i] and roi.contains(cents[i], "exclude"):
                keep[i] = False
    if bool(keep.all()):
        return detections
    return PelletDetections(
        xyxy=boxes[keep],
        conf=detections.conf[keep],
        centroid=None if detections.centroid is None else detections.centroid[keep],
        area_px=None if detections.area_px is None else detections.area_px[keep],
        track_id=None if detections.track_id is None else detections.track_id[keep],
        vanish_class=None if detections.vanish_class is None
        else [detections.vanish_class[i] for i in np.where(keep)[0]],
    )


def compute_dualtrack_gap(n_det: int | None, n_area: int | None) -> float | None:
    """双轨计数相对偏差 = |a−b| / max(a, b)。

    任一轨缺失或双轨皆为 0 → None（无偏差可言，不造数）。
    >DUALTRACK_GAP_WARN 时调用方记录质量信号 Q_dualtrack_gap。
    """
    if n_det is None or n_area is None:
        return None
    m = max(n_det, n_area)
    if m <= 0:
        return None
    return float(abs(n_det - n_area) / m)

"""面积积分计数轨（T03）：N_area = A_fg / A_single(t)。

职责（docs/06 §6 T03 内联约定 + docs/06 §1 挑战 #3 对策）：
    密集粘连时段的补充计数轨：
      前景 = HSV 明度/饱和度阈值分割（浮性料为亮色颗粒）+ 形态学开闭
      + 排除区掩膜 + 计数区（pellet_zone / arena）掩膜；
      N_area = 前景面积 / A_single(t)（A_single 按时间校正：浮料吸水膨胀）。

    逐连通域估计颗粒数 k = round(area / A_single(t))（粘连团块拆分），
    输出同构 PelletDetections（每个估计颗粒一个子框）。

纪律：
    - A_single 未标定 → 本轨 available=False，reason='A_single 未标定'，
      detect 返回空检测 + stats.available=False：调用方必须在
      FrameObservation.extra 打 unavailable 状态，**绝不硬编码猜测值**，
      也绝不把空检测当作"0 颗观测"；
    - 分割逻辑与标定脚本（pellet_dynamics.py）共用同一实现，保证
      A_single 的口径与 N_area 的分子口径一致。
"""
from __future__ import annotations

import numpy as np

from src.core.frame_context import PelletDetections
from src.core.roi import ROI
from src.pipeline.detectors.base import DetectStats, PelletDetector
from src.pipeline.pellet_dynamics import ASingleCurve, SegmentationParams, extract_blobs, segment_pellets

__all__ = ["AreaIntegralCounter"]


class AreaIntegralCounter(PelletDetector):
    """面积积分计数轨（classDiagram AreaIntegralCounter）。"""

    def __init__(
        self,
        a_single: ASingleCurve | None = None,
        params: SegmentationParams | None = None,
        isolate_conf: float = 0.95,
        merged_conf: float = 0.55,
    ) -> None:
        """
        Args:
            a_single: A_single(t) 标定曲线（scripts/00 产物）。None = 未标定。
            params: 前景分割参数。
            isolate_conf: 孤立颗粒（k=1 且面积与 A_single 吻合）置信度。
            merged_conf: 粘连团块拆分计数的置信度。
        """
        super().__init__()
        self._a_single = a_single
        self.params = params or SegmentationParams()
        self.isolate_conf = float(isolate_conf)
        self.merged_conf = float(merged_conf)

    # ------------------------------------------------------------------
    def set_A_single(self, curve: ASingleCurve | None) -> None:
        """注入/替换 A_single(t) 标定曲线（classDiagram set_A_single）。"""
        self._a_single = curve

    def name(self) -> str:
        return "area_integral"

    def available(self) -> bool:
        return self._a_single is not None and self._a_single.available

    def unavailable_reason(self) -> str | None:
        if self.available():
            return None
        return "A_single 未标定（需无鱼纯饲料视频经 scripts/00_calibrate_pellet_dynamics.py 标定；不得硬编码猜测值）"

    # ------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        t_s: float | None = None,
        roi: ROI | None = None,
    ) -> PelletDetections:
        """面积积分计数一帧。

        Returns:
            每个估计颗粒一个框的 PelletDetections；本轨不可用时返回空
            检测，last_stats.available=False（调用方据此打 unavailable 状态，
            不得当 0 颗观测）。
        """
        if not self.available():
            self.last_stats = DetectStats(
                n=0, source="area", available=False,
                reason=self.unavailable_reason(),
            )
            return PelletDetections.empty()

        mask = segment_pellets(frame, self.params)
        if roi is not None:
            mask = mask & roi.mask(mask.shape, zone="pellet_zone", subtract_exclusion=True) \
                if roi.pellet_zone is not None \
                else mask & roi.mask(mask.shape, zone="arena", subtract_exclusion=True)
        blobs = extract_blobs(mask, self.params.min_blob_area_px)
        a_single = self._a_single.value_at(t_s if t_s is not None else 0.0)
        if a_single is None or a_single <= 0:
            self.last_stats = DetectStats(
                n=0, source="area", available=False,
                reason=f"A_single(t) 在 t={t_s}s 无有效值（标定曲线缺测）",
                fg_area_px=float(mask.sum()),
            )
            return PelletDetections.empty()

        boxes: list[list[float]] = []
        confs: list[float] = []
        areas: list[float] = []
        n_total = 0
        fg_area = 0.0
        for b in blobs:
            fg_area += b.area_px
            k = max(1, int(round(b.area_px / a_single)))
            n_total += k
            # 置信度：k=1 且面积与 A_single 吻合 → 高；粘连拆分 → 低
            if k == 1 and abs(b.area_px - a_single) <= 0.35 * a_single:
                c = self.isolate_conf
            else:
                c = self.merged_conf
            sub_boxes = _subdivide_bbox(b.bbox, k)
            for sb in sub_boxes:
                boxes.append(sb)
                confs.append(c)
            areas.extend([b.area_px / k] * k)

        if not boxes:
            self.last_stats = DetectStats(
                n=0, source="area", available=True, fg_area_px=fg_area
            )
            return PelletDetections.empty()

        self.last_stats = DetectStats(
            n=n_total, source="area", available=True,
            fg_area_px=fg_area,
            extra={"a_single_px": a_single, "n_blobs": len(blobs)},
        )
        return PelletDetections(
            xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
            conf=np.asarray(confs, dtype=float),
            area_px=np.asarray(areas, dtype=float),
        )


def _subdivide_bbox(bbox: tuple[float, float, float, float], k: int) -> list[list[float]]:
    """把连通域包围盒划分为 k 个子格（近方形网格），返回子框列表。

    粘连团块内无法定位单颗位置，子框仅为接口同构的伪框（消费方只用
    计数与质心分布近似）；k=1 时返回原框。
    """
    x0, y0, x1, y1 = bbox
    if k <= 1:
        return [[x0, y0, x1, y1]]
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    w = (x1 - x0) / cols
    h = (y1 - y0) / rows
    out: list[list[float]] = []
    n_done = 0
    for r in range(rows):
        for c in range(cols):
            if n_done >= k:
                break
            out.append([x0 + c * w, y0 + r * h, x0 + (c + 1) * w, y0 + (r + 1) * h])
            n_done += 1
    return out

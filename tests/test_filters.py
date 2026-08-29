"""颗粒过滤与双轨偏差测试（T03，docs/06 §6 T03 文件清单 test_filters.py）。

覆盖：
    - filter_pellets：面积上下限 / 置信度 / ROI 区域过滤（质心在区外剔除、
      排除区剔除）、vanish_class 保序、空输入恒等；
    - compute_dualtrack_gap：None 传播、双零 → None、阈值口径
      |a−b| / max(a,b)。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.frame_context import PelletDetections
from src.core.roi import ROI
from src.pipeline.detectors.base import (
    DUALTRACK_GAP_WARN,
    compute_dualtrack_gap,
    filter_pellets,
)


def _rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float,
    )


def _dets(boxes: list[tuple[float, float, float, float]],
          confs: list[float] | None = None) -> PelletDetections:
    xyxy = np.asarray(boxes, dtype=float).reshape(-1, 4)
    conf = np.ones(len(boxes), dtype=float) if confs is None else np.asarray(confs)
    return PelletDetections(xyxy=xyxy, conf=conf)


class Test尺寸过滤:

    def test_面积下限剔除小框(self) -> None:
        d = _dets([(0, 0, 4, 4), (10, 10, 20, 20)])  # 16px² 与 100px²
        out = filter_pellets(d, min_area_px=50.0)
        assert out.n_det() == 1
        assert out.xyxy[0].tolist() == [10, 10, 20, 20]

    def test_面积上限剔除大框(self) -> None:
        d = _dets([(0, 0, 4, 4), (10, 10, 30, 30)])
        out = filter_pellets(d, max_area_px=100.0)
        assert out.n_det() == 1

    def test_置信度下限(self) -> None:
        d = _dets([(0, 0, 10, 10), (20, 0, 30, 10)], confs=[0.9, 0.1])
        out = filter_pellets(d, min_conf=0.25)
        assert out.n_det() == 1
        assert out.conf[0] == pytest.approx(0.9)


class Test区域过滤:

    def test_质心在区外剔除(self) -> None:
        roi = ROI(arena=_rect(0, 0, 100, 100), pellet_zone=_rect(50, 0, 100, 100))
        d = _dets([(10, 40, 20, 60), (70, 40, 80, 60)])  # 质心 (15,50) 与 (75,50)
        out = filter_pellets(d, roi=roi, zone="pellet_zone")
        assert out.n_det() == 1
        assert out.xyxy[0].tolist() == [70, 40, 80, 60]

    def test_排除区剔除(self) -> None:
        roi = ROI(
            arena=_rect(0, 0, 100, 100),
            exclude_zones=[_rect(40, 40, 60, 60)],
        )
        d = _dets([(45, 45, 55, 55), (10, 10, 20, 20)])
        out = filter_pellets(d, roi=roi, zone="arena")
        assert out.n_det() == 1
        assert out.xyxy[0].tolist() == [10, 10, 20, 20]

    def test_roi为None_不过滤(self) -> None:
        d = _dets([(0, 0, 5, 5)])
        out = filter_pellets(d, roi=None)
        assert out.n_det() == 1


class Test过滤保留字段:

    def test_vanish_class与track保序(self) -> None:
        d = PelletDetections(
            xyxy=np.asarray([(0, 0, 10, 10), (20, 0, 35, 10), (40, 0, 60, 10)]),
            conf=np.ones(3),
            track_id=np.asarray([1, 2, 3]),
            vanish_class=["eaten", "drifted", "unknown"],
        )
        out = filter_pellets(d, min_conf=0.99)  # 全保留
        assert out.n_det() == 3
        out = filter_pellets(d, min_area_px=150.0)  # 留 ≥150px²：后两框
        assert out.n_det() == 2
        assert out.vanish_class == ["drifted", "unknown"]
        assert out.track_id.tolist() == [2, 3]

    def test_空输入恒等(self) -> None:
        d = PelletDetections.empty()
        out = filter_pellets(d, min_area_px=1.0, min_conf=0.5)
        assert out.n_det() == 0


class Test双轨偏差:

    def test_口径_绝对差除以较大者(self) -> None:
        assert compute_dualtrack_gap(10, 5) == pytest.approx(0.5)
        assert compute_dualtrack_gap(5, 10) == pytest.approx(0.5)
        assert compute_dualtrack_gap(10, 8) == pytest.approx(DUALTRACK_GAP_WARN)

    def test_单轨缺失_None传播(self) -> None:
        assert compute_dualtrack_gap(None, 10) is None
        assert compute_dualtrack_gap(10, None) is None

    def test_双零_None不造数(self) -> None:
        assert compute_dualtrack_gap(0, 0) is None

    def test_相等_零偏差(self) -> None:
        assert compute_dualtrack_gap(7, 7) == pytest.approx(0.0)

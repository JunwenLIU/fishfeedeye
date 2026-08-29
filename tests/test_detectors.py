"""检测器测试（T03 验收：接口一致性 / 面积积分精度 / 打桩检测轨 / 轨不可用）。

策略（docs/06 §6 T03：打桩测逻辑，不依赖真实模型权重）：
    - 检测轨（YoloPelletDetector / YoloEDetector）注入 predict_fn 桩 =
      「分割 + 连通域」的先知函数，测切片/NMS/过滤逻辑；
    - 面积积分轨用真实 HSV 分割（无模型依赖）；
    - 验收口径：面积轨 100 颗密集粘连误差 ≤10%；检测轨（桩）误差 ≤20%；
    - 轨不可用（A_single 未标定 / 权重缺失）：available=False + reason，
      绝不把空检测当 0 颗观测。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.frame_context import PelletDetections
from src.core.roi import ROI
from src.pipeline.detectors.area_integral import AreaIntegralCounter
from src.pipeline.detectors.base import DetectStats, compute_dualtrack_gap
from src.pipeline.detectors.yolo_detector import (
    TiledPredictor,
    YoloPelletDetector,
    nms_xyxy,
)
from src.pipeline.detectors.yoloe_detector import YoloEDetector
from src.pipeline.pellet_dynamics import (
    ASingleCurve,
    SegmentationParams,
    extract_blobs,
    segment_pellets,
)
from tests.fixtures.synthetic import PelletSpec, render_water_frame

SIZE = (320, 240)


# ----------------------------------------------------------------------
# 桩：先知 predict_fn（分割 + 连通域 → 框 + 置信度）
# ----------------------------------------------------------------------
def _oracle_predict(tile_bgr: np.ndarray):
    params = SegmentationParams()
    blobs = extract_blobs(segment_pellets(tile_bgr, params), params.min_blob_area_px)
    return [((b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]), 0.9) for b in blobs]


def _isolated_a_single(radius: float) -> ASingleCurve:
    """用孤立单颗颗粒实测 A_single（与面积轨同一分割口径）。"""
    frame = render_water_frame(
        SIZE, 0.0, [PelletSpec(start_xy=(160.0, 120.0), radius=radius)], seed=21,
    )
    blobs = extract_blobs(
        segment_pellets(frame, SegmentationParams()),
        SegmentationParams().min_blob_area_px,
    )
    assert len(blobs) == 1
    return ASingleCurve(t_s=[0.0], area_px=[blobs[0].area_px])


def _isolated_pellets(n: int, radius: float, spacing: float) -> list[PelletSpec]:
    """n 颗等间距孤立颗粒（网格，左上角 (30, 30)）。"""
    cols = int(math.ceil(math.sqrt(n)))
    out: list[PelletSpec] = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append(
            PelletSpec(
                start_xy=(30.0 + c * spacing, 30.0 + r * spacing), radius=radius,
            )
        )
    return out


# ----------------------------------------------------------------------
# TiledPredictor（SAHI 同构切片推理）
# ----------------------------------------------------------------------
class Test切片推理:

    def test_切片原点全覆盖且带重叠(self) -> None:
        tp = TiledPredictor(_oracle_predict, tile_size=100, overlap=0.2)
        origins = tp.tile_origins(320, 240)
        xs = sorted({o[0] for o in origins})
        ys = sorted({o[1] for o in origins})
        # 覆盖：并集 [x, x+100) ⊇ [0, 320)；末点对齐保证右/下边界覆盖
        assert xs[0] == 0 and xs[-1] + 100 == 320
        assert ys[0] == 0 and ys[-1] + 100 == 240
        # 常规步长 = tile × (1 - overlap)；末段允许更密（对齐边界）
        for a, b in zip(xs[:-1], xs[1:]):
            assert 0 < b - a <= tp.stride
        for a, b in zip(ys[:-1], ys[1:]):
            assert 0 < b - a <= tp.stride

    def test_跨切片重复检测_NMS去重(self) -> None:
        # 单颗颗粒位于切片边界（x=160 落在多个 tile 内）→ 去重后恰好 1 框
        frame = render_water_frame(
            SIZE, 0.0, [PelletSpec(start_xy=(160.0, 120.0), radius=6.0)], seed=22,
        )
        tp = TiledPredictor(_oracle_predict, tile_size=100, overlap=0.2, iou_nms=0.3)
        dets = tp.predict(frame)
        assert dets.n_det() == 1
        assert dets.xyxy.shape == (1, 4)

    def test_NMS_贪心按置信度(self) -> None:
        boxes = np.asarray(
            [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)], dtype=float,
        )
        conf = np.asarray([0.9, 0.8, 0.7])
        keep = nms_xyxy(boxes, conf, iou_thr=0.5)
        assert keep == [0, 2]  # 高置信重叠对中保留先者 + 无关框

    def test_低置信过滤(self) -> None:
        frame = render_water_frame(
            SIZE, 0.0, [PelletSpec(start_xy=(160.0, 120.0), radius=6.0)], seed=23,
        )

        def low_conf(tile):
            return [((b[0], b[1], b[2], b[3]), 0.05) for b in []]  # 无返回

        tp = TiledPredictor(low_conf, conf_min=0.25)
        assert tp.predict(frame).n_det() == 0


# ----------------------------------------------------------------------
# 接口一致性（三轨同构）
# ----------------------------------------------------------------------
class Test接口一致性:

    def test_三轨返回同构PelletDetections(self) -> None:
        pellets = _isolated_pellets(5, radius=6.0, spacing=40.0)
        frame = render_water_frame(SIZE, 0.0, pellets, seed=24)
        detectors = [
            YoloPelletDetector(predict_fn=_oracle_predict),
            YoloEDetector(predict_fn=_oracle_predict),
            AreaIntegralCounter(a_single=_isolated_a_single(6.0)),
        ]
        for det in detectors:
            assert det.available() is True
            assert det.unavailable_reason() is None
            dets = det.detect(frame, t_s=0.0, roi=None)
            assert isinstance(dets, PelletDetections)
            assert dets.n_det() == 5
            assert dets.xyxy.shape == (5, 4)
            assert dets.conf.shape == (5,)
            assert float(dets.conf.min()) > 0.0
            # last_stats 就位（供 orchestrator 写 extra）
            stats = det.last_stats
            assert stats is not None and stats.available is True
            assert stats.n == 5
            assert stats.source in ("det", "area")

    def test_名称与统计可序列化(self) -> None:
        det = AreaIntegralCounter(a_single=_isolated_a_single(6.0))
        frame = render_water_frame(
            SIZE, 0.0, [PelletSpec(start_xy=(160.0, 120.0), radius=6.0)], seed=25,
        )
        det.detect(frame, t_s=0.0)
        d = det.last_stats.to_dict()
        assert d["source"] == "area" and d["available"] is True


# ----------------------------------------------------------------------
# 面积积分轨精度（验收：100 颗密集粘连 ≤10%）
# ----------------------------------------------------------------------
class Test面积积分轨:

    def test_密集粘连100颗_误差不超过10百分比(self) -> None:
        radius, spacing = 5.0, 9.0  # 间距 < 2r → 粘连团块
        pellets = _isolated_pellets(100, radius=radius, spacing=spacing)
        frame = render_water_frame(SIZE, 0.0, pellets, seed=26)
        counter = AreaIntegralCounter(a_single=_isolated_a_single(radius))
        dets = counter.detect(frame, t_s=0.0)
        stats = counter.last_stats
        assert stats.available is True
        assert stats.extra["n_blobs"] < 100  # 确认确实粘连成团
        err = abs(stats.n - 100) / 100
        assert err <= 0.10, f"面积积分计数 {stats.n} vs 真值 100（偏差 {err:.1%}）"
        assert dets.n_det() == stats.n

    def test_孤立颗粒高置信_粘连拆分低置信(self) -> None:
        isolated = render_water_frame(
            SIZE, 0.0, [PelletSpec(start_xy=(160.0, 120.0), radius=6.0)], seed=27,
        )
        counter = AreaIntegralCounter(a_single=_isolated_a_single(6.0))
        counter.detect(isolated, t_s=0.0)
        assert float(counter.last_stats.n) == 1
        # 单颗框置信 = isolate_conf（默认 0.95）
        dets = counter.detect(isolated, t_s=0.0)
        assert float(dets.conf[0]) == pytest.approx(0.95)

    def test_Asingle按时间取值(self) -> None:
        # 膨胀场景：t=0 → 100px²，t=10 → 200px²
        curve = ASingleCurve(t_s=[0.0, 10.0], area_px=[100.0, 200.0])
        counter = AreaIntegralCounter(a_single=curve)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # 前景面积 600px：t=0 → 6 颗，t=5 → 4 颗（插值 A_single=150）
        mask = np.zeros((240, 320), dtype=bool)
        mask[100:110, 100:160] = True
        frame[mask] = (60, 200, 230)
        n0 = counter.detect(frame, t_s=0.0).n_det()
        n5 = counter.detect(frame, t_s=5.0).n_det()
        assert n0 == 6
        assert n5 == 4

    def test_ROI计数区掩膜(self) -> None:
        roi = ROI(
            arena=np.array([[0, 0], [320, 0], [320, 240], [0, 240]], dtype=float),
            pellet_zone=np.array([[0, 0], [100, 0], [100, 240], [0, 240]], dtype=float),
        )
        pellets = [
            PelletSpec(start_xy=(50.0, 120.0), radius=6.0),   # 区内
            PelletSpec(start_xy=(250.0, 120.0), radius=6.0),  # 区外
        ]
        frame = render_water_frame(SIZE, 0.0, pellets, seed=28)
        counter = AreaIntegralCounter(a_single=_isolated_a_single(6.0))
        dets = counter.detect(frame, t_s=0.0, roi=roi)
        assert dets.n_det() == 1
        assert counter.last_stats.n == 1


# ----------------------------------------------------------------------
# 检测轨（桩）精度 ≤20%
# ----------------------------------------------------------------------
class Test检测轨打桩:

    def test_孤立50颗_误差不超过20百分比(self) -> None:
        pellets = _isolated_pellets(50, radius=6.0, spacing=35.0)
        frame = render_water_frame(SIZE, 0.0, pellets, seed=29)
        det = YoloEDetector(predict_fn=_oracle_predict, tile_size=100, overlap=0.2,
                            iou_nms=0.3)
        dets = det.detect(frame, t_s=0.0)
        err = abs(dets.n_det() - 50) / 50
        assert err <= 0.20, f"桩检测 {dets.n_det()} vs 真值 50（偏差 {err:.1%}）"

    def test_检测轨ROI区域过滤(self) -> None:
        roi = ROI(
            arena=np.array([[0, 0], [320, 0], [320, 240], [0, 240]], dtype=float),
            pellet_zone=np.array([[0, 0], [100, 0], [100, 240], [0, 240]], dtype=float),
        )
        pellets = [
            PelletSpec(start_xy=(50.0, 120.0), radius=6.0),
            PelletSpec(start_xy=(250.0, 120.0), radius=6.0),
        ]
        frame = render_water_frame(SIZE, 0.0, pellets, seed=30)
        det = YoloPelletDetector(predict_fn=_oracle_predict)
        dets = det.detect(frame, t_s=0.0, roi=roi)
        assert dets.n_det() == 1


# ----------------------------------------------------------------------
# 轨不可用：绝不把空检测当 0 颗观测
# ----------------------------------------------------------------------
class Test轨不可用:

    def test_面积轨A_single未标定(self) -> None:
        counter = AreaIntegralCounter(a_single=None)
        assert counter.available() is False
        reason = counter.unavailable_reason()
        assert reason is not None and "A_single" in reason
        frame = render_water_frame(SIZE, 0.0, [PelletSpec((160.0, 120.0))], seed=31)
        dets = counter.detect(frame, t_s=0.0)
        assert dets.n_det() == 0
        stats = counter.last_stats
        assert stats.available is False
        assert stats.n == 0
        assert stats.reason is not None and "A_single" in stats.reason

    def test_面积轨曲线全缺测(self) -> None:
        curve = ASingleCurve(t_s=[0.0], area_px=[None])
        counter = AreaIntegralCounter(a_single=curve)
        assert counter.available() is False

    def test_YOLO权重缺失(self, tmp_path) -> None:
        det = YoloPelletDetector(weights_path=tmp_path / "no_such_weights.pt")
        assert det.available() is False
        reason = det.unavailable_reason()
        assert reason is not None and "权重" in reason
        dets = det.detect(np.zeros((240, 320, 3), dtype=np.uint8))
        assert dets.n_det() == 0
        assert det.last_stats.available is False
        assert det.last_stats.source == "det"

    def test_set_A_single注入后可用(self) -> None:
        counter = AreaIntegralCounter(a_single=None)
        assert counter.available() is False
        counter.set_A_single(_isolated_a_single(6.0))
        assert counter.available() is True

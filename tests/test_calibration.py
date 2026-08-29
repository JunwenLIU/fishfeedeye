"""饲料颗粒动态学标定测试（T02）：分割 / A_single(t) / v_sink / 整视频标定。

覆盖：
    - segment_pellets：亮色浮料前景提取、暗色水面不误检；
    - extract_blobs：连通域计数与面积；
    - measure_a_single：孤立颗粒中位数面积、无颗粒帧 = None（缺测不插值）；
    - ASingleCurve：相邻有效点线性插值、端点外取最近、无有效点 = None；
    - measure_v_sink_px_s：漂移速度 95 分位、样本不足 → None；
    - calibrate_video：真实合成视频整链路 + YAML 往返；
    - 标定失败防御：全程无颗粒 → 报错（绝不硬编码猜测值）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.pipeline.pellet_dynamics import (
    ASingleCurve,
    PelletDynamicsCalibration,
    SegmentationParams,
    calibrate_video,
    extract_blobs,
    measure_a_single,
    measure_v_sink_px_s,
    segment_pellets,
)
from tests.fixtures.synthetic import (
    PelletSpec,
    calibration_pellets,
    make_video,
    render_water_frame,
)

SIZE = (320, 240)


class Test前景分割:

    def test_亮色颗粒检出_暗色水面不误检(self) -> None:
        pellets = [PelletSpec(start_xy=(50.0, 50.0), radius=6.0)]
        with_pellets = render_water_frame(SIZE, 0.0, pellets, seed=1)
        without = render_water_frame(SIZE, 0.0, [], seed=1)
        m1 = segment_pellets(with_pellets)
        m0 = segment_pellets(without)
        assert m1.sum() > 50          # 颗粒前景存在
        assert m0.sum() == 0          # 波浪/噪声水面不产生前景

    def test_连通域计数与面积(self) -> None:
        pellets = [
            PelletSpec(start_xy=(40.0, 40.0), radius=6.0),
            PelletSpec(start_xy=(160.0, 120.0), radius=6.0),
            PelletSpec(start_xy=(260.0, 200.0), radius=6.0),
        ]
        frame = render_water_frame(SIZE, 0.0, pellets, seed=2)
        blobs = extract_blobs(segment_pellets(frame), min_area_px=6)
        assert len(blobs) == 3
        for b in blobs:
            assert b.area_px == pytest.approx(math.pi * 36.0, rel=0.35)

    def test_噪点面积过滤(self) -> None:
        # 1 像素噪点不应产生前景（形态学开运算 + 面积下限）
        frame = render_water_frame((60, 60), 0.0, [], seed=3)
        frame[10, 10] = (60, 200, 230)  # 单像素亮点
        blobs = extract_blobs(segment_pellets(frame), min_area_px=6)
        assert blobs == []


class TestASingle曲线:

    def test_插值与端点(self) -> None:
        curve = ASingleCurve(t_s=[0.0, 2.0], area_px=[100.0, 200.0])
        assert curve.available is True
        assert curve.value_at(0.0) == pytest.approx(100.0)
        assert curve.value_at(1.0) == pytest.approx(150.0)
        assert curve.value_at(2.0) == pytest.approx(200.0)
        assert curve.value_at(-1.0) == pytest.approx(100.0)  # 端点外取最近
        assert curve.value_at(5.0) == pytest.approx(200.0)

    def test_缺测点跳过_相邻有效点插值(self) -> None:
        curve = ASingleCurve(t_s=[0.0, 1.0, 2.0], area_px=[100.0, None, 200.0])
        assert curve.value_at(1.0) == pytest.approx(150.0)
        assert curve.area_px[1] is None  # 缺测保持 None（不插值冒充观测）

    def test_无有效点_一切取值None(self) -> None:
        curve = ASingleCurve(t_s=[0.0, 1.0], area_px=[None, None])
        assert curve.available is False
        assert curve.value_at(0.5) is None

    def test_长度不一致_报错(self) -> None:
        with pytest.raises(ValueError, match="长度必须一致"):
            ASingleCurve(t_s=[0.0, 1.0], area_px=[1.0])

    def test_字典往返(self) -> None:
        curve = ASingleCurve(t_s=[0.0, 1.0], area_px=[100.0, None])
        d = curve.to_dict()
        back = ASingleCurve.from_dict(d)
        assert back.t_s == curve.t_s
        assert back.area_px == curve.area_px


class TestASingle测量:

    def test_孤立颗粒_面积中位数(self) -> None:
        pellets = calibration_pellets(n=9, radius=6.0, drift_px_s=0.0, size=SIZE)
        frames = [render_water_frame(SIZE, t, pellets, seed=4) for t in (0.0, 1.0, 2.0)]
        curve = measure_a_single(frames, [0.0, 1.0, 2.0])
        assert curve.available is True
        for a in curve.area_px:
            assert a is not None
            assert a == pytest.approx(math.pi * 36.0, rel=0.35)

    def test_无颗粒帧_记缺测None(self) -> None:
        pellets = [PelletSpec(start_xy=(50.0, 50.0), radius=6.0)]
        frames = [
            render_water_frame(SIZE, 0.0, pellets, seed=5),
            render_water_frame(SIZE, 1.0, [], seed=5),      # 无颗粒
            render_water_frame(SIZE, 2.0, pellets, seed=5),
        ]
        curve = measure_a_single(frames, [0.0, 1.0, 2.0])
        assert curve.area_px[1] is None
        assert curve.area_px[0] is not None

    def test_帧数与时间戳不一致_报错(self) -> None:
        frame = render_water_frame(SIZE, 0.0, [], seed=6)
        with pytest.raises(ValueError, match="长度必须一致"):
            measure_a_single([frame], [0.0, 1.0])


class TestVSink测量:

    def test_匀速漂移_95分位速度(self) -> None:
        pellets = [
            PelletSpec(start_xy=(40.0, 40.0), radius=6.0, velocity_px_s=(4.0, 0.0)),
            PelletSpec(start_xy=(200.0, 160.0), radius=6.0, velocity_px_s=(4.0, 0.0)),
        ]
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        frames = [render_water_frame(SIZE, t, pellets, seed=7) for t in ts]
        v = measure_v_sink_px_s(frames, ts)
        assert v is not None
        assert 3.0 < v < 5.5  # 每帧位移 4px / 1s

    def test_样本不足_None不拍脑袋(self) -> None:
        pellets = [PelletSpec(start_xy=(40.0, 40.0), radius=6.0)]
        frames = [render_water_frame(SIZE, t, pellets, seed=8) for t in (0.0, 1.0)]
        assert measure_v_sink_px_s(frames, [0.0, 1.0]) is None

    def test_无颗粒_None(self) -> None:
        frames = [render_water_frame(SIZE, t, [], seed=9) for t in range(5)]
        assert measure_v_sink_px_s(frames, list(map(float, range(5)))) is None


class Test整视频标定:

    @pytest.fixture(scope="class")
    def calib_video(self, tmp_path_factory):
        path = make_video(
            tmp_path_factory.mktemp("calib") / "pure_feed.avi",
            fps=10.0, duration_s=6.0,
            pellets=calibration_pellets(n=12, radius=6.0, drift_px_s=4.0, size=SIZE),
            size=SIZE, seed=10,
        )
        return path

    def test_标定产出与YAML往返(self, calib_video, tmp_path) -> None:
        calib = calibrate_video(calib_video, feed_id="feedT", px_per_mm=3.2)
        assert calib.n_frames == 60
        assert calib.a_single.available is True
        assert calib.v_sink_max_px_s is not None
        assert calib.v_sink_max_px_s > 0
        assert calib.v_sink_max_mm_s == pytest.approx(
            calib.v_sink_max_px_s / 3.2, rel=1e-9
        )
        # 关联半径 = 2.5 × 平均等效直径（不硬编码）
        eq_diam = 2.0 * math.sqrt((calib.mean_pellet_area_px or 0.0) / math.pi)
        assert calib.association_radius_px == pytest.approx(2.5 * eq_diam, rel=1e-6)

        out = calib.to_yaml(tmp_path / "feedT.yaml")
        back = PelletDynamicsCalibration.from_yaml(out)
        assert back.feed_id == "feedT"
        assert back.v_sink_max_px_s == pytest.approx(calib.v_sink_max_px_s, rel=1e-9)
        assert back.px_per_mm == 3.2
        assert back.a_single.area_px == pytest.approx(calib.a_single.area_px)

    def test_无px_per_mm_mm口径为None加note(self, calib_video) -> None:
        calib = calibrate_video(calib_video, feed_id="feedT2", px_per_mm=None)
        assert calib.v_sink_max_mm_s is None
        assert calib.px_per_mm is None
        assert any("px_per_mm" in n for n in calib.notes)

    def test_纯水面视频_标定失败报错(self, tmp_path) -> None:
        path = make_video(
            tmp_path / "empty_water.avi", fps=10.0, duration_s=3.0, size=SIZE, seed=11,
        )
        with pytest.raises(ValueError, match="A_single 标定失败"):
            calibrate_video(path, feed_id="bad")

    def test_帧数不足_报错(self, tmp_path) -> None:
        path = make_video(
            tmp_path / "tiny.avi", fps=10.0, duration_s=0.5, size=SIZE, seed=12,
        )
        with pytest.raises(ValueError, match="有效帧不足"):
            calibrate_video(path, feed_id="tiny")

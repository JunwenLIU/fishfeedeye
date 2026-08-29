"""颗粒关联与消失三分类测试（T03：eaten / drifted / unknown）。

覆盖（docs/06 §6 T03 消失三分类全部分支）：
    - eaten：区内部消失且关联约束成立；
    - drifted：最后位置在区外 / 贴近边界带内；
    - unknown：排除区 / ROI 缺失 / 约束未标定 / partial_window；
    - min_missing_frames 公差：断档 < 3 帧轨迹保持续接；
    - pellets=None 的帧不计未命中；
    - LinkResult 计数与 to_dict。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.frame_context import FrameObservation, PelletDetections
from src.core.roi import ROI
from src.pipeline.pellet_linker import LinkResult, PelletLinker


def _rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _roi() -> ROI:
    return ROI(
        arena=_rect(0, 0, 200, 150),
        pellet_zone=_rect(0, 0, 100, 100),
    )


def _obs(t: float, points: list[tuple[float, float]] | None,
         pellets_none: bool = False) -> FrameObservation:
    """t 秒的观测：points 为颗粒质心列表；pellets_none=True → 无检测数据帧。"""
    if pellets_none:
        pellets = None
    elif points:
        boxes = [[x - 4, y - 4, x + 4, y + 4] for x, y in points]
        pellets = PelletDetections(
            xyxy=np.asarray(boxes, dtype=float), conf=np.full(len(points), 0.9),
        )
    else:
        pellets = PelletDetections.empty()
    return FrameObservation(
        frame_idx=int(round(t * 10)), t_s=t, dt_s=None, image=None, pellets=pellets,
    )


def _linker(**kwargs) -> PelletLinker:
    """默认：物理约束已标定（v_sink=10px/s, radius=50px, 公差 3 帧）。"""
    defaults = dict(
        v_sink_max_px_s=10.0, association_radius_px=50.0,
        min_missing_frames=3, roi=_roi(),
    )
    defaults.update(kwargs)
    return PelletLinker(**defaults)


class Test消失分类_被吃掉:

    def test_区内部消失_判eaten(self) -> None:
        frames = (
            [_obs(t, [(50.0, 50.0)]) for t in (0.0, 1.0, 2.0)]
            + [_obs(t, []) for t in (3.0, 4.0, 5.0)]
        )
        result = _linker().link(frames)
        assert result.n_vanished == 1
        assert result.n_eaten == 1
        assert result.n_drifted == 0 and result.n_unknown == 0
        event = result.vanish_events[0]
        assert event.vanish_class == "eaten"
        assert event.missing_frames == 3
        assert event.partial_window is False

    def test_轨迹存活到末帧_无消失事件(self) -> None:
        frames = [_obs(t, [(50.0, 50.0)]) for t in (0.0, 1.0, 2.0, 3.0)]
        result = _linker().link(frames)
        assert result.n_vanished == 0
        assert len(result.tracks) == 1
        assert result.tracks[0].vanish_class is None


class Test消失分类_漂出:

    def test_最后位置在区外_判drifted(self) -> None:
        # 颗粒向外漂移，最后可见位置 (120, 50) 在 pellet_zone (0..100) 外
        frames = (
            [_obs(t, [p]) for t, p in (
                (0.0, (50.0, 50.0)), (1.0, (75.0, 50.0)), (2.0, (120.0, 50.0)),
            )]
            + [_obs(t, []) for t in (3.0, 4.0, 5.0)]
        )
        result = _linker().link(frames)
        assert result.n_drifted == 1
        assert result.n_eaten == 0

    def test_贴近边界带内_判drifted(self) -> None:
        # 最后位置 (96, 50)：在区内但距右边界 4px < margin 10px
        frames = (
            [_obs(t, [(50.0, 50.0)]) for t in (0.0, 1.0)]
            + [_obs(2.0, [(96.0, 50.0)])]
            + [_obs(t, []) for t in (3.0, 4.0, 5.0)]
        )
        result = _linker().link(frames)
        assert result.n_drifted == 1


class Test消失分类_不可判:

    def test_最后位置在排除区_判unknown(self) -> None:
        roi = ROI(
            arena=_rect(0, 0, 200, 150),
            pellet_zone=_rect(0, 0, 100, 100),
            exclude_zones=[_rect(40, 40, 60, 60)],  # 反光区盖住消失点
        )
        frames = (
            [_obs(t, [(50.0, 50.0)]) for t in (0.0, 1.0, 2.0)]
            + [_obs(t, []) for t in (3.0, 4.0, 5.0)]
        )
        result = _linker(roi=roi).link(frames)
        assert result.n_unknown == 1
        assert "排除区" in result.vanish_events[0].reason

    def test_ROI缺失_判unknown(self) -> None:
        frames = (
            [_obs(t, [(50.0, 50.0)]) for t in (0.0, 1.0, 2.0)]
            + [_obs(t, []) for t in (3.0, 4.0, 5.0)]
        )
        result = _linker(roi=None).link(frames)
        assert result.n_unknown == 1
        assert any("ROI 未定义" in n for n in result.notes)

    def test_约束未标定_内部消失一律unknown(self) -> None:
        frames = (
            [_obs(t, [(50.0, 50.0)]) for t in (0.0, 1.0, 2.0)]
            + [_obs(t, []) for t in (3.0, 4.0, 5.0)]
        )
        result = _linker(v_sink_max_px_s=None, association_radius_px=None).link(frames)
        assert result.association_unconstrained is True
        assert result.n_unknown == 1
        assert result.n_eaten == 0  # 没有物理约束支撑，绝不猜"被吃掉"
        assert any("未标定" in n for n in result.notes)

    def test_partial_window_判unknown(self) -> None:
        # v_sink×Δt = 2×10 = 20 ≥ radius 10 → 关联约束违反
        frames = (
            [_obs(t, [(50.0, 50.0)]) for t in (0.0, 10.0, 20.0)]
            + [_obs(t, []) for t in (30.0, 40.0, 50.0)]
        )
        result = _linker(
            v_sink_max_px_s=2.0, association_radius_px=10.0,
        ).link(frames)
        assert result.n_unknown == 1
        event = result.vanish_events[0]
        assert event.partial_window is True
        assert "partial_window" in event.reason


class Test短暂消失容忍:

    def test_断档不足3帧_轨迹续接(self) -> None:
        frames = (
            [_obs(t, [(50.0, 50.0)]) for t in (0.0, 1.0)]
            + [_obs(t, []) for t in (2.0, 3.0)]      # 仅 2 帧断档
            + [_obs(t, [(52.0, 50.0)]) for t in (4.0, 5.0)]
        )
        result = _linker().link(frames)
        assert result.n_vanished == 0
        assert len(result.tracks) == 1           # 同一条轨迹（续接）
        assert len(result.tracks[0].xy) == 4      # 2 + 2 个可见点

    def test_无检测数据帧不计未命中(self) -> None:
        frames = (
            [_obs(0.0, [(50.0, 50.0)])]
            + [_obs(t, None, pellets_none=True) for t in (1.0, 2.0, 3.0, 4.0, 5.0)]
        )
        result = _linker().link(frames)
        assert result.n_vanished == 0  # pellets=None 的帧不惩罚
        assert len(result.tracks) == 1


class Test关联行为:

    def test_超出关联半径_新建轨迹(self) -> None:
        # 半径 50：帧 0 (50,50) → 帧 1 (160,120) 位移 ~124 > 50 → 两轨
        frames = [_obs(0.0, [(50.0, 50.0)]), _obs(1.0, [(160.0, 120.0)])]
        result = _linker().link(frames)
        assert len(result.tracks) == 2

    def test_多颗粒匹配(self) -> None:
        frames = [
            _obs(0.0, [(30.0, 30.0), (70.0, 30.0)]),
            _obs(1.0, [(31.0, 30.0), (71.0, 30.0)]),
        ]
        result = _linker().link(frames)
        assert len(result.tracks) == 2
        assert all(len(tr.xy) == 2 for tr in result.tracks)

    def test_min_missing_frames下限校验(self) -> None:
        with pytest.raises(ValueError, match="min_missing_frames"):
            PelletLinker(min_missing_frames=0)

    def test_结果计数与to_dict(self) -> None:
        frames = (
            [_obs(0.0, [(50.0, 50.0), (20.0, 20.0), (90.0, 20.0)])]
            + [_obs(t, []) for t in (1.0, 2.0, 3.0)]
        )
        result = _linker().link(frames)
        d = result.to_dict()
        assert d["n_eaten"] == 3
        assert d["n_tracks"] == 3
        assert len(d["events"]) == 3
        assert d["association_unconstrained"] is False

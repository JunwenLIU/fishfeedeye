"""timing 模块测试（T01）：合成 VFR / 拼接 fixture → timing_suspect 正确触发。

验收对应 docs/06 §6 T01 验收标准 3。
fixture 全部为合成时间戳序列（VFR 无法用 VideoWriter 真实编码，合成
POS_MSEC 序列是等价且可复现的单元级 fixture）；另附一条真实小视频的
编码-解码往返冒烟（CFR，应无任何告警）。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.timing import (
    DURATION_DEVIATION_TOLERANCE,
    build_timeline,
    build_timeline_from_nominal,
)


def _cfr_ts(n: int, fps: float, start_s: float = 0.0) -> list[float]:
    """恒定帧率时间戳（毫秒）。"""
    return [(start_s + i / fps) * 1000.0 for i in range(n)]


# ======================================================================
# CFR 干净视频：无任何告警
# ======================================================================
class TestCleanCFR:
    def test_no_flags(self):
        fps = 30.0
        ts = _cfr_ts(301, fps)  # 10 s
        r = build_timeline(ts, fps)
        assert not r.timing_suspect
        assert not r.timeline_discontinuity
        assert r.timing_source == "pos_msec"
        assert r.break_indices == []
        assert r.frames_valid_for_time_metrics() == [True] * 301
        assert r.duration_deviation < DURATION_DEVIATION_TOLERANCE

    def test_tiny_deviation_within_tolerance(self):
        # 容差内的轻微抖动（<2%）不算 VFR
        fps = 30.0
        n = 301
        ts = _cfr_ts(n, fps)
        ts[-1] = ts[-1] * 1.01  # 末尾偏移 1%
        r = build_timeline(ts, fps)
        assert not r.timing_suspect

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="时间戳序列为空"):
            build_timeline([], 30.0)

    def test_bad_fps_raises(self):
        with pytest.raises(ValueError, match="帧率"):
            build_timeline(_cfr_ts(10, 30.0), 0.0)


# ======================================================================
# VFR：标称时长 vs 解码时长偏差 > 2% → timing_suspect
# ======================================================================
class TestVFRDetection:
    def test_slow_vfr_triggers_suspect(self):
        """120fps 采集 / 30fps 编码的慢动作式 VFR：实际时长比标称长 4 倍。"""
        fps_nominal = 30.0
        n = 121  # 标称 4 s
        # 实际逐帧间隔 = 1/7.5 s（VFR：播放时长 16 s）
        ts = [i * (1 / 7.5) * 1000.0 for i in range(n)]
        r = build_timeline(ts, fps_nominal)
        assert r.timing_suspect
        assert r.duration_deviation > 0.02
        assert any("VFR" in note for note in r.notes)
        # VFR 不等于断裂：单调性不受影响
        assert not r.timeline_discontinuity

    def test_faster_than_nominal_triggers_suspect(self):
        """反向 VFR（实际比标称快 >2%）：同样触发。"""
        fps = 30.0
        n = 301  # 标称 10 s
        ts = [i * (1 / 35.0) * 1000.0 for i in range(n)]  # 实际 ~8.6 s
        r = build_timeline(ts, fps)
        assert r.timing_suspect

    def test_just_below_threshold_no_trigger(self):
        """偏差恰在 2% 边界以内（1.9%）→ 不触发（避免误报）。"""
        fps = 30.0
        n = 1001  # 标称 ~33.3 s
        ts = _cfr_ts(n, fps)
        scale = 1.019
        ts = [t * scale for t in ts]
        r = build_timeline(ts, fps)
        assert not r.timing_suspect

    def test_vfr_with_monotone_still_yields_valid_time_frames(self):
        """VFR 只告警不拒绝：单调时间轴的帧全部可用于时间指标。"""
        ts = [i * (1 / 25.0) * 1000.0 for i in range(301)]  # 30fps 标称, 25fps 实际
        r = build_timeline(ts, 30.0)
        assert r.timing_suspect
        assert all(r.frames_valid_for_time_metrics())


# ======================================================================
# 拼接 / 续录：时间戳重置（非单调）→ 断裂 + 拒绝断裂后时间指标
# ======================================================================
class TestStitchedVideo:
    def test_timestamp_reset_detected(self):
        """两段各 5 s 拼接：第二段 POS_MSEC 重置为 0 → 折回。"""
        fps = 30.0
        seg1 = _cfr_ts(151, fps)               # 0 → 5 s
        seg2 = _cfr_ts(151, fps)               # 0 → 5 s（重置！）
        ts = seg1 + seg2
        r = build_timeline(ts, fps)
        assert r.timeline_discontinuity
        # 断裂点在第 151 帧（第二段首帧）
        assert 151 in r.break_indices
        # 断裂之后的时间指标一律不输出
        mask = r.frames_valid_for_time_metrics()
        assert len(mask) == 302
        assert all(mask[:151])
        assert not any(mask[151:])

    def test_timestamp_backstep_detected(self):
        """非重置型折回（第二段起于 2 s，仍小于第一段末尾 5 s）。"""
        fps = 30.0
        seg1 = _cfr_ts(151, fps)
        seg2 = _cfr_ts(151, fps, start_s=2.0)  # 2 → 7 s
        ts = seg1 + seg2
        r = build_timeline(ts, fps)
        assert r.timeline_discontinuity
        mask = r.frames_valid_for_time_metrics()
        assert all(mask[:151])
        assert not any(mask[151:])

    def test_equal_timestamp_detected(self):
        """严格单调：相等间隔也判断裂（dt == 0）。"""
        ts = _cfr_ts(10, 30.0)
        ts[5] = ts[4]  # 重复时间戳
        r = build_timeline(ts, 30.0)
        assert r.timeline_discontinuity
        assert 5 in r.break_indices

    def test_gap_jump_detected(self):
        """录制中断 30 s（间隔 > 中位数 × 3）→ 判为跳变断裂。"""
        fps = 30.0
        ts = _cfr_ts(151, fps)          # 0 → 5 s
        tail_start = 5.0 + 30.0         # 中断 30 s
        ts += [(tail_start + i / fps) * 1000.0 for i in range(151)]
        r = build_timeline(ts, fps)
        assert r.timeline_discontinuity
        assert 151 in r.break_indices
        mask = r.frames_valid_for_time_metrics()
        assert all(mask[:151]) and not any(mask[151:])


# ======================================================================
# 标称 FPS 兜底路径
# ======================================================================
class TestNominalFallback:
    def test_source_and_note(self):
        r = build_timeline_from_nominal(900, 30.0)
        assert r.timing_source == "nominal_fps"
        assert not r.timing_suspect
        assert not r.timeline_discontinuity
        assert any("VFR" in note and "系统性偏差" in note for note in r.notes)
        assert r.t_s is not None and len(r.t_s) == 900

    def test_bad_args_raise(self):
        with pytest.raises(ValueError, match="n_frames"):
            build_timeline_from_nominal(0, 30.0)
        with pytest.raises(ValueError, match="帧率"):
            build_timeline_from_nominal(10, -1.0)


# ======================================================================
# 真实视频冒烟：CFR 编码 → 逐帧 POS_MSEC 采集 → 无告警
# ======================================================================
class TestRealVideoSmoke:
    def test_cfr_video_roundtrip(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        from src.utils.video_io import collect_frame_timestamps, probe_video

        path = tmp_path / "cfr_smoke.avi"
        fps = 25.0
        n = 50
        w = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48)
        )
        assert w.isOpened()
        rng = np.random.default_rng(0)
        for _ in range(n):
            frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            w.write(frame)
        w.release()

        info = probe_video(path)
        assert info.fps == pytest.approx(fps, rel=0.01)
        assert info.n_frames == n

        ts, decoded = collect_frame_timestamps(path)
        assert decoded == n
        r = build_timeline(ts, info.fps)
        # MJPG/AVI 的 CFR 编码应单调且时长偏差在容差内
        assert not r.timeline_discontinuity, r.notes
        assert r.t_s is not None
        assert np.all(np.diff(r.t_s) > 0)
        # 允许轻微容器偏差，但不应 > 2%
        assert r.duration_deviation <= 0.02 or r.duration_s == pytest.approx(
            r.nominal_duration_s, abs=0.15
        )

"""视频接入与非对称采样测试（T02 验收：采样契约 + 基线段检测）。

覆盖：
    - plan_sampling_targets：默认参数 30+61+24 = 115 个目标；
    - select_sampling_indices：符号约束 / 去重 / 超范围跳过 / 有效性掩码；
    - ingest 真实合成视频：采样帧数 / t_s / dt_s 真实时间戳口径 /
      基线段可用性判定（≥30s）；
    - t0 边界：未提供（video_start + note）/ 超范围报错；
    - 采样结果为空的防御性报错。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.config import RunConfig
from src.pipeline.ingest import (
    BASELINE_MIN_DURATION_S,
    ingest,
    make_run_id,
    plan_sampling_targets,
    select_sampling_indices,
)
from src.core.config import SamplingParams
from tests.fixtures.synthetic import make_video


@pytest.fixture(scope="module")
def short_video(tmp_path_factory):
    """20s @ 10fps 合成视频（t0=10 时：基线 10s + 早期 10s）。"""
    return make_video(
        tmp_path_factory.mktemp("ingest_short") / "short.avi",
        fps=10.0, duration_s=20.0,
    )


@pytest.fixture(scope="module")
def long_video(tmp_path_factory):
    """80s 合成视频（t0=40 → 基线 40s ≥ 30s，基线可用）。"""
    return make_video(
        tmp_path_factory.mktemp("ingest_long") / "long.avi",
        fps=10.0, duration_s=80.0,
    )


# ----------------------------------------------------------------------
# 采样计划（纯函数）
# ----------------------------------------------------------------------
class Test采样计划:

    def test_默认参数目标数_115(self) -> None:
        targets = plan_sampling_targets(SamplingParams(), 300.0)
        assert len(targets) == 115
        # 基线 [-60, 0) 间隔 2s → 30 个
        baseline = [t for t in targets if t < 0]
        assert len(baseline) == 30
        assert baseline[0] == -60.0 and baseline[-1] == -2.0
        # 早期 [0, 60] 间隔 1s → 61 个（含端点）
        early = [t for t in targets if 0 <= t <= 60]
        assert len(early) == 61
        assert early[0] == 0.0 and early[-1] == 60.0
        # 尾段 (60, 300] 间隔 10s → 24 个
        tail = [t for t in targets if t > 60]
        assert len(tail) == 24
        assert tail[0] == 70.0 and tail[-1] == 300.0
        # 整体升序
        assert targets == sorted(targets)

    def test_观察窗不大于早期窗_报错(self) -> None:
        with pytest.raises(ValueError, match="观察窗"):
            plan_sampling_targets(SamplingParams(), 60.0)


# ----------------------------------------------------------------------
# 采样帧选择（纯函数）
# ----------------------------------------------------------------------
class Test采样帧选择:

    def test_基线目标只匹配负时间帧(self) -> None:
        t_s = np.array([-4.6, -3.5, 1.0, 2.0])
        # 基线目标 -4：只在 t<0 候选中挑最近者（帧1，距 0.5）
        # 非负目标 2：精确命中帧3；帧2（t=1，距 1 > 容差 0.5）被容差排除
        chosen = select_sampling_indices(t_s, [-4.0, 2.0])
        assert chosen == [1, 3]

    def test_同帧去重(self) -> None:
        t_s = np.array([-4.0])
        # 两个目标都最近于唯一帧 → 只能选一次
        chosen = select_sampling_indices(t_s, [-4.0, -3.9])
        assert chosen == [0]

    def test_目标超范围_跳过不造观测(self) -> None:
        t_s = np.array([0.0, 1.0, 2.0])
        chosen = select_sampling_indices(t_s, [999.0, 1.0])
        assert chosen == [1]

    def test_有效性掩码剔除断裂后帧(self) -> None:
        t_s = np.array([-5.0, -3.0, 1.0, 3.0])
        valid = [True, False, True, True]
        chosen = select_sampling_indices(t_s, [-4.0], valid=valid)
        # 帧1（t=-3）被判无效 → 只能选帧0（t=-5，距离 1 ≤ 容差 1）
        assert chosen == [0]

    def test_空时间轴_返回空(self) -> None:
        assert select_sampling_indices(np.zeros(0), [-1.0]) == []


# ----------------------------------------------------------------------
# ingest（真实合成视频）
# ----------------------------------------------------------------------
class Test接入合成视频:

    def test_20s视频_t0居中_采样与时间戳口径(self, short_video) -> None:
        result = ingest(short_video, RunConfig(), t0_s=10.0)
        # 基线目标 -2..-10（5 个）+ 早期目标 0..10（含 10→9.9 近邻，11 个）
        assert len(result.observations) == 16
        assert result.t0_source == "manual"
        # 观测按 t_s 升序，dt_s 为真实相邻差（不用标称网格推算）
        ts = [o.t_s for o in result.observations]
        assert ts == sorted(ts)
        assert result.observations[0].dt_s is None
        for prev, cur in zip(result.observations[:-1], result.observations[1:]):
            assert cur.dt_s == pytest.approx(cur.t_s - prev.t_s, abs=1e-6)
        # 基线段：8s < 30s → 不可用 + 显式 note
        assert result.baseline_duration_s == pytest.approx(8.0, abs=1e-6)
        assert result.baseline_available is False
        assert any("30s" in n for n in result.notes)
        # 每帧保留解码图（供检测阶段消费）
        assert all(o.image is not None for o in result.observations)

    def test_基线帧全为负时间戳(self, short_video) -> None:
        result = ingest(short_video, RunConfig(), t0_s=10.0)
        baseline_rows = [r for r in result.sampling_table() if r["in_baseline"]]
        non_baseline = [r for r in result.sampling_table() if not r["in_baseline"]]
        assert len(baseline_rows) == 5
        assert all(r["t_s"] < 0 for r in baseline_rows)
        assert all(r["t_s"] >= 0 for r in non_baseline)

    def test_长视频_基线可用(self, long_video) -> None:
        result = ingest(long_video, RunConfig(), t0_s=40.0)
        # 基线 40s（目标 -2..-40 全命中）≥ 30s
        assert result.baseline_duration_s == pytest.approx(38.0, abs=1e-6)
        assert result.baseline_available is True
        assert result.baseline_duration_s >= BASELINE_MIN_DURATION_S

    def test_t0未提供_按首帧处理并显式note(self, short_video) -> None:
        result = ingest(short_video, RunConfig(), t0_s=None)
        assert result.t0_source == "video_start"
        assert result.baseline_duration_s is None
        assert result.baseline_available is False
        assert any("t0 未提供" in n for n in result.notes)

    def test_t0超范围_报错(self, short_video) -> None:
        with pytest.raises(ValueError, match="超出视频时间范围"):
            ingest(short_video, RunConfig(), t0_s=99.0)

    def test_采样结果为空_防御性报错(self, short_video, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.pipeline.ingest.select_sampling_indices",
            lambda *a, **k: [],
        )
        with pytest.raises(ValueError, match="采样结果为空"):
            ingest(short_video, RunConfig(), t0_s=10.0)


class TestRunID:

    def test_run_id_规则(self) -> None:
        rid = make_run_id("/tmp/demo_video.mp4", now=None)
        assert rid.startswith("demo_video_")
        # 末两段 = 日期 8 位 + 时间 6 位
        date_part, time_part = rid.split("_")[-2:]
        assert len(date_part) == 8 and date_part.isdigit()
        assert len(time_part) == 6 and time_part.isdigit()

"""Orchestrator 管线编排测试（T02 验收 1/4 + T03 串接）。

覆盖：
    - 最小闭环：ingest → preprocess → 检测 → 关联 → run 目录留档；
    - 缓存与中断续跑：已完成帧跳过、缓存不被破坏、结果可恢复；
    - FrameObservation 逐帧输出（计数 / 置信度 / 来源轨标记）；
    - 双轨偏差 Q_dualtrack_gap 质量信号与 >20% 告警；
    - RunMeta 缺必填 → 拒绝启动；无检测器模式；
    - 人工修正留痕（corrections.jsonl + manual_count）。

检测轨用「分割 + 连通域」先知桩（不依赖模型权重）；面积积分轨用
实测 A_single（与生产链路同口径）。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.core.config import RunConfig
from src.core.frame_context import FrameObservation, PelletDetections, RunMeta
from src.core.roi import ROI
from src.pipeline.detectors.area_integral import AreaIntegralCounter
from src.pipeline.detectors.base import DetectStats, PelletDetector
from src.pipeline.meta_validation import MetaValidationError
from src.pipeline.orchestrator import Correction, Orchestrator
from src.pipeline.pellet_dynamics import (
    SegmentationParams,
    extract_blobs,
    segment_pellets,
)
from src.pipeline.pellet_linker import PelletLinker
from tests.fixtures.synthetic import PelletSpec, make_video, render_water_frame

SIZE = (160, 120)
VIDEO_S = 30.0
T0 = 10.0
N_STATIC = 8        # 投喂后静止颗粒数
N_VANISH = 3        # 其中 3 颗在 t_s=15 被吃掉（区内部消失）
VANISH_AT_VIDEO_S = 25.0


def _full_meta() -> RunMeta:
    return RunMeta(
        species="草鱼", n_fish_total=50, feed_mass_g=200.0,
        pellet_mass_mg=150.0, pellet_type="floating",
    )


def _roi() -> ROI:
    return ROI(
        arena=np.array([[0, 0], [160, 0], [160, 120], [0, 120]], dtype=float),
        pellet_zone=np.array([[10, 10], [150, 10], [150, 110], [10, 110]], dtype=float),
    )


def _pellets() -> list[PelletSpec]:
    """8 颗静态颗粒（t_s≥0 出现），其中 3 颗 t_s=15 区内部消失。"""
    xs = (30.0, 65.0, 100.0, 135.0)
    ys = (30.0, 90.0)
    out: list[PelletSpec] = []
    k = 0
    for y in ys:
        for x in xs:
            k += 1
            out.append(
                PelletSpec(
                    start_xy=(x, y), radius=6.0, appear_at_s=T0,
                    vanish_at_s=VANISH_AT_VIDEO_S if k <= N_VANISH else None,
                )
            )
    return out


@pytest.fixture(scope="module")
def analysis_video(tmp_path_factory):
    return make_video(
        tmp_path_factory.mktemp("orch") / "feeding.avi",
        fps=10.0, duration_s=VIDEO_S, pellets=_pellets(), size=SIZE, seed=41,
    )


def _a_single_curve() -> "object":
    """与生产同口径的实测 A_single（孤立单颗 r=6）。"""
    frame = render_water_frame(
        SIZE, 0.0, [PelletSpec(start_xy=(80.0, 60.0), radius=6.0)], seed=42,
    )
    params = SegmentationParams()
    blobs = extract_blobs(segment_pellets(frame, params), params.min_blob_area_px)
    assert len(blobs) == 1
    from src.pipeline.pellet_dynamics import ASingleCurve

    return ASingleCurve(t_s=[0.0], area_px=[blobs[0].area_px])


class _OracleDetRail(PelletDetector):
    """检测轨桩：分割 + 连通域（source='det'），测双轨编排逻辑。"""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self._scale = scale  # 桩偏差系数（模拟检测轨计数偏差）

    def name(self) -> str:
        return "oracle_stub"

    def available(self) -> bool:
        return True

    def detect(self, frame, t_s=None, roi=None) -> PelletDetections:
        params = SegmentationParams()
        blobs = extract_blobs(segment_pellets(frame, params), params.min_blob_area_px)
        boxes = [[b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]] for b in blobs]
        n = int(round(len(boxes) * self._scale))
        boxes = boxes[:n] if n <= len(boxes) else boxes
        self.last_stats = DetectStats(
            n=len(boxes), source="det", available=True,
            extra={"model": self.name()},
        )
        return PelletDetections(
            xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
            conf=np.full(len(boxes), 0.9),
        )


# ----------------------------------------------------------------------
# 最小闭环（验收 1：走通 ingest → preprocess；验收 4：缓存/续跑）
# ----------------------------------------------------------------------
class Test最小闭环:

    def test_面积轨全链路_检测计数与消失三分类(self, analysis_video, tmp_path) -> None:
        orch = Orchestrator(runs_root=tmp_path / "runs")
        result = orch.run(
            video_path=analysis_video,
            meta=_full_meta(),
            config=RunConfig(),
            roi=_roi(),
            t0_s=T0,
            detectors=[AreaIntegralCounter(a_single=_a_single_curve())],
            linker=PelletLinker(
                v_sink_max_px_s=5.0, association_radius_px=30.0, roi=_roi(),
            ),
        )
        # 采样：基线 5 + 早期 21 = 26 帧
        assert len(result.observations) == 26
        assert result.baseline_duration_s == pytest.approx(8.0)
        # 逐帧输出：每帧都有颗粒观测 + 来源轨标记
        for obs in result.observations:
            assert obs.pellets is not None
            assert obs.extra.get("pellet_source") == "area"
            assert obs.extra.get("area_available") is True
            assert obs.pellets.conf.shape[0] == obs.pellets.n_det()
        # 投喂后至 t_s=14：8 颗；t_s≥15：5 颗
        early = [o for o in result.observations if 0.0 <= o.t_s <= 14.0]
        late = [o for o in result.observations if o.t_s >= 15.0]
        assert early and all(o.pellets.n_det() == N_STATIC for o in early)
        assert late and all(o.pellets.n_det() == N_STATIC - N_VANISH for o in late)
        # 基线期 0 颗（t0 判定正确）
        baseline = [o for o in result.observations if o.t_s < 0.0]
        assert all(o.pellets.n_det() == 0 for o in baseline)
        # 消失三分类：3 颗区内部消失且约束成立 → eaten
        assert result.link_result is not None
        assert result.link_result.n_eaten == N_VANISH
        assert result.link_result.n_drifted == 0
        # run 目录留档（可复现锚点 + 缓存）
        assert (result.run_dir / "run_config.yaml").exists()
        assert (result.run_dir / "cache" / "detections.jsonl").exists()
        assert (result.run_dir / "cache" / "timing.json").exists()
        assert result.n_frames_processed == 26

    def test_无检测器模式_采样清单与note(self, analysis_video, tmp_path) -> None:
        orch = Orchestrator(runs_root=tmp_path / "runs")
        result = orch.run(video_path=analysis_video, t0_s=T0, meta=None)
        assert all(o.pellets is None for o in result.observations)
        assert any("未配置检测器" in n for n in result.notes)
        rows = result.sampling_table()
        assert len(rows) == 26 and rows[0]["n_pellets"] is None

    def test_Meta缺必填_拒绝启动报字段名(self, analysis_video, tmp_path) -> None:
        orch = Orchestrator(runs_root=tmp_path / "runs")
        with pytest.raises(MetaValidationError) as ei:
            orch.run(
                video_path=analysis_video, t0_s=T0,
                meta=RunMeta(species=None, n_fish_total=10),
            )
        assert "species" in str(ei.value)


# ----------------------------------------------------------------------
# 缓存与中断续跑（验收 4）
# ----------------------------------------------------------------------
class Test缓存续跑:

    def _first_run(self, analysis_video, runs_root):
        orch = Orchestrator(runs_root=runs_root)
        return orch.run(
            video_path=analysis_video, meta=_full_meta(), roi=_roi(), t0_s=T0,
            detectors=[AreaIntegralCounter(a_single=_a_single_curve())],
            linker=PelletLinker(v_sink_max_px_s=5.0, association_radius_px=30.0,
                                roi=_roi()),
        )

    def test_续跑跳过已完成帧_缓存不破坏(self, analysis_video, tmp_path) -> None:
        runs_root = tmp_path / "runs"
        first = self._first_run(analysis_video, runs_root)
        cache = first.run_dir / "cache" / "detections.jsonl"
        n_lines_first = len(cache.read_text(encoding="utf-8").splitlines())
        n_first = [o.pellets.n_det() for o in first.observations]

        # 续跑：同一 run 目录
        orch = Orchestrator(runs_root=runs_root)
        second = orch.run(
            video_path=analysis_video, meta=_full_meta(), roi=_roi(), t0_s=T0,
            detectors=[AreaIntegralCounter(a_single=_a_single_curve())],
            linker=PelletLinker(v_sink_max_px_s=5.0, association_radius_px=30.0,
                                roi=_roi()),
            run_dir=first.run_dir,
        )
        # 全帧已完成 → 0 新处理，缓存行数不变
        assert second.n_frames_processed == 0
        n_second = [o.pellets.n_det() for o in second.observations]
        assert n_second == n_first
        assert len(cache.read_text(encoding="utf-8").splitlines()) == n_lines_first
        assert any("续跑" in n for n in second.notes)
        # 续跑后消失三分类仍可复算
        assert second.link_result is not None
        assert second.link_result.n_eaten == N_VANISH

    def test_部分续跑_增量处理(self, analysis_video, tmp_path) -> None:
        runs_root = tmp_path / "runs"
        first = self._first_run(analysis_video, runs_root)
        cache = first.run_dir / "cache" / "detections.jsonl"
        # 人为截断缓存：只保留前 13 帧（模拟中途断电）
        lines = cache.read_text(encoding="utf-8").splitlines()
        cache.write_text("\n".join(lines[:13]) + "\n", encoding="utf-8")
        orch = Orchestrator(runs_root=runs_root)
        resumed = orch.run(
            video_path=analysis_video, meta=_full_meta(), roi=_roi(), t0_s=T0,
            detectors=[AreaIntegralCounter(a_single=_a_single_curve())],
            linker=PelletLinker(v_sink_max_px_s=5.0, association_radius_px=30.0,
                                roi=_roi()),
            run_dir=first.run_dir,
        )
        assert resumed.n_frames_processed == 26 - 13
        # 恢复后全部帧均有检测结果（前 13 来自缓存、后 13 本次新算）
        assert all(o.pellets is not None for o in resumed.observations)
        early = [o for o in resumed.observations if 0.0 <= o.t_s <= 14.0]
        assert all(o.pellets.n_det() == N_STATIC for o in early)

    def test_load_cache_不重新解码检测(self, analysis_video, tmp_path) -> None:
        runs_root = tmp_path / "runs"
        first = self._first_run(analysis_video, runs_root)
        orch = Orchestrator(runs_root=runs_root)
        restored = orch.load_cache(first.run_dir)
        assert len(restored.observations) == 26
        assert any("缓存恢复" in n for n in restored.notes)
        n_first = [o.pellets.n_det() for o in first.observations]
        n_restored = [o.pellets.n_det() for o in restored.observations]
        assert n_restored == n_first


# ----------------------------------------------------------------------
# 人工修正留痕（recompute_metrics）
# ----------------------------------------------------------------------
class Test人工修正:

    def test_修正写入manual_count并追加留痕(self, analysis_video, tmp_path) -> None:
        orch = Orchestrator(runs_root=tmp_path / "runs")
        result = orch.run(
            video_path=analysis_video, meta=_full_meta(), roi=_roi(), t0_s=T0,
            detectors=[AreaIntegralCounter(a_single=_a_single_curve())],
        )
        target = result.observations[15]
        corrected = orch.recompute_metrics(
            result.run_dir,
            [Correction(frame_idx=target.frame_idx, new_n=3,
                        operator="reviewer", note="人工复核粘连帧")],
        )
        by_idx = {o.frame_idx: o for o in corrected.observations}
        assert by_idx[target.frame_idx].extra["manual_count"] == 3
        assert by_idx[target.frame_idx].extra["manual_count_operator"] == "reviewer"
        log = (result.run_dir / "corrections.jsonl").read_text(encoding="utf-8")
        record = json.loads(log.strip().splitlines()[-1])
        assert record["frame_idx"] == target.frame_idx
        assert record["new_n"] == 3
        assert record["operator"] == "reviewer"
        assert record["note"] == "人工复核粘连帧"

    def test_修正未知帧号_报错(self, analysis_video, tmp_path) -> None:
        orch = Orchestrator(runs_root=tmp_path / "runs")
        result = orch.run(
            video_path=analysis_video, meta=_full_meta(), roi=_roi(), t0_s=T0,
            detectors=[AreaIntegralCounter(a_single=_a_single_curve())],
        )
        with pytest.raises(ValueError, match="不在缓存观测中"):
            orch.recompute_metrics(result.run_dir, [Correction(frame_idx=99999, new_n=1)])


# ----------------------------------------------------------------------
# 双轨偏差（T03：Q_dualtrack_gap）
# ----------------------------------------------------------------------
class Test双轨偏差:

    def test_双轨偏差超阈值_质量信号与告警(self, analysis_video, tmp_path) -> None:
        orch = Orchestrator(runs_root=tmp_path / "runs")
        result = orch.run(
            video_path=analysis_video, meta=_full_meta(), roi=_roi(), t0_s=T0,
            detectors=[
                _OracleDetRail(scale=0.4),  # 检测轨桩：~40% 计数 → 巨大双轨偏差
                AreaIntegralCounter(a_single=_a_single_curve()),
            ],
        )
        # 检测轨非零 → 主轨选 det
        early = [o for o in result.observations if 0.0 <= o.t_s <= 14.0]
        assert all(o.extra.get("pellet_source") == "det" for o in early)
        assert all("dualtrack_gap" in o.extra for o in early)
        # 面积轨 8 vs 检测桩 3 → gap = 5/8 = 62.5% > 20%
        assert result.quality_signals["Q_dualtrack_gap_max"] > 0.20
        assert any("Q_dualtrack_gap" in w for w in result.warnings)

    def test_检测轨不可用_回退面积轨并显式状态(self, analysis_video, tmp_path) -> None:
        # 面积轨 A_single 未标定 + 检测桩零检出 → 主轨回退 + 状态留痕
        area_unavailable = AreaIntegralCounter(a_single=None)
        orch = Orchestrator(runs_root=tmp_path / "runs")
        result = orch.run(
            video_path=analysis_video, meta=_full_meta(), roi=_roi(), t0_s=T0,
            detectors=[_OracleDetRail(scale=0.0), area_unavailable],
        )
        for obs in result.observations:
            # 不可用状态显式写入 extra，绝不把空检测当 0 颗观测
            assert obs.extra.get("area_available") is False
            assert obs.extra.get("area_reason") is not None
        # 检测桩零检出 → pellets 仍来自 det 轨（空检测但 available=True）
        assert all(o.pellets is not None for o in result.observations)
        assert all(o.pellets.n_det() == 0 for o in result.observations)

"""tests/test_seed_feedbox_t0.py · NFR-05 / FR-38 / FR-08 审计整改回归测试。

覆盖：
    - NFR-05：随机种子固定与留档（RunConfig.seed 往返、可复现性、随机数收敛）
    - FR-38：浮动投喂框部署状态披露（False/None/True 三态）
    - FR-08：t0 自动检测（首颗入画偏移）+ 与人工打点 >1s 偏差告警 + auto 口径披露
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.config import RunConfig
from src.core.frame_context import FrameObservation, PelletDetections, RunMeta
from src.metrics.aggregator import MetricsAggregator
from src.metrics.t0_detect import T0_DEVIATION_ALERT_S, deviation_alert, detect_first_pellet_s


# ----------------------------------------------------------------------
# 最小化观测工厂（不依赖 test_metrics 的私有 fixture，保持本文件自包含）
# ----------------------------------------------------------------------
def _make_pellets(n: int, conf: float = 0.9) -> PelletDetections:
    if n <= 0:
        return PelletDetections.empty()
    cols = int(math.ceil(math.sqrt(n)))
    xyxy: list[list[float]] = []
    for i in range(n):
        r, c = divmod(i, cols)
        x = 20.0 + c * (280.0 / max(1, cols))
        y = 20.0 + r * (200.0 / max(1, rows := int(math.ceil(n / cols))))
        s = 4.0
        xyxy.append([x - s, y - s, x + s, y + s])
    return PelletDetections(
        xyxy=np.asarray(xyxy, dtype=float),
        conf=np.full(n, conf, dtype=float),
    )


def _obs(t_s: float, n: int, frame_idx: int | None = None) -> FrameObservation:
    return FrameObservation(
        frame_idx=frame_idx if frame_idx is not None else int(round(t_s * 10)),
        t_s=float(t_s),
        dt_s=None,
        image=None,
        pellets=_make_pellets(n),
        extra={},
    )


def _aggregate(
    obs: list[FrameObservation],
    *,
    seed: int = 0,
    feedbox_deployed: bool | None = None,
    t0_source: str = "manual",
    meta: RunMeta | None = None,
) -> "MetricsAggregator":
    cfg = RunConfig(
        seed=seed,
        feedbox_deployed=feedbox_deployed,
        t0_source=t0_source,
    )
    agg = MetricsAggregator(config=cfg)
    report = agg.aggregate(obs, meta=meta if meta is not None else RunMeta())
    # 把报告挂在 aggregator 上，便于断言（同时验证聚合不抛异常）
    agg._last_report = report
    return agg  # type: ignore[return-value]


# ======================================================================
# NFR-05 随机种子
# ======================================================================
def test_seed_roundtrip_and_default():
    assert RunConfig().seed == 0
    c = RunConfig(seed=42)
    d = c.to_dict()
    assert d["seed"] == 42
    assert "seed" in c.to_dict()
    c2 = RunConfig.from_dict({"seed": 99, "feedbox_deployed": True})
    assert c2.seed == 99
    assert c2.feedbox_deployed is True


def test_seed_validate_rejects_non_int():
    problems = RunConfig(seed="abc").validate()  # type: ignore[arg-type]
    assert any("seed" in p for p in problems)


def test_seed_recorded_in_report_config():
    """seed 需随 run_config 进入报告，确保可复现留档。"""
    agg = _aggregate([_obs(0, 5)], seed=7)
    d = agg._last_report.to_dict()  # type: ignore[attr-defined]
    assert d["run_config"]["seed"] == 7


def test_chart_seed_wiring_deterministic():
    """set_chart_seed 必须真正驱动图表抖动所用的 RNG。"""
    from src.export import charts as charts_mod

    charts_mod.set_chart_seed(7)
    a = np.random.default_rng(charts_mod._CHART_SEED).random(3)
    b = np.random.default_rng(7).random(3)
    assert np.allclose(a, b)
    charts_mod.set_chart_seed(0)  # 复原默认


def test_aggregate_applies_seed_idempotently():
    """同一 seed 多次聚合给出逐位一致的可复现输出。"""
    obs = [_obs(t, int(5 * math.exp(-0.01 * t))) for t in np.arange(0, 120, 2)]
    r1 = _aggregate(obs, seed=123)._last_report.to_dict()  # type: ignore[attr-defined]
    r2 = _aggregate(obs, seed=123)._last_report.to_dict()  # type: ignore[attr-defined]
    # 关键值型指标必须完全一致（像素级复现）
    assert r1["metrics"]["A1_N0"]["value"] == r2["metrics"]["A1_N0"]["value"]


# ======================================================================
# FR-38 浮动投喂框部署状态披露
# ======================================================================
def test_feedbox_false_triggers_disclosure():
    agg = _aggregate([_obs(0, 5)], feedbox_deployed=False)
    rep = agg._last_report  # type: ignore[attr-defined]
    assert any("未布设浮动投喂框" in w for w in rep.warnings)
    assert any("未使用投喂框" in n for n in rep.notes)


def test_feedbox_none_notes_unrecorded():
    agg = _aggregate([_obs(0, 5)], feedbox_deployed=None)
    rep = agg._last_report  # type: ignore[attr-defined]
    assert any("未记录" in n for n in rep.notes)
    # None 不应误触发 False 分支的告警
    assert not any("未布设浮动投喂框" in w for w in rep.warnings)


def test_feedbox_true_no_disclosure():
    agg = _aggregate([_obs(0, 5)], feedbox_deployed=True)
    rep = agg._last_report  # type: ignore[attr-defined]
    assert not any("未布设浮动投喂框" in w for w in rep.warnings)
    assert not any("未使用投喂框" in n for n in rep.notes)


def test_feedbox_in_report_config_roundtrip():
    cfg = RunConfig(feedbox_deployed=False)
    assert cfg.to_dict()["feedbox_deployed"] is False


# ======================================================================
# FR-08 t0 自动检测
# ======================================================================
def test_t0_auto_detects_first_pellet_offset():
    obs = [_obs(t, 0) for t in [0, 1, 2, 3, 4]] + [_obs(5, 3)] + [_obs(10, 1)]
    agg = _aggregate(obs)
    rep = agg._last_report  # type: ignore[attr-defined]
    assert rep.t0_auto_detect_s == 5.0
    d = rep.to_dict()
    assert d["t0_auto_detect_s"] == 5.0


def test_t0_no_pellets_yields_none():
    obs = [_obs(t, 0) for t in [0, 1, 2, 3, 4]]
    agg = _aggregate(obs)
    rep = agg._last_report  # type: ignore[attr-defined]
    assert rep.t0_auto_detect_s is None


def test_t0_deviation_alert_when_offset_gt_threshold():
    """首颗入画于 +5s（相对人工 t0），|5| > 1s → 告警。"""
    obs = [_obs(t, 0) for t in range(5)] + [_obs(5, 3)]
    agg = _aggregate(obs)
    rep = agg._last_report  # type: ignore[attr-defined]
    assert any("偏差" in w and "t0" in w for w in rep.warnings)


def test_t0_small_offset_no_alert():
    """偏移 0.5s < 1s 不应告警。"""
    obs = [_obs(0, 0), _obs(0.5, 3)]
    agg = _aggregate(obs)
    rep = agg._last_report  # type: ignore[attr-defined]
    assert not any("偏差" in w for w in rep.warnings)


def test_t0_source_auto_disclosure():
    obs = [_obs(2, 5)]
    agg = _aggregate(obs, t0_source="auto")
    rep = agg._last_report  # type: ignore[attr-defined]
    assert any("t0_source=auto" in n for n in rep.notes)


# ======================================================================
# t0_detect 纯函数单测
# ======================================================================
def test_detect_first_pellet_returns_offset():
    samples = [(0.0, 0), (1.0, 0), (3.0, 5), (4.0, 2)]
    assert detect_first_pellet_s(samples) == 3.0


def test_detect_first_pellet_none():
    assert detect_first_pellet_s([(0.0, 0), (1.0, 0)]) is None
    assert detect_first_pellet_s([]) is None


def test_deviation_alert_threshold():
    assert deviation_alert(2.0) is not None
    assert deviation_alert(-1.5) is not None
    assert deviation_alert(0.5) is None
    assert deviation_alert(None) is None
    assert deviation_alert(0.0) is None
    # 阈值常量契约
    assert T0_DEVIATION_ALERT_S == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

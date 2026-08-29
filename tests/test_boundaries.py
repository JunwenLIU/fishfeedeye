"""T04 边界纪律测试："有没有撒谎"。

验收口径（team-lead 派单）：
    - 删失指标确实无值：censored → value=None + window_s + ">窗长"
      reason + status:censored 伪 flag（绝不为 0/NaN 冒充）；
    - 不可用必带 reason：所有 unavailable 产出（A1/A7/A12/A14/B1/B2/D4）
      必须能回答"为什么没输出"；
    - A14 不可用时：无 metadata_contradiction（没有实测即无矛盾判定权），
      有 sedimentation_risk_unverifiable，告警文案不含"实测"；
    - 覆盖必须写告警（warnings/notes），绝不静默：参考区扣除失败、
      基线不稳定、人工修正、降级决策全部留痕。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.config import Thresholds
from src.core.frame_context import (
    BaselineStats,
    FishDetections,
    FrameObservation,
    PelletDetections,
    RunMeta,
)
from src.core.roi import ROI
from src.metrics.capability import apply_capability
from src.metrics.group_a.clearance import clearance_metric, crossing_time
from src.metrics.group_a.n0_estimator import estimate_n0
from src.metrics.group_a.nonfeeding_loss import (
    nonfeeding_loss_metric,
    tally_vanish,
)
from src.metrics.group_a.pellet_curve import (
    extract_pellet_series,
    smooth_series,
)
from src.metrics.group_a.rate import rate_metrics
from src.metrics.group_a.residual import fit_exponential, residual_metrics
from src.metrics.group_b1.spatial_heterogeneity import (
    compute_activity,
    d_group_metrics,
)
from src.metrics.group_b2.zone_metrics import zone_metrics


# ----------------------------------------------------------------------
# 工具（与 test_metrics.py 同构的最小工厂）
# ----------------------------------------------------------------------
def make_pellets(n: int, conf: float = 0.9) -> PelletDetections:
    if n <= 0:
        return PelletDetections.empty()
    cols = int(math.ceil(math.sqrt(n)))
    xyxy = [
        [20.0 + (i % cols) * 12.0, 20.0 + (i // cols) * 12.0,
         24.0 + (i % cols) * 12.0, 24.0 + (i // cols) * 12.0]
        for i in range(n)
    ]
    return PelletDetections(
        xyxy=np.asarray(xyxy, dtype=float), conf=np.full(n, conf),
    )


def make_obs(t_s: float, n: int, frame_idx: int | None = None) -> FrameObservation:
    return FrameObservation(
        frame_idx=frame_idx if frame_idx is not None else int(round(t_s * 10)),
        t_s=float(t_s), dt_s=None, image=None,
        pellets=make_pellets(n), extra={},
    )


def flat_obs(t_end: float = 300.0, dt: float = 2.0, n: int = 100) -> list:
    """不消耗序列（恒 N₀）→ 所有清空时间右删失。"""
    return [make_obs(t, n) for t in np.arange(0.0, t_end, dt)]


def smoothed_flat():
    raw = extract_pellet_series(flat_obs())
    return raw, smooth_series(raw, 5)


def full_roi() -> ROI:
    arena = np.array([[10, 10], [310, 10], [310, 230], [10, 230]], dtype=float)
    return ROI(
        arena=arena,
        feeding_zone=np.array([[10, 10], [160, 10], [160, 120], [10, 120]], dtype=float),
        reference_zone=np.array([[160, 120], [310, 120], [310, 230], [160, 230]], dtype=float),
        pellet_zone=arena,
    )


def baseline(n_fz_mean: float = 4.0, n_fz_std: float = 1.0) -> BaselineStats:
    return BaselineStats(
        n_frames=30, duration_s=60.0, fish_count_mean=8.0, fish_count_std=2.0,
        n_fz_mean=n_fz_mean, n_fz_std=n_fz_std, activity_mean=1.0,
        activity_std=0.2, flow_mean=0.5, flow_std=0.1, annd_mean=50.0,
        annd_std=10.0, pellet_count_mean=0.0,
    )


# ----------------------------------------------------------------------
# 删失指标确实无值
# ----------------------------------------------------------------------
class TestCensoredNoValue:
    def test_t50_censored_has_no_value(self) -> None:
        raw, sm = smoothed_flat()
        n0 = estimate_n0(raw, None, Thresholds(), 300.0).value
        t50 = crossing_time(sm, n0, 0.5, 300.0, Thresholds())
        assert t50 is None
        mv = clearance_metric("A8_T50", t50, 300.0)
        assert mv.status == "censored"
        assert mv.value is None  # 绝不为 0
        assert mv.value is not False and mv.value != 0.0
        assert not isinstance(mv.value, float)
        assert mv.quality["window_s"] == 300.0
        assert mv.quality["censored"] is True
        assert mv.reason is not None and ">" in mv.reason
        assert "观察窗" in mv.reason and "右删失" in mv.reason

    def test_censored_pseudo_flag_in_blocking(self) -> None:
        raw, sm = smoothed_flat()
        n0 = estimate_n0(raw, None, Thresholds(), 300.0).value
        t50 = crossing_time(sm, n0, 0.5, 300.0, Thresholds())
        mv = clearance_metric("A8_T50", t50, 300.0)
        blocking = mv.blocking_flags()
        assert "status:censored" in blocking  # 单列即可筛选
        assert blocking  # 非 ok 指标 blocking_flag_count ≥ 1

    def test_censored_reason_not_pretending_number(self) -> None:
        raw, sm = smoothed_flat()
        n0 = estimate_n0(raw, None, Thresholds(), 300.0).value
        t50 = crossing_time(sm, n0, 0.1, 300.0, Thresholds())
        mv = clearance_metric("A9_T90", t50, 300.0)
        assert mv.value is None
        # 文案声明"不输出 0/NaN 冒充"
        assert "冒充" in (mv.reason or "")

    def test_v50_unavailable_not_window_substitute(self) -> None:
        raw, sm = smoothed_flat()
        n0 = estimate_n0(raw, None, Thresholds(), 300.0).value
        idx = sm.valid_mask()
        out = rate_metrics(sm.t[idx], sm.n[idx], n0, None, Thresholds(), 300.0, {})
        mv = out["A7_v50"]
        assert mv.status == "unavailable"
        assert mv.value is None
        assert mv.reason is not None and "窗长" in mv.reason  # 契约禁止代替

    def test_d4_unavailable_when_start_or_end_censored(self) -> None:
        from src.metrics.group_b1.spatial_heterogeneity import ActivityResult

        res = ActivityResult()  # 无 FA 曲线 → D2/D3 删失
        d = d_group_metrics(res, {}, Thresholds(), None)
        assert d["D4_duration"].status == "unavailable"
        assert d["D4_duration"].value is None
        assert "窗长" in (d["D4_duration"].reason or "")
        # D2 删失同样无值
        assert d["D2_T_start"].value is None
        assert d["D2_T_start"].status in ("censored", "unavailable")


# ----------------------------------------------------------------------
# 不可用必带 reason
# ----------------------------------------------------------------------
class TestUnavailableHasReason:
    def test_n0_unavailable_reason(self) -> None:
        obs = [make_obs(0.0, 100), make_obs(2.0, 100)]  # 早期窗口 < 3 帧
        mv = estimate_n0(extract_pellet_series(obs), None, Thresholds(), 300.0)
        assert mv.status == "unavailable"
        assert mv.value is None
        assert mv.reason is not None and "帧" in mv.reason

    def test_series_unavailable_reason(self) -> None:
        raw = extract_pellet_series(
            [FrameObservation(0, 0.0, None, None, None, {})]
        )
        assert raw.available is False
        assert raw.reason is not None

    def test_a12_fit_failure_reason(self) -> None:
        t = np.arange(0.0, 60.0, 2.0)
        n = np.full(t.shape, 100.0)  # 常数序列 → 零变异
        fit = fit_exponential(t, n, 100.0, Thresholds())
        assert fit["k"] is None
        assert fit["unavailable_reason"] is not None
        out = residual_metrics(t, n, 100.0, Thresholds(), 60.0, True, {})
        assert out["A12_k"].status == "unavailable"
        assert out["A12_k"].reason is not None
        assert "拟合失败" in out["A12_k"].reason

    def test_a14_untracked_reason_declares_uncorrected(self) -> None:
        tally = tally_vanish(None, [], None, Thresholds())
        mv, q = nonfeeding_loss_metric(tally, 100.0, True, Thresholds())
        assert mv.status == "unavailable"
        assert mv.value is None
        assert q is None  # 严禁默认 0
        # 必须显式声明"未校正沉降损失"
        assert "未校正" in (mv.reason or "")

    def test_zone_metrics_no_fish_reason(self) -> None:
        out, series = zone_metrics([], full_roi(), RunMeta(), None, {})
        assert series is None
        for mid in ("B2-5_N_fz_mean", "B2-6_P_fz", "B2-7_RP"):
            assert out[mid].status == "unavailable"
            assert out[mid].value is None
            assert out[mid].reason, f"{mid} 不可用无原因"

    def test_b1_unavailable_without_images_reason(self) -> None:
        obs = [make_obs(0.0, 10), make_obs(2.0, 8)]  # image=None
        res = compute_activity(obs, None, Thresholds())
        assert res.unavailable_reason is not None
        d = d_group_metrics(res, {}, Thresholds(), None)
        assert d["B1_kurtosis_mean"].status == "unavailable"
        assert d["B1_kurtosis_mean"].reason is not None

    def test_rp_closed_reason_mentions_baseline_definition(self) -> None:
        roi = full_roi()
        fish = FishDetections(
            bbox=np.array([[15, 15, 25, 25]], dtype=float),
            conf=np.array([0.8]),
        )
        obs = make_obs(0.0, 10)
        obs.extra["fish"] = fish
        # zone_metrics 消费观测**序列**（内部按 t_s 排序），单帧须包成列表
        out, _ = zone_metrics([obs], roi, RunMeta(n_fish_total=10), None, {})
        mv = out["B2-7_RP"]
        assert mv.status == "unavailable"
        assert mv.value is None
        assert "基线" in (mv.reason or "")  # 定义即相对基线


# ----------------------------------------------------------------------
# A14 不可用时：矛盾三段式的"不冒充实测"分支
# ----------------------------------------------------------------------
class TestA14UnavailableNoContradiction:
    def _signals(self, q_pelletloss) -> dict:
        return {
            "Q_det": 0.8, "Q_fdet": 0.7, "Q_vis": 20.0, "Q_track": 0.8,
            "Q_ids": None, "Q_interf": 0.05, "Q_fg": 0.05, "Q_calib": True,
            "Q_baseline": True, "Q_pelletloss": q_pelletloss,
            "Q_n0gap": 0.05, "Q_censored": False, "Q_motion": 0.5,
        }

    def test_unavailable_a14_yields_unverifiable_not_contradiction(self) -> None:
        # 真实数据流：link=None → tally 不可用 → q_pelletloss=None
        tally = tally_vanish(None, [], None, Thresholds())
        _mv, q = nonfeeding_loss_metric(tally, 100.0, True, Thresholds())
        assert q is None
        meta = RunMeta(pellet_type="floating")
        report = apply_capability(
            self._signals(q), Thresholds(), meta=meta
        )
        # 无 metadata_contradiction（无实测即无判定权）
        for flags in report.metric_flags.values():
            assert "metadata_contradiction" not in flags
        # 有 sedimentation_risk_unverifiable（不覆盖用户声明）
        assert "sedimentation_risk_unverifiable" in report.metric_flags.get(
            "A9_T90", []
        )
        # 告警文案不含"实测"
        for note in report.notes:
            if "unverifiable" in note or "不可测" in note:
                assert "实测" not in note

    def test_available_a14_contradiction_uses_measurement(self) -> None:
        meta = RunMeta(pellet_type="floating")
        report = apply_capability(
            self._signals(0.25), Thresholds(), meta=meta
        )
        assert "metadata_contradiction" in report.metric_flags.get(
            "A9_T90", []
        )
        assert any("以实测为准" in n for n in report.notes)


# ----------------------------------------------------------------------
# 覆盖必须写告警（绝不静默）
# ----------------------------------------------------------------------
class TestOverridesLeaveTrace:
    def test_reference_correction_failure_warns(self) -> None:
        from src.metrics.group_b1.reference_correction import ReferenceCorrection

        # 无参考区 → 强制 no_noise_correction 告警
        corr = ReferenceCorrection(roi=None)
        result = corr.fit([])
        assert result.alpha is None
        assert result.warning is not None
        assert "no_noise_correction" in result.warning
        # apply 不静默造 0：原样返回
        feed = np.array([1.0, 2.0, 3.0])
        out = corr.apply(feed, np.zeros(3), result)
        np.testing.assert_allclose(out, feed)

    def test_baseline_pairs_insufficient_warns(self) -> None:
        from src.metrics.group_b1.reference_correction import ReferenceCorrection

        roi = full_roi()  # 有参考区但基线样本不足
        result = ReferenceCorrection(roi).fit([(1.0, 1.0), (2.0, 2.0)])
        assert result.alpha is None
        assert result.warning is not None
        assert "不足" in result.warning

    def test_unstable_baseline_flagged_not_silent(self) -> None:
        roi = full_roi()
        fish = FishDetections(
            bbox=np.array([[15, 15, 25, 25]], dtype=float),
            conf=np.array([0.8]),
        )
        obs = make_obs(0.0, 10)
        obs.extra["fish"] = fish
        # CV = 1.5/2.0 = 0.75 > 0.5
        out, _ = zone_metrics(
            [obs], roi, RunMeta(n_fish_total=10),
            baseline(n_fz_mean=2.0, n_fz_std=1.5), {},
        )
        assert "unstable_baseline" in out["B2-7_RP"].flags
        assert out["B2-7_RP"].reason is not None

    def test_capability_decisions_all_in_notes(self) -> None:
        s = {
            "Q_det": 0.8, "Q_fdet": 0.7, "Q_vis": 100.0, "Q_track": 0.3,
            "Q_ids": None, "Q_interf": 0.05, "Q_fg": 0.05, "Q_calib": False,
            "Q_baseline": False, "Q_pelletloss": 0.3, "Q_n0gap": 0.4,
            "Q_censored": True, "Q_motion": 0.5,
        }
        report = apply_capability(s, Thresholds(), meta=RunMeta())
        # 多决策同时触发 → notes 必须非空且逐条可读
        assert report.notes
        assert report.disabled_with_reason
        for entry in report.disabled_with_reason:
            assert entry.reason
        # to_dict 序列化不丢告警
        d = report.to_dict()
        assert len(d["notes"]) == len(report.notes)
        assert len(d["disabled_with_reason"]) == len(
            report.disabled_with_reason
        )

    def test_manual_count_frames_never_overwrite_detection(self) -> None:
        # 修正轨只并列：use_manual=False 的原始轨不被 manual_count 污染
        obs = [make_obs(0.0, 100), make_obs(2.0, 90)]
        obs[1].extra["manual_count"] = 70
        orig = extract_pellet_series(obs, use_manual=False)
        corrected = extract_pellet_series(obs, use_manual=True)
        assert orig.n[1] == pytest.approx(90.0)
        assert corrected.n[1] == pytest.approx(70.0)

    def test_a14_partial_window_flagged(self) -> None:
        from src.pipeline.pellet_linker import LinkResult, VanishEvent

        evs = [
            VanishEvent(
                track_id=i, t_last_s=10.0, last_xy=(50.0, 50.0),
                vanish_class="eaten", reason="synthetic",
                missing_frames=3, partial_window=False,
            )
            for i in range(5)
        ]
        link = LinkResult(
            tracks=[], vanish_events=evs, n_eaten=5, n_drifted=0, n_unknown=0,
        )
        tally = tally_vanish(link, [], 300.0, Thresholds())
        assert tally.available
        assert tally.partial_window  # 300s > a14_window_s=60 → 显式标注
        mv, _ = nonfeeding_loss_metric(tally, 100.0, True, Thresholds())
        assert "partial_window" in mv.flags


# ----------------------------------------------------------------------
# MetricValue 构造期防撒谎兜底（冻结契约的行为验证）
# ----------------------------------------------------------------------
class TestConstructorGuardrails:
    def test_censored_with_value_rejected(self) -> None:
        from src.core.metric_value import MetricValue

        with pytest.raises(ValueError):
            MetricValue(
                metric_id="X", value=42.0, unit="s", status="censored",
                reason="x",
            )

    def test_unavailable_with_value_rejected(self) -> None:
        from src.core.metric_value import MetricValue

        with pytest.raises(ValueError):
            MetricValue(
                metric_id="X", value=0.0, unit="s", status="unavailable",
                reason="x",
            )

    def test_degraded_without_reason_rejected(self) -> None:
        from src.core.metric_value import MetricValue

        with pytest.raises(ValueError):
            MetricValue(
                metric_id="X", value=1.0, unit="s", status="degraded",
                reason=None,
            )

    def test_nan_value_rejected_any_status(self) -> None:
        from src.core.metric_value import MetricValue

        with pytest.raises(ValueError):
            MetricValue(
                metric_id="X", value=float("nan"), unit="s", status="ok",
                reason=None,
            )

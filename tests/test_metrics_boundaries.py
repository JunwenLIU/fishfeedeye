"""T04 边界纪律测试（"测有没有撒谎"）。

验收标准（docs/06 §6 T04）：
    - 删失指标确实无值（value=None，quality 带 window_s，绝不 0/NaN）；
    - 不可用指标必带 reason（解释为什么不可得，而不是空缺）；
    - A14 不可用时清空时间类指标不得声明"已校正沉降损失"
      （话术里不得出现"已校正/实测"字样）；
    - NaN 任何 status 都构造不出来（MetricValue 构造期拒绝）；
    - unavailable/censored 的 value 强制 None；
    - 空 run（零输入）也能产出全键 quality（"None 也要有行"）；
    - 序列不可用时 reason 非空且绝不用 0 冒充观测。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.config import Thresholds
from src.core.frame_context import FrameObservation, RunMeta
from src.core.metric_value import MetricValue
from src.metrics.group_a.clearance import clearance_metric, crossing_time
from src.metrics.group_a.n0_estimator import estimate_n0
from src.metrics.group_a.nonfeeding_loss import (
    nonfeeding_loss_metric,
    tally_vanish,
)
from src.metrics.group_a.pellet_curve import extract_pellet_series
from src.metrics.group_a.residual import residual_metrics
from src.metrics.quality import Q_KEYS, QualitySignals

from tests.test_metrics import exp_decay_observations, make_obs, series_from


# ----------------------------------------------------------------------
# MetricValue 构造期纪律（全局约束点）
# ----------------------------------------------------------------------
class TestMetricValueContract:
    def test_nan_value_rejected_any_status(self) -> None:
        with pytest.raises(ValueError):
            MetricValue(
                metric_id="X", value=float("nan"), unit="-", status="ok",
                reason=None,
            )
        with pytest.raises(ValueError):
            MetricValue(
                metric_id="X", value=float("nan"), unit="-", status="unavailable",
                reason="r",
            )

    def test_unavailable_with_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            MetricValue(
                metric_id="X", value=0.0, unit="-", status="unavailable",
                reason="r",
            )

    def test_censored_with_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            MetricValue(
                metric_id="X", value=0.0, unit="-", status="censored", reason="r"
            )

    def test_non_ok_requires_reason(self) -> None:
        with pytest.raises(ValueError):
            MetricValue(
                metric_id="X", value=1.0, unit="-", status="degraded", reason=None
            )

    def test_blocking_flags_includes_status_pseudo_flag(self) -> None:
        mv = MetricValue(
            metric_id="X", value=None, unit="-", status="unavailable", reason="r"
        )
        assert "status:unavailable" in mv.blocking_flags()


# ----------------------------------------------------------------------
# 删失不撒谎
# ----------------------------------------------------------------------
class TestCensoringHonesty:
    def test_censored_t50_has_no_value_and_window(self) -> None:
        # 5% 缓慢消耗：T50 窗内必然未达
        obs = [make_obs(t, int(round(100 - 0.02 * t))) for t in np.arange(0.0, 300.0, 2.0)]
        raw = extract_pellet_series(obs)
        n0 = estimate_n0(raw, None, Thresholds(), 300.0).value
        _, sm = series_from(obs)
        t50 = crossing_time(sm, n0, 0.50, 300.0, Thresholds())
        assert t50 is None
        mv = clearance_metric("A8_T50", None, 300.0)
        assert mv.value is None
        assert mv.status == "censored"
        assert mv.quality["window_s"] == 300.0
        assert mv.quality["censored"] is True
        # reason 必须输出下界语义（">窗长"），绝不为 0/NaN 冒充
        assert mv.reason and ">" in mv.reason and "300" in mv.reason

    def test_vbar50_no_window_substitution(self) -> None:
        from src.metrics.group_a.rate import rate_metrics

        obs = [make_obs(t, 100) for t in np.arange(0.0, 300.0, 2.0)]
        raw, sm = series_from(obs)
        n0 = estimate_n0(raw, None, Thresholds(), 300.0).value
        idx = sm.valid_mask()
        out = rate_metrics(sm.t[idx], sm.n[idx], n0, None, Thresholds(), 300.0, {})
        mv = out["A7_v50"]
        assert mv.status == "unavailable"
        assert mv.value is None
        assert mv.reason is not None
        assert "窗长" in mv.reason  # 显式声明禁止替换

    def test_d4_never_uses_window_as_duration(self) -> None:
        from src.metrics.group_b1.spatial_heterogeneity import (
            ActivityResult,
            d_group_metrics,
        )

        d = d_group_metrics(ActivityResult(), {}, Thresholds(), None)
        assert d["D4_duration"].value is None
        assert d["D4_duration"].status == "unavailable"
        assert "窗长" in d["D4_duration"].reason


# ----------------------------------------------------------------------
# 不可用必带 reason（不静默缺行）
# ----------------------------------------------------------------------
class TestUnavailableAlwaysExplained:
    def test_no_detection_series_reason(self) -> None:
        obs = [FrameObservation(0, t, None, None, None, {}) for t in (0.0, 2.0, 4.0)]
        raw = extract_pellet_series(obs)
        assert raw.available is False
        assert raw.reason and "无颗粒检测" in raw.reason

    def test_n0_unavailable_reason(self) -> None:
        obs = [make_obs(0.0, 100)]
        raw = extract_pellet_series(obs)
        mv = estimate_n0(raw, None, Thresholds(), 300.0)
        assert mv.status == "unavailable"
        assert mv.reason and "不足" in mv.reason

    def test_residual_unavailable_reasons(self) -> None:
        t = np.arange(0.0, 10.0, 2.0)
        n = np.full(t.shape, 100.0)
        out = residual_metrics(t, n, None, Thresholds(), 10.0, True, {})
        for mid, mv in out.items():
            assert mv.status == "unavailable"
            assert mv.reason, f"{mid} 不可用必须带 reason"
            assert mv.value is None

    def test_zone_metrics_unavailable_reasons(self) -> None:
        from src.metrics.group_b2.zone_metrics import zone_metrics

        out, series = zone_metrics([], None, RunMeta(), None, {})
        assert series is None
        for mid, mv in out.items():
            assert mv.status == "unavailable"
            assert mv.reason, f"{mid} 不可用必须带 reason"


# ----------------------------------------------------------------------
# A14 不可用时的声明纪律
# ----------------------------------------------------------------------
class TestA14ClaimDiscipline:
    def test_untracked_loss_no_corrected_claim(self) -> None:
        tally = tally_vanish(None, [], None, Thresholds())
        mv, q = nonfeeding_loss_metric(tally, 100.0, True, Thresholds())
        assert mv.status == "unavailable"
        assert mv.reason is not None
        # 必须显式声明"未校正"，且绝不出现"已校正"式话术
        assert "未校正" in mv.reason
        assert "已校正" not in mv.reason
        assert q is None  # 严禁默认为 0

    def test_clearance_metric_never_claims_corrected_when_unavailable(self) -> None:
        # Q_pelletloss=None（A14 不可用）时清空时间不得声明校正
        mv = clearance_metric(
            "A9_T90", 120.0, 300.0, q_pelletloss=None, pellet_type="floating"
        )
        assert mv.status == "ok"
        assert "已校正" not in (mv.reason or "")

    def test_unmeasured_q_pelletloss_is_none_not_zero(self) -> None:
        signals = QualitySignals().compute([])  # 空输入
        assert signals["Q_pelletloss"] is None


# ----------------------------------------------------------------------
# 空 run 的全键纪律
# ----------------------------------------------------------------------
class TestEmptyRunFullKeys:
    def test_zero_input_still_emits_all_13_rows(self) -> None:
        signals = QualitySignals().compute([])
        missing = [k for k in Q_KEYS if k not in signals]
        assert missing == []
        # 未测得一律 None（Q_censored 占位 False 除外）
        for k in Q_KEYS:
            if k == "Q_censored":
                continue
            assert signals[k] is None, f"{k} 未测得应为 None 而非 {signals[k]!r}"

    def test_meta_gaps_never_fabricated(self) -> None:
        signals = QualitySignals().compute([], meta=RunMeta(), n0_det=50.0)
        assert signals["Q_n0gap"] is None  # N0_meta 缺失不猜测
        assert signals["N0_det"] == 50.0
        assert signals["N0_meta"] is None


# ----------------------------------------------------------------------
# 数值纪律：None 与 0 的区分
# ----------------------------------------------------------------------
class TestNoneVsZero:
    def test_zero_pellets_is_valid_observation(self) -> None:
        obs = [make_obs(0.0, 100), make_obs(10.0, 0)]  # 吃完（有效 0）
        raw = extract_pellet_series(obs)
        assert raw.available
        assert raw.n[1] == 0.0

    def test_exp_decay_endgame_reaches_valid_zero(self) -> None:
        # k=0.2 的快衰减：t>40s 后 N<1 → 0 颗有效观测
        obs = [make_obs(t, int(round(100 * math.exp(-0.2 * t))))
               for t in np.arange(0.0, 300.0, 2.0)]
        raw = extract_pellet_series(obs)
        assert raw.n[-1] == 0.0
        assert raw.available

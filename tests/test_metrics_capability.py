"""T04 capability 降级矩阵测试（每条规则至少一个触发用例）。

覆盖（docs/04 §4 + team-lead 派单三规则）：
    - 元规则：自动分档（烈度 UNKNOWN/GENTLE）只加 low_sensitivity 标记，
      绝不关闭任何指标；
    - Q_track < 0.50 → C 组整组关闭 + 非随机缺失告警（informative hint）；
    - Q_baseline=False → B2-7_RP 强制关闭（相对指标定义即基线）；
    - Q_calib=False → mm 量纲指标降像素（unnormalized 降级不关闭）；
    - Q_pelletloss > 0.15 → T90/T100/RR 降级 + contains_non_feeding_loss；
    - pellet_type='sinking'（用户声明）→ T90/T100/RR 降级；
    - pellet_type 未确认 → 仅 sedimentation_risk_unassessed 标记；
    - Q_n0gap > 0.20 → denominator_suspect 传染；
    - 户外（用户确认）→ B2/C 只降级（outdoor_exploratory）不关闭；
    - 密度分档（自动）只出现在 density_tier 字段，不产生关闭。
"""
from __future__ import annotations

import pytest

from src.core.config import Thresholds
from src.core.frame_context import RunMeta
from src.metrics.capability import (
    INFORMATIVE_MISSINGNESS_HINT,
    CapabilityReport,
    apply_capability,
)


def signals(**overrides) -> dict:
    """13 Q_* 信号字典（默认全 None = 未测得）。"""
    from src.metrics.quality import Q_KEYS

    base = {k: None for k in Q_KEYS}
    base["Q_censored"] = False
    base.update(overrides)
    return base


class TestMetaRule:
    def test_vigor_unknown_only_flags_never_closes(self) -> None:
        rpt = apply_capability(signals(), Thresholds())
        assert isinstance(rpt, CapabilityReport)
        assert rpt.vigor_tier == "UNKNOWN"
        assert rpt.closed_groups == []
        assert rpt.disabled_with_reason == []
        # 只加 low_sensitivity 标记
        assert "low_sensitivity" in rpt.metric_flags.get("B1_frame_diff", [])
        assert "low_sensitivity" in rpt.metric_flags.get(
            "B1_spatial_heterogeneity", []
        )
        assert "low_sensitivity" in rpt.metric_flags.get("D2_T_start", [])

    def test_density_tier_never_closes(self) -> None:
        # 高密度（自动分档）也只出现在 density_tier，不关闭任何指标
        rpt = apply_capability(signals(Q_vis=80.0), Thresholds())
        assert rpt.density_tier == "HIGH"
        assert rpt.disabled_with_reason == []

    def test_gentle_flags_low_sensitivity(self) -> None:
        rpt = apply_capability(
            signals(), Thresholds(), fa_dynamic_range=1.2  # ≤ vigor_ratio_low
        )
        assert rpt.vigor_tier == "GENTLE"
        assert rpt.closed_groups == []
        assert rpt.disabled_with_reason == []


class TestTrackRule:
    def test_low_qtrack_closes_c_group_with_hint(self) -> None:
        rpt = apply_capability(signals(Q_track=0.3), Thresholds())
        assert "C" in rpt.closed_groups
        entries = [e for e in rpt.disabled_with_reason if e.metric == "C_group"]
        assert len(entries) == 1
        assert "Q_track" in entries[0].reason
        assert entries[0].hint == INFORMATIVE_MISSINGNESS_HINT
        assert "非随机" in entries[0].hint or "效应本身" in entries[0].hint

    def test_good_qtrack_no_close(self) -> None:
        rpt = apply_capability(signals(Q_track=0.8), Thresholds())
        assert rpt.closed_groups == []
        assert rpt.disabled_with_reason == []

    def test_unmeasured_qtrack_no_close(self) -> None:
        # None = 未测得，无权关闭
        rpt = apply_capability(signals(Q_track=None), Thresholds())
        assert rpt.closed_groups == []


class TestBaselineRule:
    def test_missing_baseline_closes_rp_only(self) -> None:
        rpt = apply_capability(signals(Q_baseline=False), Thresholds())
        closed = {e.metric for e in rpt.disabled_with_reason}
        assert "B2-7_RP" in closed
        assert "C_group" not in closed
        rp = [e for e in rpt.disabled_with_reason if e.metric == "B2-7_RP"][0]
        assert "Q_baseline" in rp.reason
        # B1 只降级（uncalibrated），不关闭
        assert "uncalibrated" in rpt.metric_flags.get("B1_frame_diff", [])

    def test_available_baseline_no_close(self) -> None:
        rpt = apply_capability(signals(Q_baseline=True), Thresholds())
        assert not any(e.metric == "B2-7_RP" for e in rpt.disabled_with_reason)


class TestCalibRule:
    def test_missing_calib_degrades_to_px_not_closes(self) -> None:
        rpt = apply_capability(signals(Q_calib=False), Thresholds())
        assert rpt.disabled_with_reason == []
        for mid in ("B1_flow_mm_s", "B2-1_ANND", "B2-2_MDC", "B2-4_FIFFB"):
            assert "unnormalized" in rpt.metric_flags.get(mid, [])
            assert mid in rpt.degrade_reasons


class TestPelletLossRule:
    def test_pelletloss_over_15pct_degrades_t90_rr(self) -> None:
        rpt = apply_capability(signals(Q_pelletloss=0.25), Thresholds())
        assert rpt.disabled_with_reason == []  # 只降级不关闭
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            assert "contains_non_feeding_loss" in rpt.metric_flags.get(mid, [])
            assert mid in rpt.degrade_reasons
            assert "命运未确认率" in rpt.degrade_reasons[mid]

    def test_pelletloss_below_threshold_no_action(self) -> None:
        rpt = apply_capability(signals(Q_pelletloss=0.05), Thresholds())
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            assert mid not in rpt.degrade_reasons

    def test_fatelost_unknown_inclusive_drives_degrade(self) -> None:
        # PRD FR-14：命运未确认率 = (漂出 + 反光区消失)/N0 驱动降级，
        # 即使已知非摄食损失率(Q_pelletloss)偏低
        rpt = apply_capability(
            signals(Q_pelletloss=0.05, Q_fatelost=0.25), Thresholds()
        )
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            assert "contains_non_feeding_loss" in rpt.metric_flags.get(mid, [])
            assert mid in rpt.degrade_reasons
        # 仅 unknown 高、已知漂出低时不应仅凭 Q_pelletloss 触发
        rpt2 = apply_capability(signals(Q_pelletloss=0.05), Thresholds())
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            assert mid not in rpt2.degrade_reasons


class TestPelletTypeRule:
    def test_sinking_user_meta_degrades(self) -> None:
        meta = RunMeta(pellet_type="sinking")
        rpt = apply_capability(signals(), Thresholds(), meta=meta)
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            assert "contains_non_feeding_loss" in rpt.metric_flags.get(mid, [])
            assert mid in rpt.degrade_reasons

    def test_unknown_pellet_type_flag_only(self) -> None:
        meta = RunMeta(pellet_type=None)
        rpt = apply_capability(signals(), Thresholds(), meta=meta)
        assert rpt.disabled_with_reason == []
        for mid in ("A9_T90", "A10_T100"):
            assert "sedimentation_risk_unassessed" in rpt.metric_flags.get(mid, [])
            assert mid not in rpt.degrade_reasons  # 未确认元数据不降级


class TestN0GapRule:
    def test_n0gap_flags_denominator_suspect_contagion(self) -> None:
        rpt = apply_capability(signals(Q_n0gap=0.35), Thresholds())
        for mid in ("A8_T50", "A9_T90", "A11_RR", "A13_AUC60"):
            assert "denominator_suspect" in rpt.metric_flags.get(mid, [])


class TestOutdoorRule:
    def test_outdoor_degrades_but_never_closes(self) -> None:
        rpt = apply_capability(
            signals(Q_baseline=True), Thresholds(), outdoor=True
        )
        assert rpt.outdoor is True
        assert rpt.disabled_with_reason == []
        for mid in ("B2-5_N_fz_mean", "B2-6_P_fz", "B2-7_RP"):
            assert "outdoor_exploratory" in rpt.metric_flags.get(mid, [])
            assert mid in rpt.degrade_reasons


class TestReportSerialization:
    def test_to_dict_json_serializable(self) -> None:
        import json

        rpt = apply_capability(signals(Q_track=0.3, Q_baseline=False), Thresholds())
        d = rpt.to_dict()
        json.dumps(d, ensure_ascii=False)  # 不抛异常即可
        assert "disabled_with_reason" in d
        assert d["closed_groups"] == ["C"]

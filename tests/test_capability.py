"""T04 降级矩阵（capability）边界行为测试。

验收口径（team-lead 派单 + docs/04 §4）：
    - 元规则：自动推断（烈度/密度档）**只降级不关闭**——vigor UNKNOWN
      时无任何 disabled 条目，只有 low_sensitivity 标记；
    - 关闭权归属：只有实测 Q_* 与用户显式元数据有权关闭
      （Q_track → C 组整组；Q_baseline=False → B2-7_RP；sinking → 降级）；
    - 沉性料/漂出超阈 → T90/T100/RR contains_non_feeding_loss + 降级；
    - 矛盾三段式：声明 floating + Q_pelletloss 超阈（A14 可用）→
      metadata_contradiction 以实测为准；Q_pelletloss 不可测（A14 不可用）→
      sedimentation_risk_unverifiable，告警文案不含"实测"；
    - FLAG_GLOSSARY：阻断 flag + status 伪 flag 全部可查到释义；
    - 每次决策覆盖都写 notes（绝不静默省略）。
"""
from __future__ import annotations

import pytest

from src.core.config import Thresholds
from src.core.frame_context import RunMeta
from src.core.metric_value import BLOCKING_FLAGS
from src.metrics.capability import (
    FLAG_GLOSSARY,
    INFORMATIVE_MISSINGNESS_HINT,
    CapabilityGate,
    apply_capability,
)


def clean_signals() -> dict:
    """全绿的 13 Q_* 信号（无任何关闭/降级触发）。"""
    return {
        "Q_det": 0.8, "Q_fdet": 0.7, "Q_vis": 20.0, "Q_track": 0.8,
        "Q_ids": None, "Q_interf": 0.05, "Q_fg": 0.05, "Q_calib": True,
        "Q_baseline": True, "Q_pelletloss": 0.02, "Q_n0gap": 0.05,
        "Q_censored": False, "Q_motion": 0.5,
    }


def floating_meta() -> RunMeta:
    return RunMeta(pellet_type="floating", n_fish_total=20)


# ----------------------------------------------------------------------
# 元规则：自动分档只降级不关闭
# ----------------------------------------------------------------------
class TestMetaRuleNoClosure:
    def test_vigor_unknown_no_closure_only_low_sensitivity(self) -> None:
        report = apply_capability(
            clean_signals(), Thresholds(), meta=floating_meta(),
            fa_dynamic_range=None,  # UNKNOWN
        )
        assert report.vigor_tier == "UNKNOWN"
        assert report.disabled_with_reason == []  # 无任何关闭
        assert report.closed_groups == []
        # 只有 low_sensitivity 标记（B1/D2）
        flags = report.metric_flags.get("B1_frame_diff", [])
        assert "low_sensitivity" in flags
        assert report.notes  # 决策留痕，不静默

    def test_density_high_tier_is_advisory_only(self) -> None:
        s = clean_signals()
        s["Q_vis"] = 100.0  # > density_med_max=40 → HIGH
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        assert report.density_tier == "HIGH"
        # 密度档是自动分档：只允许建议，不允许关闭（元规则）
        assert report.disabled_with_reason == []
        assert report.closed_groups == []

    def test_gentle_vigor_flags_but_not_closes(self) -> None:
        report = apply_capability(
            clean_signals(), Thresholds(), meta=floating_meta(),
            fa_dynamic_range=1.0,  # ≤ vigor_ratio_low → GENTLE
        )
        assert report.vigor_tier == "GENTLE"
        assert report.disabled_with_reason == []
        assert "low_sensitivity" in report.metric_flags.get(
            "B1_spatial_heterogeneity", []
        )


# ----------------------------------------------------------------------
# ① 实测质量信号（有权关闭）
# ----------------------------------------------------------------------
class TestQualitySignalClosures:
    def test_q_track_low_closes_c_group_with_hint(self) -> None:
        s = clean_signals()
        s["Q_track"] = 0.3  # < q_track_min=0.5
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        assert "C" in report.closed_groups
        entries = {d.metric: d for d in report.disabled_with_reason}
        assert "C_group" in entries
        # 非随机缺失告警：缺失可能是效应本身（docs/04 §4.3 第 4 类）
        assert entries["C_group"].hint == INFORMATIVE_MISSINGNESS_HINT
        assert "0.3" in entries["C_group"].reason  # 原因含量化数值

    def test_q_baseline_false_closes_rp_only(self) -> None:
        s = clean_signals()
        s["Q_baseline"] = False
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        closed = {d.metric for d in report.disabled_with_reason}
        assert "B2-7_RP" in closed
        # 基线缺失 → B1 只降级不关闭（uncalibrated）
        assert "uncalibrated" in report.metric_flags.get(
            "B1_frame_diff", []
        )
        assert "B1_spatial_heterogeneity" not in closed

    def test_q_interf_high_degrades_b1_only(self) -> None:
        s = clean_signals()
        s["Q_interf"] = 0.5  # > q_interf_max=0.3
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        assert "B1_frame_diff" in report.degrade_reasons
        assert "low_conf" in report.metric_flags.get("B1_frame_diff", [])
        assert report.disabled_with_reason == []  # 反光只降级不关闭


# ----------------------------------------------------------------------
# ②③ 用户元数据 + 漂出率硬规则
# ----------------------------------------------------------------------
class TestHardRules:
    def test_sinking_degrades_t90_t100_rr(self) -> None:
        meta = RunMeta(pellet_type="sinking")
        report = apply_capability(clean_signals(), Thresholds(), meta=meta)
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            assert mid in report.degrade_reasons
            assert "contains_non_feeding_loss" in report.metric_flags.get(
                mid, []
            )
        assert report.disabled_with_reason == []  # 降级为参考值，不关闭

    def test_q_pelletloss_over_threshold_degrades(self) -> None:
        s = clean_signals()
        s["Q_pelletloss"] = 0.25  # > 0.15
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            assert mid in report.degrade_reasons
            assert "contains_non_feeding_loss" in report.metric_flags.get(
                mid, []
            )

    def test_unknown_pellet_type_flag_only(self) -> None:
        report = apply_capability(
            clean_signals(), Thresholds(), meta=RunMeta()
        )  # pellet_type=None
        assert report.disabled_with_reason == []
        for mid in ("A9_T90", "A10_T100"):
            assert "sedimentation_risk_unassessed" in report.metric_flags.get(
                mid, []
            )

    def test_q_calib_false_unnormalized_flags(self) -> None:
        s = clean_signals()
        s["Q_calib"] = False
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        for mid in ("B1_flow_mm_s", "B2-1_ANND", "B2-2_MDC", "B2-4_FIFFB"):
            assert "unnormalized" in report.metric_flags.get(mid, [])

    def test_q_n0gap_denominator_suspect_spread(self) -> None:
        s = clean_signals()
        s["Q_n0gap"] = 0.30  # > 0.20
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        for mid in ("A4_P_end", "A8_T50", "A11_RR", "A13_AUC60"):
            assert "denominator_suspect" in report.metric_flags.get(mid, [])

    def test_outdoor_degrades_b2_not_closes(self) -> None:
        report = apply_capability(
            clean_signals(), Thresholds(), meta=floating_meta(), outdoor=True
        )
        assert report.outdoor is True
        for mid in ("B2-5_N_fz_mean", "B2-6_P_fz", "B2-7_RP"):
            assert mid in report.degrade_reasons
            assert "outdoor_exploratory" in report.metric_flags.get(mid, [])
        assert report.disabled_with_reason == []  # 户外只降级不关闭


# ----------------------------------------------------------------------
# ③b 矛盾三段式（docs/04 §4.5）
# ----------------------------------------------------------------------
class TestContradictionThreeWay:
    def test_floating_plus_measured_loss_uses_measurement(self) -> None:
        """分支一：A14 可用（Q_pelletloss 实测超阈）→ 以实测为准。"""
        s = clean_signals()
        s["Q_pelletloss"] = 0.25
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            assert "metadata_contradiction" in report.metric_flags.get(mid, [])
            # 实测口径同时生效
            assert "contains_non_feeding_loss" in report.metric_flags.get(
                mid, []
            )
        assert any("以实测为准" in n for n in report.notes)

    def test_floating_unverifiable_no_contradiction_no_measured_claim(self) -> None:
        """分支二：A14 不可用（Q_pelletloss=None）→ 不覆盖声明，不冒充实测。"""
        s = clean_signals()
        s["Q_pelletloss"] = None  # A14 不可用
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        # 不得出现 metadata_contradiction（无实测证据即无矛盾判定权）
        for flags in report.metric_flags.values():
            assert "metadata_contradiction" not in flags
        # 应有 sedimentation_risk_unverifiable
        for mid in ("A9_T90", "A10_T100"):
            assert "sedimentation_risk_unverifiable" in report.metric_flags.get(
                mid, []
            )
        # 告警文案不得含"实测"（此时没有实测）
        notes = [n for n in report.notes if "unverifiable" in n or "不可测" in n]
        assert notes, "A14 不可用时必须写告警（不静默）"
        for n in notes:
            assert "实测" not in n
        # 不覆盖声明：不关闭不降级（floating 声明保留）
        assert report.disabled_with_reason == []
        for mid in ("A9_T90", "A10_T100"):
            assert mid not in report.degrade_reasons


# ----------------------------------------------------------------------
# CapabilityGate 类封装（classDiagram 签名兼容）
# ----------------------------------------------------------------------
class TestCapabilityGate:
    def test_apply_delegates_to_apply_capability(self) -> None:
        gate = CapabilityGate()
        report = gate.apply(clean_signals(), meta=floating_meta())
        assert report.vigor_tier == "UNKNOWN"
        assert report.disabled_with_reason == []

    def test_user_vigor_override_takes_precedence(self) -> None:
        gate = CapabilityGate()
        report = gate.apply(
            clean_signals(), meta=floating_meta(), vigor_tier="GENTLE"
        )
        assert report.vigor_tier == "GENTLE"  # 用户声明优先于自动 UNKNOWN

    def test_camera_style_outdoor_sets_outdoor(self) -> None:
        gate = CapabilityGate()
        report = gate.apply(
            clean_signals(), meta=floating_meta(),
            camera_style="outdoor_oblique",
        )
        assert report.outdoor is True
        assert "outdoor_exploratory" in report.metric_flags.get(
            "B2-5_N_fz_mean", []
        )


# ----------------------------------------------------------------------
# FLAG_GLOSSARY 完整性（B 类边界测试 B8：flag 必须可查释义）
# ----------------------------------------------------------------------
class TestFlagGlossary:
    def test_all_blocking_flags_documented(self) -> None:
        for flag in BLOCKING_FLAGS:
            assert flag in FLAG_GLOSSARY, f"阻断 flag 无释义: {flag}"

    def test_status_pseudo_flags_documented(self) -> None:
        for status in ("ok", "degraded", "unavailable", "censored"):
            assert f"status:{status}" in FLAG_GLOSSARY

    def test_contradiction_flags_documented(self) -> None:
        assert "metadata_contradiction" in FLAG_GLOSSARY
        assert "sedimentation_risk_unverifiable" in FLAG_GLOSSARY

    def test_to_dict_roundtrip_fields(self) -> None:
        report = apply_capability(
            clean_signals(), Thresholds(), meta=floating_meta()
        )
        d = report.to_dict()
        assert d["vigor_tier"] == "UNKNOWN"
        assert isinstance(d["disabled_with_reason"], list)
        assert isinstance(d["notes"], list)


# ----------------------------------------------------------------------
# 决策留痕：覆盖必须写 notes（绝不静默省略）
# ----------------------------------------------------------------------
class TestDecisionTraceability:
    def test_every_closure_and_degrade_has_reason(self) -> None:
        s = clean_signals()
        s["Q_track"] = 0.2
        s["Q_baseline"] = False
        s["Q_pelletloss"] = 0.30
        report = apply_capability(s, Thresholds(), meta=floating_meta())
        for entry in report.disabled_with_reason:
            assert entry.reason, f"关闭无原因: {entry.metric}"
        for mid, reason in report.degrade_reasons.items():
            assert reason, f"降级无原因: {mid}"

    def test_notes_written_for_every_decision_branch(self) -> None:
        # vigor UNKNOWN → notes
        r1 = apply_capability(clean_signals(), Thresholds(), meta=floating_meta())
        assert any("UNKNOWN" in n for n in r1.notes)
        # 沉性料 → notes
        r2 = apply_capability(
            clean_signals(), Thresholds(), meta=RunMeta(pellet_type="sinking")
        )
        assert any("沉性料" in n for n in r2.notes)
        # 矛盾分支一 → notes
        s = clean_signals()
        s["Q_pelletloss"] = 0.25
        r3 = apply_capability(s, Thresholds(), meta=floating_meta())
        assert any("矛盾" in n for n in r3.notes)

"""RunMeta 交叉校验测试（T02 验收 3：元数据质量门 + 八项反查）。

覆盖：
    - 必填字段门：缺项 → MetaValidationError（报字段名，拒绝启动）；
    - 八项交叉检查逐项触发与不触发（含 n_fish vs Q_vis 单向纪律）；
    - A14 不可靠 → sedimentation_risk_unverifiable（无法验证 ≠ 测到了）。
"""
from __future__ import annotations

import pytest

from src.core.frame_context import RunMeta
from src.pipeline.meta_validation import (
    REQUIRED_META_FIELDS,
    MetaValidationError,
    cross_validate,
    validate_required,
)


def _full_meta(**overrides) -> RunMeta:
    """五项必填齐备的合法元数据。"""
    base = dict(
        species="草鱼",
        n_fish_total=50,
        feed_mass_g=200.0,
        pellet_mass_mg=150.0,
        pellet_type="floating",
        body_length_mm=300.0,
    )
    base.update(overrides)
    return RunMeta(**base)


def _codes(warnings: list) -> set[str]:
    return {w.code for w in warnings}


# ----------------------------------------------------------------------
# 必填字段门
# ----------------------------------------------------------------------
class Test必填字段门:

    def test_meta为None_报全部字段(self) -> None:
        with pytest.raises(MetaValidationError) as ei:
            validate_required(None)
        assert set(ei.value.missing_fields) == set(REQUIRED_META_FIELDS)

    def test_缺两个字段_报字段名(self) -> None:
        meta = _full_meta(species=None, feed_mass_g=None)
        with pytest.raises(MetaValidationError) as ei:
            validate_required(meta)
        assert set(ei.value.missing_fields) == {"species", "feed_mass_g"}
        # 异常消息包含字段名（用户可据此补录）
        assert "species" in str(ei.value)

    def test_齐备_通过(self) -> None:
        validate_required(_full_meta())


# ----------------------------------------------------------------------
# 八项交叉校验
# ----------------------------------------------------------------------
class Test交叉校验:

    def test_实测干净_无告警(self) -> None:
        w = cross_validate(
            _full_meta(),
            {
                "Q_pelletloss": 0.02, "a14_reliable": True,
                "N0_det": 1330, "N0_meta": 1333.3,
                "Q_vis": 40, "bodylen_measured_mm": 310.0,
                "px_per_mm_ref": 3.2, "px_per_mm_check": 3.15,
                "baseline_pellet_count_mean": 0.0,
            },
        )
        assert w == []

    # ---- 1. pellet_type vs Q_pelletloss（三段式）----
    def test_浮性料但高非摄食损失_A14可靠_矛盾告警(self) -> None:
        w = cross_validate(
            _full_meta(pellet_type="floating"),
            {"Q_pelletloss": 0.30, "a14_reliable": True},
        )
        assert "pellet_type_contradiction" in _codes(w)
        crit = [x for x in w if x.code == "pellet_type_contradiction"][0]
        assert crit.severity == "critical"

    def test_高损失但A14不可靠_不覆盖声明(self) -> None:
        w = cross_validate(
            _full_meta(pellet_type="floating"),
            {"Q_pelletloss": 0.30, "a14_reliable": False},
        )
        codes = _codes(w)
        assert "sedimentation_risk_unverifiable" in codes
        assert "pellet_type_contradiction" not in codes  # 无法验证 ≠ 测到了

    def test_声明沉性料_降级提醒(self) -> None:
        w = cross_validate(
            _full_meta(pellet_type="sinking"),
            {"Q_pelletloss": 0.02, "a14_reliable": True},
        )
        assert "sinking_declared" in _codes(w)

    def test_损失低于阈值_不告警(self) -> None:
        w = cross_validate(
            _full_meta(), {"Q_pelletloss": 0.05, "a14_reliable": True},
        )
        assert "pellet_type_contradiction" not in _codes(w)

    # ---- 2. N0 双轨偏差 ----
    def test_N0双轨偏差超20_告警(self) -> None:
        w = cross_validate(_full_meta(), {"N0_det": 900, "N0_meta": 1333})
        assert "n0_gap" in _codes(w)

    def test_N0双轨一致_不告警(self) -> None:
        w = cross_validate(_full_meta(), {"N0_det": 1300, "N0_meta": 1333})
        assert "n0_gap" not in _codes(w)

    # ---- 3. n_fish_total vs Q_vis（单向）----
    def test_可见鱼数超总数_告警(self) -> None:
        w = cross_validate(_full_meta(), {"Q_vis": 60.0})
        assert "fish_count_contradiction" in _codes(w)

    def test_可见鱼数少于总数_正常遮挡不告警(self) -> None:
        w = cross_validate(_full_meta(), {"Q_vis": 12.0})
        assert "fish_count_contradiction" not in _codes(w)

    # ---- 4. 体长 ----
    def test_体长偏差超20_告警(self) -> None:
        w = cross_validate(_full_meta(), {"bodylen_measured_mm": 500.0})
        assert "bodylen_contradiction" in _codes(w)

    def test_体长接近_不告警(self) -> None:
        w = cross_validate(_full_meta(), {"bodylen_measured_mm": 290.0})
        assert "bodylen_contradiction" not in _codes(w)

    # ---- 5. 尺度自检 ----
    def test_尺度自检偏差超10_告警(self) -> None:
        w = cross_validate(
            _full_meta(),
            {"px_per_mm_ref": 3.2, "px_per_mm_check": 2.5},
        )
        assert "calib_selfcheck_failed" in _codes(w)

    def test_尺度自检接近_不告警(self) -> None:
        w = cross_validate(
            _full_meta(),
            {"px_per_mm_ref": 3.2, "px_per_mm_check": 3.1},
        )
        assert "calib_selfcheck_failed" not in _codes(w)

    # ---- 6. N0_meta 合理性 ----
    def test_N0_meta荒谬_告警(self) -> None:
        w = cross_validate(
            _full_meta(feed_mass_g=1000.0, pellet_mass_mg=1.0), {},
        )
        assert "n0_meta_absurd" in _codes(w)

    # ---- 7. 基线颗粒计数 ----
    def test_基线有颗粒_t0可疑(self) -> None:
        w = cross_validate(_full_meta(), {"baseline_pellet_count_mean": 4.2})
        assert "t0_suspect" in _codes(w)

    def test_基线干净_不告警(self) -> None:
        w = cross_validate(_full_meta(), {"baseline_pellet_count_mean": 0.0})
        assert "t0_suspect" not in _codes(w)

    # ---- 8. timing 体检附检 ----
    def test_时间轴体检告警(self) -> None:
        w = cross_validate(None, {"timing_suspect": True, "timeline_discontinuity": True})
        codes = _codes(w)
        assert "timing_suspect" in codes
        assert "timeline_discontinuity" in codes

    def test_告警结构_三要素可序列化(self) -> None:
        w = cross_validate(_full_meta(), {"Q_vis": 60.0})
        d = w[0].to_dict()
        assert set(d) == {"code", "severity", "message"}
        assert len(d["message"]) > 10

"""MetricValue 构造期约束测试（T01，B 类边界测试：测"有没有撒谎"）。

验收对应 docs/06 §6 T01 验收标准 2：
    - 非 ok 无 reason 抛错；
    - unavailable 带 value 抛错；
    - mm 量纲缺 Q_calib 抛错；
    - BL 的 meta_mm 路径缺 Q_calib 抛错。
另覆盖：censored 带 value 抛错、BL 缺 Q_bodylen 抛错、NaN 拒绝、
status/unit_scale Literal 校验、blocking_flags 伪 flag 体系。
"""
from __future__ import annotations

import pytest

from src.core.metric_value import BLOCKING_FLAGS, MetricValue


def _ok(**kwargs) -> MetricValue:
    """合法 ok 基准（除被测字段外全部合法）。"""
    base = dict(
        metric_id="TEST_metric",
        value=1.0,
        unit="s",
        status="ok",
        reason=None,
        flags=(),
        quality={},
        unit_scale="none",
    )
    base.update(kwargs)
    return MetricValue(**base)


# ======================================================================
# 验收 2 · 四条 B 类边界
# ======================================================================
class TestReasonRequired:
    def test_degraded_without_reason_raises(self):
        with pytest.raises(ValueError, match="非 ok 状态必须给出 reason"):
            _ok(status="degraded", reason=None)

    def test_unavailable_without_reason_raises(self):
        with pytest.raises(ValueError, match="非 ok 状态必须给出 reason"):
            _ok(status="unavailable", value=None, reason=None)

    def test_censored_without_reason_raises(self):
        with pytest.raises(ValueError, match="非 ok 状态必须给出 reason"):
            _ok(status="censored", value=None, reason=None)

    def test_ok_with_reason_is_allowed(self):
        # ok 带 reason 合法（不算错，只是冗余）
        mv = _ok(reason="记录性说明")
        assert mv.status == "ok"


class TestUnavailableWithValue:
    def test_unavailable_with_value_raises(self):
        with pytest.raises(ValueError, match="不可用/删失时 value 必须为 None"):
            _ok(status="unavailable", value=0.0, reason="A_single 未标定")

    def test_unavailable_with_zero_raises(self):
        # 0 是最危险的冒充形态（0 有数值意义）
        with pytest.raises(ValueError, match="不可用/删失时 value 必须为 None"):
            _ok(status="unavailable", value=0, reason="基线缺失")

    def test_censored_with_value_raises(self):
        with pytest.raises(ValueError, match="不可用/删失时 value 必须为 None"):
            _ok(status="censored", value=300.0, reason="观察窗内未穿越")

    def test_unavailable_with_none_ok(self):
        mv = _ok(status="unavailable", value=None, reason="ROI.pellet_zone 未定义")
        assert mv.value is None


class TestMmScaleRequiresCalib:
    def test_mm_without_q_calib_raises(self):
        with pytest.raises(ValueError, match="未标定不得输出 mm 量纲"):
            _ok(unit="mm/s", unit_scale="mm")

    def test_mm_with_q_calib_false_raises(self):
        with pytest.raises(ValueError, match="未标定不得输出 mm 量纲"):
            _ok(unit="mm/s", unit_scale="mm", quality={"Q_calib": False})

    def test_mm_with_q_calib_missing_raises(self):
        # quality 里根本没有 Q_calib 键 → 同样拒绝
        with pytest.raises(ValueError, match="未标定不得输出 mm 量纲"):
            _ok(unit="mm/s", unit_scale="mm", quality={"Q_det": 0.8})

    def test_mm_with_q_calib_true_ok(self):
        mv = _ok(unit="mm/s", unit_scale="mm", quality={"Q_calib": True})
        assert mv.unit_scale == "mm"


class TestBlScaleBranches:
    def test_bl_meta_mm_without_q_calib_raises(self):
        """验收 2 第 4 条：BL 量纲，体长来自元数据(mm) 时缺 Q_calib 抛错。"""
        with pytest.raises(ValueError, match="体长来自元数据\\(mm\\)时 BL 换算依赖 Q_calib"):
            _ok(
                unit="BL/s",
                unit_scale="BL",
                quality={"Q_bodylen": True, "bodylen_source": "meta_mm"},
            )

    def test_bl_meta_mm_with_q_calib_ok(self):
        mv = _ok(
            unit="BL/s",
            unit_scale="BL",
            quality={
                "Q_bodylen": True,
                "bodylen_source": "meta_mm",
                "Q_calib": True,
            },
        )
        assert mv.unit_scale == "BL"

    def test_bl_without_q_bodylen_raises(self):
        """docs/04 §7.1 B7：体长不可得（Q_bodylen 非 True）→ BL 构造直接抛错。"""
        with pytest.raises(ValueError, match="体长不可得不得输出 BL 量纲"):
            _ok(unit="BL/s", unit_scale="BL")

    def test_bl_q_bodylen_false_raises(self):
        with pytest.raises(ValueError, match="体长不可得不得输出 BL 量纲"):
            _ok(
                unit="BL/s",
                unit_scale="BL",
                quality={"Q_bodylen": False, "Q_calib": True},
            )

    def test_bl_measured_source_without_q_calib_ok(self):
        # 实测（画面内标定）体长路径：只要求 Q_bodylen，不要求 Q_calib
        mv = _ok(
            unit="BL/s",
            unit_scale="BL",
            quality={"Q_bodylen": True, "bodylen_source": "measured"},
        )
        assert mv.unit_scale == "BL"


# ======================================================================
# 追加校验与伪 flag 体系
# ======================================================================
class TestLiteralAndNaN:
    def test_illegal_status_raises(self):
        with pytest.raises(ValueError, match="status 必须为"):
            _ok(status="fine")

    def test_illegal_unit_scale_raises(self):
        with pytest.raises(ValueError, match="unit_scale 必须为"):
            _ok(unit_scale="cm")

    def test_nan_value_raises(self):
        import math

        with pytest.raises(ValueError, match="NaN"):
            _ok(value=math.nan)

    def test_frozen(self):
        mv = _ok()
        with pytest.raises(Exception):
            mv.value = 2.0  # type: ignore[misc]  # frozen dataclass 不可变


class TestBlockingFlags:
    def test_ok_no_flags_empty(self):
        mv = _ok(flags=())
        assert mv.blocking_flags() == []

    def test_ok_nonblocking_flags_filtered(self):
        # unnormalized 不在阻断清单（跨 run 比较由 compare.py 拦截）
        mv = _ok(flags=("unnormalized", "low_conf_marker_not_real"))
        assert mv.blocking_flags() == []

    def test_blocking_flag_hit(self):
        mv = _ok(flags=("contains_non_feeding_loss", "unnormalized"))
        assert mv.blocking_flags() == ["contains_non_feeding_loss"]

    def test_status_pseudo_flag(self):
        mv = _ok(status="degraded", reason="Q_det=0.4 < 0.50")
        assert mv.blocking_flags() == ["status:degraded"]

    def test_status_pseudo_flag_plus_blocking(self):
        mv = _ok(
            status="degraded",
            reason="Q_n0gap=0.31 > 0.20",
            flags=("denominator_suspect",),
        )
        assert mv.blocking_flags() == ["denominator_suspect", "status:degraded"]

    def test_all_documented_blocking_flags_constant(self):
        # docs/04 §6.1.2 ① 阻断清单恰 8 项
        assert len(BLOCKING_FLAGS) == 8
        expected = {
            "contains_non_feeding_loss",
            "denominator_suspect",
            "sedimentation_risk_unassessed",
            "counting_unstable",
            "unstable_baseline",
            "low_conf",
            "degenerate",
            "fish_count_confounded",
        }
        assert BLOCKING_FLAGS == expected


class TestToDict:
    def test_roundtrip_dict(self):
        mv = _ok(
            metric_id="A8_T50",
            status="censored",
            value=None,
            reason="观察窗内未穿越",
            flags=("contains_non_feeding_loss",),
            quality={"Q_pelletloss": 0.2},
        )
        d = mv.to_dict()
        assert d["metric_id"] == "A8_T50"
        assert d["value"] is None
        assert d["flags"] == ["contains_non_feeding_loss"]
        assert d["quality"] == {"Q_pelletloss": 0.2}

"""MetricValue · 全局约束点（T01）。

职责：
    所有对外输出的数值指标必须经由 MetricValue 构造。构造期强制校验使
    "写错了就构造不出来"（docs/06 §3.1，11 轮交叉核验冻结的产物）：
      1. status != 'ok' 必须给出 reason；
      2. unavailable / censored 时 value 必须为 None（严禁 0/NaN 冒充）；
      3. 量纲依赖链：mm 需要 Q_calib；BL 需要 Q_bodylen，
         且 bodylen_source == 'meta_mm' 时额外依赖 Q_calib。

阻断 flag 体系（docs/04 §6.1.2）：
    blocking_flags() 返回命中阻断清单（8 项）的 flag；status != 'ok' 时
    额外包含伪 flag "status:<值>"，保证单一列即可完成筛选。

⚠️ 逐字实现契约（docs/06 §3.1 / docs/04 §3.1）：字段与构造期校验逻辑
    不得被"优化"或"简化"。允许的追加校验（Literal 成员检查 / NaN 拒绝）
    均以注释标明。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["MetricValue", "BLOCKING_FLAGS"]

# 阻断清单（8 项，内容类；docs/04 §6.1.2 ①）。
# 版本随 metrics_spec_version 统一管理（不单独设 blocking_flags_version）。
BLOCKING_FLAGS: frozenset[str] = frozenset(
    {
        "contains_non_feeding_loss",  # Q_pelletloss > 0.15 或 pellet_type == 'sinking'
        "denominator_suspect",         # Q_n0gap > 0.20
        "sedimentation_risk_unassessed",  # pellet_type == UNKNOWN
        "counting_unstable",           # N_p 非单调上升段占比 > 10%
        "unstable_baseline",           # 基线变异系数 > 0.5
        "low_conf",                    # 依赖的检测/跟踪质量低于门限
        "degenerate",                  # 几何退化（如 FIFFB 质心重复率超阈值）
        "fish_count_confounded",       # Q_overlap 超阈值（鱼数与因变量混淆）
    }
)

_ALLOWED_STATUS = ("ok", "degraded", "unavailable", "censored")
_ALLOWED_UNIT_SCALE = ("px", "mm", "BL", "none")


@dataclass(frozen=True)
class MetricValue:
    """单个指标的对外输出结构（不可变）。

    Attributes:
        metric_id: 指标编号，如 "A8_T50"、"B2-7_RP"。
        value: 数值；None 表示不可用/删失，绝不是 0。
        unit: 单位字符串，如 "s"、"%"、"颗/s"、"BL/s"；可为 None（无量纲）。
        status: 'ok' | 'degraded' | 'unavailable' | 'censored'。
        reason: status != 'ok' 时必填，否则构造期抛 ValueError。
        flags: 全部 flag（含非阻断），元组形式。
        quality: 引用 13 个 Q_* 质量信号及派生键（如 bodylen_source）。
        unit_scale: 'px' | 'mm' | 'BL' | 'none'，量纲归一化层级。
    """

    metric_id: str
    value: float | None
    unit: str | None
    status: str
    reason: str | None
    flags: tuple[str, ...] = ()
    quality: dict[str, Any] = field(default_factory=dict)
    unit_scale: str = "none"

    def __post_init__(self) -> None:
        # （追加校验，非契约原文：Literal 类型标注的运行期落实）
        if self.status not in _ALLOWED_STATUS:
            raise ValueError(
                f"status 必须为 {'/'.join(_ALLOWED_STATUS)} 之一，收到: {self.status!r}"
            )
        if self.unit_scale not in _ALLOWED_UNIT_SCALE:
            raise ValueError(
                f"unit_scale 必须为 {'/'.join(_ALLOWED_UNIT_SCALE)} 之一，"
                f"收到: {self.unit_scale!r}"
            )
        # （追加校验，落实契约 §0.3"严禁 0/NaN 冒充"：NaN 在任何 status 下都不允许）
        if self.value is not None and isinstance(self.value, float) and math.isnan(
            self.value
        ):
            raise ValueError("value 不允许为 NaN（不可用请用 status + reason 表达）")

        if self.status != "ok" and not self.reason:
            raise ValueError("非 ok 状态必须给出 reason")
        if self.status in ("unavailable", "censored") and self.value is not None:
            raise ValueError("不可用/删失时 value 必须为 None，严禁 0/NaN 冒充")
        self._check_unit_scale()

    def _check_unit_scale(self) -> None:
        """量纲依赖链校验（契约原文，逐字实现）。"""
        q = self.quality
        if self.unit_scale == "mm" and q.get("Q_calib") is not True:
            raise ValueError("未标定不得输出 mm 量纲")
        if self.unit_scale == "BL":
            if q.get("Q_bodylen") is not True:
                raise ValueError("体长不可得不得输出 BL 量纲")
            src = q.get("bodylen_source")
            if src == "meta_mm" and q.get("Q_calib") is not True:
                raise ValueError("体长来自元数据(mm)时 BL 换算依赖 Q_calib")

    def blocking_flags(self) -> list[str]:
        """返回命中阻断清单的 flag（docs/04 §6.1.2）。

        status != 'ok' 时额外追加伪 flag "status:<值>"，保证：
          - 单一列即可完成"干净"筛选（blocking_flags == [] 等价 ok 且无阻断）；
          - 非 ok 指标的 blocking_flag_count 恒 ≥ 1，不会伪装成"最干净"。
        """
        out = [f for f in self.flags if f in BLOCKING_FLAGS]
        if self.status != "ok":
            out.append(f"status:{self.status}")
        return out

    def to_dict(self) -> dict[str, Any]:
        """序列化为 summary.json 兼容的 plain dict（flags 转 list）。"""
        return {
            "metric_id": self.metric_id,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "reason": self.reason,
            "flags": list(self.flags),
            "quality": dict(self.quality),
            "unit_scale": self.unit_scale,
        }

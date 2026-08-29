"""RunMeta 交叉校验（T02）：元数据质量门 + 八项实测反查。

职责（docs/04 §4.5 硬规则 7 + 管线顺序约束）：
    管线顺序 = ingest → 检测/跟踪 → 第一遍指标 → ★ meta_validation →
    其余指标 → 输出。本模块消费"第一遍实测值"反查用户元数据：
      1. 必填字段门（缺项拒绝启动，验收标准 3）；
      2. pellet_type vs Q_pelletloss（须先过 A14 质量门，三段式话术）；
      3. feed_mass_g + pellet_mass_mg vs N0_det（Q_n0gap）；
      4. n_fish_total vs Q_vis —— **单向**：仅 Q_vis > n_fish_total 告警
         （反向是正常遮挡，双向校验会持续误报让用户学会忽略告警）；
      5. body_length_mm vs 画面实测体长；
      6. px_per_mm_ref vs 已知尺寸参照物自检；
      7. N0_meta 合理性（<1 或 >1e5）；
      8. 基线 pellet_count_mean ≠ 0 → t0 判定可疑；
      （附）timing_suspect / timeline_discontinuity 提示。

纪律：
    - 任何"自动覆盖用户元数据"的行为只出现在返回的 warnings（含
      severity + 排查引导），由 capability/报告层呈现——本模块不改写
      meta 本身（留痕不静默）；
    - A14 不可靠时**不得**声称"按实测覆盖声明"，只能输出
      sedimentation_risk_unverifiable 类告警（无法验证 ≠ 测到了）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.core.frame_context import RunMeta

__all__ = [
    "REQUIRED_META_FIELDS",
    "MetaValidationError",
    "MetaWarning",
    "validate_required",
    "cross_validate",
]

# 必填元数据字段（验收标准 3：缺必填 → 拒绝启动且报字段名）
REQUIRED_META_FIELDS: tuple[str, ...] = (
    "species",        # 鱼种
    "n_fish_total",   # 总尾数
    "feed_mass_g",    # 投喂量
    "pellet_mass_mg", # 单颗均重
    "pellet_type",    # 饲料类型
)

Severity = Literal["info", "warn", "critical"]


class MetaValidationError(ValueError):
    """必填元数据缺失 → 拒绝启动（构造期门，不是告警）。"""

    def __init__(self, missing_fields: list[str]) -> None:
        self.missing_fields = missing_fields
        super().__init__(
            "RunMeta 缺必填字段，拒绝启动: "
            + ", ".join(missing_fields)
            + "（必填: " + ", ".join(REQUIRED_META_FIELDS) + "）"
        )


@dataclass
class MetaWarning:
    """交叉校验告警（写入 warnings 数组与 capability_report.md）。"""

    code: str
    severity: Severity
    message: str          # 含"改了什么/为什么/怎么办"三要素

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


def validate_required(meta: RunMeta | None) -> None:
    """必填字段门：缺失 → 抛 MetaValidationError（拒绝启动）。"""
    if meta is None:
        raise MetaValidationError(list(REQUIRED_META_FIELDS))
    missing = [f for f in REQUIRED_META_FIELDS if getattr(meta, f) is None]
    if missing:
        raise MetaValidationError(missing)


def cross_validate(
    meta: RunMeta | None,
    measured: dict[str, Any],
    pelletloss_degrade: float = 0.15,
    n0_gap_warn: float = 0.20,
    bodylen_dev_warn: float = 0.20,
) -> list[MetaWarning]:
    """八项交叉校验（+timing 附检）。纯函数，不改写 meta。

    Args:
        meta: 用户元数据（None = 仅 timing 附检）。
        measured: 第一遍实测值字典，支持键（全部可选，缺键跳过对应检查）：
            - Q_pelletloss: 非摄食损失率（A14 产出）
            - a14_reliable: A14 是否可靠（ok；False/缺失 = 质量门未过）
            - N0_det / N0_meta: 双轨初始颗粒数
            - Q_vis: 平均可见鱼数
            - bodylen_measured_mm: 画面内实测体长（mm）
            - px_per_mm_ref: 标定尺度；px_per_mm_check: 参照物自检值
            - baseline_pellet_count_mean: 基线期颗粒计数均值
            - timing_suspect / timeline_discontinuity: 时间轴体检
        各告警阈值与 run_config 同源（默认值 = docs/04 §5）。
    """
    out: list[MetaWarning] = []
    m = measured or {}

    # ---- 1. pellet_type vs Q_pelletloss（三段式，须先过 A14 质量门）----
    q_pelletloss = m.get("Q_pelletloss")
    a14_ok = bool(m.get("a14_reliable", False))
    if q_pelletloss is not None and meta is not None:
        if q_pelletloss > pelletloss_degrade:
            if a14_ok:
                if meta.pellet_type == "floating":
                    out.append(
                        MetaWarning(
                            "pellet_type_contradiction",
                            "critical",
                            "实测显示存在显著非摄食损失（Q_pelletloss="
                            f"{q_pelletloss:.2f} > {pelletloss_degrade}），与声明的"
                            "浮性料矛盾，已按实测处理。若您确认饲料为浮性，请检查"
                            " ROI.bottom_band 是否覆盖了排水口/溢流区（浮性料高损失"
                            " 的常见原因是排水口被误判为沉降）",
                        )
                    )
            else:
                out.append(
                    MetaWarning(
                        "sedimentation_risk_unverifiable",
                        "critical",
                        "A14 不可靠（partial_window/coarse/关联未建立），无法验证"
                        "沉降损失与饲料类型声明是否矛盾：不覆盖声明，也不输出干净的"
                        " T90/T100，按保守分支处理（无法验证 ≠ 测到了）",
                    )
                )
        if a14_ok and meta.pellet_type == "sinking":
            out.append(
                MetaWarning(
                    "sinking_declared",
                    "warn",
                    "声明沉性料：T90/T100 强制降级为参考值（沉降与摄食不可分），"
                    "必须与 A14 并列展示",
                )
            )

    # ---- 2. N0 双轨交叉校验（Q_n0gap）----
    n0_det = m.get("N0_det")
    n0_meta = m.get("N0_meta")
    if n0_det is not None and n0_meta is not None and n0_meta > 0:
        gap = abs(float(n0_det) - float(n0_meta)) / float(n0_meta)
        if gap > n0_gap_warn:
            out.append(
                MetaWarning(
                    "n0_gap",
                    "critical",
                    f"N₀ 双轨偏差 {gap:.1%} > {n0_gap_warn:.0%}（N0_det={n0_det:.0f} vs "
                    f"N0_meta={n0_meta:.0f}）：denominator_suspect，可能为密集粘连"
                    "漏检或投喂量口径不符；以 N0 双轨取大者为主并复核早期密集帧",
                )
            )

    # ---- 3. n_fish_total vs Q_vis（单向！）----
    q_vis = m.get("Q_vis")
    if q_vis is not None and meta is not None and meta.n_fish_total is not None:
        if q_vis > meta.n_fish_total:
            out.append(
                MetaWarning(
                    "fish_count_contradiction",
                    "critical",
                    f"平均可见鱼数 Q_vis={q_vis:.1f} > 声明总数 n_fish_total="
                    f"{meta.n_fish_total}：denominator_suspect（污染 P_fz 与 RP 分母）。"
                    "最可能原因是检测器重复计数/误检杂物，请复核检测器而非鱼数"
                    "（反向 Q_vis < 总数是正常遮挡，不告警）",
                )
            )

    # ---- 4. body_length_mm vs 实测体长 ----
    bl = m.get("bodylen_measured_mm")
    if bl is not None and meta is not None and meta.body_length_mm is not None:
        dev = abs(bl - meta.body_length_mm) / meta.body_length_mm
        if dev > bodylen_dev_warn:
            out.append(
                MetaWarning(
                    "bodylen_contradiction",
                    "warn",
                    f"画面实测体长 {bl:.1f}mm 与声明 {meta.body_length_mm:.1f}mm 偏差 "
                    f"{dev:.1%} > {bodylen_dev_warn:.0%}：Q_bodylen 来源可疑，"
                    "BL 量纲指标降级（实测优先，已告知覆盖）",
                )
            )

    # ---- 5. px_per_mm 参照物自检 ----
    pxr = m.get("px_per_mm_ref")
    pxc = m.get("px_per_mm_check")
    if pxr is not None and pxc is not None and pxc > 0:
        dev = abs(pxr - pxc) / pxc
        if dev > 0.10:
            out.append(
                MetaWarning(
                    "calib_selfcheck_failed",
                    "critical",
                    f"尺度标定自检偏差 {dev:.1%} > 10%：Q_calib=False，所有 mm/BL "
                    "量纲指标降级（实测优先，已告知覆盖）",
                )
            )

    # ---- 6. N0_meta 合理性 ----
    if meta is not None and meta.feed_mass_g is not None and meta.pellet_mass_mg is not None:
        n0m = meta.feed_mass_g * 1000.0 / meta.pellet_mass_mg
        if n0m < 1.0 or n0m > 1e5:
            out.append(
                MetaWarning(
                    "n0_meta_absurd",
                    "critical",
                    f"N0_meta = {n0m:.2f} 颗不合理：feed_mass_g 与 pellet_mass_mg "
                    "组合请复核（口径可能不一致）",
                )
            )

    # ---- 7. 基线期颗粒计数非零 → t0 可疑 ----
    bp = m.get("baseline_pellet_count_mean")
    if bp is not None and abs(bp) > 1e-9:
        out.append(
            MetaWarning(
                "t0_suspect",
                "critical",
                f"基线期 pellet_count_mean={bp:.2f} ≠ 0：t0 判定可能有误"
                "（投喂前不应有饲料），复核 t0 打点",
            )
        )

    # ---- 8. timing 体检附检 ----
    if m.get("timing_suspect"):
        out.append(
            MetaWarning(
                "timing_suspect",
                "warn",
                "时间轴体检 timing_suspect=True：若为 VFR 录像，所有时间类指标"
                "可能存在系统性偏差",
            )
        )
    if m.get("timeline_discontinuity"):
        out.append(
            MetaWarning(
                "timeline_discontinuity",
                "critical",
                "时间轴断裂（拼接/续录）：断裂点之后的时间指标一律不输出",
            )
        )
    return out

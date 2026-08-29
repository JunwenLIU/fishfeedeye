"""capability.py · 质量分驱动的降级矩阵（T04）。

职责（docs/04 §4 + docs/06 T04 内联约定）：
    - **元规则（docs/04 §4.0，优先于一切分档）**：任何*自动推断*的分类
      （摄食烈度档、机位类型、密度档等）只被允许追加 ⚠️ 标记，绝不允许
      触发 ❌ 关闭。只有两类东西有权关闭指标：
        ① 实测质量信号 Q_*；
        ② 用户显式提供的元数据（RunMeta 非空字段）。
    - 降级矩阵（docs/04 §4.2 的 T04 子集 + team-lead 派单三规则）：
        * mm 量纲缺标定（Q_calib=False）→ 降像素量纲（unnormalized）；
        * 漂出率 > 15%（Q_pelletloss > pelletloss_degrade）→ T90/RR
          标注 contains_non_feeding_loss；
        * 无基线（Q_baseline=False）→ 相对指标关闭，降绝对值口径；
    - C 组整组门控：Q_track < q_track_min → 整组关闭，且每条
      disabled_with_reason 必须含"该缺失可能是效应本身"的非随机缺失
      告警文案（docs/04 §4.3 ⭐ 第 4 类）；
    - 户外（用户提供/用户确认）→ B2/C 组 degraded（探索性）——只降级。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.config import Thresholds
from src.core.frame_context import RunMeta

__all__ = [
    "CapabilityGate",
    "CapabilityReport",
    "DisabledEntry",
    "apply_capability",
    "INFORMATIVE_MISSINGNESS_HINT",
    "FLAG_GLOSSARY",
]

# 非随机缺失告警（docs/04 §4.3 第 4 类：缺失本身可以是混淆的）。
INFORMATIVE_MISSINGNESS_HINT = (
    "注意：该缺失可能是效应本身——高摄食强度（游速快、运动模糊加剧）"
    "会降低跟踪质量并触发本关闭，故'可用指标集'在组间可能非随机；"
    "跨组统计推断请改用 B1-1/2/3（帧差能量/光流/频谱，门控中立）"
)

# flag 术语表（B 类边界测试 B8：所有对外输出的 flag 必须可在此查到释义；
# 含 8 项阻断 flag、status 伪 flag 与常用非阻断 flag）。
FLAG_GLOSSARY: dict[str, str] = {
    # ---- 阻断清单（docs/04 §6.1.2 ①）----
    "contains_non_feeding_loss": (
        "指标值混入非摄食损失成分（Q_pelletloss > 15% 或沉性料）："
        "须与 A14 并列展示，不可单独作为摄食结论"
    ),
    "denominator_suspect": (
        "N₀ 分母可疑（Q_n0gap > 20%）：以 N₀ 为分母的指标系统性偏差风险"
    ),
    "sedimentation_risk_unassessed": (
        "pellet_type 未确认：沉降与摄食不可分的风险未评估（保守标记，不关闭）"
    ),
    "counting_unstable": (
        "N_p 曲线非单调上升段占比 > 10%：计数回补噪声显著，速率类仅供定性"
    ),
    "unstable_baseline": "基线期变异系数 > 0.5：相对基线指标不稳定",
    "low_conf": "依赖的检测/跟踪质量低于门限",
    "degenerate": "几何退化（如 FIFFB 质心重复率超阈值）",
    "fish_count_confounded": (
        "Q_overlap 超阈值：鱼数与因变量存在混淆（视场内计数不可信）"
    ),
    # ---- status 伪 flag（blocking_flags() 追加项）----
    "status:ok": "指标状态正常（伪 flag，用于 blocking_flags 单列筛选）",
    "status:degraded": "指标已降级为参考值（伪 flag）",
    "status:unavailable": "指标不可用（伪 flag；对应 value 必为 None）",
    "status:censored": "指标右删失（伪 flag；输出 '>窗长'）",
    # ---- 常用非阻断 flag ----
    "censored": "右删失：观察窗内未达阈值，输出 '>窗长' 而非数值",
    "partial_window": (
        "观察窗长于密集关联窗口（a14_window_s）：消失分类仅部分可信"
    ),
    "coarse": "bottom_band 未定义：消失三分类退化为两分",
    "low_sensitivity": (
        "烈度档 UNKNOWN/GENTLE 保守分支：该指标判据灵敏度不足（仅标记，不关闭）"
    ),
    "uncalibrated": "基线缺失：只输出未归一化绝对量，不可跨视频比较",
    "no_noise_correction": (
        "未做参考区噪声扣除：户外波浪可能虚增活跃度"
    ),
    "no_denominator": "分母缺失（如 n_fish_total 未提供）：只输出绝对值",
    "outdoor_exploratory": "户外斜拍场景：探索性输出，不进入主结论",
    "fallback": "主判据不可用，退回次级口径（如 FA→颗粒下降）",
    "window_truncated": "视频短于标称观察窗：窗口截断口径",
    "unnormalized": "未标定：mm 量纲降级为 px（单次分析内可用）",
    "window_fallback": "T50 删失导致搜索窗口退化（[t0, 0.5×窗长]）",
    # ---- 矛盾三段式（docs/04 §4.5）----
    "metadata_contradiction": (
        "用户元数据与实测信号矛盾（如声明浮性料但 Q_pelletloss 超阈）："
        "以实测口径为准，须人工复核投喂记录"
    ),
    "sedimentation_risk_unverifiable": (
        "Q_pelletloss 不可测（A14 未关联）：沉降风险无法验证，"
        "不覆盖用户声明（与 sedimentation_risk_unassessed 的区别："
        "前者是元数据缺项，本 flag 是实测通道缺失）"
    ),
}


@dataclass
class DisabledEntry:
    """一条"被关闭 + 原因"记录（summary.json capability 消费）。"""

    metric: str                       # 指标 ID 或组名（如 'C_group'）
    reason: str                       # 关闭原因（含量化数值）
    hint: str | None = None           # 非随机缺失告警（可选）

    def to_dict(self) -> dict[str, str | None]:
        return {"metric": self.metric, "reason": self.reason, "hint": self.hint}


@dataclass
class CapabilityReport:
    """降级矩阵应用结果。

    Attributes:
        density_tier: 'LOW' | 'MED' | 'HIGH' | None（自动分档，只降级）。
        vigor_tier: 'VIGOROUS' | 'GENTLE' | 'UNKNOWN'（自动，只降级）。
        outdoor: 是否户外斜拍（用户确认；B2/C degraded）。
        closed_groups: 被整组关闭的指标组。
        disabled_with_reason: 逐条关闭记录（显式列出，绝不静默省略）。
        metric_flags: metric_id → 追加 flag 列表（降级不关闭的载体）。
        degrade_reasons: metric_id → 降级 reason（status degraded 用）。
    """

    density_tier: str | None = None
    vigor_tier: str = "UNKNOWN"
    outdoor: bool = False
    closed_groups: list[str] = field(default_factory=list)
    disabled_with_reason: list[DisabledEntry] = field(default_factory=list)
    metric_flags: dict[str, list[str]] = field(default_factory=dict)
    degrade_reasons: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "density_tier": self.density_tier,
            "vigor_tier": self.vigor_tier,
            "outdoor": self.outdoor,
            "closed_groups": list(self.closed_groups),
            "disabled_with_reason": [d.to_dict() for d in self.disabled_with_reason],
            "degradation_flags": {
                k: list(v) for k, v in self.metric_flags.items() if v
            },
            "degrade_reasons": dict(self.degrade_reasons),
            "notes": list(self.notes),
        }


def _density_tier(q_vis: float | None, thresholds: Thresholds) -> str | None:
    if q_vis is None:
        return None
    if q_vis <= thresholds.density_low_max:
        return "LOW"
    if q_vis <= thresholds.density_med_max:
        return "MED"
    return "HIGH"


def _vigor_tier(
    fa_dynamic_range: float | None, thresholds: Thresholds
) -> str:
    """烈度自动分档（仅自动建议值，无权驱动关闭——docs/04 §4.0）。"""
    if fa_dynamic_range is None:
        return "UNKNOWN"
    if fa_dynamic_range > thresholds.vigor_ratio_high:
        return "VIGOROUS"
    if fa_dynamic_range <= thresholds.vigor_ratio_low:
        return "GENTLE"
    return "UNKNOWN"


def apply_capability(
    signals: dict[str, Any],
    thresholds: Thresholds,
    meta: RunMeta | None = None,
    fa_dynamic_range: float | None = None,
    outdoor: bool = False,
) -> CapabilityReport:
    """降级矩阵主入口。

    Args:
        signals: quality.compute_quality_signals 的输出（13 Q_* 全键）。
        thresholds: 阈值常量。
        meta: 用户元数据（pellet_type 等用户显式声明有权关闭/降级）。
        fa_dynamic_range: FA 峰/基比（自动烈度判据；None → UNKNOWN）。
        outdoor: 户外斜拍（用户确认位——用户输入有权降级）。

    Returns:
        CapabilityReport：关闭/降级决策，由 aggregator 应用到 MetricValue。
    """
    report = CapabilityReport(
        density_tier=_density_tier(signals.get("Q_vis"), thresholds),
        vigor_tier=_vigor_tier(fa_dynamic_range, thresholds),
        outdoor=outdoor,
    )

    def _flag(metric: str, flag: str) -> None:
        report.metric_flags.setdefault(metric, []).append(flag)

    def _degrade(metric: str, reason: str) -> None:
        report.degrade_reasons[metric] = reason

    def _close(metric: str, reason: str, hint: str | None = None) -> None:
        report.disabled_with_reason.append(
            DisabledEntry(metric=metric, reason=reason, hint=hint)
        )

    q = signals

    # ---- 元规则先声明（自动分档永不关闭，只追加标记）----
    if report.vigor_tier == "UNKNOWN":
        for mid in ("B1_frame_diff", "B1_spatial_heterogeneity", "D2_T_start"):
            _flag(mid, "low_sensitivity")
        report.notes.append(
            "摄食烈度档 UNKNOWN：按 GENTLE 保守分支加 low_sensitivity 标记"
            "（自动判据无权关闭任何指标，docs/04 §4.0）"
        )
    elif report.vigor_tier == "GENTLE":
        for mid in ("B1_frame_diff", "B1_spatial_heterogeneity"):
            _flag(mid, "low_sensitivity")

    # ---- ① 实测质量信号（有权关闭）----
    q_track = q.get("Q_track")
    if q_track is not None and q_track < thresholds.q_track_min:
        report.closed_groups.append("C")
        _close(
            "C_group",
            reason=(
                f"Q_track = {q_track:.2f} < {thresholds.q_track_min}"
                "，个体级指标（C 组）整组关闭"
            ),
            hint=INFORMATIVE_MISSINGNESS_HINT,
        )
    q_interf = q.get("Q_interf")
    if q_interf is not None and q_interf > thresholds.q_interf_max:
        _degrade(
            "B1_frame_diff",
            f"Q_interf = {q_interf:.1%} > {thresholds.q_interf_max:.0%}"
            "（反光/水花污染），活跃度降级",
        )
        _flag("B1_frame_diff", "low_conf")

    # ---- ② 用户显式元数据（有权关闭/降级）----
    pellet_type = meta.pellet_type if meta is not None else None
    if pellet_type == "sinking":
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            _flag(mid, "contains_non_feeding_loss")
            _degrade(mid, "pellet_type='sinking'（用户声明）：沉降与摄食不可分")
        report.notes.append(
            "用户声明沉性料：T90/T100/RR 降级为参考值"
            "（须与 A14 非摄食损失率并列展示）"
        )
    elif pellet_type is None:
        for mid in ("A9_T90", "A10_T100"):
            _flag(mid, "sedimentation_risk_unassessed")
        report.notes.append(
            "pellet_type 未确认（UNKNOWN）：不关闭不降级，"
            "仅加 sedimentation_risk_unassessed 标记（保守分支）"
        )

    # ---- ③ 漂出率 > 15%（team-lead 派单三规则之二）----
    q_pelletloss = q.get("Q_pelletloss")
    if q_pelletloss is not None and q_pelletloss > thresholds.pelletloss_degrade:
        for mid in ("A9_T90", "A10_T100", "A11_RR"):
            _flag(mid, "contains_non_feeding_loss")
            _degrade(
                mid,
                f"Q_pelletloss = {q_pelletloss:.1%} > "
                f"{thresholds.pelletloss_degrade:.0%}（漂出/沉降损失），"
                "降级为参考值（须与 A14 并列展示）",
            )

    # ---- ③b 矛盾三段式（docs/04 §4.5：实测反查元数据，A14 可用性分支）----
    # 用户声明 floating 但实测漂出超阈时：
    #   分支一（A14 可用，Q_pelletloss 非 None 且超阈）→ 以实测为准：
    #       标 metadata_contradiction（降级不关闭，contains_non_feeding_loss
    #       已由 ③ 给出）；
    #   分支二（A14 不可用，Q_pelletloss=None）→ 沉降风险无法验证：
    #       标 sedimentation_risk_unverifiable，**不覆盖用户声明**，
    #       告警文案不得出现"实测"字样（此时没有实测证据，只有不可测）。
    if pellet_type == "floating":
        if q_pelletloss is not None and q_pelletloss > thresholds.pelletloss_degrade:
            for mid in ("A9_T90", "A10_T100", "A11_RR"):
                _flag(mid, "metadata_contradiction")
            report.notes.append(
                "矛盾（以实测为准）：用户声明浮性料，但实测 Q_pelletloss 超阈"
                "——沉降与摄食不可分的风险实际存在，已按实测口径标注"
                " contains_non_feeding_loss + metadata_contradiction"
            )
        elif q_pelletloss is None:
            for mid in ("A9_T90", "A10_T100"):
                _flag(mid, "sedimentation_risk_unverifiable")
            report.notes.append(
                "Q_pelletloss 不可测（颗粒轨迹未关联，A14 不可用）：浮性料的"
                "沉降风险无法验证（sedimentation_risk_unverifiable），"
                "不覆盖用户声明，请结合无鱼纯饲料视频复核"
            )

    # ---- ④ 无基线 → 相对指标关闭，绝对值口径保留 ----
    q_baseline = q.get("Q_baseline")
    if q_baseline is not True:
        _close(
            "B2-7_RP",
            reason="Q_baseline = False：RP 相对基线指标强制关闭"
            "（绝对值口径 N_fz 保留，标 no_denominator/绝对口径）",
        )
        for mid in ("B1_frame_diff", "B1_spatial_heterogeneity"):
            _flag(mid, "uncalibrated")
            _degrade(
                mid,
                "基线缺失：只输出未归一化绝对量（uncalibrated=True）",
            )

    # ---- ⑤ mm 量纲缺标定 → 降像素量纲（team-lead 派单三规则之一）----
    q_calib = q.get("Q_calib")
    if q_calib is not True:
        for mid in ("B1_flow_mm_s", "B2-1_ANND", "B2-2_MDC", "B2-4_FIFFB"):
            _flag(mid, "unnormalized")
            _degrade(
                mid,
                "Q_calib = False（未标定）：mm 量纲降级为 px 量纲"
                "（unnormalized=True，单次分析内可用，不可跨视频比较）",
            )

    # ---- ⑥ N₀ 分母可疑（denominator_suspect 传染）----
    q_n0gap = q.get("Q_n0gap")
    if q_n0gap is not None and q_n0gap > thresholds.n0_gap_warn:
        for mid in (
            "A4_P_end", "A6_v_max", "A7_v50", "A8_T50", "A9_T90",
            "A10_T100", "A11_RR", "A13_AUC60",
        ):
            _flag(mid, "denominator_suspect")

    # ---- ⑦ 户外（用户确认）→ B2/C degraded（只降级不关闭）----
    if outdoor:
        for mid in ("B2-5_N_fz_mean", "B2-6_P_fz", "B2-7_RP", "C_group"):
            _degrade(mid, "户外斜拍鱼体检测不可行，仅探索性（用户确认）")
            _flag(mid, "outdoor_exploratory")

    return report


# ----------------------------------------------------------------------
# 兼容层（合并双方产出时补齐；apply_capability 核心逻辑不变）
# ----------------------------------------------------------------------
class CapabilityGate:
    """降级矩阵执行器的类封装（classDiagram CapabilityGate 签名兼容）。

    apply(signals, meta, ...) → CapabilityReport，内部委托
    apply_capability。用户显式给出的烈度档（vigor_tier）优先于自动判据：
    通过受控的 fa_dynamic_range 代理值表达（用户元数据有权分档，
    自动判据只建议——docs/04 §4.0）。
    """

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.th = thresholds if thresholds is not None else Thresholds()

    def apply(
        self,
        signals: dict[str, Any],
        meta: RunMeta | None = None,
        vigor_tier: str | None = None,
        fa_dynamic_range: float | None = None,
        camera_style: str = "outdoor_oblique",
    ) -> CapabilityReport:
        """应用降级矩阵（与 apply_capability 同构的类入口）。

        Args:
            signals: quality 层输出的 13 Q_* dict。
            meta: 用户元数据。
            vigor_tier: 用户给出的烈度档（'VIGOROUS'/'GENTLE'）；None=自动。
            fa_dynamic_range: FA 峰/基比（自动烈度判据输入）。
            camera_style: 机位；'outdoor_oblique' → outdoor=True（用户确认
                的默认机位，B2/C 降级为探索性）。
        """
        fa = fa_dynamic_range
        if vigor_tier == "VIGOROUS":
            fa = self.th.vigor_ratio_high + 1.0  # 用户声明优先：表达为高比值
        elif vigor_tier == "GENTLE":
            fa = 1.0  # ≤ vigor_ratio_low → GENTLE
        return apply_capability(
            signals,
            self.th,
            meta=meta,
            fa_dynamic_range=fa,
            outdoor=(camera_style == "outdoor_oblique"),
        )

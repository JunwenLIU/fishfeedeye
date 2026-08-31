"""stats/comparison_plan.py · 比较计划抽象（T05）。

职责（docs/06 §6 T05 内联约定 + docs/04 §4.4）：
    - **ComparisonPlan 抽象**：比较是"一组 run 按某个设计映射到分组、逐指标
      做统计推断"的过程。两组对照是它的一个特例（TwoGroupPlan），剂量梯度
      留扩展位（`DoseResponsePlan` 占位不实现，见 §6 第二轮决策 ①）；
    - **比较前一致性校验（七条拒绝规则）**：任一硬规则不一致 → 拒绝整份
      比较并列出差异，附"重跑对齐约 4 秒"的出路提示；
      ⚠️ **不提供"用户强制比较"的绕过开关**（docs/04 §4.4：覆盖开关会
      诱导用户绕过保护，而这类偏差不可察觉；给一条更好的路，而不是一个
      绕过防护的按钮）；
    - **可用指标集一致性告警**：两组被关闭的指标集不同 → 告警"缺失可能
      非随机"（docs/04 §4.3 第 4 类 + §4.4 规则 6/7）。

七条规则（docs/04 §4.4）：
    1 px_per_mm 不一致 → 拒绝比较任何带 unnormalized 的指标；
    2 metrics_spec_version 不一致 → 拒绝整份比较；
    3 model_md5 不一致 → 拒绝全部比较；
    4 window_s 不一致 或 任一 run window_truncated → 拒绝 RR 与清空时间类；
    5 ROI 不一致（面积差>5% 或 IoU<0.95）→ 告警，不拒绝；
    6 两组已关闭指标集不同 → 告警；
    7 缺失由非中立门控触发 → 拒绝对"可用子集"做统计推断，并给出
      改用 B1-1/2/3 的出路。

任务编号：T05。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.core.config import RunConfig

__all__ = [
    "ComparisonPlan",
    "RunBundle",
    "Rule",
    "Violation",
    "ConsistencyReport",
    "NON_NEUTRAL_GATES",
    "NEUTRAL_GROUP_EXIT",
    "CLEARANCE_LIKE_METRICS",
]

# 非中立门控（docs/04 §4.4 规则 7）：其触发的缺失具信息性，
# 对"可用子集"做统计推断 = 以结果为条件抽样。
NON_NEUTRAL_GATES: frozenset[str] = frozenset(
    {"Q_track", "Q_overlap", "Q_det", "Q_fg"}
)

# 规则 7 触发时给用户的出路（不是"确认继续"按钮，而是一条更好的路）。
NEUTRAL_GROUP_EXIT: str = (
    "请改用 B1-1/2/3（帧差能量 / 光流幅值 / 傅里叶频谱）做跨组比较："
    "它们的门控是外部环境量（Q_interf / Q_motion / Q_calib），"
    "误差不随被测效应变化，不会产生非随机缺失。"
)

# 受规则 4 影响的指标（观察窗口径敏感：RR@300s ≠ RR@180s）
CLEARANCE_LIKE_METRICS: frozenset[str] = frozenset(
    {"A8_T50", "A9_T90", "A10_T100", "A11_RR", "A13_AUC60", "D3_T_end",
     "D4_duration"}
)

# 重跑对齐的成本提示（docs/04 §4.4：拒绝报错必须把用户引向最低成本出路）
RERUN_HINT = "请用当前版本重新分析上述 run（单段约 4 秒）后再比较。"


@dataclass
class RunBundle:
    """参与比较的一个 run 的可比性元数据（不含逐帧数据，轻量）。"""

    run_id: str
    config: RunConfig
    metrics: dict[str, Any] = field(default_factory=dict)   # metric_id -> 行 dict
    disabled: set[str] = field(default_factory=set)          # 被关闭的指标 ID
    quality: dict[str, Any] = field(default_factory=dict)    # Q_* 信号
    window_s: float | None = None
    window_truncated: bool = False
    roi: Any | None = None                                   # ROI 对象（可选）
    group: str = ""
    pond_id: str | None = None                               # 重复结构声明（伪重复）

    def value(self, metric_id: str) -> float | None:
        """取指标数值；不可用/删失 → None（绝不用 0 冒充）。"""
        row = self.metrics.get(metric_id)
        if row is None:
            return None
        v = row.get("value")
        if v in (None, ""):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def status(self, metric_id: str) -> str:
        row = self.metrics.get(metric_id) or {}
        return str(row.get("status", "unavailable"))

    def flags(self, metric_id: str) -> set[str]:
        row = self.metrics.get(metric_id) or {}
        raw = row.get("flags") or ""
        return {f for f in str(raw).split(";") if f}

    def available_metrics(self) -> set[str]:
        """本 run 有可用数值（status ∈ ok/degraded）的指标集。"""
        return {
            mid for mid in self.metrics
            if self.status(mid) in ("ok", "degraded") and self.value(mid) is not None
        }


@dataclass
class Rule:
    """一条比较前一致性规则。"""

    rule_id: int
    name: str
    action: str            # 'reject_all' | 'reject_metrics' | 'warn'
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "action": self.action,
            "description": self.description,
        }


@dataclass
class Violation:
    """一次规则命中。"""

    rule_id: int
    rule_name: str
    action: str
    message: str
    affected_metrics: list[str] = field(default_factory=list)
    exit_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "action": self.action,
            "message": self.message,
            "affected_metrics": list(self.affected_metrics),
            "exit_hint": self.exit_hint,
        }


@dataclass
class ConsistencyReport:
    """比较前一致性校验结果。"""

    violations: list[Violation] = field(default_factory=list)
    rejected_metrics: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """是否允许进行比较（无任何 reject 级命中）。"""
        return not any(v.action.startswith("reject") for v in self.violations)

    def reject_all(self) -> bool:
        """是否被拒绝整份比较。"""
        return any(v.action == "reject_all" for v in self.violations)

    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.action == "warn"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reject_all": self.reject_all(),
            "violations": [v.to_dict() for v in self.violations],
            "rejected_metrics": sorted(self.rejected_metrics),
            "notes": list(self.notes),
        }


class ComparisonPlan(ABC):
    """比较计划抽象基类（classDiagram ComparisonPlan）。

    子类职责：
        - validate()：执行七条一致性规则，返回 ConsistencyReport；
        - execute(metric_id)：逐指标执行统计检验，返回 TestResult；
        - group_labels()：分组标签（盲法下为盲法编号）。

    设计：两组对照 = TwoGroupPlan 特例；剂量梯度留位（未实现），
    新增设计只需新增子类，不动七条规则与导出/UI 层。
    """

    def __init__(self, runs: Sequence[RunBundle]) -> None:
        if len(runs) < 2:
            raise ValueError("比较至少需要 2 个 run")
        self.runs: list[RunBundle] = list(runs)

    # ------------------------------------------------------------------
    @abstractmethod
    def validate(self) -> ConsistencyReport:
        """比较前一致性校验（七条规则）。"""
        raise NotImplementedError

    @abstractmethod
    def execute(self, metric_id: str) -> Any:
        """对单个指标执行统计检验。"""
        raise NotImplementedError

    @abstractmethod
    def group_labels(self) -> list[str]:
        """分组标签列表（盲法下为盲法编号）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 七条规则的公共实现（子类复用）
    # ------------------------------------------------------------------
    @staticmethod
    def RULES() -> list[Rule]:
        return [
            Rule(1, "px_per_mm 一致性", "reject_metrics",
                 "px_per_mm 不一致 → 拒绝比较任何带 unnormalized 的指标"),
            Rule(2, "metrics_spec_version 一致性", "reject_all",
                 "规范版本不一致 → 拒绝整份比较（阻断清单含义不同）"),
            Rule(3, "model_md5 一致性", "reject_all",
                 "模型权重不一致 → 拒绝全部比较（检测行为整体改变）"),
            Rule(4, "观察窗一致性", "reject_metrics",
                 "window_s 不一致或任一 run 窗口截断 → 拒绝 RR 与清空时间类"),
            Rule(5, "ROI 一致性", "warn",
                 "ROI 面积差>5% 或 IoU<0.95 → 告警（不同养殖单元时本应不同）"),
            Rule(6, "已关闭指标集一致性", "warn",
                 "两组可用指标集不同 → 告警（缺失可能非随机）"),
            Rule(7, "非中立门控缺失", "reject_metrics",
                 "缺失由非中立门控触发 → 拒绝对可用子集做统计推断"),
        ]

    def _check_config_rules(
        self, report: ConsistencyReport, metric_ids: Sequence[str]
    ) -> None:
        """规则 1–5：配置/ROI 层（与具体统计方法无关）。"""
        cfgs = [r.config for r in self.runs]
        base = cfgs[0]

        # ---- 规则 2：metrics_spec_version ----
        if any(c.metrics_spec_version != base.metrics_spec_version for c in cfgs):
            report.violations.append(Violation(
                rule_id=2, rule_name="metrics_spec_version 一致性",
                action="reject_all",
                message=(
                    "参与比较的 run 使用了不同的指标规范版本："
                    + "、".join(
                        f"{r.run_id}={r.config.metrics_spec_version}"
                        for r in self.runs
                    )
                    + "。版本不同意味着『什么算阻断』的清单不同，"
                      "跨批比较会失效。"
                ),
                affected_metrics=list(metric_ids),
                exit_hint=RERUN_HINT,
            ))

        # ---- 规则 3：model_md5 ----
        if any((c.model_md5 or "") != (base.model_md5 or "") for c in cfgs):
            report.violations.append(Violation(
                rule_id=3, rule_name="model_md5 一致性",
                action="reject_all",
                message=(
                    "参与比较的 run 使用了不同模型权重："
                    + "、".join(
                        f"{r.run_id}={r.config.model_md5 or '(空)'}"
                        for r in self.runs
                    )
                    + "。换/重训模型会让检测行为整体改变，几乎所有指标不可比。"
                ),
                affected_metrics=list(metric_ids),
                exit_hint=RERUN_HINT,
            ))

        # ---- 规则 1：px_per_mm → 拒绝 unnormalized 指标 ----
        px = {r.config.px_per_mm_ref for r in self.runs}
        if len(px) > 1:
            affected = [
                mid for mid in metric_ids
                if any("unnormalized" in r.flags(mid) for r in self.runs)
            ]
            report.violations.append(Violation(
                rule_id=1, rule_name="px_per_mm 一致性",
                action="reject_metrics",
                message=(
                    "各 run 的 px_per_mm 不一致（"
                    + "、".join(
                        f"{r.run_id}={r.config.px_per_mm_ref}" for r in self.runs
                    )
                    + "）：像素口径（unnormalized）指标不可跨视频比较。"
                ),
                affected_metrics=affected,
                exit_hint="完成透视标定后重跑，或改用无量纲指标（如 RR / RP / T50）。",
            ))
            report.rejected_metrics.update(affected)

        # ---- 规则 4：观察窗 ----
        windows = {round(float(r.window_s or 0.0), 6) for r in self.runs}
        truncated = [r for r in self.runs if r.window_truncated]
        if len(windows) > 1 or truncated:
            affected = [mid for mid in metric_ids if mid in CLEARANCE_LIKE_METRICS]
            if len(windows) > 1:
                msg = (
                    "各 run 的观察窗不一致（"
                    + "、".join(f"{r.run_id}={r.window_s:.0f}s" for r in self.runs)
                    + "）：RR@300s 与 RR@180s 不是同一个量。"
                )
            else:
                msg = (
                    "以下 run 的观察窗被截断（< 标称窗长）："
                    + "、".join(
                        f"{r.run_id}={r.window_s:.0f}s" for r in truncated
                    )
                    + "：截断窗口径与完整窗不可直接比较。"
                )
            report.violations.append(Violation(
                rule_id=4, rule_name="观察窗一致性",
                action="reject_metrics",
                message=msg,
                affected_metrics=affected,
                exit_hint="延长视频或统一 window_s 后重跑（单段约 4 秒）。",
            ))
            report.rejected_metrics.update(affected)

        # ---- 规则 5：ROI（只告警）----
        rois = [r.roi for r in self.runs if r.roi is not None]
        if len(rois) >= 2 and hasattr(rois[0], "arena"):
            if not self._roi_consistent(rois):
                report.violations.append(Violation(
                    rule_id=5, rule_name="ROI 一致性", action="warn",
                    message=(
                        "各 run 的 ROI 不一致（面积差 > 5% 或 IoU < 0.95）。"
                        "不同养殖单元时 ROI 本应不同（合法），"
                        "但归一化类指标会受影响，请人工确认。"
                    ),
                    affected_metrics=[],
                    exit_hint="若属不同养殖单元，此为合法差异，可继续；"
                              "否则请统一 ROI 后重跑。",
                ))

    @staticmethod
    def _roi_consistent(rois: Sequence[Any]) -> bool:
        """ROI 一致性：arena 面积差 ≤5% 且 IoU ≥0.95（缺失 polygon_area → 保守判为一致）。"""
        try:
            import numpy as np

            from src.core.roi import polygon_area
        except Exception:  # pragma: no cover
            return True

        def _rasterize(poly: Any, xs: Any, ys: Any) -> Any:
            """矢量化奇偶规则（射线法）栅格化多边形。

            ⚠️ 此处原实现用 `(xs-x_a)(y_b-y_a) − (ys-y_a)(x_b-x_a) > 0` 的
            "同侧"判据逐边 XOR——该判据对**边与网格边界重合**的轴对齐矩形
            恒不翻转（cross 恰为 0），导致整个掩膜为空 ⇒ 两个完全相同的
            ROI 算出 IoU=0 ⇒ 规则 5 对每一次合法比较都误报告警
            （已实测复现：`_roi_consistent([roi, roi]) is False`）。
            误报告警会侵蚀整套告警体系的可信度（狼来了效应），故改为
            标准射线法：只统计**跨越水平射线**的边，且用半开区间
            `(y_a > y) != (y_b > y)` 避免顶点被重复计数。
            """
            pts_p = np.asarray(poly, dtype=float)
            inside = np.zeros(xs.shape, dtype=bool)
            n = len(pts_p)
            if n < 3:
                return inside
            x_a, y_a = float(pts_p[0, 0]), float(pts_p[0, 1])
            with np.errstate(invalid="ignore", divide="ignore"):
                for i in range(1, n + 1):
                    x_b, y_b = float(pts_p[i % n, 0]), float(pts_p[i % n, 1])
                    crosses = (y_a > ys) != (y_b > ys)
                    dy = y_b - y_a
                    x_int = np.where(
                        np.abs(dy) < 1e-12,
                        np.inf,
                        x_a + (ys - y_a) / dy * (x_b - x_a),
                    )
                    inside ^= crosses & (xs < x_int)
                    x_a, y_a = x_b, y_b
            return inside

        def _poly_iou(a: Any, b: Any) -> tuple[float, float]:
            """(面积比, IoU)——栅格近似，避免引入 shapely 依赖。"""
            pts = np.vstack([np.asarray(a, dtype=float),
                             np.asarray(b, dtype=float)])
            (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
            w = h = 128
            masks: list[Any] = []
            for poly in (a, b):
                pts_p = np.asarray(poly, dtype=float).copy()
                pts_p[:, 0] = (pts_p[:, 0] - x0) / max(x1 - x0, 1e-9) * (w - 1)
                pts_p[:, 1] = (pts_p[:, 1] - y0) / max(y1 - y0, 1e-9) * (h - 1)
                ys, xs = np.mgrid[0:h, 0:w]
                masks.append(_rasterize(pts_p, xs, ys))
            mask_a, mask_b = masks
            inter = float(np.count_nonzero(mask_a & mask_b))
            union = float(np.count_nonzero(mask_a | mask_b))
            if union == 0:
                # 两个掩膜都为空（退化多边形）：无从判定 → 保守判为一致，
                # 绝不因无法计算就报"不一致"。
                return 1.0, 1.0
            return (inter / max(inter, 1e-9), inter / union)

        base = rois[0]
        for other in rois[1:]:
            try:
                area_a = float(polygon_area(np.asarray(base.arena, dtype=float)))
                area_b = float(polygon_area(np.asarray(other.arena, dtype=float)))
            except Exception:  # pragma: no cover
                continue
            if abs(area_a - area_b) / max(area_a, area_b, 1e-9) > 0.05:
                return False
            try:
                _ratio, iou = _poly_iou(base.arena, other.arena)
            except Exception:  # pragma: no cover
                return True
            if iou < 0.95:
                return False
        return True

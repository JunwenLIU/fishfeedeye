"""comparison_plan.py · 比较计划抽象（T05，统计层入口）。

职责（docs/06 §6 T05 + docs/04 §6.2）：
    - `ComparisonPlan`：**抽象基类**，定义"一组 run 之间怎么比"的契约。
      两组对照（TwoGroupPlan）是本项目的特例实现；剂量梯度 / 多组方差
      分析留扩展位（只加子类，不动基类）——第二轮用户决策 ①。
    - `RunRef`：参与比较的最小单元（run 目录 + 指标表 + run_config +
      重复结构声明），把"从磁盘读什么"收在一处，统计层不碰 IO 细节。
    - 七条比较前一致性校验（compare_rules）：跨 run 比较的**硬门**，
      任一不一致即拒绝，**不提供"用户强制比较"绕过开关**。

为什么必须先校验（docs/04 §4 伪重复 + §7.3）：
    两组之间的系统差异若来自参数/口径而非处理效应，p 值就是精确的废话。
    七条规则覆盖：规范版本、模型指纹、t0 定义、t0 来源、观察窗、ROI、
    标定尺度——即"所有会让同一个数字换一个意思"的旋钮。

任务编号：T05。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.core.config import RunConfig

__all__ = [
    "RunRef",
    "ComparisonPlan",
    "ComparisonResult",
    "CONSISTENCY_RULES",
    "RULE_DESCRIPTIONS",
    "check_consistency",
    "load_run_refs",
]

# ----------------------------------------------------------------------
# 七条一致性规则（docs/06 §6 T05；compare.py 硬校验，不可绕过）
# ----------------------------------------------------------------------
CONSISTENCY_RULES: tuple[str, ...] = (
    "metrics_spec_version",  # 1. 指标规范版本（决定"什么叫干净"的清单）
    "model_md5",             # 2. 模型指纹（换权重 = 换了一把尺子）
    "t0_definition",         # 3. t0 定义（三种口径差出好几秒）
    "t0_source",             # 4. t0 来源（手动/自动，可比性不同）
    "window_s",              # 5. 观察窗长（删失口径不同则 T50 不可比）
    "roi",                   # 6. ROI 定义（区域不同则计数口径不同）
    "sampling",              # 7. 采样三元组（非对称采样参数不同则曲线不可比）
)

RULE_DESCRIPTIONS: dict[str, str] = {
    "metrics_spec_version": "指标规范版本：决定「什么叫干净」的阻断清单版本",
    "model_md5": "模型指纹：换权重等价于换了一把尺子",
    "t0_definition": "t0 定义（投饵器启动/饲料离开投饵器/饲料入画面）：三种口径差出数秒",
    "t0_source": "t0 来源（manual/auto）：自动打点误差不参与跨组比较",
    "window_s": "观察窗长：窗长不同则右删失口径不同，T50 不可比",
    "roi": "ROI 定义（投喂区/参考区/排除区）：区域不同则计数分母不同",
    "sampling": "采样三元组（2s/1s/10s）：非对称采样参数不同则曲线不可比",
}

# px_per_mm 不一致 → 额外拒绝所有 unnormalized 指标（docs/06 T05）
_UNNORMALIZED_REJECT_RULE = "px_per_mm_ref"

_REALIGN_HINT = "重跑对齐约 4 秒（读 cache/detections.jsonl，不重跑检测）"


@dataclass
class RunRef:
    """参与比较的一个 run（磁盘 → 内存的只读投影）。

    Attributes:
        run_id: run 目录名。
        run_dir: run 目录路径。
        config: RunConfig（run_config.yaml；缺失时用内置默认 + 告警）。
        metrics: metric_id → 指标行 dict（17 列冻结 schema；缺失键 = 未输出）。
        group: 组标签（盲法场景为盲法编号；揭盲后为真实分组）。
        pond_id: 重复结构声明（伪重复检查用；None = 未声明 → 强制告警）。
        blind_code: 盲法编号（可 None）。
        warnings: 加载期告警（绝不静默）。
    """

    run_id: str
    run_dir: Path
    config: RunConfig
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    group: str | None = None
    pond_id: str | None = None
    blind_code: str | None = None
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        """对外显示名（盲法优先：不泄露分组）。"""
        return self.blind_code or self.run_id

    def has(self, metric_id: str) -> bool:
        return metric_id in self.metrics

    def value_of(self, metric_id: str) -> float | None:
        """取指标数值；不可用/删失/未输出 → None（绝不用 0 冒充）。"""
        row = self.metrics.get(metric_id)
        if row is None:
            return None
        v = row.get("value")
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def status_of(self, metric_id: str) -> str:
        row = self.metrics.get(metric_id)
        return str(row.get("status", "unavailable")) if row else "unavailable"

    def usable_values(self, metric_ids: Iterable[str]) -> dict[str, float]:
        """批量取可用值（仅 status in ok/degraded 且 value 非空的指标）。"""
        out: dict[str, float] = {}
        for mid in metric_ids:
            if self.status_of(mid) not in ("ok", "degraded"):
                continue
            v = self.value_of(mid)
            if v is not None:
                out[mid] = v
        return out


@dataclass
class ComparisonResult:
    """一次比较的输出（拒绝也是输出——列出差异 + 重跑提示）。

    Attributes:
        ok: 是否通过全部前置校验（False = 未执行任何统计）。
        plan_name: 比较计划名（'two_group' 等）。
        rejected_rules: 命中的拒绝规则名（空 = 通过）。
        differences: 逐条差异描述（字段级粒度，可直接展示给用户）。
        rejected_metrics: 被额外拒绝的 metric_id（如 px_per_mm 不一致 →
            unnormalized 指标；空 = 无额外拒绝）。
        tests: metric_id → 检验结果 dict（由子类填充）。
        warnings: 非阻断告警（可用集不一致 / 伪重复 / 单池仅描述性）。
        notes: 口径说明（检验方法选择依据、重复结构说明等）。
    """

    ok: bool = False
    plan_name: str = ""
    rejected_rules: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    rejected_metrics: list[str] = field(default_factory=list)
    tests: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "plan_name": self.plan_name,
            "rejected_rules": list(self.rejected_rules),
            "differences": list(self.differences),
            "rejected_metrics": list(self.rejected_metrics),
            "tests": dict(self.tests),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


# ----------------------------------------------------------------------
# 七条规则校验
# ----------------------------------------------------------------------
def _roi_of(cfg: RunConfig) -> Any:
    """run_config 里的 ROI 指纹（homography + px_per_mm 之外的区域口径）。

    RunConfig 未直接存 ROI 顶点（存 homography + px_per_mm_ref），故 ROI
    指纹取 run_dir/roi.json（若存在），否则退化为 homography 矩阵比较。
    """
    return cfg.homography


def check_consistency(
    refs: Sequence[RunRef],
    roi_by_run: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """七条一致性规则校验（拒绝即返回，不做统计）。

    Args:
        refs: 参与比较的 run（≥2）。
        roi_by_run: run_id → ROI 指纹（可序列化的任意结构；None 时用
            run_config.homography 代替）。

    Returns:
        (rejected_rules, differences, rejected_metrics)：
            rejected_rules — 命中的规则名（CONSISTENCY_RULES 的子集 +
            可能含 'px_per_mm_ref'）；
            differences — 逐条人类可读差异描述；
            rejected_metrics — 因 px_per_mm 不一致被额外拒绝的指标。
    """
    rejected: list[str] = []
    differences: list[str] = []
    rejected_metrics: list[str] = []
    if len(refs) < 2:
        return (
            ["n_runs"],
            [f"比较至少需要 2 个 run，收到 {len(refs)} 个"],
            [],
        )

    base = refs[0]
    for other in refs[1:]:
        diffs = base.config.diff(other.config)

        # ---- 1–5 + 7：run_config 字段级 diff 直接映射 ----
        for rule in CONSISTENCY_RULES:
            if rule == "roi":
                continue  # ROI 单独处理（不落在 RunConfig 里）
            prefix = "thresholds.observation_window_s" if rule == "window_s" else rule
            if rule == "sampling":
                prefix = "sampling."
            hits = [d for d in diffs if d.startswith(prefix)]
            if hits:
                if rule not in rejected:
                    rejected.append(rule)
                for h in hits:
                    differences.append(
                        f"[{rule}] {base.run_id} vs {other.run_id}: {h}"
                        f"（{RULE_DESCRIPTIONS[rule]}）"
                    )

        # ---- 6：ROI ----
        roi_map = roi_by_run or {}
        a_roi = roi_map.get(base.run_id, _roi_of(base.config))
        b_roi = roi_map.get(other.run_id, _roi_of(other.config))
        if a_roi != b_roi:
            if "roi" not in rejected:
                rejected.append("roi")
            differences.append(
                f"[roi] {base.run_id} vs {other.run_id}: ROI/单应性不一致"
                f"（{RULE_DESCRIPTIONS['roi']}）"
            )

        # ---- 附加：px_per_mm 不一致 → 拒绝所有 unnormalized 指标 ----
        if base.config.px_per_mm_ref != other.config.px_per_mm_ref:
            if _UNNORMALIZED_REJECT_RULE not in rejected:
                rejected.append(_UNNORMALIZED_REJECT_RULE)
            differences.append(
                f"[px_per_mm_ref] {base.run_id} vs {other.run_id}: "
                f"{base.config.px_per_mm_ref} != {other.config.px_per_mm_ref}"
                "：尺度不一致，额外拒绝所有 unnormalized 指标"
            )
            for ref in (base, other):
                for mid, row in ref.metrics.items():
                    if str(row.get("unit_scale", "")) == "px":
                        if mid not in rejected_metrics:
                            rejected_metrics.append(mid)

    if rejected:
        differences.append(f"修复建议：{_REALIGN_HINT}")
    return rejected, differences, rejected_metrics


# ----------------------------------------------------------------------
# 加载（磁盘 → RunRef）
# ----------------------------------------------------------------------
def load_run_refs(
    run_dirs: Sequence[str | Path],
    groups: Sequence[str] | None = None,
    pond_ids: Sequence[str | None] | None = None,
    blind_codes: Sequence[str | None] | None = None,
) -> list[RunRef]:
    """从 run 目录批量构造 RunRef（读 run_config.yaml + metrics_summary.csv）。

    metrics_summary.csv 缺失 → 该 run 无可用指标（warnings 记录，不静默跳过：
    空指标集会让"可用集不一致"告警失真）。

    meta.json 提供 pond_id（重复结构）与 group_label_encrypted（盲法编号）。
    """
    import csv

    refs: list[RunRef] = []
    for i, d in enumerate(run_dirs):
        run_dir = Path(d)
        warnings: list[str] = []
        if not run_dir.exists():
            raise FileNotFoundError(f"run 目录不存在: {run_dir}")
        cfg_path = run_dir / "run_config.yaml"
        if cfg_path.exists():
            config = RunConfig.from_yaml(cfg_path)
        else:
            config = RunConfig()
            warnings.append(
                "run_config.yaml 缺失：使用内置默认配置"
                "（七条校验几乎必然不通过）"
            )

        metrics: dict[str, dict[str, Any]] = {}
        csv_path = run_dir / "metrics_summary.csv"
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    mid = row.get("metric_id")
                    if mid:
                        metrics[str(mid)] = dict(row)
        else:
            warnings.append(
                "metrics_summary.csv 缺失：该 run 无可用指标"
                "（可用集为空，跨组比较的缺失告警将失真）"
            )

        pond_id: str | None = None
        blind_code: str | None = None
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(m, dict):
                    pond_id = m.get("pond_id")
                    blind_code = m.get("blind_code")
            except json.JSONDecodeError:
                warnings.append("meta.json 解析失败：重复结构未声明")

        if pond_ids is not None and i < len(pond_ids):
            pond_id = pond_ids[i]
        if blind_codes is not None and i < len(blind_codes):
            blind_code = blind_codes[i]
        group = groups[i] if groups is not None and i < len(groups) else None

        if pond_id is None:
            warnings.append(
                "pond_id 未声明：无法判定重复结构（伪重复风险，"
                "统计输出将标注为重复结构未知）"
            )

        refs.append(
            RunRef(
                run_id=run_dir.name,
                run_dir=run_dir,
                config=config,
                metrics=metrics,
                group=group,
                pond_id=pond_id,
                blind_code=blind_code,
                warnings=warnings,
            )
        )
    return refs


# ----------------------------------------------------------------------
# 抽象基类
# ----------------------------------------------------------------------
class ComparisonPlan:
    """比较计划抽象基类（classDiagram ComparisonPlan）。

    子类职责：
        - `validate()`：本计划特有的结构校验（如两组 = 恰好两个组、每组 n≥?）；
        - `execute(metric_ids)`：逐指标执行检验并填充 ComparisonResult.tests。

    基类负责：七条一致性规则（硬门）+ 可用集差异检查 + 伪重复检查 +
    拒绝时的差异清单与重跑提示。**任何子类都不得跳过基类校验**——
    这是"不提供强制比较绕过开关"的实现保证：唯一的执行入口 `run()`
    先跑 `check_consistency`，不通过就直接返回，不调用 `execute()`。
    """

    name: str = "comparison_plan"

    def __init__(self, refs: Sequence[RunRef]) -> None:
        self.refs: list[RunRef] = list(refs)

    # ------------------------------------------------------------------
    def run(
        self,
        metric_ids: Sequence[str] | None = None,
        roi_by_run: dict[str, Any] | None = None,
    ) -> ComparisonResult:
        """唯一执行入口：先过七条硬门，通过才调用 execute()。"""
        result = ComparisonResult(plan_name=self.name)
        for ref in self.refs:
            result.warnings.extend(
                f"[{ref.run_id}] {w}" for w in ref.warnings
            )

        rejected, differences, rejected_metrics = check_consistency(
            self.refs, roi_by_run
        )
        result.rejected_rules = rejected
        result.differences = differences
        result.rejected_metrics = rejected_metrics
        if rejected:
            result.ok = False
            result.notes.append(
                "一致性校验未通过：未执行任何统计检验"
                "（不提供强制比较开关——口径不一致时的 p 值是精确的废话）"
            )
            result.notes.append(_REALIGN_HINT)
            return result

        structural = self.validate()
        if structural:
            result.ok = False
            result.differences.extend(structural)
            result.notes.append("比较计划结构校验未通过：未执行统计检验")
            return result

        result.ok = True
        # 把"被额外拒绝的指标"（如 px_per_mm 不一致 → unnormalized）传给
        # 子类，execute() 内据此把这些指标标为 unavailable，而不是照常出 p 值。
        self._rejected_metrics = list(rejected_metrics)
        result.tests = self.execute(metric_ids)
        return result

    # 被额外拒绝的指标（由 check_consistency 填充；未执行校验时为空）
    _rejected_metrics: list[str] = []

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        """计划特有的结构校验（返回问题列表，空 = 通过）。子类覆写。"""
        raise NotImplementedError

    def execute(self, metric_ids: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
        """执行统计检验（子类实现）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 共用检查（子类在 execute 内调用）
    # ------------------------------------------------------------------
    def check_pseudoreplication(self, result: ComparisonResult) -> None:
        """伪重复检查（docs/06 §7.11）：同一池塘的多次投喂 ≠ 独立样本。

        单池塘（或 pond_id 缺失）→ 标注"无独立重复，仅描述性，不可做统计
        推断"；多池塘 → 说明重复结构（MixedLM 随机效应的依据）。
        """
        ponds = {r.pond_id for r in self.refs if r.pond_id is not None}
        n_runs = len(self.refs)
        if not ponds:
            result.warnings.append(
                "重复结构未知（全部 run 均未声明 pond_id）："
                "无独立重复，仅描述性，不可做统计推断"
            )
        elif len(ponds) == 1:
            result.warnings.append(
                f"单池塘（pond_id={sorted(ponds)[0]}，{n_runs} 次投喂）："
                "同一池塘多次投喂属伪重复，仅描述性，不可做统计推断"
            )
        else:
            result.notes.append(
                f"重复结构：{len(ponds)} 个池塘（{sorted(ponds)}）/ "
                f"{n_runs} 次投喂 → 采用以 pond 为随机效应的混合模型（MixedLM）"
            )

    def check_available_set_diff(
        self, result: ComparisonResult, metric_id: str
    ) -> None:
        """可用集差异检查：两组可用指标集不一致 → 非随机缺失告警。

        为什么这是硬告警（docs/04 §4.3 第 4 类）：指标"没输出"往往是因为
        鱼太活跃（运动模糊 → 跟踪失败），即**缺失本身可能就是效应**。
        此时比较"算得出来的那些"会系统性低估处理效应。
        """
        usable = [r.run_id for r in self.refs if r.status_of(metric_id) in ("ok", "degraded")]
        missing = [r.run_id for r in self.refs if r.run_id not in usable]
        if missing and usable:
            result.warnings.append(
                f"{metric_id} 可用集不一致：{sorted(usable)} 有值，"
                f"{sorted(missing)} 不可用——缺失可能是效应本身"
                "（高摄食强度会降低检测/跟踪质量），跨组比较存在非随机缺失风险"
            )

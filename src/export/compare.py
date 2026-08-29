"""compare.py · 比较前校验 + 两组统计 + 偏差诊断 + 导出（T05）。

职责（docs/06 §6 T05 内联约定 + 验收 4 + docs/04 §4.4）：
    - 从 run 目录装配 `RunBundle`（run_config.yaml + metrics_summary.csv +
      cache/quality.json + meta.json + summary.json），一次调用完成
      "加载 → 校验 → 统计 → 诊断 → 落盘"；
    - 校验规则取 docs/04 §4.4 七条与 docs/06 §6 T05 的**并集**：规则 1–7
      在 ComparisonPlan / TwoGroupPlan 内实现，规则 8–9（t0_definition /
      t0_source）由 TwoGroupPlan 按 docs/06 §6 补齐；
    - **不提供"用户强制比较"的绕过开关**：reject_all 时不产出任何 p 值，
      并把逐条差异 + "重跑对齐约 4 秒"的出路写进产物（不输出 = 必须说明）；
    - `diagnose_attenuation()`：**衰减型偏差诊断**——若两组的衰减速率 k
      不同，则所有固定窗口指标（RR@300s / AUC60 / T50）都混入了
      "窗口 × 速率"交互，组间差异可能是动力学差异的伪影而非摄食量差异；
    - px_per_mm 不一致 → 额外拒绝所有 unnormalized 指标（规则 1）。

空值纪律：视图行里所有 None 一律空字符串，绝不为 0 / "none" / "-"。

任务编号：T05。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.core.config import RunConfig
from src.core.roi import ROI
from src.export.csv_writer import read_metrics_summary_csv
from src.stats.comparison_plan import ConsistencyReport, RunBundle, Violation
from src.stats.two_group import TwoGroupPlan, describe_group

__all__ = [
    "CompareResult",
    "CompareOutcome",
    "compare_runs",
    "run_comparison",
    "load_bundles",
    "load_run_bundle",
    "build_two_group_plan",
    "comparison_view",
    "group_descriptives",
    "diagnose_attenuation",
    "render_compare_report_md",
    "write_compare_outputs",
    "write_comparison_csv",
    "write_comparison_json",
    "VIEW_COLUMNS",
    "REALIGN_HINT",
]

REALIGN_HINT = "请用当前版本重新分析上述 run（单段约 4 秒）后再比较。"

VIEW_COLUMNS: tuple[str, ...] = (
    "metric_id", "metric_name_zh", "status", "reason",
    "group_a", "group_b", "n_a", "n_b", "mean_a", "mean_b",
    "test_used", "p_value", "p_holm", "effect_size", "effect_size_type",
    "ci_low", "ci_high", "ci_level", "ci_of",
    "descriptive_only", "repeat_structure", "normality_ok", "equal_var_ok",
    "warnings", "notes",
)

# 受"窗口 × 速率"交互影响的固定窗口指标（衰减型偏差诊断的消费方）
_WINDOW_DEPENDENT_METRICS: tuple[str, ...] = (
    "A11_RR", "A13_AUC60", "A8_T50", "A9_T90", "A10_T100",
)


# ----------------------------------------------------------------------
# 输出结构
# ----------------------------------------------------------------------
@dataclass
class CompareResult:
    """一次比较的完整结果（JSON 可序列化）。

    Attributes:
        ok: 是否执行了统计（False = 被 reject_all 拒绝；view/tests 为空）。
        report: ConsistencyReport（violations / rejected_metrics / notes）。
        tests: metric_id → TestResult.to_dict()（被拒指标给 unavailable）。
        runs: 参与比较的 RunBundle。
        view: 逐指标视图行（VIEW_COLUMNS 列序）。
        descriptives: {组: {metric_id: describe_group 结果}}。
        attenuation: 衰减型偏差诊断结论（空 = 无需告警）。
        group_pair: 两组标签。
    """

    ok: bool
    report: ConsistencyReport
    tests: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs: list[RunBundle] = field(default_factory=list)
    view: list[dict[str, Any]] = field(default_factory=list)
    descriptives: dict[str, dict[str, Any]] = field(default_factory=dict)
    attenuation: list[str] = field(default_factory=list)
    group_pair: tuple[str, str] = ("A", "B")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "group_pair": list(self.group_pair),
            "report": self.report.to_dict(),
            "tests": dict(self.tests),
            "runs": [
                {
                    "run_id": r.run_id,
                    "group": r.group,
                    "pond_id": r.pond_id,
                    "window_s": r.window_s,
                    "window_truncated": r.window_truncated,
                    "disabled": sorted(r.disabled),
                }
                for r in self.runs
            ],
            "view": list(self.view),
            "descriptives": self.descriptives,
            "attenuation": list(self.attenuation),
        }


# 兼容别名（早期命名；保留以免调用方与既有脚本断裂）
CompareOutcome = CompareResult


# ----------------------------------------------------------------------
# 加载
# ----------------------------------------------------------------------
def load_run_bundle(
    run_dir: str | Path,
    group: str = "",
    pond_id: str | None = None,
    roi: ROI | None = None,
) -> RunBundle:
    """从单个 run 目录装配 RunBundle（缺失件记入 _load_notes，绝不静默）。

    读取：run_config.yaml（七条规则依据）/ metrics_summary.csv（17 列冻结）/
    cache/quality.json（规则 7 的非中立门控输入）/ meta.json（pond_id 重复
    结构）/ summary.json（window_s / window_truncated，规则 4 输入）。
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")

    notes: list[str] = []
    cfg_path = run_dir / "run_config.yaml"
    if cfg_path.exists():
        config = RunConfig.from_yaml(cfg_path)
    else:
        config = RunConfig()
        notes.append("run_config.yaml 缺失：使用内置默认配置（比较几乎必然被拒绝）")

    metrics: dict[str, Any] = {}
    disabled: set[str] = set()
    csv_path = run_dir / "metrics_summary.csv"
    if csv_path.exists():
        for row in read_metrics_summary_csv(csv_path):
            mid = row.get("metric_id")
            if not mid:
                continue
            metrics[str(mid)] = dict(row)
            if str(row.get("status", "")) in ("unavailable", "censored"):
                disabled.add(str(mid))
    else:
        notes.append(
            "metrics_summary.csv 缺失：该 run 无可用指标"
            "（可用集为空，规则 6 的缺失告警将失真）"
        )

    quality: dict[str, Any] = {}
    q_path = run_dir / "cache" / "quality.json"
    if q_path.exists():
        try:
            quality = json.loads(q_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            notes.append("cache/quality.json 解析失败：非中立门控判定不可用")

    window_s: float | None = None
    window_truncated = False
    s_path = run_dir / "summary.json"
    if s_path.exists():
        try:
            s = json.loads(s_path.read_text(encoding="utf-8"))
            window_s = s.get("window_s")
            window_truncated = bool(s.get("window_truncated", False))
        except json.JSONDecodeError:
            notes.append("summary.json 解析失败：观察窗信息不可用")

    meta_pond: str | None = None
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(m, dict):
                meta_pond = m.get("pond_id")
        except json.JSONDecodeError:
            notes.append("meta.json 解析失败：重复结构未声明")
    if pond_id is None:
        pond_id = meta_pond
    if pond_id is None:
        notes.append(
            "pond_id 未声明：无法判定重复结构（统计将标注 descriptive_only）"
        )

    bundle = RunBundle(
        run_id=run_dir.name,
        config=config,
        metrics=metrics,
        disabled=disabled,
        quality=quality,
        window_s=window_s,
        window_truncated=window_truncated,
        roi=roi,
        group=str(group),
        pond_id=pond_id,
    )
    if notes:
        # 挂在 quality 的旁路上（键名不会与 Q_* 冲突），由 report.notes 承载
        bundle.quality["_load_notes"] = notes
    return bundle


def load_bundles(
    run_dirs: Sequence[str | Path],
    groups: Sequence[str] | None = None,
    pond_ids: Sequence[str | None] | None = None,
    rois: Sequence[ROI | None] | None = None,
) -> list[RunBundle]:
    """批量装配 RunBundle（依次调用 load_run_bundle）。"""
    return [
        load_run_bundle(
            d,
            group=(groups[i] if groups is not None and i < len(groups) else ""),
            pond_id=(pond_ids[i] if pond_ids is not None and i < len(pond_ids) else None),
            roi=(rois[i] if rois is not None and i < len(rois) else None),
        )
        for i, d in enumerate(run_dirs)
    ]


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def build_two_group_plan(
    bundles: Sequence[RunBundle],
    group_a: str = "A",
    group_b: str = "B",
    alpha: float = 0.05,
) -> TwoGroupPlan:
    """由 RunBundle 列表构造 TwoGroupPlan（compare_page / 脚本共用）。"""
    return TwoGroupPlan(list(bundles), group_a=group_a, group_b=group_b, alpha=alpha)


def run_comparison(
    run_dirs: Sequence[str | Path | RunBundle],
    groups: Sequence[str] | None = None,
    pond_ids: Sequence[str | None] | None = None,
    rois: Sequence[ROI | None] | None = None,
    metric_ids: Sequence[str] | None = None,
    group_pair: tuple[str, str] = ("A", "B"),
    alpha: float = 0.05,
) -> CompareResult:
    """一站式比较：装配 → 校验（规则 1–9）→ 统计 + Holm → 衰减型偏差诊断。

    Args:
        run_dirs: run 目录路径列表（也可直接传 RunBundle，UI 复用已有对象）。
        groups: 组标签（盲法下传盲法编号）。
        pond_ids: 重复结构声明（伪重复防线）。
        rois: ROI（规则 5 输入，可 None）。
        metric_ids: 待检指标（None = 两组指标表并集）。
        group_pair: 两组标签。
        alpha: 显著性水平（同时用于正态性/方差齐性前提与 Holm 校正）。

    Returns:
        CompareResult：ok=False（reject_all）时 report.violations 给出逐条
        差异与出路，且 **tests/view 为空**——拒绝即不产出任何 p 值。
    """
    if run_dirs and isinstance(run_dirs[0], RunBundle):
        bundles = list(run_dirs)  # type: ignore[arg-type]
    else:
        bundles = load_bundles(
            [str(d) for d in run_dirs],  # type: ignore[arg-type]
            groups=groups, pond_ids=pond_ids, rois=rois,
        )

    plan = build_two_group_plan(
        bundles, group_a=group_pair[0], group_b=group_pair[1], alpha=alpha
    )
    report = plan.validate()
    for b in bundles:
        for n in (b.quality.get("_load_notes") or []):
            report.notes.append(f"[{b.run_id}] {n}")

    # 整份拒绝（reject_all）→ 不执行任何统计；
    # 仅 reject_metrics（规则 1/4/7）→ 其余指标照常比较，被拒的指标
    # 由 run_all 标为 unavailable（不给 p 值），差异仍写进 report。
    if report.reject_all():
        return CompareResult(ok=False, report=report, runs=bundles,
                             group_pair=group_pair)

    results = plan.run_all(metric_ids)
    tests = {mid: r.to_dict() for mid, r in results.items()}
    target_ids = list(metric_ids) if metric_ids else sorted(tests)
    view = comparison_view(tests, bundles, report, group_pair)
    descriptives = group_descriptives(bundles, target_ids, group_pair)
    attenuation = diagnose_attenuation(tests)
    for line in attenuation:
        report.notes.append(f"[衰减型偏差诊断] {line}")
    return CompareResult(
        ok=True, report=report, tests=tests, runs=bundles, view=view,
        descriptives=descriptives, attenuation=attenuation,
        group_pair=group_pair,
    )


# 兼容命名（与 run_comparison 同构，语义更贴近"从目录比较"的调用方）
compare_runs = run_comparison


# ----------------------------------------------------------------------
# 衰减型偏差诊断
# ----------------------------------------------------------------------
def diagnose_attenuation(
    tests: dict[str, dict[str, Any]] | CompareResult,
) -> list[str]:
    """衰减型偏差诊断：固定窗口指标是否被"窗口 × 速率"交互污染。

    问题（本项目 A 组的特异性陷阱）：
        RR@300s / AUC60 / T50 都是**固定观察窗**上的累积量。若 A 组衰减快
        （k 大）而 B 组衰减慢，即使两组的"总摄食量"完全相同，A 组在
        t=300s 的残留也会更低、AUC60 更高、T50 更短——组间差异是**动力学
        差异的伪影**，不是摄食量的差异。只看 RR 会得出反向结论。

    判据（两条同时成立才告警，避免误报）：
        ① A12_k（或 A12_tau）在两组间存在显著差异（p < 0.05）或
           相对差 |Δk|/k̄ > 20%；
        ② 至少一个固定窗口指标（A11_RR / A13_AUC60 / A8_T50…）有可用检验。

    处置建议：以 A12_k / A12_tau（与窗口无关的动力学参数）为主结论，
    固定窗口指标降为描述性；或改用与窗口无关的面积口径（A13 全窗积分）。
    """
    if isinstance(tests, CompareResult):
        tests = tests.tests
    out: list[str] = []
    k_test = tests.get("A12_k")
    if not k_test or k_test.get("status") != "ok":
        return out  # 无动力学参数 → 无从诊断（不猜测）
    p_k = k_test.get("p_value")
    mean_a, mean_b = k_test.get("mean_a"), k_test.get("mean_b")
    k_gap: float | None = None
    if mean_a is not None and mean_b is not None:
        k_bar = (abs(mean_a) + abs(mean_b)) / 2.0
        if k_bar > 0:
            k_gap = abs(mean_a - mean_b) / k_bar
    significant = (p_k is not None and p_k < 0.05) or (k_gap is not None and k_gap > 0.20)
    if not significant:
        return out

    hit = [m for m in _WINDOW_DEPENDENT_METRICS
           if tests.get(m, {}).get("status") == "ok"]
    if hit:
        gap_txt = f"{k_gap:.1%}" if k_gap is not None else "不可量化"
        p_txt = f"{p_k:.4f}" if p_k is not None else "不可用"
        out.append(
            f"两组衰减速率不同（A12_k p={p_txt}，相对差 {gap_txt}）："
            f"{'、'.join(hit)} 是固定观察窗上的累积量，其组间差异可能来自"
            "「窗口 × 速率」交互（动力学差异的伪影）而非摄食量差异。"
        )
        out.append(
            "处置：以 A12_k / A12_tau（与窗口无关的动力学参数）为主结论，"
            "固定窗口指标降为描述性；或比较全窗积分口径（A13）并统一观察窗。"
        )
    return out


# ----------------------------------------------------------------------
# 视图
# ----------------------------------------------------------------------
def comparison_view(
    tests: dict[str, dict[str, Any]],
    bundles: Sequence[RunBundle],
    report: ConsistencyReport | None = None,
    group_pair: tuple[str, str] = ("A", "B"),
) -> list[dict[str, Any]]:
    """逐指标视图行（六要素齐全，空值为空字符串，绝不为 0）。"""
    names: dict[str, str] = {}
    for b in bundles:
        for mid, row in b.metrics.items():
            zh = str(row.get("metric_name_zh") or "")
            if zh and mid not in names:
                names[mid] = zh

    rejected = set((report.rejected_metrics if report else set()) or set())
    warn_by_metric: dict[str, list[str]] = {}
    for v in (report.violations if report else []):
        for mid in v.affected_metrics:
            warn_by_metric.setdefault(mid, []).append(
                f"[规则{v.rule_id}·{v.rule_name}] {v.message}"
            )

    rows: list[dict[str, Any]] = []
    for mid in sorted(tests):
        t = tests[mid]
        warnings = list(t.get("warnings", []))
        warnings.extend(warn_by_metric.get(mid, []))
        if mid in rejected:
            warnings.append(
                "该指标被比较前一致性校验拒绝（口径不一致或非中立门控缺失）"
            )
        rows.append(
            {
                "metric_id": mid,
                "metric_name_zh": names.get(mid, ""),
                "status": t.get("status", ""),
                "reason": t.get("reason") or "",
                "group_a": group_pair[0],
                "group_b": group_pair[1],
                "n_a": t.get("n_a", ""),
                "n_b": t.get("n_b", ""),
                "mean_a": _fmt(t.get("mean_a")),
                "mean_b": _fmt(t.get("mean_b")),
                "test_used": t.get("test_used") or "",
                "p_value": _fmt(t.get("p_value")),
                "p_holm": _fmt(t.get("p_holm")),
                "effect_size": _fmt(t.get("effect_size")),
                "effect_size_type": t.get("effect_size_type") or "",
                "ci_low": _fmt(t.get("ci_low")),
                "ci_high": _fmt(t.get("ci_high")),
                "ci_level": _fmt(t.get("ci_level")),
                "ci_of": t.get("ci_of") or "",
                "descriptive_only": _tri(t.get("descriptive_only")),
                "repeat_structure": t.get("repeat_structure", ""),
                "normality_ok": _tri(t.get("normality_ok")),
                "equal_var_ok": _tri(t.get("equal_var_ok")),
                "warnings": " | ".join(warnings),
                "notes": " | ".join(t.get("notes", [])),
            }
        )
    return rows


def group_descriptives(
    bundles: Sequence[RunBundle],
    metric_ids: Sequence[str],
    group_pair: tuple[str, str] = ("A", "B"),
) -> dict[str, dict[str, Any]]:
    """两组描述统计（不跑检验也能展示分布；n 不足如实显示，不画假均值）。"""
    out: dict[str, dict[str, Any]] = {}
    for g in group_pair:
        per_metric: dict[str, Any] = {}
        for mid in metric_ids:
            vals = [
                b.value(mid) for b in bundles
                if b.group == g and b.status(mid) in ("ok", "degraded")
                and b.value(mid) is not None
            ]
            per_metric[mid] = describe_group(vals)
        out[g] = per_metric
    return out


# ----------------------------------------------------------------------
# 报告与落盘
# ----------------------------------------------------------------------
def render_compare_report_md(result: CompareResult) -> str:
    """渲染 comparison_report.md（人类可读；被拒绝时解释"为什么没有 p 值"）。"""
    g_a, g_b = result.group_pair
    lines: list[str] = [
        "# 分组比较报告",
        "",
        f"- 比较计划：two_group（{g_a} vs {g_b}）",
        f"- 执行统计：{'是' if result.ok else '否（被比较前校验拒绝）'}",
        f"- 参与 run 数：{len(result.runs)}",
        "",
    ]

    if not result.ok or not result.view:
        lines.append("## 未执行统计的原因")
        lines.append("")
        for v in result.report.violations:
            lines.append(
                f"- **规则 {v.rule_id} · {v.rule_name}**（{v.action}）：{v.message}"
            )
            if v.exit_hint:
                lines.append(f"  - 出路：{v.exit_hint}")
        if result.report.rejected_metrics:
            lines.append(
                f"- 被拒绝的指标：{', '.join(sorted(result.report.rejected_metrics))}"
            )
        for n in result.report.notes:
            lines.append(f"- 备注：{n}")
        lines.extend(["", f"> {REALIGN_HINT}", ""])
        return "\n".join(lines)

    lines.append("## 逐指标检验结果（六要素）")
    lines.append("")
    lines.append(
        "| 指标 | 状态 | n(A/B) | 均值(A/B) | 检验方法 | p | p(Holm) | 效应量 | 95%CI | 重复结构 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in result.view:
        es = (
            f"{row['effect_size']} ({row['effect_size_type']})"
            if row["effect_size"] else ""
        )
        ci = (
            f"[{row['ci_low']}, {row['ci_high']}]"
            if row["ci_low"] or row["ci_high"] else ""
        )
        marker = " ⚠️" if row["descriptive_only"] == "true" else ""
        lines.append(
            f"| {row['metric_id']} {row['metric_name_zh']} | {row['status']} | "
            f"{row['n_a']}/{row['n_b']} | {row['mean_a']}/{row['mean_b']} | "
            f"{row['test_used']} | {row['p_value']} | {row['p_holm']} | {es} | "
            f"{ci} | {row['repeat_structure']}{marker} |"
        )
        if row["reason"]:
            lines.append(f"  - 原因：{row['reason']}")
        if row["warnings"]:
            lines.append(f"  - ⚠ {row['warnings']}")

    if result.attenuation:
        lines.extend(["", "## 衰减型偏差诊断", ""])
        for line in result.attenuation:
            lines.append(f"- ⚠ {line}")

    warns = [v for v in result.report.violations if v.action == "warn"]
    if warns:
        lines.extend(["", "## 告警（不阻断）", ""])
        for v in warns:
            lines.append(f"- 规则 {v.rule_id} · {v.rule_name}：{v.message}")
            if v.exit_hint:
                lines.append(f"  - 出路：{v.exit_hint}")

    lines.extend(["", "## 备注", ""])
    for n in result.report.notes:
        lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


def write_compare_outputs(
    result: CompareResult, out_dir: str | Path
) -> list[Path]:
    """落盘三件套：comparison.csv / comparison.json / comparison_report.md。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        write_comparison_csv(result, out_dir / "comparison.csv"),
        write_comparison_json(result, out_dir / "comparison.json"),
    ]
    md = out_dir / "comparison_report.md"
    md.write_text(render_compare_report_md(result), encoding="utf-8")
    written.append(md)
    return written


def write_comparison_csv(result: CompareResult, path: str | Path) -> Path:
    """落盘 comparison.csv。

    被拒绝时**仍然写文件**，但内容是差异清单与规则释义——让用户拿到的
    产物解释了"为什么没有 p 值"（docs/04 §0.4：不输出 = 必须显式说明）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        if result.ok and result.view:
            w = csv.DictWriter(fh, fieldnames=list(VIEW_COLUMNS))
            w.writeheader()
            for row in result.view:
                w.writerow(row)
        else:
            w2 = csv.writer(fh)
            w2.writerow(["comparison_rejected", "detail"])
            for v in result.report.violations:
                w2.writerow([
                    f"rule_{v.rule_id}_{v.rule_name}",
                    f"[{v.action}] {v.message}"
                    + (f" | 出路：{v.exit_hint}" if v.exit_hint else ""),
                ])
            if result.report.rejected_metrics:
                w2.writerow([
                    "rejected_metrics",
                    ";".join(sorted(result.report.rejected_metrics)),
                ])
            for line in result.attenuation:
                w2.writerow(["attenuation", line])
            for n in result.report.notes:
                w2.writerow(["note", n])
            w2.writerow(["hint", REALIGN_HINT])
            if result.ok and not result.view:
                w2.writerow(["note", "比较通过但无可用指标（两组指标表交集为空）"])
    return path


def write_comparison_json(result: CompareResult, path: str | Path) -> Path:
    """落盘 comparison.json（完整结构化结果，供 UI 与复核消费）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------------
# 内部
# ----------------------------------------------------------------------
def _fmt(v: Any) -> Any:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def _tri(v: Any) -> str:
    """三态输出：True→true / False→false / None（未测得）→ 空字符串。"""
    if v is None or v == "":
        return ""
    return "true" if bool(v) else "false"

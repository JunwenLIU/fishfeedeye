"""scripts/06_validate_against_groundtruth.py · 自动 vs 人工偏差报告（T05，G1 误差实测）。

职责（docs/06 §6 T05 内联约定）：
    消费人工打点产物 `manual_counts.csv`（TallyRecorder 的输出，
    src/export/csv_writer.write_manual_counts_csv）与自动轨的
    `metrics_timeseries.csv`，输出**自动 vs 人工的偏差报告**：
        - 逐帧配对（按时间戳最近邻，容差可配）→ 偏差分布；
        - 汇总：MAE / RMSE / MAPE / 偏差中位数 / 相关系数 / 系统偏差（Bias）；
        - 累计消耗事件数对照（人工打点累计 vs 自动 N₀ − N_end）；
        - 打点人、打点时间、配对帧数等审计信息。

纪律（docs/06 §3.2 打点工具条款，本项目铁律）：
    - 打点结果**不修正自动指标**，只并列——本脚本只出偏差报告，
      绝不写回 metrics_summary.csv，也不修改任何自动指标；
    - 不可配对/样本不足 → 报告 unavailable + reason，绝不输出 0 偏差
      （"没配对上" ≠ "两者一致"）；
    - 空值一律 None + 原因，绝不用 0 冒充。

用法::

    python scripts/06_validate_against_groundtruth.py \\
        --run runs/demo_20260830_143022 \\
        --manual runs/demo_20260830_143022/manual_counts.csv \\
        --out   runs/demo_20260830_143022/validation_report.md

退出码：0 正常完成；2 输入缺失（报明确的文件名）。
任务编号：T05。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# 项目根加入 sys.path（脚本直跑支持）
_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

__all__ = [
    "PairedPoint",
    "ValidationReport",
    "load_manual_events",
    "load_auto_series",
    "pair_by_timestamp",
    "validate",
    "render_report_md",
    "main",
]

# 配对容差默认值（秒）：非对称采样下尾段间隔 10s，容差须覆盖半间隔
DEFAULT_TOLERANCE_S = 5.0
# 偏差汇总所需的最小配对点数（少于此值不输出汇总统计，只报配对详情）
MIN_PAIRED_POINTS = 3


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class PairedPoint:
    """一个配对点（自动值 vs 人工值）。"""

    t_s: float
    auto_value: float
    manual_value: float

    @property
    def diff(self) -> float:
        """auto − manual（正 = 自动高于人工）。"""
        return self.auto_value - self.manual_value

    @property
    def rel(self) -> float | None:
        """相对偏差（manual 为 0 时无定义 → None，绝不除零）。"""
        if self.manual_value == 0:
            return None
        return self.diff / abs(self.manual_value)


@dataclass
class ValidationReport:
    """偏差报告（空值 = None + reason，绝不用 0 冒充）。"""

    run_id: str = ""
    status: str = "unavailable"          # 'ok' | 'unavailable'
    reason: str | None = None
    n_manual_events: int = 0
    n_auto_points: int = 0
    n_paired: int = 0
    tolerance_s: float = DEFAULT_TOLERANCE_S
    operators: list[str] = field(default_factory=list)
    created_at_first: str | None = None
    created_at_last: str | None = None
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None            # 系统偏差 = mean(auto − manual)
    median_diff: float | None = None
    mape: float | None = None            # 平均绝对相对偏差（%）
    corr: float | None = None            # Pearson 相关系数
    manual_total_events: int | None = None
    auto_total_consumed: float | None = None
    pairs: list[PairedPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "reason": self.reason,
            "n_manual_events": self.n_manual_events,
            "n_auto_points": self.n_auto_points,
            "n_paired": self.n_paired,
            "tolerance_s": self.tolerance_s,
            "operators": list(self.operators),
            "created_at_first": self.created_at_first,
            "created_at_last": self.created_at_last,
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "median_diff": self.median_diff,
            "mape": self.mape,
            "corr": self.corr,
            "manual_total_events": self.manual_total_events,
            "auto_total_consumed": self.auto_total_consumed,
            "pairs": [
                {"t_s": p.t_s, "auto": p.auto_value,
                 "manual": p.manual_value, "diff": p.diff}
                for p in self.pairs
            ],
            "warnings": list(self.warnings),
        }


# ----------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------
def load_manual_events(path: str | Path) -> list[dict[str, Any]]:
    """读 manual_counts.csv（event_id / t_s / phase / operator / created_at）。

    兼容两种表头：TallyRecorder 的 `count`（累计计数）与逐事件打点的
    `n`（每次 +1）；两者都归一为 t_s 时刻的累计人工计数。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manual_counts.csv 不存在: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        # 跳过文件头注释行（TallyRecorder.to_csv 会写 `# video:` /
        # `# operator:` / `# exported_at:` / `# 打点计数为独立证据链…` 四行
        # 审计头；不做预处理会把注释行当成表头，导致后续整表读空）。
        data_lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    if not data_lines:
        return rows
    for raw in csv.DictReader(data_lines):
            t_s = raw.get("t_s")
            if t_s in (None, ""):
                continue
            rows.append(
                {
                    "t_s": float(t_s),
                    "count": (
                        float(raw["count"]) if raw.get("count") not in (None, "")
                        else (float(raw["n"]) if raw.get("n") not in (None, "")
                              else 1.0)
                    ),
                    "phase": raw.get("phase") or "",
                    "operator": raw.get("operator") or "",
                    "created_at": raw.get("created_at") or "",
                }
            )
    rows.sort(key=lambda r: r["t_s"])
    return rows


def load_auto_series(path: str | Path, metric_id: str = "A2_Np") -> list[tuple[float, float]]:
    """读 metrics_timeseries.csv 的指定指标列（原生不规则时间戳，不重采样）。

    Returns:
        [(t_s, value)]，按时间升序；空值（未采样到该指标的点）跳过。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metrics_timeseries.csv 不存在: {path}")
    out: list[tuple[float, float]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or metric_id not in reader.fieldnames:
            return out
        for row in reader:
            t_raw, v_raw = row.get("t_seconds"), row.get(metric_id)
            if t_raw in (None, "") or v_raw in (None, ""):
                continue
            try:
                out.append((float(t_raw), float(v_raw)))
            except ValueError:
                continue
    out.sort(key=lambda x: x[0])
    return out


# ----------------------------------------------------------------------
# 配对与统计
# ----------------------------------------------------------------------
def pair_by_timestamp(
    manual: Sequence[dict[str, Any]],
    auto: Sequence[tuple[float, float]],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> list[PairedPoint]:
    """按时间戳最近邻配对（人工事件 → 自动曲线）。

    人工打点是**累计计数**（第 k 次击键 = 累计 k 个摄食事件），自动曲线
    A2_Np 是**剩余颗粒数**。二者量纲相反，故配对时自动侧取累计消耗
    C(t) = N₀ − N_p(t)：与人工累计计数同向可比。
    未能在容差内找到自动采样点的人工事件 → 丢弃（并计入告警）。
    """
    if not auto:
        return []
    # 自动侧：N₀ 取序列最大值（最早期的剩余量）
    n0 = max(v for _t, v in auto)
    auto_t = np.array([t for t, _v in auto], dtype=float)
    auto_c = np.array([n0 - v for _t, v in auto], dtype=float)

    pairs: list[PairedPoint] = []
    for ev in manual:
        if auto_t.size == 0:
            break
        i = int(np.argmin(np.abs(auto_t - ev["t_s"])))
        if abs(float(auto_t[i]) - ev["t_s"]) > tolerance_s:
            continue  # 容差外：不配对（不猜测）
        pairs.append(
            PairedPoint(
                t_s=float(ev["t_s"]),
                auto_value=float(auto_c[i]),
                manual_value=float(ev["count"]),
            )
        )
    return pairs


def validate(
    run_id: str,
    manual_events: Sequence[dict[str, Any]],
    auto_series: Sequence[tuple[float, float]],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> ValidationReport:
    """计算偏差报告（样本不足 → unavailable + reason，绝不输出 0 偏差）。"""
    rep = ValidationReport(
        run_id=run_id,
        n_manual_events=len(manual_events),
        n_auto_points=len(auto_series),
        tolerance_s=tolerance_s,
    )
    if manual_events:
        created = [e["created_at"] for e in manual_events if e["created_at"]]
        if created:
            rep.created_at_first = min(created)
            rep.created_at_last = max(created)
        for e in manual_events:
            if e["operator"] and e["operator"] not in rep.operators:
                rep.operators.append(e["operator"])

    if not manual_events:
        rep.status = "unavailable"
        rep.reason = "无人工打点事件（manual_counts.csv 为空）：偏差无从计算"
        return rep
    if not auto_series:
        rep.status = "unavailable"
        rep.reason = (
            "自动曲线不可用（metrics_timeseries.csv 无 A2_Np 列或全为空）："
            "偏差无从计算"
        )
        return rep

    pairs = pair_by_timestamp(manual_events, auto_series, tolerance_s)
    rep.pairs = pairs
    rep.n_paired = len(pairs)
    dropped = len(manual_events) - len(pairs)
    if dropped:
        rep.warnings.append(
            f"{dropped} 个人工事件在 ±{tolerance_s}s 内未找到自动采样点，"
            "未参与偏差统计（非对称采样下尾段间隔达 10s 属预期）"
        )
    if rep.n_paired < MIN_PAIRED_POINTS:
        rep.status = "unavailable"
        rep.reason = (
            f"配对点数 {rep.n_paired} < {MIN_PAIRED_POINTS}："
            "样本不足以给出偏差统计（不输出 0 偏差冒充一致）"
        )
        return rep

    d = np.array([p.diff for p in pairs], dtype=float)
    a = np.array([p.auto_value for p in pairs], dtype=float)
    m = np.array([p.manual_value for p in pairs], dtype=float)
    rep.mae = float(np.mean(np.abs(d)))
    rep.rmse = float(math.sqrt(float(np.mean(d * d))))
    rep.bias = float(np.mean(d))
    rep.median_diff = float(np.median(d))
    rels = [p.rel for p in pairs if p.rel is not None and p.manual_value != 0]
    rep.mape = float(np.mean([abs(r) for r in rels]) * 100.0) if rels else None
    if a.size >= 2 and float(np.std(a)) > 0 and float(np.std(m)) > 0:
        rep.corr = float(np.corrcoef(a, m)[0, 1])
    rep.manual_total_events = int(max(m[-1] if m.size else 0, len(manual_events)))
    rep.auto_total_consumed = float(max(a) if a.size else 0.0)
    rep.status = "ok"

    if rep.mape is not None and rep.mape > 20.0:
        rep.warnings.append(
            f"平均绝对相对偏差 {rep.mape:.1f}% > 20%：自动轨与人工打点口径"
            "差异显著，请核对 t0 定义与打点口径（每「一口」是否计为一次）"
        )
    if rep.corr is not None and rep.corr < 0.8:
        rep.warnings.append(
            f"相关系数 {rep.corr:.3f} < 0.8：两条曲线走势一致性不足，"
            "偏差可能不是常数偏移（存在时变系统误差）"
        )
    return rep


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def render_report_md(rep: ValidationReport) -> str:
    """渲染 validation_report.md（人类可读；不可用时解释原因）。"""
    lines: list[str] = [
        "# 自动 vs 人工偏差报告（G1 误差实测）",
        "",
        f"- run：`{rep.run_id}`",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 状态：**{rep.status}**"
        + (f"（{rep.reason}）" if rep.reason else ""),
        "",
        "## 1. 输入与配对",
        "",
        f"- 人工打点事件数：{rep.n_manual_events}",
        f"- 自动曲线采样点数：{rep.n_auto_points}",
        f"- 配对点数：{rep.n_paired}（时间戳容差 ±{rep.tolerance_s:g}s）",
        f"- 打点人：{', '.join(rep.operators) if rep.operators else '（未记录）'}",
        f"- 打点时间范围：{rep.created_at_first or '—'} ~ {rep.created_at_last or '—'}",
        "",
        "> 配对口径：人工打点是**累计**摄食事件计数，自动 A2_Np 是**剩余**颗粒数，"
        "故自动侧取累计消耗 C(t) = N₀ − N_p(t) 后配对（同向可比）。",
        "",
    ]

    if rep.status != "ok":
        lines.extend([
            "## 2. 偏差统计",
            "",
            f"未输出：{rep.reason}",
            "",
            "> 零值纪律：不可用时报告 unavailable 并说明原因，"
            "绝不以 0 偏差冒充「两者一致」。",
            "",
        ])
        _append_warnings(lines, rep)
        return "\n".join(lines)

    lines.extend([
        "## 2. 偏差统计（auto − manual，单位：颗）",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| MAE 平均绝对偏差 | {rep.mae:.3f} |" if rep.mae is not None else "| MAE | — |",
        f"| RMSE 均方根偏差 | {rep.rmse:.3f} |" if rep.rmse is not None else "| RMSE | — |",
        f"| Bias 系统偏差 | {rep.bias:+.3f} |" if rep.bias is not None else "| Bias | — |",
        f"| 偏差中位数 | {rep.median_diff:+.3f} |" if rep.median_diff is not None else "| 中位数 | — |",
        f"| MAPE 平均绝对相对偏差 | {rep.mape:.2f}% |" if rep.mape is not None else "| MAPE | — |",
        f"| Pearson 相关系数 | {rep.corr:.4f} |" if rep.corr is not None else "| 相关系数 | — |",
        "",
        "## 3. 累计量对照",
        "",
        f"- 人工累计事件数：{rep.manual_total_events}",
        f"- 自动累计消耗（C(t) 峰值）："
        + (f"{rep.auto_total_consumed:.3f} 颗" if rep.auto_total_consumed is not None else "—"),
        "",
        "## 4. 配对明细",
        "",
        "| t_s (s) | 自动 C(t) | 人工累计 | 偏差 |",
        "|---|---|---|---|",
    ])
    for p in rep.pairs:
        lines.append(
            f"| {p.t_s:.2f} | {p.auto_value:.3f} | {p.manual_value:.3f} | {p.diff:+.3f} |"
        )
    lines.append("")
    _append_warnings(lines, rep)
    lines.extend([
        "## 5. 纪律声明",
        "",
        "- 本报告**不修正**任何自动指标：打点结果与自动曲线并列展示、永不合并"
        "（docs/06 §3.2）；",
        "- 偏差结论只用于 G1 误差实测与审稿答辩，不回写 metrics_summary.csv。",
        "",
    ])
    return "\n".join(lines)


def _append_warnings(lines: list[str], rep: ValidationReport) -> None:
    if rep.warnings:
        lines.extend(["## 告警", ""])
        for w in rep.warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="自动 vs 人工打点偏差报告（G1 误差实测）"
    )
    ap.add_argument("--run", required=True, help="run 目录（含 metrics_timeseries.csv）")
    ap.add_argument("--manual", default=None,
                    help="manual_counts.csv 路径（默认 <run>/manual_counts.csv）")
    ap.add_argument("--metric", default="A2_Np",
                    help="对照的自动曲线列（默认 A2_Np 剩余颗粒数）")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S,
                    help=f"配对时间戳容差（秒，默认 {DEFAULT_TOLERANCE_S}）")
    ap.add_argument("--out", default=None,
                    help="报告输出路径（默认 <run>/validation_report.md）")
    ap.add_argument("--json-out", default=None,
                    help="结构化结果输出路径（默认 <run>/validation.json）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"[06] run 目录不存在: {run_dir}", file=sys.stderr)
        return 2
    manual_path = Path(args.manual) if args.manual else run_dir / "manual_counts.csv"
    ts_path = run_dir / "metrics_timeseries.csv"
    if not manual_path.exists():
        print(
            f"[06] 缺少人工打点文件: {manual_path}"
            "（先在 UI 回放页打点，或用 TallyRecorder 生成）",
            file=sys.stderr,
        )
        return 2
    if not ts_path.exists():
        print(f"[06] 缺少自动时序文件: {ts_path}（先跑一次完整分析）",
              file=sys.stderr)
        return 2

    manual_events = load_manual_events(manual_path)
    auto_series = load_auto_series(ts_path, args.metric)
    rep = validate(run_dir.name, manual_events, auto_series, args.tolerance)

    out_md = Path(args.out) if args.out else run_dir / "validation_report.md"
    out_md.write_text(render_report_md(rep), encoding="utf-8")
    out_json = Path(args.json_out) if args.json_out else run_dir / "validation.json"
    out_json.write_text(
        json.dumps(rep.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"[06] run_id = {rep.run_id}")
    print(f"[06] 人工事件 {rep.n_manual_events} / 自动采样点 {rep.n_auto_points} "
          f"/ 配对 {rep.n_paired}")
    if rep.status == "ok":
        print(f"[06] MAE={rep.mae:.3f} RMSE={rep.rmse:.3f} Bias={rep.bias:+.3f} "
              f"MAPE={rep.mape:.2f}% corr={rep.corr:.4f}")
    else:
        print(f"[06] 未输出统计：{rep.reason}")
    for w in rep.warnings:
        print(f"[06] ⚠ {w}")
    print(f"[06] 报告 → {out_md}")
    print(f"[06] 结构化 → {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

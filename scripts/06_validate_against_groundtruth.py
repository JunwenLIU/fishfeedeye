"""scripts/06_validate_against_groundtruth.py · 真值对照误差实测（T05，G1）。

职责（docs/04 §7.1 + docs/06 §6 T05）：
    把「未检出缺陷」升级为「实测误差 = X」——**唯一**能做到这件事的途径
    （架构师 v6 决策 6）。输入人工真值帧（`n_pellets_gt` / `n_fish_gt`）
    与自动检测值，输出误差报告 `validation_report.md`。

五条纪律（docs/04 §7.1 ③，防止"真值"成为新的欺骗源）：
    1. **禁止基于 < 30 帧的真值自动施加任何校正系数**——只报告偏差，
       由用户决定是否采信（小样本自动校正 = 用 5 个点的噪声修正全部数据）；
    2. **必须报告真值自身的噪声下限**——同一帧两人独立计数，给出
       评分者间差异（真值噪声）；
    3. 🔴 **实测误差 < 噪声下限 → 必须输出"无法测量"，不得输出精度数字**
       （没测出来 ≠ 没有；同一陷阱第三次出现）；
    4. **不得在 < 30 帧时用"减噪声"的代数方式算修正后误差**
       （σ_model² ≈ MS − σ_ref² 在小样本下可得负值：荒谬但看起来专业）；
    5. **Q_n0gap 不是真值的替代品**——它看不见逐帧系统性比例偏差
       （稳定漏检 15% 时总量对得上）。两者互补，都要做。

用法::

    python scripts/06_validate_against_groundtruth.py \\
        --gt gt_counts.csv --auto auto_counts.csv --run-id demo \\
        --out runs/demo/validation_report.md

    # gt_counts.csv 列：frame_idx, t_s, n_pellets_gt, counter
    #                   （可选 n_fish_gt；同一帧多人计数即多行）
    # auto_counts.csv 列：frame_idx, value

退出码：0 正常完成；2 输入缺失（报明确的文件名/列名）。
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
from typing import Any, Iterable, Mapping, Sequence

# 项目根加入 sys.path（脚本直跑支持）
_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

__all__ = [
    "GroundTruthRow",
    "ValidationReport",
    "MIN_FRAMES_FOR_CORRECTION",
    "validate",
    "load_gt_rows",
    "load_auto_values",
    "render_report_md",
    "main",
]

# 纪律 1/4 的样本量门槛：少于此帧数只报告、禁止任何自动校正
MIN_FRAMES_FOR_CORRECTION = 30

# 纪律 3 的强制输出模板（docs/04 §7.1 ③，逐字照抄）
_UNMEASURABLE_TEMPLATE = (
    "本验证无法区分模型误差与真值噪声（实测误差 < 噪声下限）。"
    "要获得有效结论，需提高真值质量"
    "（增加计数人数 / 更高分辨率人工复核 / 改用已知颗粒数的标定板）。"
)

# 精度数字的降级样式（不可测量时抑制高亮，但仍可见——不静默删除）
_DEEMPHASIS_STYLE = "color:#999"


# ----------------------------------------------------------------------
# 输入结构
# ----------------------------------------------------------------------
@dataclass
class GroundTruthRow:
    """一行人工真值（docs/04 §7.1：同一帧多人计数 = 多行）。

    Attributes:
        frame_idx: 帧号（与自动检测值的配对键）。
        t_s: 相对 t0 的时间戳（秒，仅用于展示与排序）。
        n_pellets_gt: 人工数出的颗粒数（真值）。
        counter: 计数人标识（用于计算评分者间差异 = 真值噪声下限）。
        n_fish_gt: 人工数出的鱼数（可选；§7.1 要求真值必须含鱼数）。
    """

    frame_idx: int
    t_s: float
    n_pellets_gt: float
    counter: str = "A"
    n_fish_gt: float | None = None


@dataclass
class ValidationReport:
    """误差实测报告（不可测量时精度数字仍在，但降级展示 + 明确结论）。

    Attributes:
        n_frames: 参与对照的帧数（真值与自动值都有的帧）。
        n_counters: 计数人数。
        noise_floor: 真值噪声下限（评分者间平均绝对差；单人计数 → 0.0
            且 noise_floor_estimable=False，绝不假装知道）。
        noise_floor_estimable: 噪声下限是否可估计（≥2 人独立计数）。
        measurable: 实测误差是否可测（mae > noise_floor）。
        verdict: 结论文本（不可测量时用强制模板）。
        mae / mape / bias: 平均绝对误差 / 平均绝对相对误差 / 有符号偏差。
        rmse: 均方根误差。
        corr: 自动值 vs 真值的 Pearson 相关系数（None = 无法计算）。
        agreement: 评分者间一致性（1 − 平均相对差；None = 单人计数）。
        warnings: 纪律告警（小样本禁止校正等）。
        per_frame: 逐帧对照明细。
    """

    n_frames: int = 0
    n_counters: int = 0
    noise_floor: float = 0.0
    noise_floor_estimable: bool = False
    measurable: bool = False
    verdict: str = ""
    mae: float | None = None
    mape: float | None = None
    bias: float | None = None
    rmse: float | None = None
    corr: float | None = None
    agreement: float | None = None
    warnings: list[str] = field(default_factory=list)
    per_frame: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "n_counters": self.n_counters,
            "noise_floor": self.noise_floor,
            "noise_floor_estimable": self.noise_floor_estimable,
            "measurable": self.measurable,
            "verdict": self.verdict,
            "mae": self.mae,
            "mape": self.mape,
            "bias": self.bias,
            "rmse": self.rmse,
            "corr": self.corr,
            "agreement": self.agreement,
            "warnings": list(self.warnings),
            "per_frame": list(self.per_frame),
        }


# ----------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------
def load_gt_rows(path: str | Path) -> list[GroundTruthRow]:
    """读真值 CSV（frame_idx, t_s, n_pellets_gt, counter[, n_fish_gt]）。

    跳过 `#` 注释行（人工标注表常带审计头）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"真值文件不存在: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"真值文件为空（无数据行）: {path}")
    reader = csv.DictReader(lines)
    required = {"frame_idx", "n_pellets_gt"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"真值文件缺列 {sorted(missing)}：{path}")
    rows: list[GroundTruthRow] = []
    for raw in reader:
        if not (raw.get("frame_idx") or "").strip():
            continue
        fish = raw.get("n_fish_gt")
        rows.append(
            GroundTruthRow(
                frame_idx=int(float(raw["frame_idx"])),
                t_s=float(raw.get("t_s") or 0.0),
                n_pellets_gt=float(raw["n_pellets_gt"]),
                counter=str(raw.get("counter") or "A"),
                n_fish_gt=(None if fish in (None, "") else float(fish)),
            )
        )
    if not rows:
        raise ValueError(f"真值文件无有效数据行: {path}")
    return rows


def load_auto_values(path: str | Path) -> dict[int, float]:
    """读自动检测值 CSV（frame_idx, value）→ {frame_idx: value}。

    缺失/非法行跳过（不填 0——缺失帧在 validate 里显式告警）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"自动检测值文件不存在: {path}")
    out: dict[int, float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    if not lines:
        return out
    reader = csv.DictReader(lines)
    if "frame_idx" not in (reader.fieldnames or []) or "value" not in (
        reader.fieldnames or []
    ):
        raise ValueError(f"自动检测值文件需含 frame_idx 与 value 两列: {path}")
    for raw in reader:
        f_raw, v_raw = raw.get("frame_idx"), raw.get("value")
        if f_raw in (None, "") or v_raw in (None, ""):
            continue
        try:
            out[int(float(f_raw))] = float(v_raw)
        except ValueError:
            continue
    return out


# ----------------------------------------------------------------------
# 核心：误差实测
# ----------------------------------------------------------------------
def validate(
    rows: Sequence[GroundTruthRow],
    auto: Mapping[int, float],
) -> ValidationReport:
    """真值 vs 自动值的误差实测（五条纪律的执行体）。

    Args:
        rows: 人工真值行（同一帧多人计数 = 多行）。
        auto: {frame_idx: 自动检测值}。

    Returns:
        ValidationReport：measurable=False 时 verdict 为 docs/04 §7.1 ③
        的强制模板，精度数字虽在但**报告中标为降级信息**（不静默删除、
        不高亮）。
    """
    rep = ValidationReport()
    if not rows:
        rep.verdict = "无真值行：误差无从实测（不输出任何精度数字）"
        rep.warnings.append("真值表为空：请先按 docs/04 §7.1 的输入规范采集真值帧")
        return rep

    # ---- 逐帧聚合真值（多人计数取均值 = 该帧真值）----
    by_frame: dict[int, list[GroundTruthRow]] = {}
    for r in rows:
        by_frame.setdefault(int(r.frame_idx), []).append(r)
    counters = {str(r.counter) for r in rows if r.counter}
    rep.n_counters = len(counters) or 1

    # ---- 真值噪声下限（纪律 2）：同一帧多人计数的平均绝对差 ----
    diffs: list[float] = []
    for _f, rs in by_frame.items():
        if len(rs) >= 2:
            vals = [r.n_pellets_gt for r in rs]
            diffs.append(float(np.mean(np.abs(np.diff(sorted(vals))))))
    if diffs:
        rep.noise_floor = float(np.mean(diffs))
        rep.noise_floor_estimable = True
        gt_all = [r.n_pellets_gt for r in rows]
        gt_mean = float(np.mean(gt_all)) if gt_all else 0.0
        rep.agreement = (
            float(1.0 - rep.noise_floor / abs(gt_mean)) if gt_mean else None
        )
    else:
        rep.noise_floor = 0.0
        rep.noise_floor_estimable = False
        rep.agreement = None

    # ---- 配对：只取真值与自动值都有的帧（缺失跳过，不填 0）----
    paired: list[tuple[int, float, float]] = []   # (frame_idx, gt, auto)
    missing: list[int] = []
    for f in sorted(by_frame):
        rs = by_frame[f]
        gt = float(np.mean([r.n_pellets_gt for r in rs]))
        if f not in auto:
            missing.append(f)
            continue
        paired.append((f, gt, float(auto[f])))
        rep.per_frame.append(
            {
                "frame_idx": f,
                "t_s": float(rs[0].t_s),
                "gt": gt,
                "auto": float(auto[f]),
                "error": float(auto[f]) - gt,
            }
        )
    rep.n_frames = len(paired)

    if missing:
        rep.warnings.append(
            f"{len(missing)} 个真值帧缺少自动检测值（帧号 {missing[:10]}"
            f"{'…' if len(missing) > 10 else ''}）：已跳过，"
            "**不填 0、不插补**（缺失 ≠ 0 颗）"
        )
    if rep.n_frames == 0:
        rep.verdict = "无可用配对帧：误差无从实测（不输出任何精度数字）"
        return rep

    # ---- 统计量 ----
    err = np.array([a - g for _f, g, a in paired], dtype=float)
    gt_arr = np.array([g for _f, g, _a in paired], dtype=float)
    auto_arr = np.array([a for _f, _g, a in paired], dtype=float)
    rep.mae = float(np.mean(np.abs(err)))
    rep.bias = float(np.mean(err))
    rep.rmse = float(math.sqrt(float(np.mean(err * err))))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(gt_arr != 0, np.abs(err) / np.abs(gt_arr), np.nan)
    rep.mape = (
        float(np.nanmean(rel) * 100.0) if np.any(~np.isnan(rel)) else None
    )
    if rep.n_frames >= 2 and float(np.std(gt_arr)) > 0 and float(
        np.std(auto_arr)
    ) > 0:
        rep.corr = float(np.corrcoef(auto_arr, gt_arr)[0, 1])

    # ---- 纪律 3：可测性判定（实测误差必须**超过**真值噪声下限）----
    if rep.noise_floor_estimable:
        rep.measurable = bool(rep.mae > rep.noise_floor)
        if not rep.measurable:
            rep.verdict = _UNMEASURABLE_TEMPLATE
        else:
            rep.verdict = _verdict_measurable(rep)
    else:
        # 单人计数：噪声下限不可估计 → 可测性**无从判定**，
        # 按"误差 > 0 即报告"处理，但必须显式声明噪声未知（绝不假装知道）。
        rep.measurable = bool(rep.mae > 0.0)
        rep.verdict = _verdict_measurable(rep)
        rep.warnings.append(
            "真值噪声下限**不可估计**（每帧仅 1 人计数）：无法执行纪律 3 的"
            "「误差 vs 噪声」判定，本报告的 mae 可能落在真值自身噪声内。"
            "建议同一帧安排 2 人独立计数（docs/04 §7.1 纪律 2）。"
        )

    # ---- 纪律 1 / 4：小样本只报告，禁止任何自动校正 ----
    if rep.n_frames < MIN_FRAMES_FOR_CORRECTION:
        rep.warnings.append(
            f"真值帧数 {rep.n_frames} < {MIN_FRAMES_FOR_CORRECTION}："
            "**禁止**据此外推任何校正系数，也**禁止**用「减噪声」的代数方式"
            "（σ_model² ≈ MS − σ_ref²）算修正后误差——小样本下方差估计极不稳定，"
            "相减还可能得负值。本脚本只并列展示，不做代数修正"
            "（docs/04 §7.1 纪律 1/4）。"
        )
    # ---- 纪律 5：Q_n0gap 不是替代品 ----
    rep.warnings.append(
        "Q_n0gap 不是真值对照的替代品：它只能捕捉总量级粗大误差，看不见"
        "逐帧系统性比例偏差（稳定漏检 15% 时总量对得上）。两者互补，都要做"
        "（docs/04 §7.1 纪律 5）。"
    )
    return rep


def _verdict_measurable(rep: ValidationReport) -> str:
    """可测时的结论文本（区分"稳定低估/高估"与"随机抖动"）。"""
    mae, bias = rep.mae or 0.0, rep.bias or 0.0
    if mae <= 0:
        return "自动值与真值逐帧完全一致（罕见，请复核真值采集是否独立）。"
    ratio = abs(bias) / mae
    if ratio >= 0.5:
        direction = "稳定低估" if bias < 0 else "稳定高估"
        return (
            f"{direction}：有符号偏差 {bias:+.2f} 颗占 MAE {mae:.2f} 颗的 "
            f"{ratio:.0%}（≥50%），误差以系统性方向偏差为主，"
            "非随机抖动——对应难点 #9，可考虑按方向修正计数口径后复测。"
        )
    return (
        f"随机抖动为主：有符号偏差 {bias:+.2f} 颗仅占 MAE {mae:.2f} 颗的 "
        f"{ratio:.0%}（<50%），未见显著方向性系统偏差。"
    )


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def render_report_md(rep: ValidationReport, run_id: str = "") -> str:
    """渲染 validation_report.md。

    纪律 3 的展示要求：不可测量时**并列展示**噪声下限与实测误差，
    并**抑制精度数字的高亮**（用灰色样式），结论用强制模板。
    精度数字仍可见——不静默删除（删除 = 另一种隐瞒）。
    """
    style_open = f'<span style="{_DEEMPHASIS_STYLE}">'
    style_close = "</span>"
    lines: list[str] = [
        "# 真值对照误差实测报告（G1）",
        "",
        f"- run：`{run_id or '—'}`",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 参与对照帧数：**{rep.n_frames}**",
        f"- 计数人数：{rep.n_counters}",
        "",
        "## 1. 结论",
        "",
        f"> {rep.verdict}" if rep.verdict else "> （无结论：无可用配对帧）",
        "",
    ]

    if not rep.measurable and rep.n_frames > 0:
        lines.extend([
            "",
            "⚠️ 本节精度数字**不可作为精度声明**：实测误差落在真值自身噪声内，"
            "本方法在当前真值质量下**无法测量**该误差。",
            "",
        ])

    lines.extend([
        "## 2. 误差 vs 真值噪声下限（纪律 2 / 3，必须并列展示）",
        "",
        "| 量 | 值 | 说明 |",
        "|---|---|---|",
        f"| **真值噪声下限** | {_fmt(rep.noise_floor)} 颗 | "
        + (
            "同帧多人计数的平均绝对差"
            if rep.noise_floor_estimable else
            "**不可估计**（单人计数）"
        )
        + " |",
        f"| **实测 MAE** | {_fmt(rep.mae)} 颗 | 平均绝对误差 |",
        f"| 判定 | {'可测量' if rep.measurable else '**无法测量**'} | "
        "实测误差必须**超过**噪声下限才算测到 |",
        "",
    ])

    lines.extend([
        "## 3. 统计量",
        "",
        "| 统计量 | 值 |",
        "|---|---|",
    ])
    stats: list[tuple[str, Any, str]] = [
        ("MAE 平均绝对误差", rep.mae, " 颗"),
        ("MAPE 平均绝对相对误差", rep.mape, "%"),
        ("Bias 有符号偏差", rep.bias, " 颗"),
        ("RMSE 均方根误差", rep.rmse, " 颗"),
        ("Pearson 相关系数", rep.corr, ""),
        ("评分者间一致性", rep.agreement, ""),
    ]
    for name, value, unit in stats:
        shown = _fmt(value) + unit
        if not rep.measurable and name.startswith(("MAE", "MAPE")):
            shown = f"{style_open}{shown}（不可作为精度声明）{style_close}"
        lines.append(f"| {name} | {shown} |")
    lines.append("")

    if rep.per_frame:
        lines.extend([
            "## 4. 逐帧对照表",
            "",
            "| frame_idx | t_s | 真值 | 自动值 | 误差(auto−gt) |",
            "|---|---|---|---|---|",
        ])
        for row in rep.per_frame:
            lines.append(
                f"| {row['frame_idx']} | {row['t_s']:.2f} | "
                f"{row['gt']:.3f} | {row['auto']:.3f} | {row['error']:+.3f} |"
            )
        lines.append("")

    if rep.warnings:
        lines.extend(["## 5. 纪律告警", ""])
        for w in rep.warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    lines.extend([
        "## 6. 纪律声明（docs/04 §7.1 ③）",
        "",
        "1. 本报告**不施加任何校正系数**：只报告偏差，由用户决定是否采信；",
        "2. 真值噪声下限与实测误差**并列展示**，不做代数相减；",
        "3. 实测误差 < 噪声下限 → 输出「无法测量」，精度数字降级为灰色"
        "（不删除、不高亮）；",
        "4. 真值对照与 `Q_n0gap` 互补：后者看不见逐帧系统性比例偏差。",
        "",
    ])
    return "\n".join(lines)


def _fmt(v: float | None) -> str:
    """数值格式化：None → '—'（不可用绝不写成 0）。"""
    if v is None:
        return "—"
    return f"{v:.4g}"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="真值对照误差实测（G1，docs/04 §7.1 五条纪律）"
    )
    ap.add_argument("--gt", required=True,
                    help="人工真值 CSV（frame_idx,t_s,n_pellets_gt,counter）")
    ap.add_argument("--auto", required=True,
                    help="自动检测值 CSV（frame_idx,value）")
    ap.add_argument("--run-id", default="", help="run 标识（仅展示用）")
    ap.add_argument("--out", default=None,
                    help="报告输出路径（默认 validation_report.md）")
    ap.add_argument("--json-out", default=None,
                    help="结构化结果输出路径（默认 validation.json）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_gt_rows(args.gt)
        auto = load_auto_values(args.auto)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[06] 输入错误：{exc}", file=sys.stderr)
        return 2

    rep = validate(rows, auto)
    out_md = Path(args.out) if args.out else Path("validation_report.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_report_md(rep, args.run_id), encoding="utf-8")
    out_json = Path(args.json_out) if args.json_out else out_md.with_name(
        "validation.json"
    )
    out_json.write_text(
        json.dumps(rep.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"[06] 配对帧 {rep.n_frames} / 计数人 {rep.n_counters} / "
          f"噪声下限 {_fmt(rep.noise_floor)} 颗"
          f"{'' if rep.noise_floor_estimable else '（不可估计）'}")
    print(f"[06] MAE={_fmt(rep.mae)} Bias={_fmt(rep.bias)} "
          f"MAPE={_fmt(rep.mape)} corr={_fmt(rep.corr)}")
    print(f"[06] 可测量：{'是' if rep.measurable else '否'}")
    print(f"[06] 结论：{rep.verdict}")
    for w in rep.warnings:
        print(f"[06] ⚠ {w}")
    print(f"[06] 报告 → {out_md}")
    print(f"[06] 结构化 → {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

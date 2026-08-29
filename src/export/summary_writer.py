"""export/summary_writer.py · 报告落盘（T05）。

职责（docs/04 §4.3 ① + §6.1.2 ⑤ + docs/06 §6 T05 导出六件套）：
    - summary.json：完整结果（meta / run_config / quality / capability /
      metrics / warnings），JSON 可序列化；
    - capability_report.md：**人类可读**报告，必须含
        · 本次运行的质量信号总览（含 threshold/passed）；
        · **"本组因何原因未输出哪些指标"**（disabled_with_reason，强制，
          绝不静默省略——docs/04 §4.3 第 4 类工程对策 1）；
        · **flag 术语表**（docs/04 §6.1.2 ⑤）；
    - flag_glossary.csv：与 metrics_summary.csv **并列同目录**
      （只看 CSV 的用户不会去打开 .md，术语表必须在他们的动线上）。

任务编号：T05。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.metrics.capability import FLAG_GLOSSARY

__all__ = [
    "write_summary_json",
    "write_flag_glossary_csv",
    "render_capability_report_md",
    "write_capability_report_md",
]


def _jsonify(obj: Any) -> Any:
    """递归转 JSON 安全类型（numpy 标量 → Python 标量；NaN → None）。"""
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.floating):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def write_summary_json(payload: dict[str, Any], path: str | Path) -> Path:
    """写 summary.json（ensure_ascii=False 保中文可读）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonify(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_flag_glossary_csv(path: str | Path) -> Path:
    """写 flag_glossary.csv（flag_token / 中文名 / 含义 / 建议动作）。"""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["flag_token", "中文名", "含义", "建议动作"]
        )
        w.writeheader()
        for token, (zh, meaning, action) in FLAG_GLOSSARY.items():
            w.writerow({
                "flag_token": token,
                "中文名": zh,
                "含义": meaning,
                "建议动作": action,
            })
    return path


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "（空）"      # 未测得：明示，绝不用 0 冒充
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def render_capability_report_md(
    run_id: str,
    metrics_spec_version: str,
    quality_table: Sequence[dict[str, Any]],
    capability: Any | None,
    warnings: Sequence[str],
    notes: Sequence[str],
    metric_rows: Sequence[dict[str, Any]] = (),
) -> str:
    """渲染 capability_report.md（人类可读；关闭清单为强制章节）。"""
    lines: list[str] = []
    lines.append(f"# 指标能力报告 · {run_id}")
    lines.append("")
    lines.append(f"- 指标规范版本 `metrics_spec_version`: **{metrics_spec_version}**")
    lines.append("- 本报告由 `src/export/summary_writer.py` 生成（可复现）")
    lines.append("")

    # ---- ① 本组因何原因未输出哪些指标（强制，绝不静默省略）----
    lines.append("## 1. 未输出的指标及原因（强制列出）")
    lines.append("")
    closed: list[tuple[str, str, str]] = []
    if capability is not None:
        for entry in capability.disabled_with_reason:
            closed.append((entry.metric, entry.reason, entry.hint or ""))
    unavailable = [
        r for r in metric_rows
        if r.get("status") in ("unavailable", "censored")
    ]
    if not closed and not unavailable:
        lines.append("本次运行无指标被关闭。")
    else:
        if closed:
            lines.append("### 1.1 被降级矩阵关闭")
            lines.append("")
            lines.append("| 指标 / 组 | 关闭原因 | 非随机缺失提示 |")
            lines.append("|---|---|---|")
            for metric, reason, hint in closed:
                lines.append(
                    f"| `{metric}` | {reason} | {hint or '—'} |"
                )
            lines.append("")
        if unavailable:
            lines.append("### 1.2 不可用 / 右删失的指标（有行无值）")
            lines.append("")
            lines.append("| 指标 | 状态 | 原因 | 观察窗下界 |")
            lines.append("|---|---|---|---|")
            for r in unavailable:
                lines.append(
                    f"| `{r.get('metric_id')}` | {r.get('status')} | "
                    f"{r.get('reason') or '—'} | "
                    f"{_fmt_cell(r.get('window_s')) if r.get('status') == 'censored' else '—'} |"
                )
            lines.append("")
        lines.append(
            "> ⚠️ 缺失本身可能是效应：高摄食强度会带来更强的运动模糊与"
            "更密的聚集，从而降低跟踪/计数质量并触发上述关闭。"
            "跨组比较时若两组的可用指标集不同，直接比较可得部分会产生"
            "选择性偏差（docs/04 §4.3 第 4 类）。"
        )
    lines.append("")

    # ---- ② 质量信号总览（含 threshold / passed）----
    lines.append("## 2. 质量信号 Q_*（含门限与是否通过）")
    lines.append("")
    lines.append("| 信号 | 实测值 | 门限 | 通过 |")
    lines.append("|---|---|---|---|")
    for row in quality_table:
        passed = row.get("passed")
        passed_s = "—" if passed is None else ("✅" if passed else "❌")
        lines.append(
            f"| `{row.get('signal')}` | {_fmt_cell(row.get('value'))} | "
            f"{_fmt_cell(row.get('threshold'))} | {passed_s} |"
        )
    lines.append("")
    lines.append(
        '> 空值 = 未测得（不是 0，也不是「通过」）。门限取自 '
        "configs/default.yaml，随 run_config.yaml 一并留档。"
    )
    lines.append("")

    # ---- ③ 告警与备注 ----
    lines.append("## 3. 告警与备注")
    lines.append("")
    if warnings:
        lines.append("### 3.1 告警")
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.append("### 3.1 告警")
        lines.append("- 无")
    lines.append("")
    lines.append("### 3.2 备注")
    if notes:
        for n in notes:
            lines.append(f"- {n}")
    else:
        lines.append("- 无")
    lines.append("")

    # ---- ④ flag 术语表（docs/04 §6.1.2 ⑤）----
    lines.append("## 4. flag 术语表")
    lines.append("")
    lines.append("| flag | 中文名 | 含义 | 建议动作 |")
    lines.append("|---|---|---|---|")
    for token, (zh, meaning, action) in FLAG_GLOSSARY.items():
        lines.append(f"| `{token}` | {zh} | {meaning} | {action} |")
    lines.append("")
    return "\n".join(lines)


def write_capability_report_md(
    path: str | Path,
    run_id: str,
    metrics_spec_version: str,
    quality_table: Sequence[dict[str, Any]],
    capability: Any | None,
    warnings: Sequence[str],
    notes: Sequence[str],
    metric_rows: Sequence[dict[str, Any]] = (),
) -> Path:
    """写 capability_report.md。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_capability_report_md(
            run_id=run_id,
            metrics_spec_version=metrics_spec_version,
            quality_table=quality_table,
            capability=capability,
            warnings=warnings,
            notes=notes,
            metric_rows=metric_rows,
        ),
        encoding="utf-8",
    )
    return path

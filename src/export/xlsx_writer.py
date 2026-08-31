"""export/xlsx_writer.py · 汇总指标 Excel 多 sheet 导出（T05，FR-32）。

FR-32 要求：汇总指标 Excel（多 sheet：汇总指标 / 质量信号 / 人工修正记录 /
元数据 / 参数；含 flag 术语表 sheet）。本模块从 MetricsReport 直接生成 6-sheet
工作簿，与既有 CSV 导出（metrics_summary.csv 等）并列同目录。

任务编号：T05。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["write_summary_xlsx"]


def _flatten(d: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """嵌套 dict/list 展平为 (path, value) 行（参数 sheet 用）。"""
    rows: list[tuple[str, Any]] = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            rows.extend(_flatten(v, key + "."))
        elif isinstance(v, list):
            rows.append(
                (key, ", ".join(str(x) for x in v) if v else "（空）")
            )
        else:
            rows.append((key, "（空）" if v is None else v))
    return rows


def write_summary_xlsx(
    report: Any,
    run_dir: str | Path,
    path: str | Path | None = None,
) -> Path:
    """生成 6-sheet 汇总工作簿。

    Sheets:
        1. 汇总指标      —— 17 列冻结 schema（report.summary_rows()）
        2. 质量信号      —— report.quality_table
        3. 人工修正记录  —— run_dir/corrections.jsonl（无则标注）
        4. 元数据        —— run 级信息（run_id / 规范版本 / 观察窗 / 料型）
        5. 参数          —— run_config.yaml 全量（含采样/阈值/标定）
        6. flag术语表    —— FLAG_GLOSSARY（flag/中文名/含义/建议动作）

    Args:
        report: MetricsReport（需提供 summary_rows / quality_table /
            config / to_dict 接口）。
        run_dir: run 目录（用于定位 corrections.jsonl）。
        path: 输出路径；缺省为 run_dir/metrics_summary.xlsx。
    Returns:
        写出文件的 Path。
    """
    # 延迟导入：openpyxl 仅在真正导出时才需要，避免无此依赖时阻断其他导出
    from openpyxl import Workbook
    from openpyxl.styles import Font

    from src.core.checklists import (
        ChecklistsState,
        FR03_ITEMS,
        FR35_ITEMS,
    )
    from src.metrics.aggregator import SUMMARY_COLUMNS
    from src.metrics.capability import FLAG_GLOSSARY
    from src.metrics.corrections import load_corrections_jsonl

    run_dir = Path(run_dir)
    out = (
        Path(path)
        if path is not None
        else (run_dir / "metrics_summary.xlsx")
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    header_font = Font(bold=True)

    # ---- 1. 汇总指标 ----
    ws = wb.active
    ws.title = "汇总指标"
    ws.append(list(SUMMARY_COLUMNS))
    for c in ws[1]:
        c.font = header_font
    for row in report.summary_rows():
        ws.append([row.get(col, "") for col in SUMMARY_COLUMNS])

    # ---- 2. 质量信号 ----
    ws = wb.create_sheet("质量信号")
    ws.append(["signal", "value", "threshold", "passed"])
    for c in ws[1]:
        c.font = header_font
    for row in report.quality_table:
        val = row.get("value")
        thr = row.get("threshold")
        passed = row.get("passed")
        ws.append([
            row.get("signal", ""),
            "" if val is None else val,
            "" if thr is None else thr,
            "" if passed is None else ("通过" if passed else "未通过"),
        ])

    # ---- 3. 人工修正记录 ----
    ws = wb.create_sheet("人工修正记录")
    corr_path = run_dir / "corrections.jsonl"
    if corr_path.exists():
        records = load_corrections_jsonl(corr_path)
        if records:
            keys = list(records[0].keys())
            ws.append(keys)
            for c in ws[1]:
                c.font = header_font
            for rec in records:
                ws.append([rec.get(k, "") for k in keys])
        else:
            ws.append(["（无人工修正记录）"])
    else:
        ws.append(["（无 corrections.jsonl，未执行人工修正）"])

    # ---- 4. 元数据 ----
    ws = wb.create_sheet("元数据")
    full = report.to_dict()
    meta_pairs = [
        ("run_id", report.run_id),
        ("metrics_spec_version", full.get("metrics_spec_version")),
        ("window_s", full.get("window_s")),
        ("window_truncated", full.get("window_truncated")),
        ("pellet_type", getattr(report.config, "pellet_type", None)),
        ("seed", getattr(report.config, "seed", None)),
        ("feedbox_deployed", getattr(report.config, "feedbox_deployed", None)),
        # FR-08：t0 自动检测到的首颗入画相对时刻（秒）
        ("t0_auto_detect_s", full.get("t0_auto_detect_s")),
    ]
    ws.append(["字段", "值"])
    for c in ws[1]:
        c.font = header_font
    for k, v in meta_pairs:
        ws.append([k, "（空）" if v is None else v])

    # ---- 5. 参数 ----
    ws = wb.create_sheet("参数")
    ws.append(["参数路径", "值"])
    for c in ws[1]:
        c.font = header_font
    for k, v in _flatten(report.config.to_dict()):
        ws.append([k, v])

    # ---- 6. flag术语表 ----
    ws = wb.create_sheet("flag术语表")
    ws.append(["flag", "中文名", "含义", "建议动作"])
    for c in ws[1]:
        c.font = header_font
    for token, (zh, meaning, action) in FLAG_GLOSSARY.items():
        ws.append([token, zh, meaning, action])

    # ---- 7. 采集偏差与实验设计（FR-03 / FR-35）----
    ws = wb.create_sheet("采集偏差与实验设计")
    ws.append(["清单", "条目ID", "条目", "是否满足", "不做会怎样"])
    for c in ws[1]:
        c.font = header_font
    cl = getattr(report, "checklists", None)
    if isinstance(cl, ChecklistsState):
        for group, items, checked_map in (
            ("FR-03 拍摄规范", FR03_ITEMS, cl.fr03),
            ("FR-35 实验设计", FR35_ITEMS, cl.fr35),
        ):
            for it in items:
                ok = bool(checked_map.get(it.id, False))
                note = "" if ok else it.consequence
                ws.append([
                    group, it.id, it.label,
                    "是" if ok else "否（已知采集偏差）" if group.startswith("FR-03") else "否",
                    note,
                ])
    else:
        ws.append(["（未关联 FR-03/FR-35 清单，无法披露采集偏差）"])

    wb.save(out)
    return out

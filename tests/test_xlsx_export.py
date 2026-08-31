"""FR-32 · 汇总指标 Excel 多 sheet 导出测试。"""
from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

from src.core.checklists import ChecklistsState
from src.core.config import RunConfig
from src.export.xlsx_writer import write_summary_xlsx
from src.metrics.aggregator import SUMMARY_COLUMNS


class _StubReport:
    """最小可用 MetricsReport 接口桩（仅暴露 xlsx _writer 所需方法）。"""

    def __init__(self, checklists: ChecklistsState | None = None) -> None:
        self.run_id = "test_run"
        self.config = RunConfig()
        self.checklists = checklists
        self._full = {
            "metrics_spec_version": "ms-v1",
            "window_s": 120.0,
            "window_truncated": False,
        }

    def summary_rows(self):
        row = {c: "" for c in SUMMARY_COLUMNS}
        row["metric_id"] = "A1_N0"
        row["value"] = 100.0
        return [row]

    @property
    def quality_table(self):
        return [
            {"signal": "Q_det", "value": 0.9, "threshold": 0.5, "passed": True}
        ]

    def to_dict(self):
        return self._full


def test_write_summary_xlsx_six_sheets(tmp_path: Path) -> None:
    out = write_summary_xlsx(_StubReport(), tmp_path)
    assert out.exists()
    assert out.name == "metrics_summary.xlsx"

    wb = load_workbook(out)
    assert wb.sheetnames == [
        "汇总指标",
        "质量信号",
        "人工修正记录",
        "元数据",
        "参数",
        "flag术语表",
        "采集偏差与实验设计",
    ]

    # 汇总指标首行 = 17 列列头
    ws = wb["汇总指标"]
    header = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
    assert header == list(SUMMARY_COLUMNS)
    # 数据行已写入
    assert ws.max_row >= 2

    # 参数 sheet 含 pellet_type 路径
    ws_param = wb["参数"]
    rows = {r[0]: r[1] for r in ws_param.iter_rows(min_row=2, values_only=True)}
    assert "pellet_type" in rows

    # flag 术语表非空
    assert wb["flag术语表"].max_row >= 2

    # 无 corrections.jsonl 时给出明确标注
    note = list(wb["人工修正记录"].iter_rows(values_only=True))[0][0]
    assert "corrections.jsonl" in note


def test_write_summary_xlsx_checklist_sheet(tmp_path: Path) -> None:
    """FR-03/FR-35 清单 sheet 在 checklists 提供时逐条列出。"""
    cl = ChecklistsState.default()
    cl.set_item("fr03", "camera_fixed", True)  # 勾选 1 条
    out = write_summary_xlsx(_StubReport(checklists=cl), tmp_path)

    wb = load_workbook(out)
    ws = wb["采集偏差与实验设计"]
    data = list(ws.iter_rows(min_row=2, values_only=True))
    # FR-03(8) + FR-35(6) = 14 行
    assert len(data) == 14
    # 已勾选的 camera_fixed 标"是"
    camera_row = [r for r in data if r[1] == "camera_fixed"][0]
    assert camera_row[3] == "是"
    # 未勾选的某 FR-03 条目标"否（已知采集偏差）"
    unmet = [r for r in data if r[0] == "FR-03 拍摄规范" and r[3].startswith("否")]
    assert unmet  # 7 条未勾选


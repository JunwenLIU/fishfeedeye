"""FR-03 / FR-35 拍摄规范与实验设计清单测试。

覆盖：
    - 清单项数达到 PRD 下限（FR-03 ≥8，FR-35 ≥6）；
    - 默认全未勾选（披露纪律）；
    - from_dict/to_dict 往返无损 + 容错（未知项丢弃、缺失项补 False）；
    - unmet_fr03 / collection_bias markdown 内容；
    - MetricsReport.to_dict 在提供 checklists 时含已知采集偏差与实验设计状态；
    - ProjectState 持久化（写入 project.json 后重载还原）。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.app.state import ProjectState
from src.core.checklists import (
    ChecklistsState,
    FR03_ITEMS,
    FR35_ITEMS,
)
from src.metrics.aggregator import MetricsReport
from src.core.config import RunConfig


def test_fr03_has_at_least_8_items() -> None:
    assert len(FR03_ITEMS) >= 8


def test_fr35_has_at_least_6_items() -> None:
    assert len(FR35_ITEMS) >= 6


def test_default_all_unchecked() -> None:
    cl = ChecklistsState.default()
    assert all(not v for v in cl.fr03.values())
    assert all(not v for v in cl.fr35.values())
    # 默认即"全部已知采集偏差"
    assert cl.n_fr03_unmet() == len(FR03_ITEMS)


def test_set_item_and_unmet() -> None:
    cl = ChecklistsState.default()
    cl.set_item("fr03", "camera_fixed", True)
    assert cl.fr03["camera_fixed"] is True
    assert cl.n_fr03_unmet() == len(FR03_ITEMS) - 1
    unmet_ids = {it.id for it in cl.unmet_fr03_items()}
    assert "camera_fixed" not in unmet_ids


def test_to_from_dict_roundtrip() -> None:
    cl = ChecklistsState.default()
    cl.set_item("fr03", "single_take", True)
    cl.set_item("fr35", "crossover_design", True)
    d = cl.to_dict()
    restored = ChecklistsState.from_dict(d)
    assert restored.to_dict() == d
    assert restored.fr03["single_take"] is True
    assert restored.fr35["crossover_design"] is True


def test_from_dict_forgiving() -> None:
    # 缺失项补 False；未知项丢弃；绝不误判为已满足
    raw = {"fr03": {"camera_fixed": True, "ghost_item": True}, "fr35": {}}
    cl = ChecklistsState.from_dict(raw)
    assert cl.fr03["camera_fixed"] is True
    assert "ghost_item" not in cl.fr03
    # 其余 FR-03 项缺省 False
    assert cl.fr03["single_take"] is False
    # FR-35 缺省全部 False（即便 raw.fr35 为空）
    assert all(not v for v in cl.fr35.values())


def test_collection_bias_markdown_content() -> None:
    cl = ChecklistsState.default()  # 全未勾选
    md = cl.collection_bias_markdown()
    assert "已知采集偏差" in md
    # 每条未勾选项的"做不到会怎样"文本应出现在偏差说明里
    assert "单镜到底" not in md  # label 不一定出现，但 consequence 一定
    # 勾选全部后应为无偏差
    for it in FR03_ITEMS:
        cl.set_item("fr03", it.id, True)
    assert "无已知采集偏差" in cl.collection_bias_markdown()


def test_report_to_dict_includes_checklists() -> None:
    cl = ChecklistsState.default()
    cl.set_item("fr03", "camera_fixed", True)
    report = MetricsReport(run_id="r1", config=RunConfig(), checklists=cl)
    d = report.to_dict()
    assert "known_collection_bias" in d
    assert "design_checklist" in d
    bias_ids = {b["id"] for b in d["known_collection_bias"]}
    assert "camera_fixed" not in bias_ids
    assert d["known_collection_bias"]  # 仍有未勾选项
    # 每条设计清单含 checked 字段
    assert all("checked" in row for row in d["design_checklist"])
    assert len(d["design_checklist"]) == len(FR35_ITEMS)


def test_report_to_dict_no_checklists_is_empty_lists() -> None:
    report = MetricsReport(run_id="r2", config=RunConfig(), checklists=None)
    d = report.to_dict()
    assert d["known_collection_bias"] == []
    assert d["design_checklist"] == []


def test_project_state_persists_checklists(tmp_path: Path) -> None:
    proj = ProjectState(project_dir=tmp_path)
    proj.set_checklist("fr03", "size_reference", True)
    proj.set_checklist("fr35", "acclimation", True)
    # 重新从磁盘加载
    reloaded = ProjectState(project_dir=tmp_path)
    assert reloaded.checklists.fr03["size_reference"] is True
    assert reloaded.checklists.fr35["acclimation"] is True
    # 磁盘文件含 checklists 键
    raw = json.loads((tmp_path / "project.json").read_text(encoding="utf-8"))
    assert "checklists" in raw
    assert raw["checklists"]["fr03"]["size_reference"] is True

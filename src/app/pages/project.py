"""app/pages/project.py · 项目管理页签（T05）。

职责（docs/06 §6 T05）：
    - 批量导入视频（本地文件多选）；
    - 组标签 / 池塘编号（**盲法编号自动生成，UI 不显示真实分组**）；
    - RunMeta 表单（鱼种/总尾数/体长/投喂量/单颗均重/饲料类型/水温）；
    - 盲法开关 + 盲法映射表落盘（project/blind_map.enc）+ 审计日志。

任务编号：T05。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from src.app.state import ProjectState, VideoEntry

__all__ = ["build_project_page"]

META_FIELDS: list[tuple[str, str, str]] = [
    ("species", "鱼种", "text"),
    ("n_fish_total", "总尾数", "number"),
    ("body_length_mm", "体长(mm)", "number"),
    ("feed_mass_g", "投喂量(g)", "number"),
    ("pellet_mass_mg", "单颗均重(mg)", "number"),
    ("pellet_type", "饲料类型", "text"),
    ("water_temp_c", "水温(℃)", "number"),
]

TABLE_HEADERS = ["编号", "视频文件", "池塘/网箱", "状态", "run_id"]


def build_project_page(state: ProjectState) -> gr.components.Component:
    """构建"项目"页签（返回 gr.Tab 内容容器）。"""
    with gr.Column() as page:
        gr.Markdown(
            "### ① 项目管理 · 批量导入 / 分组 / 元数据\n"
            "盲法开启时，本页与分析页**只显示盲法编号**（run_001…）；"
            "真实分组写入 `project/blind_map.enc`（异或混淆），"
            "揭盲操作写入 `project/audit_log.jsonl`。"
        )

        with gr.Row():
            files = gr.File(
                label="选择视频文件（可多选）", file_count="multiple",
                file_types=["video"],
            )
            add_btn = gr.Button("导入选中视频", variant="primary")

        with gr.Row():
            # 盲法纪律：默认值与标签**不得**预填真实分组名（否则分组标签会
            # 出现在页面 HTML 里，盲法形同虚设）。留空由用户填写，值只写
            # 入 project/blind_map.enc。
            group_tb = gr.Textbox(
                label="组标签（盲法下仅写入映射表，不在页面展示）", value=""
            )
            pond_tb = gr.Textbox(
                label="池塘/网箱编号 pond_id（重复结构声明，伪重复治理必需）",
                placeholder="pond_A",
            )
            operator_tb = gr.Textbox(label="操作员", value="user")

        table = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=["str", "str", "str", "str", "str"],
            label="项目清单（盲法编号）",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("#### RunMeta 元数据（对选中行生效）")
        with gr.Row():
            row_idx = gr.Number(label="行号（0 起）", value=0, precision=0)
            apply_meta_btn = gr.Button("写入元数据")
        meta_boxes: dict[str, gr.components.Component] = {}
        with gr.Row():
            for key, label, _kind in META_FIELDS:
                meta_boxes[key] = gr.Textbox(label=label, value="")

        with gr.Row():
            blind_ck = gr.Checkbox(label="启用盲法", value=state.blind_enabled)
            reveal_btn = gr.Button("揭盲（写审计日志）")
            hide_btn = gr.Button("重新遮蔽")
            save_btn = gr.Button("保存项目")
        status_md = gr.Markdown("")

        # ------------------------------------------------------------------
        def _on_add(file_paths: Any, group: str, pond: str,
                    operator: str) -> tuple[list[list[str]], str]:
            """导入视频（去重；盲法编号自动分配）。"""
            paths = file_paths if isinstance(file_paths, list) else [file_paths]
            existing = {v.video_path for v in state.videos}
            added = 0
            for p in paths:
                path = str(getattr(p, "name", p))
                if not path or path in existing:
                    continue
                state.videos.append(VideoEntry(
                    video_path=path, group=group or "对照组",
                    pond_id=pond or "", operator=operator or "user",
                ))
                existing.add(path)
                added += 1
            if added:
                state.save()
                state.log("import", f"导入 {added} 个视频，组标签={group}")
            return state.table_rows(), f"已导入 {added} 个视频（共 {len(state.videos)}）"

        def _on_apply_meta(idx: float, *values: str) -> str:
            i = int(idx or 0)
            if i >= len(state.videos):
                return f"行号 {i} 越界（共 {len(state.videos)} 行）"
            meta: dict[str, Any] = {}
            for (key, _label, kind), val in zip(META_FIELDS, values):
                if val in (None, ""):
                    continue
                if kind == "number":
                    try:
                        meta[key] = float(val)
                    except ValueError:
                        return f"{key} 不是数字：{val!r}"
                else:
                    meta[key] = str(val)
            state.videos[i].meta.update(meta)
            state.save()
            return f"已写入第 {i} 行元数据：{meta}"

        def _on_blind_change(enabled: bool) -> tuple[list[list[str]], str]:
            state.blind_enabled = bool(enabled)
            state.save()
            state.log("blind_toggle", f"盲法={'开启' if enabled else '关闭'}")
            return state.table_rows(), f"盲法已{'开启' if enabled else '关闭'}"

        def _on_reveal(operator: str) -> tuple[list[list[str]], str]:
            state.reveal(operator=operator or "user")
            return state.table_rows(), (
                "已揭盲：分组映射解锁（操作已写入审计日志 "
                f"{state.audit_log_path.name}）"
            )

        def _on_hide(operator: str) -> tuple[list[list[str]], str]:
            state.hide(operator=operator or "user")
            return state.table_rows(), "已重新遮蔽分组映射（写入审计日志）"

        def _on_save() -> str:
            p = state.save()
            return f"项目已保存：{p}（盲法映射 {state.blind_map_path.name}）"

        # ------------------------------------------------------------------
        add_btn.click(
            _on_add, [files, group_tb, pond_tb, operator_tb], [table, status_md]
        )
        apply_meta_btn.click(
            _on_apply_meta,
            [row_idx, *[meta_boxes[k] for k, _l, _t in META_FIELDS]],
            [status_md],
        )
        blind_ck.change(_on_blind_change, [blind_ck], [table, status_md])
        reveal_btn.click(_on_reveal, [operator_tb], [table, status_md])
        hide_btn.click(_on_hide, [operator_tb], [table, status_md])
        save_btn.click(_on_save, [], [status_md])
        table.value = state.table_rows()
    return page

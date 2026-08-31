"""app/pages/compare_page.py · 分组统计对比 + 揭盲（T05）。

职责（docs/06 §6 T05）：
    - 选两组成员 runs → `export.compare.run_comparison()` 一站式完成
      "装配 → 校验（规则 1–9）→ 统计 + Holm → 衰减型偏差诊断"；
    - 六要素展示：p 值 / Cohen's d(Hedges g) / 95%CI / 检验方法名 / n /
      重复结构说明；
    - **衰减型偏差**一节（固定窗口指标被"窗口×速率"交互污染的诊断）；
    - **揭盲**（写审计日志）+ 导出（md / csv / json / 对比图）；
    - 盲法开启时，本页用中性名（组A/组B）而非真实分组标签。

⚠️ 规则 7 / reject_all 触发时**不提供强制比较按钮**（docs/04 §4.4：
给一条更好的路，而不是一个绕过防护的开关）。

任务编号：T05。
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.app.state import ProjectState

__all__ = ["build_compare_page"]

# 对比表列（六要素 + 诊断信息）
VIEW_HEADERS = [
    "指标", "中文名", "状态", "n(A/B)", "均值A", "均值B", "检验方法",
    "p", "p_holm", "效应量", "效应量口径", "95%CI", "重复结构",
    "仅描述性", "备注",
]


def _parse_dirs(text: str) -> tuple[list[str], list[str], list[str | None]]:
    """解析 run 目录列表文本 → (目录, 分组, pond_id)。

    支持三种写法：
        `路径`                  → 分组留空（由 group_pair 统一指定时不可用）
        `路径,分组`             → pond_id 从 run/meta.json 读
        `路径,分组,pond_id`     → 显式声明重复结构
    """
    dirs: list[str] = []
    groups: list[str] = []
    ponds: list[str | None] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("，", ",").split(",")]
        dirs.append(parts[0])
        groups.append(parts[1] if len(parts) > 1 else "")
        ponds.append(parts[2] if len(parts) > 2 and parts[2] else None)
    return dirs, groups, ponds


def build_compare_page(state: ProjectState) -> gr.components.Component:
    """构建"对比"页签。"""
    with gr.Column() as page:
        gr.Markdown(
            "### ⑥ 两组统计对比 · 七条规则 + 揭盲\n"
            "比较前先跑一致性校验（规则 1–9）；**不提供【强制比较】开关**"
            "——口径不一致时的 p 值是精确的废话。\n\n"
            "写法：`run目录路径,分组[,pond_id]`（每行一个）。"
        )
        with gr.Row():
            dirs_tb = gr.Textbox(
                label="run 目录列表（每行一个）", lines=8,
                placeholder=str(state.runs_root / "run_xxx") + ",A,pond_A",
            )
            with gr.Column():
                ga_tb = gr.Textbox(label="A 组标签", value="A 组")
                gb_tb = gr.Textbox(label="B 组标签", value="B 组")
                blind_ck = gr.Checkbox(
                    label="盲法显示（表中组名用中性名）",
                    value=state.blind_enabled,
                )
                op_tb = gr.Textbox(label="操作者", value="user")
                reveal_btn = gr.Button("🔓 揭盲（写审计日志）")
        compare_btn = gr.Button("执行比较", variant="primary")

        cons_md = gr.Markdown("")
        gr.Markdown("#### 统计结果（六要素：p / d / 95%CI / 方法 / n / 重复结构）")
        result_df = gr.Dataframe(
            headers=VIEW_HEADERS, datatype=["str"] * len(VIEW_HEADERS),
            label="两组比较", interactive=False, wrap=True,
        )
        gr.Markdown("#### 效应可能被低估/放大的情形（衰减型偏差诊断）")
        atten_df = gr.Dataframe(
            headers=["诊断结论"], datatype=["str"],
            label="衰减型偏差", interactive=False, wrap=True,
        )
        with gr.Row():
            out_tb = gr.Textbox(
                label="导出目录", value=str(state.project_dir / "compare")
            )
            export_btn = gr.Button("导出对比报告（md/csv/json/图表）")
        export_md = gr.Markdown("")

        # ------------------------------------------------------------------
        def _rows_from_view(view: list[dict]) -> list[list[str]]:
            """视图行 → 表格行（空值一律空字符串，绝不为 0）。"""
            rows: list[list[str]] = []
            for v in view:
                ci = (
                    f"[{v.get('ci_low')}, {v.get('ci_high')}]"
                    if v.get("ci_low") not in (None, "") else "—"
                )
                rows.append([
                    str(v.get("metric_id", "")),
                    str(v.get("metric_name_zh", "")),
                    str(v.get("status", "")),
                    f"{v.get('n_a', '')}/{v.get('n_b', '')}",
                    str(v.get("mean_a", "")),
                    str(v.get("mean_b", "")),
                    str(v.get("test_used", "")),
                    str(v.get("p_value", "")),
                    str(v.get("p_holm", "")),
                    str(v.get("effect_size", "")),
                    str(v.get("effect_size_type", "")),
                    ci,
                    str(v.get("repeat_structure", "")),
                    str(v.get("descriptive_only", "")),
                    str(v.get("reason") or v.get("warnings") or ""),
                ])
            return rows

        def _on_compare(dirs_text: str, ga: str, gb: str, blind: bool):
            from src.export.compare import (
                render_compare_report_md,
                run_comparison,
            )

            dirs, groups, ponds = _parse_dirs(dirs_text)
            if len(dirs) < 2:
                return ([], [], "至少需要 2 个 run 目录")
            # 未显式写分组时，按 A/B 组标签交替无意义 → 要求显式分组
            if not all(groups):
                return ([], [], "请为每个 run 显式指定分组（写法：`路径,分组`）")
            # 盲法：表中组名替换为中性名（真实标签不进 UI）
            pair = ("组A", "组B") if (blind and not state.revealed) else (ga, gb)
            group_map = {ga: pair[0], gb: pair[1]}
            mapped = [group_map.get(g, g) for g in groups]
            try:
                result = run_comparison(
                    dirs, groups=mapped, pond_ids=ponds,
                    group_pair=(pair[0], pair[1]),
                )
            except (FileNotFoundError, ValueError) as exc:
                return ([], [], f"比较失败：{exc}")

            rows = _rows_from_view(result.view)
            atten = [[line] for line in result.attenuation]
            state.log("compare",
                      f"{pair[0]} vs {pair[1]}：{len(rows)} 个指标，"
                      f"ok={result.ok}")
            md = render_compare_report_md(result)
            return (rows, atten, md)

        def _on_reveal(operator: str) -> str:
            state.reveal(operator=operator or "user")
            return (
                "已揭盲：分组映射解锁（写入 "
                f"{state.audit_log_path.name}）。审计日志共 "
                f"{len(state.audit_log)} 条。"
            )

        def _on_export(dirs_text: str, ga: str, gb: str, out_dir: str) -> str:
            from src.export.charts import plot_group_comparison
            from src.export.compare import (
                run_comparison,
                write_compare_outputs,
            )

            dirs, groups, ponds = _parse_dirs(dirs_text)
            if len(dirs) < 2 or not all(groups):
                return "需要 ≥2 个 run 目录且每个都指定分组"
            result = run_comparison(dirs, groups=groups, pond_ids=ponds,
                                    group_pair=(ga, gb))
            out = Path(out_dir or (state.project_dir / "compare"))
            written = write_compare_outputs(result, out)
            # 逐指标对比图（仅有可用检验的指标）
            for mid, t in result.tests.items():
                if t.get("status") != "ok":
                    continue
                vals_a, vals_b = [], []
                for b in result.runs:
                    v = b.value(mid)
                    if v is None:
                        continue
                    (vals_a if b.group == ga else vals_b).append(float(v))
                if not vals_a and not vals_b:
                    continue
                plot_group_comparison(
                    out / f"compare_{mid}.png", mid, vals_a, vals_b,
                    label_a=ga, label_b=gb, p_value=t.get("p_value"),
                    effect_size=t.get("effect_size"),
                    inferable=not t.get("descriptive_only", False),
                )
            state.log("export", f"导出对比报告 → {out}")
            return "已导出：" + "、".join(p.name for p in written) + " + 对比图"

        # ------------------------------------------------------------------
        compare_btn.click(
            _on_compare, [dirs_tb, ga_tb, gb_tb, blind_ck],
            [result_df, atten_df, cons_md],
        )
        reveal_btn.click(_on_reveal, [op_tb], [cons_md])
        export_btn.click(
            _on_export, [dirs_tb, ga_tb, gb_tb, out_tb], [export_md]
        )
    return page

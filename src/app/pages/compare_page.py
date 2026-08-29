"""app/pages/compare_page.py · 分组统计对比 + 揭盲（T05）。

职责（docs/06 §6 T05）：
    - 选两组成员 runs → 构造 TwoGroupPlan → 七条拒绝规则 → 逐指标统计；
    - 六要素展示：p 值 / Cohen's d / 95%CI / 检验方法名 / n / 重复结构说明；
    - **衰减型偏差**一节（门控当结果变量，显著即警告"效应可能被低估"）；
    - **揭盲**（写审计日志）+ 导出时可选包含/排除分组列；
    - 盲法开启时，本页对比表用盲法编号（run_001 / 组A、组B 的中性名）。

⚠️ 规则 7 触发时**不提供强制比较按钮**（docs/04 §4.4：给一条更好的路，
而不是一个绕过防护的开关）。

任务编号：T05。
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.app.state import ProjectState

__all__ = ["build_compare_page"]


def _bundles_from_dirs(state: ProjectState, dirs_text: str,
                       group_a: str, group_b: str,
                       pond_map: dict[str, str] | None = None):
    """run 目录列表文本 → (bundles, group_a, group_b)。

    pond_id 从 run 目录的 meta.json 读；缺失 → 未声明（伪重复告警）。
    """
    from src.export.compare import load_run_bundle

    lines = [x.strip() for x in (dirs_text or "").splitlines() if x.strip()]
    bundles = []
    for line in lines:
        # 支持 "路径,分组" 或 "路径" 两种写法
        if "," in line:
            path, grp = line.split(",", 1)
        else:
            path, grp = line, ""
        path = path.strip()
        grp = grp.strip()
        d = Path(path)
        if not d.exists():
            continue
        pond = None
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                pond = meta.get("pond_id") or None
            except json.JSONDecodeError:
                pond = None
        if pond is None and pond_map:
            pond = pond_map.get(d.name)
        bundles.append(load_run_bundle(d, group=grp, pond_id=pond))
    return bundles


def build_compare_page(state: ProjectState) -> gr.components.Component:
    """构建"对比"页签。"""
    with gr.Column() as page:
        gr.Markdown(
            "### ⑥ 两组统计对比 · 七条拒绝规则 + 揭盲\n"
            "比较前先跑 `run_config` 一致性校验；**不提供【强制比较】开关**"
            "（这类偏差不可察觉，绕过开关只会诱导误用）。"
        )
        with gr.Row():
            dirs_tb = gr.Textbox(
                label="run 目录列表（每行一个，可写 `路径,分组`）",
                lines=6,
                placeholder=str(state.runs_root / "run_xxx"),
            )
            with gr.Column():
                # 同上：默认值用中性名，避免真实分组名进入页面 HTML
                ga_tb = gr.Textbox(label="A 组标签", value="A 组")
                gb_tb = gr.Textbox(label="B 组标签", value="B 组")
                blind_ck = gr.Checkbox(
                    label="盲法显示（对比表用中性名 组A/组B）",
                    value=state.blind_enabled,
                )
                reveal_btn = gr.Button("🔓 揭盲（写审计日志）")
                op_tb = gr.Textbox(label="操作者", value="user")
        compare_btn = gr.Button("执行比较", variant="primary")

        cons_md = gr.Markdown("")
        gr.Markdown("#### 统计结果（六要素）")
        result_df = gr.Dataframe(
            headers=["指标", "n(A/B)", "差值", "方法", "p", "Cohen's d",
                     "95%CI", "重复结构", "可推断", "备注"],
            datatype=["str"] * 10,
            label="两组比较", interactive=False, wrap=True,
        )
        gr.Markdown("#### 效应可能被低估/放大的情形（门控当结果变量）")
        atten_df = gr.Dataframe(
            headers=["门控", "均值(A/B)", "p", "显著", "解读"],
            datatype=["str"] * 5,
            label="衰减型偏差诊断", interactive=False, wrap=True,
        )
        with gr.Row():
            out_tb = gr.Textbox(
                label="导出目录", value=str(state.project_dir / "compare")
            )
            export_btn = gr.Button("导出对比报告（md/csv/json/图表）")
        export_md = gr.Markdown("")

        # ------------------------------------------------------------------
        def _on_compare(dirs_text: str, ga: str, gb: str, blind: bool):
            from src.export.compare import (
                build_two_group_plan,
                render_compare_report_md,
                run_comparison,
            )

            bundles = _bundles_from_dirs(state, dirs_text, ga, gb)
            if len(bundles) < 2:
                return ([], [], "至少需要 2 个有效 run 目录")
            labels = {b.group for b in bundles if b.group}
            if len(labels) != 2:
                return ([], [], f"需要恰好 2 个分组标签，收到 {sorted(labels)}"
                                "（请在每行写 `路径,分组`）")
            try:
                plan = build_two_group_plan(bundles, group_a=ga, group_b=gb)
            except ValueError as exc:
                return ([], [], f"构造比较计划失败：{exc}")
            result = run_comparison(plan)

            # 盲法：中性名
            la, lb = ("组A", "组B") if (blind and not state.revealed) else (ga, gb)
            rows: list[list[str]] = []
            for t in result.tests:
                if not t.available:
                    rows.append([t.metric_id, "—", "—", "—", "—", "—", "—",
                                 "—", "—", t.reason or ""])
                    continue
                ci = (
                    f"[{t.ci_low:.4g}, {t.ci_high:.4g}]"
                    if t.ci_low is not None and t.ci_high is not None else "—"
                )
                rows.append([
                    t.metric_id,
                    f"{t.n_a}/{t.n_b}",
                    f"{t.diff:.4g}" if t.diff is not None else "—",
                    t.test_model,
                    f"{t.p_value:.4f}" if t.p_value is not None else "—",
                    f"{t.effect_size:.3f}" if t.effect_size is not None else "—",
                    ci, t.replication,
                    "是" if t.inferable else "否（仅描述性）",
                    t.reason or "",
                ])
            atten = [
                [e["gate"], f"{e['mean_a']:.4g} / {e['mean_b']:.4g}",
                 f"{e['p_value']:.4f}" if e["p_value"] is not None else "—",
                 "⚠️ 是" if e["significant"] else "否",
                 e.get("interpretation", "")]
                for e in result.attenuation
            ]
            state.log("compare", f"{la} vs {lb}：{len(rows)} 个指标")
            md = render_compare_report_md(result)
            return (rows, atten, md)

        def _on_reveal(operator: str) -> str:
            state.reveal(operator=operator or "user")
            return (
                "已揭盲：分组映射解锁（写入 "
                f"{state.audit_log_path.name}）。审计日志条目："
                f"{len(state.audit_log)} 条。"
            )

        def _on_export(dirs_text: str, ga: str, gb: str, out_dir: str) -> str:
            from src.export.charts import plot_group_comparison
            from src.export.compare import (
                build_two_group_plan,
                run_comparison,
                write_compare_outputs,
            )

            bundles = _bundles_from_dirs(state, dirs_text, ga, gb)
            if len(bundles) < 2:
                return "至少需要 2 个有效 run 目录"
            plan = build_two_group_plan(bundles, group_a=ga, group_b=gb)
            result = run_comparison(plan)
            out = Path(out_dir or (state.project_dir / "compare"))
            written = write_compare_outputs(result, out)
            # 逐指标对比图
            for t in result.tests:
                if not t.available or t.p_value is None:
                    continue
                vals_a = [b.value(t.metric_id) for b in plan.runs_a]
                vals_b = [b.value(t.metric_id) for b in plan.runs_b]
                if not any(v is not None for v in vals_a + vals_b):
                    continue
                plot_group_comparison(
                    out / f"compare_{t.metric_id}.png", t.metric_id,
                    [v for v in vals_a if v is not None],
                    [v for v in vals_b if v is not None],
                    label_a=ga, label_b=gb, p_value=t.p_value,
                    effect_size=t.effect_size, inferable=t.inferable,
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

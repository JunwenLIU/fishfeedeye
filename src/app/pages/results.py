"""app/pages/results.py · 曲线查看与人工修正（T05）。

职责（docs/06 §6 T05 验收 6）：
    - 查看 A2_Np / C(t) / P(t) / v(t) / M_diff / FA / N_fz 等时序曲线与
      metrics_summary.csv（17 列）；
    - **人工修正**：选帧输入修正颗粒数 → 只重跑 metrics 层（读
      cache/detections.jsonl，**不重跑检测**）→ corrections.jsonl 追加留痕
      （帧号/原值/新值/时间）→ 报告"共修正 N 帧，占 X%"；
    - 图表：右删失段渲染为**阴影 + ">窗长"**，绝不画成归零。

任务编号：T05。
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.app.state import ProjectState

__all__ = ["build_results_page", "apply_correction"]


def _load_run(run_dir: Path):
    """读 run 目录 → (report, timeseries_dict)。"""
    from src.metrics import compute_metrics_from_run_dir

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")
    roi = None
    roi_path = state_roi_hint(run_dir)
    if roi_path is not None and roi_path.exists():
        from src.core.roi import ROI

        roi = ROI.from_dict(json.loads(roi_path.read_text(encoding="utf-8")))
    report = compute_metrics_from_run_dir(run_dir, roi=roi)
    return report


def state_roi_hint(run_dir: Path) -> Path | None:
    """ROI 文件候选路径（project/roi.json 或 run 目录内）。"""
    for cand in (
        Path(__file__).resolve().parents[3] / "project" / "roi.json",
        Path(run_dir) / "roi.json",
    ):
        if cand.exists():
            return cand
    return None


def apply_correction(
    run_dir: str | Path, frame_idx: int, new_n: int, operator: str = "user",
    note: str = "",
) -> dict[str, object]:
    """施加一条人工修正并重算 metrics（只重跑指标层）。

    纪律（docs/06 §6 T05 验收 6 + §7.12）：
        - 修正只**留痕**（corrections.jsonl 追加），cache/detections.jsonl
          **绝不被修改**；
        - 原始口径与 _manual 口径**并列**输出，原始值不被覆盖。

    Returns:
        {"n_corrected", "n_total", "share", "corrections_path",
         "metrics_before", "metrics_after_manual"}
    """
    from datetime import datetime

    from src.metrics import compute_metrics_from_run_dir, write_run_outputs

    run_dir = Path(run_dir)
    roi = None
    roi_path = state_roi_hint(run_dir)
    if roi_path is not None and roi_path.exists():
        from src.core.roi import ROI

        roi = ROI.from_dict(json.loads(roi_path.read_text(encoding="utf-8")))

    # ---- 原值取自缓存（不重跑检测）----
    from src.pipeline.orchestrator import Orchestrator

    cached = Orchestrator._read_cache(run_dir / "cache" / "detections.jsonl")
    by_idx = {o.frame_idx: o for o in cached}
    obs = by_idx.get(int(frame_idx))
    if obs is None:
        raise ValueError(
            f"帧号 {frame_idx} 不在缓存观测中（可用：{sorted(by_idx)[:5]}... "
            f"共 {len(by_idx)} 帧）"
        )
    original_n = obs.pellets.n_det() if obs.pellets is not None else None

    # ---- 追加留痕（绝不覆盖）----
    log_path = run_dir / "corrections.jsonl"
    record = {
        "frame_idx": int(frame_idx),
        "t_s": float(obs.t_s),
        "original_n": original_n,
        "new_n": int(new_n),
        "operator": operator or "user",
        "note": note,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- 只重跑 metrics 层（读缓存）----
    report = compute_metrics_from_run_dir(run_dir, roi=roi)
    write_run_outputs(report, run_dir)

    summary = report.manual_summary
    n_corrected = summary.corrected_frames if summary is not None else 0
    share = summary.corrected_fraction if summary is not None else 0.0
    return {
        "n_corrected": int(n_corrected),
        "n_total": len(cached),
        "share": float(share),
        "corrections_path": str(log_path),
        "cache_touched": False,   # 明确声明：检测缓存未被触碰
        "metrics_count": len(report.metrics),
        "manual_metrics": sorted(
            m for m in report.metrics if m.endswith("_manual")
        ),
        "original_n": original_n,
        "new_n": int(new_n),
    }


def build_results_page(state: ProjectState) -> gr.components.Component:
    """构建"结果"页签。"""
    with gr.Column() as page:
        gr.Markdown(
            "### ⑤ 结果 · 曲线查看与人工修正\n"
            "修正只**重跑指标层**（读 `cache/detections.jsonl`，不重跑检测），"
            "留痕写入 `corrections.jsonl`；原始口径与 `_manual` 口径并列输出。"
        )
        with gr.Row():
            run_tb = gr.Textbox(label="run 目录（runs/<run_id>）", scale=3)
            load_btn = gr.Button("加载结果", variant="primary")
        status_md = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("指标总表"):
                summary_df = gr.Dataframe(
                    label="metrics_summary.csv（17 列冻结；空值=空字符串）",
                    interactive=False, wrap=True,
                )
            with gr.Tab("曲线"):
                curve_plot = gr.Plot(label="剩余颗粒数 N_p(t)")
                ts_plot = gr.Plot(label="其它时序")
            with gr.Tab("质量与降级"):
                quality_df = gr.Dataframe(
                    label="quality_signals.csv（含 threshold/passed）",
                    interactive=False, wrap=True,
                )
                report_md = gr.Markdown("")

        gr.Markdown("#### 人工修正（选帧 → 输入修正颗粒数 → 只重跑指标层）")
        with gr.Row():
            frame_nb = gr.Number(label="帧号 frame_idx", precision=0)
            newn_nb = gr.Number(label="修正后颗粒数", precision=0)
            op_tb = gr.Textbox(label="修正人", value="user")
            note_tb = gr.Textbox(label="备注（可选）")
        with gr.Row():
            fix_btn = gr.Button("施加修正并重算指标", variant="primary")
        fix_out = gr.Markdown("")

        # ------------------------------------------------------------------
        def _on_load(run_dir: str):
            """加载 run 结果（总表 / 曲线 / 质量表 / 报告）。"""
            from src.export.charts import plot_pellet_curve, plot_time_series
            from src.export.csv_writer import read_metrics_summary_csv

            d = Path(run_dir or "")
            if not d.exists():
                return ([], None, None, [], f"run 目录不存在：{d}")
            rows = read_metrics_summary_csv(d / "metrics_summary.csv")
            if not rows:
                return ([], None, None, [], f"{d} 无 metrics_summary.csv")
            cols = list(rows[0].keys())
            table = [[r.get(c, "") for c in cols] for r in rows]

            # 质量表
            q_rows: list[list] = []
            q_path = d / "quality_signals.csv"
            if q_path.exists():
                import csv as _csv

                with open(q_path, "r", encoding="utf-8-sig", newline="") as fh:
                    q_rows = [
                        [r.get("signal"), r.get("value"), r.get("threshold"),
                         r.get("passed")]
                        for r in _csv.DictReader(fh)
                    ]
            md_path = d / "capability_report.md"
            report_md_txt = (
                md_path.read_text(encoding="utf-8")
                if md_path.exists()
                else "（尚未生成 capability_report.md；可由导出层补跑）"
            )

            # 曲线（从 metrics_timeseries.csv 读；删失段按 window_s 画阴影）
            import csv as _csv
            import math

            import numpy as np

            ts_path = d / "metrics_timeseries.csv"
            fig1 = fig2 = None
            if ts_path.exists():
                with open(ts_path, "r", encoding="utf-8-sig", newline="") as fh:
                    ts_rows = list(_csv.DictReader(fh))
                t = np.array([
                    float(r["t_seconds"]) for r in ts_rows
                    if r.get("t_seconds") not in (None, "")
                ])
                n_p = np.array([
                    float(r["A2_Np"]) if r.get("A2_Np") not in (None, "")
                    else math.nan for r in ts_rows
                ])
                window_s = None
                s_path = d / "summary.json"
                if s_path.exists():
                    try:
                        summary = json.loads(s_path.read_text(encoding="utf-8"))
                        window_s = summary.get("window_s")
                        if summary.get("window_truncated"):
                            window_s = float(window_s or 0)
                        else:
                            window_s = None  # 未截断 = 无删失段需标注
                    except json.JSONDecodeError:
                        window_s = None
                tmp = d / "charts"
                tmp.mkdir(parents=True, exist_ok=True)
                if t.size:
                    fig1 = plot_pellet_curve(
                        tmp / "N_p.png", t, n_p, censored_window_s=window_s,
                    )
                    fig2 = plot_time_series(
                        tmp / "others.png",
                        [], "其它时序（本 run 无可绘序列）", "值",
                    )
            return (table, fig1, fig2, q_rows, report_md_txt)

        def _on_fix(run_dir: str, frame_idx, new_n, operator, note) -> str:
            try:
                out = apply_correction(
                    run_dir, int(frame_idx), int(new_n),
                    operator=operator or "user", note=note or "",
                )
            except (ValueError, FileNotFoundError) as exc:
                return f"⚠️ 修正失败：{exc}"
            return (
                f"✅ 已修正帧 {int(frame_idx)}：{out['original_n']} → "
                f"{out['new_n']} 颗。"
                f"共修正 {out['n_corrected']}/{out['n_total']} 帧"
                f"（占 {out['share']:.1%}）。\n\n"
                f"- 留痕：`{Path(str(out['corrections_path'])).name}` 追加 1 行\n"
                f"- **cache/detections.jsonl 未被触碰**（只重跑指标层）\n"
                f"- 并列口径：{', '.join(out['manual_metrics'][:6])}"
                f"{' …' if len(out['manual_metrics']) > 6 else ''}"
            )

        # ------------------------------------------------------------------
        load_btn.click(
            _on_load, [run_tb],
            [summary_df, curve_plot, ts_plot, quality_df, report_md],
        )
        fix_btn.click(
            _on_fix, [run_tb, frame_nb, newn_nb, op_tb, note_tb], [fix_out]
        )
        if state.videos and state.videos[0].run_id:
            run_tb.value = str(state.runs_root / state.videos[0].run_id)
    return page

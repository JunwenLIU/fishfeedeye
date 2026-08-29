"""app/pages/analysis.py · 批量分析与中断续跑（T05）。

职责（docs/06 §6 T05）：
    - 按项目清单批量执行 Orchestrator（ingest→preprocess→detect→link）
      + metrics（聚合→质量→降级）+ 导出六件套；
    - 进度回调（阶段 + 百分比）+ 中断续跑（cache/detections.jsonl 已有
      帧号跳过，不重复处理）；
    - **盲法**：本页表格与日志只出现盲法编号，绝不出现真实分组标签。

任务编号：T05。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from src.app.state import ProjectState

__all__ = ["build_analysis_page", "run_batch"]

HEADERS = ["编号", "视频", "阶段", "进度", "run_id", "状态"]


def _load_meta(state: ProjectState) -> dict[str, Any]:
    """读 RunMeta 表单（project 页写入 + 标定页的 t0）。"""
    meta: dict[str, Any] = {}
    for i, v in enumerate(state.videos):
        meta[state.blind_code(i)] = v.meta
    return meta


def run_batch(
    state: ProjectState,
    indices: list[int],
    detectors_choice: str,
    calibration: str,
    keep_images: bool,
    outdoor: bool,
    progress: gr.Progress | None = None,
) -> list[list[str]]:
    """执行批量分析（每个视频一行；返回表格行）。

    Args:
        detectors_choice: 'area' | 'auto' | 'none'。
        calibration: A_single 标定 YAML 路径（面积积分轨需要）。
        keep_images: 是否保留采样帧图像（B1/D 组需要）。
        outdoor: 户外斜拍（用户确认）→ B2/C 组降级为探索性。
    """
    from src.core.config import RunConfig
    from src.core.frame_context import RunMeta
    from src.core.roi import ROI
    from src.metrics import compute_run_metrics, write_run_outputs
    from src.pipeline.detectors.area_integral import AreaIntegralCounter
    from src.pipeline.orchestrator import Orchestrator
    from src.pipeline.pellet_dynamics import PelletDynamicsCalibration

    rows: list[list[str]] = []
    calib = None
    if calibration and Path(calibration).exists():
        try:
            calib = PelletDynamicsCalibration.from_yaml(calibration)
        except Exception as exc:  # 标定文件损坏 → 显式失败，不静默跳过
            rows.append(["—", calibration, "标定读取失败", "0%",
                         str(exc), "❌"])
            return rows

    roi = None
    roi_path = state.project_dir / "roi.json"
    if roi_path.exists():
        roi = ROI.from_dict(json.loads(roi_path.read_text(encoding="utf-8")))

    t0_cfg: dict[str, Any] = {}
    t0_path = state.project_dir / "t0.json"
    if t0_path.exists():
        t0_cfg = json.loads(t0_path.read_text(encoding="utf-8"))

    orch = Orchestrator(runs_root=state.runs_root)
    total = max(1, len(indices))
    for k, idx in enumerate(indices):
        if idx >= len(state.videos):
            continue
        entry = state.videos[idx]
        label = state.blind_code(idx)  # 盲法：只显示编号
        if progress is not None:
            progress((k) / total, desc=f"{label} 分析中")

        detectors: list[Any] = []
        if detectors_choice in ("area", "auto"):
            detectors.append(AreaIntegralCounter(
                a_single=calib.a_single if calib is not None else None
            ))
        if detectors_choice == "auto":
            try:
                from src.pipeline.detectors.yoloe_detector import YoloEDetector

                ov = YoloEDetector()
                if ov.available():
                    detectors.insert(0, ov)
            except Exception:
                pass  # 冷启动缺权重 → 静默回退面积轨（检测器层已有 reason）

        meta = RunMeta(**{
            key: val for key, val in (entry.meta or {}).items()
            if key in RunMeta.__dataclass_fields__
        }) if entry.meta else None

        cfg = RunConfig.load_default()
        if t0_cfg.get("t0_definition"):
            cfg.t0_definition = str(t0_cfg["t0_definition"])
        if t0_cfg.get("t0_source"):
            cfg.t0_source = str(t0_cfg["t0_source"])
        cal_path = state.project_dir / "calibration.json"
        if cal_path.exists():
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            if cal.get("px_per_mm_ref"):
                cfg.px_per_mm_ref = float(cal["px_per_mm_ref"])
        if calib is not None and calib.association_radius_px:
            cfg.thresholds.association_radius_px = float(
                calib.association_radius_px
            )

        try:
            from src.pipeline.pellet_linker import PelletLinker

            linker = PelletLinker(
                v_sink_max_px_s=calib.v_sink_max_px_s if calib else None,
                association_radius_px=(
                    calib.association_radius_px if calib else None
                ),
                roi=roi,
            )
            result = orch.run(
                video_path=entry.video_path,
                meta=meta,
                config=cfg,
                roi=roi,
                t0_s=t0_cfg.get("t0_s"),
                detectors=detectors,
                linker=linker if detectors else None,
                keep_images=bool(keep_images),
                progress=lambda stage, frac, _lbl=label: None,
            )
            entry.run_id = result.run_id
            if meta is not None:
                report = compute_run_metrics(result, roi=roi, outdoor=outdoor)
                write_run_outputs(report, result.run_dir)
                (result.run_dir / "meta.json").write_text(
                    json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                entry.status = "已完成（含指标）"
                rows.append([label, Path(entry.video_path).name, "done", "100%",
                             result.run_id, "✅"])
            else:
                entry.status = "已分析（无元数据，未出指标）"
                rows.append([label, Path(entry.video_path).name, "done", "100%",
                             result.run_id, "⚠️ 无 RunMeta"])
        except Exception as exc:  # 单个视频失败不中断整批（记录原因）
            entry.status = f"失败：{exc}"
            rows.append([label, Path(entry.video_path).name, "error", "0%",
                         "", f"❌ {exc}"])
        state.log("analyze", f"{label} 分析完成：{entry.status}")
    if progress is not None:
        progress(1.0, desc="批量分析结束")
    state.save()
    return rows


def build_analysis_page(state: ProjectState) -> gr.components.Component:
    """构建"分析"页签。"""
    with gr.Column() as page:
        gr.Markdown(
            "### ③ 批量分析 · 进度 / 中断续跑\n"
            "中断后重跑会自动跳过 `cache/detections.jsonl` 中已完成的帧"
            "（不重复处理）。**本页只显示盲法编号**，不出现分组标签。"
        )
        with gr.Row():
            range_tb = gr.Textbox(
                label="待分析行号（逗号分隔，空=全部）",
                placeholder="0,1,2",
            )
            det_rd = gr.Radio(
                label="检测轨", choices=["area", "auto", "none"], value="area",
            )
            calib_tb = gr.Textbox(
                label="A_single 标定 YAML（scripts/00 产出，可空）",
                placeholder="configs/calibration/feedA.yaml",
            )
        with gr.Row():
            keep_ck = gr.Checkbox(label="保留采样帧图像（B1/D 组需要，耗内存）",
                                  value=False)
            outdoor_ck = gr.Checkbox(
                label="户外斜拍机位（B2/C 组降级为探索性）", value=True
            )
        run_btn = gr.Button("开始批量分析", variant="primary")
        table = gr.Dataframe(
            headers=HEADERS,
            datatype=["str", "str", "str", "str", "str", "str"],
            label="分析进度（盲法编号）", interactive=False, wrap=True,
        )
        log_md = gr.Markdown("")

        def _on_run(range_text: str, det: str, calib: str, keep: bool,
                    outdoor: bool, prog: gr.Progress = gr.Progress()):
            text = (range_text or "").strip()
            if text:
                try:
                    idxs = [int(x) for x in text.replace("，", ",").split(",")
                            if x.strip()]
                except ValueError:
                    return [], "行号解析失败：请填逗号分隔的整数"
            else:
                idxs = list(range(len(state.videos)))
            idxs = [i for i in idxs if 0 <= i < len(state.videos)]
            if not idxs:
                return [], "无有效行号（项目清单为空？请先在①项目页导入）"
            rows = run_batch(state, idxs, det, calib, keep, outdoor,
                             progress=prog)
            n_ok = sum(1 for r in rows if r[5].startswith("✅"))
            return rows, f"完成 {n_ok}/{len(rows)}（runs 目录：{state.runs_root}）"

        run_btn.click(
            _on_run, [range_tb, det_rd, calib_tb, keep_ck, outdoor_ck],
            [table, log_md],
        )
        table.value = [
            [state.blind_code(i), Path(v.video_path).name, v.pond_id,
             v.status, v.run_id, ""]
            for i, v in enumerate(state.videos)
        ]
    return page

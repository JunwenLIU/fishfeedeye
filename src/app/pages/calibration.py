"""app/pages/calibration.py · ROI + 透视标定向导 + t0 打点（T05）。

职责（docs/06 §6 T05 内联约定）：
    - 标定向导**先问"有无投喂框"**：
        · 有 → 点四角 + 填实际尺寸（浮动投喂框为**主标定物**，
          四角即单应性四点 + pellet_zone 边界）；
        · 无 → 走尺寸参照物备份路径（两点 + 实际长度 → px_per_mm_ref）；
    - **t0 打点三选一**（投饵器启动 / 饲料离开投饵器 / 饲料出现在画面）
      ——默认第三项 `pellet_in_frame`；`t0_definition` 与 `t0_source`
      都写入 run_config（docs/04 §1 ③，compare 规则必校项）；
    - UI 必填下拉框 + 自动判据预填默认值（docs/04 §4.0 UI 要求：纯提示
      用户一定会跳过，那这些字段就永远是 UNKNOWN）。

任务编号：T05。
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.app.state import ProjectState
from src.core.homography import HomographyCalibrator

__all__ = ["build_calibration_page", "T0_DEFINITIONS"]

# t0 定义三选一（docs/04 §1 ②；默认 pellet_in_frame）
T0_DEFINITIONS: list[tuple[str, str]] = [
    ("pellet_in_frame", "C · 第一颗饲料出现在画面/入水（默认，客观可复现）"),
    ("feeder_start", "A · 投饵器启动（电机开始转动）"),
    ("pellet_released", "B · 第一颗饲料离开投饵器"),
]


def build_calibration_page(state: ProjectState) -> gr.components.Component:
    """构建"标定"页签。"""
    with gr.Column() as page:
        gr.Markdown(
            "### ② 标定 · ROI / 透视 / t0 定义\n"
            "**先选标定路径**：有浮动投喂框 → 点四角（主路径，四角即单应性"
            "四点 + 颗粒计数区边界）；无 → 用尺寸参照物走备份路径。"
        )

        with gr.Row():
            video_tb = gr.Textbox(label="视频路径", scale=3)
            frame_no = gr.Number(label="抽帧序号（用于标定取图）", value=0,
                                 precision=0)
            load_btn = gr.Button("抽取标定帧")
        calib_img = gr.Image(label="标定帧（点击选取点位）", interactive=False)

        with gr.Row():
            has_frame_rd = gr.Radio(
                label="是否布设了浮动投喂框？",
                choices=["有（主路径：四角标定）", "无（备份路径：尺寸参照物）"],
                value="有（主路径：四角标定）",
            )
        with gr.Row(visible=True) as frame_row:
            c1 = gr.Textbox(label="角点1 (x,y)", placeholder="100,80")
            c2 = gr.Textbox(label="角点2 (x,y)", placeholder="540,80")
            c3 = gr.Textbox(label="角点3 (x,y)", placeholder="540,400")
            c4 = gr.Textbox(label="角点4 (x,y)", placeholder="100,400")
            fw = gr.Number(label="框实际宽 (m)", value=1.0)
            fh = gr.Number(label="框实际高 (m)", value=0.8)
        with gr.Row(visible=False) as ref_row:
            p1 = gr.Textbox(label="参照物端点1 (x,y)", placeholder="100,200")
            p2 = gr.Textbox(label="参照物端点2 (x,y)", placeholder="300,200")
            ref_len = gr.Number(label="参照物实际长度 (mm)", value=100.0)

        fit_btn = gr.Button("执行标定", variant="primary")
        calib_out = gr.Markdown("")

        gr.Markdown("#### t0 定义与来源（compare 必校项）")
        with gr.Row():
            t0_def = gr.Dropdown(
                label="t0 定义（三选一，默认 C）",
                choices=[d for _v, d in T0_DEFINITIONS],
                value=T0_DEFINITIONS[0][1],
            )
            t0_src = gr.Dropdown(
                label="t0 来源",
                choices=["manual（人工统一打点，跨组比较主场景）",
                         "auto（自动检测，仅单次分析）"],
                value="manual（人工统一打点，跨组比较主场景）",
            )
        with gr.Row():
            t0_s = gr.Number(label="t0 在视频内的绝对时刻 (s)", value=0.0)
            save_btn = gr.Button("写入 run_config（t0_definition / t0_source）")
        t0_out = gr.Markdown("")

        # ------------------------------------------------------------------
        def _on_load(video: str, idx: float):
            """抽取标定帧（无 cv2/文件缺失 → 显式报错，不静默）。"""
            import cv2

            path = Path(video or "")
            if not path.exists():
                return None, f"视频不存在：{path}"
            cap = cv2.VideoCapture(str(path))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx or 0))
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                return None, f"抽帧失败（序号 {int(idx or 0)}）"
            return frame[:, :, ::-1], f"已抽取第 {int(idx or 0)} 帧"

        def _on_toggle(choice: str):
            has = choice.startswith("有")
            return gr.Row(visible=has), gr.Row(visible=not has)

        @staticmethod
        def _parse_pt(text: str, name: str) -> tuple[float, float]:
            raw = (text or "").replace("，", ",").strip()
            parts = [x for x in raw.split(",") if x.strip()]
            if len(parts) != 2:
                raise ValueError(f"{name} 需形如 'x,y'，收到 {text!r}")
            return float(parts[0]), float(parts[1])

        def _on_fit(choice: str, a1: str, a2: str, a3: str, a4: str,
                    width_m: float, height_m: float,
                    r1: str, r2: str, ref_mm: float) -> str:
            """执行标定：四角单应性（主）或两点尺度（备份）。"""
            try:
                if choice.startswith("有"):
                    corners = [
                        _parse_pt(a1, "角点1"), _parse_pt(a2, "角点2"),
                        _parse_pt(a3, "角点3"), _parse_pt(a4, "角点4"),
                    ]
                    w_m = float(width_m or 0)
                    h_m = float(height_m or 0)
                    if w_m <= 0 or h_m <= 0:
                        return "框实际尺寸必须为正（标定物尺寸是 px→mm 的唯一依据）"
                    world = [
                        [0.0, 0.0], [w_m, 0.0], [w_m, h_m], [0.0, h_m],
                    ]
                    cal = HomographyCalibrator()
                    ok = cal.fit(corners, world)
                    if not ok:
                        return "单应性标定失败：四点退化（共线或重复点）"
                    px_per_mm = cal.px_per_mm(*corners[0])
                    lines = [
                        "✅ 四角单应性标定成功（主路径）",
                        f"- px_per_mm（角点1 处）≈ {px_per_mm:.4f}",
                        f"- 尺度场可用：px_per_mm(x, y) 随位置变化"
                        f"（斜拍场景下不同位置尺度不同）",
                        "",
                        "⚠️ **能力边界声明**：单应性只修**尺度**，不修**遮挡**"
                        "——被鱼体挡住的颗粒、反光区、重叠个体，标定无法补救。",
                    ]
                else:
                    pa = _parse_pt(r1, "参照物端点1")
                    pb = _parse_pt(r2, "参照物端点2")
                    length_px = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                    mm = float(ref_mm or 0)
                    if mm <= 0 or length_px <= 1e-6:
                        return "参照物实际长度必须为正，且两点不得重合"
                    px_per_mm = length_px / mm
                    lines = [
                        "✅ 尺寸参照物标定成功（备份路径）",
                        f"- 参照物像素长度 {length_px:.2f}px / {mm:.1f}mm "
                        f"→ px_per_mm = {px_per_mm:.4f}",
                        "",
                        "⚠️ 备份路径只给出**单一尺度**，不建模斜拍的尺度场；"
                        "画面不同位置的尺度误差未被校正。",
                    ]
            except ValueError as exc:
                return f"标定输入有误：{exc}"

            state.project_dir.mkdir(parents=True, exist_ok=True)
            out = state.project_dir / "calibration.json"
            out.write_text(json.dumps(
                {"px_per_mm_ref": px_per_mm, "method": (
                    "homography_4pt" if choice.startswith("有") else "reference_2pt"
                )},
                ensure_ascii=False, indent=2,
            ), encoding="utf-8")
            state.log("calibrate", f"px_per_mm={px_per_mm:.4f}（{out.name}）")
            lines.append(f"\n已写入 `{out}`。")
            return "\n".join(lines)

        def _on_save_t0(def_label: str, src_label: str, t0: float) -> str:
            definition = next(
                (v for v, d in T0_DEFINITIONS if d == def_label), "pellet_in_frame"
            )
            source = "manual" if src_label.startswith("manual") else "auto"
            state.project_dir.mkdir(parents=True, exist_ok=True)
            out = state.project_dir / "t0.json"
            out.write_text(json.dumps(
                {"t0_definition": definition, "t0_source": source,
                 "t0_s": float(t0 or 0.0)},
                ensure_ascii=False, indent=2,
            ), encoding="utf-8")
            state.log("t0", f"definition={definition}, source={source}, "
                            f"t0={t0}")
            return (
                f"已写入 `{out.name}`：t0_definition=`{definition}`，"
                f"t0_source=`{source}`，t0={float(t0 or 0.0):.2f}s。"
                "这两项随 run_config 留档，跨组比较时由 compare 规则硬校验"
                "（不同定义的两批数据不可直接比较）。"
            )

        # ------------------------------------------------------------------
        load_btn.click(_on_load, [video_tb, frame_no], [calib_img, calib_out])
        has_frame_rd.change(_on_toggle, [has_frame_rd], [frame_row, ref_row])
        fit_btn.click(
            _on_fit,
            [has_frame_rd, c1, c2, c3, c4, fw, fh, p1, p2, ref_len],
            [calib_out],
        )
        save_btn.click(_on_save_t0, [t0_def, t0_src, t0_s], [t0_out])
    return page

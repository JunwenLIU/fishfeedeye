"""app/main.py · Gradio UI 入口（T05）。

职责（docs/06 §6 T05 验收 1）：
    六个页签组装 + 全局盲法开关：
        ① 项目管理（批量导入 / 组标签 / 盲法编号 / 元数据表单）
        ② 标定（ROI + 透视标定向导 + t0 定义三选一）
        ③ 批量分析（进度 / 中断续跑）
        ④ 打点计数（P0，独立证据链）
        ⑤ 结果（曲线查看 + 人工修正，只重跑 metrics 层）
        ⑥ 两组对比（七条拒绝规则 + 揭盲）

启动：
    python -m src.app.main            （默认 127.0.0.1:7860）
    python -m src.app.main --share    （需要外网访问时）
    python -m src.app.main --port 7861

盲法：分析页 HTML 中不得出现真实分组标签（只显示 run_001…）；
分组映射表 project/blind_map.enc（异或混淆），揭盲写 project/audit_log.jsonl。

任务编号：T05。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from src.app.pages.analysis import build_analysis_page
from src.app.pages.calibration import build_calibration_page
from src.app.pages.compare_page import build_compare_page
from src.app.pages.project import build_project_page
from src.app.pages.results import build_results_page
from src.app.pages.tally import build_tally_page
from src.app.state import ProjectState

__all__ = ["build_ui", "main", "TAB_TITLES"]

TAB_TITLES: list[str] = [
    "① 项目", "② 标定", "③ 分析", "④ 打点计数", "⑤ 结果", "⑥ 两组对比",
]


def build_ui(state: ProjectState | None = None) -> gr.Blocks:
    """构建 Gradio Blocks（六页签）。

    Args:
        state: 项目状态（共享；缺省新建，落 project/ 目录）。
    """
    st = state if state is not None else ProjectState()
    with gr.Blocks(
        title="鱼类摄食行为视频分析工具",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# 鱼类摄食行为视频分析工具\n"
            "户外池塘 · 浮性膨化料 · 科研级诱食性对比实验\n\n"
            "**主结论建立在 A 组颗粒曲线上**；鱼体相关指标（B2/C）在户外"
            "斜拍下默认降级为探索性输出。所有不可用指标一律输出"
            "`unavailable + reason`，**绝不用 0 冒充**。"
        )
        with gr.Tabs():
            with gr.Tab(TAB_TITLES[0]):
                build_project_page(st)
            with gr.Tab(TAB_TITLES[1]):
                build_calibration_page(st)
            with gr.Tab(TAB_TITLES[2]):
                build_analysis_page(st)
            with gr.Tab(TAB_TITLES[3]):
                build_tally_page(st)
            with gr.Tab(TAB_TITLES[4]):
                build_results_page(st)
            with gr.Tab(TAB_TITLES[5]):
                build_compare_page(st)
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="鱼类摄食行为视频分析工具（Gradio）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成公网访问链接")
    ap.add_argument("--project-dir", default=None,
                    help="项目目录（默认 <项目根>/project）")
    ap.add_argument("--no-browser", action="store_true",
                    help="不自动打开浏览器（测试/CI 用）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_dir = (
        Path(args.project_dir)
        if args.project_dir
        else Path(__file__).resolve().parents[2] / "project"
    )
    state = ProjectState(project_dir=project_dir)
    demo = build_ui(state)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=bool(args.share),
        inbrowser=not args.no_browser,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

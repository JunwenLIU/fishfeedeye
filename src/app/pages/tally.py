"""app/pages/tally.py · 打点计数工具（T05，第二轮决策④从 P1 提到 P0）。

职责（docs/06 §3.2 + §6 T05 验收 3）：
    - 回放视频 + **大按钮**（点击 = 一次目击摄食事件）；
    - 时间戳来源：点"开始打点"后与视频播放**同时起表**（单调时钟），
      每次点击记录距起表的秒数 —— 保证时间戳单调，且不需要 Gradio 暴露
      播放器内部时间（docs/06 §4 UNCLEAR ① 的保底方案：鼠标大按钮）；
      也可手动改写时间戳框（如按帧号回看定位）；
    - 输出 manual_counts.csv（event_id/t_s/phase/operator/created_at/note），
      与自动曲线**并列展示、永不合并**；
    - 打点人/时间写入文件头，多次打点取最后一次为当前版，历史留痕。

纪律（docs/06 §3.2）：打点结果**不修正自动指标**，只并列。

任务编号：T05。
"""
from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from src.app.state import ProjectState
from src.export.csv_writer import TallyRecorder

__all__ = ["build_tally_page", "TallySession", "PHASES"]

PHASES = ["基线（投喂前）", "投喂期", "尾段"]


class TallySession:
    """一次打点会话（起表 + 记录 + 落盘）。

    时间戳 = 距"开始打点"的秒数（单调时钟，与视频播放同步起表），
    因此天然单调递增；不依赖 Gradio 暴露播放器内部时间。
    """

    def __init__(self, operator: str = "user", video_name: str = "") -> None:
        self.recorder = TallyRecorder(operator=operator, video_name=video_name)
        self.t_start: float | None = None
        self.phase: str = PHASES[1]
        self.saved_path: str = ""

    def start(self) -> str:
        """起表（与视频播放同时）。"""
        self.t_start = time.monotonic()
        self.recorder.events = []
        return "已起表：请开始播放视频，每次目击摄食点击【打点】"

    def stop(self) -> str:
        if self.t_start is None:
            return "尚未起表"
        elapsed = time.monotonic() - self.t_start
        self.t_start = None
        return f"已停止：共 {len(self.recorder.events)} 次打点，时长 {elapsed:.1f}s"

    def click(self, manual_ts: float | None = None) -> tuple[float, int]:
        """记录一次打点。Returns: (时间戳, 事件序号)。"""
        if self.t_start is None:
            raise RuntimeError("请先点击【开始打点】起表（时间戳需与视频同步）")
        ts = float(manual_ts) if manual_ts not in (None, "") else (
            time.monotonic() - self.t_start
        )
        event_id = self.recorder.on_key(ts, phase=self.phase)
        return float(ts), int(event_id)

    def undo(self) -> int:
        """撤销最后一次打点（留痕：不静默）。Returns: 剩余事件数。"""
        if self.recorder.events:
            self.recorder.events.pop()
        return len(self.recorder.events)

    def to_csv(self, path: str | Path) -> Path:
        p = self.recorder.to_csv(path)
        self.saved_path = str(p)
        return p


def build_tally_page(state: ProjectState) -> gr.components.Component:
    """构建"打点计数"页签。"""
    session = TallySession(operator="user")

    with gr.Column() as page:
        gr.Markdown(
            "### ④ 打点计数 · 人工真值锚点（独立证据链）\n"
            "回放视频，每次目击【鱼吃一颗/一口】点击一次【打点】。\n"
            "⚠️ **打点结果不会修正任何自动指标**，只与自动曲线**并列**展示"
            "（导出时独立成节），用于误差实测（G1）与审稿答辩。"
        )
        with gr.Row():
            video_in = gr.Video(label="回放视频", interactive=True)
        with gr.Row():
            operator_tb = gr.Textbox(label="打点人", value="user")
            phase_rd = gr.Radio(label="阶段", choices=PHASES, value=PHASES[1])
            start_btn = gr.Button("▶ 开始打点（与播放同时起表）",
                                  variant="primary")
            stop_btn = gr.Button("⏹ 停止")

        with gr.Row():
            tally_btn = gr.Button(
                "🔴 打点（目击一次摄食）", variant="primary", size="lg",
                scale=3,
            )
            ts_nb = gr.Number(
                label="时间戳覆盖（留空=用起表计时）", value=None
            )
            undo_btn = gr.Button("撤销最后一次")

        with gr.Row():
            count_md = gr.Markdown("**已打点：0 次**")
            last_md = gr.Markdown("最近一次：—")

        events_df = gr.Dataframe(
            headers=["event_id", "t_s", "phase", "operator", "created_at"],
            datatype=["number", "number", "str", "str", "str"],
            label="打点事件（时间戳单调递增）", interactive=False, wrap=True,
        )
        with gr.Row():
            out_tb = gr.Textbox(
                label="manual_counts.csv 输出路径",
                value=str(state.project_dir / "manual_counts.csv"),
            )
            save_btn = gr.Button("导出 manual_counts.csv", variant="primary")
        status_md = gr.Markdown("")

        # ------------------------------------------------------------------
        def _on_start(operator: str) -> str:
            session.recorder.operator = operator or "user"
            return session.start()

        def _on_click(manual_ts, phase) -> tuple[str, str, list[list]]:
            session.phase = phase or PHASES[1]
            try:
                ts, event_id = session.click(manual_ts)
            except RuntimeError as exc:
                return f"⚠️ {exc}", str(len(session.recorder.events)), _rows()
            return (
                f"**已打点：{len(session.recorder.events)} 次**",
                f"最近一次：#{event_id} @ {ts:.3f}s（{session.phase}）",
                _rows(),
            )

        def _rows() -> list[list]:
            return [
                [ev.event_id, round(ev.t_s, 6), ev.phase, ev.operator,
                 ev.created_at]
                for ev in sorted(session.recorder.events, key=lambda e: e.t_s)
            ]

        def _on_undo() -> tuple[str, str, list[list]]:
            n = session.undo()
            return (
                f"**已打点：{n} 次**",
                f"已撤销一次（剩余 {n} 次）",
                _rows(),
            )

        def _on_save(path: str, operator: str) -> str:
            p = Path(path or (state.project_dir / "manual_counts.csv"))
            session.recorder.operator = operator or "user"
            written = session.to_csv(p)
            state.log("tally",
                      f"导出 manual_counts.csv：{len(session.recorder.events)} "
                      f"次打点 → {written}")
            return (
                f"已导出 `{written}`（{len(session.recorder.events)} 行）。"
                "该文件与自动颗粒曲线**并列展示、永不合并**；"
                "可用 `scripts/06_validate_against_groundtruth.py` 做误差实测。"
            )

        # ------------------------------------------------------------------
        start_btn.click(_on_start, [operator_tb], [status_md])
        stop_btn.click(lambda: session.stop(), [], [status_md])
        tally_btn.click(
            _on_click, [ts_nb, phase_rd], [count_md, last_md, events_df]
        )
        undo_btn.click(_on_undo, [], [count_md, last_md, events_df])
        save_btn.click(_on_save, [out_tb, operator_tb], [status_md])
    return page


# ----------------------------------------------------------------------
# 供测试/脚本直接调用的无 UI 接口（点击 20 次 → 20 行）
# ----------------------------------------------------------------------
def simulate_clicks(
    n: int, operator: str = "user", video_name: str = "demo.mp4",
    dt: float = 0.5,
) -> TallySession:
    """模拟 n 次打点（时间戳按 dt 递增，用于测试与演示）。

    Args:
        n: 点击次数。
        dt: 相邻两次的时间间隔（秒）。

    Returns:
        TallySession（session.recorder.events 恰 n 条，时间戳单调递增）。
    """
    sess = TallySession(operator=operator, video_name=video_name)
    sess.t_start = 0.0
    for i in range(n):
        sess.recorder.on_key(round(i * dt, 6), phase=PHASES[1])
    return sess

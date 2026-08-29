"""时间轴构建与时间基准风险检测（T01）。

职责（docs/04 §1 时间基准约定 + docs/06 T01 内联约定）：
    时间轴一律逐帧 PTS/POS_MSEC 优先，容器标称 FPS 仅兜底。检测：
      ① VFR：标称时长 vs 实际解码时长偏差 > 2% → timing_suspect=True；
      ② 拼接/中断续录：POS_MSEC 非严格单调递增 → timeline_discontinuity=True，
         断裂点之后的时间指标一律不输出（返回 valid_for_time_metrics 掩码）；
      ③ 跳变：相邻帧间隔 > 中位数 3 倍 → 判为断裂位置；
      ④ 慢动作/延时：文件内部无法检测，仅预留 timing_scale_unknown 标记位
         （由 UI 询问用户后填充，docs/04 §1 ⑤）。

使用方式：
    ts = collect_frame_timestamps(video_path)          # utils/video_io.py
    report = build_timeline(ts, fps)                    # 本模块
    if report.timeline_discontinuity:
        mask = report.frames_valid_for_time_metrics()   # 断裂后帧 False
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["TimingReport", "build_timeline", "build_timeline_from_nominal", "DURATION_DEVIATION_TOLERANCE"]

# VFR 检测容差：标称时长 vs 解码时长相对偏差 > 2% → timing_suspect（契约值）
DURATION_DEVIATION_TOLERANCE: float = 0.02
# 跳变判定：相邻帧间隔 > 中位数 × 3 → 断裂（契约值）
GAP_MEDIAN_FACTOR: float = 3.0


@dataclass
class TimingReport:
    """时间轴体检报告。"""

    n_frames: int
    timing_source: str                     # 'pos_msec' | 'nominal_fps'
    duration_s: float                      # 实际解码时长（首帧到末帧）
    nominal_duration_s: float              # 标称时长（帧数/标称FPS）
    duration_deviation: float              # |实际-标称|/标称（相对）
    timing_suspect: bool = False           # VFR / 时长偏差告警
    timeline_discontinuity: bool = False   # 非单调（拼接）告警
    break_indices: list[int] = field(default_factory=list)   # 断裂帧号（0-based）
    break_times_s: list[float] = field(default_factory=list)  # 断裂处时间戳
    notes: list[str] = field(default_factory=list)
    t_s: np.ndarray | None = None          # 逐帧时间戳（秒），供下游消费

    def frames_valid_for_time_metrics(self) -> list[bool]:
        """逐帧有效性掩码：首个断裂点之后一律 False（断裂后时间指标不输出）。"""
        if not self.timeline_discontinuity or not self.break_indices:
            return [True] * self.n_frames
        first_break = min(self.break_indices)
        return [i < first_break for i in range(self.n_frames)]

    def to_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "timing_source": self.timing_source,
            "duration_s": self.duration_s,
            "nominal_duration_s": self.nominal_duration_s,
            "duration_deviation": self.duration_deviation,
            "timing_suspect": self.timing_suspect,
            "timeline_discontinuity": self.timeline_discontinuity,
            "break_indices": list(self.break_indices),
            "break_times_s": list(self.break_times_s),
            "notes": list(self.notes),
        }


def build_timeline(
    pos_msec: Sequence[float],
    nominal_fps: float,
) -> TimingReport:
    """由逐帧 PTS/POS_MSEC 构建时间轴并体检（唯一可信源路径）。

    Args:
        pos_msec: 逐帧时间戳（毫秒，cv2.CAP_PROP_POS_MSEC 顺序解码收集）。
        nominal_fps: 容器标称帧率（仅用于时长交叉校验，不用于推算时间轴）。

    Returns:
        TimingReport；t_s 为秒制逐帧时间戳。
    """
    t = np.asarray(pos_msec, dtype=float) / 1000.0
    n = int(t.shape[0])
    if n == 0:
        raise ValueError("时间戳序列为空")
    if nominal_fps <= 0:
        raise ValueError(f"标称帧率必须为正，收到 {nominal_fps}")

    notes: list[str] = []
    duration_s = float(t[-1] - t[0]) if n > 1 else 0.0
    # 标称时长 = (n-1) / fps（首帧到末帧的口径，与解码口径对齐）
    nominal_duration_s = (n - 1) / float(nominal_fps)
    deviation = (
        abs(duration_s - nominal_duration_s) / nominal_duration_s
        if nominal_duration_s > 0
        else 0.0
    )
    timing_suspect = bool(deviation > DURATION_DEVIATION_TOLERANCE)
    if timing_suspect:
        notes.append(
            f"VFR 疑似：标称时长 {nominal_duration_s:.3f}s vs 解码时长 "
            f"{duration_s:.3f}s，相对偏差 {deviation:.1%} > 2%；"
            "时间类指标可能存在系统性偏差，须逐帧 PTS 核实"
        )

    break_indices: list[int] = []
    break_times: list[float] = []
    discontinuity = False
    if n > 1:
        dt = np.diff(t)
        # ① 非严格单调递增（回退/重置 → 拼接续录）
        nonmono = np.where(dt <= 0)[0]
        for i in nonmono:
            break_indices.append(int(i + 1))  # 折回发生在第 i+1 帧处
            break_times.append(float(t[i + 1]))
            discontinuity = True
        if nonmono.size:
            notes.append(
                f"时间戳非单调（拼接/续录重置）：共 {nonmono.size} 处回退，"
                "断裂点之后的时间指标一律不输出"
            )
        # ② 跳变：间隔 > 中位数 × 3
        if dt.size >= 3:
            med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
            if med > 0:
                jumps = np.where(dt > med * GAP_MEDIAN_FACTOR)[0]
                for i in jumps:
                    idx = int(i + 1)
                    if idx not in break_indices:
                        break_indices.append(idx)
                        break_times.append(float(t[idx]))
                    discontinuity = True
                if jumps.size:
                    notes.append(
                        f"帧间隔跳变：{jumps.size} 处间隔 > 中位数×{GAP_MEDIAN_FACTOR:g}"
                    )

    return TimingReport(
        n_frames=n,
        timing_source="pos_msec",
        duration_s=duration_s,
        nominal_duration_s=nominal_duration_s,
        duration_deviation=float(deviation),
        timing_suspect=timing_suspect,
        timeline_discontinuity=discontinuity,
        break_indices=sorted(break_indices),
        break_times_s=break_times,
        notes=notes,
        t_s=t,
    )


def build_timeline_from_nominal(n_frames: int, nominal_fps: float) -> TimingReport:
    """标称 FPS 兜底路径（拿不到逐帧 PTS 时）。

    契约要求：标 timing_source='nominal_fps'，报告显式提示
    「若为 VFR 录像，所有时间类指标可能存在系统性偏差」。
    """
    if n_frames <= 0:
        raise ValueError("n_frames 必须为正")
    if nominal_fps <= 0:
        raise ValueError(f"标称帧率必须为正，收到 {nominal_fps}")
    t = np.arange(n_frames, dtype=float) / float(nominal_fps)
    duration_s = float(t[-1] - t[0]) if n_frames > 1 else 0.0
    return TimingReport(
        n_frames=int(n_frames),
        timing_source="nominal_fps",
        duration_s=duration_s,
        nominal_duration_s=duration_s,
        duration_deviation=0.0,
        timing_suspect=False,
        timeline_discontinuity=False,
        notes=[
            "timing_source='nominal_fps'（逐帧 PTS 不可得，标称 FPS 兜底）："
            "若为 VFR 录像，所有时间类指标可能存在系统性偏差"
        ],
        t_s=t,
    )

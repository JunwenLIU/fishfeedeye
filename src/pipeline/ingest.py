"""视频接入与非对称采样（T02）。

职责（docs/06 §6 T02 内联约定 + docs/04 §3 采样约定）：
    1. 解码视频（复用 utils/video_io.py），逐帧 PTS/POS_MSEC 构建时间轴
       （复用 pipeline/timing.py），容器标称 FPS 仅兜底；
    2. 非对称采样（硬设计，非优化）：
         基线 [-60s, 0)   间隔 2s  → 30 帧
         早期 [0, 60s]     间隔 1s  → 61 帧（A14 关联只在窗口内成立）
         尾段 (60s, 300s]  间隔 10s → 24 帧
       合计 ≈115 帧（契约口径"≈114 帧"，t0 与视频端点截断后通常为 114±1）；
    3. 基线段检测：投喂前 ≥30s 基线可用性判定（不足 → Q_baseline=False，
       由下游强制关闭相对基线指标，本模块只报告不关闭）；
    4. 断裂防护：时间轴断裂（拼接/续录）之后的帧不参与采样
       （timing.frames_valid_for_time_metrics 掩码）。

纪律：
    - 采样帧的 t_s / dt_s 一律取自真实解码时间戳（PTS），绝不按标称 FPS
      或目标网格推算——"原生不规则时间戳是唯一可信源"（docs/04 §3 ①）；
    - 基线目标只允许匹配 t < t0 的帧，早期/尾段目标只允许匹配 t ≥ t0 的帧
      （避免把投喂后帧误标为基线）；
    - 随机 seek 不可靠（utils/video_io 注释），采样帧一律顺序解码获取。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.core.config import RunConfig, SamplingParams
from src.core.frame_context import FrameObservation
from src.pipeline.timing import TimingReport, build_timeline, build_timeline_from_nominal
from src.utils.logging import get_logger
from src.utils.video_io import VideoInfo, VideoReader, collect_frame_timestamps, probe_video

__all__ = [
    "IngestResult",
    "plan_sampling_targets",
    "select_sampling_indices",
    "ingest",
]

_log = get_logger("pipeline.ingest")

# 基线段可用性契约值：投喂前 ≥30s（docs/04 §1：最短 30s）
BASELINE_MIN_DURATION_S: float = 30.0


# ----------------------------------------------------------------------
# 采样计划（纯函数，可独立单测）
# ----------------------------------------------------------------------
def plan_sampling_targets(
    sampling: SamplingParams,
    observation_window_s: float,
) -> list[float]:
    """生成相对 t0 的目标采样时间戳（秒，负值 = 基线期）。

    基线 [-baseline_window, 0) 间隔 baseline_interval；
    早期 [0, early_window] 间隔 early_interval（含端点）；
    尾段 (early_window, observation_window] 间隔 tail_interval。

    Returns:
        升序目标时间戳列表。默认参数下共 30+61+24 = 115 个。
    """
    if observation_window_s <= sampling.early_window_s:
        raise ValueError(
            f"观察窗 ({observation_window_s}s) 必须大于早期窗口 ({sampling.early_window_s}s)"
        )
    targets: list[float] = []
    # 基线段：[-bw, 0)，不含 0（0 = t0，属早期段）
    n_base = int(round(sampling.baseline_window_s / sampling.baseline_interval_s))
    for i in range(n_base, 0, -1):
        targets.append(-i * sampling.baseline_interval_s)
    # 早期段：[0, early_window]，含两端
    n_early = int(round(sampling.early_window_s / sampling.early_interval_s))
    for i in range(n_early + 1):
        targets.append(i * sampling.early_interval_s)
    # 尾段：(early_window, observation_window]
    t = sampling.early_window_s + sampling.tail_interval_s
    while t <= observation_window_s + 1e-9:
        targets.append(t)
        t += sampling.tail_interval_s
    return targets


def select_sampling_indices(
    t_s: np.ndarray,
    targets: Sequence[float],
    valid: Sequence[bool] | None = None,
    half_interval_s: float | None = None,
) -> list[int]:
    """为目标时间戳挑选真实帧号（贪心最近邻 + 单调去重）。

    Args:
        t_s: 逐帧真实时间戳（秒，相对 t0；基线期为负）。
        targets: plan_sampling_targets 的输出。
        valid: 逐帧有效性掩码（时间轴断裂后为 False）；None = 全部有效。
        half_interval_s: 匹配容差（目标与最近候选帧的最大允许距离）。
            None 时按目标所属段自动取（基线/早期/尾段各自间隔的一半）。

    规则：
        - 基线目标（<0）只匹配 t < 0 的帧；非负目标只匹配 t ≥ 0 的帧
          （防止把 t0 附近帧误标为基线，或反之）；
        - 每帧至多被选一次（去重），选择顺序按目标升序 → 帧号天然升序；
        - 目标超出视频时间范围（±容差）→ 跳过（不造观测点）。
    """
    t = np.asarray(t_s, dtype=float)
    n = int(t.shape[0])
    if n == 0:
        return []
    v = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    chosen: list[int] = []
    used: set[int] = set()
    for target in sorted(targets):
        # 判定目标所属段 → 容差与符号约束
        if half_interval_s is not None:
            tol = float(half_interval_s)
        elif target < 0:
            tol = 1.0  # 基线 2s 的一半
        elif target <= 60.0 + 1e-9:
            tol = 0.5  # 早期 1s 的一半
        else:
            tol = 5.0  # 尾段 10s 的一半
        need_negative = target < 0
        cand = np.where(v)[0]
        # 符号约束：基线目标只要 t<0 的帧；非负目标只要 t>=0 的帧
        mask_sign = (t < 0.0) if need_negative else (t >= 0.0)
        idx_pool = cand[mask_sign[cand]]
        if idx_pool.size == 0:
            continue
        dist = np.abs(t[idx_pool] - target)
        j = int(np.argmin(dist))
        if dist[j] > tol + 1e-9:
            continue  # 目标超出可用范围，跳过（绝不外推）
        fi = int(idx_pool[j])
        if fi in used:
            continue  # 已被更近的目标占用（去重）
        used.add(fi)
        chosen.append(fi)
    return sorted(chosen)


# ----------------------------------------------------------------------
# 接入结果
# ----------------------------------------------------------------------
@dataclass
class IngestResult:
    """ingest 阶段输出（FrameObservation 列表 + 时间轴体检 + 基线可用性）。"""

    info: VideoInfo
    timing: TimingReport
    observations: list[FrameObservation]
    t0_s: float                          # 视频内绝对时间戳（秒）
    t0_source: str                       # 'manual' | 'video_start'
    baseline_duration_s: float | None    # None = 无基线帧
    baseline_available: bool             # 投喂前 ≥30s
    notes: list[str] = field(default_factory=list)

    def sampling_table(self) -> list[dict]:
        """采样帧清单（frame_idx / t_s / dt_s / in_baseline），CLI 与报告消费。"""
        rows: list[dict] = []
        for obs in self.observations:
            rows.append(
                {
                    "frame_idx": obs.frame_idx,
                    "t_s": obs.t_s,
                    "dt_s": obs.dt_s,
                    "in_baseline": obs.t_s < 0.0,
                }
            )
        return rows


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def ingest(
    video_path: str | Path,
    config: RunConfig,
    t0_s: float | None = None,
    keep_images: bool = True,
    progress: Callable[[str, float], None] | None = None,
) -> IngestResult:
    """接入视频：时间轴构建 + 非对称采样 + 基线段检测。

    Args:
        video_path: 视频文件路径。
        config: RunConfig（消费 sampling 与 thresholds.observation_window_s）。
        t0_s: 投喂起点在视频内的绝对时间戳（秒）。None = 未提供 →
            按视频首帧处理（无基线段，显式 note，绝不静默）。
        keep_images: 是否在 FrameObservation 中保留解码帧（检测阶段需要；
            缓存回放/纯清单场景可关以省内存）。
        progress: 进度回调 (阶段名, 0-1)。

    Returns:
        IngestResult。observations 按 t_s 升序，dt_s 为相邻采样的真实间隔。
    """
    if progress is None:
        progress = lambda _stage, _frac: None  # noqa: E731

    notes: list[str] = []
    info = probe_video(video_path)
    _log.info("视频元信息: %s×%s @%.3ffps 标称 %d 帧 (%.1fs)",
              info.width, info.height, info.fps, info.n_frames, info.duration_s)

    # ---- 时间轴（逐帧 PTS 优先）----
    progress("collect_timestamps", 0.0)
    pos_msec, decoded = collect_frame_timestamps(video_path)
    if decoded == 0:
        raise IOError(f"视频解码 0 帧（文件损坏或编码器不支持）: {video_path}")
    timing = build_timeline(pos_msec, info.fps)
    if decoded != info.n_frames:
        notes.append(
            f"容器标称帧数 {info.n_frames} 与实际解码 {decoded} 不一致"
            "（VFR/拼接信号，详见 timing 报告）"
        )
    if timing.timing_suspect:
        notes.append("timing_suspect=True：时间类指标可能存在系统性偏差")
    if timing.timeline_discontinuity:
        notes.append(
            "timeline_discontinuity=True：断裂点之后的帧不参与采样，"
            "断裂后时间指标一律不输出"
        )
    if timing.t_s is None:  # pragma: no cover - build_timeline 必然填充
        timing = build_timeline_from_nominal(decoded, info.fps)
    t_abs = np.asarray(timing.t_s, dtype=float)

    # ---- t0 ----
    if t0_s is None:
        t0_s = float(t_abs[0])
        t0_source = "video_start"
        notes.append(
            "t0 未提供：按视频首帧处理，基线段不可用（相对基线指标将被强制关闭）"
        )
    else:
        t0_source = "manual"
        if not (t_abs[0] - 1e-6 <= t0_s <= t_abs[-1] + 1e-6):
            raise ValueError(
                f"t0={t0_s}s 超出视频时间范围 [{t_abs[0]:.2f}, {t_abs[-1]:.2f}]s"
            )

    # ---- 非对称采样 ----
    progress("plan_sampling", 0.5)
    targets = plan_sampling_targets(config.sampling, config.thresholds.observation_window_s)
    t_rel = t_abs - t0_s
    valid = timing.frames_valid_for_time_metrics()
    selected = select_sampling_indices(t_rel, targets, valid)
    if not selected:
        raise ValueError(
            f"采样结果为空：视频时长 {timing.duration_s:.1f}s，t0={t0_s:.1f}s，"
            "没有任何目标时间戳落在有效范围内"
        )

    # ---- 解码采样帧（顺序解码，不用随机 seek）----
    progress("retrieve_frames", 0.6)
    frames = _retrieve_frames(video_path, selected) if keep_images else {}

    observations: list[FrameObservation] = []
    prev_t: float | None = None
    for fi in selected:
        t = float(t_rel[fi])
        dt = None if prev_t is None else t - prev_t
        observations.append(
            FrameObservation(
                frame_idx=fi,
                t_s=t,
                dt_s=dt,
                image=frames.get(fi),
                pellets=None,
                extra={},
            )
        )
        prev_t = t
    progress("ingest_done", 1.0)

    # ---- 基线段检测 ----
    baseline_ts = [obs.t_s for obs in observations if obs.t_s < 0.0]
    if baseline_ts:
        baseline_duration = float(max(baseline_ts) - min(baseline_ts))
    else:
        baseline_duration = None
    baseline_available = baseline_duration is not None and baseline_duration >= BASELINE_MIN_DURATION_S
    if baseline_ts and not baseline_available:
        notes.append(
            f"基线段仅 {baseline_duration:.1f}s < 30s：Q_baseline=False，"
            "所有相对基线指标（RP/归一化活跃度）将被强制关闭"
        )

    _log.info("采样完成: %d 帧（基线 %d 帧 %.1fs），t0_source=%s",
              len(observations), len(baseline_ts),
              baseline_duration or 0.0, t0_source)
    return IngestResult(
        info=info,
        timing=timing,
        observations=observations,
        t0_s=float(t0_s),
        t0_source=t0_source,
        baseline_duration_s=baseline_duration,
        baseline_available=baseline_available,
        notes=notes,
    )


def _retrieve_frames(
    video_path: str | Path, indices: Sequence[int]
) -> dict[int, np.ndarray]:
    """顺序解码并取出指定帧号集合（随机 seek 不可靠，见 utils/video_io）。"""
    want = {int(i) for i in indices}
    out: dict[int, np.ndarray] = {}
    with VideoReader(video_path) as vr:
        for idx, _t, frame in vr:
            if idx in want:
                out[idx] = frame
                if len(out) == len(want):
                    break
    missing = want - set(out)
    if missing:
        raise IOError(f"顺序解码未取到帧: {sorted(missing)}")
    return out


def make_run_id(video_path: str | Path, now: datetime | None = None) -> str:
    """run_id 规则：视频文件名去扩展名 + 分析时间戳（docs/06 §7.2）。"""
    stem = Path(video_path).stem
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}"

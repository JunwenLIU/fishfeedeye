"""颗粒时空最近邻关联 + 消失三分类（T03）。

职责（docs/06 §6 T03 + docs/04 §3.1 A14）：
    1. 帧间关联：带物理约束的时空最近邻（不用匈牙利匹配）：
         关联可行条件 = v_sink_max × Δt < association_radius
         （v_sink_max 由无鱼纯饲料视频标定，禁止硬编码；条件不满足 →
         该次关联标记 partial_window，仍取最近邻但可信度下降）；
    2. 消失三分类（轨迹级）：
         drifted  —— 轨迹穿越投喂区边界向外（最后位置在区外，或贴近
                     边界带内）→ 非摄食损失；
         unknown  —— 最后位置落在排除区（反光/遮挡），或关联物理约束
                     未标定 / partial_window 不可靠 → **严禁猜测归类**；
         eaten    —— 颗粒在投喂区内部消失且关联约束成立；
    3. 短暂消失容忍：连续 ≥ min_missing_frames（默认 3）个采样帧未匹配
       才判消失（波浪短暂淹没容忍）；不足 N 帧的断档按遮挡处理，
       轨迹保持（再次出现续接）。

纪律：
    - v_sink_max / association_radius 缺失（未标定）→ 关联照做（纯最近邻）
      但 association_unconstrained=True，且内部消失一律 unknown
      （没有物理约束支撑"被吃掉"的推断，不猜）；
    - 消失分类只基于几何与约束证据，从不基于"概率猜测"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.core.frame_context import FrameObservation, PelletDetections
from src.core.roi import ROI, point_in_polygon

__all__ = ["Track", "VanishEvent", "LinkResult", "PelletLinker"]


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class Track:
    """单条颗粒轨迹（采样帧级）。"""

    track_id: int
    t: list[float] = field(default_factory=list)           # 相对 t0 秒
    xy: list[tuple[float, float]] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)
    vanish_class: str | None = None                        # 存活到末帧 = None
    vanish_reason: str | None = None
    partial_window: bool = False                           # 关联约束曾被违反
    misses: int = 0                                        # 末尾连续未匹配帧数


@dataclass
class VanishEvent:
    """一次消失事件（轨迹级三分类）。"""

    track_id: int
    t_last_s: float
    last_xy: tuple[float, float]
    vanish_class: str                  # 'eaten' | 'drifted' | 'unknown'
    reason: str
    missing_frames: int
    partial_window: bool


@dataclass
class LinkResult:
    """link 阶段输出。"""

    tracks: list[Track]
    vanish_events: list[VanishEvent]
    association_unconstrained: bool = False   # v_sink/radius 未标定
    n_eaten: int = 0
    n_drifted: int = 0
    n_unknown: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_vanished(self) -> int:
        return len(self.vanish_events)

    def to_dict(self) -> dict:
        return {
            "n_tracks": len(self.tracks),
            "association_unconstrained": self.association_unconstrained,
            "n_eaten": self.n_eaten,
            "n_drifted": self.n_drifted,
            "n_unknown": self.n_unknown,
            "events": [
                {
                    "track_id": e.track_id,
                    "t_last_s": e.t_last_s,
                    "last_xy": list(e.last_xy),
                    "vanish_class": e.vanish_class,
                    "reason": e.reason,
                    "missing_frames": e.missing_frames,
                    "partial_window": e.partial_window,
                }
                for e in self.vanish_events
            ],
            "notes": list(self.notes),
        }


# ----------------------------------------------------------------------
# 关联器
# ----------------------------------------------------------------------
class PelletLinker:
    """时空最近邻关联 + 消失三分类（classDiagram PelletLinker）。"""

    def __init__(
        self,
        v_sink_max_px_s: float | None = None,
        association_radius_px: float | None = None,
        min_missing_frames: int = 3,
        roi: ROI | None = None,
        boundary_margin_px: float = 10.0,
    ) -> None:
        """
        Args:
            v_sink_max_px_s: 颗粒最大位移速度（px/s，标定值，禁止硬编码）。
            association_radius_px: 关联半径（px，标定值）。
            min_missing_frames: 连续未匹配采样帧数达到该值才判消失（默认 3）。
            roi: 区域定义（pellet_zone 边界 → drifted；排除区 → unknown）。
            boundary_margin_px: 边界带宽度（贴边消失 → drifted）。
        """
        if min_missing_frames < 1:
            raise ValueError("min_missing_frames 必须 ≥ 1")
        self.v_sink_max = v_sink_max_px_s
        self.radius = association_radius_px
        self.min_missing_frames = int(min_missing_frames)
        self.roi = roi
        self.boundary_margin_px = float(boundary_margin_px)

    # ------------------------------------------------------------------
    def link(self, frames: Sequence[FrameObservation]) -> LinkResult:
        """逐帧最近邻关联 → 轨迹 + 消失三分类。

        frames 须按 t_s 升序；pellets 为 None 的帧（无检测数据）跳过但
        不计入未匹配（不惩罚检测缺失帧）。
        """
        notes: list[str] = []
        constrained = (
            self.v_sink_max is not None
            and self.radius is not None
            and self.v_sink_max > 0
            and self.radius > 0
        )
        if not constrained:
            notes.append(
                "关联物理约束未标定（v_sink_max / association_radius 缺失）："
                "关联退化为纯最近邻，投喂区内部消失一律 unknown（不猜）"
            )
        if self.roi is None:
            notes.append("ROI 未定义：边界穿越判定不可用，内部消失一律 unknown")

        active: list[Track] = []
        finished: list[Track] = []
        events: list[VanishEvent] = []
        next_id = 1

        for obs in frames:
            if obs.pellets is None:
                continue  # 无检测数据的帧不参与关联
            dets: PelletDetections = obs.pellets
            cents = dets.centroids() if dets.n_det() > 0 else np.zeros((0, 2))
            confs = dets.conf if dets.n_det() > 0 else np.zeros((0,))

            # 距上一匹配帧的 Δt（用上一有效观测的 t，而非上一采样帧）
            prev_t = max((tr.t[-1] for tr in active), default=None)
            dt = (obs.t_s - prev_t) if prev_t is not None else None

            # ---- 匹配（贪心最近邻，按检测顺序）----
            unmatched_tracks = list(range(len(active)))
            used_det = np.zeros(len(cents), dtype=bool)
            assign: list[tuple[int, int]] = []  # (track_pos, det_i)
            for i in range(len(cents)):
                best_j, best_d = -1, np.inf
                for j in unmatched_tracks:
                    ax, ay = active[j].xy[-1]
                    d = float(np.hypot(cents[i][0] - ax, cents[i][1] - ay))
                    if d < best_d:
                        best_d, best_j = d, j
                if best_j < 0:
                    continue
                # 关联半径：已标定 → 半径约束；未标定 → 不限（但记不可信）
                if constrained and best_d > self.radius:
                    best_d_ok = False
                else:
                    best_d_ok = True
                if not best_d_ok:
                    continue  # 超出物理可达范围，不关联（新建轨迹）
                assign.append((best_j, i))
                unmatched_tracks.remove(best_j)
                used_det[i] = True

            # ---- 更新匹配轨迹 / partial_window ----
            for j, i in assign:
                tr = active[j]
                if constrained and dt is not None:
                    if self.v_sink_max * dt >= self.radius:
                        tr.partial_window = True  # 关联约束违反，可信度降
                tr.t.append(float(obs.t_s))
                tr.xy.append((float(cents[i][0]), float(cents[i][1])))
                tr.conf.append(float(confs[i]))
                tr.misses = 0

            # ---- 未匹配检测 → 新轨迹 ----
            for i in range(len(cents)):
                if used_det[i]:
                    continue
                tr = Track(
                    track_id=next_id,
                    t=[float(obs.t_s)],
                    xy=[(float(cents[i][0]), float(cents[i][1]))],
                    conf=[float(confs[i])],
                )
                next_id += 1
                active.append(tr)

            # ---- 未匹配轨迹：misses + 1；达到阈值 → 消失分类 ----
            still_active: list[Track] = []
            for j, tr in enumerate(active):
                if j in unmatched_tracks:
                    tr.misses += 1
                    if tr.misses >= self.min_missing_frames:
                        cls, reason = self.classify_vanish(tr)
                        tr.vanish_class = cls
                        tr.vanish_reason = reason
                        events.append(
                            VanishEvent(
                                track_id=tr.track_id,
                                t_last_s=tr.t[-1],
                                last_xy=tr.xy[-1],
                                vanish_class=cls,
                                reason=reason,
                                missing_frames=tr.misses,
                                partial_window=tr.partial_window,
                            )
                        )
                        finished.append(tr)
                        continue  # 已消失的轨迹不再保留
                still_active.append(tr)
            active = still_active

        tracks = finished + active
        n_eaten = sum(1 for e in events if e.vanish_class == "eaten")
        n_drifted = sum(1 for e in events if e.vanish_class == "drifted")
        n_unknown = sum(1 for e in events if e.vanish_class == "unknown")
        return LinkResult(
            tracks=tracks,
            vanish_events=events,
            association_unconstrained=not constrained,
            n_eaten=n_eaten,
            n_drifted=n_drifted,
            n_unknown=n_unknown,
            notes=notes,
        )

    # ------------------------------------------------------------------
    def classify_vanish(self, track: Track) -> tuple[str, str]:
        """消失三分类（几何 + 约束证据；classDiagram classify_vanish）。

        判定顺序（从严到宽，证据不足 → unknown，绝不猜）：
          1. 最后位置在排除区 → unknown（反光/遮挡，原因不可判）；
          2. ROI 未定义 → unknown；
          3. 最后位置在 pellet_zone 外（或 arena 外）→ drifted；
          4. 贴近 pellet_zone 边界（< boundary_margin_px）→ drifted；
          5. 关联约束未标定 → unknown；
          6. 轨迹含 partial_window（关联不可信）→ unknown；
          7. 其余（区内部消失且约束成立）→ eaten。
        """
        last_xy = track.xy[-1]
        constrained = (
            self.v_sink_max is not None and self.radius is not None
            and self.v_sink_max > 0 and self.radius > 0
        )
        if self.roi is not None and self.roi.exclude_zones:
            if any(point_in_polygon(last_xy, z) for z in self.roi.exclude_zones):
                return (
                    "unknown",
                    "最后位置位于排除区（反光/遮挡），消失原因不可判，严禁猜测归类",
                )
        if self.roi is None:
            return "unknown", "ROI 未定义，无法判定消失位置类别"
        zone = self.roi.pellet_zone if self.roi.pellet_zone is not None else self.roi.arena
        if not point_in_polygon(last_xy, zone):
            return (
                "drifted",
                "轨迹穿越投喂区边界向外（最后位置在区外）→ 非摄食损失",
            )
        if _dist_to_polygon_edge(last_xy, zone) < self.boundary_margin_px:
            return (
                "drifted",
                f"最后位置贴近投喂区边界（< {self.boundary_margin_px:g}px）→ 判漂出",
            )
        if not constrained:
            return (
                "unknown",
                "关联物理约束未标定（v_sink_max/association_radius 缺失）："
                "无证据支撑『被吃掉』推断，不猜测归类",
            )
        if track.partial_window:
            return (
                "unknown",
                "关联存在 partial_window（v_sink_max×Δt ≥ association_radius），"
                "轨迹可信度不足，消失原因不可判",
            )
        return (
            "eaten",
            "颗粒在投喂区内部消失且关联约束成立（被吃掉）",
        )


def _dist_to_polygon_edge(
    point: tuple[float, float], polygon: np.ndarray
) -> float:
    """点到多边形边界（各边线段）的最小距离（像素）。"""
    poly = np.asarray(polygon, dtype=float)
    px, py = float(point[0]), float(point[1])
    best = np.inf
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 <= 1e-12:
            d = float(np.hypot(px - ax, py - ay))
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
            d = float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
        best = min(best, d)
    return best

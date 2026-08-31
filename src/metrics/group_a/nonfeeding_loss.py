"""A14 · 非摄食损失率（T04 · 质量警示项，不参与主指标计算）。

职责（docs/04 §3.1 A14）：
    - 消费颗粒轨迹消失事件三分类（eaten / drifted / unknown，
      由 pipeline/pellet_linker 产出）：drifted（区外/贴边消失）计为
      非摄食损失 n_loss；unknown 不猜测归类；
    - ratio = n_loss / N₀（Q_pelletloss 的来源）；
    - 仅密集窗口（a14_window_s，默认 60s）内的消失事件可信：观察窗更长
      → partial_window=True；
    - bottom_band 未定义 → 分类退化为两分，标记 coarse=True。

失效条件（契约原文）：
    - track_id 全程缺失 → 不可用，且必须显式声明"未校正沉降损失"
      （严禁默认为 0）；
    - 失效时任何清空时间类指标都不得声明"已校正沉降损失"。

任务编号：T04。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.core.config import Thresholds
from src.core.frame_context import FrameObservation
from src.core.metric_value import MetricValue
from src.pipeline.pellet_linker import LinkResult

__all__ = ["VanishTally", "tally_vanish", "nonfeeding_loss_metric"]

_NOT_TRACKED_REASON = (
    "颗粒轨迹未关联（track_id=None）：未校正沉降损失（严禁默认为 0）"
)


@dataclass
class VanishTally:
    """A14 消失事件统计。

    Attributes:
        n_loss: 非摄食损失颗数（drifted）。
        n_eaten: 摄食消失颗数。
        n_unknown: unknown 颗数（不猜测归类）。
        available: 统计是否可用。
        reason: 不可用原因。
        coarse: bottom_band 缺失 → 两分退化。
        partial_window: 观察窗超出密集关联窗口。
    """

    n_loss: int = 0
    n_eaten: int = 0
    n_unknown: int = 0
    available: bool = False
    reason: str | None = None
    coarse: bool = False
    partial_window: bool = False

    @property
    def unknown_frac(self) -> float | None:
        """unknown 占比（越高越不可信）；无事件 → None。"""
        total = self.n_loss + self.n_eaten + self.n_unknown
        if total == 0:
            return None
        return self.n_unknown / float(total)


def tally_vanish(
    link_result: LinkResult | None,
    observations: Sequence[FrameObservation],
    t_max_s: float | None,
    thresholds: Thresholds | None = None,
) -> VanishTally:
    """从关联结果统计消失三分类。

    Args:
        link_result: PelletLinker 输出（vanish_events 含分类）。
        observations: 帧观测（link_result 缺失时从 vanish_class 回退统计）。
        t_max_s: 序列最晚时间戳（partial_window 判定）。
        thresholds: a14_window_s 密集窗口阈值。
    """
    th = thresholds if thresholds is not None else Thresholds()
    events: list[tuple[float, str]] = []
    if link_result is not None and link_result.vanish_events:
        for e in link_result.vanish_events:
            events.append((e.t_last_s, e.vanish_class))
    else:
        # 回退：帧级 vanish_class 标记（orchestrator 回写）
        for obs in observations:
            pel = obs.pellets
            if pel is None or pel.vanish_class is None:
                continue
            for cls in pel.vanish_class:
                if cls is not None:
                    events.append((obs.t_s, cls))

    has_track_ids = any(
        obs.pellets is not None and obs.pellets.track_id is not None
        for obs in observations
    )
    if not events and not has_track_ids:
        return VanishTally(available=False, reason=_NOT_TRACKED_REASON)
    if not events:
        return VanishTally(available=False, reason=_NOT_TRACKED_REASON)

    tally = VanishTally(available=True)
    for t_ev, cls in events:
        # 密集窗口外的事件仍计数，但整体标记 partial_window
        if cls == "drifted":
            tally.n_loss += 1
        elif cls == "eaten":
            tally.n_eaten += 1
        else:  # unknown（不猜测归类）
            tally.n_unknown += 1
    if t_max_s is not None and t_max_s > th.a14_window_s:
        tally.partial_window = True
    if link_result is not None:
        tally.coarse = False  # 分类来自 linker（含 bottom_band 判定）
    return tally


def nonfeeding_loss_metric(
    tally: VanishTally,
    n0: float,
    bottom_band_defined: bool = False,
    thresholds: Thresholds | None = None,
) -> tuple[MetricValue, float | None, float | None]:
    """A14 输出 + Q_pelletloss（已知非摄食损失率）+ Q_fatelost（命运未确认率）。

    Returns:
        (MetricValue A14, Q_pelletloss, Q_fatelost)：
        - Q_pelletloss = n_loss/N₀（已知漂出损失，用于展示与元数据反查）；
        - Q_fatelost  = (n_loss + n_unknown)/N₀（漂出 + 反光区消失等命运未确认，
          驱动 T90/T100/RR 降级与矛盾标记，对应 PRD FR-14 的
          “n_drifted + n_unknown > 15%” 触发条件）；
        A14 不可用时二者均为 None（不猜测，不得 0 冒充）。
    """
    th = thresholds if thresholds is not None else Thresholds()
    quality: dict[str, Any] = {
        "n_loss": tally.n_loss,
        "n_eaten": tally.n_eaten,
        "n_unknown": tally.n_unknown,
        "unknown_frac": tally.unknown_frac,
    }
    flags: list[str] = []
    if not bottom_band_defined:
        flags.append("coarse")
    if tally.partial_window:
        flags.append("partial_window")

    if not tally.available:
        return (
            MetricValue(
                metric_id="A14_NonFeedingLoss",
                value=None,
                unit="%",
                status="unavailable",
                reason=tally.reason or _NOT_TRACKED_REASON,
                flags=tuple(flags),
                quality=quality,
                unit_scale="none",
            ),
            None,
            None,
        )

    if n0 is None or n0 <= 0:
        return (
            MetricValue(
                metric_id="A14_NonFeedingLoss",
                value=None,
                unit="%",
                status="unavailable",
                reason="N₀ 不可用：损失率分母缺失，不猜测（严禁默认为 0）",
                flags=tuple(flags),
                quality=quality,
                unit_scale="none",
            ),
            None,
            None,
        )

    ratio = tally.n_loss / float(n0)  # 已知非摄食损失率（Q_pelletloss）
    # PRD FR-14：降级/矛盾触发以「命运未确认率」为准 = (漂出 + 反光区消失)/N₀
    ratio_fatelost = (tally.n_loss + tally.n_unknown) / float(n0)
    q_pelletloss: float | None = ratio
    q_fatelost: float | None = ratio_fatelost
    if (
        ratio_fatelost is not None
        and ratio_fatelost > th.pelletloss_degrade
    ):
        flags.append("contains_non_feeding_loss")
    return (
        MetricValue(
            metric_id="A14_NonFeedingLoss",
            value=None if ratio is None else ratio * 100.0,
            unit="%",
            status="ok",
            reason=None,
            flags=tuple(flags),
            quality=quality,
            unit_scale="none",
        ),
        q_pelletloss,
        q_fatelost,
    )

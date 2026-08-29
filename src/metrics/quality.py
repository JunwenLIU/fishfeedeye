"""13 个 Q_* 全局质量信号（T04，docs/04 §2）。

职责：
    每次运行必算、必输出——即使值为 None 也要有行（None = 未测得，
    绝不冒充 0 或 False）。各指标的失效条件引用这些信号；
    capability.py 消费本模块输出做降级/关闭决策。

可计算性说明（L0 冷启动阶段的诚实边界）：
    - Q_det / Q_fdet / Q_vis：从 FrameObservation.pellets / extra["fish"]
      直接可算；
    - Q_track：需要 Tracks（轻量关联轨迹）输入，缺输入 → None；
    - Q_ids：ID 切换率需要跟踪器逐帧 ID 记录，L0 不可测 → 恒 None；
    - Q_interf / Q_fg / Q_motion：消费 pipeline 在 extra 里逐帧写入的
      interf_frac / fg_frac / motion_residual（缺失 → None）；
      ⚠️ Q_fg 的计算区域必须排除 pellet_zone（pipeline 侧责任，见
      docs/04 §4.3 🔴——否则被颗粒数污染，用被测变量门控另一个指标）；
    - Q_calib / Q_baseline：布尔（标定/基线可用性）；
    - Q_pelletloss / Q_n0gap：依赖 A14 / A1 的产出（由 aggregator 注入）；
    - Q_censored：右删失标记，由 aggregator 在清空时间指标算完后回填。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.config import Thresholds
from src.core.frame_context import (
    BaselineStats,
    FrameObservation,
    RunMeta,
    Tracks,
)
from src.pipeline.pellet_linker import LinkResult

__all__ = ["QualitySignals", "Q_KEYS", "Q_SIGNAL_KEYS", "to_quality_rows"]

# 13 个质量信号键（docs/04 §2 表格顺序）
Q_KEYS: tuple[str, ...] = (
    "Q_det",          # 颗粒检测置信度中位数
    "Q_fdet",         # 鱼体检测置信度中位数
    "Q_vis",          # 平均可见鱼数（密度分档）
    "Q_track",        # 可追踪个体比例
    "Q_ids",          # ID 切换率（L0 不可测 → None）
    "Q_interf",       # 反光/水花污染帧占比
    "Q_fg",           # 前景像素占比（须排除 pellet_zone，pipeline 侧）
    "Q_calib",        # 标定可用性（bool）
    "Q_baseline",     # 基线可用性（bool）
    "Q_pelletloss",   # 非摄食损失率（A14 产出）
    "Q_n0gap",        # N₀ 交叉校验偏差（A1 产出）
    "Q_censored",     # 右删失标记（aggregator 回填）
    "Q_motion",       # 相机稳定性（配准残差）
)

# Q_track 有效轨迹的最小存活帧数（工程约定；docs/04 §2 未给阈值）
_TRACK_MIN_ALIVE_FRAMES = 5


class QualitySignals:
    """质量信号计算器（纯函数式：compute 一次返回全部 13 个键）。"""

    def compute(
        self,
        frames: list[FrameObservation],
        *,
        tracks: Tracks | None = None,
        link_result: LinkResult | None = None,
        baseline: BaselineStats | None = None,
        meta: RunMeta | None = None,
        n0_det: float | None = None,
        n0_meta: float | None = None,
        px_per_mm: float | None = None,
        thresholds: Thresholds | None = None,
    ) -> dict[str, Any]:
        """计算 13 个 Q_* 信号（全部键必存在，未测得为 None）。

        Args:
            frames: 采样帧观测（t_s 相对 t0，基线期为负）。
            tracks: 颗粒/鱼体轻量关联轨迹（Q_track 消费；缺省 None）。
            link_result: A14 关联结果（Q_pelletloss 消费）。
            baseline: 基线统计（None = 基线缺失）。
            meta: 用户元数据。
            n0_det / n0_meta: A1 双轨初始颗粒数（Q_n0gap 消费）。
            px_per_mm: 标定尺度（Q_calib）。
            thresholds: 阈值表（缺省用内置默认）。

        Returns:
            dict，键 = Q_KEYS 全部 13 项。
        """
        th = thresholds if thresholds is not None else Thresholds()
        signals: dict[str, Any] = {}

        # ---- Q_det：全程颗粒置信度中位数 ----
        all_conf: list[np.ndarray] = []
        for obs in frames:
            if obs.pellets is not None and obs.pellets.n_det() > 0:
                all_conf.append(np.asarray(obs.pellets.conf, dtype=float))
        signals["Q_det"] = (
            float(np.median(np.concatenate(all_conf))) if all_conf else None
        )

        # ---- Q_fdet / Q_vis：鱼体（extra["fish"] = FishDetections）----
        fish_conf: list[np.ndarray] = []
        fish_counts: list[int] = []
        for obs in frames:
            fish = obs.extra.get("fish")
            if fish is not None and fish.n_det() > 0:
                fish_conf.append(np.asarray(fish.conf, dtype=float))
                fish_counts.append(fish.n_det())
            elif fish is not None:
                fish_counts.append(0)
        signals["Q_fdet"] = (
            float(np.median(np.concatenate(fish_conf))) if fish_conf else None
        )
        signals["Q_vis"] = (
            float(np.mean(fish_counts)) if fish_counts else None
        )

        # ---- Q_track：有效轨迹数 / 平均可见数 ----
        if tracks is not None and signals["Q_vis"] and signals["Q_vis"] > 0:
            n_valid = sum(
                1 for alive in tracks.n_frames_alive
                if alive >= _TRACK_MIN_ALIVE_FRAMES
            )
            signals["Q_track"] = float(n_valid) / float(signals["Q_vis"])
        else:
            signals["Q_track"] = None

        # ---- Q_ids：ID 切换率（L0 无跟踪器逐帧 ID 记录，不可测）----
        signals["Q_ids"] = None

        # ---- Q_interf / Q_fg / Q_motion：pipeline 逐帧 extra 注入 ----
        signals["Q_interf"] = _mean_of_extra(frames, "interf_frac")
        signals["Q_fg"] = _mean_of_extra(frames, "fg_frac")
        signals["Q_motion"] = _mean_of_extra(frames, "motion_residual")

        # ---- Q_calib / Q_baseline ----
        signals["Q_calib"] = px_per_mm is not None
        signals["Q_baseline"] = baseline is not None and baseline.ok

        # ---- Q_pelletloss：非摄食损失率（A14；n0 缺失 → None）----
        if (
            link_result is not None
            and n0_det is not None
            and n0_det > 0
        ):
            signals["Q_pelletloss"] = float(link_result.n_drifted) / float(
                n0_det
            )
        else:
            signals["Q_pelletloss"] = None

        # ---- Q_n0gap：N₀ 双轨交叉校验 ----
        if n0_det is not None and n0_meta is not None and n0_meta > 0:
            signals["Q_n0gap"] = abs(n0_det - n0_meta) / float(n0_meta)
        else:
            signals["Q_n0gap"] = None

        # ---- Q_censored：占位（aggregator 在清空时间指标后回填）----
        signals["Q_censored"] = False

        # 元数据反查所需的一并暴露（docs/04 §4.5 管线顺序约束）
        signals["N0_det"] = n0_det
        signals["N0_meta"] = n0_meta
        signals["baseline_duration_s"] = (
            baseline.duration_s if baseline is not None else None
        )
        signals["pellet_type_meta"] = meta.pellet_type if meta else None

        assert set(Q_KEYS) <= set(signals), "13 个 Q_* 键必须全部输出"
        return signals


def _mean_of_extra(
    frames: list[FrameObservation], key: str
) -> float | None:
    """取逐帧 extra[key] 的均值；一帧都没有 → None（未测得，不冒充 0）。"""
    vals = [
        float(obs.extra[key])
        for obs in frames
        if obs.extra.get(key) is not None
    ]
    return float(np.mean(vals)) if vals else None


# ----------------------------------------------------------------------
# 兼容层（合并双方产出时补齐；核心 compute 逻辑不变）
# ----------------------------------------------------------------------
# Q_SIGNAL_KEYS：Q_KEYS 的规范别名（aggregator / 早期 __init__ 消费名）
Q_SIGNAL_KEYS: tuple[str, ...] = Q_KEYS

# Q_glare（反光污染帧占比）为 Q_interf 的反光子集口径：docs/06 T04 要求
# quality_signals.csv "含 Q_glare/Q_interf" 两行都存在，即使值为 None。
_Q_GLARE = "Q_glare"

# 有门限的信号（docs/04 §5 阈值表；None = 无固定门限/布尔量）
_Q_THRESHOLD_FIELDS: dict[str, str | None] = {
    "Q_det": "q_det_min",
    "Q_fdet": None,
    "Q_vis": None,
    "Q_track": "q_track_min",
    "Q_ids": None,
    "Q_interf": "q_interf_max",
    "Q_fg": "q_fg_min",
    "Q_calib": None,
    "Q_baseline": None,
    "Q_pelletloss": "pelletloss_degrade",
    "Q_n0gap": "n0_gap_warn",
    "Q_censored": None,
    "Q_motion": None,
}


def to_quality_rows(
    signals: dict[str, Any], thresholds: Thresholds | None = None
) -> list[dict[str, Any]]:
    """quality_signals.csv 行（signal / value / threshold / passed）。

    docs/04 §6.3：附 threshold 与 passed 两列，用户排查"为什么这个指标
    没输出"时不必回头翻 run_config.yaml。

    passed 规则：
        - 有门限的数值信号：值未越门限 = True，越门限 = False；
        - 无门限或值 None：passed = None（无法判定，绝不写 False 冒充）。
    """
    th = thresholds if thresholds is not None else Thresholds()
    rows: list[dict[str, Any]] = []
    keys = list(Q_KEYS) + [_Q_GLARE]
    for k in keys:
        if k == _Q_GLARE:
            val = signals.get(_Q_GLARE, signals.get("Q_interf"))
        else:
            val = signals.get(k)
        tname = _Q_THRESHOLD_FIELDS.get(k)
        thr = getattr(th, tname, None) if tname is not None else None
        passed: bool | None = None
        if val is not None and thr is not None and isinstance(val, (int, float)):
            if k in ("Q_det", "Q_track", "Q_fg"):
                passed = bool(float(val) >= float(thr))
            else:
                passed = bool(float(val) <= float(thr))
        rows.append(
            {"signal": k, "value": val, "threshold": thr, "passed": passed}
        )
    return rows

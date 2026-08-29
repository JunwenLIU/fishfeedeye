"""B1 · 参考区噪声扣除 α（T04）。

职责（docs/04 §3.2 B1 内联约定 + docs/06 T04）：
    - 参考区扣除系数 α **必须由基线期回归得到**（禁止拍脑袋默认值）：
        α = Σ(feed·ref) / Σ(ref²)   （过原点最小二乘）
      基线期样本不足（< min_samples）或参考区信号零变异 → α=None + note，
      此时**不得静默不扣除**——必须显式 flag no_noise_correction 并提示
      "户外波浪可能虚增活跃度"；
    - 参考区未定义 → 同样 no_noise_correction + 告警。

两个使用层次：
    - 函数层（estimate_alpha / apply_reference_correction）：纯数值内核，
      单测直接调用；
    - 类层（ReferenceCorrection / ReferenceCorrectionResult）：带 ROI 与
      基线语义的门面（spatial_heterogeneity 消费），fit() 负责样本数/参考区
      存在性/零变异的全部失效分支，apply() 在 α 缺失时原样返回并保留告警。

任务编号：T04。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.core.roi import ROI

__all__ = [
    "ReferenceCorrection",
    "ReferenceCorrectionResult",
    "estimate_alpha",
    "apply_reference_correction",
    "NO_NOISE_CORRECTION_FLAG",
    "no_reference_warning",
]

NO_NOISE_CORRECTION_FLAG = "no_noise_correction"
# 基线期回归 α 的最小样本数（docs/04 §5 未给此阈值 → 工程取值 5）。
# 取值依据：过原点最小二乘只有 1 个自由度参数，5 个基线期样本足以辨识 α；
# 少于 5 时 α 的抽样误差过大，按契约「α 必须由基线回归得到，不设默认值」
# 处理为不可估计 + no_noise_correction 告警。
# 注：T02 preprocess.ReferenceZoneCorrector.min_baseline_samples=8 属于
# **逐像素**层面的另一条链路（样本量充裕得多），与本处的曲线级口径不同源，
# 不强行对齐（此前声称"对齐"是错的：两者样本量级差两个数量级）。
MIN_BASELINE_SAMPLES = 5


def estimate_alpha(
    feed_signal: Sequence[float],
    ref_signal: Sequence[float],
    min_samples: int = MIN_BASELINE_SAMPLES,
) -> tuple[float | None, str | None]:
    """基线期回归估计扣除系数 α（过原点最小二乘）。

    Args:
        feed_signal: 投喂区信号序列（基线期）。
        ref_signal: 参考区信号序列（与 feed 一一对应）。
        min_samples: 最小基线样本数（默认 MIN_BASELINE_SAMPLES=5）。

    Returns:
        (alpha, note)：alpha=None 时 note 给出不可估计原因。
    """
    f = np.asarray(feed_signal, dtype=float)
    r = np.asarray(ref_signal, dtype=float)
    if f.size != r.size:
        return None, f"投喂区/参考区信号长度不一致（{f.size} != {r.size}）"
    if f.size < min_samples:
        return None, f"基线期样本不足（{f.size} < {min_samples}）：无法回归 α"
    denom = float(np.sum(r * r))
    if denom <= 1e-12:
        return None, "参考区信号零变异：α 不可辨识"
    alpha = float(np.sum(f * r) / denom)
    return alpha, None


def apply_reference_correction(
    feed_signal: np.ndarray, ref_signal: np.ndarray, alpha: float
) -> np.ndarray:
    """扣除后的投喂区信号：feed − α·ref。"""
    return np.asarray(feed_signal, dtype=float) - alpha * np.asarray(
        ref_signal, dtype=float
    )


def no_reference_warning() -> str:
    """无参考区/α 不可得时的强制告警文案（docs/06 T04）。"""
    return (
        "未做参考区噪声扣除（no_noise_correction）：户外波浪可能虚增活跃度，"
        "跨组比较时两组的波浪条件需一致性检查"
    )


def _warning_with_reason(reason: str | None) -> str:
    """把"具体失效原因"拼进告警文案。

    为什么：只说"未做噪声扣除"，用户无从判断该补参考区还是延长基线段。
    告警必须自带可执行的下一步（docs/04 §4.5 排查引导纪律）。
    """
    base = no_reference_warning()
    if not reason:
        return base
    return f"{base}；原因：{reason}"


@dataclass
class ReferenceCorrectionResult:
    """ReferenceCorrection.fit 的产物。

    Attributes:
        alpha: 扣除系数（None = 不可估计，绝不默认 0/1 冒充）。
        n_samples: 参与回归的基线期样本数。
        warning: α 不可得时的告警（None = 校正有效）。告警由通报文案 +
            **具体失效原因**（无参考区 / 样本不足 / 零变异）拼成——
            用户据此才知道该补参考区还是补基线段，故 warning 必须自带原因。
        note: 附加口径说明。

    @property
        available: α 是否可用（不可用时调用方必须挂 no_noise_correction）。
    """

    alpha: float | None = None
    n_samples: int = 0
    warning: str | None = None
    note: str | None = None

    @property
    def available(self) -> bool:
        """α 是否可用（不可用时调用方必须挂 no_noise_correction）。"""
        return self.alpha is not None


class ReferenceCorrection:
    """参考区噪声扣除门面（B1 消费；α 只能来自基线期回归）。

    用法::

        corr = ReferenceCorrection(roi)
        result = corr.fit([(feed_i, ref_i), ...])   # 基线期样本对
        m_corrected = corr.apply(m_feed, m_ref, result)
    """

    def __init__(
        self,
        roi: ROI | None = None,
        min_samples: int = MIN_BASELINE_SAMPLES,
    ) -> None:
        self._roi = roi
        self._min_samples = int(min_samples)

    def has_reference_zone(self) -> bool:
        """ROI 是否定义了参考区。"""
        return self._roi is not None and self._roi.reference_zone is not None

    def fit(
        self, baseline_pairs: Sequence[tuple[float, float]]
    ) -> ReferenceCorrectionResult:
        """从基线期 (投喂区, 参考区) 信号对回归 α。

        Args:
            baseline_pairs: [(feed_i, ref_i), ...]，基线期（t < 0）逐帧样本。

        Returns:
            ReferenceCorrectionResult：α=None 时 warning 必非空
            （无参考区 / 样本不足 / 零变异，三个失效分支全覆盖）。
        """
        if not self.has_reference_zone():
            note = "ROI.reference_zone 未定义：无噪声参照，不做扣除"
            return ReferenceCorrectionResult(
                alpha=None,
                n_samples=0,
                warning=_warning_with_reason(note),
                note=note,
            )
        pairs = list(baseline_pairs)
        n = len(pairs)
        if n == 0:
            note = "基线期无帧差样本（无基线或基线帧无图像）：α 不可估计"
            return ReferenceCorrectionResult(
                alpha=None,
                n_samples=0,
                warning=_warning_with_reason(note),
                note=note,
            )
        feed = np.asarray([p[0] for p in pairs], dtype=float)
        ref = np.asarray([p[1] for p in pairs], dtype=float)
        if n < self._min_samples:
            note = (
                f"基线期样本不足（{n} < {self._min_samples}）："
                "α 必须由基线回归得到，不设默认值"
            )
            return ReferenceCorrectionResult(
                alpha=None,
                n_samples=n,
                warning=_warning_with_reason(note),
                note=note,
            )
        alpha, note = estimate_alpha(feed, ref, self._min_samples)
        if alpha is None:
            return ReferenceCorrectionResult(
                alpha=None, n_samples=n,
                warning=_warning_with_reason(note), note=note,
            )
        return ReferenceCorrectionResult(
            alpha=alpha,
            n_samples=n,
            warning=None,
            note=(
                "α = Σ(feed·ref)/Σ(ref²)（过原点最小二乘，"
                f"n={n} 基线期样本）"
            ),
        )

    def apply(
        self,
        feed_signal: np.ndarray,
        ref_signal: np.ndarray,
        result: ReferenceCorrectionResult,
    ) -> np.ndarray:
        """应用扣除；α 不可得时原样返回（告警由调用方挂 flag 表达）。"""
        feed = np.asarray(feed_signal, dtype=float)
        ref = np.asarray(ref_signal, dtype=float)
        if result.alpha is None or feed.shape != ref.shape:
            return feed
        return feed - result.alpha * ref

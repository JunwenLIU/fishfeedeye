"""预处理（T02）：稳像 / 透视校正应用 / CLAHE / 参考区噪声扣除。

职责（docs/06 §2 preprocess.py + T02 任务定义）：
    1. 稳像：三脚架微动场景（第二轮决策 ③），用相位相关估计帧间全局
       平移并累积补偿；平移幅度超限（机位真实移动）时停止补偿并显式
       note，绝不把"机位移动"硬拉回参考系；
    2. ROI 掩膜与透视校正应用：复用 core/roi.py 掩膜生成与
       core/homography.py 的 warp（可选整幅校正，默认关闭——校正后像素
       坐标系改变，ROI 必须同步定义在校正后坐标系，此边界显式声明）；
    3. CLAHE 对比度增强（明度通道，保色）；
    4. 参考区噪声扣除（docs/06 §1 挑战 #2 对策）：户外波浪/反光用参考区
       逐帧扣除，扣除系数 α 必须由基线期回归得到（禁止拍脑袋定值）。

纪律：
    - 稳像只修平移抖动，不修旋转/遮挡（能力边界，须向用户声明）；
    - 参考区缺失或基线不足 → α=None + flag no_noise_correction +
      明示"户外波浪可能虚增活跃度"，绝不静默跳过。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.core.config import RunConfig
from src.core.homography import HomographyCalibrator
from src.core.roi import ROI

__all__ = [
    "Stabilizer",
    "PreprocessResult",
    "Preprocessor",
    "ReferenceZoneCorrector",
    "zone_frame_diff",
]

try:  # OpenCV 可选导入（无 cv2 时退化为恒等预处理，纯逻辑仍可单测）
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False


# ----------------------------------------------------------------------
# 稳像（三脚架微动）
# ----------------------------------------------------------------------
@dataclass
class Stabilizer:
    """帧间全局平移估计与补偿（相位相关，三脚架微动场景）。

    Attributes:
        max_correction_px: 累积补偿上限（像素）。超过判定为机位真实移动
            （非抖动），停止补偿并记录 note——硬拉回会撕裂画面语义。
    """

    max_correction_px: float = 25.0

    def __post_init__(self) -> None:
        self._ref_gray: np.ndarray | None = None
        self._offset_xy: tuple[float, float] = (0.0, 0.0)
        self.shift_history: list[tuple[float, float]] = []
        self.notes: list[str] = []

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """处理一帧，返回（补偿后帧, 累积补偿量 (dx, dy) 像素）。

        首帧成为参考帧（零偏移）；后续帧与参考帧做相位相关得全局平移，
        用累积偏移的相反方向 warp 回参考坐标系。
        """
        if not _HAS_CV2:
            self.shift_history.append((0.0, 0.0))
            return frame_bgr, (0.0, 0.0)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._ref_gray is None:
            self._ref_gray = gray
            self.shift_history.append((0.0, 0.0))
            return frame_bgr, (0.0, 0.0)

        # 相位相关：当前帧相对参考帧的平移 (dx, dy)。
        # OpenCV 版本差异：4.x 返回 (dx, dy)；5.x 返回 ((dx, dy), response)。
        ret = cv2.phaseCorrelate(
            np.float32(self._ref_gray), np.float32(gray)
        )
        if len(ret) == 2 and isinstance(ret[0], (tuple, list, np.ndarray)):
            dx, dy = float(ret[0][0]), float(ret[0][1])
        else:
            dx, dy = float(ret[0]), float(ret[1])
        self.shift_history.append((float(dx), float(dy)))
        cand = (self._offset_xy[0] + dx, self._offset_xy[1] + dy)
        mag = float(np.hypot(*cand))
        if mag > self.max_correction_px:
            self.notes.append(
                f"累积平移 {mag:.1f}px 超过补偿上限 {self.max_correction_px}px："
                "判定为机位移动（非抖动），本帧起停止补偿"
            )
            self._ref_gray = gray  # 参考系切换到当前帧
            self._offset_xy = (0.0, 0.0)
            return frame_bgr, (0.0, 0.0)
        self._offset_xy = cand

        h, w = frame_bgr.shape[:2]
        M = np.float32([[1.0, 0.0, -self._offset_xy[0]],
                        [0.0, 1.0, -self._offset_xy[1]]])
        out = cv2.warpAffine(
            frame_bgr, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return out, self._offset_xy

    @property
    def mean_abs_shift_px(self) -> float:
        """Q_motion 代理量：帧间平移幅度均值（px/帧）。"""
        if not self.shift_history:
            return 0.0
        return float(np.mean([np.hypot(s[0], s[1]) for s in self.shift_history]))


# ----------------------------------------------------------------------
# 预处理器
# ----------------------------------------------------------------------
@dataclass
class PreprocessResult:
    """单帧预处理输出。"""

    frame: np.ndarray
    shift_px: tuple[float, float]     # 稳像累积补偿量（本帧相对参考系）
    warped: bool                      # 是否做了整幅透视校正
    clahe_applied: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class Preprocessor:
    """预处理管线：稳像 → （可选）透视校正 → CLAHE。

    Attributes:
        roi: 区域定义（像素坐标系 = 输入帧坐标系）。
        calibrator: 已拟合的单应性标定器（None = 未标定）。
        apply_warp: 是否整幅透视校正。默认 False：校正会改变像素坐标系，
            ROI/检测坐标须同步换算；面积积分轨依赖原始像素面积时建议关闭。
        stabilize: 是否稳像。
        apply_clahe: 是否 CLAHE 增强。
        clahe_clip / clahe_grid: CLAHE 参数。
    """

    roi: ROI | None = None
    calibrator: HomographyCalibrator | None = None
    apply_warp: bool = False
    stabilize: bool = True
    apply_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    def __post_init__(self) -> None:
        self._stabilizer = Stabilizer()
        self._clahe = None
        if _HAS_CV2 and self.apply_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=float(self.clahe_clip),
                tileGridSize=(int(self.clahe_grid), int(self.clahe_grid)),
            )
        self.notes: list[str] = []
        if self.apply_warp and (self.calibrator is None or not self.calibrator.fitted):
            self.notes.append(
                "apply_warp=True 但单应性未标定：透视校正跳过（只修尺度、不修遮挡）"
            )

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> PreprocessResult:
        """处理一帧。调用方负责按时间顺序喂帧（稳像依赖帧序）。"""
        notes: list[str] = []
        out = frame_bgr
        shift: tuple[float, float] = (0.0, 0.0)
        if self.stabilize:
            out, shift = self._stabilizer.process(out)
            notes.extend(self._stabilizer.notes)
            self._stabilizer.notes = []

        warped = False
        if self.apply_warp and self.calibrator is not None and self.calibrator.fitted:
            out = self.calibrator.warp(out)
            warped = True
            notes.append("已整幅透视校正：ROI/检测坐标须位于校正后坐标系")

        clahe_applied = False
        if self._clahe is not None:
            out = self._clahe_bgr(out)
            clahe_applied = True
        return PreprocessResult(
            frame=out, shift_px=shift, warped=warped,
            clahe_applied=clahe_applied, notes=notes,
        )

    def _clahe_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        """CLAHE 仅作用于明度通道（YUV 的 Y），保持色彩信息。"""
        if not _HAS_CV2:
            return frame_bgr
        yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = self._clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @property
    def mean_abs_shift_px(self) -> float:
        """稳像帧间平移均值（Q_motion 代理量，写质量信号）。"""
        return self._stabilizer.mean_abs_shift_px


# ----------------------------------------------------------------------
# 参考区噪声扣除（α 由基线期回归）
# ----------------------------------------------------------------------
def zone_frame_diff(
    prev_gray: np.ndarray, cur_gray: np.ndarray, mask: np.ndarray
) -> float:
    """两帧灰度在掩膜内的平均绝对差（区域信号提取，B1-1 同源口径）。"""
    if prev_gray.shape != cur_gray.shape or prev_gray.shape != mask.shape:
        raise ValueError(
            f"形状不一致: prev{prev_gray.shape} cur{cur_gray.shape} mask{mask.shape}"
        )
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0.0
    return float(np.mean(np.abs(cur_gray[m].astype(float) - prev_gray[m].astype(float))))


@dataclass
class ReferenceZoneCorrector:
    """参考区噪声扣除器（docs/06 §1 挑战 #2 对策）。

    α 必须由基线期（t<0，无饲料无摄食）的「投喂区信号 ~ 参考区信号」
    最小二乘回归得到——此时两区信号同源于水面波浪/光照漂移；投喂后
    投喂区信号 = 摄食活动 + 波浪，扣除 α×参考区信号即得活动成分。

    纪律：
        - 无参考区 / 基线样本不足（<8）/ 参考区信号无变异 → α=None，
          调用方必须打 no_noise_correction flag 并明示风险，不得拍脑袋；
        - 回归用「去均值后的过原点斜率」（截距吸收基线均值差）。
    """

    min_baseline_samples: int = 8

    def __post_init__(self) -> None:
        self.alpha: float | None = None
        self.fit_notes: list[str] = []

    def fit(
        self, feed_signals: Sequence[float], ref_signals: Sequence[float]
    ) -> float | None:
        """基线期回归求 α。

        Returns:
            α（斜率）；无法可靠估计时 None（绝不返回 0 冒充"无噪声"）。
        """
        a = np.asarray(feed_signals, dtype=float)
        b = np.asarray(ref_signals, dtype=float)
        if a.shape != b.shape:
            raise ValueError("feed/ref 信号长度必须一致")
        if a.size < self.min_baseline_samples:
            self.alpha = None
            self.fit_notes.append(
                f"基线样本 {a.size} < {self.min_baseline_samples}：α 无法回归，"
                "参考区扣除不可用（no_noise_correction）"
            )
            return None
        da, db = a - a.mean(), b - b.mean()
        denom = float(np.dot(db, db))
        if denom <= 1e-12:
            self.alpha = None
            self.fit_notes.append(
                "参考区信号零变异：α 无法回归，参考区扣除不可用（no_noise_correction）"
            )
            return None
        self.alpha = float(np.dot(da, db) / denom)
        return self.alpha

    def correct(self, feed_signal: float, ref_signal: float) -> float | None:
        """扣除参考区噪声：feed − α×(ref − ref_baseline_mean 尚未存) 的简式。

        返回 feed − α×ref；α 不可得时返回 None（调用方不得当作 0）。
        基线均值的扣除由调用方在聚合层完成（本类只做系数与点态扣除）。
        """
        if self.alpha is None:
            return None
        return float(feed_signal - self.alpha * ref_signal)

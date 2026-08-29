"""YOLOE 开放词汇检测轨（T03 冷启动）：零训练检测颗粒。

职责（docs/06 §6 T03 + docs/06 §1 挑战 #1）：
    L0 无标注阶段的主检测通道：COCO 没有 fish/feed pellet 类，YOLOE
    开放词汇用文本提示（默认 "feed pellet"）零训练跑。复用 yolo_detector
    的 TiledPredictor 切片推理（SAHI 同构：tile + overlap + NMS）。

    L1 微调权重出现后由调用方优先切换 YoloPelletDetector（本轨降级为
    交叉校验）；切换决策不写死在检测器内部。

可测性设计：
    predict_fn 注入 → 无 torch/权重也能测切片/NMS/过滤逻辑。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.frame_context import PelletDetections
from src.core.roi import ROI
from src.pipeline.detectors.base import DetectStats, PelletDetector, filter_pellets
from src.pipeline.detectors.yolo_detector import (
    BoxConf,
    PredictFn,
    TiledPredictor,
    _make_ultralytics_predict_fn,
)

__all__ = ["YoloEDetector", "DEFAULT_YOLOE_PROMPT", "DEFAULT_YOLOE_WEIGHTS"]

# 开放词汇提示：浮性膨化饲料颗粒（docs/06 T03 内联约定）
DEFAULT_YOLOE_PROMPT = "feed pellet"
DEFAULT_YOLOE_WEIGHTS = "yoloe-v8s-seg.pt"


class YoloEDetector(PelletDetector):
    """YOLOE 开放词汇冷启动检测轨（无需训练）。"""

    def __init__(
        self,
        prompt: str = DEFAULT_YOLOE_PROMPT,
        weights: str = DEFAULT_YOLOE_WEIGHTS,
        predict_fn: PredictFn | None = None,
        tile_size: int = 640,
        overlap: float = 0.2,
        conf_min: float = 0.25,
        iou_nms: float = 0.5,
        imgsz: int = 1280,
        min_area_px: float | None = None,
        max_area_px: float | None = None,
        device: str | None = None,
    ) -> None:
        """
        Args:
            prompt: 开放词汇文本提示。
            weights: YOLOE 预训练权重名或路径。
            predict_fn: 测试注入（提供时跳过模型加载与下载）。
            其余参数同 TiledPredictor / 检测轨通用过滤。
        """
        super().__init__()
        self.prompt = prompt
        self._unavailable_reason: str | None = None
        self._tiled: TiledPredictor | None = None
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px
        if predict_fn is not None:
            self._tiled = TiledPredictor(predict_fn, tile_size, overlap, conf_min, iou_nms)
            return
        try:
            from ultralytics import YOLOE  # 延迟导入：依赖缺失不阻断包导入
        except Exception as exc:
            self._unavailable_reason = (
                f"ultralytics YOLOE 不可用: {exc}（冷启动轨需要 ultralytics>=8.3 "
                "与网络下载预训练权重）"
            )
            return
        try:
            model = YOLOE(weights)
            # 开放词汇：把文本提示注册为检测类别
            model.set_classes([self.prompt], model.get_text_pe([self.prompt]))
            if device is not None:
                model.to(device)
            self._tiled = TiledPredictor(
                _make_ultralytics_predict_fn(model, imgsz, conf_min),
                tile_size, overlap, conf_min, iou_nms,
            )
        except Exception as exc:  # 权重下载失败等
            self._unavailable_reason = f"YOLOE 初始化失败: {exc}"

    def name(self) -> str:
        return "yoloe_ov"

    def available(self) -> bool:
        return self._tiled is not None

    def unavailable_reason(self) -> str | None:
        if self.available():
            return None
        return self._unavailable_reason

    def detect(
        self,
        frame: np.ndarray,
        t_s: float | None = None,
        roi: ROI | None = None,
    ) -> PelletDetections:
        if self._tiled is None:
            self.last_stats = DetectStats(
                n=0, source="det", available=False,
                reason=self.unavailable_reason(),
            )
            return PelletDetections.empty()
        dets = self._tiled.predict(frame)
        zone = "pellet_zone" if (roi is not None and roi.pellet_zone is not None) else "arena"
        dets = filter_pellets(
            dets, min_area_px=self.min_area_px, max_area_px=self.max_area_px,
            roi=roi, zone=zone,
        )
        self.last_stats = DetectStats(
            n=dets.n_det(), source="det", available=True,
            extra={"model": self.name(), "prompt": self.prompt},
        )
        return dets

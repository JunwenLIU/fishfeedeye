"""YOLO 颗粒检测轨（T03）：Ultralytics + SAHI 切片推理封装。

职责（docs/06 §6 T03）：
    L1 微调权重（models/pellet_user.pt）出现后的主通道。小目标必需：
    imgsz=1280 + 切片推理（tile + overlap + 回拼 + 全局 NMS）。
    SAHI 库可用时直接复用其切片约定；否则用本模块的等效实现
    （网格切片 + NMS，与 SAHI 核心算法同构，避免硬依赖）。

可测性设计：
    构造期可注入 predict_fn(tile_bgr) -> list[(xyxy, conf)]，
    使切片/NMS/过滤逻辑无需真实权重即可单测（docs/06 T03 验收：
    打桩测逻辑）。真实权重路径延迟导入 ultralytics。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.core.frame_context import PelletDetections
from src.core.roi import ROI
from src.pipeline.detectors.base import DetectStats, PelletDetector, filter_pellets

__all__ = ["TiledPredictor", "YoloPelletDetector", "nms_xyxy"]

# 单框 (x0, y0, x1, y1, conf)
BoxConf = tuple[tuple[float, float, float, float], float]
PredictFn = Callable[[np.ndarray], list[BoxConf]]


# ----------------------------------------------------------------------
# 切片推理核心（SAHI 同构：网格切片 + overlap + 回拼 + 全局 NMS）
# ----------------------------------------------------------------------
class TiledPredictor:
    """切片推理器：把整帧切成重叠 tile，逐 tile 调 predict_fn，回拼去重。"""

    def __init__(
        self,
        predict_fn: PredictFn,
        tile_size: int = 640,
        overlap: float = 0.2,
        conf_min: float = 0.25,
        iou_nms: float = 0.5,
    ) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size 必须为正，收到 {tile_size}")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap 须在 [0,1)，收到 {overlap}")
        self.predict_fn = predict_fn
        self.tile_size = int(tile_size)
        self.stride = max(1, int(round(tile_size * (1.0 - overlap))))
        self.conf_min = float(conf_min)
        self.iou_nms = float(iou_nms)

    # ------------------------------------------------------------------
    def tile_origins(self, width: int, height: int) -> list[tuple[int, int]]:
        """网格切片原点（最后一格对齐右/下边界，保证全覆盖）。"""
        origins: list[tuple[int, int]] = []
        xs = _axis_positions(width, self.tile_size, self.stride)
        ys = _axis_positions(height, self.tile_size, self.stride)
        for y in ys:
            for x in xs:
                origins.append((x, y))
        return origins

    def predict(self, frame_bgr: np.ndarray) -> PelletDetections:
        """整帧切片推理 → 全局 NMS → PelletDetections。"""
        h, w = frame_bgr.shape[:2]
        boxes: list[list[float]] = []
        confs: list[float] = []
        for (x0, y0) in self.tile_origins(w, h):
            x1 = min(x0 + self.tile_size, w)
            y1 = min(y0 + self.tile_size, h)
            tile = frame_bgr[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            for (bx, conf) in self.predict_fn(tile):
                if conf < self.conf_min:
                    continue
                # 切片内坐标 → 整帧坐标
                boxes.append([bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0])
                confs.append(float(conf))
        if not boxes:
            return PelletDetections.empty()
        arr = np.asarray(boxes, dtype=float)
        cf = np.asarray(confs, dtype=float)
        keep = nms_xyxy(arr, cf, self.iou_nms)
        return PelletDetections(xyxy=arr[keep], conf=cf[keep])


def _axis_positions(total: int, tile: int, stride: int) -> list[int]:
    """单轴切片原点（含末点对齐）。"""
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    last = total - tile
    if pos[-1] != last:
        pos.append(last)
    return pos


def nms_xyxy(boxes: np.ndarray, conf: np.ndarray, iou_thr: float) -> list[int]:
    """非极大值抑制（按置信度降序贪心）。返回保留索引（升序）。"""
    if boxes.shape[0] == 0:
        return []
    x0 = boxes[:, 0]; y0 = boxes[:, 1]
    x1 = boxes[:, 2]; y1 = boxes[:, 3]
    areas = np.maximum(x1 - x0, 0.0) * np.maximum(y1 - y0, 0.0)
    order = np.argsort(-conf)
    keep: list[int] = []
    suppressed = np.zeros(boxes.shape[0], dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        xx0 = np.maximum(x0[i], x0[order])
        yy0 = np.maximum(y0[i], y0[order])
        xx1 = np.minimum(x1[i], x1[order])
        yy1 = np.minimum(y1[i], y1[order])
        inter = np.maximum(xx1 - xx0, 0.0) * np.maximum(yy1 - yy0, 0.0)
        union = areas[i] + areas[order] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        hit = (iou > iou_thr) & (~suppressed[order])
        # 抑制除自身外的命中框
        for j in np.asarray(order)[hit]:
            if j != i:
                suppressed[int(j)] = True
    return sorted(keep)


# ----------------------------------------------------------------------
# Ultralytics 适配（真实权重路径，延迟导入）
# ----------------------------------------------------------------------
def _make_ultralytics_predict_fn(
    model,
    imgsz: int,
    conf_min: float,
) -> PredictFn:
    """把 ultralytics 模型适配为 TiledPredictor 的 predict_fn。"""

    def fn(tile_bgr: np.ndarray) -> list[BoxConf]:
        results = model.predict(
            tile_bgr, imgsz=int(imgsz), conf=float(conf_min), verbose=False
        )
        out: list[BoxConf] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                xyxy = b.xyxy.cpu().numpy().reshape(-1)
                out.append((
                    (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    float(b.conf.item()),
                ))
        return out

    return fn


def default_weights_path() -> Path:
    """L1 微调权重默认位置（项目根 models/pellet_user.pt；.gitignore 不入库）。"""
    return Path(__file__).resolve().parents[3] / "models" / "pellet_user.pt"


# ----------------------------------------------------------------------
# 检测轨
# ----------------------------------------------------------------------
class YoloPelletDetector(PelletDetector):
    """YOLO 检测轨（L1 微调权重主通道 + SAHI 切片）。"""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        predict_fn: PredictFn | None = None,
        tile_size: int = 640,
        overlap: float = 0.2,
        conf_min: float = 0.25,
        iou_nms: float = 0.5,
        imgsz: int = 1280,
        min_area_px: float | None = None,
        max_area_px: float | None = None,
    ) -> None:
        """
        Args:
            weights_path: 权重路径；None = 自动探测 models/pellet_user.pt。
            predict_fn: 测试注入的伪推理函数（提供时跳过权重加载）。
            tile_size / overlap / conf_min / iou_nms: 切片推理参数。
            imgsz: ultralytics 推理分辨率（小目标必需 1280）。
            min_area_px / max_area_px: 颗粒框尺寸过滤。
        """
        super().__init__()
        self._unavailable_reason: str | None = None
        self._tiled: TiledPredictor | None = None
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px
        if predict_fn is not None:
            self._tiled = TiledPredictor(predict_fn, tile_size, overlap, conf_min, iou_nms)
            return
        wp = Path(weights_path) if weights_path is not None else default_weights_path()
        if not wp.exists():
            self._unavailable_reason = (
                f"YOLO 权重不存在: {wp}（L1 微调后自动优先；冷启动请用 "
                "YoloEDetector 开放词汇轨）"
            )
            return
        try:
            from ultralytics import YOLO  # 延迟导入
        except Exception as exc:  # pragma: no cover - 依赖缺失路径
            self._unavailable_reason = f"ultralytics 不可用: {exc}"
            return
        model = YOLO(str(wp))
        self._tiled = TiledPredictor(
            _make_ultralytics_predict_fn(model, imgsz, conf_min),
            tile_size, overlap, conf_min, iou_nms,
        )

    def name(self) -> str:
        return "yolo_det"

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
            reason=None,
            extra={"model": self.name()},
        )
        return dets

"""饲料颗粒动态学标定（T02）：A_single(t) 与 v_sink_max 的测量逻辑。

职责（docs/06 §6 T02 + docs/04 §5.1）：
    消费「无鱼纯饲料」对照视频，产出：
      1. A_single(t) 曲线 —— 单颗颗粒投影面积时序（面积积分轨的基准）。
         浮性膨化料吸水膨胀，A_single 随时间变化，**必须按时间校正**；
      2. v_sink_max —— 颗粒位移速度 95 分位（A14 关联的物理约束）。
         ⚠️ 禁止硬编码默认值，不同饲料/粒径/水温差异极大；
      3. association_radius —— 建议 2–3 倍平均粒径（取 2.5 倍等效直径）。

    有鱼视频不能用于标定（消失/移动被摄食污染，测出的是"摄食+沉降"
    混合速度）。

纪律：
    - A_single 只取「孤立颗粒」（面积落在全局中位数 ±25% 带内）的面积，
      粘连团块不参与（其面积是多颗之和，混入会系统性虚增基准）；
    - 某帧没有任何孤立颗粒 → 该点为 None（缺测），绝不插值冒充观测；
      面积积分轨消费时用相邻有效点的线性插值（这是"定位量值"的合法
      插值，docs/04 §3 ④）；
    - 全程无有效点 → 标定失败（曲线整体 unavailable，不硬编码猜测值）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from src.utils.logging import get_logger
from src.utils.video_io import VideoReader

__all__ = [
    "SegmentationParams",
    "PelletBlob",
    "ASingleCurve",
    "PelletDynamicsCalibration",
    "segment_pellets",
    "extract_blobs",
    "measure_a_single",
    "measure_v_sink_px_s",
    "calibrate_video",
]

_log = get_logger("pipeline.pellet_dynamics")

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False


# ----------------------------------------------------------------------
# 前景分割参数（浮性膨化料 = 亮色颗粒；与面积积分轨共用同一实现）
# ----------------------------------------------------------------------
@dataclass
class SegmentationParams:
    """HSV 明度/饱和度阈值 + 形态学参数（浮性料亮色分割）。

    取值依据：水面/背景明度低（V<100），浮性膨化料亮（V 高）且带
    色泽（S 中等）；高饱和纯色多为反光/浮标，剔除。
    """

    hsv_v_min: int = 160          # 明度下限（0-255）
    hsv_s_min: int = 30           # 饱和度下限（剔除灰白水沫）
    hsv_s_max: int = 220          # 饱和度上限（剔除高饱和反光/异物）
    min_blob_area_px: int = 6     # 噪点过滤
    morph_open_k: int = 3         # 开运算核（去噪）
    morph_close_k: int = 5        # 闭运算核（填孔）


def segment_pellets(frame_bgr: np.ndarray, params: SegmentationParams | None = None) -> np.ndarray:
    """浮性饲料前景分割 → 布尔掩膜 (H, W)。

    HSV 阈值分割 + 形态学开闭。帧须为 BGR；无 cv2 时退化为灰度阈值
    （仅测试场景可用，能力边界显式声明）。
    """
    p = params or SegmentationParams()
    if _HAS_CV2:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, p.hsv_s_min, p.hsv_v_min], dtype=np.uint8),
            np.array([179, p.hsv_s_max, 255], dtype=np.uint8),
        )
        if p.morph_open_k > 1:
            k = np.ones((p.morph_open_k, p.morph_open_k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if p.morph_close_k > 1:
            k = np.ones((p.morph_close_k, p.morph_close_k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask.astype(bool)
    # 无 cv2 兜底（纯 numpy）：灰度高亮度阈值
    gray = frame_bgr.mean(axis=2)
    return (gray >= p.hsv_v_min).astype(bool)


@dataclass
class PelletBlob:
    """单个连通域（可能是孤立颗粒或粘连团块）。"""

    centroid: tuple[float, float]  # (x, y) 像素
    area_px: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1


def extract_blobs(mask: np.ndarray, min_area_px: int = 6) -> list[PelletBlob]:
    """连通域分析 → 颗粒色块列表（按面积降序）。

    无 cv2 时用纯 numpy 的简单洪泛标记（小图可接受，性能边界显式）。
    """
    m = np.asarray(mask, dtype=bool)
    if m.any() and _HAS_CV2:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), connectivity=8
        )
        blobs: list[PelletBlob] = []
        for i in range(1, n):  # 0 = 背景
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue
            x0 = float(stats[i, cv2.CC_STAT_LEFT])
            y0 = float(stats[i, cv2.CC_STAT_TOP])
            w = float(stats[i, cv2.CC_STAT_WIDTH])
            h = float(stats[i, cv2.CC_STAT_HEIGHT])
            blobs.append(
                PelletBlob(
                    centroid=(float(centroids[i, 0]), float(centroids[i, 1])),
                    area_px=area,
                    bbox=(x0, y0, x0 + w, y0 + h),
                )
            )
        blobs.sort(key=lambda b: -b.area_px)
        return blobs
    # 纯 numpy 兜底：行扫描连通域（4-连通近似，仅测试小图）
    return _blobs_numpy(m, min_area_px)


def _blobs_numpy(mask: np.ndarray, min_area_px: int) -> list[PelletBlob]:
    """无 cv2 的连通域兜底（两遍扫描并查集）。"""
    h, w = mask.shape
    parent = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    labels = np.zeros((h, w), dtype=int)
    nxt = 1
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            left = labels[y, x - 1] if x > 0 else 0
            up = labels[y - 1, x] if y > 0 else 0
            if left and up:
                labels[y, x] = left
                union(left, up)
            elif left or up:
                labels[y, x] = left or up
            else:
                labels[y, x] = nxt
                parent[nxt] = nxt
                nxt += 1
    groups: dict[int, list[tuple[int, int]]] = {}
    for y in range(h):
        for x in range(w):
            if labels[y, x]:
                groups.setdefault(find(labels[y, x]), []).append((x, y))
    blobs: list[PelletBlob] = []
    for pts in groups.values():
        if len(pts) < min_area_px:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        blobs.append(
            PelletBlob(
                centroid=(float(np.mean(xs)), float(np.mean(ys))),
                area_px=float(len(pts)),
                bbox=(float(min(xs)), float(min(ys)),
                      float(max(xs)) + 1.0, float(max(ys)) + 1.0),
            )
        )
    blobs.sort(key=lambda b: -b.area_px)
    return blobs


# ----------------------------------------------------------------------
# A_single(t)
# ----------------------------------------------------------------------
@dataclass
class ASingleCurve:
    """A_single(t) 标定曲线（面积积分轨的基准）。

    Attributes:
        t_s: 采样时间戳（秒，相对标定视频起点；消费侧换算到 run 时间轴）。
        area_px: 单颗投影面积（像素²）；缺测帧为 None。
    """

    t_s: list[float] = field(default_factory=list)
    area_px: list[float | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.t_s) != len(self.area_px):
            raise ValueError("t_s 与 area_px 长度必须一致")

    @property
    def available(self) -> bool:
        """至少有一个有效观测点（None 点 = 缺测，不算）。"""
        return any(a is not None for a in self.area_px)

    def value_at(self, t: float) -> float | None:
        """t 时刻的 A_single：相邻有效点线性插值；端点外取最近有效点。

        这是「定位量值」的合法插值（docs/04 §3 ④）；若无任何有效点
        返回 None（调用方不得当 0）。
        """
        if not self.available:
            return None
        pts = [(t_, a) for t_, a in zip(self.t_s, self.area_px) if a is not None]
        ts = np.array([p[0] for p in pts], dtype=float)
        as_ = np.array([p[1] for p in pts], dtype=float)
        if t <= ts[0]:
            return float(as_[0])
        if t >= ts[-1]:
            return float(as_[-1])
        j = int(np.searchsorted(ts, t))
        t0, t1 = ts[j - 1], ts[j]
        a0, a1 = as_[j - 1], as_[j]
        if t1 <= t0:
            return float(a0)
        return float(a0 + (a1 - a0) * (t - t0) / (t1 - t0))

    def to_dict(self) -> dict:
        return {"t_s": [float(t) for t in self.t_s],
                "area_px": [None if a is None else float(a) for a in self.area_px]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ASingleCurve":
        if not d:
            return cls()
        return cls(t_s=[float(t) for t in d.get("t_s", [])],
                   area_px=[None if a is None else float(a)
                            for a in d.get("area_px", [])])


def measure_a_single(
    frames: Sequence[np.ndarray],
    t_s: Sequence[float],
    params: SegmentationParams | None = None,
    isolate_ratio: float = 0.25,
) -> ASingleCurve:
    """逐帧测量 A_single(t)。

    每帧取「孤立颗粒」面积（落在全局中位数 × (1±isolate_ratio) 带内的
    连通域）的中位数；无孤立颗粒的帧记 None（缺测，不插值冒充观测）。

    Args:
        frames: BGR 帧序列（标定视频解码帧或子采样）。
        t_s: 各帧时间戳（秒）。
        params: 分割参数。
        isolate_ratio: 孤立判定带宽（相对全局中位数）。
    """
    if len(frames) != len(t_s):
        raise ValueError("frames 与 t_s 长度必须一致")
    p = params or SegmentationParams()
    # 第一遍：全局面积中位数（孤立带中心）
    all_areas: list[float] = []
    per_frame_blobs: list[list[PelletBlob]] = []
    for f in frames:
        blobs = extract_blobs(segment_pellets(f, p), p.min_blob_area_px)
        per_frame_blobs.append(blobs)
        all_areas.extend(b.area_px for b in blobs)
    if not all_areas:
        return ASingleCurve(t_s=[float(t) for t in t_s],
                            area_px=[None] * len(t_s))
    med = float(np.median(all_areas))
    lo, hi = med * (1.0 - isolate_ratio), med * (1.0 + isolate_ratio)

    areas: list[float | None] = []
    for blobs in per_frame_blobs:
        isolated = [b.area_px for b in blobs if lo <= b.area_px <= hi]
        areas.append(float(np.median(isolated)) if isolated else None)
    return ASingleCurve(t_s=[float(t) for t in t_s], area_px=areas)


# ----------------------------------------------------------------------
# v_sink_max（95 分位位移速度）
# ----------------------------------------------------------------------
def measure_v_sink_px_s(
    frames: Sequence[np.ndarray],
    t_s: Sequence[float],
    params: SegmentationParams | None = None,
    link_radius_factor: float = 2.5,
) -> float | None:
    """帧间最近邻跟踪 → 位移速度 95 分位（px/s）。

    关联半径 = link_radius_factor × 全局颗粒等效半径（不硬编码）。
    速度 = 位移 / 真实 Δt（VFR 安全）。样本 < 5 → None（不得输出拍脑袋值）。
    """
    if len(frames) != len(t_s):
        raise ValueError("frames 与 t_s 长度必须一致")
    p = params or SegmentationParams()
    per_frame: list[list[PelletBlob]] = [
        extract_blobs(segment_pellets(f, p), p.min_blob_area_px) for f in frames
    ]
    all_areas = [b.area_px for blobs in per_frame for b in blobs]
    if not all_areas:
        return None
    mean_area = float(np.mean(all_areas))
    radius = float(np.sqrt(mean_area / np.pi))
    link_r = link_radius_factor * radius

    # 活动轨迹：[(上帧位置, 预计到达时间)]
    active: list[tuple[float, float]] = []
    speeds: list[float] = []
    for k, blobs in enumerate(per_frame):
        if k == 0:
            active = [b.centroid for b in blobs]
            continue
        dt = float(t_s[k] - t_s[k - 1])
        if dt <= 0:
            continue
        unmatched = list(active)
        new_active: list[tuple[float, float]] = []
        for b in blobs:
            best_j, best_d = -1, np.inf
            for j, (ax, ay) in enumerate(unmatched):
                d = float(np.hypot(b.centroid[0] - ax, b.centroid[1] - ay))
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= link_r:
                (ax, ay) = unmatched.pop(best_j)
                speeds.append(best_d / dt)
                new_active.append(b.centroid)
            else:
                new_active.append(b.centroid)
        active = new_active + unmatched  # 未匹配旧目标保留（短暂消失容忍）
    if len(speeds) < 5:
        return None
    return float(np.percentile(speeds, 95))


# ----------------------------------------------------------------------
# 标定结果容器与整视频标定
# ----------------------------------------------------------------------
@dataclass
class PelletDynamicsCalibration:
    """一次标定的全部产物（configs/calibration/<feed_id>.yaml 的载体）。"""

    feed_id: str
    a_single: ASingleCurve
    v_sink_max_px_s: float | None          # 95 分位位移速度（px/s）
    v_sink_max_mm_s: float | None          # 需要 px_per_mm；未提供 = None
    association_radius_px: float | None    # 建议 2.5 × 平均等效直径
    mean_pellet_area_px: float | None
    px_per_mm: float | None
    n_frames: int
    duration_s: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feed_id": self.feed_id,
            "a_single": self.a_single.to_dict(),
            "v_sink_max_px_s": self.v_sink_max_px_s,
            "v_sink_max_mm_s": self.v_sink_max_mm_s,
            "association_radius_px": self.association_radius_px,
            "mean_pellet_area_px": self.mean_pellet_area_px,
            "px_per_mm": self.px_per_mm,
            "n_frames": self.n_frames,
            "duration_s": self.duration_s,
            "notes": list(self.notes),
        }

    def to_yaml(self, path: str | Path) -> Path:
        """写出 configs/calibration/<feed_id>.yaml。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PelletDynamicsCalibration":
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        if not isinstance(d, dict):
            raise ValueError(f"标定文件内容不是映射: {path}")
        return cls(
            feed_id=str(d.get("feed_id", Path(path).stem)),
            a_single=ASingleCurve.from_dict(d.get("a_single")),
            v_sink_max_px_s=d.get("v_sink_max_px_s"),
            v_sink_max_mm_s=d.get("v_sink_max_mm_s"),
            association_radius_px=d.get("association_radius_px"),
            mean_pellet_area_px=d.get("mean_pellet_area_px"),
            px_per_mm=d.get("px_per_mm"),
            n_frames=int(d.get("n_frames", 0)),
            duration_s=float(d.get("duration_s", 0.0)),
            notes=list(d.get("notes", [])),
        )


def calibrate_video(
    video_path: str | Path,
    feed_id: str,
    px_per_mm: float | None = None,
    frame_step: int = 1,
    params: SegmentationParams | None = None,
    progress=None,
) -> PelletDynamicsCalibration:
    """标定入口：无鱼纯饲料视频 → A_single(t) / v_sink / 关联半径。

    Args:
        video_path: 无鱼纯饲料视频（≥30fps、覆盖完整漂散过程）。
        feed_id: 饲料批次标识（输出文件名）。
        px_per_mm: 像素/毫米尺度（有标定时提供；None → v_sink_max_mm_s
            为 None，但 px 版与 A_single 仍产出）。
        frame_step: 帧抽样步长（标定测量无需每帧；默认 1 = 全帧）。
    """
    notes: list[str] = []
    frames: list[np.ndarray] = []
    ts: list[float] = []
    with VideoReader(video_path) as vr:
        for idx, t_msec, frame in vr:
            if idx % max(1, frame_step) == 0:
                frames.append(frame)
                ts.append(t_msec / 1000.0)
    if len(frames) < 10:
        raise ValueError(
            f"标定视频有效帧不足（{len(frames)} < 10）：无法标定 A_single"
        )
    duration_s = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    a_single = measure_a_single(frames, ts, params)
    if not a_single.available:
        raise ValueError(
            "A_single 标定失败：全程未测到任何孤立颗粒面积（分割阈值请复核；"
            "不硬编码猜测值）"
        )
    v_px = measure_v_sink_px_s(frames, ts, params)

    # 平均等效直径 → 关联半径 = 2.5 × 等效直径
    all_areas = [
        b.area_px
        for f in frames
        for b in extract_blobs(segment_pellets(f, params or SegmentationParams()),
                               (params or SegmentationParams()).min_blob_area_px)
    ]
    mean_area = float(np.mean(all_areas)) if all_areas else None
    if mean_area:
        eq_diam = 2.0 * float(np.sqrt(mean_area / np.pi))
        radius = 2.5 * eq_diam
    else:
        radius = None
        notes.append("无法测得平均颗粒面积：association_radius 未标定")

    v_mm: float | None = None
    if v_px is None:
        notes.append(
            "位移样本 < 5：v_sink_max 未标定（A14 关联物理约束缺失，"
            "消失分类将退化为 unknown，不得猜测）"
        )
    elif px_per_mm is None:
        notes.append(
            "未提供 px_per_mm：v_sink_max_mm_s 为 None（仅 px 口径）；"
            "提供尺度后可换算"
        )
    else:
        v_mm = v_px / px_per_mm

    return PelletDynamicsCalibration(
        feed_id=feed_id,
        a_single=a_single,
        v_sink_max_px_s=v_px,
        v_sink_max_mm_s=v_mm,
        association_radius_px=radius,
        mean_pellet_area_px=mean_area,
        px_per_mm=px_per_mm,
        n_frames=len(frames),
        duration_s=duration_s,
        notes=notes,
    )

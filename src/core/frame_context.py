"""帧级观测契约层（T01）：pipeline → metrics 的标准输入结构。

职责（docs/04 §1 + docs/06 §3.1 classDiagram 合并）：
    - FrameContext / FrameObservation：帧级时间与上下文（t 相对 t0，基线期为负）；
    - PelletDetections / FishDetections：颗粒与鱼体检测输出；
    - Tracks：跨帧累积轨迹；
    - RunMeta：用户元数据（缺项 = None，绝不用 0 填充）；
    - BaselineStats：基线期统计（基线缺失则整个对象为 None）。

纪律：
    - 字段缺失一律 None，不得用 0 填充（0 是有意义的数值）；
    - 指标层只消费本模块契约，不 import 检测器（可脱离模型单测）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "FrameContext",
    "FrameObservation",
    "PelletDetections",
    "FishDetections",
    "Tracks",
    "RunMeta",
    "BaselineStats",
]


# ----------------------------------------------------------------------
# 帧级上下文
# ----------------------------------------------------------------------
@dataclass
class FrameContext:
    """单帧上下文（docs/04 §1）。

    Attributes:
        t: 秒，相对投喂起点 t0；t < 0 表示基线期（投喂前）。
        frame_idx: 源视频帧号。
        width / height: 帧尺寸（像素）。
        fps_source: 源视频标称帧率。
        px_per_mm: 标定尺度；未标定为 None（绝不为 0）。
        in_baseline: 是否属于基线期。
    """

    t: float
    frame_idx: int
    width: int
    height: int
    fps_source: float
    px_per_mm: float | None
    in_baseline: bool


@dataclass
class FrameObservation:
    """采样帧的完整观测（metrics 层的实际消费单元）。

    Attributes:
        frame_idx: 源视频帧号。
        t_s: 相对 t0 的时间戳（秒，基线期为负）。
        dt_s: 距上一个采样点的真实间隔（秒）；首个采样点为 None。
            没有它用户/下游积分会默认等间隔而算错——最便宜的防呆。
        image: 解码帧（BGR np.ndarray）；测试/缓存回放场景可为 None。
        pellets: 该帧的颗粒检测结果；可为 None（无检测数据）。
        extra: 附加通道（如 fish 检测、光流幅值等）。
    """

    frame_idx: int
    t_s: float
    dt_s: float | None
    image: np.ndarray | None
    pellets: "PelletDetections | None" = None
    extra: dict[str, Any] = field(default_factory=dict)

    def in_baseline(self, t0: float = 0.0) -> bool:
        """是否属于基线期（t < t0）。"""
        return self.t_s < t0


# ----------------------------------------------------------------------
# 检测结构
# ----------------------------------------------------------------------
@dataclass
class PelletDetections:
    """颗粒检测结果（classDiagram 主定义 + docs/04 §1 消费字段合并）。

    Attributes:
        xyxy: (N, 4) float，左上/右下角点（像素）。
        conf: (N,) float，检测置信度。
        centroid: (N, 2) float 或 None，质心（像素）；
            未显式给出时由 xyxy 中心派生（惰性，经 centroids()）。
        area_px: (N,) float 或 None，单颗面积（像素²）。
        track_id: (N,) int 或 None，轻量关联 ID（A14 消费）。
        vanish_class: 逐轨迹消失三分类标签列表（eaten/drifted/unknown），
            由 pellet_linker 填充；未关联时为 None。
    """

    xyxy: np.ndarray
    conf: np.ndarray
    centroid: np.ndarray | None = None
    area_px: np.ndarray | None = None
    track_id: np.ndarray | None = None
    vanish_class: list[str] | None = None

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.asarray(self.conf, dtype=float).reshape(-1)
        if self.xyxy.shape[0] != self.conf.shape[0]:
            raise ValueError(
                f"xyxy 行数 ({self.xyxy.shape[0]}) 与 conf 长度 ({self.conf.shape[0]}) 不一致"
            )
        if self.centroid is not None:
            self.centroid = np.asarray(self.centroid, dtype=float).reshape(-1, 2)
            if self.centroid.shape[0] != self.xyxy.shape[0]:
                raise ValueError("centroid 行数与 xyxy 不一致")
        if self.area_px is not None:
            self.area_px = np.asarray(self.area_px, dtype=float).reshape(-1)
            if self.area_px.shape[0] != self.xyxy.shape[0]:
                raise ValueError("area_px 长度与 xyxy 不一致")
        if self.track_id is not None:
            self.track_id = np.asarray(self.track_id).reshape(-1)
            if self.track_id.shape[0] != self.xyxy.shape[0]:
                raise ValueError("track_id 长度与 xyxy 不一致")

    def n_det(self) -> int:
        """检测数量。"""
        return int(self.xyxy.shape[0])

    def centroids(self) -> np.ndarray:
        """质心数组 (N, 2)：显式给出则返回，否则由框中心派生。"""
        if self.centroid is not None:
            return self.centroid
        return np.column_stack(
            ((self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0,
             (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0)
        )

    @classmethod
    def empty(cls) -> "PelletDetections":
        """空检测结果（N=0）。"""
        return cls(xyxy=np.zeros((0, 4)), conf=np.zeros((0,)))


@dataclass
class FishDetections:
    """鱼体检测结果（docs/04 §1；B2/C 组探索性消费）。"""

    bbox: np.ndarray                      # (M, 4)
    conf: np.ndarray                      # (M,)
    mask: np.ndarray | None = None        # (M, H, W) 可选分割
    track_id: np.ndarray | None = None
    heading: np.ndarray | None = None     # (M,) 弧度，可选

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=float).reshape(-1, 4)
        self.conf = np.asarray(self.conf, dtype=float).reshape(-1)
        if self.bbox.shape[0] != self.conf.shape[0]:
            raise ValueError("bbox 行数与 conf 长度不一致")

    def n_det(self) -> int:
        return int(self.bbox.shape[0])

    def centroids(self) -> np.ndarray:
        return np.column_stack(
            ((self.bbox[:, 0] + self.bbox[:, 2]) / 2.0,
             (self.bbox[:, 1] + self.bbox[:, 3]) / 2.0)
        )


@dataclass
class Tracks:
    """跨帧累积轨迹（docs/04 §1；C 组与 A14 消费）。"""

    track_id: list[int]
    t: list[np.ndarray]           # 每条轨迹的时间戳数组
    xy: list[np.ndarray]          # 每条轨迹的质心数组 (k, 2)
    n_frames_alive: list[int]
    heading: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        n = len(self.track_id)
        if len(self.t) != n or len(self.xy) != n or len(self.n_frames_alive) != n:
            raise ValueError("Tracks 各字段长度必须一致")
        if self.heading is not None and len(self.heading) != n:
            raise ValueError("heading 长度与 track_id 不一致")
        for tid, t_arr, xy_arr, alive in zip(
            self.track_id, self.t, self.xy, self.n_frames_alive
        ):
            if t_arr.shape[0] != xy_arr.shape[0]:
                raise ValueError(f"轨迹 {tid} 的时间戳与质心长度不一致")
            if alive <= 0:
                raise ValueError(f"轨迹 {tid} 的存活帧数必须为正")


# ----------------------------------------------------------------------
# 元数据与基线
# ----------------------------------------------------------------------
@dataclass
class RunMeta:
    """用户元数据（docs/04 §1 + classDiagram 合并；缺项 = None）。

    pond_id 用于伪重复检查（跨 run 统计前必查重复结构，docs/06 §7.11）；
    group_label_encrypted / blind_code 服务盲法（T05）。
    """

    species: str | None = None
    n_fish_total: int | None = None
    body_length_mm: float | None = None
    body_weight_g: float | None = None
    feed_mass_g: float | None = None          # 投喂量（g）
    pellet_mass_mg: float | None = None       # 单颗均重（mg）
    pellet_type: str | None = None            # 'floating'|'sinking'|'slow-sinking'
    water_temp_c: float | None = None
    pond_id: str | None = None
    group_label_encrypted: str | None = None
    blind_code: str | None = None

    _VALID_PELLET_TYPES = ("floating", "sinking", "slow-sinking")

    def cross_validate(self) -> list[str]:
        """基础交叉校验，返回告警字符串列表（空 = 通过）。

        完整八项交叉校验（含 n_fish_total 单向告警等实测反查）在
        pipeline/meta_validation.py（T02）；此处仅做无输入数据即可判定
        的静态一致性检查。
        """
        warnings: list[str] = []
        if self.pellet_type is not None and self.pellet_type not in self._VALID_PELLET_TYPES:
            warnings.append(
                f"pellet_type 非法: {self.pellet_type!r}（应为 "
                f"{'/'.join(self._VALID_PELLET_TYPES)} 或 None）"
            )
        if self.n_fish_total is not None and self.n_fish_total <= 0:
            warnings.append(f"n_fish_total 非正: {self.n_fish_total}")
        if self.body_length_mm is not None and self.body_length_mm <= 0:
            warnings.append(f"body_length_mm 非正: {self.body_length_mm}")
        if self.feed_mass_g is not None and self.feed_mass_g <= 0:
            warnings.append(f"feed_mass_g 非正: {self.feed_mass_g}")
        if self.pellet_mass_mg is not None and self.pellet_mass_mg <= 0:
            warnings.append(f"pellet_mass_mg 非正: {self.pellet_mass_mg}")
        if (
            self.feed_mass_g is not None
            and self.pellet_mass_mg is not None
            and self.pellet_mass_mg > 0
        ):
            # N0_meta = feed_mass_g × 1000 / pellet_mass_mg
            n0_meta = self.feed_mass_g * 1000.0 / self.pellet_mass_mg
            if n0_meta < 1.0:
                warnings.append(
                    f"N0_meta = {n0_meta:.2f} 颗 < 1：feed_mass_g 与 pellet_mass_mg 组合不合理"
                )
        return warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "n_fish_total": self.n_fish_total,
            "body_length_mm": self.body_length_mm,
            "body_weight_g": self.body_weight_g,
            "feed_mass_g": self.feed_mass_g,
            "pellet_mass_mg": self.pellet_mass_mg,
            "pellet_type": self.pellet_type,
            "water_temp_c": self.water_temp_c,
            "pond_id": self.pond_id,
            "group_label_encrypted": self.group_label_encrypted,
            "blind_code": self.blind_code,
        }


@dataclass
class BaselineStats:
    """基线期（t < 0）统计（docs/04 §1）；基线缺失则整个对象为 None。

    pellet_count_mean 应恒为 0，用于校验投喂起点 t0 的正确性。
    """

    n_frames: int
    duration_s: float
    fish_count_mean: float
    fish_count_std: float
    n_fz_mean: float
    n_fz_std: float
    activity_mean: float
    activity_std: float
    flow_mean: float
    flow_std: float
    annd_mean: float
    annd_std: float
    pellet_count_mean: float

    def __post_init__(self) -> None:
        if self.n_frames <= 0:
            raise ValueError("BaselineStats.n_frames 必须为正")
        if self.duration_s <= 0:
            raise ValueError("BaselineStats.duration_s 必须为正")
        if self.pellet_count_mean != 0:
            raise ValueError(
                "基线期 pellet_count_mean 应恒为 0（非零意味着 t0 判定可能有误）"
            )

    @property
    def ok(self) -> bool:
        """Q_baseline：BaselineStats 存在且 duration_s ≥ 30。"""
        return self.duration_s >= 30.0

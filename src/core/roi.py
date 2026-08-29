"""ROI 多边形契约（T01）。

职责：
    投喂区 / 参考区 / 颗粒计数区 / 排除区 / 底部带 / 有效水域的多边形容器，
    提供点在多边形内判定与掩膜生成（docs/04 §1 ROI + classDiagram ROI 合并）。

设计要点：
    - Polygon = (N, 2) float ndarray，顶点顺序任意（内判定对顺/逆时针均成立）；
    - 点在多边形内用射线法纯 numpy 实现（core 层不强制依赖 cv2，
      保证契约层可脱离 OpenCV 单测）；
    - 掩膜生成优先用 cv2.fillPoly（快），cv2 不可用时回退到逐点判定（慢，
      仅测试场景可接受）；
    - 顶点可序列化为普通列表（run_config.yaml 留档 + compare.py ROI 一致性校验）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Polygon", "point_in_polygon", "polygon_area", "ROI"]

Polygon = np.ndarray  # (N, 2) float，约定类型别名


def _as_polygon(poly: Polygon | list) -> np.ndarray:
    arr = np.asarray(poly, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError(f"多边形必须是 (N≥3, 2) 数组，收到 shape={arr.shape}")
    return arr


def point_in_polygon(point: tuple[float, float] | np.ndarray, polygon: Polygon) -> bool:
    """射线法点在多边形内判定（边界上的点视为在内，容差 1e-9）。

    对凸/凹多边形、顺/逆时针顶点顺序均成立。
    """
    poly = _as_polygon(polygon)
    px, py = float(point[0]), float(point[1])
    x, y = poly[:, 0], poly[:, 1]
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        # 边界判定：点恰在边 i-j 上
        cross = (xj - xi) * (py - yi) - (yj - yi) * (px - xi)
        if (
            abs(cross) < 1e-9
            and min(xi, xj) - 1e-9 <= px <= max(xi, xj) + 1e-9
            and min(yi, yj) - 1e-9 <= py <= max(yi, yj) + 1e-9
        ):
            return True
        if (yi > py) != (yj > py):
            x_int = xi + (py - yi) * (xj - xi) / (yj - yi)
            if px < x_int:
                inside = not inside
        j = i
    return inside


def polygon_area(polygon: Polygon) -> float:
    """鞋带公式多边形面积（绝对值，像素²）。"""
    poly = _as_polygon(polygon)
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


@dataclass
class ROI:
    """区域集合（多边形坐标均为像素）。

    Attributes:
        arena: 有效水域（必填）。
        feeding_zone: 投喂区（RP / 停留时长依赖）。
        reference_zone: 参考区（B1 波浪扣除 α 回归用）。
        pellet_zone: 颗粒计数区（通常 = 投喂框内）。
        exclude_zones: 排除区列表（反光 / 遮挡 / 池外 / 设备）。
        bottom_band: 底部带（判定颗粒沉降离开）。
        feed_center: 投喂点 R（像素）。
    """

    arena: Polygon
    feeding_zone: Polygon | None = None
    reference_zone: Polygon | None = None
    pellet_zone: Polygon | None = None
    exclude_zones: list[Polygon] = field(default_factory=list)
    bottom_band: Polygon | None = None
    feed_center: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.arena = _as_polygon(self.arena)
        if self.feeding_zone is not None:
            self.feeding_zone = _as_polygon(self.feeding_zone)
        if self.reference_zone is not None:
            self.reference_zone = _as_polygon(self.reference_zone)
        if self.pellet_zone is not None:
            self.pellet_zone = _as_polygon(self.pellet_zone)
        if self.bottom_band is not None:
            self.bottom_band = _as_polygon(self.bottom_band)
        self.exclude_zones = [_as_polygon(z) for z in self.exclude_zones]

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------
    def contains(
        self, point: tuple[float, float] | np.ndarray, zone: str = "arena"
    ) -> bool:
        """点是否落在指定区域内。

        Args:
            point: (x, y) 像素坐标。
            zone: 'arena' | 'feeding_zone' | 'reference_zone' | 'pellet_zone' |
                  'bottom_band' | 'exclude'（任一排除区命中即 True）。
        """
        if zone == "exclude":
            return any(point_in_polygon(point, z) for z in self.exclude_zones)
        poly = getattr(self, zone, None)
        if poly is None:
            raise ValueError(f"区域 {zone!r} 未定义（None）")
        return point_in_polygon(point, poly)

    def in_arena_not_excluded(self, point: tuple[float, float]) -> bool:
        """点在有效水域且不在任何排除区（掩膜逻辑的逐点版）。"""
        return point_in_polygon(point, self.arena) and not self.contains(point, "exclude")

    # ------------------------------------------------------------------
    # 掩膜
    # ------------------------------------------------------------------
    def mask(
        self,
        shape: tuple[int, int],
        zone: str = "arena",
        subtract_exclusion: bool = True,
    ) -> np.ndarray:
        """生成布尔掩膜 (H, W)，True = 区域内。

        Args:
            shape: (height, width)。
            zone: 生成哪个区域的掩膜（'arena' 等；'exclude' 生成排除区并集）。
            subtract_exclusion: 仅对非 exclude 区域生效，从结果中扣除排除区。
        """
        h, w = int(shape[0]), int(shape[1])
        try:  # 优先 cv2（快）
            import cv2  # type: ignore

            def _fill(poly: np.ndarray) -> np.ndarray:
                m = np.zeros((h, w), dtype=np.uint8)
                pts = np.round(np.asarray(poly, dtype=float)).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
                return m.astype(bool)

        except Exception:  # 回退：逐行射线判定（慢，仅测试可接受）

            def _fill(poly: np.ndarray) -> np.ndarray:
                yy, xx = np.mgrid[0:h, 0:w]
                m = np.zeros((h, w), dtype=bool)
                for row in range(h):
                    for col in range(w):
                        if point_in_polygon((col, row), poly):
                            m[row, col] = True
                return m

        if zone == "exclude":
            out = np.zeros((h, w), dtype=bool)
            for z in self.exclude_zones:
                out |= _fill(z)
            return out

        poly = getattr(self, zone, None)
        if poly is None:
            raise ValueError(f"区域 {zone!r} 未定义（None）")
        out = _fill(poly)
        if subtract_exclusion and self.exclude_zones:
            out &= ~self.mask(shape, zone="exclude", subtract_exclusion=False)
        return out

    # ------------------------------------------------------------------
    # 序列化（run_config.yaml 顶点留档 + compare.py 一致性校验）
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """顶点展平为普通列表（YAML 兼容）。"""

        def _poly(p: np.ndarray | None) -> list[list[float]] | None:
            return None if p is None else [[float(v[0]), float(v[1])] for v in p]

        return {
            "arena": _poly(self.arena),
            "feeding_zone": _poly(self.feeding_zone),
            "reference_zone": _poly(self.reference_zone),
            "pellet_zone": _poly(self.pellet_zone),
            "exclude_zones": [_poly(z) for z in self.exclude_zones],
            "bottom_band": _poly(self.bottom_band),
            "feed_center": None
            if self.feed_center is None
            else [float(self.feed_center[0]), float(self.feed_center[1])],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ROI":
        feed_center = d.get("feed_center")
        return cls(
            arena=np.asarray(d["arena"], dtype=float),
            feeding_zone=None if d.get("feeding_zone") is None else np.asarray(d["feeding_zone"], dtype=float),
            reference_zone=None if d.get("reference_zone") is None else np.asarray(d["reference_zone"], dtype=float),
            pellet_zone=None if d.get("pellet_zone") is None else np.asarray(d["pellet_zone"], dtype=float),
            exclude_zones=[np.asarray(z, dtype=float) for z in d.get("exclude_zones") or []],
            bottom_band=None if d.get("bottom_band") is None else np.asarray(d["bottom_band"], dtype=float),
            feed_center=None if feed_center is None else (float(feed_center[0]), float(feed_center[1])),
        )

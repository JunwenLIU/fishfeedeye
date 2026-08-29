"""合成视频 / 合成帧工厂（T02/T03 测试 fixture）。

设计（docs/06 §8：fixtures 先合成、真实数据到位后补回归测试）：
    - 水面背景：暗色 BGR 基底 + 低幅噪声 + 缓变"波浪"正弦纹理
      （HSV 明度 ~80，低于分割阈值 160，不会被误检为前景）；
    - 浮性饲料颗粒：亮黄色圆 (BGR 60,200,230)（V=230、S≈188，
      落在 SegmentationParams 阈值带内）；
    - PelletSpec：起点 / 半径 / 速度 / 消失时刻，覆盖
      静止（标定）、漂移（drifted）、区内消失（eaten）、排除区消失
      （unknown）四类场景；
    - make_video：MJPG/AVI 真实编码（CFR，POS_MSEC 可靠）。

所有工厂函数纯确定性（固定种子），不依赖网络与真实权重。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

__all__ = ["PelletSpec", "render_water_frame", "make_video", "WATER_BASE_BGR", "PELLET_BGR"]

WATER_BASE_BGR: tuple[int, int, int] = (60, 70, 80)   # 暗色水面（V≈80 < 160）
PELLET_BGR: tuple[int, int, int] = (60, 200, 230)    # 亮黄浮性料（V=230, S≈188）


@dataclass
class PelletSpec:
    """单颗合成颗粒的行为脚本。"""

    start_xy: tuple[float, float]
    radius: float = 6.0
    velocity_px_s: tuple[float, float] = (0.0, 0.0)
    vanish_at_s: float | None = None       # None = 全程存活
    color_bgr: tuple[int, int, int] = PELLET_BGR
    appear_at_s: float = 0.0               # 出现时刻（默认 0）


def render_water_frame(
    size: tuple[int, int],
    t_s: float,
    pellets: Sequence[PelletSpec],
    seed: int = 0,
    wave_amplitude: int = 10,
) -> np.ndarray:
    """渲染一帧合成水面（含颗粒）。

    Args:
        size: (width, height)。
        t_s: 帧时刻（秒），决定颗粒位置与波浪相位。
        pellets: 颗粒行为脚本列表。
        seed: 噪声种子（帧内确定性）。
        wave_amplitude: 波浪纹理幅度（低幅，不会误检为前景）。
    """
    w, h = int(size[0]), int(size[1])
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.array(WATER_BASE_BGR, dtype=np.float32)
    # 缓变波浪（相位随时间漂移，制造真实帧差但保持暗色）
    wave = wave_amplitude * np.sin(xx / 18.0 + t_s * 1.3) * np.cos(yy / 24.0 - t_s * 0.7)
    frame = base[None, None, :] + wave[:, :, None]
    # 噪声种子：必须支持**负时刻**（基线期 t < 0 是合法且常见的输入），
    # 故做非负映射（seed 与时刻双因子，避免 t 与 -t 共种子）。
    rng = np.random.default_rng(
        int(seed) * 1_000_003 + int(round(t_s * 1000.0)) + 1_000_000
    )
    frame += rng.integers(-8, 9, (h, w, 1))  # 帧内噪声（去相关由种子 t 变化）
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    for p in pellets:
        if t_s < p.appear_at_s - 1e-9:
            continue
        if p.vanish_at_s is not None and t_s >= p.vanish_at_s - 1e-9:
            continue
        x = p.start_xy[0] + p.velocity_px_s[0] * (t_s - p.appear_at_s)
        y = p.start_xy[1] + p.velocity_px_s[1] * (t_s - p.appear_at_s)
        if -p.radius <= x < w + p.radius and -p.radius <= y < h + p.radius:
            cv2.circle(frame, (int(round(x)), int(round(y))), int(round(p.radius)),
                       p.color_bgr, -1)
    return frame


def make_video(
    path: str | Path,
    fps: float = 10.0,
    duration_s: float = 20.0,
    pellets: Sequence[PelletSpec] | None = None,
    size: tuple[int, int] = (160, 120),
    seed: int = 0,
) -> Path:
    """写一段 CFR 合成视频（MJPG/AVI）。返回文件路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(fps * duration_s))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (int(size[0]), int(size[1]))
    )
    if not writer.isOpened():
        raise IOError(f"VideoWriter 打开失败: {path}")
    try:
        for i in range(n):
            t = i / fps
            writer.write(render_water_frame(size, t, pellets or [], seed=seed))
    finally:
        writer.release()
    return path


# ----------------------------------------------------------------------
# 预置场景工厂
# ----------------------------------------------------------------------
def calibration_pellets(n: int = 12, radius: float = 6.0, drift_px_s: float = 4.0,
                        size: tuple[int, int] = (320, 240)) -> list[PelletSpec]:
    """标定视频颗粒：网格均布、缓慢漂移（无鱼纯饲料场景）。"""
    pellets: list[PelletSpec] = []
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    x0 = size[0] * 0.15
    y0 = size[1] * 0.15
    dx = size[0] * 0.7 / max(1, cols - 1)
    dy = size[1] * 0.7 / max(1, rows - 1)
    for i in range(n):
        r, c = divmod(i, cols)
        vx = drift_px_s if i % 2 == 0 else -drift_px_s
        vy = drift_px_s * 0.5 if i % 3 == 0 else 0.0
        pellets.append(
            PelletSpec(
                start_xy=(x0 + c * dx, y0 + r * dy),
                radius=radius,
                velocity_px_s=(vx, vy),
            )
        )
    return pellets

"""OpenCV 视频解码封装 + 元信息（T01）。

职责：
    统一的读取入口，解封装元信息（fps/帧数/尺寸/时长），逐帧顺序解码
    并收集 CAP_PROP_POS_MSEC（timing.py 的唯一可信时间源）。

注意：
    - CAP_PROP_FRAME_COUNT 与 CAP_PROP_FPS 均为容器标称值（VFR 视频不
      可信，仅作交叉校验与兜底，时间轴构建见 pipeline/timing.py）；
    - 随机 seek（CAP_PROP_POS_FRAMES）在部分编码器上不可靠，read_frame
      仅供抽帧预览，不用于时间轴构建。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

__all__ = ["VideoInfo", "probe_video", "collect_frame_timestamps", "VideoReader"]


@dataclass
class VideoInfo:
    """视频容器元信息（全部为标称值，VFR 下不保证精确）。"""

    path: str
    width: int
    height: int
    fps: float           # 标称帧率
    n_frames: int        # 标称帧数
    duration_s: float    # 标称时长 = n_frames / fps
    fourcc: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "duration_s": self.duration_s,
            "fourcc": self.fourcc,
        }


def _open(video_path: str | Path) -> cv2.VideoCapture:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"视频无法打开（编码器不支持或文件损坏）: {path}")
    return cap


def probe_video(video_path: str | Path) -> VideoInfo:
    """读取视频容器元信息（不解码帧）。"""
    cap = _open(video_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = (
            "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
            if fourcc_int > 0
            else ""
        )
        if fps <= 0:
            raise IOError(f"容器标称帧率非法（{fps}）: {video_path}")
        duration_s = n_frames / fps if n_frames > 0 else 0.0
        return VideoInfo(
            path=str(video_path),
            width=width,
            height=height,
            fps=fps,
            n_frames=n_frames,
            duration_s=duration_s,
            fourcc=fourcc,
        )
    finally:
        cap.release()


def collect_frame_timestamps(
    video_path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[float], int]:
    """顺序解码全程，逐帧收集 CAP_PROP_POS_MSEC（毫秒）。

    Returns:
        (pos_msec 列表, 实际解出帧数)。实际帧数与容器标称帧数不一致
        本身就是 VFR/拼接的信号，交给 timing.build_timeline 判定。

    注意：必须顺序解码（POS_MSEC 只在顺序读时可靠）。
    """
    cap = _open(video_path)
    pos_msec: list[float] = []
    try:
        total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            pos_msec.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
            if progress is not None and total_hint > 0 and len(pos_msec) % 500 == 0:
                progress(len(pos_msec), total_hint)
    finally:
        cap.release()
    return pos_msec, len(pos_msec)


class VideoReader:
    """上下文管理器式逐帧读取器（顺序解码）。

    用法::

        with VideoReader(path) as vr:
            for frame_idx, t_msec, frame in vr:
                ...
    """

    def __init__(self, video_path: str | Path) -> None:
        self.path = Path(video_path)
        self.cap = _open(video_path)
        self.info = VideoInfo(
            path=str(video_path),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self.cap.get(cv2.CAP_PROP_FPS)),
            n_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_s=0.0,
            fourcc="",
        )
        if self.info.fps > 0 and self.info.n_frames > 0:
            self.info.duration_s = self.info.n_frames / self.info.fps
        self._frame_idx = 0

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None  # type: ignore[assignment]

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        return self

    def __next__(self) -> tuple[int, float, np.ndarray]:
        if self.cap is None:
            raise StopIteration
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        t_msec = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        idx = self._frame_idx
        self._frame_idx += 1
        return idx, t_msec, frame

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        """按帧号 seek 读取单帧（仅供抽帧预览；seek 精度因编码器而异，
        不用于时间轴构建）。"""
        if self.cap is None:
            raise RuntimeError("VideoReader 已关闭")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = self.cap.read()
        return frame if ok else None

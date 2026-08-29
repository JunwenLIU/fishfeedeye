"""HomographyCalibrator · 4 点单应性标定 + 尺度场（T01）。

职责（docs/06 T01 内联约定）：
    输入画面内 4 点（投喂框四角，像素）+ 世界坐标（框实际尺寸，米），
    输出单应矩阵 H（像素 → 世界米）与 px_per_mm(x, y) 尺度场。

    斜拍为确定机位 → 透视校正是 P0 硬需求；尺度随画面位置变化
    （px_per_mm 不是常数），点标注生成框时按位置取半径。

能力边界（须向用户声明）：单应性只修尺度、不修遮挡。

实现要点：
    - fit(): cv2.findHomography(帧内4点, 世界4点)，4 点精确解（method=0），
      带重投影误差校验（阈值可配，默认 5 mm）；
    - px_per_mm(x, y): 对 H 做数值微分取局部雅可比 → 该点处
      x / y 两个方向的像素-毫米尺度，返回几何平均
      （方向差异本身是透视信号，由 scale_anisotropy() 暴露）；
    - warp(): 用 H 把整帧投影到世界坐标平面（俯视平面）。
"""
from __future__ import annotations

import numpy as np

__all__ = ["HomographyCalibrator"]


class HomographyCalibrator:
    """4 点单应性标定器（像素 → 世界米）。"""

    def __init__(self, reproj_tol_m: float = 0.005) -> None:
        """
        Args:
            reproj_tol_m: fit 时重投影误差容限（米），默认 5 mm。
        """
        self.H: np.ndarray | None = None        # 3×3，像素 → 世界米
        self.frame_corners: np.ndarray | None = None  # (4, 2) 像素
        self.world_corners_m: np.ndarray | None = None  # (4, 2) 米
        self.reproj_err_m: float | None = None
        self._reproj_tol_m = float(reproj_tol_m)

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------
    def fit(
        self,
        frame_corners: np.ndarray | list,
        world_corners_m: np.ndarray | list,
    ) -> bool:
        """由 4 组对应点求解单应矩阵。

        Args:
            frame_corners: (4, 2) 帧内像素坐标（投喂框四角，顶点顺序须与世界坐标一一对应）。
            world_corners_m: (4, 2) 世界坐标（米，框实际尺寸实测值）。

        Returns:
            True = 标定成功（H 与重投影误差已就位）；False = 失败（点数不足
            / 几何退化 / 重投影误差超容限），此时 H 保持 None，绝不给出
            半成品标定。
        """
        fp = np.asarray(frame_corners, dtype=float)
        wp = np.asarray(world_corners_m, dtype=float)
        if fp.shape != (4, 2) or wp.shape != (4, 2):
            self.H = None
            return False
        # 几何退化检查：4 点中任意 3 点不能共线（像素域与世界域都查）
        if _any_three_collinear(fp) or _any_three_collinear(wp):
            self.H = None
            return False

        import cv2  # 延迟导入：core 层 import 不触发 OpenCV

        H, _mask = cv2.findHomography(fp, wp, method=0)  # 4 点精确解
        if H is None:
            self.H = None
            return False

        # 重投影误差校验（世界坐标，米）
        proj = self._apply(H, fp)
        err = float(np.max(np.linalg.norm(proj - wp, axis=1)))
        if err > self._reproj_tol_m:
            self.H = None
            self.reproj_err_m = err
            return False

        self.H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H  # 归一化
        self.frame_corners = fp
        self.world_corners_m = wp
        self.reproj_err_m = err
        return True

    @property
    def fitted(self) -> bool:
        """是否已完成有效标定。"""
        return self.H is not None

    # ------------------------------------------------------------------
    # 尺度场
    # ------------------------------------------------------------------
    def _to_world(self, x: float, y: float) -> tuple[float, float]:
        """像素点经 H 映射到世界坐标（米）。须已标定。"""
        if self.H is None:
            raise RuntimeError("未标定：先调用 fit()")
        wx, wy = self._apply(self.H, np.array([[x, y]], dtype=float))[0]
        return float(wx), float(wy)

    def px_per_mm(self, x: float, y: float) -> float:
        """该像素点处的局部尺度（像素/毫米），x/y 两方向几何平均。

        原理：对 H 做数值微分（步长 1 px）得局部雅可比，求出
        1 像素在 x / y 方向对应的世界位移（米）→ 换算 px/mm。
        透视下该值随位置变化，即"尺度场"。
        """
        if self.H is None:
            raise RuntimeError("未标定：先调用 fit()")
        eps = 1.0  # 像素
        # x 方向：1 px 的世界位移
        w0 = np.array(self._to_world(x, y))
        wdx = np.array(self._to_world(x + eps, y))
        # y 方向
        wdy = np.array(self._to_world(x, y + eps))
        dx_m = float(np.linalg.norm(wdx - w0))   # 米 / 像素（x 方向）
        dy_m = float(np.linalg.norm(wdy - w0))   # 米 / 像素（y 方向）
        if dx_m <= 0 or dy_m <= 0:
            raise ValueError(f"点 ({x}, {y}) 处尺度退化（可能位于灭线附近）")
        px_per_mm_x = 1.0 / (dx_m * 1000.0)      # 1 mm = 1000×dx_m 像素
        px_per_mm_y = 1.0 / (dy_m * 1000.0)
        return float(np.sqrt(px_per_mm_x * px_per_mm_y))

    def scale_anisotropy(self, x: float, y: float) -> float:
        """该点 x / y 两方向尺度比（>1 表示透视畸变显著，通常 <1.2 可接受）。"""
        if self.H is None:
            raise RuntimeError("未标定：先调用 fit()")
        eps = 1.0
        w0 = np.array(self._to_world(x, y))
        dx_m = float(np.linalg.norm(np.array(self._to_world(x + eps, y)) - w0))
        dy_m = float(np.linalg.norm(np.array(self._to_world(x, y + eps)) - w0))
        if dx_m <= 0 or dy_m <= 0:
            raise ValueError(f"点 ({x}, {y}) 处尺度退化")
        return max(dx_m, dy_m) / min(dx_m, dy_m)

    def px_per_mm_ref(self) -> float:
        """参考尺度：世界四角中心点处的 px_per_mm（run_config.px_per_mm_ref 留档值）。"""
        if self.world_corners_m is None or self.frame_corners is None:
            raise RuntimeError("未标定：先调用 fit()")
        cx = float(self.frame_corners[:, 0].mean())
        cy = float(self.frame_corners[:, 1].mean())
        return self.px_per_mm(cx, cy)

    # ------------------------------------------------------------------
    # 投影
    # ------------------------------------------------------------------
    def warp(
        self,
        frame: np.ndarray,
        px_per_m: float = 1000.0,
        out_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """整帧透视校正到世界平面（俯视平面）。

        Args:
            frame: BGR 或灰度帧。
            px_per_m: 输出分辨率（像素/米），默认 1000（即 1 mm/px）。
            out_size: 显式指定 (w, h)；缺省由世界包围盒 × px_per_m 计算。

        Returns:
            校正后帧。注意：只修尺度，不修遮挡（能力边界，须向用户声明）。
        """
        if self.H is None:
            raise RuntimeError("未标定：先调用 fit()")
        import cv2  # 延迟导入

        if out_size is None:
            wmin, wmax = self.world_corners_m[:, 0].min(), self.world_corners_m[:, 0].max()
            hmin, hmax = self.world_corners_m[:, 1].min(), self.world_corners_m[:, 1].max()
            out_w = max(1, int(np.ceil((wmax - wmin) * px_per_m)))
            out_h = max(1, int(np.ceil((hmax - hmin) * px_per_m)))
            # 平移使世界最小角对齐输出原点
            T = np.array(
                [
                    [1.0, 0.0, -wmin * px_per_m],
                    [0.0, 1.0, -hmin * px_per_m],
                    [0.0, 0.0, 1.0],
                ]
            )
            H_use = T @ np.diag([px_per_m, px_per_m, 1.0]) @ self.H
        else:
            out_w, out_h = int(out_size[0]), int(out_size[1])
            H_use = self.H
        return cv2.warpPerspective(frame, H_use, (out_w, out_h))

    def to_dict(self) -> dict | None:
        """H 展平为 9 元素列表（run_config.yaml homography 字段留档）。"""
        if self.H is None:
            return None
        return [float(v) for v in self.H.reshape(-1)]

    @classmethod
    def from_dict(cls, d: list[float] | None) -> "HomographyCalibrator":
        """从 run_config.yaml 恢复（不含原始 4 点，仅 H；尺度场仍可用）。"""
        calib = cls()
        if d is not None:
            if len(d) != 9:
                raise ValueError("homography 必须为 3×3 展平的 9 元素列表")
            calib.H = np.asarray(d, dtype=float).reshape(3, 3)
        return calib

    # ------------------------------------------------------------------
    @staticmethod
    def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """齐次坐标批量应用 H（自动做透视除法）。"""
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        proj = pts_h @ H.T
        return proj[:, :2] / proj[:, 2:3]


def _any_three_collinear(pts: np.ndarray, tol: float = 1e-9) -> bool:
    """4 点中任意 3 点是否共线（几何退化检查）。"""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = pts[i], pts[j], pts[k]
                area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < tol:
                    return True
    return False

"""预处理测试（T02）：稳像 / CLAHE / 参考区噪声扣除。

覆盖：
    - Stabilizer：真实平移的相位相关估计与补偿、超限钳制（机位移动）
      与 note、mean_abs_shift_px（Q_motion 代理量）；
    - Preprocessor：CLAHE 作用标记、透视校正默认关闭、未标定时 warp
      跳过 + note；
    - ReferenceZoneCorrector：α 回归、样本不足 / 零变异 → None + note
      （绝不拍脑袋）、correct 点态扣除；
    - zone_frame_diff：区域平均绝对差。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.pipeline.preprocess import (
    Preprocessor,
    ReferenceZoneCorrector,
    Stabilizer,
    zone_frame_diff,
)


def _noise_canvas(seed: int = 7, size: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


# ----------------------------------------------------------------------
# 稳像
# ----------------------------------------------------------------------
class Test稳像器:

    def test_平移估计与补偿(self) -> None:
        canvas = _noise_canvas()
        ref = canvas[0:200, 0:200]
        # 内容平移 (+5, +3)：cur(x,y) = canvas(y+3, x+5)
        cur = canvas[3:203, 5:205]
        st = Stabilizer()
        out0, shift0 = st.process(ref)
        assert shift0 == (0.0, 0.0)  # 首帧 = 参考帧
        out1, shift1 = st.process(cur)
        # phaseCorrelate 返回对齐方向偏移：内容 (+5,+3) → 补偿 (-5,-3)
        assert shift1[0] == pytest.approx(-5.0, abs=0.5)
        assert shift1[1] == pytest.approx(-3.0, abs=0.5)
        # 补偿后与参考帧对齐（比较内部区域，边界有 REPLICATE 复制伪影）
        assert np.mean(np.abs(
            out1[20:-20, 20:-20].astype(float) - ref[20:-20, 20:-20].astype(float)
        )) < 2.0

    def test_超限钳制_判定机位移动(self) -> None:
        canvas = _noise_canvas(seed=11)
        ref = canvas[0:200, 0:200]
        cur = canvas[3:203, 5:205]  # 平移约 5.8px > 上限 2px
        st = Stabilizer(max_correction_px=2.0)
        st.process(ref)
        out, shift = st.process(cur)
        # 停止补偿：原帧返回、偏移清零、参考系切换、显式 note
        assert np.array_equal(out, cur)
        assert shift == (0.0, 0.0)
        assert any("机位移动" in n for n in st.notes)
        # 参考系已切换到当前帧：下一帧零偏移（相位相关数值噪声 ~1e-8）
        out2, shift2 = st.process(cur)
        assert shift2 == pytest.approx((0.0, 0.0), abs=1e-6)

    def test_mean_abs_shift_px_代理量(self) -> None:
        canvas = _noise_canvas(seed=13)
        st = Stabilizer()
        st.process(canvas[0:200, 0:200])
        st.process(canvas[0:200, 0:200])  # 静止 → 0（相位相关数值噪声 ~1e-8）
        assert st.mean_abs_shift_px == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# 预处理器
# ----------------------------------------------------------------------
class Test预处理器:

    def test_默认管线_稳像加CLAHE_不做透视校正(self) -> None:
        canvas = _noise_canvas(seed=21)
        pre = Preprocessor()  # 默认 apply_warp=False
        res = pre.process(canvas[0:100, 0:100])
        assert res.warped is False
        assert res.clahe_applied is True
        assert res.shift_px == (0.0, 0.0)

    def test_请求透视校正但未标定_跳过并note(self) -> None:
        canvas = _noise_canvas(seed=23)
        pre = Preprocessor(apply_warp=True, calibrator=None)
        res = pre.process(canvas[0:100, 0:100])
        assert res.warped is False
        assert any("单应性未标定" in n for n in pre.notes)

    def test_关闭全部增强_恒等(self) -> None:
        canvas = _noise_canvas(seed=25)[0:100, 0:100]
        pre = Preprocessor(stabilize=False, apply_clahe=False)
        res = pre.process(canvas)
        assert res.warped is False
        assert res.clahe_applied is False
        assert np.array_equal(res.frame, canvas)

    def test_mean_abs_shift_px_透出(self) -> None:
        pre = Preprocessor(stabilize=True, apply_clahe=False)
        pre.process(_noise_canvas(seed=27)[0:100, 0:100])
        assert pre.mean_abs_shift_px >= 0.0


# ----------------------------------------------------------------------
# 参考区噪声扣除
# ----------------------------------------------------------------------
class Test参考区噪声扣除:

    def test_基线回归_估计α(self) -> None:
        rng = np.random.default_rng(31)
        ref = rng.normal(0.0, 5.0, 30)
        feed = 0.7 * ref + rng.normal(0.0, 0.5, 30)
        corr = ReferenceZoneCorrector()
        alpha = corr.fit(feed, ref)
        assert alpha is not None
        assert 0.5 < alpha < 0.9
        assert corr.alpha == alpha

    def test_点态扣除(self) -> None:
        corr = ReferenceZoneCorrector()
        corr.alpha = 0.5
        assert corr.correct(feed_signal=10.0, ref_signal=4.0) == pytest.approx(8.0)

    def test_α不可得时correct返回None(self) -> None:
        corr = ReferenceZoneCorrector()
        corr.alpha = None
        assert corr.correct(10.0, 4.0) is None

    def test_基线样本不足_None加note(self) -> None:
        corr = ReferenceZoneCorrector()
        alpha = corr.fit([1.0] * 5, [2.0] * 5)  # 5 < 8
        assert alpha is None
        assert any("no_noise_correction" in n for n in corr.fit_notes)

    def test_参考区零变异_None加note(self) -> None:
        corr = ReferenceZoneCorrector()
        alpha = corr.fit([1.0, 2.0, 3.0] * 5, [7.0] * 15)  # 无变异
        assert alpha is None
        assert any("零变异" in n for n in corr.fit_notes)

    def test_长度不一致_报错(self) -> None:
        corr = ReferenceZoneCorrector()
        with pytest.raises(ValueError, match="长度必须一致"):
            corr.fit([1.0, 2.0], [1.0, 2.0, 3.0])


class Test区域帧差:

    def test_同帧_零差(self) -> None:
        g = np.full((20, 20), 100, dtype=np.uint8)
        mask = np.ones((20, 20), dtype=bool)
        assert zone_frame_diff(g, g, mask) == 0.0

    def test_已知差异_均值口径(self) -> None:
        prev = np.zeros((10, 10), dtype=np.uint8)
        cur = np.full((10, 10), 20, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        mask[:, :5] = True  # 只取左半
        assert zone_frame_diff(prev, cur, mask) == pytest.approx(20.0)

    def test_空掩膜_零差(self) -> None:
        g = np.zeros((10, 10), dtype=np.uint8)
        assert zone_frame_diff(g, g + 5, np.zeros((10, 10), dtype=bool)) == 0.0

    def test_形状不一致_报错(self) -> None:
        g = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="形状不一致"):
            zone_frame_diff(g, np.zeros((12, 10), dtype=np.uint8), np.ones((10, 10), dtype=bool))

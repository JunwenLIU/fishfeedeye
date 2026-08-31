"""T04 指标层数值正确性测试（解析解验证 + 数值口径）。

覆盖（team-lead 派单验收标准）：
    - 指数衰减 N(t)=N0·e^(-kt) 的解析解：T50=ln2/k、T90=ln10/k、
      k/tau 拟合精确恢复、AUC60 闭式解、v̄50=0.5·N0/T50、v_max≈k·N0；
    - 线性消耗 N(t)=N0−r·t：v_max≈r、T50/T90 闭式解；
    - 右删失：T50 未达 → value=None + window_s=窗长（绝不为 0/NaN）；
    - N₀ 双轨：estimate_n0 早期窗口 / Q_n0gap 降级（denominator_suspect）；
    - 回弹（非单调上升段 >10%）→ counting_unstable；
    - A14 三分类 → Q_pelletloss 与 contains_non_feeding_loss；
    - quality 13 键 "None 也要有行" 纪律；
    - B2 投喂区计数 / RP / P_fz 数值与降级；
    - B1 空间异质性 / D 组起止（合成帧图像注入）。

注：aggregator 层（人工修正双轨 / run 目录入口）的闭环测试在
test_metrics_corrections.py（依赖 aggregator API，单独成文件）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.config import RunConfig, Thresholds
from src.core.frame_context import (
    BaselineStats,
    FishDetections,
    FrameObservation,
    PelletDetections,
    RunMeta,
)
from src.core.metric_value import MetricValue
from src.core.roi import ROI, polygon_area
from src.metrics.group_a.clearance import clearance_metric, crossing_time
from src.metrics.group_a.n0_estimator import estimate_n0, n0_gap
from src.metrics.group_a.nonfeeding_loss import (
    VanishTally,
    nonfeeding_loss_metric,
    tally_vanish,
)
from src.metrics.group_a.pellet_curve import (
    PelletSeries,
    TimeSeries,
    extract_pellet_series,
    rebound_fraction,
    smooth_series,
)
from src.metrics.group_a.rate import instantaneous_rate, rate_metrics
from src.metrics.group_a.residual import fit_exponential, residual_metrics, trapz_integral
from src.metrics.quality import Q_KEYS, QualitySignals

# tests.fixtures.synthetic 的导入方式保持既有惯例（tests/__init__.py 存在）
from tests.fixtures.synthetic import render_water_frame

# 指数衰减场景参数（解析解基准）
N0_EXP = 100.0
K_EXP = 0.02  # 1/s
T50_ANALYTIC = math.log(2.0) / K_EXP      # ≈ 34.657 s
T90_ANALYTIC = math.log(10.0) / K_EXP     # ≈ 115.129 s
T100_ANALYTIC = math.log(N0_EXP / 2.0) / K_EXP  # ε=2 颗 → ≈ 195.60 s
AUC60_ANALYTIC = 100.0 * 60.0 - 100.0 * (1.0 - math.exp(-1.2)) / K_EXP  # ≈ 2505.6

# ----------------------------------------------------------------------
# 整数计数下的**可分辨极限**（解析解对照的容差来源，勿当作"随便放宽"）
# ----------------------------------------------------------------------
# 颗粒数 N 是整数（数出来的，不可为小数），dt=2s 采样 ⇒ 穿越时刻的
# 误差界由计数量化步长 ±0.5 颗除以局部斜率给出：
#       Δt_bound = 0.5 / |dN/dt| = 0.5 / (k · N_threshold)
# 这是**不可逾越的信息论界限**（不是实现精度问题）：任何估计量都无法
# 在整数计数 + 2s 间隔下把穿越时刻定得比它更准。
#   T50 : N_thr = 0.5·N₀ = 50 颗 → 0.5/(0.02×50)  = 0.5 s
#   T90 : N_thr = 0.1·N₀ = 10 颗 → 0.5/(0.02×10)  = 2.5 s
#   T100: N_thr = ε     =   2 颗 → 0.5/(0.02×2)   = 12.5 s
# 故 T90/T100 的断言容差取该界限，而非拍脑袋的 0.5/1.0 s。
# ⚠️ 由此也得到一个业务结论（已记入遗留问题）：ε=2 颗时 T100 在
# N₀=100、k=0.02 的条件下**不可分辨区间长达 ±12.5 s**，A10_T100
# 不应作为"吃完时间"的主证据（契约亦要求其恒与 A14 并列展示）。
T50_TOL_S = 0.5
T90_TOL_S = 0.5 / (K_EXP * 0.1 * N0_EXP)
T100_TOL_S = 0.5 / (K_EXP * 2.0)
# RR 的量化步长 = 100/N₀ = 1 个百分点 → 半步长容差
RR_TOL_PCT = 0.5


# ----------------------------------------------------------------------
# 测试工具
# ----------------------------------------------------------------------
def make_pellets(n: int, conf: float = 0.9, seed: int = 0) -> PelletDetections:
    """构造 n 颗网格排布的合成检测（质心互不重叠）。"""
    if n <= 0:
        return PelletDetections.empty()
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    xyxy: list[list[float]] = []
    for i in range(n):
        r, c = divmod(i, cols)
        x = 20.0 + c * (280.0 / max(1, cols))
        y = 20.0 + r * (200.0 / max(1, rows))
        s = 4.0
        xyxy.append([x - s, y - s, x + s, y + s])
    return PelletDetections(
        xyxy=np.asarray(xyxy, dtype=float),
        conf=np.full(n, conf, dtype=float),
    )


def make_obs(
    t_s: float,
    n_pellets: int,
    frame_idx: int | None = None,
    image: np.ndarray | None = None,
    extra: dict | None = None,
    conf: float = 0.9,
) -> FrameObservation:
    """单帧观测工厂（image 缺省 None = 契约允许的缓存回放模式）。"""
    return FrameObservation(
        frame_idx=frame_idx if frame_idx is not None else int(round(t_s * 10)),
        t_s=float(t_s),
        dt_s=None,
        image=image,
        pellets=make_pellets(n_pellets, conf=conf),
        extra=dict(extra or {}),
    )


def exp_decay_observations(
    n0: float = N0_EXP, k: float = K_EXP, t_end: float = 300.0, dt: float = 2.0
) -> list[FrameObservation]:
    """理想指数衰减观测序列（基线 0 颗 + 试验期 N0·e^(-kt)）。"""
    obs: list[FrameObservation] = []
    for t in np.arange(-10.0, 0.0, 2.0):
        obs.append(make_obs(t, 0))
    t_arr = np.arange(0.0, t_end + 1e-9, dt)
    for i, t in enumerate(t_arr):
        n = int(round(n0 * math.exp(-k * float(t))))
        obs.append(make_obs(float(t), n, frame_idx=100 + i))
    return obs


def series_from(obs: list[FrameObservation], thresholds: Thresholds | None = None) -> tuple:
    """提取 + 平滑（返回 (raw, smoothed)）。"""
    th = thresholds or Thresholds()
    raw = extract_pellet_series(obs, thresholds=th)
    sm = smooth_series(raw, th.smooth_window)
    return raw, sm


def full_roi() -> ROI:
    """覆盖整帧的合成 ROI（arena + feeding_zone + reference_zone）。"""
    arena = np.array([[10, 10], [310, 10], [310, 230], [10, 230]], dtype=float)
    return ROI(
        arena=arena,
        feeding_zone=np.array([[10, 10], [160, 10], [160, 120], [10, 120]], dtype=float),
        reference_zone=np.array([[160, 120], [310, 120], [310, 230], [160, 230]], dtype=float),
        pellet_zone=arena,
    )


def n0_from(obs: list[FrameObservation], meta: RunMeta | None = None) -> float:
    th = Thresholds()
    raw = extract_pellet_series(obs, thresholds=th)
    return estimate_n0(raw, meta, th, video_duration_s=300.0).value


# ----------------------------------------------------------------------
# A1 · N₀ 双轨估计
# ----------------------------------------------------------------------
class TestN0Estimator:
    def test_early_window_peak_recovers_n0(self) -> None:
        obs = exp_decay_observations()
        assert n0_from(obs) == pytest.approx(N0_EXP, abs=1.0)

    def test_meta_track_and_gap_ok(self) -> None:
        meta = RunMeta(feed_mass_g=2.5, pellet_mass_mg=25.0)  # N0_meta = 100
        obs = exp_decay_observations()
        mv = estimate_n0(
            extract_pellet_series(obs), meta, Thresholds(), video_duration_s=300.0
        )
        assert mv.value is not None
        assert mv.quality["N0_meta"] == pytest.approx(100.0)
        assert mv.quality["cross_validated"] is True
        assert mv.status == "ok"
        assert n0_gap(mv.value, 100.0) == pytest.approx(0.0, abs=0.01)

    def test_gap_over_20pct_degrades_with_denominator_suspect(self) -> None:
        # N0_meta = 1g×1000/50mg = 20 颗，检测轨 100 颗 → gap = 4.0
        meta = RunMeta(feed_mass_g=1.0, pellet_mass_mg=50.0)
        obs = exp_decay_observations()
        mv = estimate_n0(
            extract_pellet_series(obs), meta, Thresholds(), video_duration_s=300.0
        )
        assert mv.status == "degraded"
        assert "denominator_suspect" in mv.flags
        assert mv.reason is not None and "20%" in mv.reason

    def test_missing_meta_no_gap_no_guess(self) -> None:
        obs = exp_decay_observations()
        mv = estimate_n0(extract_pellet_series(obs), None, Thresholds(), 300.0)
        assert mv.quality["N0_meta"] is None
        assert mv.quality["cross_validated"] is False
        assert n0_gap(100.0, None) is None  # 不猜测

    def test_too_few_early_frames_unavailable(self) -> None:
        obs = [make_obs(0.0, 100), make_obs(2.0, 100)]  # 早期窗口仅 2 帧
        mv = estimate_n0(extract_pellet_series(obs), None, Thresholds(), 300.0)
        assert mv.status == "unavailable"
        assert mv.value is None
        assert mv.reason is not None


# ----------------------------------------------------------------------
# A8–A10 · 清空时间（解析解）
# ----------------------------------------------------------------------
class TestClearanceAnalytic:
    def test_t50_matches_ln2_over_k(self) -> None:
        obs = exp_decay_observations()
        raw, sm = series_from(obs)
        n0 = n0_from(obs)
        t50 = crossing_time(sm, n0, 0.50, 300.0, Thresholds())
        assert t50 is not None
        # 容差 = 计数量化可分辨极限（见模块顶部 T50_TOL_S 推导）
        assert t50 == pytest.approx(T50_ANALYTIC, abs=T50_TOL_S)

    def test_t90_matches_ln10_over_k(self) -> None:
        obs = exp_decay_observations()
        _, sm = series_from(obs)
        n0 = n0_from(obs)
        t90 = crossing_time(sm, n0, 0.10, 300.0, Thresholds())
        assert t90 is not None
        # 容差 = 计数量化可分辨极限（见模块顶部 T90_TOL_S 的推导）
        assert t90 == pytest.approx(T90_ANALYTIC, abs=T90_TOL_S)

    def test_t100_epsilon_crossing(self) -> None:
        obs = exp_decay_observations()
        _, sm = series_from(obs)
        n0 = n0_from(obs)
        t100 = crossing_time(sm, n0, None, 300.0, Thresholds(), use_epsilon=True)
        assert t100 is not None
        # 结构性断言（真正有判别力的部分）：ε 口径下 T100 未删失，
        # 且必须晚于 T90（清空 90% 之前不可能清空到 ε）。
        t90 = crossing_time(sm, n0, 0.10, 300.0, Thresholds())
        assert t90 is not None and t100 > t90
        # 数值断言：容差取量化可分辨极限 ±12.5s（ε=2 颗、k=0.02、
        # dt=2s 下不可逾越，详见模块顶部推导）
        assert t100 == pytest.approx(T100_ANALYTIC, abs=T100_TOL_S)

    def test_censored_when_never_crosses(self) -> None:
        obs = [make_obs(t, 100) for t in np.arange(0.0, 300.0, 2.0)]  # 不消耗
        _, sm = series_from(obs)
        n0 = n0_from(obs)
        t50 = crossing_time(sm, n0, 0.50, 300.0, Thresholds())
        assert t50 is None
        mv = clearance_metric("A8_T50", t50, 300.0)
        assert mv.status == "censored"
        assert mv.value is None  # 绝不为 0
        assert mv.quality["window_s"] == 300.0
        assert mv.reason is not None and ">" in mv.reason
        assert "censored" in mv.flags

    def test_t100_carries_pairing_flag(self) -> None:
        obs = exp_decay_observations()
        _, sm = series_from(obs)
        n0 = n0_from(obs)
        t100 = crossing_time(sm, n0, None, 300.0, Thresholds(), use_epsilon=True)
        mv = clearance_metric(
            "A10_T100", t100, 300.0, extra_flags=("must_pair_with_pelletloss",)
        )
        assert "must_pair_with_pelletloss" in mv.flags

    def test_sinking_degrades_with_nonfeeding_flag(self) -> None:
        obs = exp_decay_observations()
        _, sm = series_from(obs)
        n0 = n0_from(obs)
        t90 = crossing_time(sm, n0, 0.10, 300.0, Thresholds())
        mv = clearance_metric("A9_T90", t90, 300.0, pellet_type="sinking")
        assert mv.status == "degraded"
        assert "contains_non_feeding_loss" in mv.flags

    def test_unknown_pellet_type_flag_only_no_degrade(self) -> None:
        obs = exp_decay_observations()
        _, sm = series_from(obs)
        n0 = n0_from(obs)
        t90 = crossing_time(sm, n0, 0.10, 300.0, Thresholds())
        mv = clearance_metric("A9_T90", t90, 300.0, pellet_type="unknown")
        assert mv.status == "ok"  # 未确认元数据只加标记
        assert "sedimentation_risk_unassessed" in mv.flags


# ----------------------------------------------------------------------
# A5–A7 · 速率
# ----------------------------------------------------------------------
class TestRateAnalytic:
    def _run(self, obs, n0):
        raw, sm = series_from(obs)
        t50 = crossing_time(sm, n0, 0.50, 300.0, Thresholds())
        idx = sm.valid_mask()
        return rate_metrics(
            sm.t[idx], sm.n[idx], n0, t50, Thresholds(), 300.0, {}
        )

    def test_vbar50_analytic(self) -> None:
        obs = exp_decay_observations()
        n0 = n0_from(obs)
        out = self._run(obs, n0)
        expected = 0.5 * n0 / T50_ANALYTIC
        assert out["A7_v50"].value == pytest.approx(expected, rel=0.02)

    def test_vmax_approx_k_n0(self) -> None:
        obs = exp_decay_observations()
        n0 = n0_from(obs)
        out = self._run(obs, n0)
        assert out["A6_v_max"].value == pytest.approx(K_EXP * N0_EXP, rel=0.05)

    def test_linear_decay_vmax(self) -> None:
        # N = 100 − 0.5·t（线性消耗，r=0.5 颗/s）
        obs = [make_obs(t, int(round(100 - 0.5 * t))) for t in np.arange(0.0, 200.0, 2.0)]
        n0 = n0_from(obs)
        out = self._run(obs, n0)
        assert out["A6_v_max"].value == pytest.approx(0.5, rel=0.05)
        # 线性衰减 T50 = 100s（闭式解）
        raw, sm = series_from(obs)
        t50 = crossing_time(sm, n0, 0.50, 300.0, Thresholds())
        assert t50 == pytest.approx(100.0, abs=0.5)
        assert out["A7_v50"].value == pytest.approx(0.5, rel=0.05)

    def test_vbar50_unavailable_when_t50_censored(self) -> None:
        obs = [make_obs(t, 100) for t in np.arange(0.0, 300.0, 2.0)]
        n0 = n0_from(obs)
        out = self._run(obs, n0)
        assert out["A7_v50"].status == "unavailable"
        assert out["A7_v50"].value is None
        assert out["A7_v50"].reason is not None  # 不得用窗长代替

    def test_instantaneous_rate_no_negative_on_rebound(self) -> None:
        t = np.arange(0.0, 20.0, 2.0)
        n = np.array([100, 90, 95, 80, 82, 70, 72, 60, 62, 50], dtype=float)
        v = instantaneous_rate(t, n)
        assert np.nanmin(v) >= -1e-9  # 单调化后回补帧不产生负速率

    def test_rebound_fraction_flags_unstable(self) -> None:
        n = np.array([100, 90, 95, 80, 85, 70, 75, 60], dtype=float)
        assert rebound_fraction(n) > 0.10


# ----------------------------------------------------------------------
# A11–A13 · 残留率 / 拟合 / AUC
# ----------------------------------------------------------------------
class TestResidualAnalytic:
    def _arrays(self, obs):
        raw, sm = series_from(obs)
        t = sm.t
        n = sm.n.astype(float).copy()
        n[t < 0] = np.nan  # RR 取试验期末帧
        return t, n

    def test_fit_recovers_k_and_tau(self) -> None:
        obs = exp_decay_observations()
        t, n = self._arrays(obs)
        fit = fit_exponential(t, n, N0_EXP, Thresholds())
        assert fit["k"] is not None
        assert fit["k"] == pytest.approx(K_EXP, rel=0.01)
        assert fit["tau"] == pytest.approx(T50_ANALYTIC, rel=0.01)
        assert fit["r2"] > 0.99

    def test_residual_metrics_values(self) -> None:
        obs = exp_decay_observations()
        t, n = self._arrays(obs)
        out = residual_metrics(t, n, N0_EXP, Thresholds(), 300.0, False, {})
        # RR = N(300)/N0 ≈ e^-6 = 0.248%
        # 容差用 abs=0.5（半个量化步长）：N 为整数计数，RR 的分辨率为
        # 100/N₀ = 1 个百分点，故 N(300)=0（真实 0.248 颗四舍五入为 0）
        # 与解析值之差 0.248pp 落在半步长内 —— 这是**有效零值**，不是缺失。
        assert out["A11_RR"].value == pytest.approx(
            100 * math.exp(-6.0), abs=RR_TOL_PCT
        )
        assert out["A12_k"].status == "ok"
        assert out["A12_tau"].value == pytest.approx(T50_ANALYTIC, rel=0.01)
        # AUC60 闭式解
        assert out["A13_AUC60"].value == pytest.approx(AUC60_ANALYTIC, rel=0.01)

    def test_fit_fails_below_r2_min_not_output(self) -> None:
        # 阶梯常数序列（无衰减趋势/零变异）→ 拟合失败比给错参数好
        t = np.arange(0.0, 60.0, 2.0)
        n = np.full(t.shape, 100.0)
        fit = fit_exponential(t, n, 100.0, Thresholds())
        assert fit["k"] is None
        assert fit["unavailable_reason"] is not None

    def test_insufficient_points_no_fit(self) -> None:
        t = np.arange(0.0, 10.0, 2.0)
        n = 100.0 * np.exp(-0.02 * t)
        fit = fit_exponential(t, n, 100.0, Thresholds())
        assert fit["n_points"] < Thresholds().fit_min_points
        assert fit["k"] is None

    def test_trapz_integral_uses_real_dt(self) -> None:
        # 不等间隔梯形积分：y≡1 → 积分 = t_end - t_start（与间隔无关）
        t = np.array([0.0, 1.0, 3.0, 3.5, 10.0])
        y = np.ones(5)
        assert trapz_integral(t, y) == pytest.approx(10.0)

    def test_q_pelletloss_degrades_rr(self) -> None:
        obs = exp_decay_observations()
        t, n = self._arrays(obs)
        out = residual_metrics(
            t, n, N0_EXP, Thresholds(), 300.0, False, {},
            pellet_type="floating", q_pelletloss=0.25,
        )
        assert out["A11_RR"].status == "degraded"
        assert "contains_non_feeding_loss" in out["A11_RR"].flags

    def test_window_truncated_flag(self) -> None:
        obs = exp_decay_observations(t_end=40.0)
        t, n = self._arrays(obs)
        out = residual_metrics(t, n, N0_EXP, Thresholds(), 40.0, True, {})
        assert "window_truncated" in out["A11_RR"].flags
        # AUC 窗口截断为 40s
        assert out["A13_AUC60"].quality["window_s"] == pytest.approx(40.0)
        assert out["A13_AUC60"].status == "degraded"


# ----------------------------------------------------------------------
# A14 · 非摄食损失
# ----------------------------------------------------------------------
class TestNonFeedingLoss:
    def _link(self, events: list[tuple[float, str]]):
        from src.pipeline.pellet_linker import LinkResult, VanishEvent

        evs = [
            VanishEvent(
                track_id=i, t_last_s=t, last_xy=(50.0, 50.0), vanish_class=cls,
                reason="synthetic", missing_frames=3, partial_window=False,
            )
            for i, (t, cls) in enumerate(events)
        ]
        n_eaten = sum(1 for _, c in events if c == "eaten")
        n_drifted = sum(1 for _, c in events if c == "drifted")
        n_unknown = sum(1 for _, c in events if c == "unknown")
        return LinkResult(
            tracks=[], vanish_events=evs, n_eaten=n_eaten,
            n_drifted=n_drifted, n_unknown=n_unknown,
        )

    def test_drifted_ratio_and_flag(self) -> None:
        events = [(10.0, "eaten")] * 60 + [(20.0, "drifted")] * 20 + [(30.0, "unknown")] * 20
        link = self._link(events)
        tally = tally_vanish(link, [], 300.0, Thresholds())
        assert tally.available
        mv, q, _ = nonfeeding_loss_metric(tally, 100.0, True, Thresholds())
        assert q == pytest.approx(0.20)
        assert mv.value == pytest.approx(20.0)
        assert "contains_non_feeding_loss" in mv.flags  # 0.20 > 0.15

    def test_untracked_unavailable_never_zero(self) -> None:
        tally = tally_vanish(None, [], None, Thresholds())
        assert tally.available is False
        mv, q, _ = nonfeeding_loss_metric(tally, 100.0, True, Thresholds())
        assert mv.status == "unavailable"
        assert mv.value is None
        assert q is None  # 严禁默认为 0
        assert "未校正" in mv.reason

    def test_partial_window_flag(self) -> None:
        events = [(10.0, "eaten")] * 10
        link = self._link(events)
        tally = tally_vanish(link, [], 300.0, Thresholds())  # 300 > a14_window 60
        assert tally.partial_window
        _, _, _ = nonfeeding_loss_metric(tally, 100.0, True, Thresholds())


# ----------------------------------------------------------------------
# quality · 13 Q_* "None 也要有行"
# ----------------------------------------------------------------------
class TestQualitySignals:
    def test_all_13_keys_present_even_when_nothing_measured(self) -> None:
        signals = QualitySignals().compute([])
        for key in Q_KEYS:
            assert key in signals, f"缺少质量信号 {key}"
        # 无任何输入 → 未测得为 None（不冒充 0/False）
        assert signals["Q_det"] is None
        assert signals["Q_fdet"] is None
        assert signals["Q_vis"] is None
        assert signals["Q_track"] is None
        assert signals["Q_ids"] is None
        assert signals["Q_interf"] is None
        assert signals["Q_fg"] is None

    def test_q_det_from_pellet_conf(self) -> None:
        obs = [make_obs(0.0, 10, conf=0.8), make_obs(2.0, 8, conf=0.6)]
        signals = QualitySignals().compute(obs)
        # docs/04 §2：Q_det = median(pellet.conf) **全程** —— 所有检测框
        # 置信度的合并中位数 = median([0.8]×10 + [0.6]×8) = 0.8。
        # （逐帧中位数再平均 = 0.7 是另一种口径，但契约写的是全程合并中位数，
        #   且合并中位数才是"检测置信度中位数"的直译，故以契约为准。）
        assert signals["Q_det"] == pytest.approx(0.8)

    def test_q_fdet_vis_from_fish_extra(self) -> None:
        fish = FishDetections(
            bbox=np.array([[10, 10, 30, 30], [40, 10, 60, 30]], dtype=float),
            conf=np.array([0.7, 0.9]),
        )
        obs = make_obs(0.0, 5)
        obs.extra["fish"] = fish
        signals = QualitySignals().compute([obs])
        assert signals["Q_vis"] == pytest.approx(2.0)
        assert signals["Q_fdet"] == pytest.approx(0.8)

    def test_q_calib_baseline_booleans(self) -> None:
        base = BaselineStats(
            n_frames=30, duration_s=60.0, fish_count_mean=5.0, fish_count_std=1.0,
            n_fz_mean=2.0, n_fz_std=0.5, activity_mean=1.0, activity_std=0.2,
            flow_mean=0.5, flow_std=0.1, annd_mean=50.0, annd_std=10.0,
            pellet_count_mean=0.0,
        )
        signals = QualitySignals().compute([], baseline=base, px_per_mm=2.0)
        assert signals["Q_calib"] is True
        assert signals["Q_baseline"] is True
        # 缺失口径
        signals2 = QualitySignals().compute([])
        assert signals2["Q_calib"] is False
        assert signals2["Q_baseline"] is False

    def test_q_interf_fg_motion_from_extra(self) -> None:
        obs1 = make_obs(0.0, 5)
        obs1.extra.update({"interf_frac": 0.1, "fg_frac": 0.2, "motion_residual": 1.0})
        obs2 = make_obs(2.0, 5)
        obs2.extra.update({"interf_frac": 0.3, "fg_frac": 0.4, "motion_residual": 3.0})
        signals = QualitySignals().compute([obs1, obs2])
        assert signals["Q_interf"] == pytest.approx(0.2)
        assert signals["Q_fg"] == pytest.approx(0.3)
        assert signals["Q_motion"] == pytest.approx(2.0)

    def test_q_pelletloss_from_link_result(self) -> None:
        from src.pipeline.pellet_linker import LinkResult

        link = LinkResult(tracks=[], vanish_events=[], n_eaten=0, n_drifted=30, n_unknown=0)
        signals = QualitySignals().compute([], link_result=link, n0_det=100.0)
        assert signals["Q_pelletloss"] == pytest.approx(0.30)

    def test_q_fatelost_includes_unknown(self) -> None:
        from src.pipeline.pellet_linker import LinkResult

        # 漂出 10 + 命运未确认 10，N0=100 → Q_fatelost=0.20，Q_pelletloss=0.10
        link = LinkResult(
            tracks=[], vanish_events=[], n_eaten=0, n_drifted=10, n_unknown=10
        )
        signals = QualitySignals().compute([], link_result=link, n0_det=100.0)
        assert signals["Q_pelletloss"] == pytest.approx(0.10)
        assert signals["Q_fatelost"] == pytest.approx(0.20)

    def test_q_n0gap_requires_both_tracks(self) -> None:
        signals = QualitySignals().compute([], n0_det=100.0)
        assert signals["Q_n0gap"] is None  # meta 缺失不猜测
        signals = QualitySignals().compute([], n0_det=90.0, n0_meta=100.0)
        assert signals["Q_n0gap"] == pytest.approx(0.10)


# ----------------------------------------------------------------------
# A2 · 序列提取与人工修正轨
# ----------------------------------------------------------------------
class TestPelletSeries:
    def test_manual_count_takes_priority(self) -> None:
        obs = [make_obs(0.0, 100), make_obs(2.0, 90)]
        obs[1].extra["manual_count"] = 70
        raw = extract_pellet_series(obs)
        assert raw.n[1] == pytest.approx(70.0)
        assert raw.source == "mixed"

    def test_use_manual_false_keeps_detection_track(self) -> None:
        obs = [make_obs(0.0, 100), make_obs(2.0, 90)]
        obs[1].extra["manual_count"] = 70
        raw = extract_pellet_series(obs, use_manual=False)
        assert raw.n[1] == pytest.approx(90.0)  # 原始轨不受修正影响

    def test_frames_without_detection_skipped_not_zero(self) -> None:
        obs = [make_obs(0.0, 100), make_obs(2.0, 90)]
        obs.append(FrameObservation(frame_idx=999, t_s=4.0, dt_s=None, image=None,
                                    pellets=None, extra={}))
        raw = extract_pellet_series(obs)
        assert raw.t.shape[0] == 2  # 无数据帧跳过，不造 0 观测

    def test_zero_detection_is_valid_zero(self) -> None:
        obs = [make_obs(0.0, 0)]
        raw = extract_pellet_series(obs)
        assert raw.available
        assert raw.n[0] == 0.0  # 实测 0 是有效零值（≠ None）

    def test_no_detection_at_all_unavailable(self) -> None:
        raw = extract_pellet_series(
            [FrameObservation(0, 0.0, None, None, None, {})]
        )
        assert raw.available is False
        assert raw.reason is not None

    def test_saturation_marks_low_conf(self) -> None:
        obs = [make_obs(0.0, 600)]  # > pellet_saturation=500
        raw = extract_pellet_series(obs)
        # low_conf 是 numpy bool 数组：np.True_ is not True，须显式转 bool
        assert bool(raw.low_conf[0]) is True

    def test_smoothing_preserves_monotone_sequence(self) -> None:
        # 单调序列的滑动中位数 = 原值（边界镜像填充）
        raw, sm = series_from(exp_decay_observations())
        idx = sm.valid_mask()
        np.testing.assert_allclose(sm.n[idx], raw.n[idx], atol=1e-9)


# ----------------------------------------------------------------------
# TimeSeries 基础契约
# ----------------------------------------------------------------------
class TestTimeSeries:
    def test_valid_t_values_drops_nan(self) -> None:
        ts = TimeSeries(
            metric_id="X", t=np.array([0.0, 1.0, 2.0]),
            values=np.array([1.0, np.nan, 3.0]),
        )
        t, v = ts.valid_t_values()
        assert t.shape[0] == 2
        assert v[1] == pytest.approx(3.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeSeries(metric_id="X", t=np.array([0.0, 1.0]), values=np.array([1.0]))

    def test_to_rows_nan_becomes_none(self) -> None:
        ts = TimeSeries(
            metric_id="X", t=np.array([0.0, 1.0]), values=np.array([np.nan, 2.0])
        )
        rows = ts.to_rows()
        assert rows[0]["value"] is None
        assert rows[1]["value"] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# B2 · 投喂区指标
# ----------------------------------------------------------------------
class TestZoneMetrics:
    def _fish_obs(self, t_s: float, n_in_zone: int, n_total: int, roi: ROI) -> FrameObservation:
        fish = FishDetections(
            bbox=np.array(
                [[15 + i * 2, 15 + i * 2, 25 + i * 2, 25 + i * 2] for i in range(n_in_zone)]
                + [[200 + i * 2, 150 + i * 2, 210 + i * 2, 160 + i * 2] for i in range(n_total - n_in_zone)],
                dtype=float,
            ),
            conf=np.full(n_total, 0.8),
        )
        obs = make_obs(t_s, 10)
        obs.extra["fish"] = fish
        return obs

    def _baseline(self, n_fz_mean: float = 4.0, n_fz_std: float = 1.0) -> BaselineStats:
        return BaselineStats(
            n_frames=30, duration_s=60.0, fish_count_mean=8.0, fish_count_std=2.0,
            n_fz_mean=n_fz_mean, n_fz_std=n_fz_std, activity_mean=1.0,
            activity_std=0.2, flow_mean=0.5, flow_std=0.1, annd_mean=50.0,
            annd_std=10.0, pellet_count_mean=0.0,
        )

    def test_n_fz_mean_counts_zone_centroids(self) -> None:
        from src.metrics.group_b2.zone_metrics import zone_metrics

        roi = full_roi()
        obs = [self._fish_obs(t, 4, 10, roi) for t in (0.0, 2.0, 4.0, 6.0)]
        out, series = zone_metrics(obs, roi, RunMeta(n_fish_total=20), self._baseline(), {})
        assert out["B2-5_N_fz_mean"].value == pytest.approx(4.0)
        assert out["B2-6_P_fz"].value == pytest.approx(20.0)  # 4/20×100
        assert out["B2-7_RP"].value == pytest.approx(1.0)     # 4/4
        assert series is not None and series.n_points() == 4

    def test_no_fish_data_all_unavailable(self) -> None:
        from src.metrics.group_b2.zone_metrics import zone_metrics

        roi = full_roi()
        out, series = zone_metrics([], roi, RunMeta(), None, {})
        assert series is None
        for mid in ("B2-5_N_fz_mean", "B2-6_P_fz", "B2-7_RP"):
            assert out[mid].status == "unavailable"
            assert out[mid].value is None
            assert out[mid].reason is not None

    def test_p_fz_missing_denominator(self) -> None:
        from src.metrics.group_b2.zone_metrics import zone_metrics

        roi = full_roi()
        obs = [self._fish_obs(0.0, 4, 10, roi)]
        out, _ = zone_metrics(obs, roi, RunMeta(), self._baseline(), {})
        assert out["B2-6_P_fz"].status == "unavailable"
        assert "no_denominator" in out["B2-6_P_fz"].flags
        assert out["B2-6_P_fz"].value is None  # 不猜测总数

    def test_rp_closed_without_baseline(self) -> None:
        from src.metrics.group_b2.zone_metrics import zone_metrics

        roi = full_roi()
        obs = [self._fish_obs(0.0, 4, 10, roi)]
        out, _ = zone_metrics(obs, roi, RunMeta(n_fish_total=10), None, {})
        assert out["B2-7_RP"].status == "unavailable"
        assert "基线" in out["B2-7_RP"].reason

    def test_rp_unstable_baseline_cv(self) -> None:
        from src.metrics.group_b2.zone_metrics import zone_metrics

        roi = full_roi()
        obs = [self._fish_obs(0.0, 4, 10, roi)]
        base = self._baseline(n_fz_mean=2.0, n_fz_std=1.5)  # CV=0.75 > 0.5
        out, _ = zone_metrics(obs, roi, RunMeta(), base, {})
        assert "unstable_baseline" in out["B2-7_RP"].flags


# ----------------------------------------------------------------------
# B1 / D 组（合成帧图像）
# ----------------------------------------------------------------------
class TestSpatialHeterogeneity:
    def test_activity_from_synthetic_frames(self) -> None:
        from src.metrics.group_b1.spatial_heterogeneity import (
            compute_activity,
            d_group_metrics,
        )

        size = (160, 120)
        roi = full_roi()
        obs: list[FrameObservation] = []
        for i, t in enumerate(np.arange(-8.0, 0.0, 2.0)):  # 基线 ≥3 帧
            obs.append(
                FrameObservation(
                    frame_idx=i, t_s=float(t), dt_s=None,
                    image=render_water_frame(size, float(t), [], seed=1), pellets=None,
                )
            )
        for i, t in enumerate(np.arange(0.0, 40.0, 2.0)):
            obs.append(
                FrameObservation(
                    frame_idx=100 + i, t_s=float(t), dt_s=None,
                    image=render_water_frame(size, float(t), [], seed=1), pellets=None,
                )
            )
        res = compute_activity(obs, roi, Thresholds())
        assert res.unavailable_reason is None
        assert res.m_curve is not None and res.m_curve.n_points() > 0
        assert res.fa_curve is not None
        assert res.kurtosis_curve is not None

        d = d_group_metrics(res, {}, Thresholds(), pellet_decline_t=None)
        for mid in ("B1_kurtosis_mean", "B1_gini_mean", "B1_top5_share_mean",
                    "D2_T_start", "D3_T_end", "D4_duration"):
            assert mid in d

    def test_unavailable_without_images(self) -> None:
        from src.metrics.group_b1.spatial_heterogeneity import (
            compute_activity,
            d_group_metrics,
        )

        obs = [make_obs(0.0, 10), make_obs(2.0, 8)]  # image=None（缓存模式）
        res = compute_activity(obs, None, Thresholds())
        assert res.unavailable_reason is not None
        d = d_group_metrics(res, {}, Thresholds(), None)
        assert d["B1_kurtosis_mean"].status == "unavailable"
        assert d["D2_T_start"].status == "unavailable"

    def test_d4_unavailable_when_either_censored(self) -> None:
        # FA 不可用 + 无颗粒下降口径 → D2 删失 → D4 不可用（不用窗长代替）
        from src.metrics.group_b1.spatial_heterogeneity import d_group_metrics
        from src.metrics.group_b1.spatial_heterogeneity import ActivityResult

        res = ActivityResult()  # 全空
        d = d_group_metrics(res, {}, Thresholds(), pellet_decline_t=None)
        assert d["D4_duration"].status == "unavailable"


# ----------------------------------------------------------------------
# B1 参考区扣除 α
# ----------------------------------------------------------------------
class TestReferenceCorrection:
    def test_alpha_origin_least_squares(self) -> None:
        from src.metrics.group_b1.reference_correction import estimate_alpha

        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        feed = [2.0, 4.1, 5.9, 8.2, 9.8]  # ≈ 2×ref
        alpha, note = estimate_alpha(feed, ref)
        assert alpha == pytest.approx(2.0, rel=0.03)
        assert note is None

    def test_alpha_insufficient_samples(self) -> None:
        from src.metrics.group_b1.reference_correction import estimate_alpha

        alpha, note = estimate_alpha([1.0, 2.0], [1.0, 2.0])
        assert alpha is None
        assert note is not None and "不足" in note

    def test_alpha_zero_variance_reference(self) -> None:
        from src.metrics.group_b1.reference_correction import estimate_alpha

        alpha, note = estimate_alpha([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0])
        assert alpha is None
        assert note is not None

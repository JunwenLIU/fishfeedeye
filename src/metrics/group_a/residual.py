"""A11–A13 · 残留率 / 衰减拟合 / 累计消耗积分（T04）。

职责（docs/04 §3.1 A11/A12/A13）：
    - A11 RR = N_p(T_end)/N₀ × 100（%）：T_end = min(t0+观察窗, 视频末帧)，
      视频短于观察窗 → window_truncated=True（改用视频末帧，显式标注）；
    - A12 指数衰减拟合 N(t) = N₀·exp(−k·t)：k（1/s）、τ = ln2/k（s）、R²、
      RMSE；有效点 < 8 → 不拟合；R² < 0.60 → 不输出（拟合失败比给错
      参数好）；
    - A13 AUC₆₀ = ∫C(t)dt（颗·s）：基于真实 dt 的梯形法（np.trapz），
      观察窗 < 60s → 改用实际窗长 + window_truncated。

纪律：
    - 积分一律 np.trapz(y, t)（契约 integral_method=trapz 固定），
      绝不用 sum(y)/fps 的等间隔假设；
    - 不可用 = None + reason，绝不用 0 冒充。
"""
from __future__ import annotations

import numpy as np

from src.core.config import Thresholds
from src.core.metric_value import MetricValue

__all__ = ["residual_metrics", "fit_exponential", "trapz_integral"]


def trapz_integral(t: np.ndarray, y: np.ndarray) -> float:
    """基于真实 dt 的梯形积分（契约 §3 采样约定 ②）。

    numpy 2.0 移除 np.trapz → 使用 np.trapezoid（老版本回退 np.trapz）。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        raise ValueError("梯形积分至少需要 2 个点")
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapezoid(y, t))


def fit_exponential(
    t: np.ndarray, n: np.ndarray, n0: float, thresholds: Thresholds,
) -> dict:
    """A12 指数衰减拟合（ln 域**加权**最小二乘 + 原域 R²/RMSE）。

    Returns:
        {"k", "tau", "r2", "rmse", "n_points", "unavailable_reason"}
        —— 拟合失败时 k/tau/r2/rmse 为 None，reason 非空。
    """
    t = np.asarray(t, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = (~np.isnan(n)) & (n > 0) & (t >= 0)
    n_points = int(np.sum(ok))
    if n_points < thresholds.fit_min_points:
        return {"k": None, "tau": None, "r2": None, "rmse": None,
                "n_points": n_points,
                "unavailable_reason": (
                    f"有效点 {n_points} < {thresholds.fit_min_points}，不拟合"
                )}
    ts, ns = t[ok], n[ok]
    # 零变异先判（常数曲线）：R² 的分母为 0，此时任何回归参数都无意义，
    # 必须在进入回归之前拦掉（否则会先撞上"斜率非负"分支，给出误导性原因）。
    sst = float(np.sum((ns - ns.mean()) ** 2))
    if sst <= 1e-12:
        return {"k": None, "tau": None, "r2": None, "rmse": None,
                "n_points": n_points,
                "unavailable_reason": (
                    "拟合失败（序列零变异，常数曲线）：R² 无定义，不输出参数"
                )}
    # ln 域加权线性回归：ln N = ln N0 − k·t
    # ⚠️ 权重为什么是 n²（不是等权）：
    #   N 是**整数计数**，量化误差 σ_N ≈ 1/√12（均匀分布）；
    #   一阶传播到对数域：Var(ln N) ≈ σ_N² / N² ∝ 1/N²。
    #   ⇒ 逆方差权重 w = N²。等权 ln 拟合会让尾部的相对误差（N=1~3 时
    #   量化误差可达 ±50%）主导回归，**系统性低估 k**（实测偏低约 3.7%，
    #   对 τ = ln2/k 与"半衰期"结论是直接偏差）。
    w = ns * ns
    sw = np.sqrt(w)
    design = np.vstack([ts, np.ones_like(ts)]).T * sw[:, None]
    slope, _intercept = np.linalg.lstsq(design, np.log(ns) * sw, rcond=None)[0]
    k = -float(slope)
    if k <= 0:
        return {"k": None, "tau": None, "r2": None, "rmse": None,
                "n_points": n_points,
                "unavailable_reason": (
                    "拟合失败（拟合斜率非负，无衰减趋势）：k 无意义，不输出参数"
                )}
    tau = float(np.log(2.0) / k)
    pred = float(n0) * np.exp(-k * ts)
    sse = float(np.sum((ns - pred) ** 2))
    r2 = 1.0 - sse / sst
    rmse = float(np.sqrt(sse / n_points))
    return {"k": k, "tau": tau, "r2": float(r2), "rmse": rmse,
            "n_points": n_points, "unavailable_reason": None}


def residual_metrics(
    t: np.ndarray,
    n_smooth: np.ndarray,
    n0: float | None,
    thresholds: Thresholds,
    window_s: float,
    truncated: bool,
    quality: dict,
    base_flags: tuple[str, ...] = (),
    pellet_type: str = "floating",
    q_pelletloss: float | None = None,
) -> dict[str, MetricValue]:
    """计算 A11_RR / A12_k / A12_tau / A13_AUC60。

    Args:
        t: 原生采样时刻（秒）。
        n_smooth: 平滑后的有效计数序列（NaN 表示该点无观测）。
        n0: 主口径 N₀。
        thresholds: 阈值（fit_min_points / fit_r2_min / observation_window_s）。
        window_s: 实际观察窗长度（视频末帧时刻）。
        truncated: 视频短于标称观察窗。
        quality: MetricValue 构造所需 Q_* 引用。
        base_flags: 继承 flag。
        pellet_type: 饲料类型（sinking → contains_non_feeding_loss）。
        q_pelletloss: 非摄食损失率（> 0.15 → contains_non_feeding_loss）。
    """
    out: dict[str, MetricValue] = {}
    t = np.asarray(t, dtype=float)
    n_smooth = np.asarray(n_smooth, dtype=float)
    nonfeeding = (
        (q_pelletloss is not None and q_pelletloss > thresholds.pelletloss_degrade)
        or pellet_type == "sinking"
    )
    flags = tuple(base_flags)
    if nonfeeding:
        flags = tuple(dict.fromkeys(flags + ("contains_non_feeding_loss",)))
    if truncated:
        flags = tuple(dict.fromkeys(flags + ("window_truncated",)))

    n_valid_pts = int(np.sum(~np.isnan(n_smooth)))
    q_base = dict(quality)
    q_base["n_frames_used"] = n_valid_pts

    # ---- A11 RR ----
    if n0 is None or n0 <= 0 or n_valid_pts == 0:
        out["A11_RR"] = MetricValue(
            metric_id="A11_RR", value=None, unit="%", status="unavailable",
            reason="N₀ 不可用或无计数观测，残留率无从定义", flags=flags,
            quality=dict(q_base), unit_scale="none",
        )
    else:
        n_end = float(n_smooth[~np.isnan(n_smooth)][-1])
        rr = n_end / float(n0) * 100.0
        q11 = dict(q_base)
        q11["window_s"] = float(thresholds.observation_window_s)
        q11["window_truncated"] = bool(truncated)
        q11["method"] = "integral=trapz-free;N_end/N0"
        q11["n_p_end"] = n_end
        # 状态与 reason 必须自洽：任何非空 reason 都对应非 ok 状态。
        # 窗口截断改变了 RR 的物理含义（RR@80s ≠ RR@300s，跨 run 不可比），
        # 故与 A13_AUC60 保持一致按 degraded 处理（compare 规则 4 会拦截）。
        status = "degraded" if (nonfeeding or truncated) else "ok"
        reason: str | None = None
        if nonfeeding:
            reason = ("存在非摄食损失（Q_pelletloss 超限或沉性料）：RR 含非摄食"
                      "成分，标记 contains_non_feeding_loss")
        elif truncated:
            reason = (f"视频短于观察窗（实际 {window_s:.0f}s < "
                      f"{thresholds.observation_window_s}s），RR 为截断窗口径"
                      "（window_truncated=True，不可与长窗 run 直接比较）")
        out["A11_RR"] = MetricValue(
            metric_id="A11_RR", value=rr, unit="%", status=status,
            reason=reason, flags=flags, quality=q11, unit_scale="none",
        )

    # ---- A12 k / τ ----
    if n0 is None or n0 <= 0:
        reason = "N₀ 不可用，衰减拟合无从定义"
        for mid in ("A12_k", "A12_tau"):
            out[mid] = MetricValue(
                metric_id=mid, value=None, unit="1/s" if mid == "A12_k" else "s",
                status="unavailable", reason=reason, flags=flags,
                quality=dict(q_base), unit_scale="none",
            )
    else:
        fit = fit_exponential(t, n_smooth, float(n0), thresholds)
        q12 = dict(q_base)
        q12["method"] = "fit=exp;ln-domain-ls"
        q12["n_points"] = fit["n_points"]
        q12["R2"] = fit["r2"]
        if fit["unavailable_reason"] or fit["r2"] is not None and fit["r2"] < thresholds.fit_r2_min:
            reason = fit["unavailable_reason"] or (
                f"R² = {fit['r2']:.3f} < {thresholds.fit_r2_min:.2f}："
                "拟合失败比给错参数好，不输出"
            )
            for mid in ("A12_k", "A12_tau"):
                out[mid] = MetricValue(
                    metric_id=mid, value=None,
                    unit="1/s" if mid == "A12_k" else "s",
                    status="unavailable", reason=reason, flags=flags,
                    quality=dict(q12), unit_scale="none",
                )
        else:
            r2 = float(fit["r2"])
            status = "ok" if r2 >= 0.80 else "degraded"
            reason = None
            if status == "degraded":
                reason = f"R² = {r2:.3f} < 0.80（告警档），参数仅供参考"
            out["A12_k"] = MetricValue(
                metric_id="A12_k", value=fit["k"], unit="1/s", status=status,
                reason=reason, flags=flags, quality=dict(q12), unit_scale="none",
            )
            out["A12_tau"] = MetricValue(
                metric_id="A12_tau", value=fit["tau"], unit="s", status=status,
                reason=reason, flags=flags, quality=dict(q12), unit_scale="none",
            )

    # ---- A13 AUC₆₀ ----
    auc_hi = min(60.0, float(window_s)) if window_s > 0 else 0.0
    q13 = dict(q_base)
    q13["method"] = "integral=trapz"
    q13["window_s"] = auc_hi
    auc_truncated = bool(window_s < 60.0 - 1e-9)
    q13["window_truncated"] = auc_truncated
    flags13 = flags if not auc_truncated else tuple(
        dict.fromkeys(flags + ("window_truncated",))
    )
    if n0 is None or n0 <= 0:
        out["A13_AUC60"] = MetricValue(
            metric_id="A13_AUC60", value=None, unit="颗·s", status="unavailable",
            reason="N₀ 不可用，C(t) = N₀ − N_p(t) 无从定义", flags=flags13,
            quality=q13, unit_scale="none",
        )
    else:
        ok = (~np.isnan(n_smooth)) & (t >= 0) & (t <= auc_hi + 1e-9)
        n_pts = int(np.sum(ok))
        q13["n_frames_used"] = n_pts
        if n_pts < 2:
            out["A13_AUC60"] = MetricValue(
                metric_id="A13_AUC60", value=None, unit="颗·s",
                status="unavailable",
                reason=f"积分区间 [0, {auc_hi:.0f}]s 内有效点 {n_pts} < 2，无法积分",
                flags=flags13, quality=q13, unit_scale="none",
            )
        else:
            c = float(n0) - n_smooth[ok]
            auc = trapz_integral(t[ok], c)
            status = "ok"
            reason = None
            if auc_truncated:
                status = "degraded"
                reason = (f"观察窗 {window_s:.0f}s < 60s：改用实际窗长积分，"
                          "window_truncated=True")
            out["A13_AUC60"] = MetricValue(
                metric_id="A13_AUC60", value=auc, unit="颗·s", status=status,
                reason=reason, flags=flags13, quality=q13, unit_scale="none",
            )
    return out

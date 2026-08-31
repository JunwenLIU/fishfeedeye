"""export/charts.py · matplotlib 图表（T05）。

职责（docs/04 §3 ③ + docs/06 §6 T05 验收 5）：
    - 颗粒曲线 N_p(t) / C(t) / P(t)：**默认绘制原生序列**（真实采样点为
      marker），平滑/插值线若叠加，必须与原生点同图显示；
    - **右删失段渲染为阴影区 + ">窗长" 标注，绝不画成归零**
      （画成 0 是最典型的"静默欺骗"：用户会把"没测到"读成"测到了 0"）；
    - 活跃度 FA(t)、投喂区鱼数 N_fz(t)、人工打点累计曲线（与自动曲线
      并列，绝不合并）。

纪律：
    - 每张图必须标注口径（单位、平滑窗口、是否归一化）；
    - 删失段用阴影 + 文字标注下界，不用虚线延拓（延拓是凭空造数据）。

任务编号：T05。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")  # 无显示环境（服务端/批处理）必须
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文图表字体：优先选用系统中已安装的 CJK 字体，避免标题/标注里的中文
# 渲染成方块（'Glyph X missing from font(s) DejaVu Sans'）。找不到时退回默认
# 字体（此时中文可能缺字，但不报错，英文/数字正常）。
_CJK_FONT_CANDIDATES = [
    "Microsoft YaHei", "Source Han Sans CN", "Noto Sans SC",
    "SimHei", "WenQuanYi Zen Hei", "SimSun",
]


def _configure_cjk_font() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = [n for n in _CJK_FONT_CANDIDATES if n in available]
    if chosen:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False  # CJK 字体常缺独立减号字形


_configure_cjk_font()

__all__ = [
    "SeriesSpec",
    "plot_pellet_curve",
    "plot_time_series",
    "plot_manual_vs_auto",
    "plot_group_comparison",
    "CENSOR_HATCH_COLOR",
]

CENSOR_HATCH_COLOR = "#d0d0d0"  # 删失阴影（灰，不用醒目的红/绿，避免暗示"结果"）


@dataclass
class SeriesSpec:
    """一条待绘制的时序。"""

    t: np.ndarray
    values: np.ndarray
    label: str
    unit: str = ""
    style: str = "-o"          # '-o' 原生点；'-' 平滑/插值线
    color: str | None = None
    markersize: float = 3.5


def _valid(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = ~np.isnan(v)
    return t[ok], v[ok]


def _new_fig(title: str, xlabel: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=120)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig


def _draw_censored(
    ax: plt.Axes, window_s: float | None, note: str = ""
) -> None:
    """把观察窗之外的区域画成阴影 + ">窗长" 标注（绝不画成归零）。"""
    if window_s is None:
        return
    xlim = ax.get_xlim()
    if window_s >= xlim[1]:
        return
    ax.axvspan(window_s, xlim[1], color=CENSOR_HATCH_COLOR, alpha=0.55,
               zorder=0)
    ax.annotate(
        f">{window_s:.0f}s（右删失：观察窗内未达阈值，非 0）",
        xy=(window_s, 0.04), xycoords=("data", "axes fraction"),
        xytext=(6, 0), textcoords="offset points",
        fontsize=8, color="#555555", va="bottom",
    )
    if note:
        ax.annotate(note, xy=(window_s, 0.12),
                    xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color="#555555", va="bottom")


def plot_pellet_curve(
    path: str | Path,
    t: np.ndarray,
    n_pellets: np.ndarray,
    n0: float | None = None,
    censored_window_s: float | None = None,
    title: str = "剩余颗粒数 N_p(t)",
    smooth: np.ndarray | None = None,
    t50: float | None = None,
    t90: float | None = None,
) -> Path:
    """颗粒曲线图（原生点为 marker；删失段阴影 + ">窗长"）。

    Args:
        censored_window_s: 右删失下界（观察窗长）。清空时间未达时传入，
            用于画阴影区；**不得**把曲线延拓到该区间并画成 0。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tt, vv = _valid(t, n_pellets)
    fig = _new_fig(title, "t (s, 相对投喂起点 t0)", "剩余颗粒数 N_p (颗)")
    ax = fig.axes[0]
    if smooth is not None:
        st, sv = _valid(t, smooth)
        if st.size:
            ax.plot(st, sv, "-", color="#9ecae1", linewidth=2.0,
                    label=f"平滑（{title}）", zorder=1)
    if tt.size:
        ax.plot(tt, vv, "-o", color="#1f77b4", markersize=3.5, linewidth=1.2,
                label="原生采样点（真实观测）", zorder=2)
    if n0 is not None:
        ax.axhline(float(n0), color="#888888", linestyle=":", linewidth=1.0,
                   label=f"N0 = {n0:.0f} 颗")
    for tv, name, color in ((t50, "T50", "#2ca02c"), (t90, "T90", "#ff7f0e")):
        if tv is not None:
            ax.axvline(float(tv), color=color, linestyle="--", linewidth=1.0,
                       label=f"{name} = {float(tv):.1f}s")
    _draw_censored(ax, censored_window_s)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_time_series(
    path: str | Path,
    series: Sequence[SeriesSpec],
    title: str,
    ylabel: str,
    censored_window_s: float | None = None,
    hlines: Sequence[tuple[float, str]] = (),
) -> Path:
    """通用时序图（多条曲线；原生点默认带 marker）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = _new_fig(title, "t (s, 相对投喂起点 t0)", ylabel)
    ax = fig.axes[0]
    for spec in series:
        tt, vv = _valid(spec.t, spec.values)
        if tt.size == 0:
            continue
        ax.plot(tt, vv, spec.style, color=spec.color,
                markersize=spec.markersize, linewidth=1.3, label=spec.label,
                zorder=2)
    for level, label in hlines:
        ax.axhline(float(level), color="#999999", linestyle=":", linewidth=1.0,
                   label=label)
    _draw_censored(ax, censored_window_s)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_manual_vs_auto(
    path: str | Path,
    auto_t: np.ndarray,
    auto_consumed: np.ndarray,
    manual_series: Sequence[tuple[float, int]],
    title: str = "自动消耗曲线 vs 人工打点（并列，不合并）",
) -> Path:
    """自动累计消耗 C(t) 与人工打点累计曲线的**并列**对比图。

    纪律：两条曲线各自独立绘制，**不做任何对齐/拟合/合并**；图例与标题
    均须写明"并列，不合并"（docs/06 §3.2）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = _new_fig(title, "t (s, 相对投喂起点 t0)", "累计消耗（颗）")
    ax = fig.axes[0]
    tt, vv = _valid(auto_t, auto_consumed)
    if tt.size:
        ax.plot(tt, vv, "-o", color="#1f77b4", markersize=3.5, linewidth=1.3,
                label="自动 C(t)（颗粒曲线）", zorder=2)
    if manual_series:
        mt = np.array([p[0] for p in manual_series], dtype=float)
        mv = np.array([p[1] for p in manual_series], dtype=float)
        ax.step(mt, mv, where="post", color="#d62728", linewidth=1.6,
                label=f"人工打点累计（n={len(manual_series)}）", zorder=3)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_group_comparison(
    path: str | Path,
    metric_id: str,
    values_a: Sequence[float],
    values_b: Sequence[float],
    label_a: str = "A 组",
    label_b: str = "B 组",
    p_value: float | None = None,
    effect_size: float | None = None,
    inferable: bool = True,
) -> Path:
    """两组比较箱线图 + 散点（叠加 p 值/效应量；不可推断时明确标注）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = _new_fig(f"{metric_id} · 两组比较", "分组", metric_id)
    ax = fig.axes[0]
    data = [
        [v for v in values_a if v is not None],
        [v for v in values_b if v is not None],
    ]
    ax.boxplot(data, tick_labels=[label_a, label_b], showmeans=True)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if not vals:
            continue
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.7,
                   color="#333333", zorder=3)
    note_parts: list[str] = []
    if p_value is not None:
        note_parts.append(f"p = {p_value:.4f}")
    if effect_size is not None:
        note_parts.append(f"Cohen's d = {effect_size:.3f}")
    if note_parts:
        suffix = "" if inferable else "（仅描述性，不可做统计推断）"
        ax.set_title(f"{metric_id} · 两组比较：{', '.join(note_parts)}{suffix}")
    elif not inferable:
        ax.set_title(f"{metric_id} · 两组比较（仅描述性，不可做统计推断）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

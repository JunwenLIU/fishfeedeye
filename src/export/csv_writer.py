"""export/csv_writer.py · CSV 导出（T05）。

职责（docs/04 §6.1/§6.2/§6.3 + docs/06 §6 T05）：
    - metrics_summary.csv：17 列冻结 schema（列名与顺序冻结，空值纪律）；
    - metrics_timeseries.csv：原生**不等间隔**时间戳 + 必带的 dt_s 列 +
      每个时序一列 `<metric_id>_low_conf`；
    - metrics_timeseries_1hz.csv：**可选**的画图便利层（默认不导出）。
      每行带 interpolated 与 source_interval_s，文件头写明
      「插值数据，不可用于积分、统计或显著性检验」；
    - manual_counts.csv：打点计数产物（与自动曲线并列，永不合并）。

纪律：
    - 空值 = 空字符串，绝不为 0 / "none" / "-"；
    - 1 Hz 版是插值，"不导出"是默认值（docs/04 §3 ③）。

任务编号：T05。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "TIMESERIES_1HZ_HEADER_NOTE",
    "write_metrics_summary_csv",
    "write_timeseries_csv",
    "write_timeseries_1hz_csv",
    "write_manual_counts_csv",
    "read_metrics_summary_csv",
    "resample_to_1hz",
]

# 1 Hz 便利层文件头强制声明（docs/04 §3 ③ 原文要求）
TIMESERIES_1HZ_HEADER_NOTE = (
    "插值数据，不可用于积分、统计或显著性检验"
    "（原生不等间隔序列见 metrics_timeseries.csv）"
)


def _fmt(v: Any) -> Any:
    """CSV 值格式化：None/NaN → 空字符串（绝不为 0）。"""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{float(v):.6g}"
    return v


def write_metrics_summary_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    columns: Sequence[str],
) -> Path:
    """写 metrics_summary.csv（17 列冻结 schema，UTF-8-BOM 便于 Excel）。

    Args:
        rows: 行列表（每行为 dict，键覆盖 columns）。
        path: 输出路径。
        columns: 列名与列序（冻结，顺序即 CSV 列序）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for row in rows:
            w.writerow({c: _fmt(row.get(c)) for c in columns})
    return path


def write_timeseries_csv(
    series_list: Sequence[Any], path: str | Path
) -> Path:
    """写 metrics_timeseries.csv（原生时间戳 + dt_s + 每个时序一列 low_conf）。

    宽表格式（每指标两列），非长表——时间序列数量可控，宽表对 Excel
    用户友好得多（docs/04 §6.2）。缺失点为空字符串，绝不补 0。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["t_seconds", "dt_s", "in_baseline"]
    for ts in series_list:
        cols.append(ts.metric_id)
        cols.append(f"{ts.metric_id}_low_conf")

    all_t: dict[float, None] = {}
    for ts in series_list:
        for t in np.asarray(ts.t, dtype=float):
            all_t[float(t)] = None
    t_list = sorted(all_t)

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        prev: float | None = None
        for t in t_list:
            row: dict[str, Any] = {
                "t_seconds": f"{t:.6g}",
                "dt_s": "" if prev is None else f"{t - prev:.6g}",
                "in_baseline": "true" if t < 0 else "false",
            }
            for ts in series_list:
                tt = np.asarray(ts.t, dtype=float)
                vv = np.asarray(ts.values, dtype=float)
                hit = np.where(np.isclose(tt, t))[0]
                if hit.size:
                    v = float(vv[int(hit[0])])
                    row[ts.metric_id] = "" if np.isnan(v) else f"{v:.6g}"
                    row[f"{ts.metric_id}_low_conf"] = "false"
                else:
                    row[ts.metric_id] = ""
                    row[f"{ts.metric_id}_low_conf"] = ""
            w.writerow(row)
            prev = t
    return path


@dataclass
class ResampledPoint:
    """1 Hz 重采样后的一个点（自带插值标记，防误用）。"""

    t: float
    value: float | None
    interpolated: bool
    source_interval_s: float | None = None


def resample_to_1hz(
    t: np.ndarray, values: np.ndarray
) -> list[ResampledPoint]:
    """把原生序列重采样到 1 Hz（仅画图便利层，不可用于积分/统计）。

    原生采样点 → interpolated=False；插值点 → True，并附该处原生间隔
    source_interval_s（让用户知道这一格被"拉长"了多少）。
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(values, dtype=float)
    out: list[ResampledPoint] = []
    if t.size == 0:
        return out
    grid = np.arange(math_floor(t[0]), math_ceil(t[-1]) + 1e-9, 1.0)
    for tg in grid:
        idx = int(np.argmin(np.abs(t - tg)))
        if t.size and abs(float(t[idx]) - float(tg)) < 1e-6:
            val = None if np.isnan(v[idx]) else float(v[idx])
            interval = (
                float(t[idx + 1] - t[idx]) if idx + 1 < t.size
                else (float(t[idx] - t[idx - 1]) if idx > 0 else None)
            )
            out.append(ResampledPoint(
                t=float(tg), value=val, interpolated=False,
                source_interval_s=interval,
            ))
            continue
        # 线性插值（仅在已有观测之间；区间外不插值——那才是"制造观测"）
        lo = np.where(t <= tg)[0]
        hi = np.where(t >= tg)[0]
        if lo.size == 0 or hi.size == 0:
            out.append(ResampledPoint(t=float(tg), value=None, interpolated=True))
            continue
        i, j = int(lo[-1]), int(hi[0])
        if i == j:
            val = None if np.isnan(v[i]) else float(v[i])
            out.append(ResampledPoint(
                t=float(tg), value=val, interpolated=True,
                source_interval_s=None,
            ))
            continue
        if np.isnan(v[i]) or np.isnan(v[j]):
            out.append(ResampledPoint(t=float(tg), value=None, interpolated=True))
            continue
        frac = (float(tg) - float(t[i])) / max(float(t[j]) - float(t[i]), 1e-9)
        out.append(ResampledPoint(
            t=float(tg), value=float(v[i]) + frac * float(v[j] - v[i]),
            interpolated=True,
            source_interval_s=float(t[j] - t[i]),
        ))
    return out


def math_floor(x: float) -> float:
    import math

    return float(math.floor(x))


def math_ceil(x: float) -> float:
    import math

    return float(math.ceil(x))


def write_timeseries_1hz_csv(
    series_list: Sequence[Any], path: str | Path
) -> Path:
    """写 metrics_timeseries_1hz.csv（可选便利层；默认不导出）。

    文件首行为强制声明，每行带 interpolated 与 source_interval_s。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["t_seconds", "interpolated", "source_interval_s"]
    for ts in series_list:
        cols.append(ts.metric_id)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        fh.write(f"# {TIMESERIES_1HZ_HEADER_NOTE}\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        resampled = {
            ts.metric_id: resample_to_1hz(
                np.asarray(ts.t, dtype=float), np.asarray(ts.values, dtype=float)
            )
            for ts in series_list
        }
        n = max((len(v) for v in resampled.values()), default=0)
        for i in range(n):
            row: dict[str, Any] = {
                "t_seconds": "",
                "interpolated": "",
                "source_interval_s": "",
            }
            filled = False
            for ts in series_list:
                pts = resampled[ts.metric_id]
                if i >= len(pts):
                    row[ts.metric_id] = ""
                    continue
                p = pts[i]
                row[ts.metric_id] = "" if p.value is None else f"{p.value:.6g}"
                if not filled:
                    row["t_seconds"] = f"{p.t:.6g}"
                    row["interpolated"] = "true" if p.interpolated else "false"
                    row["source_interval_s"] = (
                        "" if p.source_interval_s is None
                        else f"{p.source_interval_s:.6g}"
                    )
                    filled = True
            w.writerow(row)
    return path


# ----------------------------------------------------------------------
# 打点计数（独立证据链，永不与自动曲线合并）
# ----------------------------------------------------------------------
@dataclass
class TallyEvent:
    """一次人工目击摄食事件（打点）。"""

    t_s: float
    operator: str = "user"
    created_at: str = ""
    phase: str = ""
    note: str = ""
    event_id: int = 0

    def to_row(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "t_s": f"{self.t_s:.6g}",
            "phase": self.phase,
            "operator": self.operator,
            "created_at": self.created_at,
            "note": self.note,
        }


@dataclass
class TallyRecorder:
    """打点计数器（Gradio 回放页消费；classDiagram TallyRecorder）。

    纪律（docs/06 §3.2）：
        - 打点结果**不修正自动指标**，只并列展示、独立成节导出；
        - 打点人与时间写入文件头（审计）；
        - 每段视频可多次打点，最后一次为当前版，历史留痕（文件不覆盖）。
    """

    events: list[TallyEvent] = field(default_factory=list)
    operator: str = "user"
    video_name: str = ""

    def on_key(self, timestamp_s: float, phase: str = "", note: str = "") -> int:
        """记录一次打点（时间戳单调性由调用方保证；此处只留痕）。

        Returns:
            事件序号（1 起）。
        """
        import datetime

        ev = TallyEvent(
            t_s=float(timestamp_s),
            operator=self.operator,
            created_at=datetime.datetime.now().isoformat(timespec="seconds"),
            phase=phase,
            note=note,
            event_id=len(self.events) + 1,
        )
        self.events.append(ev)
        return ev.event_id

    def series(self) -> list[tuple[float, int]]:
        """累计消耗事件序列 [(t_s, 累计次数), ...]（供与自动曲线并列对比）。"""
        out: list[tuple[float, int]] = []
        for i, ev in enumerate(sorted(self.events, key=lambda e: e.t_s), start=1):
            out.append((ev.t_s, i))
        return out

    def to_csv(self, path: str | Path) -> Path:
        """写 manual_counts.csv（含文件头审计信息）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import datetime

        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            fh.write(f"# video: {self.video_name}\n")
            fh.write(f"# operator: {self.operator}\n")
            fh.write(
                f"# exported_at: "
                f"{datetime.datetime.now().isoformat(timespec='seconds')}\n"
            )
            fh.write("# 打点计数为独立证据链：与自动颗粒曲线并列展示，永不合并\n")
            w = csv.DictWriter(
                fh,
                fieldnames=["event_id", "t_s", "phase", "operator",
                            "created_at", "note"],
            )
            w.writeheader()
            for ev in sorted(self.events, key=lambda e: e.t_s):
                w.writerow(ev.to_row())
        return path


def write_manual_counts_csv(events: Sequence[TallyEvent], path: str | Path,
                            video_name: str = "", operator: str = "") -> Path:
    """函数式写 manual_counts.csv（TallyRecorder.to_csv 的无状态版本）。"""
    rec = TallyRecorder(events=list(events), operator=operator or "user",
                        video_name=video_name)
    return rec.to_csv(path)


def read_metrics_summary_csv(path: str | Path) -> list[dict[str, Any]]:
    """读回 metrics_summary.csv（compare / UI 消费；空字符串 → None）。"""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: list[dict[str, Any]] = []
    for r in rows:
        d: dict[str, Any] = {}
        for k, v in r.items():
            if k is None:
                continue
            d[k] = None if v == "" else v
        out.append(d)
    return out

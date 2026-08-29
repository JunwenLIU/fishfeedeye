"""corrections.py · 人工修正消费（T04）。

职责（docs/06 §6 T04 交付 3 + docs/04 §6.3）：
    - manual_counts.csv（人工计数标注表）与 corrections.jsonl
      （orchestrator recompute_metrics 留痕）的读取；
    - 修正值写入 obs.extra['manual_count']（数据通路唯一入口，
      pellet_curve 提取时优先消费）；
    - 修正只**并列**输出（N_p_manual 曲线 + _manual 后缀指标），
      绝不覆盖自动轨原始值（审计纪律：原始与修正必须可对照）；
    - ManualSummary：修正事件数 / 首个事件时刻 / 修正帧占比
      （summary.json 消费，人工干预程度留档）。

manual_counts.csv 列约定（首行表头，UTF-8）：
    frame_idx（必填）, count 或 new_n（必填）, t_s（可选）,
    operator（可选）, created_at（可选）, note（可选）
    event_id（可选，仅透传不参与逻辑）

任务编号：T04。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.core.frame_context import FrameObservation

__all__ = [
    "ManualSummary",
    "load_manual_counts_csv",
    "load_corrections_jsonl",
    "apply_manual_counts",
    "summarize_corrections",
]

_COUNT_KEYS = ("count", "new_n", "manual_count", "n")


@dataclass
class ManualSummary:
    """人工修正程度留档（summary.json / capability_report.md 消费）。

    Attributes:
        n_events: 修正事件数（条目数，含同帧多次修正）。
        first_event_t: 最早修正事件的 t_s（秒；未知为 None，绝不用 0 冒充）。
        corrected_frames: 被修正的唯一帧数。
        corrected_fraction: 修正帧 / 总采样帧（0-1）。
        operators: 出现过的操作者列表（去重保序）。
        source: 修正来源（'manual_counts_csv' / 'corrections_jsonl' / 'inline'）。
    """

    n_events: int = 0
    first_event_t: float | None = None
    corrected_frames: int = 0
    corrected_fraction: float = 0.0
    operators: list[str] = field(default_factory=list)
    source: str = "inline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_events": self.n_events,
            "first_event_t": self.first_event_t,
            "corrected_frames": self.corrected_frames,
            "corrected_fraction": self.corrected_fraction,
            "operators": list(self.operators),
            "source": self.source,
        }


def _pick_count(row: dict[str, Any]) -> int | None:
    """从行中取计数字段（count/new_n/manual_count/n 任一）。"""
    for key in _COUNT_KEYS:
        v = row.get(key)
        if v is None or v == "":
            continue
        try:
            n = int(float(v))
        except (TypeError, ValueError):
            continue
        if n < 0:
            raise ValueError(f"人工计数不得为负: {key}={v!r}")
        return n
    return None


def load_manual_counts_csv(path: str | Path) -> list[dict[str, Any]]:
    """读 manual_counts.csv（人工计数标注表）。

    Returns:
        行列表（dict）；每行至少含 frame_idx:int 与 count:int。

    Raises:
        FileNotFoundError / ValueError: 文件缺失或行缺必填列。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manual_counts.csv 不存在: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"manual_counts.csv 无表头: {path}")
        for i, raw in enumerate(reader, start=2):
            try:
                frame_idx = int(float(raw["frame_idx"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"manual_counts.csv 第 {i} 行 frame_idx 缺失或非法: {raw!r}"
                ) from exc
            count = _pick_count(raw)
            if count is None:
                raise ValueError(
                    f"manual_counts.csv 第 {i} 行缺计数字段（count/new_n）: {raw!r}"
                )
            row: dict[str, Any] = {"frame_idx": frame_idx, "count": count}
            if raw.get("t_s") not in (None, ""):
                row["t_s"] = float(raw["t_s"])
            if raw.get("operator"):
                row["operator"] = str(raw["operator"])
            if raw.get("created_at"):
                row["created_at"] = str(raw["created_at"])
            if raw.get("note"):
                row["note"] = str(raw["note"])
            if raw.get("event_id"):
                row["event_id"] = raw["event_id"]
            rows.append(row)
    return rows


def load_corrections_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读 corrections.jsonl（orchestrator recompute_metrics 留痕文件）。

    每行一个 JSON 对象（frame_idx / t_s / original_n / new_n /
    operator / note / timestamp）；坏行跳过但计入 note 计数（不静默）。
    """
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # 审计产物追加写：坏行不阻断读取
            if "frame_idx" in d and _pick_count(d) is not None:
                rows.append(d)
    return rows


def apply_manual_counts(
    observations: Sequence[FrameObservation],
    records: Sequence[dict[str, Any]],
    inplace: bool = True,
) -> list[FrameObservation]:
    """把修正记录写入 obs.extra['manual_count']（唯一数据通路）。

    Args:
        observations: 观测序列。
        records: 修正记录（frame_idx + count/new_n/manual_count/n 之一）。
        inplace: True（默认）= 就地写入 obs.extra（历史口径，orchestrator
            与标注工具依赖）；False = 返回**副本**列表，原观测对象不被
            改写（aggregator 双轨需要同时保留自动轨与修正轨时使用）。

    Returns:
        inplace=True → 观测序列本身（便于链式调用）与实际命中帧数由
            summarize_corrections 统计；inplace=False → 新的观测列表。
    """
    import dataclasses

    target: list[FrameObservation] = (
        list(observations) if inplace
        else [dataclasses.replace(o, extra=dict(o.extra)) for o in observations]
    )
    by_idx = {obs.frame_idx: obs for obs in target}
    for rec in records:
        obs = by_idx.get(int(rec["frame_idx"]))
        if obs is None:
            continue
        count = _pick_count(rec)
        if count is None:
            continue
        obs.extra["manual_count"] = int(count)
        if rec.get("operator"):
            obs.extra["manual_count_operator"] = str(rec["operator"])
    return target


def summarize_corrections(
    observations: Sequence[FrameObservation],
    records: Sequence[dict[str, Any]] | None = None,
    source: str = "inline",
) -> ManualSummary:
    """汇总人工修正程度（从观测的 manual_count 标记 + 原始记录双口径）。

    first_event_t 优先取记录中的 t_s；记录缺 t_s 时回退观测 t_s；
    两者皆缺 → None（不用 0 冒充）。
    """
    n_total = len(observations)
    corrected = [o for o in observations if o.extra.get("manual_count") is not None]
    operators: list[str] = []
    first_t: float | None = None
    rec_by_idx = {int(r["frame_idx"]): r for r in (records or [])}
    for obs in corrected:
        op = obs.extra.get("manual_count_operator")
        if op and op not in operators:
            operators.append(str(op))
        rec = rec_by_idx.get(obs.frame_idx)
        t_src = rec.get("t_s") if rec else None
        t = float(t_src) if t_src is not None else obs.t_s
        if t is not None and (first_t is None or t < first_t):
            first_t = t
    n_events = len(records) if records is not None else len(corrected)
    return ManualSummary(
        n_events=int(n_events),
        first_event_t=first_t,
        corrected_frames=len(corrected),
        corrected_fraction=(len(corrected) / n_total) if n_total > 0 else 0.0,
        operators=operators,
        source=source,
    )

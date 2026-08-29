"""Orchestrator · 管线编排 + 缓存 + 中断续跑（T02 骨架 / T03 串接）。

职责（docs/06 §2 + §3.3 流程一）：
    ingest → preprocess → detect（双轨）→ link → cache 落盘。
    指标聚合（metrics 层）、质量信号、capability、导出六件套由 T04/T05
    接管；本模块保证：
      - run_id 规则与 run 目录隔离（不覆盖，保审计，docs/06 §7.2）；
      - run_config.yaml 留档（可复现锚点）；
      - cache/detections.jsonl 逐帧落盘；重跑（resume）时按已完成帧号
        跳过，不重复处理（T02 验收 4）；
      - FrameObservation 逐帧输出（颗粒计数、置信度、来源轨标记）；
      - 双轨偏差 >20% → 记录 Q_dualtrack_gap 质量信号（供告警）。

纪律：
    - RunMeta 缺必填字段 → 拒绝启动（MetaValidationError 报字段名）；
    - meta=None 仅允许"接入/预处理"模式（无指标产出，显式 note）；
    - 人工修正（recompute_metrics）只留痕不静默：corrections.jsonl 追加，
      修正值进入 extra['manual_count']，metrics 层（T04）优先消费。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.core.config import RunConfig
from src.core.frame_context import FrameObservation, PelletDetections, RunMeta
from src.core.roi import ROI
from src.pipeline.detectors.base import (
    DUALTRACK_GAP_WARN,
    PelletDetector,
    compute_dualtrack_gap,
)
from src.pipeline.ingest import IngestResult, ingest, make_run_id
from src.pipeline.pellet_linker import LinkResult, PelletLinker
from src.pipeline.preprocess import Preprocessor
from src.utils.logging import get_logger

__all__ = ["RunResult", "Correction", "Orchestrator"]

_log = get_logger("pipeline.orchestrator")

ProgressFn = Callable[[str, float], None]


@dataclass
class RunResult:
    """一次 run 的结果（内存态；持久化在 run 目录）。"""

    run_id: str
    run_dir: Path
    config: RunConfig
    meta: RunMeta | None
    timing: dict
    observations: list[FrameObservation]      # image 已释放（None），帧号/时间/检测在
    baseline_duration_s: float | None
    baseline_available: bool
    link_result: LinkResult | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    n_frames_processed: int = 0               # 本次实际检测的帧数（续跑=增量）
    quality_signals: dict = field(default_factory=dict)  # Q_dualtrack_gap 等

    def sampling_table(self) -> list[dict]:
        rows = []
        for obs in self.observations:
            rows.append(
                {
                    "frame_idx": obs.frame_idx,
                    "t_s": obs.t_s,
                    "dt_s": obs.dt_s,
                    "n_pellets": obs.pellets.n_det() if obs.pellets is not None else None,
                    "source": obs.extra.get("pellet_source"),
                }
            )
        return rows


@dataclass
class Correction:
    """人工修正一条记录（帧号 → 修正颗粒数）。"""

    frame_idx: int
    new_n: int
    operator: str = "user"
    note: str | None = None


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
class Orchestrator:
    """串管线 + 进度回调 + 缓存读写 + 中断续跑（classDiagram Orchestrator）。"""

    def __init__(self, runs_root: str | Path | None = None) -> None:
        if runs_root is None:
            runs_root = Path(__file__).resolve().parents[2] / "runs"
        self.runs_root = Path(runs_root)

    # ------------------------------------------------------------------
    def run(
        self,
        video_path: str | Path,
        meta: RunMeta | None = None,
        config: RunConfig | None = None,
        roi: ROI | None = None,
        t0_s: float | None = None,
        detectors: Sequence[PelletDetector] | None = None,
        linker: PelletLinker | None = None,
        run_dir: str | Path | None = None,
        progress: ProgressFn | None = None,
    ) -> RunResult:
        """执行主分析管线（流程一）。

        Args:
            meta: 用户元数据；None = 仅接入/预处理模式（不产出指标，
                显式 note）。提供时必填字段缺失 → MetaValidationError。
            config: 运行配置；None = configs/default.yaml。
            roi: 区域定义（检测过滤 / 面积积分计数区 / 消失分类）。
            t0_s: 投喂起点绝对时间戳；None = 视频首帧（无基线，显式 note）。
            detectors: 检测器列表（双轨）；None/空 = 不检测。
            linker: 颗粒关联器；None = 不关联。
            run_dir: 既有 run 目录（续跑）；None = 新建 run。
        """
        if progress is None:
            progress = lambda _s, _f: None  # noqa: E731
        from src.pipeline.meta_validation import validate_required

        cfg = config if config is not None else RunConfig.load_default()
        if meta is not None:
            validate_required(meta)  # 拒绝启动（报字段名）

        # ---- run 目录 ----
        if run_dir is not None:
            run_dir = Path(run_dir)
            if not run_dir.exists():
                raise FileNotFoundError(f"续跑目录不存在: {run_dir}")
            run_id = run_dir.name
        else:
            run_id = make_run_id(video_path)
            run_dir = self.runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
        _log.info("run 开始: %s → %s", video_path, run_dir)

        # ---- ingest ----
        progress("ingest", 0.0)
        ing: IngestResult = ingest(video_path, cfg, t0_s=t0_s, keep_images=True)

        # ---- preprocess ----
        progress("preprocess", 0.3)
        pre = Preprocessor(roi=roi, stabilize=True, apply_clahe=True, apply_warp=False)
        for obs in ing.observations:
            if obs.image is not None:
                res = pre.process(obs.image)
                obs.image = res.frame
        notes = list(ing.notes)
        notes.append(f"稳像帧间平移均值 {pre.mean_abs_shift_px:.3f}px（Q_motion 代理量）")

        # ---- 检测（含续跑）----
        progress("detect", 0.5)
        cache_path = run_dir / "cache" / "detections.jsonl"
        done_frames = self._load_done_frames(cache_path) if run_dir.exists() else set()
        detectors = [d for d in (detectors or [])]
        n_processed = 0
        if detectors:
            for k, obs in enumerate(ing.observations):
                if obs.frame_idx in done_frames:
                    obs.image = None  # 已缓存帧不再处理
                    continue
                self._detect_frame(obs, detectors, roi)
                n_processed += 1
                obs.image = None  # 检测完成即释放，防内存膨胀
                progress("detect", 0.5 + 0.4 * (k + 1) / len(ing.observations))
            # 续跑：恢复已缓存帧的检测结果
            if done_frames:
                self._restore_cached(ing.observations, cache_path)
                notes.append(
                    f"续跑：跳过已完成 {len(done_frames)} 帧，本次新处理 {n_processed} 帧"
                )
            self._write_cache(
                ing.observations, cache_path,
                only_indices={o.frame_idx for o in ing.observations
                              if o.frame_idx not in done_frames},
            )
        else:
            for obs in ing.observations:
                obs.image = None
            notes.append("未配置检测器：仅执行 ingest→preprocess（采样帧清单见 sampling_table）")

        # ---- 关联 ----
        link_result: LinkResult | None = None
        if linker is not None and any(o.pellets is not None for o in ing.observations):
            progress("link", 0.9)
            link_result = linker.link(ing.observations)
            # 消失三分类回写（A14 消费方按事件取，帧级仅写标记）
            vanished_ids = {
                e.track_id for e in link_result.vanish_events
            }
            for obs in ing.observations:
                if obs.pellets is not None and obs.pellets.track_id is not None:
                    obs.pellets.vanish_class = [
                        (next(
                            (e.vanish_class for e in link_result.vanish_events
                             if e.track_id == tid),
                            None,
                        ))
                        for tid in obs.pellets.track_id
                    ]
            if vanished_ids:
                notes.append(
                    f"消失三分类: eaten={link_result.n_eaten}, "
                    f"drifted={link_result.n_drifted}, unknown={link_result.n_unknown}"
                )

        # ---- run_config 留档 + timing 留档 ----
        cfg.to_yaml(run_dir / "run_config.yaml")
        timing_cache = run_dir / "cache" / "timing.json"
        timing_cache.parent.mkdir(parents=True, exist_ok=True)
        timing_cache.write_text(
            json.dumps(ing.timing.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        warnings = list(ing.timing.notes) if hasattr(ing.timing, "notes") else []

        quality_signals: dict = {}
        gaps = [
            o.extra.get("dualtrack_gap")
            for o in ing.observations
            if o.extra.get("dualtrack_gap") is not None
        ]
        if gaps:
            quality_signals["Q_dualtrack_gap_max"] = float(max(gaps))
            if max(gaps) > DUALTRACK_GAP_WARN:
                warnings.append(
                    f"Q_dualtrack_gap = {max(gaps):.1%} > {DUALTRACK_GAP_WARN:.0%}"
                    "：检测轨与面积积分轨计数偏差过大（密集粘连或分割阈值需复核）"
                )

        result = RunResult(
            run_id=run_id,
            run_dir=run_dir,
            config=cfg,
            meta=meta,
            timing=ing.timing.to_dict(),
            observations=ing.observations,
            baseline_duration_s=ing.baseline_duration_s,
            baseline_available=ing.baseline_available,
            link_result=link_result,
            warnings=warnings,
            notes=notes,
            n_frames_processed=n_processed,
            quality_signals=quality_signals,
        )
        progress("done", 1.0)
        _log.info("run 完成: %d 帧观测（本次检测 %d），缓存 %s",
                  len(ing.observations), n_processed, cache_path)
        return result

    # ------------------------------------------------------------------
    def load_cache(self, run_dir: str | Path) -> RunResult:
        """从缓存恢复 run（不重新解码检测；classDiagram load_cache）。"""
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run 目录不存在: {run_dir}")
        cfg = RunConfig.from_yaml(run_dir / "run_config.yaml")
        cache_path = run_dir / "cache" / "detections.jsonl"
        observations = self._read_cache(cache_path)
        timing_path = run_dir / "cache" / "timing.json"
        timing: dict = {}
        if timing_path.exists():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        return RunResult(
            run_id=run_dir.name,
            run_dir=run_dir,
            config=cfg,
            meta=None,
            timing=timing,
            observations=observations,
            baseline_duration_s=None,
            baseline_available=False,
            notes=["结果来自缓存恢复（未重新检测）"],
        )

    # ------------------------------------------------------------------
    def recompute_metrics(
        self,
        run_dir: str | Path,
        corrections: Sequence[Correction],
    ) -> RunResult:
        """人工修正后重算（只重跑 metrics 层的入口；classDiagram recompute_metrics）。

        读 cache/detections.jsonl（不重跑检测），把修正写入
        extra['manual_count'] 并追加 corrections.jsonl 留痕（帧号/原值/
        新值/时间）。指标聚合本体在 T04 aggregator；本方法保证修正的
        数据通路与留痕纪律成立。
        """
        run_dir = Path(run_dir)
        result = self.load_cache(run_dir)
        by_idx = {obs.frame_idx: obs for obs in result.observations}
        log_path = run_dir / "corrections.jsonl"
        n_applied = 0
        with open(log_path, "a", encoding="utf-8") as fh:
            for c in corrections:
                obs = by_idx.get(c.frame_idx)
                if obs is None:
                    raise ValueError(
                        f"修正帧号 {c.frame_idx} 不在缓存观测中"
                        f"（可用: {sorted(by_idx)[:5]}... 共 {len(by_idx)} 帧）"
                    )
                original_n = obs.pellets.n_det() if obs.pellets is not None else None
                obs.extra["manual_count"] = int(c.new_n)
                obs.extra["manual_count_operator"] = c.operator
                fh.write(json.dumps(
                    {
                        "frame_idx": c.frame_idx,
                        "t_s": obs.t_s,
                        "original_n": original_n,
                        "new_n": int(c.new_n),
                        "operator": c.operator,
                        "note": c.note,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                ) + "\n")
                n_applied += 1
        result.notes.append(
            f"人工修正 {n_applied} 帧（占 {n_applied / max(1, len(result.observations)):.1%}），"
            "已追加 corrections.jsonl；指标重算由 metrics 层（T04）消费 manual_count"
        )
        return result

    # ------------------------------------------------------------------
    # 检测一帧（双轨 + 来源标记 + 双轨偏差）
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_frame(
        obs: FrameObservation,
        detectors: Sequence[PelletDetector],
        roi: ROI | None,
    ) -> None:
        frame = obs.image
        det_rail: PelletDetections | None = None
        area_rail: PelletDetections | None = None
        n_det: int | None = None
        n_area: int | None = None
        extra: dict = {}
        for det in detectors:
            dets = det.detect(frame, t_s=obs.t_s, roi=roi)
            stats = det.last_stats
            if stats is not None:
                extra[f"{stats.source}_available"] = stats.available
                extra[f"{stats.source}_reason"] = stats.reason
                if stats.source == "det":
                    n_det = stats.n if stats.available else None
                    if stats.available:
                        det_rail = dets
                elif stats.source == "area":
                    n_area = stats.n if stats.available else None
                    if stats.available:
                        area_rail = dets
                        if stats.fg_area_px is not None:
                            extra["fg_area_px"] = stats.fg_area_px
        gap = compute_dualtrack_gap(n_det, n_area)
        if gap is not None:
            extra["dualtrack_gap"] = gap
        # 主轨选择：检测轨优先（召回率高）；零检出回退面积轨；
        # 仅当检测轨不可用（None）或零检出且面积轨可用时才让位。
        # 注意：检测轨 available 且实测 0 颗 = 有效的零值观测（不是"无数据"），
        # 不得回退成 pellets=None（零值纪律：None ≠ 0）。
        if det_rail is not None and (det_rail.n_det() > 0 or area_rail is None):
            obs.pellets = det_rail
            extra["pellet_source"] = "det"
        elif area_rail is not None:
            obs.pellets = area_rail
            extra["pellet_source"] = "area"
        else:
            obs.pellets = None
            extra["pellet_source"] = None
        if n_det is not None:
            extra["n_det"] = n_det
        if n_area is not None:
            extra["n_area"] = n_area
        obs.extra.update(extra)

    # ------------------------------------------------------------------
    # 缓存读写
    # ------------------------------------------------------------------
    @staticmethod
    def _load_done_frames(cache_path: Path) -> set[int]:
        if not cache_path.exists():
            return set()
        done: set[int] = set()
        with open(cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") == "frame":
                    done.add(int(d["frame_idx"]))
        return done

    @staticmethod
    def _write_cache(
        observations: Sequence[FrameObservation],
        cache_path: Path,
        only_indices: set[int] | None = None,
    ) -> None:
        """逐帧落盘 cache/detections.jsonl（追加模式，不破坏既有缓存）。

        only_indices 为 None = 全部；已落盘的帧号绝不重写（续跑幂等，
        追加语义保证之前完成的帧记录不会被截断丢失——缓存是审计产物）。
        """
        existing = Orchestrator._load_done_frames(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as fh:
            for obs in observations:
                if only_indices is not None and obs.frame_idx not in only_indices:
                    continue
                if obs.frame_idx in existing:
                    continue
                if obs.pellets is None and not obs.extra:
                    continue
                fh.write(json.dumps(_frame_to_dict(obs), ensure_ascii=False) + "\n")

    @classmethod
    def _restore_cached(
        cls,
        observations: Sequence[FrameObservation],
        cache_path: Path,
    ) -> None:
        """把缓存中的检测结果回填到对应观测（续跑场景）。"""
        cached = cls._read_cache(cache_path, keep_timing=False)
        by_idx = {o.frame_idx: o for o in cached}
        for obs in observations:
            hit = by_idx.get(obs.frame_idx)
            if hit is not None:
                obs.t_s = hit.t_s
                obs.dt_s = hit.dt_s
                obs.pellets = hit.pellets
                obs.extra = hit.extra

    @staticmethod
    def _read_cache(
        cache_path: Path, keep_timing: bool = True
    ) -> list[FrameObservation]:
        if not cache_path.exists():
            return []
        out: list[FrameObservation] = []
        with open(cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") != "frame":
                    continue
                pellets = None
                pd = d.get("pellets")
                if pd is not None:
                    pellets = PelletDetections(
                        xyxy=np.asarray(pd.get("xyxy", []), dtype=float).reshape(-1, 4),
                        conf=np.asarray(pd.get("conf", []), dtype=float).reshape(-1),
                        area_px=(
                            None if pd.get("area_px") is None
                            else np.asarray(pd["area_px"], dtype=float).reshape(-1)
                        ),
                        track_id=(
                            None if pd.get("track_id") is None
                            else np.asarray(pd["track_id"]).reshape(-1)
                        ),
                        vanish_class=pd.get("vanish_class"),
                    )
                out.append(
                    FrameObservation(
                        frame_idx=int(d["frame_idx"]),
                        t_s=float(d["t_s"]),
                        dt_s=None if d.get("dt_s") is None else float(d["dt_s"]),
                        image=None,
                        pellets=pellets,
                        extra=dict(d.get("extra") or {}),
                    )
                )
        return out


def _frame_to_dict(obs: FrameObservation) -> dict:
    d: dict = {
        "type": "frame",
        "frame_idx": int(obs.frame_idx),
        "t_s": float(obs.t_s),
        "dt_s": None if obs.dt_s is None else float(obs.dt_s),
        "pellets": None,
        "extra": _jsonify(obs.extra),
    }
    if obs.pellets is not None:
        p = obs.pellets
        d["pellets"] = {
            "xyxy": p.xyxy.reshape(-1, 4).tolist(),
            "conf": p.conf.tolist(),
            "area_px": None if p.area_px is None else p.area_px.tolist(),
            "track_id": None if p.track_id is None else p.track_id.tolist(),
            "vanish_class": p.vanish_class,
        }
    return d


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

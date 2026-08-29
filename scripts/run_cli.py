"""scripts/run_cli.py · 无 UI 命令行入口（T02 最小闭环）。

职责（docs/06 §6 T02 验收 1 + 最小闭环）：
    python scripts/run_cli.py --video demo.mp4 --config configs/default.yaml
    走通 ingest → preprocess；无检测器时输出采样帧清单（帧号/t_s/dt_s/
    基线标记）。提供 --detectors / --calibration 后串接双轨检测与关联
    （T03），结果落 runs/<run_id>/。

用法示例：
    # 仅接入+预处理（采样清单）
    python scripts/run_cli.py --video demo.mp4 --config configs/default.yaml

    # 面积积分轨（需先跑 00 标定）
    python scripts/run_cli.py --video demo.mp4 --meta meta.json \
        --t0 65.0 --roi roi.json --calibration configs/calibration/feedA.yaml \
        --detectors area

    # 开放词汇检测轨（需 ultralytics；自动回退面积轨）
    python scripts/run_cli.py --video demo.mp4 --detectors auto ...

退出码：0 正常；2 元数据必填缺失（报字段名）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 项目根加入 sys.path（脚本直跑支持）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.core.config import RunConfig  # noqa: E402
from src.core.frame_context import RunMeta  # noqa: E402
from src.core.roi import ROI  # noqa: E402
from src.pipeline.detectors.area_integral import AreaIntegralCounter  # noqa: E402
from src.pipeline.meta_validation import MetaValidationError  # noqa: E402
from src.pipeline.orchestrator import Orchestrator  # noqa: E402
from src.pipeline.pellet_dynamics import PelletDynamicsCalibration  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="鱼类摄食行为视频分析 CLI（最小闭环）")
    ap.add_argument("--video", required=True, help="视频文件路径")
    ap.add_argument("--config", default=None, help="RunConfig YAML（默认 configs/default.yaml）")
    ap.add_argument("--meta", default=None, help="RunMeta JSON 文件（做指标必填）")
    ap.add_argument("--t0", type=float, default=None,
                    help="投喂起点绝对时间戳（秒；缺省=视频首帧，无基线）")
    ap.add_argument("--roi", default=None, help="ROI JSON 文件（ROI.to_dict 格式）")
    ap.add_argument("--calibration", default=None,
                    help="A_single 标定 YAML（configs/calibration/<feed_id>.yaml）")
    ap.add_argument("--detectors", choices=["none", "area", "auto"], default="none",
                    help="检测轨：none=仅接入预处理；area=面积积分轨；"
                         "auto=优先 YOLOE 开放词汇，缺失回退面积轨")
    ap.add_argument("--runs-dir", default=None, help="runs 根目录（默认项目根 runs/）")
    ap.add_argument("--resume-run", default=None, help="续跑：既有 run 目录")
    ap.add_argument("--list-frames", action="store_true",
                    help="无检测器时打印完整采样帧清单")
    return ap.parse_args(argv)


def load_meta(path: str | None) -> RunMeta | None:
    if path is None:
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {
        f for f in d.keys()
        if f in RunMeta.__dataclass_fields__
    }
    meta = RunMeta(**{k: d[k] for k in known})
    return meta


def load_roi(path: str | None) -> ROI | None:
    if path is None:
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return ROI.from_dict(d)


def build_detectors(args: argparse.Namespace, calib: PelletDynamicsCalibration | None):
    """按 --detectors 组装检测轨列表。"""
    from src.pipeline.detectors.base import PelletDetector

    detectors: list[PelletDetector] = []
    if args.detectors == "none":
        return detectors
    # 面积积分轨（标定可用即挂载；不可用也挂载——由 available 状态显式暴露）
    area = AreaIntegralCounter(
        a_single=calib.a_single if calib is not None else None
    )
    if args.detectors == "area":
        detectors.append(area)
        return detectors
    # auto：优先 YOLOE 开放词汇（冷启动），不可用回退面积轨
    try:
        from src.pipeline.detectors.yoloe_detector import YoloEDetector

        ov = YoloEDetector()
        if ov.available():
            detectors.append(ov)
        else:
            print(f"[run_cli] YOLOE 轨不可用：{ov.unavailable_reason()}")
    except Exception as exc:
        print(f"[run_cli] YOLOE 轨加载失败：{exc}")
    detectors.append(area)
    return detectors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = (
        RunConfig.from_yaml(args.config)
        if args.config
        else RunConfig.load_default(ROOT / "configs" / "default.yaml")
    )
    try:
        meta = load_meta(args.meta)
    except json.JSONDecodeError as exc:
        print(f"[run_cli] meta JSON 解析失败: {exc}", file=sys.stderr)
        return 2

    roi = load_roi(args.roi)
    calib = None
    if args.calibration:
        calib = PelletDynamicsCalibration.from_yaml(args.calibration)

    detectors = build_detectors(args, calib)

    # 关联器：标定齐备才带物理约束；否则纯最近邻 + unknown 兜底
    from src.pipeline.pellet_linker import PelletLinker

    linker = PelletLinker(
        v_sink_max_px_s=calib.v_sink_max_px_s if calib else None,
        association_radius_px=calib.association_radius_px if calib else None,
        roi=roi,
    )

    orch = Orchestrator(runs_root=args.runs_dir or ROOT / "runs")
    try:
        result = orch.run(
            video_path=args.video,
            meta=meta,
            config=config,
            roi=roi,
            t0_s=args.t0,
            detectors=detectors,
            linker=linker if detectors else None,
            run_dir=args.resume_run,
            progress=lambda stage, frac: print(
                f"[run_cli] {stage} {frac:>5.0%}", file=sys.stderr
            ) if frac in (0.0, 0.3, 0.5, 0.9, 1.0) else None,
        )
    except MetaValidationError as exc:
        print(f"[run_cli] 拒绝启动：{exc}", file=sys.stderr)
        return 2

    # ---- 报告 ----
    print(f"\n[run_cli] run_id = {result.run_id}")
    print(f"[run_cli] run_dir = {result.run_dir}")
    print(f"[run_cli] 采样 {len(result.observations)} 帧，"
          f"基线 {result.baseline_duration_s or 0.0:.1f}s "
          f"({'可用' if result.baseline_available else '不可用(<30s)'})")
    for note in result.notes:
        print(f"[run_cli] note: {note}")
    for w in result.warnings:
        print(f"[run_cli] ⚠ warning: {w}")

    if args.detectors == "none" or not detectors:
        print("[run_cli] 无检测器模式：采样帧清单（t_s 相对 t0，基线为负）")
        print(f"{'frame_idx':>10} {'t_s':>10} {'dt_s':>8}  baseline")
        for row in result.sampling_table():
            print(f"{row['frame_idx']:>10d} {row['t_s']:>10.2f} "
                  f"{'' if row['dt_s'] is None else format(row['dt_s'], '8.2f')}  "
                  f"{'yes' if row['t_s'] < 0 else 'no'}")
        print("[run_cli] 提示：--meta/--calibration/--detectors 可接入检测与指标闭环")
    else:
        n_with = sum(1 for o in result.observations if o.pellets is not None)
        srcs: dict[str, int] = {}
        for o in result.observations:
            s = o.extra.get("pellet_source")
            if s:
                srcs[s] = srcs.get(s, 0) + 1
        print(f"[run_cli] 检测帧 {n_with}/{len(result.observations)}，来源轨 {srcs}")
        if result.link_result is not None:
            lr = result.link_result
            print(f"[run_cli] 关联: {len(lr.tracks)} 轨，消失 "
                  f"eaten={lr.n_eaten} drifted={lr.n_drifted} unknown={lr.n_unknown}")
        if result.quality_signals:
            print(f"[run_cli] 质量信号: {result.quality_signals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

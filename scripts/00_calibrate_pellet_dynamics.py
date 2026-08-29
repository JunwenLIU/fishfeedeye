"""scripts/00_calibrate_pellet_dynamics.py · A_single(t) 预标定脚本（T02）。

职责（docs/06 §6 T02 + docs/04 §5.1）：
    输入「无鱼纯饲料」对照视频（≥30fps、覆盖完整沉降/漂散过程）：
      ★ 主产出 A_single(t) 曲线（面积积分轨必需）；
      ★ v_sink_max（95 分位位移速度；A14 关联物理约束，禁止硬编码）；
      ★ association_radius（建议 2.5 × 平均等效直径）。
    输出 configs/calibration/<feed_id>.yaml。

用法：
    python scripts/00_calibrate_pellet_dynamics.py \
        --video pure_feed.mp4 --feed-id feedA \
        [--px-per-mm 3.2] [--frame-step 2] [--out configs/calibration]

注意：
    - 有鱼视频不能用于标定（消失/移动被摄食污染）；
    - v_sink_max_mm_s 需要 --px-per-mm（单应性标定产出）；未提供则只
      输出 px 口径并显式 note。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根加入 sys.path（脚本直跑支持）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.pellet_dynamics import (  # noqa: E402
    PelletDynamicsCalibration,
    SegmentationParams,
    calibrate_video,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="无鱼纯饲料视频 → A_single(t) / v_sink_max / association_radius 标定"
    )
    ap.add_argument("--video", required=True, help="无鱼纯饲料视频路径")
    ap.add_argument("--feed-id", required=True, help="饲料批次标识（输出文件名）")
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="像素/毫米尺度（单应性标定产出；缺省则 v_sink_max_mm_s 为 null）")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="帧抽样步长（默认 1 = 全帧）")
    ap.add_argument("--v-min", type=int, default=None,
                    help="分割明度下限（覆盖默认 160）")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 configs/calibration）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    params = SegmentationParams()
    if args.v_min is not None:
        params.hsv_v_min = int(args.v_min)
    out_dir = Path(args.out) if args.out else ROOT / "configs" / "calibration"

    calib: PelletDynamicsCalibration = calibrate_video(
        args.video,
        feed_id=args.feed_id,
        px_per_mm=args.px_per_mm,
        frame_step=max(1, args.frame_step),
        params=params,
    )
    out_path = calib.to_yaml(out_dir / f"{args.feed_id}.yaml")

    a_vals = [a for a in calib.a_single.area_px if a is not None]
    print(f"[00_calibrate] feed_id={calib.feed_id}")
    print(f"  帧数 {calib.n_frames}，时长 {calib.duration_s:.1f}s")
    print(f"  A_single 有效点 {len(a_vals)}/{len(calib.a_single.area_px)}"
          f"（中位 {sorted(a_vals)[len(a_vals)//2]:.1f}px²）" if a_vals else
          "  A_single 无有效点（标定失败将抛错，不应到达此处）")
    print(f"  v_sink_max: px={calib.v_sink_max_px_s}, mm/s={calib.v_sink_max_mm_s}")
    print(f"  association_radius_px: {calib.association_radius_px}")
    for note in calib.notes:
        print(f"  ⚠ {note}")
    print(f"  写出: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

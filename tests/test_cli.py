"""CLI 与标定脚本冒烟测试（T02 验收 1：scripts/run_cli.py 最小闭环）。

子进程直跑真实脚本（与用户用法一致）：
    - 无检测器模式：exit 0 + 采样帧清单输出；
    - meta 缺必填：exit 2 + 字段名（拒绝启动）；
    - 00_calibrate 标定脚本：真实合成无鱼纯饲料视频 → YAML；
    - 面积积分轨全链路 CLI：--detectors area + --calibration + --roi。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.fixtures.synthetic import PelletSpec, calibration_pellets, make_video

ROOT = Path(__file__).resolve().parents[1]
SIZE = (320, 240)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """直跑脚本（与用户命令行一致），UTF-8 输出防 GBK 乱码。"""
    env = {"PYTHONIOENCODING": "utf-8", "PATH": ""}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_cli.py")] + args,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env, timeout=300,
    )


def _run_00(args: list[str]) -> subprocess.CompletedProcess:
    env = {"PYTHONIOENCODING": "utf-8", "PATH": ""}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "00_calibrate_pellet_dynamics.py")] + args,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env, timeout=300,
    )


@pytest.fixture(scope="module")
def feeding_video(tmp_path_factory):
    """30s 投喂视频：t0=10，10 颗颗粒 t_s≥0 出现。"""
    pellets = [
        PelletSpec(start_xy=(30.0 + 35.0 * i, 60.0), radius=6.0, appear_at_s=10.0)
        for i in range(10)
    ]
    return make_video(
        tmp_path_factory.mktemp("cli") / "feeding.avi",
        fps=10.0, duration_s=30.0, pellets=pellets, size=SIZE, seed=51,
    )


@pytest.fixture(scope="module")
def pure_feed_video(tmp_path_factory):
    """6s 无鱼纯饲料标定视频。"""
    return make_video(
        tmp_path_factory.mktemp("cli_calib") / "pure_feed.avi",
        fps=10.0, duration_s=6.0,
        pellets=calibration_pellets(n=12, radius=6.0, drift_px_s=4.0, size=SIZE),
        size=SIZE, seed=52,
    )


class TestCLI冒烟:

    def test_无检测器模式_采样清单_退出码0(self, feeding_video, tmp_path) -> None:
        proc = _run([
            "--video", str(feeding_video), "--t0", "10.0",
            "--runs-dir", str(tmp_path / "runs"),
        ])
        assert proc.returncode == 0, proc.stderr
        assert "run_id" in proc.stdout
        assert "frame_idx" in proc.stdout          # 采样帧清单
        assert "t_s" in proc.stdout
        # run 目录留档
        run_dirs = list((tmp_path / "runs").iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "run_config.yaml").exists()

    def test_元数据缺必填_退出码2且报字段名(self, feeding_video, tmp_path) -> None:
        meta_path = tmp_path / "meta_bad.json"
        meta_path.write_text(
            json.dumps({"n_fish_total": 50}, ensure_ascii=False),  # 缺 4 项必填
            encoding="utf-8",
        )
        proc = _run([
            "--video", str(feeding_video), "--t0", "10.0",
            "--meta", str(meta_path),
            "--runs-dir", str(tmp_path / "runs"),
        ])
        assert proc.returncode == 2
        assert "拒绝启动" in proc.stderr
        assert "species" in proc.stderr

    def test_完整元数据_退出码0(self, feeding_video, tmp_path) -> None:
        meta_path = tmp_path / "meta_ok.json"
        meta_path.write_text(
            json.dumps(
                {
                    "species": "草鱼", "n_fish_total": 50,
                    "feed_mass_g": 200.0, "pellet_mass_mg": 150.0,
                    "pellet_type": "floating",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        proc = _run([
            "--video", str(feeding_video), "--t0", "10.0", "--meta", str(meta_path),
            "--runs-dir", str(tmp_path / "runs"),
        ])
        assert proc.returncode == 0, proc.stderr


class Test标定脚本冒烟:

    def test_标定产出YAML(self, pure_feed_video, tmp_path) -> None:
        out_dir = tmp_path / "calibration"
        proc = _run_00([
            "--video", str(pure_feed_video), "--feed-id", "feedSmoke",
            "--out", str(out_dir),
        ])
        assert proc.returncode == 0, proc.stderr
        assert "feed_id=feedSmoke" in proc.stdout
        out = out_dir / "feedSmoke.yaml"
        assert out.exists()
        d = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert d["feed_id"] == "feedSmoke"
        assert any(a is not None for a in d["a_single"]["area_px"])
        assert d["v_sink_max_px_s"] is not None and d["v_sink_max_px_s"] > 0
        assert d["association_radius_px"] is not None
        assert d["v_sink_max_mm_s"] is None  # 未提供 px_per_mm

    def test_无颗粒视频_非零退出(self, tmp_path) -> None:
        empty = make_video(
            tmp_path / "empty.avi", fps=10.0, duration_s=3.0, size=SIZE, seed=53,
        )
        proc = _run_00([
            "--video", str(empty), "--feed-id", "badFeed",
            "--out", str(tmp_path / "calibration"),
        ])
        assert proc.returncode != 0
        assert "A_single" in proc.stderr


class Test面积轨CLI闭环:

    def test_标定加检测加关联_退出码0(self, pure_feed_video, feeding_video, tmp_path) -> None:
        # 1) 标定
        out_dir = tmp_path / "calibration"
        proc = _run_00([
            "--video", str(pure_feed_video), "--feed-id", "feedLoop",
            "--out", str(out_dir),
        ])
        assert proc.returncode == 0, proc.stderr
        # 2) ROI：全画面（arena）+ 中央计数区（pellet_zone）
        roi_path = tmp_path / "roi.json"
        roi_path.write_text(
            json.dumps({
                "arena": [[0, 0], [320, 0], [320, 240], [0, 240]],
                "pellet_zone": [[10, 10], [310, 10], [310, 230], [10, 230]],
            }),
            encoding="utf-8",
        )
        # 3) 面积积分轨全链路
        proc = _run([
            "--video", str(feeding_video), "--t0", "10.0",
            "--roi", str(roi_path),
            "--calibration", str(out_dir / "feedLoop.yaml"),
            "--detectors", "area",
            "--runs-dir", str(tmp_path / "runs"),
        ])
        assert proc.returncode == 0, proc.stderr
        assert "检测帧" in proc.stdout
        assert "run_id" in proc.stdout
        # 检测与关联留档可查
        run_dirs = list((tmp_path / "runs").iterdir())
        cache = run_dirs[0] / "cache" / "detections.jsonl"
        assert cache.exists()
        lines = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines()]
        assert all(entry["type"] == "frame" for entry in lines)

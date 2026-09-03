#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AV-FFIA 显式下载器（Zenodo, CC-BY-4.0）
=====================================
设计目标：
  1. 实时进度 —— 终端内原地刷新进度条；后台/重定向运行时每 LOG_EVERY 秒输出一行，
     并持续写 avffia_status.json，方便你随时查看实时进度。
  2. 断点续传 —— 依赖 curl -C - 的 Range 续传（已在本机 Zenodo 验证可用），
     每次中断后从当前字节数继续，绝不重复下载。
  3. 可随时重跑 —— 脚本幂等：已完成且校验通过的 zip 自动跳过；
     中断后重新执行同一条命令即可接着下。
  4. 单实例保护 —— PID 锁，防止两个进程同时写同一个 zip 造成损坏；
     进程已死时锁自动失效（这正是"意外中断后能继续"的关键）。

用法（Git Bash / PowerShell 均可，推荐 Git Bash）：
  查看实时状态（不下载）：
      python download_avffia.py --status
  开始/继续下载（前台，进度条实时刷新）：
      python download_avffia.py
  只下某一个文件（可重复给 --file）：
      python download_avffia.py --file video_6.27-6.29.zip
  仅校验已下载 zip 完整性：
      python download_avffia.py --validate
  跳过下载后校验：
      python download_avffia.py --no-validate
  忽略单实例锁（确认旧进程已死时使用）：
      python download_avffia.py --force
"""

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import deque
from pathlib import Path

# ---------- Windows 控制台中文输出保护 ----------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 路径解析：脚本放在 scripts/ 或 datasets/raw_videos/ 都能工作——
# 向上逐级寻找含 datasets/raw_videos 的项目根，数据与日志统一落在 datasets/raw_videos/ 下。
_HERE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in [_HERE.parent, *_HERE.parents]
     if (p / "datasets" / "raw_videos").is_dir()),
    _HERE.parent,
)
BASE = PROJECT_ROOT / "datasets" / "raw_videos"
DEST = BASE / "AV-FFIA"
LOG_PATH = BASE / "avffia_progress.log"
STATUS_JSON = BASE / "avffia_status.json"
LOCK = DEST / ".download.lock"

LOG_EVERY = 10          # 非 TTY 时每多少秒输出一行进度
SPEED_WINDOW = 60       # 速度平滑窗口（秒）
MAX_ATTEMPTS = 500      # 单文件最大重连次数（每次都是续传，不会浪费流量）
RETRY_SLEEP = 3         # 重连前等待秒数

# 远端精确大小（字节），由 `curl -sIL -r 0-0 <url>` 探测得到；
# 探测失败时作为兜底，保证进度百分比与完成判定依然正确。
MANIFEST = [
    {"name": "audio_dataset.zip",    "record": "11059975", "size": 17213268034},
    {"name": "video_6.19-6.22.zip",  "record": "11059975", "size": 21764413408},
    {"name": "video_6.13-6.18.zip",  "record": "11060195", "size": 30674390494},
    {"name": "video_6.27-6.29.zip",  "record": "11060195", "size": 13049105990},
    {"name": "video_6.23-6.26.zip",  "record": "11060370", "size": 17282820624},
]

for _m in MANIFEST:
    _m["url"] = f"https://zenodo.org/api/records/{_m['record']}/files/{_m['name']}/content"


# ---------------- 工具函数 ----------------
def human(n: float) -> str:
    """字节数转人类可读"""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TB"


def hms(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fsize(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def probe_total(url: str, fallback: int) -> int:
    """用 curl 探测远端总大小（加 -r 0-0，否则 Zenodo 会返回异常响应）"""
    try:
        out = subprocess.run(
            ["curl", "-sIL", "-r", "0-0", "--connect-timeout", "20", url],
            capture_output=True, text=True, timeout=90,
        ).stdout
        for line in out.splitlines():
            low = line.lower()
            if low.startswith("content-range:") and "/" in line:
                tail = line.split("/")[-1].strip()
                if tail.isdigit() and int(tail) > 0:
                    return int(tail)
        for line in out.splitlines():
            if line.lower().startswith("content-length:"):
                v = line.split(":", 1)[1].strip()
                if v.isdigit() and int(v) > 1024:
                    return int(v)
    except Exception:
        pass
    return fallback


def pid_alive(pid: int) -> bool:
    """Windows 下判断进程是否存活"""
    if pid <= 0:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        return str(pid) in out
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False


def acquire_lock(force: bool) -> bool:
    DEST.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        raw = LOCK.read_text(encoding="utf-8", errors="ignore").strip()
        old = int(raw) if raw.isdigit() else -1
        if old > 0 and old != os.getpid() and pid_alive(old):
            if not force:
                print(f"[锁] 已有下载进程在运行（PID {old}），本实例退出以避免并发写坏 zip。")
                print(f"     确认该进程已死后可加 --force 强制启动；或先结束它再重跑。")
                return False
            print(f"[锁] 忽略存活进程 PID {old}（--force）")
        else:
            print(f"[锁] 清除失效锁（旧 PID {old or '未知'} 已不存在）")
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_lock():
    try:
        if LOCK.exists():
            raw = LOCK.read_text(encoding="utf-8", errors="ignore").strip()
            if raw.isdigit() and int(raw) == os.getpid():
                LOCK.unlink()
    except Exception:
        pass


def emit(line: str, end: str = "\n"):
    """统一输出：同时写日志文件，便于 tail -f 实时查看"""
    try:
        print(line, end=end, flush=True)
    except Exception:
        pass
    try:
        with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass


def write_status(rows):
    """写 avffia_status.json，供外部随时读取实时进度"""
    try:
        tot = sum(r["total"] or 0 for r in rows)
        got = sum(r["have"] for r in rows)
        data = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_bytes": got,
            "overall_total": tot,
            "overall_pct": round(got / tot * 100, 2) if tot else 0.0,
            "files": rows,
        }
        tmp = STATUS_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(STATUS_JSON)
    except Exception:
        pass


def snapshot():
    """读取当前所有文件状态（不下载）"""
    rows = []
    for m in MANIFEST:
        dest = DEST / m["name"]
        have = fsize(dest)
        total = m["size"]
        pct = round(have / total * 100, 2) if total else 0.0
        if total and have >= total:
            state = "完成"
        elif have > 0:
            state = "续传中"
        else:
            state = "未开始"
        rows.append({
            "name": m["name"], "have": have, "total": total,
            "pct": pct, "state": state,
        })
    return rows


def print_status(rows):
    tot = sum(r["total"] or 0 for r in rows)
    got = sum(r["have"] for r in rows)
    print()
    print("=" * 78)
    print(f"AV-FFIA 下载状态    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    print(f"{'文件':<24}{'已下载':>14}{'总大小':>14}{'进度':>9}   状态")
    print("-" * 78)
    for r in rows:
        print(f"{r['name']:<24}{human(r['have']):>14}{human(r['total']):>14}"
              f"{r['pct']:>8.1f}%   {r['state']}")
    print("-" * 78)
    print(f"{'总计':<24}{human(got):>14}{human(tot):>14}"
          f"{(got / tot * 100 if tot else 0):>8.1f}%")
    print("=" * 78)
    print()


def validate(path: Path) -> bool:
    """zip 完整性校验"""
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
        if bad is None:
            print(f"[校验] OK  {path.name}")
            return True
        print(f"[校验] 损坏：{path.name} 内 {bad} CRC 错误")
        return False
    except Exception as e:
        print(f"[校验] 失败：{path.name} -> {e}")
        return False


# ---------------- 下载核心 ----------------
def download_one(m: dict, do_validate: bool = True) -> str:
    """下载单个文件；curl 负责续传，Python 负责实时进度与重连"""
    name, url = m["name"], m["url"]
    dest = DEST / name
    DEST.mkdir(parents=True, exist_ok=True)

    total = probe_total(url, m["size"])
    if not total:
        print(f"[{name}] 无法探测远端大小，跳过（请检查网络）")
        return "error"

    have = fsize(dest)
    if have > total:
        print(f"[{name}] 本地 {human(have)} 超过远端 {human(total)}，截断后重下")
        with open(dest, "r+b") as f:
            f.truncate(total)
        have = total

    if have >= total:
        print(f"[{name}] 已完整（{human(have)}），跳过")
        return "skipped"

    isatty = sys.stdout.isatty()
    print(f"[{name}] 目标 {human(total)}，从 {human(have)} 续传 "
          f"({have / total * 100:.1f}%)")

    cmd = [
        "curl", "-C", "-", "-L",
        "--retry", "5", "--retry-delay", "10",
        "--retry-all-errors",
        "--connect-timeout", "30",
        "--speed-time", "120", "--speed-limit", "1024",
        "-o", str(dest), url,
    ]

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        have = fsize(dest)
        if have >= total:
            break

        attempt += 1
        samples = deque()
        t0 = time.time()
        last_log = 0.0

        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"[{name}] 第 {attempt} 次连接（续传自 {human(have)}）", flush=True)

        stalled_for = 0
        last_size = have
        while True:
            if proc.poll() is not None:
                break
            time.sleep(1)
            cur = fsize(dest)
            now = time.time()
            samples.append((now, cur))
            while samples and samples[0][0] < now - SPEED_WINDOW:
                samples.popleft()

            # 速度 & ETA
            if len(samples) >= 2:
                dt = samples[-1][0] - samples[0][0]
                dv = samples[-1][1] - samples[0][1]
                speed = dv / dt if dt > 0 else 0.0
            else:
                speed = 0.0
            eta = (total - cur) / speed if speed > 1024 else -1

            # 卡死检测：120 秒完全无增长则重启连接
            if cur <= last_size:
                stalled_for += 1
                if stalled_for >= 120:
                    print(f"\n[{name}] 检测到 120 秒无数据增长，重启连接", flush=True)
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except Exception:
                        proc.kill()
                    break
            else:
                stalled_for = 0
                last_size = cur

            pct = cur / total * 100 if total else 0
            bl = 26
            filled = int(pct / 100 * bl)
            bar = "#" * filled + "-" * (bl - filled)
            line = (f"[{bar}] {pct:5.1f}%  {human(cur)}/{human(total)}  "
                    f"{human(speed)}/s  ETA {hms(eta) if eta >= 0 else '--:--:--'}")

            if isatty:
                sys.stdout.write("\r" + name[:22].ljust(22) + line)
                sys.stdout.flush()
            elif now - last_log >= LOG_EVERY:
                emit(f"{time.strftime('%H:%M:%S')} {name} {line}")
                last_log = now

            write_status(snapshot())

            if cur >= total:
                # 等 curl 自然退出
                try:
                    proc.wait(timeout=30)
                except Exception:
                    proc.terminate()
                break

        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        cur = fsize(dest)
        if isatty:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if cur >= total:
            print(f"[{name}] 下载完成：{human(cur)}（用时 {hms(time.time() - t0)}，"
                  f"重连 {attempt} 次）", flush=True)
            write_status(snapshot())
            if do_validate:
                if not validate(dest):
                    print(f"[{name}] !! 校验未通过，建议手动删除后重跑本脚本重新下载")
                    return "corrupt"
            return "done"

        print(f"[{name}] 连接中断于 {human(cur)}（{cur / total * 100:.1f}%），"
              f"{RETRY_SLEEP}s 后自动续传", flush=True)
        write_status(snapshot())
        time.sleep(RETRY_SLEEP)

    print(f"[{name}] 达到最大重连次数仍未完成，请检查网络后重跑本脚本（会自动续传）",
          flush=True)
    return "failed"


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="AV-FFIA 显式下载器（实时进度 / 断点续传）")
    ap.add_argument("--status", action="store_true", help="只查看当前进度，不下载")
    ap.add_argument("--validate", action="store_true", help="只校验已下载的 zip")
    ap.add_argument("--file", action="append", default=[], help="只处理指定文件，可重复")
    ap.add_argument("--no-validate", action="store_true", help="下载后不校验")
    ap.add_argument("--force", action="store_true", help="忽略单实例锁")
    args = ap.parse_args()

    if not DEST.exists():
        DEST.mkdir(parents=True, exist_ok=True)

    if args.status:
        rows = snapshot()
        print_status(rows)
        write_status(rows)
        return 0

    if args.validate:
        ok = True
        for m in MANIFEST:
            p = DEST / m["name"]
            if p.exists() and fsize(p) >= m["size"]:
                ok &= validate(p)
            else:
                print(f"[跳过] {m['name']} 未下载完整")
        return 0 if ok else 1

    if not acquire_lock(args.force):
        return 1

    try:
        targets = MANIFEST
        if args.file:
            wanted = set(args.file)
            targets = [m for m in MANIFEST if m["name"] in wanted]
            if not targets:
                print(f"未匹配到文件：{args.file}")
                return 1

        # 先下最接近完成的，尽早产出可用文件
        rows = {r["name"]: r["pct"] for r in snapshot()}
        targets = sorted(targets, key=lambda m: -rows.get(m["name"], 0.0))

        print()
        print("=" * 78)
        print(f"AV-FFIA 下载开始   {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"目标目录：{DEST}")
        print(f"本实例 PID：{os.getpid()}")
        print("=" * 78)
        print("提示：另开一个终端执行 "
              f"tail -f \"{LOG_PATH}\" 可实时查看进度")
        print("=" * 78)
        print()

        results = {}
        for m in targets:
            results[m["name"]] = download_one(m, do_validate=not args.no_validate)
            print()

        print("=" * 78)
        print("本轮结束，汇总：")
        for k, v in results.items():
            print(f"  {k:<24} {v}")
        print("=" * 78)
        print("如中途中断，重新执行同一条命令即可续传。")
        print()
        rows = snapshot()
        print_status(rows)
        return 0
    finally:
        release_lock()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n已手动中断 —— 进度已保存，重新执行同一条命令即可续传。\n")
        release_lock()
        sys.exit(130)

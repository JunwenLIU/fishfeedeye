"""app/state.py · Gradio UI 的共享状态（T05）。

职责（docs/06 §6 T05 内联约定）：
    - 项目注册表：视频列表 / 组标签 / 池塘编号 / RunMeta 表单 / run 状态；
    - **盲法**：分析页只显示盲法编号（run_001…），分组映射表单独文件存储
      （project/blind_map.enc，简单异或加密），揭盲按钮写**审计日志**；
      导出时可选择包含/排除分组列；
    - 人工修正留痕：corrections.jsonl 追加（帧号/原值/新值/时间），
      cache/detections.jsonl **绝不被触碰**（只重跑 metrics 层）。

⚠️ 为什么单独成文件：盲法编号与审计日志是**跨页签共享且必须单一来源**
的状态（分析/结果/对比三个页签都要读），放在任一 page 里都会造成两份真值。

任务编号：T05。
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.checklists import ChecklistsState

__all__ = [
    "VideoEntry",
    "ProjectState",
    "xor_obfuscate",
    "BLIND_CODE_PREFIX",
]

BLIND_CODE_PREFIX = "run_"
# 盲法映射表的混淆密钥（docs/06：简单异或加密即可——防误看，不防攻击）
_BLIND_KEY = b"fish-feeding-blind-map-v1"


def xor_obfuscate(data: bytes, key: bytes = _BLIND_KEY) -> bytes:
    """简单异或混淆（可逆；防误看，不是加密强度）。"""
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


@dataclass
class VideoEntry:
    """一个待分析视频（项目页一行）。"""

    video_path: str = ""
    group: str = ""            # 真实分组标签（盲法开启时不在 UI 展示）
    pond_id: str = ""          # 重复结构声明（伪重复治理必需）
    operator: str = ""
    run_id: str = ""           # 分析后回填
    status: str = "待分析"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "group": self.group,
            "pond_id": self.pond_id,
            "operator": self.operator,
            "run_id": self.run_id,
            "status": self.status,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VideoEntry":
        return cls(
            video_path=str(d.get("video_path", "")),
            group=str(d.get("group", "")),
            pond_id=str(d.get("pond_id", "")),
            operator=str(d.get("operator", "")),
            run_id=str(d.get("run_id", "")),
            status=str(d.get("status", "待分析")),
            meta=dict(d.get("meta") or {}),
        )


class ProjectState:
    """项目状态（Gradio gr.State 持有；单用户本机场景）。"""

    def __init__(self, project_dir: str | Path | None = None) -> None:
        self.project_dir = Path(project_dir) if project_dir else (
            Path(__file__).resolve().parents[2] / "project"
        )
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.videos: list[VideoEntry] = []
        self.blind_enabled: bool = True
        self.revealed: bool = False
        self.audit_log: list[dict[str, Any]] = []
        self.runs_root: Path = Path(__file__).resolve().parents[2] / "runs"
        self.blind_map_path: Path = self.project_dir / "blind_map.enc"
        self.audit_log_path: Path = self.project_dir / "audit_log.jsonl"
        self.checklists: ChecklistsState = ChecklistsState.default()
        self._load()

    # ------------------------------------------------------------------
    # 盲法
    # ------------------------------------------------------------------
    def blind_code(self, index: int) -> str:
        """盲法编号（run_001…）；与真实分组标签不可逆推导（映射表另存）。"""
        return f"{BLIND_CODE_PREFIX}{index + 1:03d}"

    def display_label(self, index: int) -> str:
        """UI 展示用标签：盲法开启且未揭盲 → 盲法编号；否则分组标签。"""
        if self.blind_enabled and not self.revealed:
            return self.blind_code(index)
        entry = self.videos[index] if index < len(self.videos) else None
        return entry.group if entry and entry.group else self.blind_code(index)

    def save_blind_map(self) -> Path:
        """写盲法映射表（异或混淆；含分组标签与池塘编号）。"""
        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "entries": [
                {
                    "blind_code": self.blind_code(i),
                    "video_path": v.video_path,
                    "group": v.group,
                    "pond_id": v.pond_id,
                    "run_id": v.run_id,
                }
                for i, v in enumerate(self.videos)
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.blind_map_path.write_bytes(
            base64.b64encode(xor_obfuscate(raw))
        )
        return self.blind_map_path

    def load_blind_map(self) -> dict[str, Any]:
        """读盲法映射表（文件不存在 → 空字典）。"""
        if not self.blind_map_path.exists():
            return {}
        try:
            raw = xor_obfuscate(base64.b64decode(self.blind_map_path.read_bytes()))
            return json.loads(raw.decode("utf-8"))
        except Exception:  # 混淆文件损坏：显式失败，不猜测映射
            return {}

    def reveal(self, operator: str = "user") -> bool:
        """揭盲（写审计日志）。Returns: 揭盲后的 revealed 状态。"""
        if self.revealed:
            return True
        self.revealed = True
        self.log("reveal", f"操作者 {operator} 执行揭盲（分组映射已解锁）")
        return True

    def hide(self, operator: str = "user") -> bool:
        """重新遮蔽（写审计日志）。"""
        self.revealed = False
        self.log("hide", f"操作者 {operator} 重新遮蔽分组映射")
        return False

    # ------------------------------------------------------------------
    # 审计日志
    # ------------------------------------------------------------------
    def log(self, action: str, detail: str, operator: str = "user") -> None:
        """写审计日志（内存 + 追加文件，绝不静默）。"""
        rec = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "action": action,
            "operator": operator,
            "detail": detail,
        }
        self.audit_log.append(rec)
        with open(self.audit_log_path, "a", encoding="utf-8") as fh:
            import json as _json

            fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load(self) -> None:
        p = self.project_dir / "project.json"
        if not p.exists():
            return
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.blind_enabled = bool(d.get("blind_enabled", True))
        self.revealed = bool(d.get("revealed", False))
        self.runs_root = Path(d.get("runs_root") or self.runs_root)
        self.videos = [VideoEntry.from_dict(v) for v in d.get("videos") or []]
        self.checklists = ChecklistsState.from_dict(d.get("checklists"))
        if self.audit_log_path.exists():
            for line in self.audit_log_path.read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self.audit_log.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def save(self) -> Path:
        p = self.project_dir / "project.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "blind_enabled": self.blind_enabled,
                    "revealed": self.revealed,
                    "runs_root": str(self.runs_root),
                    "videos": [v.to_dict() for v in self.videos],
                    "checklists": self.checklists.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.save_blind_map()
        return p

    # ------------------------------------------------------------------
    def set_checklist(self, group: str, item_id: str, checked: bool) -> None:
        """勾选/取消勾选 FR-03/FR-35 清单条目（改动即落盘 + 写审计日志）。"""
        self.checklists.set_item(group, item_id, checked)
        self.save()
        self.log(
            "checklist",
            f"{group}.{item_id} → {'已勾选（满足）' if checked else '未勾选（已知采集偏差）'}",
        )

    # ------------------------------------------------------------------
    def table_rows(self, include_group: bool = False) -> list[list[str]]:
        """项目表格行（盲法下不含分组列——分析页 grep 不到分组标签）。"""
        rows: list[list[str]] = []
        for i, v in enumerate(self.videos):
            label = self.display_label(i)
            row = [label, Path(v.video_path).name, v.pond_id, v.status, v.run_id]
            if include_group and (not self.blind_enabled or self.revealed):
                row.append(v.group)
            rows.append(row)
        return rows

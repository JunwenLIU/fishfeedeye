"""core/checklists.py · 拍摄规范清单（FR-03）与实验设计清单（FR-35）。

两条"科学信度支柱"的数据驱动实现（PRD §4.1.1 FR-03 / §4.1.5 FR-35）：

- FR-03 拍摄规范自检清单：内置可勾选清单，逐条说明"为什么"与"做不到会怎样"；
  **未勾选的条目在报告中列为"已知采集偏差"**（docs/03-PRD.md:134）。
- FR-35 实验设计规范清单：交叉设计、A/B 投喂点位置互换、适应期、重复次数、
  阳性/阴性对照、随机化与盲法；逐条说明"不做会怎样"；
  **用户勾选状态写入报告**（docs/03-PRD.md:188）。

设计约定：
    - 清单项为冻结常量（IDs 即契约）；新增项须追加在末尾，不得改 ID。
    - 默认全为未勾选（False）：用户必须**主动确认**每条已满足，才能使该条
      不进入"已知采集偏差"——这是披露纪律的核心，绝不为空。
    - ChecklistsState 是清单状态的唯一内存载体，to_dict/from_dict 往返无损，
      缺失项一律按 False（未勾选）补齐，绝不误判为"已满足"。

任务编号：本文件为审计整改（FR-03/FR-35 NOT-MET）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ChecklistItem",
    "FR03_ITEMS",
    "FR35_ITEMS",
    "ChecklistsState",
]


@dataclass(frozen=True)
class ChecklistItem:
    """清单中的一条目（冻结，ID 即契约）。"""

    id: str
    label: str
    rationale: str      # 为什么（要做到）
    consequence: str    # 做不到会怎样


# ----------------------------------------------------------------------
# FR-03 拍摄规范自检清单（≥8 条，PRD §4.1.1）
# ----------------------------------------------------------------------
FR03_ITEMS: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        id="camera_fixed",
        label="相机固定",
        rationale="相机晃动引入运动模糊与帧间配准误差，破坏面积积分计数与轨迹关联的稳定性。",
        consequence="颗粒重叠区误检/漏检上升，N_p(t) 出现非真实抖动，T50/T90 估计偏差增大。",
    ),
    ChecklistItem(
        id="single_take",
        label="一镜到底",
        rationale="剪辑拼接产生时间戳不连续与帧率跳变，时间轴自检会拒绝输出断裂后的时间指标。",
        consequence="时间基准失效，衰减曲线无法按真实 dt 积分，速率类指标不可用。",
    ),
    ChecklistItem(
        id="no_slowmo_timelapse",
        label="非慢动作/延时",
        rationale="慢动作/延时会改变时间标度，使 t0 与消耗速率失去物理意义。",
        consequence="所有时间相关指标须强制标注『疑似可变帧率/时间标度失真』，跨视频比较作废。",
    ),
    ChecklistItem(
        id="pre_feed_empty_30s",
        label="含投喂前 ≥30 s 空镜头",
        rationale="基线期是 RP 与归一化活跃度的硬输入，也用于参考区扣除回归 α 的估计。",
        consequence="基线不足 30 s 时相对基线指标强制关闭，仅保留绝对值口径。",
    ),
    ChecklistItem(
        id="size_reference",
        label="画面内有尺寸参照物",
        rationale="像素—物理换算（mm/BL）必须依赖已知尺寸参照物，否则物理量纲指标无意义。",
        consequence="所有物理量纲指标降级为像素口径并标注『不可跨视频比较』。",
    ),
    ChecklistItem(
        id="avoid_glare",
        label="避免过度反光",
        rationale="反光区会淹没颗粒前景并触发 unknown 消失分类，污染 N₀ 与丢失命运判定。",
        consequence="反光区消失计入 n_unknown，命运未确认率上升，T90/RR 标注『含非摄食损失』。",
    ),
    ChecklistItem(
        id="raw_stream",
        label="原始码流不二次压缩",
        rationale="二次压缩（转码/重编码）抹除高频颗粒边缘，降低小目标召回。",
        consequence="颗粒检测置信度 Q_det 下降，N₀ 系统性低估。",
    ),
    ChecklistItem(
        id="stable_lighting",
        label="光照稳定",
        rationale="光照漂移改变帧差能量基准，使 B1 活跃度出现伪趋势。",
        consequence="活跃度曲线混入光照伪差，Q_interf 上升，建议人工复核。",
    ),
)

# ----------------------------------------------------------------------
# FR-35 实验设计检查清单（PRD §4.1.5）
# ----------------------------------------------------------------------
FR35_ITEMS: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        id="crossover_design",
        label="交叉设计",
        rationale="同一批鱼先后接受对照/实验处理，可把个体间差异作为随机效应吸收。",
        consequence="个体差异与处理效应混淆，组间差异可能只是鱼群本身不同。",
    ),
    ChecklistItem(
        id="ab_position_swap",
        label="A/B 投喂点位置互换（防条件反射）",
        rationale="固定投喂点会让鱼形成位置偏好（条件反射），位置本身成为混杂变量。",
        consequence="处理效应与位置效应不可分，结论可能归因于错误的因素。",
    ),
    ChecklistItem(
        id="acclimation",
        label="适应期",
        rationale="新环境/新饲料下鱼需适应，适应期不足时摄食行为不具代表性。",
        consequence="摄食强度被应激抑制，低估诱食效果。",
    ),
    ChecklistItem(
        id="replication_ge6",
        label="重复次数建议（≥6）",
        rationale="池塘/网箱才是独立重复单元，单池多次投喂 ≠ 独立样本（伪重复）。",
        consequence="把重复测量当独立样本会严重高估显著性，p 值失真。",
    ),
    ChecklistItem(
        id="positive_negative_control",
        label="阳性/阴性对照",
        rationale="阳性对照验证系统能检出已知效应，阴性对照排除背景噪声。",
        consequence="无法判断『无差异』是真实无效应还是系统失效。",
    ),
    ChecklistItem(
        id="randomization_blinding",
        label="随机化与盲法",
        rationale="随机化平衡未知混杂，盲法防止分析者主观偏倚。",
        consequence="分组不均衡与确认偏误进入结果，削弱论文可信度。",
    ),
)

# 反向索引（id → item），供状态层按 id 取元数据
FR03_BY_ID: dict[str, ChecklistItem] = {it.id: it for it in FR03_ITEMS}
FR35_BY_ID: dict[str, ChecklistItem] = {it.id: it for it in FR35_ITEMS}


def _default_group(items: tuple[ChecklistItem, ...]) -> dict[str, bool]:
    """生成"全未勾选"的默认状态（披露纪律：默认即偏差，须主动勾除）。"""
    return {it.id: False for it in items}


# ----------------------------------------------------------------------
# 状态容器（唯一内存载体）
# ----------------------------------------------------------------------
@dataclass
class ChecklistsState:
    """FR-03 与 FR-35 清单的勾选状态（项目级，落 project.json）。

    fr03 / fr35 为 {item_id: checked(bool)}；缺失项在 from_dict 时按 False 补齐，
    多余/未知项被丢弃（向前兼容，不因旧版脏数据误判为已满足）。
    """

    fr03: dict[str, bool] = field(default_factory=lambda: _default_group(FR03_ITEMS))
    fr35: dict[str, bool] = field(default_factory=lambda: _default_group(FR35_ITEMS))

    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "ChecklistsState":
        """全未勾选的初始状态。"""
        return cls()

    @classmethod
    def from_dict(cls, d: Any | None) -> "ChecklistsState":
        """从持久化 dict 还原（容错：缺失/未知项安全补齐或丢弃）。"""
        d = d if isinstance(d, dict) else {}
        fr03_raw = d.get("fr03") if isinstance(d.get("fr03"), dict) else {}
        fr35_raw = d.get("fr35") if isinstance(d.get("fr35"), dict) else {}
        fr03 = {
            it.id: bool(fr03_raw.get(it.id, False)) for it in FR03_ITEMS
        }
        fr35 = {
            it.id: bool(fr35_raw.get(it.id, False)) for it in FR35_ITEMS
        }
        return cls(fr03=fr03, fr35=fr35)

    def to_dict(self) -> dict[str, Any]:
        """往返无损的序列化（仅含已知 item_id）。"""
        return {
            "fr03": {it.id: bool(self.fr03.get(it.id, False)) for it in FR03_ITEMS},
            "fr35": {it.id: bool(self.fr35.get(it.id, False)) for it in FR35_ITEMS},
        }

    # ------------------------------------------------------------------
    def set_item(self, group: str, item_id: str, checked: bool) -> None:
        """勾选/取消勾选一条目（group ∈ {'fr03','fr35'}）。"""
        if group == "fr03" and item_id in FR03_BY_ID:
            self.fr03[item_id] = bool(checked)
        elif group == "fr35" and item_id in FR35_BY_ID:
            self.fr35[item_id] = bool(checked)
        else:
            raise KeyError(f"未知清单分组或条目：group={group!r} item={item_id!r}")

    # ------------------------------------------------------------------
    # FR-03：已知采集偏差
    # ------------------------------------------------------------------
    def unmet_fr03_items(self) -> list[ChecklistItem]:
        """未勾选（未满足）的 FR-03 条目 → 即"已知采集偏差"集合。"""
        return [it for it in FR03_ITEMS if not self.fr03.get(it.id, False)]

    def collection_bias_rows(self) -> list[dict[str, Any]]:
        """已知采集偏差的明细行（供报告/表格）。"""
        return [
            {
                "id": it.id,
                "label": it.label,
                "rationale": it.rationale,
                "consequence": it.consequence,
            }
            for it in self.unmet_fr03_items()
        ]

    def collection_bias_markdown(self) -> str:
        """渲染"已知采集偏差"章节（无未勾选项 → 明确写明"无"，绝不静默）。"""
        unmet = self.unmet_fr03_items()
        lines: list[str] = []
        if not unmet:
            lines.append("**无已知采集偏差**：拍摄规范清单全部勾选满足。")
            return "\n".join(lines)
        lines.append(
            f"共 **{len(unmet)}** 条拍摄规范未勾选，列为已知采集偏差"
            "（这些偏差可能影响对应指标的可靠性，审稿人问起须如实披露）："
        )
        for it in unmet:
            lines.append(f"- **{it.label}**：{it.consequence}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # FR-35：实验设计清单状态
    # ------------------------------------------------------------------
    def design_checklist_rows(self) -> list[dict[str, Any]]:
        """FR-35 勾选状态明细（供报告/表格）。"""
        rows: list[dict[str, Any]] = []
        for it in FR35_ITEMS:
            rows.append(
                {
                    "id": it.id,
                    "label": it.label,
                    "checked": bool(self.fr35.get(it.id, False)),
                    "rationale": it.rationale,
                    "consequence": it.consequence,
                }
            )
        return rows

    def design_checklist_markdown(self) -> str:
        """渲染实验设计清单章节（逐条标注是否满足 + 不做会怎样）。"""
        lines: list[str] = []
        for it in FR35_ITEMS:
            mark = "✅ 已满足" if self.fr35.get(it.id, False) else "⬜ 未满足"
            lines.append(f"- [{mark}] **{it.label}**：{it.consequence}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 报告聚合
    # ------------------------------------------------------------------
    def to_report_section(self) -> dict[str, Any]:
        """供 MetricsReport.to_dict() 直接并入的片段。"""
        return {
            "known_collection_bias": self.collection_bias_rows(),
            "design_checklist": self.design_checklist_rows(),
        }

    def n_fr03_unmet(self) -> int:
        return len(self.unmet_fr03_items())

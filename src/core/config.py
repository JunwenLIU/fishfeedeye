"""RunConfig · 可复现锚点（T01）。

职责：
    一次分析的完整参数留档（configs/default.yaml 的运行时载体），写入
    runs/<run_id>/run_config.yaml，是跨 run 比较可比性的锚点（docs/06 §7.3）。
    compare.py 七条拒绝规则消费本类的 diff() 结果。

关键能力：
    - to_yaml / from_yaml 往返无损（数值类型、None、嵌套结构均保真）；
    - diff(other) 返回逐字段差异描述列表（供 compare.py 硬校验）；
    - load_default() 从 configs/default.yaml 构造默认配置。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["SamplingParams", "Thresholds", "RunConfig"]


@dataclass
class SamplingParams:
    """非对称采样三元组（硬设计，非优化；docs/06 T02 内联约定）。

    基线 [-60s, 0) 间隔 2s；早期 [0, 60s] 间隔 1s；尾段 (60s, 300s] 间隔
    10s；尾段终点由 Thresholds.observation_window_s 决定。合计约 114 帧。
    """

    baseline_interval_s: float = 2.0
    early_interval_s: float = 1.0
    tail_interval_s: float = 10.0
    baseline_window_s: float = 60.0
    early_window_s: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_interval_s": self.baseline_interval_s,
            "early_interval_s": self.early_interval_s,
            "tail_interval_s": self.tail_interval_s,
            "baseline_window_s": self.baseline_window_s,
            "early_window_s": self.early_window_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "SamplingParams":
        if not d:
            return cls()
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class Thresholds:
    """26+ 阈值常量（docs/04 §5 逐条照抄，configs/default.yaml 同源）。

    v_sink_max_mm_s / association_radius_px 由无鱼纯饲料视频标定得到，
    禁止硬编码默认值，缺省为 None（下游必须显式处理"未标定"分支）。
    """

    baseline_window_s: int = 60
    n0_early_window_s: int = 30
    n0_gap_warn: float = 0.20
    t100_epsilon_min_pellets: int = 2
    t100_epsilon_frac: float = 0.02
    observation_window_s: int = 300
    smooth_window: int = 5
    pellet_saturation: int = 500
    pelletloss_degrade: float = 0.15
    q_det_min: float = 0.50
    q_track_min: float = 0.50
    q_interf_max: float = 0.30
    q_fg_min: float = 0.01
    annd_min_n: int = 5
    fiffb_min_n: int = 8
    fit_min_points: int = 8
    fit_r2_min: float = 0.60
    onset_sigma: float = 3.0
    onset_hold_s: float = 1.0
    offset_sigma: float = 2.0
    offset_hold_s: float = 3.0
    density_low_max: int = 15
    density_med_max: int = 40
    vigor_ratio_high: float = 3.0
    vigor_ratio_low: float = 1.5
    v_sink_max_mm_s: float | None = None
    association_radius_px: float | None = None
    a14_window_s: int = 60
    integral_method: str = "trapz"
    export_1hz_timeseries: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            out[f.name] = getattr(self, f.name)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Thresholds":
        if not d:
            return cls()
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def t100_epsilon(self, n0: float) -> float:
        """清空判定残留阈值 = max(t100_epsilon_min_pellets, t100_epsilon_frac × N₀)。"""
        return max(self.t100_epsilon_min_pellets, self.t100_epsilon_frac * n0)


@dataclass
class RunConfig:
    """单次分析的完整运行配置（可复现锚点，docs/06 §2 runs/<id>/run_config.yaml）。"""

    metrics_spec_version: str = "ms-v1"
    sampling: SamplingParams = field(default_factory=SamplingParams)
    thresholds: Thresholds = field(default_factory=Thresholds)
    model_md5: str = ""
    # t0 定义三选一：feeder_start / pellet_released / pellet_in_frame（默认 C）
    t0_definition: str = "pellet_in_frame"
    # t0 来源：manual（用户手动打点，跨组比较主场景）/ auto（自动检测）
    t0_source: str = "manual"
    # 3×3 单应性矩阵（行优先展平的 9 元素列表）；未标定为 None
    homography: list[float] | None = None
    px_per_mm_ref: float | None = None
    pellet_type: str = "floating"
    # 随机种子（NFR-05）：固定并随 run_config.yaml 留档，保证可复现。
    # 任何随机步骤（图表抖动、采样、NMS 等）须消费此种子。
    seed: int = 0
    # 是否布设浮动投喂框（FR-38）：False = 未布设（颗粒更易漂出 ROI，
    # 漂出类指标应标注为参考值）；None = 未记录（未知，须披露）。
    feedbox_deployed: bool | None = None

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics_spec_version": self.metrics_spec_version,
            "sampling": self.sampling.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "model_md5": self.model_md5,
            "t0_definition": self.t0_definition,
            "t0_source": self.t0_source,
            "homography": self.homography,
            "px_per_mm_ref": self.px_per_mm_ref,
            "pellet_type": self.pellet_type,
            "seed": self.seed,
            "feedbox_deployed": self.feedbox_deployed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        return cls(
            metrics_spec_version=d.get("metrics_spec_version", "ms-v1"),
            sampling=SamplingParams.from_dict(d.get("sampling")),
            thresholds=Thresholds.from_dict(d.get("thresholds")),
            model_md5=d.get("model_md5", ""),
            t0_definition=d.get("t0_definition", "pellet_in_frame"),
            t0_source=d.get("t0_source", "manual"),
            homography=d.get("homography"),
            px_per_mm_ref=d.get("px_per_mm_ref"),
            pellet_type=d.get("pellet_type", "floating"),
            seed=int(d.get("seed", 0)),
            feedbox_deployed=d.get("feedbox_deployed", None),
        )

    def to_yaml(self, path: str | Path) -> None:
        """写出 run_config.yaml（allow_unicode + 保序，保证人工可读且往返无损）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                self.to_dict(),
                fh,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """从 run_config.yaml 读回（与 to_yaml 往返无损）。"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        if not isinstance(d, dict):
            raise ValueError(f"run_config.yaml 内容不是映射: {path}")
        return cls.from_dict(d)

    @classmethod
    def load_default(cls, defaults_path: str | Path | None = None) -> "RunConfig":
        """从 configs/default.yaml 构造默认配置；文件缺失时用内置默认值。"""
        if defaults_path is None:
            defaults_path = (
                Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
            )
        defaults_path = Path(defaults_path)
        if not defaults_path.exists():
            return cls()
        d = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError(f"default.yaml 内容不是映射: {defaults_path}")
        cfg = cls.from_dict(d)
        # 保持"metrics_spec_version 是第一个键"的契约：即使 yaml 被外部重排，
        # from_dict 已显式读取；此处再做一次存在性断言。
        if "metrics_spec_version" not in d:
            raise ValueError("default.yaml 缺少 metrics_spec_version（必须为第一个键）")
        return cfg

    # ------------------------------------------------------------------
    # 比较与校验
    # ------------------------------------------------------------------
    def diff(self, other: "RunConfig") -> list[str]:
        """逐字段比较，返回差异描述列表（空列表 = 完全一致）。

        嵌套 dict（sampling / thresholds）递归展开为点路径（如
        "thresholds.observation_window_s: 300 != 180"），保证 compare.py
        七条拒绝规则（spec_version / model_md5 / t0_definition / t0_source /
        window_s / px_per_mm / 采样）可在字段级粒度直接过滤消费。
        """
        a = _flatten(self.to_dict())
        b = _flatten(other.to_dict())
        diffs: list[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in b:
                diffs.append(f"{key}: 仅存在于本配置 ({a[key]!r})")
            elif key not in a:
                diffs.append(f"{key}: 仅存在于对方配置 ({b[key]!r})")
            elif a[key] != b[key]:
                diffs.append(f"{key}: {a[key]!r} != {b[key]!r}")
        return diffs

    def validate(self) -> list[str]:
        """构造期语义校验，返回问题列表（空 = 通过）。"""
        problems: list[str] = []
        if not self.metrics_spec_version:
            problems.append("metrics_spec_version 为空")
        if self.t0_definition not in ("feeder_start", "pellet_released", "pellet_in_frame"):
            problems.append(f"t0_definition 非法: {self.t0_definition!r}")
        if self.t0_source not in ("manual", "auto"):
            problems.append(f"t0_source 非法: {self.t0_source!r}")
        if self.pellet_type not in ("floating", "sinking", "slow-sinking", "unknown"):
            problems.append(f"pellet_type 非法: {self.pellet_type!r}")
        if not isinstance(self.seed, int):
            problems.append(f"seed 必须为整数: {self.seed!r}")
        if self.homography is not None and len(self.homography) != 9:
            problems.append("homography 必须为 3x3 展平的 9 元素列表")
        s = self.sampling
        if s.baseline_interval_s <= 0 or s.early_interval_s <= 0 or s.tail_interval_s <= 0:
            problems.append("采样间隔必须为正数")
        return problems


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """嵌套 dict 展平为点路径键（list 原样保留，不做下标展开）。"""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out

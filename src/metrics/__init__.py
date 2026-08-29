"""指标与质量层（T04，业务核心）。

职责（docs/06 §2 文件清单 + §6 T04）：
    - group_a：饲料/颗粒维度指标（A1–A14）；
    - group_b1：活跃度（帧差能量 + 16×16 网格空间异质性 + 参考区扣除）；
    - group_b2：聚集度/投喂区（N_fz / P_fz / RP）；
    - quality：13+ 个 Q_* 质量信号（每次必算必输出，值为 None 也输出行）；
    - capability：降级矩阵（自动推断只降级不关闭；仅 Q_* 与用户元数据可关闭）；
    - aggregator：汇总（组间物理隔离 dict，严禁合成单一标量）+ 五件套导出；
    - corrections：人工修正消费（manual_counts.csv / corrections.jsonl）。

纪律（docs/04 §0）：
    - 所有 A 组输出必须经 MetricValue 构造（构造期强制校验）；
    - 右删失 ≠ 失败：T50 未达 → status='censored', value=None, window_s=窗长；
    - 字段缺失用 None，不得用 0 填充（0 是有意义的数值）；
    - 指标层只消费 FrameObservation 契约，不 import 检测器。
"""
from src.metrics.aggregator import (
    METRIC_NAMES_ZH,
    SUMMARY_COLUMNS,
    Aggregator,
    MetricsAggregator,
    MetricsReport,
    MetricsSummary,
    compute_run_metrics,
    write_run_outputs,
)
from src.metrics.capability import (
    FLAG_GLOSSARY,
    CapabilityGate,
    CapabilityReport,
    DisabledEntry,
    apply_capability,
)
from src.metrics.corrections import (
    ManualSummary,
    apply_manual_counts,
    load_corrections_jsonl,
    load_manual_counts_csv,
    summarize_corrections,
)
from src.metrics.quality import (
    Q_KEYS,
    Q_SIGNAL_KEYS,
    QualitySignals,
    to_quality_rows,
)

__all__ = [
    "SUMMARY_COLUMNS",
    "METRIC_NAMES_ZH",
    "MetricsSummary",
    "MetricsAggregator",
    "Aggregator",
    "MetricsReport",
    "compute_run_metrics",
    "write_run_outputs",
    "FLAG_GLOSSARY",
    "CapabilityGate",
    "CapabilityReport",
    "DisabledEntry",
    "apply_capability",
    "ManualSummary",
    "load_manual_counts_csv",
    "load_corrections_jsonl",
    "apply_manual_counts",
    "summarize_corrections",
    "Q_KEYS",
    "Q_SIGNAL_KEYS",
    "QualitySignals",
    "to_quality_rows",
]

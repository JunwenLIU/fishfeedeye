"""指标与质量层（T04，业务核心）。

职责（docs/06 §2 文件清单 + §6 T04）：
    - group_a：饲料/颗粒维度指标（A1–A14）；
    - group_b1：活跃度（帧差能量 + 16×16 网格空间异质性 + 参考区扣除）；
    - group_b2：聚集度/投喂区（N_fz / P_fz / RP）；
    - quality：13+ 个 Q_* 质量信号（每次必算必输出，值为 None 也输出行）；
    - capability：降级矩阵（自动推断只降级不关闭；仅 Q_* 与用户元数据
      可关闭）+ **FLAG_GLOSSARY 唯一权威份**；
    - aggregator：汇总（组间物理隔离 dict，严禁合成单一标量）+ 五件套导出；
    - corrections：人工修正消费（manual_counts.csv / corrections.jsonl）。

纪律（docs/04 §0）：
    - 所有 A 组输出必须经 MetricValue 构造（构造期强制校验）；
    - 右删失 ≠ 失败：T50 未达 → status='censored', value=None, window_s=窗长；
    - 字段缺失用 None，不得用 0 填充（0 是有意义的数值）；
    - 指标层只消费 FrameObservation 契约，不 import 检测器。

T04 收口说明（本轮修复的历史并发写入分裂）：
    - 权威口径一律以 docs/04 + docs/06 为准，重复实现只保留一份：
      FLAG_GLOSSARY → capability；ManualSummary / 修正读写 → corrections；
      quality_signals 行生成 → quality.to_quality_rows。
    - aggregator 侧保留同名再导出（quality_rows / load_corrections /
      FLAG_GLOSSARY），调用方无需改动，但**实现只有一处**。
"""
from src.metrics.aggregator import (
    METRIC_GROUPS,
    METRIC_NAMES_ZH,
    MANUAL_SUFFIX,
    SUMMARY_COLUMNS,
    Aggregator,
    MetricsAggregator,
    MetricsReport,
    MetricsSummary,
    apply_manual_corrections,
    build_baseline_stats,
    compute_metrics_from_run_dir,
    compute_run_metrics,
    quality_rows,
    summary_row,
    write_run_outputs,
)
from src.metrics.capability import (
    FLAG_GLOSSARY,
    INFORMATIVE_MISSINGNESS_HINT,
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
    # ---- 汇总与导出（aggregator）----
    "SUMMARY_COLUMNS",
    "METRIC_NAMES_ZH",
    "METRIC_GROUPS",
    "MANUAL_SUFFIX",
    "MetricsSummary",
    "MetricsAggregator",
    "Aggregator",
    "MetricsReport",
    "build_baseline_stats",
    "compute_run_metrics",
    "compute_metrics_from_run_dir",
    "write_run_outputs",
    "quality_rows",
    "summary_row",
    # ---- 降级矩阵（capability）----
    "FLAG_GLOSSARY",
    "INFORMATIVE_MISSINGNESS_HINT",
    "CapabilityGate",
    "CapabilityReport",
    "DisabledEntry",
    "apply_capability",
    # ---- 人工修正（corrections）----
    "ManualSummary",
    "load_manual_counts_csv",
    "load_corrections_jsonl",
    "apply_manual_counts",
    "apply_manual_corrections",
    "summarize_corrections",
    # ---- 质量信号（quality）----
    "Q_KEYS",
    "Q_SIGNAL_KEYS",
    "QualitySignals",
    "to_quality_rows",
]

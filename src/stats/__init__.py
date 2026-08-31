"""统计层（T05）。

    - comparison_plan.py：ComparisonPlan 抽象 + 七条比较前一致性规则
      1–5 与 7 的公共实现（两组对照是特例；剂量梯度留扩展位）；
    - two_group.py：TwoGroupPlan（Welch t / Mann–Whitney / MixedLM +
      Cohen's d(Hedges g) + 95%CI + 重复结构说明 + Holm 多重比较校正），
      并按 docs/06 §6 补齐规则 8–9（t0_definition / t0_source 一致性）。

纪律：
    - 宁可不给 p 值，不可给占位 p 值（p=1.0/0.5 一律不输出）；
    - 两个检验的 p 值都给（不隐藏与主检验不一致的那一个——挑检验是
      p 值造假的经典入口）；
    - 伪重复必须显式标注（descriptive_only）。
"""
from src.stats.comparison_plan import (
    CLEARANCE_LIKE_METRICS,
    NEUTRAL_GROUP_EXIT,
    NON_NEUTRAL_GATES,
    RERUN_HINT,
    ComparisonPlan,
    ConsistencyReport,
    RunBundle,
    Rule,
    Violation,
)
from src.stats.two_group import (
    HOLM_ALPHA,
    MIN_N_PER_GROUP,
    NORMALITY_MIN_N,
    TestResult,
    TwoGroupPlan,
    describe_group,
)

__all__ = [
    "ComparisonPlan",
    "ConsistencyReport",
    "RunBundle",
    "Rule",
    "Violation",
    "NON_NEUTRAL_GATES",
    "NEUTRAL_GROUP_EXIT",
    "CLEARANCE_LIKE_METRICS",
    "RERUN_HINT",
    "TwoGroupPlan",
    "TestResult",
    "describe_group",
    "MIN_N_PER_GROUP",
    "NORMALITY_MIN_N",
    "HOLM_ALPHA",
]

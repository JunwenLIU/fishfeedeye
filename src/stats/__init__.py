"""统计层（T05）。

    - comparison_plan.py：ComparisonPlan 抽象 + 七条比较前一致性规则
      （两组对照是特例；剂量梯度留扩展位）；
    - two_group.py：TwoGroupPlan（Welch t / Mann-Whitney / MixedLM +
      Cohen's d + 95%CI + 重复结构说明）。

纪律：宁可不给 p 值，不可给错的 p 值；伪重复必须显式标注。
"""
from src.stats.comparison_plan import (
    CLEARANCE_LIKE_METRICS,
    NEUTRAL_GROUP_EXIT,
    NON_NEUTRAL_GATES,
    ComparisonPlan,
    ConsistencyReport,
    RunBundle,
    Rule,
    Violation,
)
from src.stats.two_group import (
    MIN_N_PER_GROUP,
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
    "TwoGroupPlan",
    "TestResult",
    "describe_group",
    "MIN_N_PER_GROUP",
]

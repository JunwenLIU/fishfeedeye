"""统计层（T05，跨 run 比较与两组对照）。

    - comparison_plan：ComparisonPlan 抽象 + RunRef + 七条比较前一致性
      校验规则（拒绝即列出差异 + 重跑提示，**无强制比较开关**）；
    - two_group：TwoGroupPlan（两组对照 = ComparisonPlan 特例）——
      Welch t / Mann–Whitney 自动选择 + Cohen's d（Hedges 校正）+
      95%CI + n + 重复结构说明 + 多池塘 MixedLM + Holm 多重比较校正。

纪律：
    - 样本量不足不输出 p 值（绝不占位）；
    - 两个检验的 p 值都给，结论不一致时显式告警；
    - 单池 / pond_id 未声明 → descriptive_only（仅描述性，不可推断）。

任务编号：T05。
"""
from src.stats.comparison_plan import (
    CONSISTENCY_RULES,
    RULE_DESCRIPTIONS,
    ComparisonPlan,
    ComparisonResult,
    RunRef,
    check_consistency,
    load_run_refs,
)
from src.stats.two_group import TwoGroupPlan

__all__ = [
    "CONSISTENCY_RULES",
    "RULE_DESCRIPTIONS",
    "ComparisonPlan",
    "ComparisonResult",
    "RunRef",
    "TwoGroupPlan",
    "check_consistency",
    "load_run_refs",
]

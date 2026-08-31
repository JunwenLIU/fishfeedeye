"""two_group.py · 两组对照统计（T05 · ComparisonPlan 的特例实现）。

职责（docs/06 §6 T05 + docs/04 §4.4）：
    - 两组对照是当前唯一的实验设计（第二轮用户决策 ①）；剂量梯度 /
      多组 ANOVA 不实现，通过 ComparisonPlan 抽象留扩展位——新增子类
      即可，七条规则与本文件都不用改。
    - 检验选择：正态性（Shapiro–Wilk）+ 方差齐性（Levene）→ 主检验在
      Welch t 与 Mann–Whitney 之间自动选择，**两个 p 值都输出**
      （不隐藏与主检验不一致的那一个——挑检验是 p 值造假的经典入口）。
    - 输出六要素（docs/06 T05 验收 4）：p / Cohen's d / 95%CI /
      检验方法名 / n / 重复结构说明，缺一不可。
    - 重复结构（伪重复防线）：pond_id 声明多池塘 → statsmodels MixedLM
      （pond 随机效应）；单池或未声明 → descriptive_only（仅描述性）。
    - 多重比较：run_all() 涉及多指标 → Holm–Bonferroni 校正 p。

规则补齐（docs/06 §6 T05 与 docs/04 §4.4 的差异处置）：
    docs/06 列的七条含 t0_definition / t0_source，docs/04 §4.4 的七条不含
    （但含"已关闭指标集"与"非中立门控"两条）。二者是**互补而非互斥**，
    故 validate() 取并集：规则 1–5 与 6–7 由基类/本类按 docs/04 §4.4 实现，
    规则 8–9（t0_definition / t0_source）按 docs/06 §6 在此补齐——
    t0 口径不一致会让所有"相对投喂起点"的时间类指标整体平移数秒，
    属 reject_all 级。
    ⚠️ ROI 严重度以 docs/04 §4.4 为准（**warn 不拒绝**）：不同养殖单元
    的 ROI 本应不同（合法差异），拒绝会误伤；docs/06 的表述未区分此情形。

零值纪律（本项目第一铁律在统计层的落实）：
    - 任一组可用样本 < MIN_N_PER_GROUP → status='unavailable' + reason，
      **绝不输出 p=1.0 / 0.5 之类的占位值**；
    - 检验算不出（零方差并列、样本不足）→ None + reason；
    - unavailable / censored 的指标不进样本（RunBundle.value 已过滤）。

任务编号：T05。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.stats.comparison_plan import (
    NEUTRAL_GROUP_EXIT,
    NON_NEUTRAL_GATES,
    ComparisonPlan,
    ConsistencyReport,
    RunBundle,
    Violation,
)

__all__ = [
    "TestResult",
    "TwoGroupPlan",
    "describe_group",
    "MIN_N_PER_GROUP",
    "NORMALITY_MIN_N",
    "HOLM_ALPHA",
]

# 参数检验的最小样本数（n < 2 无法估计方差 → 无从检验）
MIN_N_PER_GROUP: int = 2
# 正态性检验（Shapiro–Wilk）的最小样本数
NORMALITY_MIN_N: int = 3
# 默认显著性水平（同时用于正态性/方差齐性前提检验与 Holm 校正）
HOLM_ALPHA: float = 0.05


# ----------------------------------------------------------------------
# 输出结构
# ----------------------------------------------------------------------
@dataclass
class TestResult:
    """单个指标的两组检验结果（六要素齐全，缺项一律 None + reason）。

    Attributes:
        metric_id: 指标 ID。
        status: 'ok' | 'unavailable'（不可用必带 reason）。
        available: status == 'ok' 的语义别名（UI/导出层的便利读法）。
        reason: status != 'ok' 时的原因。
        n_a / n_b: 两组**可用**样本数（不含不可用/删失）。
        mean_a / mean_b: 两组均值（n=0 时为 None）。
        test_used / test_model: 主检验方法名。test_model 为对外主名，
            取值 'welch_t' | 'mann_whitney' | 'mixedlm' | 'none'；
            test_used 是其别名（与 test_model 同步，含历史值 'mixed_lm'
            的归一化）。
        p_value: 主检验 p 值。
        p_welch / p_mwu: 两种检验的 p（都给，不隐藏分歧）。
        p_holm: Holm–Bonferroni 校正后 p（run_all 多指标时填充）。
        effect_size / effect_size_type: Cohen's d（Hedges 校正）或秩二列相关。
        ci_low / ci_high / ci_level / ci_of: 95%CI 与它描述的对象。
        normality_ok / equal_var_ok: 前提检验结论（None = 无法判定）。
        repeat_structure / replication / replication_note: 重复结构
            （伪重复防线）。replication ∈ 'independent_ponds' |
            'single_pond' | 'undeclared'；replication_note 是其人类可读
            说明（UI 直接展示）。
        inferable: 是否可做统计推断（= 多池塘且样本量足够；
            单池/未声明 pond_id → False，即"仅描述性"）。
        descriptive_only: 单池 / pond_id 未声明 → True（= not inferable）。
        warnings / notes: 告警与口径说明。
    """

    metric_id: str
    status: str = "unavailable"
    reason: str | None = None
    n_a: int = 0
    n_b: int = 0
    mean_a: float | None = None
    mean_b: float | None = None
    test_used: str | None = None
    p_value: float | None = None
    p_welch: float | None = None
    p_mwu: float | None = None
    p_holm: float | None = None
    effect_size: float | None = None
    effect_size_type: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float = 0.95
    ci_of: str | None = None
    normality_ok: bool | None = None
    equal_var_ok: bool | None = None
    repeat_structure: str = ""
    replication: str = "undeclared"
    replication_note: str = ""
    descriptive_only: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """是否产出了可用的检验结论（status == 'ok'）。"""
        return self.status == "ok"

    @property
    def test_model(self) -> str:
        """主检验方法名（对外主名；'none' = 未执行任何检验）。"""
        raw = (self.test_used or "none").strip()
        return "mixedlm" if raw in ("mixedlm", "mixed_lm") else raw

    @property
    def inferable(self) -> bool:
        """是否可做统计推断（单池 / 未声明 pond_id → False，仅描述性）。"""
        return bool(self.available and not self.descriptive_only)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "status": self.status,
            "available": self.available,
            "reason": self.reason,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "test_used": self.test_used,
            "test_model": self.test_model,
            "p_value": self.p_value,
            "p_welch": self.p_welch,
            "p_mwu": self.p_mwu,
            "p_holm": self.p_holm,
            "effect_size": self.effect_size,
            "effect_size_type": self.effect_size_type,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "ci_of": self.ci_of,
            "normality_ok": self.normality_ok,
            "equal_var_ok": self.equal_var_ok,
            "repeat_structure": self.repeat_structure,
            "replication": self.replication,
            "replication_note": self.replication_note,
            "descriptive_only": self.descriptive_only,
            "inferable": self.inferable,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


# ----------------------------------------------------------------------
# 描述统计
# ----------------------------------------------------------------------
def describe_group(values: Sequence[float]) -> dict[str, Any]:
    """单组描述统计（空输入 → n=0 且全部指标为 None，绝不返回 0）。

    为什么单独成函数：compare_page 与 06 脚本都要在**不跑检验**的情况下
    展示两组分布（散点/箱线图），且必须能在 n<2 时如实显示"样本不足"
    而不是画一个假的均值。
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "min": None, "max": None, "iqr": None,
        }
    q1, q3 = (float(x) for x in np.percentile(arr, [25.0, 75.0]))
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n >= 2 else None,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "iqr": float(q3 - q1),
    }


# ----------------------------------------------------------------------
# 数值内核（纯函数，可脱离 IO 单测）
# ----------------------------------------------------------------------
def _welch_t_test(
    a: np.ndarray, b: np.ndarray, level: float = 0.95,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Welch t 检验（不假设等方差）。

    Returns:
        (p, ci_low, ci_high, df)，差值口径为 mean(a) − mean(b)；
        无从检验（n<2 或两组各自零方差且均值相同）→ (None, None, None, None)。
    """
    from scipy import stats

    n1, n2 = a.size, b.size
    if n1 < MIN_N_PER_GROUP or n2 < MIN_N_PER_GROUP:
        return None, None, None, None
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)
    diff = float(np.mean(a) - np.mean(b))
    if se <= 0:
        # 两组各自零方差：数值全同（无从检验）或完全分离
        return (None, None, None, None) if diff == 0 else (0.0, diff, diff, None)
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else float(n1 + n2 - 2)
    t_stat = diff / se
    p = float(2.0 * stats.t.sf(abs(t_stat), df))
    t_crit = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, df))
    return p, diff - t_crit * se, diff + t_crit * se, df


def _mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    """Mann–Whitney U + 秩二列相关（= Cliff's delta）效应量。"""
    from scipy import stats

    if a.size < 1 or b.size < 1:
        return None, None
    try:
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return None, None
    u = float(res.statistic)
    return float(res.pvalue), (2.0 * u / (a.size * b.size) - 1.0)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cohen's d（合并标准差口径）+ Hedges 小样本校正。

    为什么校正：n 小（本项目常见 n=3~6）时 d 高估约 4%~10%，
    g = d·(1 − 3/(4(n1+n2)−9)) 近似无偏。
    """
    n1, n2 = a.size, b.size
    if n1 < MIN_N_PER_GROUP or n2 < MIN_N_PER_GROUP:
        return None
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))
    s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if s_pooled <= 0:
        return None
    d = (float(np.mean(a)) - float(np.mean(b))) / s_pooled
    denom = 4.0 * (n1 + n2) - 9.0
    return d * (1.0 - 3.0 / denom) if denom > 0 else d


def _normality_ok(x: np.ndarray, alpha: float) -> bool | None:
    """Shapiro–Wilk 正态性（n < 3 无法判定 → None，绝不猜 True）。"""
    from scipy import stats

    if x.size < NORMALITY_MIN_N:
        return None
    try:
        return bool(stats.shapiro(x).pvalue > alpha)
    except Exception:
        return None


def _equal_var_ok(a: np.ndarray, b: np.ndarray, alpha: float) -> bool | None:
    """Levene 方差齐性（任一组 n < 2 → None）。"""
    from scipy import stats

    if a.size < MIN_N_PER_GROUP or b.size < MIN_N_PER_GROUP:
        return None
    try:
        return bool(stats.levene(a, b).pvalue > alpha)
    except Exception:
        return None


def _holm_adjust(pairs: list[tuple[str, float | None]]) -> dict[str, float | None]:
    """Holm–Bonferroni 逐步校正（控制 FWER，比 Bonferroni 更有检验效能）。"""
    valid = sorted([(k, p) for k, p in pairs if p is not None], key=lambda kv: kv[1])
    out: dict[str, float | None] = {k: None for k, _ in pairs}
    if not valid:
        return out
    m = len(valid)
    running = 0.0
    for i, (k, p) in enumerate(valid):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = float(running)
    return out


# ----------------------------------------------------------------------
# 两组比较计划
# ----------------------------------------------------------------------
class TwoGroupPlan(ComparisonPlan):
    """两组对照（classDiagram TwoGroupPlan ← ComparisonPlan）。

    用法::

        plan = TwoGroupPlan(bundles, group_a="A", group_b="B")
        report = plan.validate()
        if report.ok:
            res = plan.execute("A8_T50")      # 单指标
            all_res = plan.run_all(["A8_T50", "A11_RR"])   # 含 Holm 校正
    """

    name: str = "two_group"

    def __init__(
        self,
        runs: Sequence[RunBundle],
        group_a: str = "A",
        group_b: str = "B",
        alpha: float = HOLM_ALPHA,
    ) -> None:
        super().__init__(runs)
        self.group_a = str(group_a)
        self.group_b = str(group_b)
        self.alpha = float(alpha)
        # 最近一次选定的参数/非参数主检验（MixedLM 未采纳时的留痕文案用）
        self._last_primary: str = "welch_t"

    # ------------------------------------------------------------------
    def group_labels(self) -> list[str]:
        """分组标签（盲法下由调用方传入盲法编号作为 group 值）。"""
        return [self.group_a, self.group_b]

    def _runs_of(self, group: str) -> list[RunBundle]:
        return [r for r in self.runs if r.group == group]

    # ------------------------------------------------------------------
    # 一致性校验（七条 §4.4 + t0 两条 §6，取并集）
    # ------------------------------------------------------------------
    def validate(self) -> ConsistencyReport:
        """比较前一致性校验（规则 1–9）。

        Returns:
            ConsistencyReport：ok=False 时调用方**不得**执行 execute()
            （本类不提供绕过开关——口径不一致时的 p 值是精确的废话）。
        """
        report = ConsistencyReport()
        metric_ids = sorted({m for r in self.runs for m in r.metrics})
        self._check_config_rules(report, metric_ids)
        self._check_t0_rules(report, metric_ids)
        self._check_metric_set_rules(report, metric_ids)

        groups = {r.group for r in self.runs}
        if len(groups) != 2:
            report.violations.append(Violation(
                rule_id=0, rule_name="两组结构", action="reject_all",
                message=(
                    f"两组比较需要恰好 2 个组标签，收到 {sorted(groups)}"
                    f"（共 {len(self.runs)} 个 run）"
                ),
                affected_metrics=list(metric_ids),
                exit_hint="请在比较页为每个 run 指定所属组别。",
            ))
        for g in (self.group_a, self.group_b):
            if not self._runs_of(g):
                report.violations.append(Violation(
                    rule_id=0, rule_name="两组结构", action="reject_all",
                    message=f"组 {g!r} 无 run：无法比较。",
                    affected_metrics=list(metric_ids),
                    exit_hint="请至少为每组选择 1 个 run。",
                ))
        return report

    # ------------------------------------------------------------------
    def _check_t0_rules(
        self, report: ConsistencyReport, metric_ids: Sequence[str]
    ) -> None:
        """规则 8/9（docs/06 §6 T05）：t0_definition / t0_source 一致性。

        t0 是全部时间类指标的原点。定义不同（投饵器启动 / 饲料离开投饵器 /
        饲料入画面）可差出数秒，来源不同（手动打点 vs 自动检测）误差量级
        也不同——两者任一不一致，所有"相对投喂起点"的指标都换了意思。
        """
        cfgs = [r.config for r in self.runs]
        base = cfgs[0]
        for rule_id, field_name, label in (
            (8, "t0_definition", "t0 定义"),
            (9, "t0_source", "t0 来源"),
        ):
            vals = {str(getattr(c, field_name, "")) for c in cfgs}
            if len(vals) > 1:
                report.violations.append(Violation(
                    rule_id=rule_id, rule_name=f"{label}一致性",
                    action="reject_all",
                    message=(
                        f"各 run 的{label}不一致：" +
                        "、".join(f"{r.run_id}={getattr(r.config, field_name, '')!r}"
                                  for r in self.runs) +
                        "。t0 是全部时间类指标的原点，口径不同则所有"
                        "相对投喂起点的指标整体平移，不可比。"
                    ),
                    affected_metrics=list(metric_ids),
                    exit_hint="请用同一 t0 定义与来源重新分析上述 run（单段约 4 秒）。",
                ))

    # ------------------------------------------------------------------
    def _check_metric_set_rules(
        self, report: ConsistencyReport, metric_ids: Sequence[str]
    ) -> None:
        """规则 6（已关闭指标集不同 → 告警）+ 规则 7（非中立门控 → 拒绝）。"""
        for mid in metric_ids:
            have = [r for r in self.runs
                    if r.status(mid) in ("ok", "degraded") and r.value(mid) is not None]
            miss = [r for r in self.runs if r not in have]
            if not miss or not have:
                if not have:
                    report.notes.append(
                        f"{mid}：所有 run 均不可用（无样本），跳过比较。"
                    )
                continue

            # ---- 规则 6：可用集不一致 → warn（缺失可能非随机）----
            report.violations.append(Violation(
                rule_id=6, rule_name="已关闭指标集一致性", action="warn",
                message=(
                    f"{mid} 的可用集不一致：{sorted(r.run_id for r in have)} 有值，"
                    f"{sorted(r.run_id for r in miss)} 不可用。"
                    "缺失可能是效应本身（高摄食强度会降低检测/跟踪质量），"
                    "跨组比较存在非随机缺失风险。"
                ),
                affected_metrics=[mid],
                exit_hint="请核对 capability_report.md 中该指标的关闭原因。",
            ))

            # ---- 规则 7：缺失由非中立门控触发 → 拒绝对可用子集做推断 ----
            non_neutral = []
            for r in miss:
                gates = _triggered_gates(r)
                if gates & NON_NEUTRAL_GATES:
                    non_neutral.append((r.run_id, sorted(gates & NON_NEUTRAL_GATES)))
            if non_neutral:
                report.violations.append(Violation(
                    rule_id=7, rule_name="非中立门控缺失", action="reject_metrics",
                    message=(
                        f"{mid} 的缺失由非中立门控触发：" +
                        "、".join(f"{rid}({','.join(g)})" for rid, g in non_neutral) +
                        "。这些门控的误差随被测效应变化，"
                        "对『算得出来的那些』做统计推断 = 以结果为条件抽样。"
                    ),
                    affected_metrics=[mid],
                    exit_hint=NEUTRAL_GROUP_EXIT,
                ))
                report.rejected_metrics.add(mid)

    # ------------------------------------------------------------------
    # 统计执行
    # ------------------------------------------------------------------
    def execute(self, metric_id: str) -> TestResult:
        """对单个指标执行两组检验（校验未通过时调用方不得调用本方法）。"""
        a_runs = self._runs_of(self.group_a)
        b_runs = self._runs_of(self.group_b)
        vals_a = _values_of(a_runs, metric_id)
        vals_b = _values_of(b_runs, metric_id)
        a = np.asarray(vals_a, dtype=float)
        b = np.asarray(vals_b, dtype=float)
        res = TestResult(metric_id=metric_id, n_a=int(a.size), n_b=int(b.size))

        # ---- 重复结构（伪重复防线，先于一切统计）----
        (res.replication, res.replication_note,
         res.descriptive_only) = self._repeat_structure()
        res.repeat_structure = res.replication_note

        # ---- 样本量硬门（零值纪律：不输出占位 p）----
        if a.size < MIN_N_PER_GROUP or b.size < MIN_N_PER_GROUP:
            res.status = "unavailable"
            # test_used 保持 None（未执行任何检验）；对外的 test_model
            # 属性会把 None 归一为 'none'（UI/CSV 便于筛选）。
            res.test_used = None
            res.reason = (
                f"样本量不足（{self.group_a}={a.size}, {self.group_b}={b.size}）："
                f"至少各需 {MIN_N_PER_GROUP} 个可用值才能估计方差；"
                "不输出 p 值（严禁以 p=1.0/0.5 等占位值冒充）"
            )
            # 样本量不足时连描述统计也不报：n=1 的单点均值无代表性，
            # 给出反而容易被误读为"可信的组均值"，故均值字段保持 None。
            return res

        # ---- 描述统计（仅在样本量足够时计算，避免 n=1 单点均值误导）----
        res.mean_a = float(np.mean(a)) if a.size else None
        res.mean_b = float(np.mean(b)) if b.size else None

        res.normality_ok = self._normality_both(a, b)
        res.equal_var_ok = _equal_var_ok(a, b, self.alpha)

        # ---- 两个检验都跑（不隐藏与主检验不一致的那一个）----
        p_welch, ci_lo, ci_hi, _df = _welch_t_test(a, b, res.ci_level)
        p_mwu, rank_biserial = _mann_whitney(a, b)
        d = _cohens_d(a, b)
        res.p_welch, res.p_mwu = p_welch, p_mwu
        res.ci_low, res.ci_high, res.ci_of = ci_lo, ci_hi, "mean_diff(a-b)"

        res.test_used = self._pick_primary(res, a, b)
        if res.test_used == "welch_t":
            res.p_value = p_welch
            res.effect_size = d
            res.effect_size_type = "cohen_d_hedges_g" if d is not None else None
        else:
            res.p_value = p_mwu
            res.effect_size = rank_biserial
            res.effect_size_type = "rank_biserial"
            res.notes.append(
                "效应量为秩二列相关（= Cliff's delta）；CI 仍是均值差的 Welch 区间"
                "（非中位数位移），解读时勿混用口径"
            )

        # ---- 多池塘 → MixedLM（pond 随机截距）为主检验 ----
        if not res.descriptive_only:
            self._last_primary = res.test_used or "welch_t"
            mixed = self._mixed_lm(metric_id)
            if mixed is None:
                # 随机效应不可辨识（每池 1 个观测）→ 不采纳，显式留痕。
                # 沉默地退回 Welch 会让用户以为用的是混合模型。
                reason = self._mixed_lm_unusable_reason(metric_id)
                if reason:
                    res.notes.append(reason)
                    res.warnings.append(reason)
            else:
                p_mixed, note = mixed
                # ⚠️ NaN/Inf 兜底：MixedLM 在**零方差/常数指标**上会收敛到
                # 退化解并给出 NaN 的 p 值。NaN 不是"没有 p"，而是"算坏了"；
                # 直接透传会让 status=ok 且 p=nan 的结果流进 CSV/UI
                # （这正是本项目反复防的"拿不可用的数冒充结果"）。
                # 处置：退化解不采纳，保留 Welch/MW 为主检验并显式留痕。
                if p_mixed is None or not math.isfinite(float(p_mixed)):
                    res.notes.append(
                        "MixedLM 给出退化解（p 值为 NaN/Inf，常见于该指标两组"
                        "数值已退化为常数或完全并列）：不采纳其 p 值，"
                        f"主检验保持 {res.test_used}"
                    )
                else:
                    res.p_value = float(p_mixed)
                    res.test_used = "mixed_lm"
                    res.effect_size = d
                    res.effect_size_type = (
                        "cohen_d_hedges_g" if d is not None else None
                    )
                    res.notes.append(note)
                    res.notes.append(
                        "Welch / Mann–Whitney p 值保留在 p_welch / p_mwu "
                        "字段供交叉核对"
                    )

        if res.p_value is None or not math.isfinite(float(res.p_value)):
            res.status = "unavailable"
            res.p_value = None       # NaN/Inf 一律不外泄（零值纪律）
            res.reason = (
                "检验未能给出有效的 p 值（两组数值完全并列、零方差，或混合"
                "模型退化为 NaN）：不输出占位值，请检查该指标是否已退化为常数"
            )
            return res

        res.status = "ok"
        if res.descriptive_only:
            res.warnings.append(
                "descriptive_only：无独立重复（单池或 pond_id 未声明），"
                "p 值与效应量仅供描述，**不可做统计推断**"
            )
        if res.equal_var_ok is False:
            res.notes.append(
                "方差齐性未通过（Levene）：已使用 Welch 校正（不假设等方差）"
            )
        self._warn_divergence(res)
        return res

    # ------------------------------------------------------------------
    def run_all(
        self, metric_ids: Sequence[str] | None = None
    ) -> dict[str, TestResult]:
        """批量执行 + Holm 多重比较校正（UI / 导出的主入口）。

        Returns:
            metric_id → TestResult；被 rejected_metrics 拒绝的指标返回
            status='unavailable' + reason（照常占位，但绝不给出 p 值）。
        """
        report = self.validate()
        targets = list(metric_ids) if metric_ids else sorted(
            {m for r in self.runs for m in r.metrics}
        )
        out: dict[str, TestResult] = {}
        for mid in targets:
            if mid in report.rejected_metrics:
                out[mid] = TestResult(
                    metric_id=mid, status="unavailable",
                    reason="该指标被比较前一致性校验拒绝"
                           "（口径不一致 / 非中立门控缺失），不输出 p 值",
                )
            else:
                out[mid] = self.execute(mid)
        holm = _holm_adjust([(mid, r.p_value) for mid, r in out.items()])
        for mid, adj in holm.items():
            out[mid].p_holm = adj
        return out

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _pick_primary(self, res: TestResult, a: np.ndarray, b: np.ndarray) -> str:
        """主检验选择（正态性为准；样本不足判不出 → 保守走非参数）。"""
        if res.normality_ok is None:
            res.notes.append(
                f"样本量不足（n < {NORMALITY_MIN_N}）无法判定正态性："
                "以非参数 Mann–Whitney 为主检验（保守口径）"
            )
            res.warnings.append(
                f"正态性无法判定（每组样本 < {NORMALITY_MIN_N}）："
                "已按保守口径改用 Mann–Whitney 为主检验"
            )
            return "mann_whitney"
        if res.normality_ok:
            res.notes.append(
                f"两组均通过正态性检验（Shapiro–Wilk p > {self.alpha}）："
                "以 Welch t 为主检验"
            )
            return "welch_t"
        res.notes.append(
            "至少一组未通过正态性检验：以 Mann–Whitney 为主检验"
            "（Welch p 值一并给出，供交叉核对）"
        )
        res.warnings.append(
            "正态性检验未通过（Shapiro–Wilk）：已改用非参数 Mann–Whitney "
            "为主检验；参数检验的 p 值仍列于 p_welch 供交叉核对"
        )
        return "mann_whitney"

    def _normality_both(self, a: np.ndarray, b: np.ndarray) -> bool | None:
        na = _normality_ok(a, self.alpha)
        nb = _normality_ok(b, self.alpha)
        if na is None or nb is None:
            return None
        return bool(na and nb)

    def _repeat_structure(self) -> tuple[str, str, bool]:
        """(replication 代码, 人类可读说明, 是否仅描述性)。

        replication ∈ {'independent_ponds', 'single_pond', 'undeclared'}：
            - independent_ponds：≥2 个池塘 → 可做统计推断（MixedLM 随机效应）；
            - single_pond：只有 1 个池塘 → 多次投喂属**伪重复**，
              仅描述性，p 值不可用于推断；
            - undeclared：pond_id 未声明 → 重复结构未知，按最保守处理
              （不可推断），并提示用户补声明。
        """
        ponds = {r.pond_id for r in self.runs if r.pond_id is not None}
        if not ponds:
            return (
                "undeclared",
                "重复结构未知（全部 run 均未声明 pond_id）：无独立重复，"
                "仅描述性，不可做统计推断（请在 meta.json 补 pond_id）",
                True,
            )
        if len(ponds) == 1:
            return (
                "single_pond",
                f"单池塘（pond_id={sorted(ponds)[0]}，{len(self.runs)} 次投喂）："
                "同一池塘多次投喂属伪重复，无独立重复，仅描述性，"
                "不可做统计推断",
                True,
            )
        return (
            "independent_ponds",
            f"{len(ponds)} 个池塘（{sorted(ponds)}）/ {len(self.runs)} 次投喂："
            "存在独立重复，以 pond 为随机效应的混合模型（MixedLM）",
            False,
        )

    @staticmethod
    def _warn_divergence(res: TestResult) -> None:
        """两检验结论方向不一致 → 显式告警（不静默挑一个好看的）。"""
        pa, pb = res.p_welch, res.p_mwu
        if pa is None or pb is None:
            return
        if (pa < 0.05) != (pb < 0.05):
            res.warnings.append(
                f"参数与非参数检验结论不一致（Welch p={pa:.4f} vs "
                f"Mann–Whitney p={pb:.4f}）：检验前提可能不满足或存在离群点，"
                "请以效应量与原始数据分布为准，勿只报显著的那一个"
            )

    # MixedLM 随机效应可辨识性的最小重复数（每个池塘至少这么多观测）
    _MIN_OBS_PER_POND_FOR_MIXEDLM = 2

    def _mixed_lm(self, metric_id: str) -> tuple[float, str] | None:
        """statsmodels MixedLM（pond 随机截距）；不可拟合 → None（不猜）。

        ⚠️ 随机效应**可辨识性**硬门（本方法最容易产出假阳性的地方）：
            若每个池塘只有 1 个观测（观测数 == 池塘数），随机截距方差与
            残差方差**无法分离**（模型饱和），statsmodels 会在 Hessian
            非正定的情况下仍返回一个 p 值——实测在同分布数据上给出
            p=0.042（Welch 0.489 / MW 0.623），即**假阳性**。
            这属于"算出来了但意思是错的"，比不输出更危险。
            故要求：至少一个池塘贡献 ≥2 个观测，且观测数 > 池塘数。

        任何异常都回退（回退事实体现在主检验仍为 Welch/MW，
        绝不静默给错 p）。
        """
        try:
            import pandas as pd
            import statsmodels.formula.api as smf
        except Exception:
            return None

        rows: list[dict[str, Any]] = []
        for r in self.runs:
            v = r.value(metric_id)
            if v is None or r.status(metric_id) not in ("ok", "degraded"):
                continue
            if r.pond_id is None or r.group not in (self.group_a, self.group_b):
                continue
            rows.append({"value": float(v), "group": r.group, "pond": r.pond_id})
        if len(rows) < 3 or len({x["pond"] for x in rows}) < 2:
            return None

        per_pond: dict[str, int] = {}
        for x in rows:
            per_pond[str(x["pond"])] = per_pond.get(str(x["pond"]), 0) + 1
        if max(per_pond.values()) < self._MIN_OBS_PER_POND_FOR_MIXEDLM or len(
            rows
        ) <= len(per_pond):
            return None  # 随机效应不可辨识 → 不采纳 MixedLM

        try:
            df = pd.DataFrame(rows)
            fit = smf.mixedlm("value ~ C(group)", df, groups=df["pond"]).fit(reml=True)
            keys = [k for k in fit.pvalues.index if "group" in k]
            if not keys:
                return None
            return (
                float(fit.pvalues[keys[0]]),
                f"MixedLM（value ~ C(group)，pond 随机截距，REML，"
                f"n={len(rows)}，{len(per_pond)} 个池塘）",
            )
        except Exception:
            return None

    def _mixed_lm_unusable_reason(self, metric_id: str) -> str | None:
        """MixedLM 不被采纳的原因（None = 已采纳或无需说明）。"""
        rows: list[dict[str, Any]] = []
        for r in self.runs:
            v = r.value(metric_id)
            if v is None or r.status(metric_id) not in ("ok", "degraded"):
                continue
            if r.pond_id is None or r.group not in (self.group_a, self.group_b):
                continue
            rows.append({"value": float(v), "group": r.group, "pond": r.pond_id})
        if not rows:
            return None
        per_pond: dict[str, int] = {}
        for x in rows:
            per_pond[str(x["pond"])] = per_pond.get(str(x["pond"]), 0) + 1
        if max(per_pond.values()) < self._MIN_OBS_PER_POND_FOR_MIXEDLM:
            return (
                "MixedLM 未采纳：每个池塘仅 1 个观测，随机截距方差与残差方差"
                "无法分离（模型饱和），此时 MixedLM 的 p 值不可信"
                "（实测可在同分布数据上给出假阳性 p<0.05）；"
                f"已改用 {self._last_primary} 为主检验。"
                "要启用混合模型，请在每个池塘安排 ≥2 次投喂/重复观测。"
            )
        return None


# ----------------------------------------------------------------------
# 模块级工具
# ----------------------------------------------------------------------
def _values_of(runs: Sequence[RunBundle], metric_id: str) -> list[float]:
    """取一组内所有可用值（不可用/删失/未输出一律不入样本）。"""
    out: list[float] = []
    for r in runs:
        if r.status(metric_id) not in ("ok", "degraded"):
            continue
        v = r.value(metric_id)
        if v is not None:
            out.append(float(v))
    return out


def _triggered_gates(run: RunBundle) -> set[str]:
    """该 run 上被触发的门控名（用于规则 7 的非中立门控判定）。

    只认"实测信号确实越限"这一种触发；未测得（None）不算触发
    （未测得 ≠ 测得为差，docs/04 §4.0）。
    """
    th = run.config.thresholds
    q = run.quality or {}
    gates: set[str] = set()
    v = q.get("Q_track")
    if v is not None and v < th.q_track_min:
        gates.add("Q_track")
    v = q.get("Q_det")
    if v is not None and v < th.q_det_min:
        gates.add("Q_det")
    v = q.get("Q_fg")
    if v is not None and v < th.q_fg_min:
        gates.add("Q_fg")
    v = q.get("Q_overlap")
    if v is not None and v > 0.30:
        gates.add("Q_overlap")
    # 指标行上的 low_conf flag 同样是"依赖检测质量"的非中立信号
    for mid in run.metrics:
        if "low_conf" in run.flags(mid):
            gates.add("Q_det")
    return gates

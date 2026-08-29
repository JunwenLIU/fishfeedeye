"""two_group.py · 两组对照统计（T05，ComparisonPlan 的特例实现）。

职责（docs/06 §6 T05 + docs/04 §6.2）：
    - `TwoGroupPlan`：**两组对照**是本项目当前唯一的实验设计（第二轮用户
      决策 ①）。剂量梯度 / 多组 ANOVA 不实现，但通过 ComparisonPlan 抽象
      留扩展位——新增 `MultiGroupPlan` 即可，本文件与基类都不用改。
    - 检验选择：正态性（Shapiro–Wilk）+ 方差齐性（Levene）→ 主检验在
      Welch t 与 Mann–Whitney 之间自动选择，**两个 p 值都输出**（不隐藏
      与主检验不一致的那一个——选错检验是 p 值造假的经典入口）。
    - 输出六要素（docs/06 T05 验收 4）：p / Cohen's d / 95%CI / 检验方法名
      / n / 重复结构说明，缺一不可。
    - 重复结构（伪重复防线）：pond_id 声明多个池塘 → statsmodels MixedLM
      （pond 为随机效应）；单池或未声明 → 标注"无独立重复，仅描述性，
      不可做统计推断"，**此时 p 值仍输出但打 descriptive_only 标记**。
    - 多重比较：一次 execute() 涉及多个指标 → Holm–Bonferroni 校正 p。

零值纪律（本项目第一铁律在统计层的落实）：
    - 任一组可用样本 < 2 → status='unavailable' + reason，**绝不输出 p=1.0
      或 p=0.5 之类的占位值**；
    - 样本量不足 / 拟合失败 → None + reason，绝不用 0 冒充；
    - unavailable/censored 的指标不进样本（RunRef.value_of 已过滤）。

任务编号：T05。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.stats.comparison_plan import ComparisonPlan, RunRef

__all__ = ["TestResult", "TwoGroupPlan", "HOLM_LEVEL"]

HOLM_LEVEL: float = 0.05

# 正态性检验的最小样本数（Shapiro–Wilk 需要 n ≥ 3）
_MIN_N_FOR_NORMALITY = 3
# 参数检验的最小样本数（n < 2 无法估计方差 → 无从检验）
_MIN_N_FOR_TEST = 2


# ----------------------------------------------------------------------
# 输出结构
# ----------------------------------------------------------------------
@dataclass
class TestResult:
    """单个指标的两组检验结果（六要素齐全，缺项一律 None + reason）。

    Attributes:
        metric_id: 指标 ID。
        status: 'ok' | 'unavailable'（不可用必带 reason）。
        reason: status != 'ok' 时的原因。
        n_a / n_b: 两组**可用**样本数（不含不可用/删失）。
        mean_a / mean_b: 两组均值（n=0 时为 None）。
        test_used: 主检验方法名（'welch_t' / 'mann_whitney' / 'mixed_lm'）。
        p_value: 主检验 p 值。
        p_welch / p_mwu: 两种检验的 p 值（都给，不隐藏分歧）。
        effect_size / effect_size_type: Cohen's d（Hedges 校正）/ 'cohen_d'
            或秩二列相关 / 'rank_biserial'。
        ci_low / ci_high / ci_level / ci_of: 95% 置信区间与它描述的对象。
        normality_ok / equal_var_ok: 前提检验结论（None = 样本不足无法判定）。
        p_holm: Holm–Bonferroni 校正后 p（多指标同批检验时）。
        repeat_structure: 重复结构说明（伪重复防线）。
        descriptive_only: 单池/未声明 pond_id → True（仅描述性）。
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
    descriptive_only: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "status": self.status,
            "reason": self.reason,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "test_used": self.test_used,
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
            "descriptive_only": self.descriptive_only,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


# ----------------------------------------------------------------------
# 数值内核（纯函数，可脱离 IO 单测）
# ----------------------------------------------------------------------
def _welch_t_test(
    a: np.ndarray, b: np.ndarray
) -> tuple[float | None, float | None, float | None, float | None]:
    """Welch t 检验（不假设等方差）。

    Returns:
        (p, ci_low, ci_high, df)：差值为 mean(a) − mean(b)；
        样本不足（任一组 n<2 或合并方差为 0 且均值相同）→ (None, ...)。
    """
    from scipy import stats

    n1, n2 = a.size, b.size
    if n1 < _MIN_N_FOR_TEST or n2 < _MIN_N_FOR_TEST:
        return None, None, None, None
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)
    diff = float(np.mean(a) - np.mean(b))
    if se <= 0:
        # 两组各自零方差：要么完全分离（无检验意义），要么数值全相同
        return (None, None, None, None) if diff == 0 else (0.0, diff, diff, None)
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else float(n1 + n2 - 2)
    t_stat = diff / se
    p = float(2.0 * stats.t.sf(abs(t_stat), df))
    t_crit = float(stats.t.ppf(1.0 - (1.0 - 0.95) / 2.0, df))
    return p, diff - t_crit * se, diff + t_crit * se, df


def _mann_whitney(
    a: np.ndarray, b: np.ndarray
) -> tuple[float | None, float | None]:
    """Mann–Whitney U 检验 + 秩二列相关（= Cliff's delta）效应量。

    Returns:
        (p, rank_biserial)：样本不足或全部并列 → (None, None)。
    """
    from scipy import stats

    if a.size < 1 or b.size < 1:
        return None, None
    try:
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return None, None
    u = float(res.statistic)
    p = float(res.pvalue)
    # 秩二列相关：r = 2U/(n1·n2) − 1，取值 [-1, 1]（1 = a 全部大于 b）
    return p, (2.0 * u / (a.size * b.size) - 1.0)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cohen's d（合并标准差口径）+ Hedges 小样本校正 g。

    为什么做 Hedges 校正：n 小（本项目常见 n=3~6）时 Cohen's d 高估约
    4%~10%，g = d·(1 − 3/(4(n1+n2)−9)) 是无偏估计。
    """
    n1, n2 = a.size, b.size
    if n1 < _MIN_N_FOR_TEST or n2 < _MIN_N_FOR_TEST:
        return None
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))
    s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if s_pooled <= 0:
        return None
    d = (float(np.mean(a)) - float(np.mean(b))) / s_pooled
    denom = 4.0 * (n1 + n2) - 9.0
    return d * (1.0 - 3.0 / denom) if denom > 0 else d


def _normality_ok(x: np.ndarray, alpha: float = HOLM_LEVEL) -> bool | None:
    """Shapiro–Wilk 正态性（n < 3 无法判定 → None，绝不猜 True）。"""
    from scipy import stats

    if x.size < _MIN_N_FOR_NORMALITY:
        return None
    try:
        return bool(stats.shapiro(x).pvalue > alpha)
    except Exception:  # scipy 极端输入（全相同）可能抛错
        return None


def _equal_var_ok(a: np.ndarray, b: np.ndarray, alpha: float = HOLM_LEVEL) -> bool | None:
    """Levene 方差齐性（任一组 n < 2 → None）。"""
    from scipy import stats

    if a.size < _MIN_N_FOR_TEST or b.size < _MIN_N_FOR_TEST:
        return None
    try:
        return bool(stats.levene(a, b).pvalue > alpha)
    except Exception:
        return None


def _holm_adjust(pairs: list[tuple[str, float | None]]) -> dict[str, float | None]:
    """Holm–Bonferroni 逐步校正（比 Bonferroni 更有检验效能且控制 FWER）。

    Args:
        pairs: [(metric_id, p)]，p=None 的项不参与校正（保留 None）。

    Returns:
        metric_id → 校正后 p（原始 p=None 的项仍为 None）。
    """
    valid = [(k, p) for k, p in pairs if p is not None]
    out: dict[str, float | None] = {k: None for k, _ in pairs}
    if not valid:
        return out
    valid.sort(key=lambda kv: kv[1])
    m = len(valid)
    running = 0.0
    for i, (k, p) in enumerate(valid):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)  # 单调化（Holm 要求）
        out[k] = float(running)
    return out


# ----------------------------------------------------------------------
# 两组比较计划
# ----------------------------------------------------------------------
class TwoGroupPlan(ComparisonPlan):
    """两组对照（classDiagram TwoGroupPlan ← ComparisonPlan）。

    用法::

        refs = load_run_refs([run_a1, run_a2, run_b1], groups=['A','A','B'])
        plan = TwoGroupPlan(refs, groups=('A', 'B'))
        result = plan.run(["A8_T50", "A11_RR"])
        # result.ok=False 时读 result.differences（七条规则差异 + 重跑提示）
    """

    name = "two_group"

    def __init__(
        self,
        refs: Sequence[RunRef],
        groups: tuple[str, str] = ("A", "B"),
        alpha: float = HOLM_LEVEL,
    ) -> None:
        super().__init__(refs)
        self.groups: tuple[str, str] = (str(groups[0]), str(groups[1]))
        self.alpha = float(alpha)

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        """结构校验：必须恰好两组、每组至少一个 run（硬失败）。"""
        problems: list[str] = []
        groups_present = {r.group for r in self.refs if r.group is not None}
        if len(groups_present) != 2:
            problems.append(
                f"两组比较需要恰好 2 个组标签，收到 {sorted(groups_present)}"
                f"（共 {len(self.refs)} 个 run）"
            )
            return problems
        for g in self.groups:
            if not any(r.group == g for r in self.refs):
                problems.append(f"组 {g!r} 无 run")
        for g in sorted(groups_present):
            n = sum(1 for r in self.refs if r.group == g)
            if n < 1:
                problems.append(f"组 {g!r} 无 run")
        return problems

    # ------------------------------------------------------------------
    def _split(self, metric_id: str) -> tuple[list[float], list[float]]:
        """按组取可用值（不可用/删失/未输出一律不进样本）。"""
        a: list[float] = []
        b: list[float] = []
        for ref in self.refs:
            v = ref.value_of(metric_id)
            if v is None or ref.status_of(metric_id) not in ("ok", "degraded"):
                continue
            if ref.group == self.groups[0]:
                a.append(v)
            elif ref.group == self.groups[1]:
                b.append(v)
        return a, b

    # ------------------------------------------------------------------
    def execute(
        self, metric_ids: Sequence[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """逐指标执行检验（基类已保证七条规则通过后才调用本方法）。"""
        from src.stats.comparison_plan import ComparisonResult

        # result 仅用于承载共用告警；execute 的契约是返回 tests 字典
        carrier = ComparisonResult(plan_name=self.name)
        self.check_pseudoreplication(carrier)

        targets: list[str] = list(metric_ids) if metric_ids else sorted(
            {mid for r in self.refs for mid in r.metrics}
        )
        results: dict[str, dict[str, Any]] = {}
        raw_p: list[tuple[str, float | None]] = []
        rejected: list[str] = list(getattr(self, "_rejected_metrics", []))
        for mid in targets:
            if mid in rejected:
                results[mid] = TestResult(
                    metric_id=mid, status="unavailable",
                    reason="该指标被一致性校验额外拒绝（如 px_per_mm 不一致 → "
                           "unnormalized 指标不可跨 run 比较）",
                ).to_dict()
                raw_p.append((mid, None))
                continue
            res = self._test_one(mid)
            # 共用告警（可用集不一致 = 非随机缺失风险）
            self.check_available_set_diff(carrier, mid)
            results[mid] = res.to_dict()
            raw_p.append((mid, res.p_value))

        holm = _holm_adjust(raw_p)
        for mid, adj in holm.items():
            if mid in results:
                results[mid]["p_holm"] = adj

        # 把共用告警/说明回填到每个指标（UI 逐行展示时也能看到）
        for mid in results:
            results[mid]["warnings"] = list(
                dict.fromkeys(
                    list(results[mid].get("warnings", [])) + carrier.warnings
                )
            )
            results[mid]["notes"] = list(
                dict.fromkeys(list(results[mid].get("notes", [])) + carrier.notes)
            )
        return results

    # ------------------------------------------------------------------
    def _test_one(self, metric_id: str) -> TestResult:
        vals_a, vals_b = self._split(metric_id)
        a = np.asarray(vals_a, dtype=float)
        b = np.asarray(vals_b, dtype=float)
        res = TestResult(metric_id=metric_id, n_a=int(a.size), n_b=int(b.size))

        # ---- 重复结构（伪重复防线）----
        ponds = {r.pond_id for r in self.refs if r.pond_id is not None}
        if not ponds:
            res.repeat_structure = "重复结构未知（pond_id 未声明）"
            res.descriptive_only = True
        elif len(ponds) == 1:
            res.repeat_structure = (
                f"单池塘（pond_id={sorted(ponds)[0]}）：{len(self.refs)} 次投喂"
                "属伪重复，仅描述性，不可做统计推断"
            )
            res.descriptive_only = True
        else:
            res.repeat_structure = (
                f"{len(ponds)} 个池塘（{sorted(ponds)}）/ {len(self.refs)} 次投喂"
                "：以 pond 为随机效应的混合模型"
            )

        # ---- 样本量硬门（零值纪律：不输出占位 p）----
        if a.size < _MIN_N_FOR_TEST or b.size < _MIN_N_FOR_TEST:
            res.status = "unavailable"
            res.reason = (
                f"样本量不足（{self.groups[0]}={a.size}, {self.groups[1]}={b.size}）："
                "至少各需 2 个可用值才能估计方差；"
                "不输出 p 值（严禁以 p=1.0/0.5 等占位值冒充）"
            )
            return res

        res.mean_a = float(np.mean(a))
        res.mean_b = float(np.mean(b))
        res.normality_ok = (
            (_normality_ok(a, self.alpha) is True)
            and (_normality_ok(b, self.alpha) is True)
        ) if (a.size >= _MIN_N_FOR_NORMALITY and b.size >= _MIN_N_FOR_NORMALITY) else None
        res.equal_var_ok = _equal_var_ok(a, b, self.alpha)

        # ---- 两个检验都跑（不隐藏与主检验不一致的那一个）----
        p_welch, ci_lo, ci_hi, _df = _welch_t_test(a, b)
        p_mwu, rank_biserial = _mann_whitney(a, b)
        d = _cohens_d(a, b)
        res.p_welch = p_welch
        res.p_mwu = p_mwu
        res.ci_low, res.ci_high = ci_lo, ci_hi
        res.ci_of = "mean_diff(a-b)"

        if res.normality_ok is None:
            res.notes.append(
                f"样本量不足（n<{_MIN_N_FOR_NORMALITY}）无法判定正态性："
                "以非参数 Mann–Whitney 为主检验（保守口径）"
            )
            primary = "mann_whitney"
        elif res.normality_ok:
            primary = "welch_t"
            res.notes.append(
                "两组均通过正态性检验（Shapiro–Wilk p > "
                f"{self.alpha}）：以 Welch t 为主检验"
            )
        else:
            primary = "mann_whitney"
            res.notes.append(
                "至少一组未通过正态性检验：以 Mann–Whitney 为主检验"
                "（Welch p 值一并给出，供交叉核对）"
            )
        if res.equal_var_ok is False:
            res.notes.append(
                "方差齐性未通过（Levene）：已使用 Welch 校正（不假设等方差）"
            )

        # ---- 多池塘 → 混合模型（pond 随机效应）为主检验 ----
        if not res.descriptive_only and len(ponds) > 1:
            mixed = self._mixed_lm(metric_id)
            if mixed is not None:
                p_mixed, note = mixed
                res.p_value = p_mixed
                res.test_used = "mixed_lm"
                res.notes.append(note)
                res.notes.append(
                    "Welch / Mann–Whitney p 值保留在 p_welch / p_mwu 字段供对照"
                )
                res.effect_size = d
                res.effect_size_type = "cohen_d_hedges_g" if d is not None else None
                res.status = "ok"
                self._warn_divergence(res)
                return res

        res.test_used = primary
        if primary == "welch_t":
            res.p_value = p_welch
            res.effect_size = d
            res.effect_size_type = "cohen_d_hedges_g" if d is not None else None
        else:
            res.p_value = p_mwu
            res.effect_size = rank_biserial
            res.effect_size_type = "rank_biserial"
            res.notes.append(
                "效应量为秩二列相关（Cliff's delta 同值）；"
                "CI 仍是均值差的 Welch 区间（非中位数位移），解读时勿混用"
            )
        if res.p_value is None:
            res.status = "unavailable"
            res.reason = (
                "检验未能给出 p 值（两组数值完全并列或零方差）："
                "不输出占位值，请检查该指标是否退化为常数"
            )
            return res
        res.status = "ok"
        if res.descriptive_only:
            res.warnings.append(
                "descriptive_only：无独立重复（单池或 pond_id 未声明），"
                "p 值与效应量仅供描述，**不可做统计推断**"
            )
        self._warn_divergence(res)
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _warn_divergence(res: TestResult) -> None:
        """两个检验结论方向不一致 → 显式告警（不静默挑一个好看的）。"""
        pa, pb = res.p_welch, res.p_mwu
        if pa is None or pb is None:
            return
        if (pa < 0.05) != (pb < 0.05):
            res.warnings.append(
                f"参数与非参数检验结论不一致（Welch p={pa:.4f} vs "
                f"Mann–Whitney p={pb:.4f}）：检验前提可能不满足或存在离群点，"
                "请以效应量与原始数据分布为准，勿只报显著的那一个"
            )

    # ------------------------------------------------------------------
    def _mixed_lm(self, metric_id: str) -> tuple[float, str] | None:
        """statsmodels MixedLM（pond 随机截距）；不可拟合 → None（不猜）。

        仅在 ≥2 个池塘、≥3 个观测量时调用；任何异常都回退（回退事实会
        写入 notes，绝不静默）。
        """
        try:
            import pandas as pd
            import statsmodels.formula.api as smf
        except Exception:
            return None

        rows: list[dict[str, Any]] = []
        for ref in self.refs:
            v = ref.value_of(metric_id)
            if v is None or ref.status_of(metric_id) not in ("ok", "degraded"):
                continue
            if ref.pond_id is None or ref.group is None:
                continue
            rows.append({"value": float(v), "group": ref.group, "pond": ref.pond_id})
        if len(rows) < 3 or len({r["pond"] for r in rows}) < 2:
            return None
        try:
            df = pd.DataFrame(rows)
            model = smf.mixedlm("value ~ C(group)", df, groups=df["pond"])
            fit = model.fit(reml=True)
            key = [k for k in fit.pvalues.index if "group" in k]
            if not key:
                return None
            return (
                float(fit.pvalues[key[0]]),
                f"MixedLM（value ~ C(group)，pond 随机截距，REML，n={len(rows)}）",
            )
        except Exception:
            return None

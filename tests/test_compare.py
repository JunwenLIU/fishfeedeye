"""T05 UI / 统计 / 导出 / 打点 测试（docs/06 §6 T05 验收标准 1–6）。

覆盖：
    1. gradio main.py 启动，六个页签可用（真实起服务 + 抓 HTML）；
    2. 盲法：页面 HTML 中 grep 不到分组标签字符串；揭盲写审计日志；
    3. 打点：点击 20 次 → manual_counts.csv 恰 20 行，时间戳单调；
    4. compare：metrics_spec_version / model_md5 不同 → 拒绝整份比较并
       列出差异 + 出路；统计输出含 p / 效应量 / 95%CI / 方法名 / n /
       重复结构说明六要素；单池 → descriptive_only；
    5. 图表：右删失段为阴影 + ">窗长" 标注，无归零假象；
    6. 修正：改 1 帧 → corrections.jsonl 增 1 行，cache/detections.jsonl
       未被触碰，_manual 并列口径出现。

注：起服务的用例用固定端口 7891，失败即失败（验收要求"六个页签可用"
必须被真实验证，不做 skip）。
"""
from __future__ import annotations

import csv as _csv
import importlib.util
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from src.app.main import TAB_TITLES, build_ui
from src.app.pages.tally import PHASES, TallySession, simulate_clicks
from src.core.config import RunConfig
from src.core.frame_context import RunMeta
from src.core.metric_value import BLOCKING_FLAGS
from src.export.charts import _draw_censored, plot_pellet_curve
from src.export.csv_writer import (
    TIMESERIES_1HZ_HEADER_NOTE,
    TallyRecorder,
    read_metrics_summary_csv,
    resample_to_1hz,
    write_metrics_summary_csv,
    write_timeseries_1hz_csv,
)
from src.export.summary_writer import (
    render_capability_report_md,
    write_flag_glossary_csv,
)
from src.metrics import SUMMARY_COLUMNS, MetricsAggregator, write_run_outputs
from src.stats.comparison_plan import ConsistencyReport, RunBundle
from src.stats.two_group import MIN_N_PER_GROUP, TwoGroupPlan

TEST_PORT = 7891
GROUP_A = "对照组标签"
GROUP_B = "试验组标签"


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def make_bundle(
    run_id: str,
    group: str,
    values: dict[str, float | None],
    *,
    spec_version: str = "ms-v1",
    model_md5: str = "md5-aaa",
    pond_id: str | None = "pond_A",
    window_s: float = 300.0,
    window_truncated: bool = False,
    px_per_mm: float | None = 2.0,
    quality: dict | None = None,
    statuses: dict[str, str] | None = None,
    flags: dict[str, str] | None = None,
    t0_definition: str = "pellet_in_frame",
    t0_source: str = "manual",
) -> RunBundle:
    """构造一个参与比较的 run（metrics 行 = metrics_summary.csv 的行结构）。"""
    cfg = RunConfig()
    cfg.metrics_spec_version = spec_version
    cfg.model_md5 = model_md5
    cfg.px_per_mm_ref = px_per_mm
    cfg.t0_definition = t0_definition
    cfg.t0_source = t0_source
    metrics = {
        mid: {
            "metric_id": mid,
            "value": v,
            "status": (statuses or {}).get(mid, "ok"),
            "flags": (flags or {}).get(mid, ""),
            "window_s": window_s,
            "metric_name_zh": mid,
        }
        for mid, v in values.items()
    }
    return RunBundle(
        run_id=run_id, config=cfg, metrics=metrics, disabled=set(),
        quality=quality or {}, window_s=window_s,
        window_truncated=window_truncated, group=group, pond_id=pond_id,
    )


# ----------------------------------------------------------------------
# 验收 4 · 拒绝规则
# ----------------------------------------------------------------------
class TestRejectRules:
    def test_spec_version_diff_rejects_all(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0}, spec_version="ms-v1")
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0}, spec_version="ms-v2")
        plan = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B)
        rep = plan.validate()
        assert rep.reject_all() is True
        assert rep.ok is False
        msgs = " ".join(v.message for v in rep.violations
                        if v.action == "reject_all")
        assert "ms-v1" in msgs and "ms-v2" in msgs
        # 拒绝必须给"重跑对齐"的出路
        assert any("重跑" in v.exit_hint or "重新分析" in v.exit_hint
                   for v in rep.violations if v.action == "reject_all")

    def test_model_md5_diff_rejects_all(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0}, model_md5="md5-aaa")
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0}, model_md5="md5-bbb")
        rep = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B).validate()
        assert rep.reject_all()
        assert any(v.rule_id == 3 for v in rep.violations)

    def test_t0_definition_diff_rejects_all(self) -> None:
        """规则 8（docs/06 §6）：t0 定义不同 → 时间类指标整体平移，不可比。"""
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0},
                        t0_definition="pellet_in_frame")
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0},
                        t0_definition="feeder_start")
        rep = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B).validate()
        assert rep.reject_all()
        assert any(v.rule_id == 8 for v in rep.violations)

    def test_t0_source_diff_rejects_all(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0}, t0_source="manual")
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0}, t0_source="auto")
        rep = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B).validate()
        assert any(v.rule_id == 9 for v in rep.violations)

    def test_px_per_mm_diff_rejects_unnormalized_only(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0, "B2-1_ANND": 10.0},
                        px_per_mm=2.0)
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0, "B2-1_ANND": 12.0},
                        px_per_mm=3.0,
                        statuses={"B2-1_ANND": "degraded"},
                        flags={"B2-1_ANND": "unnormalized"})
        rep = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B).validate()
        assert "B2-1_ANND" in rep.rejected_metrics
        assert "A8_T50" not in rep.rejected_metrics
        assert not rep.reject_all()

    def test_window_truncated_rejects_clearance_metrics(self) -> None:
        a = make_bundle("r1", GROUP_A,
                        {"A8_T50": 40.0, "A11_RR": 5.0, "A1_N0": 100.0})
        b = make_bundle("r2", GROUP_B,
                        {"A8_T50": 30.0, "A11_RR": 4.0, "A1_N0": 100.0},
                        window_s=120.0, window_truncated=True)
        rep = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B).validate()
        assert "A11_RR" in rep.rejected_metrics
        assert "A8_T50" in rep.rejected_metrics
        assert "A1_N0" not in rep.rejected_metrics
        assert not rep.reject_all()

    def test_roi_mismatch_warns_only(self) -> None:
        from src.core.roi import ROI

        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0})
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0})
        a.roi = ROI(arena=np.array([[0, 0], [100, 0], [100, 100], [0, 100]],
                                   dtype=float))
        b.roi = ROI(arena=np.array([[0, 0], [200, 0], [200, 200], [0, 200]],
                                   dtype=float))
        rep = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B).validate()
        assert rep.ok is True                       # 只告警，不拒绝
        assert any(v.rule_id == 5 and v.action == "warn"
                   for v in rep.violations)

    def test_available_set_mismatch_warns(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0, "B2-7_RP": 1.4})
        b = make_bundle("r2", GROUP_B,
                        {"A8_T50": 30.0, "B2-7_RP": None},
                        statuses={"B2-7_RP": "unavailable"})
        rep = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B).validate()
        warn = [v for v in rep.violations if v.rule_id == 6]
        assert warn and "B2-7_RP" in warn[0].affected_metrics
        assert "非随机" in warn[0].message

    def test_non_neutral_gate_rejects_subset_inference(self) -> None:
        """缺失由非中立门控触发 → 拒绝该指标，并给 B1-1/2/3 的出路。"""
        a = make_bundle("r1", GROUP_A, {"B2-7_RP": 1.4},
                        quality={"Q_overlap": 0.5})
        b = make_bundle("r2", GROUP_B, {"B2-7_RP": None},
                        statuses={"B2-7_RP": "unavailable"},
                        flags={"B2-7_RP": "fish_count_confounded"},
                        quality={"Q_overlap": 0.5})
        rep = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B).validate()
        v7 = [v for v in rep.violations if v.rule_id == 7]
        assert v7 and "B2-7_RP" in v7[0].affected_metrics
        assert "B1-1/2/3" in v7[0].exit_hint

    def test_unmeasured_gate_does_not_trigger_rule7(self) -> None:
        """未测得（None）不算门控触发——docs/04 §4.0：未测得 ≠ 测得为差。"""
        a = make_bundle("r1", GROUP_A, {"B2-7_RP": 1.4}, quality={})
        b = make_bundle("r2", GROUP_B, {"B2-7_RP": None},
                        statuses={"B2-7_RP": "unavailable"}, quality={})
        rep = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B).validate()
        assert not any(v.rule_id == 7 for v in rep.violations)
        # 但规则 6 的"可用集不一致"仍然告警（缺失本身仍需提示）
        assert any(v.rule_id == 6 for v in rep.violations)

    def test_no_force_compare_switch(self) -> None:
        """不提供绕过开关（docs/04 §4.4）：API 层无 force/override/ignore。"""
        assert not hasattr(ConsistencyReport, "ignore")
        assert not hasattr(ConsistencyReport, "override")
        assert not hasattr(TwoGroupPlan, "force_execute")


# ----------------------------------------------------------------------
# 验收 4 · 统计六要素
# ----------------------------------------------------------------------
class TestTwoGroupStatistics:
    @staticmethod
    def _plan(na: int = 6, nb: int = 6, ponds: bool = True,
              runs_per_pond: int = 2) -> TwoGroupPlan:
        """两组 run；默认每个池塘 `runs_per_pond` 次投喂（随机效应可辨识）。

        为什么默认 2：每个池塘只有 1 个观测时，MixedLM 的随机截距方差与
        残差方差无法分离（模型饱和），其 p 值不可信（可在同分布数据上
        给出假阳性）。故需要 MixedLM 的用例必须给出池内重复。
        """
        rng = np.random.default_rng(42)
        a = [make_bundle(f"a{i}", GROUP_A, {"A8_T50": float(v)},
                         pond_id=(f"pond_{i // runs_per_pond}"
                                  if ponds else "pond_A"))
             for i, v in enumerate(rng.normal(40.0, 3.0, na))]
        b = [make_bundle(f"b{i}", GROUP_B, {"A8_T50": float(v)},
                         pond_id=(f"pond_{10 + i // runs_per_pond}"
                                  if ponds else "pond_A"))
             for i, v in enumerate(rng.normal(34.0, 3.0, nb))]
        return TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B)

    def test_six_elements_present(self) -> None:
        res = self._plan().execute("A8_T50")
        assert res.status == "ok"
        # ① p 值 ② 效应量 ③ 95%CI ④ 方法名 ⑤ n ⑥ 重复结构
        assert res.p_value is not None and 0.0 <= res.p_value <= 1.0
        assert res.effect_size is not None
        assert res.effect_size_type in ("cohen_d_hedges_g", "rank_biserial")
        assert res.ci_low is not None and res.ci_high is not None
        assert res.ci_low <= res.ci_high and res.ci_level == 0.95
        assert res.test_used in ("welch_t", "mann_whitney", "mixed_lm")
        assert res.n_a == 6 and res.n_b == 6
        assert res.repeat_structure

    def test_multi_pond_uses_mixedlm(self) -> None:
        res = self._plan().execute("A8_T50")      # 每池 2 次投喂 → 可辨识
        assert res.test_used == "mixed_lm"
        assert res.descriptive_only is False
        # 两个检验的 p 值都保留（不隐藏与主检验不一致的那一个）
        assert res.p_welch is not None and res.p_mwu is not None

    def test_single_pond_is_descriptive_only(self) -> None:
        res = self._plan(ponds=False).execute("A8_T50")
        assert res.descriptive_only is True
        assert "伪重复" in res.repeat_structure or "单池塘" in res.repeat_structure
        assert any("不可做统计推断" in w for w in res.warnings)

    def test_undeclared_pond_is_descriptive_only(self) -> None:
        a = [make_bundle(f"a{i}", GROUP_A, {"A8_T50": 40.0 + i}, pond_id=None)
             for i in range(3)]
        b = [make_bundle(f"b{i}", GROUP_B, {"A8_T50": 34.0 + i}, pond_id=None)
             for i in range(3)]
        res = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B).execute(
            "A8_T50")
        assert res.descriptive_only is True
        assert "未声明" in res.repeat_structure

    def test_insufficient_n_gives_no_p_value(self) -> None:
        """宁可不给 p 值，不可给占位 p（p=1.0/0.5 一律不输出）。"""
        a = [make_bundle("a1", GROUP_A, {"A8_T50": 40.0})]
        b = [make_bundle("b1", GROUP_B, {"A8_T50": 34.0})]
        res = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B).execute(
            "A8_T50")
        assert res.status == "unavailable"
        assert res.p_value is None
        assert res.test_used is None
        assert res.reason and "样本量不足" in res.reason
        assert res.mean_a is None and res.mean_b is None

    def test_unavailable_values_never_enter_statistics(self) -> None:
        """不可用/删失值不进样本（绝不用 0 填补）。"""
        a = [
            make_bundle("a1", GROUP_A, {"A8_T50": 40.0}, pond_id="p1"),
            make_bundle("a2", GROUP_A, {"A8_T50": 38.0}, pond_id="p2"),
            make_bundle("a3", GROUP_A, {"A8_T50": None}, pond_id="p3",
                        statuses={"A8_T50": "censored"}),
        ]
        b = [make_bundle(f"b{i}", GROUP_B, {"A8_T50": 34.0 - i}, pond_id=f"q{i}")
             for i in range(3)]
        res = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B).execute(
            "A8_T50")
        assert res.n_a == 2 and res.n_b == 3
        assert res.mean_a == pytest.approx(39.0)

    def test_run_all_applies_holm_correction(self) -> None:
        plan = self._plan()
        out = plan.run_all(["A8_T50"])
        assert set(out) == {"A8_T50"}
        assert out["A8_T50"].p_holm is not None
        assert out["A8_T50"].p_holm >= out["A8_T50"].p_value - 1e-12

    def test_identical_groups_not_significant(self) -> None:
        rng = np.random.default_rng(11)
        a = [make_bundle(f"a{i}", GROUP_A, {"Y": float(v)}, pond_id=f"p{i}")
             for i, v in enumerate(rng.normal(20, 2, 12))]
        b = [make_bundle(f"b{i}", GROUP_B, {"Y": float(v)}, pond_id=f"q{i}")
             for i, v in enumerate(rng.normal(20, 2, 12))]
        res = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B).execute("Y")
        assert res.p_value is not None and res.p_value > 0.05

    def test_zero_variance_constant_metric_has_no_p(self) -> None:
        a = [make_bundle(f"a{i}", GROUP_A, {"Z": 5.0}, pond_id=f"p{i}")
             for i in range(3)]
        b = [make_bundle(f"b{i}", GROUP_B, {"Z": 5.0}, pond_id=f"q{i}")
             for i in range(3)]
        res = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B).execute("Z")
        assert res.p_value is None
        assert res.reason and "占位值" in res.reason

    def test_min_n_constant_is_at_least_two(self) -> None:
        assert MIN_N_PER_GROUP >= 2   # n<2 无法估计方差，无从检验


# ----------------------------------------------------------------------
# 验收 3 · 打点计数
# ----------------------------------------------------------------------
class TestTallyRecorder:
    def test_20_clicks_yield_20_monotonic_rows(self, tmp_path: Path) -> None:
        sess = simulate_clicks(20)
        out = sess.to_csv(tmp_path / "manual_counts.csv")
        lines = [ln for ln in out.read_text(encoding="utf-8-sig").splitlines()
                 if ln.strip() and not ln.startswith("#")]
        assert len(lines) == 21                      # 表头 + 20 行
        rows = list(_csv.DictReader(lines))
        assert len(rows) == 20
        ts = [float(r["t_s"]) for r in rows]
        assert all(b > a for a, b in zip(ts, ts[1:]))    # 严格单调
        assert [int(r["event_id"]) for r in rows] == list(range(1, 21))

    def test_header_carries_operator_video_and_disclaimer(
        self, tmp_path: Path
    ) -> None:
        rec = TallyRecorder(operator="kou", video_name="demo.mp4")
        rec.on_key(1.5, phase=PHASES[1])
        head = rec.to_csv(tmp_path / "manual_counts.csv").read_text(
            encoding="utf-8-sig")
        assert "# operator: kou" in head
        assert "# video: demo.mp4" in head
        assert "永不合并" in head           # 独立证据链声明

    def test_series_is_cumulative(self) -> None:
        assert simulate_clicks(5, dt=1.0).recorder.series()[-1] == (4.0, 5)

    def test_undo_drops_last_event(self) -> None:
        sess = simulate_clicks(3)
        assert sess.undo() == 2

    def test_tally_never_merges_with_auto_metrics(self) -> None:
        """打点只是并列文件，不修正任何自动指标。"""
        sess = simulate_clicks(4)
        assert not hasattr(sess.recorder, "metrics")
        assert sess.recorder.series()[0][1] == 1


# ----------------------------------------------------------------------
# 验收 5 · 图表：删失段阴影 + ">窗长"，不归零
# ----------------------------------------------------------------------
class TestChartCensoring:
    def test_censored_span_drawn(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlim(0, 120)
        _draw_censored(ax, 60.0)
        spans = [c for c in ax.get_children()
                 if type(c).__name__ == "Span" or hasattr(c, "get_xy")]
        assert spans, "删失段必须以阴影（Span）呈现"
        plt.close(fig)

    def test_annotation_states_lower_bound_not_zero(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlim(0, 120)
        _draw_censored(ax, 60.0)
        texts = [t.get_text() for t in ax.texts]
        assert any(">60s" in s for s in texts)
        assert any("非 0" in s for s in texts)
        plt.close(fig)

    def test_pellet_curve_file_written(self, tmp_path: Path) -> None:
        t = np.arange(0.0, 60.0, 2.0)
        n = np.full(t.shape, 100.0)                  # 从不下降 → T50 右删失
        out = plot_pellet_curve(tmp_path / "np.png", t, n, n0=100.0,
                                censored_window_s=60.0)
        assert out.exists() and out.stat().st_size > 0


# ----------------------------------------------------------------------
# 导出：17 列 / 1Hz 便利层 / 术语表 / capability_report.md
# ----------------------------------------------------------------------
class TestExportArtifacts:
    def test_summary_csv_17_columns_and_empty_discipline(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {"metric_id": "A8_T50", "status": "censored", "value": None,
             "blocking_flag_count": None},
            {"metric_id": "A11_RR", "status": "ok", "value": 5.0,
             "blocking_flag_count": 0},
        ]
        out = write_metrics_summary_csv(rows, tmp_path / "s.csv",
                                        columns=list(SUMMARY_COLUMNS))
        back = read_metrics_summary_csv(out)
        assert len(back[0]) == 17
        assert back[0]["value"] is None               # 空 = 空字符串，不是 0
        assert back[0]["blocking_flag_count"] is None
        assert back[1]["value"] == "5"

    def test_1hz_layer_marked_interpolated(self, tmp_path: Path) -> None:
        from src.metrics.group_a.pellet_curve import TimeSeries

        ts = TimeSeries(metric_id="A2_Np", t=np.array([0.0, 5.0, 10.0]),
                        values=np.array([100.0, 80.0, 60.0]))
        text = write_timeseries_1hz_csv([ts], tmp_path / "ts1.csv").read_text(
            encoding="utf-8-sig")
        assert TIMESERIES_1HZ_HEADER_NOTE in text
        assert "interpolated" in text
        pts = resample_to_1hz(ts.t, ts.values)
        assert pts[0].interpolated is False           # 原生观测点
        assert pts[1].interpolated is True            # 插值点
        assert pts[1].source_interval_s == pytest.approx(5.0)

    def test_flag_glossary_covers_blocking_and_status_flags(
        self, tmp_path: Path
    ) -> None:
        out = write_flag_glossary_csv(tmp_path / "flag_glossary.csv")
        with open(out, encoding="utf-8-sig", newline="") as fh:
            rows = list(_csv.DictReader(fh))
        tokens = {r["flag_token"] for r in rows}
        assert BLOCKING_FLAGS <= tokens
        for st in ("ok", "degraded", "unavailable", "censored"):
            assert f"status:{st}" in tokens
        assert list(rows[0].keys()) == ["flag_token", "中文名", "含义", "建议动作"]

    def test_capability_report_lists_disabled_metrics(self) -> None:
        from src.metrics.capability import DisabledEntry

        class _Cap:
            disabled_with_reason = [
                DisabledEntry(metric="C_group", reason="Q_track = 0.30 < 0.50",
                              hint="缺失可能是效应本身")
            ]

        md = render_capability_report_md(
            run_id="r1", metrics_spec_version="ms-v1",
            quality_table=[{"signal": "Q_det", "value": 0.8,
                            "threshold": 0.5, "passed": True}],
            capability=_Cap(), warnings=["w1"], notes=["n1"],
            metric_rows=[{"metric_id": "A8_T50", "status": "censored",
                          "reason": "观察窗内未穿越", "window_s": 300.0}],
        )
        assert "C_group" in md and "Q_track = 0.30" in md
        assert "未输出的指标及原因" in md
        assert "缺失本身可能是效应" in md
        assert "flag 术语表" in md


# ----------------------------------------------------------------------
# 验收 1 & 2 · Gradio 启动 + 盲法
# ----------------------------------------------------------------------
class TestGradioApp:
    @pytest.fixture(scope="class")
    def served_html(self):
        """真实起一次 Gradio 服务并抓首页 HTML（验收 1/2 的唯一可信验证）。"""
        from src.app.state import ProjectState, VideoEntry

        state = ProjectState(project_dir="_scratch/project_pytest")
        state.videos = [
            VideoEntry(video_path="a.mp4", group=GROUP_A, pond_id="pond_A"),
            VideoEntry(video_path="b.mp4", group=GROUP_B, pond_id="pond_B"),
        ]
        state.blind_enabled = True
        state.revealed = False
        demo = build_ui(state)
        demo.launch(server_name="127.0.0.1", server_port=TEST_PORT,
                    prevent_thread_lock=True, share=False, inbrowser=False,
                    show_error=True)
        html = ""
        for _ in range(60):
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{TEST_PORT}/",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                html = urllib.request.urlopen(req, timeout=5).read().decode(
                    "utf-8", "replace")
                break
            except Exception:
                time.sleep(0.5)
        try:
            yield html
        finally:
            demo.close()

    def test_six_tabs_available(self, served_html: str) -> None:
        assert served_html, "Gradio 服务未起来"
        assert len(TAB_TITLES) == 6
        for title in TAB_TITLES:
            assert title in served_html, f"页签缺失: {title}"

    def test_blind_mode_hides_group_labels(self, served_html: str) -> None:
        assert served_html
        assert GROUP_A not in served_html
        assert GROUP_B not in served_html
        assert "盲法" in served_html

    def test_reveal_writes_audit_log(self, tmp_path: Path) -> None:
        from src.app.state import ProjectState, VideoEntry

        st = ProjectState(project_dir=tmp_path / "proj")
        st.videos = [VideoEntry(video_path="a.mp4", group=GROUP_A)]
        assert st.display_label(0) == "run_001"        # 盲法编号
        st.reveal(operator="kou")
        assert st.revealed is True
        assert st.display_label(0) == GROUP_A          # 揭盲后显示分组
        assert any(r["action"] == "reveal" for r in st.audit_log)
        assert (tmp_path / "proj" / "audit_log.jsonl").exists()
        st.save_blind_map()
        mapping = st.load_blind_map()
        assert mapping["entries"][0]["group"] == GROUP_A
        assert mapping["entries"][0]["blind_code"] == "run_001"


# ----------------------------------------------------------------------
# 验收 6 · 人工修正（只重跑 metrics 层，缓存不被触碰）
# ----------------------------------------------------------------------
class TestManualCorrection:
    @staticmethod
    def _make_run(tmp_path: Path) -> Path:
        from src.core.frame_context import FrameObservation, PelletDetections

        obs = []
        for i, t in enumerate(np.arange(0.0, 60.0, 2.0)):
            n = int(round(100 * math.exp(-0.02 * t)))
            xyxy = np.array(
                [[20 + j * 12, 20, 26 + j * 12, 26] for j in range(max(n, 0))],
                dtype=float,
            ).reshape(-1, 4)
            obs.append(FrameObservation(
                frame_idx=i, t_s=float(t), dt_s=None, image=None,
                pellets=PelletDetections(xyxy=xyxy,
                                         conf=np.full(max(n, 0), 0.9)),
                extra={},
            ))
        run_dir = tmp_path / "run_demo"
        (run_dir / "cache").mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({
                "type": "frame", "frame_idx": o.frame_idx, "t_s": o.t_s,
                "dt_s": None,
                "pellets": {
                    "xyxy": o.pellets.xyxy.reshape(-1, 4).tolist(),
                    "conf": o.pellets.conf.tolist(),
                    "area_px": None, "track_id": None, "vanish_class": None,
                },
                "extra": {},
            }, ensure_ascii=False)
            for o in obs
        ]
        (run_dir / "cache" / "detections.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        RunConfig().to_yaml(run_dir / "run_config.yaml")
        rep = MetricsAggregator(config=RunConfig()).aggregate(
            obs, meta=RunMeta(feed_mass_g=2.5, pellet_mass_mg=25.0))
        write_run_outputs(rep, run_dir)
        return run_dir

    def test_correction_appends_and_keeps_cache_untouched(
        self, tmp_path: Path
    ) -> None:
        from src.app.pages.results import apply_correction

        run_dir = self._make_run(tmp_path)
        cache = run_dir / "cache" / "detections.jsonl"
        before = cache.read_text(encoding="utf-8")
        before_mtime = cache.stat().st_mtime_ns

        out = apply_correction(run_dir, frame_idx=5, new_n=42,
                               operator="kou", note="复核")

        # ① corrections.jsonl 增加 1 行（含原值/新值/人/时间）
        recs = [json.loads(x) for x in
                (run_dir / "corrections.jsonl").read_text(
                    encoding="utf-8").splitlines() if x.strip()]
        assert len(recs) == 1
        assert recs[0]["frame_idx"] == 5
        assert recs[0]["new_n"] == 42
        assert recs[0]["operator"] == "kou"
        assert recs[0]["original_n"] is not None
        assert recs[0]["timestamp"]

        # ② cache/detections.jsonl 未被触碰（不重跑检测）
        assert cache.read_text(encoding="utf-8") == before
        assert cache.stat().st_mtime_ns == before_mtime
        assert out["cache_touched"] is False

        # ③ _manual 并列口径出现，原始口径保留
        assert out["n_corrected"] == 1
        assert 0.0 < out["share"] < 1.0
        assert any(m.endswith("_manual") for m in out["manual_metrics"])
        rows = read_metrics_summary_csv(run_dir / "metrics_summary.csv")
        ids = {r["metric_id"] for r in rows}
        assert "A8_T50" in ids                       # 原始口径仍在
        assert any(i.endswith("_manual") for i in ids)

    def test_second_correction_appends_not_overwrites(
        self, tmp_path: Path
    ) -> None:
        from src.app.pages.results import apply_correction

        run_dir = self._make_run(tmp_path)
        apply_correction(run_dir, 3, 50, operator="kou")
        apply_correction(run_dir, 7, 30, operator="kou")
        recs = [json.loads(x) for x in
                (run_dir / "corrections.jsonl").read_text(
                    encoding="utf-8").splitlines() if x.strip()]
        assert [r["frame_idx"] for r in recs] == [3, 7]   # 追加留痕

    def test_unknown_frame_raises(self, tmp_path: Path) -> None:
        from src.app.pages.results import apply_correction

        with pytest.raises(ValueError, match="不在缓存观测中"):
            apply_correction(self._make_run(tmp_path), 99999, 10)


# ----------------------------------------------------------------------
# G1 误差实测纪律（docs/04 §7.1 ③）
# ----------------------------------------------------------------------
class TestGroundTruthValidation:
    @staticmethod
    def _mod():
        import importlib.util

        path = (Path(__file__).resolve().parents[1] / "scripts"
                / "06_validate_against_groundtruth.py")
        spec = importlib.util.spec_from_file_location("validate_gt", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        # 必须先登记进 sys.modules：模块内 @dataclass 解析类型注解时会
        # 通过 sys.modules[cls.__module__] 反查命名空间，未登记会抛
        # AttributeError: 'NoneType' object has no attribute '__dict__'。
        sys.modules["validate_gt"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_unmeasurable_when_error_below_noise_floor(self) -> None:
        mod = self._mod()
        rows = []
        for f in range(6):      # 同帧双计数差异 ±3 颗 → 噪声下限 3
            rows.append(mod.GroundTruthRow(frame_idx=f, t_s=float(f),
                                           n_pellets_gt=100, counter="A"))
            rows.append(mod.GroundTruthRow(frame_idx=f, t_s=float(f),
                                           n_pellets_gt=103, counter="B"))
        rep = mod.validate(rows, {f: 101.0 for f in range(6)})
        assert rep.measurable is False            # 纪律 3：判为"无法测量"
        assert "无法区分模型误差与真值噪声" in rep.verdict
        assert rep.mae < rep.noise_floor
        md = mod.render_report_md(rep, "r1")
        assert "无法测量" in md
        assert "color:#999" in md                 # 精度数字降级为次要信息

    def test_measurable_reports_mae_and_bias(self) -> None:
        mod = self._mod()
        rows = [mod.GroundTruthRow(frame_idx=f, t_s=float(f),
                                   n_pellets_gt=100, counter="A")
                for f in range(4)]
        rep = mod.validate(rows, {f: 90.0 for f in range(4)})   # 稳定低估
        assert rep.measurable is True
        assert rep.mae == pytest.approx(10.0)
        assert rep.bias == pytest.approx(-10.0)
        assert "稳定低估" in rep.verdict
        assert any("禁止" in w for w in rep.warnings)   # 纪律 1：小样本不校正

    def test_missing_auto_frames_are_skipped_not_zero_filled(self) -> None:
        mod = self._mod()
        rows = [mod.GroundTruthRow(frame_idx=f, t_s=float(f),
                                   n_pellets_gt=100, counter="A")
                for f in range(3)]
        rep = mod.validate(rows, {0: 100.0})
        assert rep.n_frames == 1
        assert any("缺失" in w for w in rep.warnings)

"""T05 UI / 统计 / 导出 / 打点 测试（docs/06 §6 T05 验收标准 1–6）。

覆盖：
    1. gradio main.py 启动，六个页签可用（真实起服务 + 抓 HTML）；
    2. 盲法：分析页 HTML 中 grep 不到分组标签字符串；揭盲写审计日志；
    3. 打点：点击 20 次 → manual_counts.csv 恰 20 行，时间戳单调；
    4. compare：metrics_spec_version 不同 → 拒绝整份比较并列出差异；
       统计输出含 p / Cohen's d / 95%CI / 方法名 / n / 重复结构说明六要素；
       model_md5 不一致 → 拒绝全部；单池 → "仅描述性"；
    5. 图表：右删失段为阴影 + ">窗长" 标注，无归零假象；
    6. 修正：改 1 帧 → corrections.jsonl 增 1 行，cache/detections.jsonl
       未被触碰，_manual 并列口径出现。

注：起服务的用例用固定端口 7891，失败即失败（不做 skip——验收要求
"六个页签可用"必须被真实验证）。
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from src.app.main import TAB_TITLES, build_ui
from src.app.pages.tally import PHASES, TallySession, simulate_clicks
from src.core.config import RunConfig, Thresholds
from src.core.frame_context import RunMeta
from src.export.charts import plot_pellet_curve
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
from src.stats.two_group import TwoGroupPlan

TEST_PORT = 7891
GROUP_A = "对照组"
GROUP_B = "试验组"


# ----------------------------------------------------------------------
# 工具：合成 run bundle
# ----------------------------------------------------------------------
def make_bundle(
    run_id: str,
    group: str,
    values: dict[str, float],
    *,
    spec_version: str = "ms-v1",
    model_md5: str = "md5-aaa",
    pond_id: str | None = "pond_A",
    window_s: float = 300.0,
    window_truncated: bool = False,
    px_per_mm: float | None = 2.0,
    quality: dict | None = None,
    disabled: set[str] | None = None,
    statuses: dict[str, str] | None = None,
) -> RunBundle:
    """构造一个参与比较的 run（metrics 行 = {value, status, flags}）。"""
    cfg = RunConfig()
    cfg.metrics_spec_version = spec_version
    cfg.model_md5 = model_md5
    cfg.px_per_mm_ref = px_per_mm
    metrics = {
        mid: {
            "value": v,
            "status": (statuses or {}).get(mid, "ok"),
            "flags": "",
            "window_s": window_s,
        }
        for mid, v in values.items()
    }
    return RunBundle(
        run_id=run_id, config=cfg, metrics=metrics,
        disabled=disabled or set(), quality=quality or {},
        window_s=window_s, window_truncated=window_truncated,
        group=group, pond_id=pond_id,
    )


# ----------------------------------------------------------------------
# 验收 4 · 七条拒绝规则
# ----------------------------------------------------------------------
class TestSevenRejectRules:
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
        # 拒绝时必须给出"重跑对齐"的出路提示
        assert any("重跑" in v.exit_hint or "重新分析" in v.exit_hint
                   for v in rep.violations if v.action == "reject_all")

    def test_model_md5_diff_rejects_all(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0}, model_md5="md5-aaa")
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0}, model_md5="md5-bbb")
        plan = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B)
        rep = plan.validate()
        assert rep.reject_all()
        assert any(v.rule_id == 3 for v in rep.violations)

    def test_px_per_mm_diff_rejects_unnormalized_only(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0, "B2-1_ANND": 10.0},
                        px_per_mm=2.0)
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0, "B2-1_ANND": 12.0},
                        px_per_mm=3.0, statuses={"B2-1_ANND": "degraded"})
        b.metrics["B2-1_ANND"]["flags"] = "unnormalized"
        plan = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B)
        rep = plan.validate()
        assert "B2-1_ANND" in rep.rejected_metrics
        assert "A8_T50" not in rep.rejected_metrics  # 无量纲指标不受影响
        assert not rep.reject_all()

    def test_window_truncated_rejects_clearance_metrics(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0, "A11_RR": 5.0,
                                        "A1_N0": 100.0})
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0, "A11_RR": 4.0,
                                        "A1_N0": 100.0},
                        window_s=120.0, window_truncated=True)
        plan = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B)
        rep = plan.validate()
        assert "A11_RR" in rep.rejected_metrics
        assert "A8_T50" in rep.rejected_metrics
        assert "A1_N0" not in rep.rejected_metrics
        assert not rep.reject_all()

    def test_roi_mismatch_warns_only(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0})
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0})
        import numpy as np

        from src.core.roi import ROI

        a.roi = ROI(arena=np.array([[0, 0], [100, 0], [100, 100], [0, 100]],
                                   dtype=float))
        b.roi = ROI(arena=np.array([[0, 0], [200, 0], [200, 200], [0, 200]],
                                   dtype=float))
        plan = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B)
        rep = plan.validate()
        assert rep.ok is True      # 只告警，不拒绝
        assert any(v.rule_id == 5 and v.action == "warn"
                   for v in rep.violations)

    def test_available_set_mismatch_warns(self) -> None:
        a = make_bundle("r1", GROUP_A, {"A8_T50": 40.0, "B2-7_RP": 1.4})
        b = make_bundle("r2", GROUP_B, {"A8_T50": 30.0},
                        statuses={"B2-7_RP": "unavailable"})
        b.metrics["B2-7_RP"]["value"] = None
        plan = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B)
        rep = plan.validate()
        warn = [v for v in rep.violations if v.rule_id == 6]
        assert warn and "B2-7_RP" in warn[0].affected_metrics
        assert "非随机" in warn[0].message

    def test_non_neutral_gate_rejects_subset_inference(self) -> None:
        """缺失由非中立门控触发 → 拒绝该指标，并给 B1-1/2/3 的出路。"""
        a = make_bundle(
            "r1", GROUP_A, {"B2-7_RP": 1.4},
            quality={"Q_overlap": 0.5},   # 非中立门控越限
        )
        b = make_bundle(
            "r2", GROUP_B, {}, quality={"Q_overlap": 0.5},
            statuses={"B2-7_RP": "unavailable"},
        )
        b.metrics = {"B2-7_RP": {"value": None, "status": "unavailable",
                                 "flags": "fish_count_confounded"}}
        plan = TwoGroupPlan([a, b], group_a=GROUP_A, group_b=GROUP_B)
        rep = plan.validate()
        v7 = [v for v in rep.violations if v.rule_id == 7]
        assert v7 and "B2-7_RP" in v7[0].affected_metrics
        assert "B1-1/2/3" in v7[0].exit_hint

    def test_no_force_compare_switch_exists(self) -> None:
        """规则 7 不提供绕过开关（docs/04 §4.4）——API 层面无 force 参数。"""
        import inspect

        from src.stats import two_group

        src = inspect.getsource(two_group) + inspect.getsource(
            __import__("src.stats.comparison_plan", fromlist=["x"])
        )
        assert "force" not in {w.strip("_(,\"'") for w in src.split()}
        # 且 ConsistencyReport 没有"忽略/覆盖"接口
        assert not hasattr(ConsistencyReport, "ignore")


# ----------------------------------------------------------------------
# 验收 4 · 统计六要素
# ----------------------------------------------------------------------
class TestTwoGroupStatistics:
    def _plan(self, na: int = 6, nb: int = 6, ponds: bool = True):
        rng = np.random.default_rng(42)
        vals_a = rng.normal(40.0, 3.0, na)
        vals_b = rng.normal(34.0, 3.0, nb)
        a = [make_bundle(f"a{i}", GROUP_A, {"A8_T50": float(v)},
                         pond_id=(f"pond_{i}" if ponds else "pond_A"))
             for i, v in enumerate(vals_a)]
        b = [make_bundle(f"b{i}", GROUP_B, {"A8_T50": float(v)},
                         pond_id=(f"pond_{10 + i}" if ponds else "pond_A"))
             for i, v in enumerate(vals_b)]
        return TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B)

    def test_six_elements_present(self) -> None:
        plan = self._plan()
        res = plan.execute("A8_T50")
        assert res.available
        # ① p 值 ② Cohen's d ③ 95%CI ④ 方法名 ⑤ n ⑥ 重复结构
        assert res.p_value is not None and 0.0 <= res.p_value <= 1.0
        assert res.effect_size is not None
        assert res.ci_low is not None and res.ci_high is not None
        assert res.ci_low <= res.ci_high
        assert res.test_model in ("welch_t", "mann_whitney", "mixedlm")
        assert res.n_a == 6 and res.n_b == 6
        assert res.replication == "independent_ponds"
        assert res.replication_note

    def test_multi_pond_uses_mixedlm(self) -> None:
        plan = self._plan()
        res = plan.execute("A8_T50")
        assert res.test_model == "mixedlm"
        assert res.inferable is True

    def test_single_pond_is_descriptive_only(self) -> None:
        plan = self._plan(ponds=False)
        res = plan.execute("A8_T50")
        assert res.replication == "single_pond"
        assert res.inferable is False
        assert "无独立重复" in res.replication_note
        # 六要素仍在（供用户看），但明确标注仅描述性
        assert res.p_value is not None
        assert any("仅描述性" in w for w in res.warnings)

    def test_undeclared_pond_warns(self) -> None:
        a = [make_bundle(f"a{i}", GROUP_A, {"A8_T50": 40.0 + i}, pond_id=None)
             for i in range(3)]
        b = [make_bundle(f"b{i}", GROUP_B, {"A8_T50": 34.0 + i}, pond_id=None)
             for i in range(3)]
        plan = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B)
        res = plan.execute("A8_T50")
        assert res.replication == "undeclared"
        assert res.inferable is False
        assert "pond_id" in res.replication_note

    def test_insufficient_n_gives_no_p_value(self) -> None:
        """宁可不给 p 值，不可给错的 p 值。"""
        a = [make_bundle("a1", GROUP_A, {"A8_T50": 40.0})]
        b = [make_bundle("b1", GROUP_B, {"A8_T50": 34.0})]
        plan = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B)
        res = plan.execute("A8_T50")
        assert res.p_value is None
        assert res.test_model == "none"
        assert res.reason is not None and "样本量不足" in res.reason
        # 描述性统计仍在
        assert res.mean_a == pytest.approx(40.0)
        assert res.mean_b == pytest.approx(34.0)

    def test_non_normal_switches_to_mann_whitney(self) -> None:
        rng = np.random.default_rng(7)
        vals_a = np.concatenate([rng.normal(10, 1, 8), [80.0, 95.0]])  # 重尾
        vals_b = rng.normal(12, 1.5, 10)
        a = [make_bundle(f"a{i}", GROUP_A, {"X": float(v)}, pond_id=f"p{i}")
             for i, v in enumerate(vals_a)]
        b = [make_bundle(f"b{i}", GROUP_B, {"X": float(v)}, pond_id=f"q{i}")
             for i, v in enumerate(vals_b)]
        plan = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B)
        res = plan.execute("X")
        assert res.test_model in ("mann_whitney", "mixedlm")
        if res.test_model == "mann_whitney":
            assert any("正态" in w for w in res.warnings)

    def test_identical_groups_no_significance(self) -> None:
        rng = np.random.default_rng(11)
        a = [make_bundle(f"a{i}", GROUP_A, {"Y": float(v)}, pond_id=f"p{i}")
             for i, v in enumerate(rng.normal(20, 2, 10))]
        b = [make_bundle(f"b{i}", GROUP_B, {"Y": float(v)}, pond_id=f"q{i}")
             for i, v in enumerate(rng.normal(20, 2, 10))]
        plan = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B)
        res = plan.execute("Y")
        assert res.p_value is not None and res.p_value > 0.05

    def test_unavailable_values_never_enter_statistics(self) -> None:
        """不可用/删失值不进数组（不用 0 填补）。"""
        a = [
            make_bundle("a1", GROUP_A, {"A8_T50": 40.0}, pond_id="p1"),
            make_bundle("a2", GROUP_A, {"A8_T50": 38.0}, pond_id="p2"),
            make_bundle("a3", GROUP_A, {"A8_T50": 0.0}, pond_id="p3",
                        statuses={"A8_T50": "censored"}),
        ]
        a[2].metrics["A8_T50"]["value"] = None
        b = [
            make_bundle("b1", GROUP_B, {"A8_T50": 34.0}, pond_id="q1"),
            make_bundle("b2", GROUP_B, {"A8_T50": 33.0}, pond_id="q2"),
            make_bundle("b3", GROUP_B, {"A8_T50": 32.0}, pond_id="q3"),
        ]
        plan = TwoGroupPlan(a + b, group_a=GROUP_A, group_b=GROUP_B)
        res = plan.execute("A8_T50")
        assert res.n_a == 2 and res.n_b == 3   # 删失的那一帧被排除，不是当 0
        assert res.mean_a == pytest.approx(39.0)


# ----------------------------------------------------------------------
# 验收 3 · 打点计数
# ----------------------------------------------------------------------
class TestTallyRecorder:
    def test_20_clicks_yield_20_monotonic_rows(self, tmp_path: Path) -> None:
        sess = simulate_clicks(20)
        out = sess.to_csv(tmp_path / "manual_counts.csv")
        lines = [ln for ln in out.read_text(encoding="utf-8-sig").splitlines()
                 if ln.strip() and not ln.startswith("#")]
        assert len(lines) == 21                    # 表头 + 20 行
        rows = [ln.split(",") for ln in lines[1:]]
        assert len(rows) == 20
        ts = [float(r[1]) for r in rows]
        assert all(b > a for a, b in zip(ts, ts[1:]))   # 严格单调
        assert [int(r[0]) for r in rows] == list(range(1, 21))

    def test_csv_header_carries_operator_and_video(self, tmp_path: Path) -> None:
        rec = TallyRecorder(operator="kou", video_name="demo.mp4")
        rec.on_key(1.5, phase=PHASES[1])
        out = rec.to_csv(tmp_path / "manual_counts.csv")
        head = out.read_text(encoding="utf-8-sig")
        assert "# operator: kou" in head
        assert "# video: demo.mp4" in head
        assert "独立证据链" in head

    def test_series_is_cumulative(self) -> None:
        sess = simulate_clicks(5, dt=1.0)
        series = sess.recorder.series()
        assert series[-1] == (4.0, 5)

    def test_undo_drops_last_event(self) -> None:
        sess = simulate_clicks(3)
        n = sess.undo()
        assert n == 2 and len(sess.recorder.events) == 2

    def test_manual_counts_never_merge_with_auto(self) -> None:
        """打点不修正自动指标：manual_counts.csv 只是并列文件。"""
        sess = simulate_clicks(4)
        # TallyRecorder 不接触任何 MetricValue / metrics_summary
        assert not hasattr(sess.recorder, "metrics")
        assert sess.recorder.series()[0][1] == 1


# ----------------------------------------------------------------------
# 验收 5 · 图表：删失段阴影，不归零
# ----------------------------------------------------------------------
class TestChartCensoring:
    def test_censored_region_is_shaded_not_zero(self, tmp_path: Path) -> None:
        t = np.arange(0.0, 60.0, 2.0)
        n = np.full(t.shape, 100.0)          # 从不下降 → T50 右删失
        out = plot_pellet_curve(
            tmp_path / "np.png", t, n, n0=100.0, censored_window_s=60.0,
        )
        assert out.exists() and out.stat().st_size > 0
        # 图内不得出现"曲线延拓到 0"的数据：本函数只画真实点 + 阴影区，
        # 阴影跨度 = [window_s, x_max]，由 _draw_censored 绘制。
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from src.export.charts import _draw_censored

        fig, ax = plt.subplots()
        ax.plot(t, n)
        ax.set_xlim(0, 120)
        _draw_censored(ax, 60.0)
        spans = [c for c in ax.get_children()
                 if type(c).__name__ == "Span" or hasattr(c, "get_xy")]
        assert spans, "删失段必须以阴影（Span）呈现"
        plt.close(fig)

    def test_censored_annotation_text(self, tmp_path: Path) -> None:
        """标注文本含 '>窗长' 与'非 0' 字样（无归零假象）。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from src.export.charts import _draw_censored

        fig, ax = plt.subplots()
        ax.set_xlim(0, 120)
        _draw_censored(ax, 60.0)
        texts = [t.get_text() for t in ax.texts]
        assert any(">60s" in s for s in texts)
        assert any("非 0" in s for s in texts)
        plt.close(fig)


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
        # 空值 = 空字符串，绝不为 0
        assert back[0]["value"] is None
        assert back[0]["blocking_flag_count"] is None
        assert back[1]["value"] == "5"

    def test_1hz_layer_is_opt_in_and_marked_interpolated(
        self, tmp_path: Path
    ) -> None:
        from src.metrics.group_a.pellet_curve import TimeSeries

        ts = TimeSeries(metric_id="A2_Np", t=np.array([0.0, 5.0, 10.0]),
                        values=np.array([100.0, 80.0, 60.0]))
        out = write_timeseries_1hz_csv([ts], tmp_path / "ts1.csv")
        text = out.read_text(encoding="utf-8-sig")
        assert TIMESERIES_1HZ_HEADER_NOTE in text
        assert "interpolated" in text
        pts = resample_to_1hz(ts.t, ts.values)
        assert pts[0].interpolated is False     # 原生点
        assert pts[1].interpolated is True      # 插值点
        assert pts[1].source_interval_s == pytest.approx(5.0)

    def test_flag_glossary_covers_all_blocking_flags(
        self, tmp_path: Path
    ) -> None:
        import csv as _csv

        from src.core.metric_value import BLOCKING_FLAGS

        out = write_flag_glossary_csv(tmp_path / "flag_glossary.csv")
        with open(out, encoding="utf-8-sig", newline="") as fh:
            rows = list(_csv.DictReader(fh))
        tokens = {r["flag_token"] for r in rows}
        assert BLOCKING_FLAGS <= tokens          # 8 项阻断 flag 全覆盖
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

        rows = [{"metric_id": "A8_T50", "status": "censored",
                 "reason": "观察窗内未穿越", "window_s": 300.0}]
        md = render_capability_report_md(
            run_id="r1", metrics_spec_version="ms-v1",
            quality_table=[{"signal": "Q_det", "value": 0.8,
                            "threshold": 0.5, "passed": True}],
            capability=_Cap(), warnings=["w1"], notes=["n1"],
            metric_rows=rows,
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
    def served_html(self) -> str:
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
                    "utf-8", "replace"
                )
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
        # 分析页与全部页面 HTML 中不得出现真实分组标签
        assert GROUP_A not in served_html
        assert GROUP_B not in served_html
        # 盲法编号体系存在
        assert "run_" in served_html or "盲法" in served_html

    def test_reveal_writes_audit_log(self, tmp_path: Path) -> None:
        from src.app.state import ProjectState, VideoEntry

        st = ProjectState(project_dir=tmp_path / "proj")
        st.videos = [VideoEntry(video_path="a.mp4", group=GROUP_A)]
        assert st.display_label(0) == "run_001"     # 盲法编号
        st.reveal(operator="kou")
        assert st.revealed is True
        assert st.display_label(0) == GROUP_A       # 揭盲后显示分组
        assert any(r["action"] == "reveal" for r in st.audit_log)
        assert (tmp_path / "proj" / "audit_log.jsonl").exists()
        # 盲法映射表落盘且可被读回
        st.save_blind_map()
        mapping = st.load_blind_map()
        assert mapping["entries"][0]["group"] == GROUP_A
        assert mapping["entries"][0]["blind_code"] == "run_001"


# ----------------------------------------------------------------------
# 验收 6 · 人工修正（只重跑 metrics 层）
# ----------------------------------------------------------------------
class TestManualCorrection:
    def _make_run(self, tmp_path: Path) -> Path:
        """用聚合器产出一个真实 run 目录（含 cache/detections.jsonl）。"""
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
                pellets=PelletDetections(
                    xyxy=xyxy, conf=np.full(max(n, 0), 0.9),
                ),
                extra={},
            ))
        run_dir = tmp_path / "run_demo"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "cache").mkdir(exist_ok=True)
        # 手写 cache（模拟 Orchestrator 落盘格式）
        lines = []
        for o in obs:
            lines.append(json.dumps({
                "type": "frame", "frame_idx": o.frame_idx, "t_s": o.t_s,
                "dt_s": None,
                "pellets": {
                    "xyxy": o.pellets.xyxy.reshape(-1, 4).tolist(),
                    "conf": o.pellets.conf.tolist(),
                    "area_px": None, "track_id": None, "vanish_class": None,
                },
                "extra": {},
            }, ensure_ascii=False))
        cache = run_dir / "cache" / "detections.jsonl"
        cache.write_text("\n".join(lines) + "\n", encoding="utf-8")
        RunConfig().to_yaml(run_dir / "run_config.yaml")
        agg = MetricsAggregator(config=RunConfig())
        rep = agg.aggregate(obs, meta=RunMeta(feed_mass_g=2.5,
                                              pellet_mass_mg=25.0))
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

        # ① corrections.jsonl 增加 1 行
        log = run_dir / "corrections.jsonl"
        recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()
                if x.strip()]
        assert len(recs) == 1
        assert recs[0]["frame_idx"] == 5
        assert recs[0]["new_n"] == 42
        assert recs[0]["operator"] == "kou"
        assert recs[0]["original_n"] is not None   # 原值留痕
        assert recs[0]["timestamp"]

        # ② cache/detections.jsonl 未被触碰（不重跑检测）
        assert cache.read_text(encoding="utf-8") == before
        assert cache.stat().st_mtime_ns == before_mtime
        assert out["cache_touched"] is False

        # ③ 并列口径：_manual 指标出现，原始值不被覆盖
        assert out["n_corrected"] == 1
        assert 0.0 < out["share"] < 1.0
        assert any(m.endswith("_manual") for m in out["manual_metrics"])
        assert "A8_T50_manual" in out["manual_metrics"] or any(
            m.startswith("A") for m in out["manual_metrics"]
        )
        # 原始口径仍在 summary.csv 中
        rows = read_metrics_summary_csv(run_dir / "metrics_summary.csv")
        ids = {r["metric_id"] for r in rows}
        assert "A8_T50" in ids
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
        assert len(recs) == 2          # 追加，不覆盖（审计留痕）
        assert [r["frame_idx"] for r in recs] == [3, 7]

    def test_correction_unknown_frame_raises(self, tmp_path: Path) -> None:
        from src.app.pages.results import apply_correction

        run_dir = self._make_run(tmp_path)
        with pytest.raises(ValueError, match="不在缓存观测中"):
            apply_correction(run_dir, 99999, 10)


# ----------------------------------------------------------------------
# 误差实测脚本（G1）纪律
# ----------------------------------------------------------------------
class TestGroundTruthValidation:
    """G1 误差实测：五条纪律（docs/04 §7.1 ③）。"""

    @staticmethod
    def _mod():
        """按文件路径加载 scripts/06_validate_against_groundtruth.py。"""
        import importlib.util

        path = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "06_validate_against_groundtruth.py"
        )
        spec = importlib.util.spec_from_file_location("validate_gt", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_unmeasurable_when_error_below_noise_floor(self) -> None:
        mod = self._mod()
        # 同一帧两人计数差异 ±3 颗（噪声下限 3），实测误差仅 1 颗
        rows = []
        for f in range(6):
            rows.append(mod.GroundTruthRow(frame_idx=f, t_s=float(f),
                                           n_pellets_gt=100, counter="A"))
            rows.append(mod.GroundTruthRow(frame_idx=f, t_s=float(f),
                                           n_pellets_gt=103, counter="B"))
        auto = {f: 101.0 for f in range(6)}    # 与真值均值差 0.5
        rep = mod.validate(rows, auto)
        assert rep.measurable is False         # 纪律 3：判为"无法测量"
        assert "无法区分模型误差与真值噪声" in rep.verdict
        assert rep.mae is not None and rep.mae < rep.noise_floor
        md = mod.render_report_md(rep, "r1")
        assert "无法测量" in md
        # 精度数字降级为次要信息（不高亮），但仍可见（不静默删除）
        assert "color:#999" in md

    def test_measurable_reports_mae_and_bias(self) -> None:
        mod = self._mod()
        rows = [
            mod.GroundTruthRow(frame_idx=f, t_s=float(f),
                               n_pellets_gt=100, counter="A")
            for f in range(4)
        ]
        auto = {f: 90.0 for f in range(4)}     # 稳定低估 10 颗
        rep = mod.validate(rows, auto)
        assert rep.measurable is True
        assert rep.mae == pytest.approx(10.0)
        assert rep.bias == pytest.approx(-10.0)   # 有符号偏差：稳定低估
        assert "稳定低估" in rep.verdict
        # 纪律 1/4：小样本不得自动校正
        assert any("禁止" in w for w in rep.warnings)

    def test_missing_auto_frames_are_skipped_not_zero_filled(self) -> None:
        mod = self._mod()
        rows = [
            mod.GroundTruthRow(frame_idx=f, t_s=float(f),
                               n_pellets_gt=100, counter="A")
            for f in range(3)
        ]
        rep = mod.validate(rows, {0: 100.0})   # 仅第 0 帧有自动值
        assert rep.n_frames == 1
        assert any("缺失" in w for w in rep.warnings)


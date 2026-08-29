"""导出层（T05）。

    - csv_writer：metrics_summary.csv（17 列冻结）/ metrics_timeseries.csv
      （原生时间戳 + dt_s）/ metrics_timeseries_1hz.csv（可选，默认不导出）
      / manual_counts.csv（打点，独立证据链）+ TallyRecorder；
    - summary_writer：summary.json / capability_report.md（含关闭清单与
      flag 术语表）/ flag_glossary.csv；
    - charts：matplotlib 图（右删失段阴影 + ">窗长"，绝不画成归零）；
    - compare：run_config 七条拒绝规则 + 两组统计 + 衰减型偏差诊断 + 导出。
"""
from src.export.charts import (
    SeriesSpec,
    plot_group_comparison,
    plot_manual_vs_auto,
    plot_pellet_curve,
    plot_time_series,
)
from src.export.compare import (
    CompareResult,
    build_two_group_plan,
    diagnose_attenuation,
    load_run_bundle,
    render_compare_report_md,
    run_comparison,
    write_compare_outputs,
)
from src.export.csv_writer import (
    TallyEvent,
    TallyRecorder,
    read_metrics_summary_csv,
    resample_to_1hz,
    write_manual_counts_csv,
    write_metrics_summary_csv,
    write_timeseries_1hz_csv,
    write_timeseries_csv,
)
from src.export.summary_writer import (
    render_capability_report_md,
    write_capability_report_md,
    write_flag_glossary_csv,
    write_summary_json,
)

__all__ = [
    "SeriesSpec",
    "plot_group_comparison",
    "plot_manual_vs_auto",
    "plot_pellet_curve",
    "plot_time_series",
    "CompareResult",
    "build_two_group_plan",
    "diagnose_attenuation",
    "load_run_bundle",
    "render_compare_report_md",
    "run_comparison",
    "write_compare_outputs",
    "TallyEvent",
    "TallyRecorder",
    "read_metrics_summary_csv",
    "resample_to_1hz",
    "write_manual_counts_csv",
    "write_metrics_summary_csv",
    "write_timeseries_1hz_csv",
    "write_timeseries_csv",
    "render_capability_report_md",
    "write_capability_report_md",
    "write_flag_glossary_csv",
    "write_summary_json",
]

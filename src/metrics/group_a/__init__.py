"""A 组指标（T04，docs/04 §3.1 饲料/颗粒维度 A1–A14）。"""
from src.metrics.group_a.clearance import (
    clearance_metric,
    crossing_threshold,
    crossing_time,
)
from src.metrics.group_a.n0_estimator import early_window_s, estimate_n0, n0_gap
from src.metrics.group_a.nonfeeding_loss import (
    VanishTally,
    nonfeeding_loss_metric,
    tally_vanish,
)
from src.metrics.group_a.pellet_curve import (
    PelletSeries,
    TimeSeries,
    extract_pellet_series,
    monotonize,
    moving_median,
    rebound_fraction,
    smooth_series,
)
from src.metrics.group_a.rate import instantaneous_rate, rate_metrics
from src.metrics.group_a.residual import (
    fit_exponential,
    residual_metrics,
    trapz_integral,
)

__all__ = [
    "PelletSeries",
    "TimeSeries",
    "extract_pellet_series",
    "moving_median",
    "monotonize",
    "smooth_series",
    "rebound_fraction",
    "early_window_s",
    "estimate_n0",
    "n0_gap",
    "crossing_time",
    "crossing_threshold",
    "clearance_metric",
    "instantaneous_rate",
    "rate_metrics",
    "fit_exponential",
    "trapz_integral",
    "residual_metrics",
    "VanishTally",
    "tally_vanish",
    "nonfeeding_loss_metric",
]

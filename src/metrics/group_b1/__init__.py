"""B1 组（活跃度/空间异质性）指标模块（T04）。"""
from src.metrics.group_b1.reference_correction import (
    NO_NOISE_CORRECTION_FLAG,
    MIN_BASELINE_SAMPLES,
    ReferenceCorrection,
    ReferenceCorrectionResult,
    apply_reference_correction,
    estimate_alpha,
    no_reference_warning,
)
from src.metrics.group_b1.spatial_heterogeneity import (
    ActivityResult,
    GRID_N,
    compute_activity,
    d_group_metrics,
)

__all__ = [
    "ReferenceCorrection",
    "ReferenceCorrectionResult",
    "estimate_alpha",
    "apply_reference_correction",
    "no_reference_warning",
    "NO_NOISE_CORRECTION_FLAG",
    "MIN_BASELINE_SAMPLES",
    "ActivityResult",
    "GRID_N",
    "compute_activity",
    "d_group_metrics",
]

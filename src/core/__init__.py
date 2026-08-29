"""core · 契约层（T01）。

包含全部跨层共享的数据结构与约束点：
    config.py       RunConfig（可复现锚点，YAML 往返无损 + diff）
    metric_value.py MetricValue（全局约束点：写错了构造不出来）
    frame_context.py FrameObservation / 各类 Detections / RunMeta / BaselineStats
    roi.py          ROI 多边形（投喂区/参考区/排除区）
    homography.py   HomographyCalibrator（4 点标定 + 尺度场）

本模块只做无副作用的再导出，便于 `from src.core import MetricValue` 式引用。
"""

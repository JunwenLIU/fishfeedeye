"""鱼类摄食行为视频分析工具 · 源码包。

分层结构（docs/06 §1.3）：
    core/     契约层（T01，最先写、最严测试）
    pipeline/ 处理管线（T02/T03）
    metrics/  指标层（T04，业务核心）
    stats/    统计层（T05）
    export/   输出层（T05）
    app/      Gradio UI（T05）
    utils/    通用工具

本包 __init__ 保持轻量：不在此处触发任何重型依赖（cv2/torch 等）的导入。
"""

"""pytest 路径引导：把项目根加入 sys.path，使 `import src.*` 可用。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

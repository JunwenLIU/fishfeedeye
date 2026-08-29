"""tests 包标记：显式声明为正规包。

原因：site-packages 中某些第三方包（如 clip_anytorch）会携带顶层 `tests`
目录，与本项目 `tests/` 命名空间包冲突导致 `import tests.fixtures` 被
遮蔽。声明为正规包后，conftest.py 将项目根插入 sys.path[0]，本包优先。
"""

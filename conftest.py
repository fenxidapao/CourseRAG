"""pytest 全局收集前归一 profile：课程套件在 import 时就需要 APP_PROFILE=course。

不设这一行的后果（2026-11-02 实测）：无参数 `pytest` 时论文套件先被收集，
config 以 paper 导入，rag_core 把 rag_search_utils.ACTIVE_SOURCE_RULES 置空，
于是 test_api 的 4 条上传/删除与 test_core 的 3 条 route_sources 假失败。
论文套件（test_api_papers 等）在 fixture 里 monkeypatch config.APP_PROFILE，不依赖这里的值。
"""
import os

os.environ["APP_PROFILE"] = "course"

"""test_config_profile.py — profile 相关默认值的行为契约（v3 P0，2026-10-30）

config 在 import 时读环境变量，进程内改 env 无效，故每条断言都用子进程。
锁定的契约：
  - paper 默认不再跑生成后接地复核（8 GB 卡上 7B 生成 + 9B 裁判无法同驻）；
  - course 行为不变（默认 true）；
  - 显式 GROUNDED_VERIFY_ENABLED=true 仍能覆盖默认（开关保留可用）。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _grounded(profile, extra=None):
    env = os.environ.copy()
    env.pop("GROUNDED_VERIFY_ENABLED", None)
    env["APP_PROFILE"] = profile
    env.update(extra or {})
    proc = subprocess.run(
        [sys.executable, "-c", "import config; print(config.GROUNDED_VERIFY_ENABLED)"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_paper_default_off():
    assert _grounded("paper") == "False"


def test_course_default_on():
    assert _grounded("course") == "True"


def test_explicit_true_still_works():
    assert _grounded("paper", {"GROUNDED_VERIFY_ENABLED": "true"}) == "True"

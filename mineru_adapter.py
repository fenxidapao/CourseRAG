"""mineru_adapter.py — MinerU 2.7.6 子进程适配层。

主应用只通过本模块 subprocess 调用 MinerU，不 import mineru（避免 torch 版本冲突）。
CLI 探测顺序：venv 里的 mineru.exe → `python -m mineru`；参数以实际 --help 为准。
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import config

_DRIVER = None


def mineru_python():
    p = Path(config.MINERU_PYTHON)
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


def _env():
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if config.MINERU_MODEL_SOURCE:
        env["MINERU_MODEL_SOURCE"] = config.MINERU_MODEL_SOURCE
    return env


def _run(argv, timeout=None):
    t0 = time.time()
    proc = subprocess.run(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout, env=_env(),
    )
    out = (proc.stdout or b"").decode("utf-8", "replace")
    return proc.returncode, out, time.time() - t0


def driver():
    """返回 (命令前缀, --help 文本)。"""
    global _DRIVER
    if _DRIVER is not None:
        return _DRIVER
    py = mineru_python()
    candidates = []
    exe = py.parent / "mineru.exe"
    if exe.exists():
        candidates.append([str(exe)])
    candidates.append([str(py), "-m", "mineru"])
    for cmd in candidates:
        code, out, _ = _run(cmd + ["--help"], timeout=120)
        if code == 0:
            _DRIVER = (cmd, out)
            return _DRIVER
    raise RuntimeError(
        "MinerU CLI 不可用：先创建 venv-mineru 并安装 mineru[pipeline]==2.7.6"
    )


def version():
    code, out, _ = _run(
        [str(mineru_python()), "-c",
         "import importlib.metadata as m; print(m.version('mineru'))"],
        timeout=60,
    )
    return out.strip().splitlines()[-1] if code == 0 and out.strip() else ""


def resolve_device(requested=None):
    requested = (requested or config.MINERU_DEVICE or "auto").strip().lower()
    if requested != "auto":
        return requested
    code, out, _ = _run(
        [str(mineru_python()), "-c",
         "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
        timeout=120,
    )
    return out.strip().splitlines()[-1] if code == 0 and out.strip() else "cpu"


def build_parse_argv(pdf, out_dir, device=None, max_pages=None, lang=None):
    cmd, help_text = driver()
    argv = cmd + [
        "-p", str(pdf),
        "-o", str(out_dir),
        "-b", config.MINERU_BACKEND,
        "-l", lang or config.MINERU_LANG,
        "-d", resolve_device(device),
    ]
    if max_pages:
        if "--max-pages" in help_text:
            argv += ["--max-pages", str(int(max_pages))]
        elif "-e" in help_text or "--end" in help_text:
            argv += ["-e", str(int(max_pages) - 1)]
        else:
            print("[warn] MinerU CLI 无 max-pages 参数，忽略 --max-pages")
    src = (config.MINERU_MODEL_SOURCE or "").strip().lower()
    if src:
        if "--source" in help_text and src in ("huggingface", "modelscope", "local"):
            argv += ["--source", src]
        else:
            print(f"[warn] MinerU CLI 不支持 --source {src}，忽略")
    return argv


def find_outputs(out_dir):
    out_dir = Path(out_dir)

    def newest(pattern):
        items = sorted(out_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(items[0]) if items else None

    return {
        "content_list": newest("*content_list.json"),
        "middle": newest("*middle.json"),
        "markdown": newest("*.md"),
    }


def count_pages(middle_path, content_list_path=None):
    if middle_path and Path(middle_path).exists():
        try:
            data = json.loads(Path(middle_path).read_text(encoding="utf-8"))
            return len(data.get("pdf_info") or [])
        except Exception:
            pass
    if content_list_path and Path(content_list_path).exists():
        try:
            data = json.loads(Path(content_list_path).read_text(encoding="utf-8"))
            return max((int(x.get("page_idx", 0)) for x in data), default=-1) + 1
        except Exception:
            pass
    return 0


def parse(pdf, out_dir, device=None, max_pages=None, timeout=None, log_path=None, lang=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = build_parse_argv(pdf, out_dir, device=device, max_pages=max_pages, lang=lang)
    rc, log, elapsed = _run(argv, timeout=timeout or config.MINERU_TIMEOUT)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(log, encoding="utf-8")
    outputs = find_outputs(out_dir)
    outputs.update(
        {
            "returncode": rc,
            "log": log,
            "elapsed": elapsed,
            "argv": argv,
            "pages": count_pages(outputs["middle"], outputs["content_list"]),
        }
    )
    return outputs


if __name__ == "__main__":
    print("mineru_python:", mineru_python())
    print("version:", version())
    print("device:", resolve_device())
    try:
        cmd, help_text = driver()
        print("driver:", cmd)
        print(help_text[:800])
    except Exception as e:
        print("driver error:", e)

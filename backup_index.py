"""
backup_index.py — 索引版本化备份 / 恢复（工程化 P0：T10）

为什么需要它（见 2026-08-24 点评文档 P0「索引备份 / 版本化」）：
  当前索引目录是单点，重建一次要跑 OCR/切分/嵌入（成本不低）。没有备份，
  一次误删或 chroma 格式损坏（历史上有过 SQLite readonly 崩溃）就丢全部索引。
  生产落地必须有「可回滚的索引版本」。

用法：
  python backup_index.py backup            # 快照当前索引 -> chroma_db_backup_<ts>_<nchunks>/
  python backup_index.py list              # 列出所有备份（含块数/时间）
  python backup_index.py restore <name>    # 恢复指定备份（先自动备份当前态再覆盖）

实现：纯文件操作，仅 chromadb.PersistentClient 读块数，不触发 Ollama / 嵌入，可随时跑。
"""
import os
import sys
import json
import shutil
import datetime

import chromadb

import config

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(ROOT, config.PERSIST_DIRECTORY))
# profile 分目录：course 沿用旧目录，paper 单独放，避免 restore 覆盖错库
BACKUP_ROOT = os.path.join(
    ROOT,
    "chroma_db_backups" if config.APP_PROFILE == "course" else f"chroma_db_backups_{config.APP_PROFILE}",
)


def _now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _chunk_count():
    """读当前索引块数（不加载嵌入函数，只读集合计数）。"""
    if not os.path.isdir(SRC):
        return 0
    try:
        client = chromadb.PersistentClient(path=SRC)
        col = client.get_or_create_collection(config.COLLECTION_NAME)
        return col.count()
    except Exception as e:
        print(f"[warn] 读取块数失败（按 0 计）: {e!r}")
        return 0


def backup():
    if not os.path.isdir(SRC):
        print(f"✗ 当前没有索引目录（{SRC}），无法备份（先建库）。")
        return None
    n = _chunk_count()
    name = f"chroma_db_backup_{_now_ts()}_{n}chunks"
    dest = os.path.join(BACKUP_ROOT, name)
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    shutil.copytree(SRC, dest)
    manifest = {
        "name": name,
        "created": _now_ts(),
        "chunk_count": n,
        "source": os.path.basename(SRC),
    }
    with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"✓ 已备份 {n} 块 -> {dest}")
    return dest


def list_backups():
    if not os.path.isdir(BACKUP_ROOT):
        print("（无备份目录）")
        return []
    items = []
    for d in sorted(os.listdir(BACKUP_ROOT)):
        p = os.path.join(BACKUP_ROOT, d)
        if not os.path.isdir(p):
            continue
        man = os.path.join(p, "manifest.json")
        info = {"name": d}
        if os.path.exists(man):
            try:
                info.update(json.load(open(man, encoding="utf-8")))
            except Exception:
                pass
        items.append(info)
    if not items:
        print("（无备份）")
    for it in items:
        cc = it.get("chunk_count", "?")
        print(f"  {it['name']}  (块数={cc})")
    return items


def restore(name):
    """恢复备份。支持完整名或唯一子串片段（如 restore 225924 → 自动补全）。"""
    if not os.path.isdir(BACKUP_ROOT):
        print("✗ 没有备份目录，无法恢复。")
        return False
    names = [d for d in os.listdir(BACKUP_ROOT)
             if os.path.isdir(os.path.join(BACKUP_ROOT, d))]
    dest = os.path.join(BACKUP_ROOT, name)
    if not os.path.isdir(dest):
        # 子串匹配（时间戳片段 225924 在目录名中间，startswith 匹配不上；2026-08-24 实测修正）
        matches = [n for n in names if name in n]
        if len(matches) == 1:
            name = matches[0]
            dest = os.path.join(BACKUP_ROOT, name)
            print(f"  （按片段匹配到: {name}）")
        elif len(matches) > 1:
            print(f"✗ 片段 {name} 匹配到多个备份，请指定更精确片段：")
            for m in matches:
                print(f"    {m}")
            return False
        else:
            print(f"✗ 备份不存在: {name}")
            return False
    # 先备份当前态（仅当当前库可读且非空，避免残缺库产生垃圾备份）
    if os.path.isdir(SRC) and _chunk_count() > 0:
        current_backup = backup()
        print(f"  （恢复前已自动备份当前态: {os.path.basename(current_backup) if current_backup else 'n/a'}）")
    # 用 os.rename 原子移动旧库到临时名。
    # 注意：不要用 shutil.move——它在 Windows 上 rename 失败后会 fallback 到
    # copytree+rmtree，而 rmtree 遇被占用文件（如 uvicorn 持有 chroma.sqlite3）会
    # 中途失败 → 源目录被删残（2026-08-24 实测：chroma_db 只剩 sqlite3）。
    # os.rename 失败是原子的（不破坏源），直接提示先关占用进程。
    tmp = SRC + f".restore_tmp_{_now_ts()}"
    if os.path.exists(SRC):
        try:
            os.rename(SRC, tmp)
        except OSError as e:
            print(f"✗ 无法移动当前索引目录（{os.path.basename(SRC)}）——可能被 uvicorn/streamlit 等进程占用。")
            print(f"  请先关闭占用索引的服务（Ctrl+C）再重试；源目录未被改动。")
            print(f"  详情: {e!r}")
            return False
    shutil.copytree(dest, SRC)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"✓ 已恢复 {name} -> {os.path.basename(SRC)}")
    return True


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backup"
    if cmd == "backup":
        backup()
    elif cmd == "list":
        list_backups()
    elif cmd == "restore":
        if len(sys.argv) < 3:
            print("用法: python backup_index.py restore <name>")
            sys.exit(1)
        ok = restore(sys.argv[2])
        sys.exit(0 if ok else 1)
    else:
        print("用法: backup_index.py [backup|list|restore <name>]")
        sys.exit(1)


if __name__ == "__main__":
    main()

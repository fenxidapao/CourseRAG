"""`visual_rerank` 的离线单测：不调用 VLM、不需要 GPU / Ollama。

覆盖的都是**曾经真的坏过**的地方（见 `visual_rerank.py` 顶部说明），
加上词面图像通道这条纯本地的召回捷径。
所以这些用例可以在没有 Ollama、没有 CUDA 的机器上跑。

跑法：

    $py -m pytest test_visual_rerank.py -q
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import config  # noqa: E402
import visual_rerank as vr  # noqa: E402

_CAPTIONS = Path(vr._PROJECT_ROOT) / "data" / "images" / config.IMAGE_CAPTIONS_CSV
_HAVE_CAPTIONS = _CAPTIONS.is_file()
_needs_captions = pytest.mark.skipif(not _HAVE_CAPTIONS, reason="caption 语料缺失")


# ---------------------------------------------------------------------------
# bug 1：`unicode_escape` 把真中文打成乱码 → 证据门对所有中文问题永远拒绝
# ---------------------------------------------------------------------------
class TestUnescape:
    def test_real_chinese_is_left_untouched(self):
        """真 UTF-8 中文必须原样返回（这正是旧实现毁掉的那种输入）。"""
        s = "项目管理者, 产品组, 设计组, 开发组, 测试组"
        assert vr._unescape(s) == s

    def test_json_escapes_are_decoded(self):
        assert vr._unescape("\\u5c0f\\u7a0b\\u5e8f") == "小程序"

    def test_escaped_quote_is_decoded(self):
        assert vr._unescape('a\\"b') == 'a"b'

    def test_ascii_passthrough(self):
        assert vr._unescape("tab1.wxml, tab2.wxml, tab3.wxml") == "tab1.wxml, tab2.wxml, tab3.wxml"


class TestNormIdent:
    def test_strips_space_and_lowercases(self):
        """VLM 写成 `tab1. wxml` 也要能匹配上。"""
        assert vr._norm_ident("Tab1. WXML") == "tab1.wxml"


# ---------------------------------------------------------------------------
# bug 2 的门：文件名门必须要求**整个标识符**命中
# ---------------------------------------------------------------------------
class TestEvidenceGate:
    def test_filename_gate_rejects_decoy_png(self):
        """历史假阳性：问题问 avatar.png，VLM 抄的是 guest.png。
        旧门因两者共有 token `png` 而放行 → 无关图被置顶。"""
        assert vr._evidence_supports(
            "哪个微信小程序项目的目录里有 avatar.png 这个文件？", "images, guest.png"
        ) is False

    def test_filename_gate_accepts_exact_match(self):
        assert vr._evidence_supports(
            "哪个微信小程序项目的目录里有 avatar.png 这个文件？", "images, avatar.png"
        ) is True

    def test_empty_evidence_is_rejected(self):
        assert vr._evidence_supports("哪个项目的目录里有 avatar.png？", "") is False

    def test_fail_open_when_question_has_no_discriminative_word(self):
        """问题里全是泛化词（项目/结构/目录都在停用词表里）时，门必须放行而不是全拒。"""
        assert vr._evidence_supports("项目结构目录是什么？", "随便什么字都行") is True


# ---------------------------------------------------------------------------
# 词面图像通道：B3 之后「判别性文字已在 caption 里」的纯本地快路
# ---------------------------------------------------------------------------
class TestCaptionDf:
    @_needs_captions
    def test_discriminative_words_are_rare(self):
        df = vr._caption_df()
        assert df, "DF 表不该为空（caption 语料在盘上）"
        for w in ("个人信息", "比较数字大小", "用户登录", "电影列表"):
            assert df.get(w, 0.0) < vr._EVIDENCE_MAX_DF, f"{w} 的 DF={df.get(w)} 应按稀有词处理"

    @_needs_captions
    def test_generic_words_are_not_rare(self):
        """泛化词必须被 DF 门挡住，否则词面通道会把全库图片都捞进来。"""
        df = vr._caption_df()
        for w in ("项目", "结构"):
            assert df.get(w, 0.0) >= vr._EVIDENCE_MAX_DF, f"{w} 的 DF={df.get(w)} 不该当判别词"

    @_needs_captions
    def test_content_tokens_drops_generic_keeps_rare(self):
        toks = vr._content_tokens("比较数字大小这个项目的目录结构是什么？")
        assert toks, "判别词不该为空"
        assert "项目" not in toks and "结构" not in toks and "目录" not in toks, toks
        # 注意：jieba 会把「比较数字大小」拆成 比较/数字/大小 —— **这正是不加引号时
        # 唯一可用的粒度**，也是 `_QUOTED_RE` 存在的理由（见下一个用例）。
        assert {"比较", "数字", "大小"} <= toks, toks


class TestLexicalImageChannel:
    @_needs_captions
    def test_quoted_proper_noun_pulls_the_right_image(self):
        """「用户登录」是 p167_i00.png 像素里独有的字样，B3 之后进了 caption。
        向量通道分不开（`nomic-embed-text` 的已知天花板），词面通道应当分得开。"""
        docs = vr._lexical_image_channel("微信小程序里哪个项目的目录里有「用户登录」这个页面？", None)
        paths = [str(d.metadata.get("image_path")) for d in docs]
        assert paths, "词面通道一张图都没召回"
        assert any(p.endswith("微信小程序开发实战/p167_i00.png") for p in paths[:3]), \
            f"目标图不在前三：{paths[:3]}"

    @_needs_captions
    def test_filename_identifier_pulls_the_right_image(self):
        docs = vr._lexical_image_channel("哪个项目的目录里有 avatar.png 这个文件？", None)
        paths = [str(d.metadata.get("image_path")) for d in docs]
        assert paths, "词面通道一张图都没召回"
        assert any(p.endswith("微信小程序开发实战/p045_i00.png") for p in paths[:3]), \
            f"目标图不在前三：{paths[:3]}"

    @_needs_captions
    def test_unquoted_phrase_still_pulls_the_right_image(self):
        """不加引号时 jieba 会把「比较数字大小」拆成 比较/数字/大小，
        但通道是按**子串**匹配 caption 的，所以照样能命中 p082_i00.png。"""
        docs = vr._lexical_image_channel("哪个项目的目录里写着比较数字大小？", None)
        paths = [str(d.metadata.get("image_path")) for d in docs]
        assert paths, "词面通道一张图都没召回"
        assert any(p.endswith("微信小程序开发实战/p082_i00.png") for p in paths[:3]), \
            f"目标图不在前三：{paths[:3]}"

    @_needs_captions
    def test_generic_question_does_not_flood(self):
        """全是泛化词的问题不该把全库图片都捞出来（那等于没有召回）。"""
        docs = vr._lexical_image_channel("这个项目的目录结构是什么？", None)
        assert len(docs) == 0, f"泛化问题召回了 {len(docs)} 张图"


# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
class TestResolveImagePath:
    def test_empty_returns_none(self):
        assert vr._resolve_image_path("") is None

    def test_missing_file_returns_none(self):
        assert vr._resolve_image_path("data/images/不存在的书/p001_i00.png") is None

    @_needs_captions
    def test_resolves_a_real_asset(self):
        with open(_CAPTIONS, encoding="utf-8-sig") as f:
            png_path = next(csv.DictReader(f))["png_path"]
        resolved = vr._resolve_image_path(png_path)
        assert resolved is not None and resolved.is_file(), png_path

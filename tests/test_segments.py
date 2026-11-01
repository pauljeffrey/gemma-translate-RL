# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest

from gemma_mt.rl.segments import align, split_sentences as S


@pytest.mark.parametrize("text,n", [
    ("Hello. How are you? I am fine.", 3),
    ("It costs 3.5 dollars. OK.", 2),
    ("Dr. Smith went home. He slept.", 2),
    ("I met J. Smith at 5 p.m. today. Then we left.", 2),
    ("1. First item. 2. Second item.", 2),
    ("See www.example.org for more. Thanks.", 2),
    ("He said \"go home.\" She left.", 2),
    ("e.g. the dog. It barked.", 2),
    ("No terminator here", 1),
    ("", 0),
])
def test_sentence_counts(text, n):
    assert len(S(text)) == n


def test_newlines_are_plain_whitespace_and_ethiopic_full_stop_splits():
    assert S("Line one\nstill one. Second.") == ["Line one still one.", "Second."]
    assert S("እንዴት ነህ። ደህና ነኝ።") == ["እንዴት ነህ።", "ደህና ነኝ።"]


def test_align_requires_equal_counts_of_at_least_two():
    s, o, ok = align("A b. C d.", "E f. G h.")
    assert ok and s == ["A b.", "C d."] and o == ["E f.", "G h."]
    assert align("A b. C d.", "E f g h.")[2] is False
    assert align("One only.", "Eyi nikan.")[2] is False


def test_gale_church_handles_merged_split_and_omitted_sentences():
    from gemma_mt.rl.segments import align_chunks, gale_church

    src = "Aaa bbb ccc ddd eee. Ggg hhh iii jjj kkk. Mmm nnn ooo ppp qqq. Sss ttt uuu vvv www. Yyy zzz aaa bbb ccc."
    ch, failed, how = align_chunks(src, "Aaa bbb ccc ddd eee. Ggg hhh iii jjj kkk. Mmm nnn ooo ppp qqq. Sss ttt uuu vvv www. Yyy zzz aaa bbb ccc.")
    assert how == "aligned" and failed == 0 and len(ch) == 5
    ch, failed, how = align_chunks(src, "Aaa bbb ccc ddd eee, ggg hhh iii jjj kkk. Mmm nnn ooo ppp qqq. Sss ttt uuu vvv www, yyy zzz aaa bbb ccc.")
    assert how == "gale_church" and len(ch) == 3 and failed == 0
    ch, failed, how = align_chunks(src, "Aaa bbb ccc ddd eee, ggg hhh iii jjj kkk, mmm nnn ooo ppp qqq, sss ttt uuu vvv www, yyy zzz aaa bbb ccc.")
    assert how == "whole" and len(ch) == 1 and failed == 0                     # whole text, nothing counted as omitted
    # the beads always cover every source and every output sentence exactly once, in order
    beads = gale_church(["a b c d."] * 7, ["e f g h."] * 4)
    assert [i for si, _ in beads for i in si] == list(range(7)) and [j for _, oj in beads for j in oj] == list(range(4))

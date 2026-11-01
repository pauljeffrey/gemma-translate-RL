# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-59: the language floor of PromptSchedule (every language >= m prompts in every aligned block of P steps), default off, stateless, resume-safe."""
import collections

import numpy as np
import pytest

from gemma_mt.rl.data import MIXED, PromptSchedule

LANGS = "yor hau zul amh swh ibo lin kin som orm wol twi ewe fon bam sna xho afr nya sot tsn mlg tir lug nso tso ven ssw bem tum".split()
BINS = ("b64", "b256", "b512", "b1024", "b2048")


def _pool(per_cell=6, langs=LANGS):
    iso, lb, dom = [], [], []
    for d in ("science", "general"):
        for b in BINS:
            for l in langs:
                for _ in range(per_cell):
                    iso.append(l); lb.append(b); dom.append(d)
    return iso, lb, dom


def _sched(floor_steps=0, floor_prompts=1, weights=None, **kw):
    iso, lb, dom = _pool()
    # ONE weak-draw language (weight 0.01) so that without the floor it is almost never drawn
    lw = {LANGS[-1]: 0.01}
    lw.update(weights or {})
    return PromptSchedule(iso, alpha=0.0, seed=3, length_bin=lb, curriculum={"b64": 0}, domain=dom, domain_weights={"science": 0.7, "general": 0.3},
                          language_weights=lw, floor_steps=floor_steps, floor_prompts=floor_prompts, **kw), iso


def test_every_language_gets_the_floor_in_every_block_even_a_nearly_never_drawn_one():
    s, iso = _sched(floor_steps=10, floor_prompts=2)
    assert s.bin_of_step(0) == MIXED
    for block in range(6):
        c = collections.Counter(iso[i] for st in range(block * 10, block * 10 + 10) for i in s.indices(st, 15))
        assert set(c) == set(LANGS) and min(c.values()) >= 2, (block, c.most_common()[-3:])
    s0, iso0 = _sched()
    c0 = collections.Counter(iso0[i] for st in range(60) for i in s0.indices(st, 15))
    assert c0[LANGS[-1]] < 2 * 6          # without the floor the down-weighted language is nearly absent (the point of the test)


def test_floor_slots_are_spread_evenly_deterministic_and_stateless():
    s, _ = _sched(floor_steps=10, floor_prompts=1)
    sizes = [len(s.floor_slots(st)) for st in range(10)]
    assert sum(sizes) == len(LANGS) and max(sizes) - min(sizes) <= 1
    assert s.floor_slots(7) == s.floor_slots(7) and s.floor_slots(7) != s.floor_slots(8)
    assert s.floor_slots(7) == _sched(floor_steps=10, floor_prompts=1)[0].floor_slots(7)       # a restart (new object) gives the same stream
    a, b = s.indices(23, 20, rank=0, world=2), s.indices(23, 20, rank=1, world=2)
    assert sorted(a + b) == sorted(s.indices(23, 40))


def test_off_by_default_stream_is_bit_identical_and_floor_needs_the_domain_mixture():
    iso, lb, dom = _pool()
    a = PromptSchedule(iso, alpha=0.0, seed=3, length_bin=lb, curriculum={"b64": 0}, domain=dom, domain_weights={"science": 0.7, "general": 0.3})
    b = PromptSchedule(iso, alpha=0.0, seed=3, length_bin=lb, curriculum={"b64": 0}, domain=dom, domain_weights={"science": 0.7, "general": 0.3}, floor_steps=0)
    assert all(a.indices(st, 20) == b.indices(st, 20) for st in range(20))
    with pytest.raises(ValueError, match="domain mixture"):
        PromptSchedule(iso, alpha=0.0, seed=3, length_bin=lb, floor_steps=5)
    with pytest.raises(ValueError):
        PromptSchedule(iso, alpha=0.0, seed=3, length_bin=lb, domain=dom, domain_weights={"science": 1.0}, floor_steps=5, floor_prompts=0)


def test_too_few_prompts_per_step_for_the_floor_raises():
    s, _ = _sched(floor_steps=2, floor_prompts=1)       # 30 forced slots in 2 steps = 15 per step
    with pytest.raises(ValueError, match="forced prompts"):
        s.indices(0, 10)
    assert len(s.indices(0, 15)) == 15


def test_forced_language_falls_back_when_the_drawn_pool_lacks_it():
    iso, lb, dom = _pool()
    keep = [k for k in range(len(iso)) if not (iso[k] == LANGS[0] and dom[k] == "science" and lb[k] == "b64")]       # LANGS[0] has no science b64 prompt
    iso, lb, dom = [iso[k] for k in keep], [lb[k] for k in keep], [dom[k] for k in keep]
    s = PromptSchedule(iso, alpha=0.0, seed=1, length_bin=lb, curriculum={"b64": 0}, domain=dom, domain_weights={"science": 0.7, "general": 0.3},
                       domain_bin_weights={"science": {"b64": 1.0}}, floor_steps=3, floor_prompts=1)
    for st in range(9):
        assert len(s.indices(st, 20)) == 20                # no crash; the language is still drawn through the general pool most of the time

"""Whole finite gluing uses original structural maps and global predicates."""
import gc

import pytest

from rexgraph_cpp import (
    FiniteSearchCancellation,
    NativeFiniteFamily,
    NativeFiniteGluing,
    NativeFiniteSource,
    NativeInputUseAsset,
    NativeSource,
)


def source(side, extra=False):
    vertices = ((42,), (11, 42), (42, 99), (11, 42, 99))[side]
    relations = []
    if side in (1, 3):
        relations.append(dict(grade=1, key=50, type=0, occurrences=tuple(
            dict(key=i + 1, lower=v, negative=negative, magnitude=magnitude, exists=exists, aspects=(0,))
            for i, (v, negative, magnitude, exists) in enumerate(
                ((11, True, 1, True), (42, False, 1, True), (11, True, 2, False), (42, False, 0, True))))))
    if side in (2, 3):
        start = 5 if side == 3 else 1
        relations.append(dict(grade=1, key=51, type=0, occurrences=(
            dict(key=start, lower=42, negative=True, magnitude=1, exists=True, aspects=(0,)),
            dict(key=start + 1, lower=99, negative=False, magnitude=1, exists=True, aspects=(0,)))))
    owner = NativeSource.from_declaration(dict(profile='declared-role-source-v1', namespace=880100 + side,
        semantic_scale=2, max_grade=1, role_aspects=('boundary',),
        types=({'name': 'original pair and repeated flags', 'min_arity': 0, 'max_arity': 8},),
        witnesses=vertices, relations=tuple(relations), fields=(), metrics=(), carriers=()))
    if side == 0: return owner
    asset = NativeInputUseAsset(owner.archive, dict(profile='rexgraph.relational-input-use-asset.v1', key=1,
        channel=1, encoding='UTF-8', bytes=b'x', uses=(dict(key=1, cells=tuple((0, v) for v in vertices),
        occurrences=(), spans=((0, 1),)),)))
    base = {1: 101, 2: 201, 3: 501}[side]
    charts = []
    if side in (1, 3):
        charts.append(dict(key=1, name='left equality', patch=((0, 0), (0, 1)), support=((0, 0), (0, 1)),
            allowed=((base, base), (base + 1, base + 1))))
    if side in (2, 3):
        a, b = (0, 1) if side == 2 else (1, 2)
        charts.append(dict(key=2, name='right inequality', patch=((0, a), (0, b)), support=((0, a), (0, b)),
            allowed=((base, base + 1), (base + 1, base))))
    if extra: charts.append(dict(key=3, name='global observation', patch=((0, 0),), support=((0, 0),), allowed=((base,),)))
    return NativeFiniteSource(asset, dict(profile='rexgraph.relational-finite-source.v1', key=27, revision=1,
        symbols=({'key': base, 'definition': b'a'}, {'key': base + 1, 'definition': b'b'}),
        variables=tuple(dict(cell=(0, i), domain=(base, base + 1)) for i in range(len(vertices))),
        bindings=tuple(dict(key=i + 1, use=1, cell=(0, i)) for i in range(len(vertices))), charts=tuple(charts)))


def fixture():
    common, left, right, global_source = source(0), source(1), source(2), source(3, True)
    def mapping(cells, occurrences): return dict(cells=cells, occurrences=occurrences)
    maps = dict(key=1, revision=1,
        common_left=mapping(((1,), ()), ((), ())), common_right=mapping(((0,), ()), ((), ())),
        left_glued=mapping(((0, 1), (0,)), ((), (0, 1, 2, 3))),
        right_glued=mapping(((1, 2), (1,)), ((), (4, 5))))
    symbols = dict(key=2, revision=1, left=((501, 502), (501, 502)), right=((501, 502), (501, 502)))
    glue = NativeFiniteGluing(common.archive, left, right, global_source, maps, symbols)
    def family(owner, cell):
        return NativeFiniteFamily.projection(owner, dict(key=7, revision=1, cells=((0, cell),),
            window=None, fixed=(), distinct_relations=()))
    return glue, family(left, 1), family(right, 0), (common, left, right, global_source), maps, symbols


def test_glued_field_retains_actual_parent_owners_and_all_original_predicates():
    glue, left, right, owners, maps, symbols = fixture()
    assert glue.maps() == {'source_maps': maps, 'symbol_maps': symbols}
    a = left.extend((101,))['extension']; b = right.extend((201,))['extension']
    result = glue.glue(a, b)
    assert result['status'] == 'compatible'
    section = result['section']
    assert section.gluing().same_owner(glue)
    assert glue.source('left').same_owner(owners[1]) and glue.source('global').same_owner(owners[3])
    assert section.parent().family().same_owner(left) and section.parent(right=True).family().same_owner(right)
    assert section.select(0, 3) == ((0, 0, 501), (0, 1, 501), (0, 2, 502))
    assert section.inspect() == {'variable_count': 3, 'left_leaf': 0, 'right_leaf': 0}
    del glue, left, right, owners, a, b
    gc.collect()
    assert section.select(1, 1) == ((0, 1, 501),)


def test_pair_disagreement_and_global_refutation_are_distinct():
    glue, left, right, _, _, _ = fixture()
    a = left.extend((101,))['extension']; b = right.extend((202,))['extension']
    mismatch = glue.glue(a, b)
    assert mismatch['status'] == 'incompatible' and mismatch['section'] is None
    a = left.extend((102,))['extension']
    refuted = glue.glue(a, b)
    assert refuted['status'] == 'refuted_by_global_source' and refuted['section'] is None


def test_gluing_admission_cancellation_and_full_maps_are_checked():
    glue, left, right, owners, maps, symbols = fixture()
    a = left.extend((101,))['extension']; b = right.extend((201,))['extension']
    stop = FiniteSearchCancellation(); stop.cancel()
    cancelled = glue.glue(a, b, cancellation=stop)
    assert cancelled['status'] == 'cancelled' and cancelled['section'] is None
    refused = glue.glue(a, b, family_limits=(65536, 0, 16777216, 65536, 65536))
    assert refused['status'] == 'budget_exceeded' and refused['section'] is None
    with pytest.raises(ValueError):
        NativeFiniteGluing(owners[0].archive, *owners[1:], maps, {**symbols, 'left': ((501, 501), (501, 502))})
    with pytest.raises(ValueError):
        NativeFiniteGluing(owners[0].archive, *owners[1:], {**maps, 'left_glued': {'cells': ((0, 1), (0,)), 'occurrences': ((), ())}}, symbols)
    with pytest.raises((ValueError, RuntimeError)):
        glue.glue(a, b)['section'].select(0, 3, max_records=1)

"""Raw v2 packets retain canonical exclusions and complete original fields."""
from concurrent.futures import ThreadPoolExecutor
import gc
import struct

import pytest

from rexgraph_cpp import NativeFiniteFamily, NativeFiniteSource, PreparedFiniteSearch, FiniteSearchCancellation
from test_relational_finite_field import asset, declaration
from test_prepared_finite_search import projection


def decode_proof(packet):
    offset = 0

    def word():
        nonlocal offset
        if offset + 8 > len(packet):
            raise ValueError("truncated proof word")
        value = struct.unpack_from("<Q", packet, offset)[0]; offset += 8
        return value

    def sequence(read):
        count = word()
        if count > len(packet) // 8:
            raise ValueError("proof count exceeds the packet")
        return tuple(read() for _ in range(count))

    def words():
        return sequence(word)

    def blob():
        nonlocal offset
        size = word()
        if size > len(packet) - offset:
            raise ValueError("truncated proof bytes")
        value = packet[offset:offset + size]; offset += size
        return value

    pair = lambda: (word(), word())
    triple = lambda: (word(), word(), word())
    search_work = lambda: tuple(word() for _ in range(12))
    profile = blob(); source = blob(); key, revision = pair()
    cells = sequence(pair); window = sequence(words) if word() else None
    fixed, groups = sequence(triple), sequence(pair)
    source_bounds, maximum, limits, status = words(), word(), words(), word()
    work, native_work = tuple(word() for _ in range(15)), search_work()
    pruning = None
    if word():
        complete, enumeration = bool(word()), sequence(words)
        unsatisfiable = word() if word() else None
        def propagation():
            leaf, state, reused, executed = word(), word(), bool(word()), search_work()
            root = None
            if word():
                preparation, propagated, retained = search_work(), search_work(), word()
                root_fixed, root_groups = sequence(triple), sequence(pair)
                domains = sequence(lambda: (word(), word(), words()))
                root = dict(preparation=preparation, propagation=propagated, retained=retained,
                            fixed=root_fixed, groups=root_groups, domains=domains)
            return dict(leaf=leaf, status=state, reused=reused, work=executed, root=root)
        pruning = dict(complete=complete, window=enumeration, unsatisfiable_leaf=unsatisfiable,
                       propagations=sequence(propagation))
    def query():
        values, state, sources, steps = words(), word(), word(), word()
        decisive = word() if word() else None
        return dict(tuple=values, status=state, source_queries=sources, native_steps=steps, decisive_leaf=decisive)
    queries = sequence(query)
    image = None
    if word():
        image = sequence(lambda: (words(), sequence(lambda: sequence(triple))))
    if offset != len(packet):
        raise ValueError("unclaimed proof bytes")
    return dict(profile=profile, source=source, key=key, revision=revision, cells=cells, window=window,
        fixed=fixed, groups=groups, source_bounds=source_bounds, maximum=maximum, limits=limits,
        status=status, work=work, native_work=native_work, pruning=pruning, queries=queries, image=image)


def make(mode="unique"):
    d = declaration()
    if mode == "unique":
        d['charts'] = (dict(d['charts'][0], allowed=((102, 102),)), d['charts'][1])
    elif mode == "unsatisfiable":
        d['charts'] = (dict(d['charts'][0], allowed=((101, 101),)),
                       dict(d['charts'][1], allowed=((102, 102),)))
    return NativeFiniteSource(asset(), d)


def test_raw_v2_retains_full_request_limits_pruning_fields_and_whole_witness():
    source = make(); view = projection()
    family = NativeFiniteFamily.projection(source, view)
    limits = (65536, 64000000, 16777216, 65536, 65536)
    run = family.materialize(family_limits=limits); image = run['image']
    assert run['status'] == 'complete' and image.inspect() == dict(row_count=1, query_count=1)
    assert run['work']['propagations'] == 1 and run['work']['pruned_values'] == 4
    assert family.declaration() == view and family.inspect()['window'] == ((101, 102, 103),) * 2
    info = image.pruning(); root = image.pruning_root(0)
    assert info == run['queries'].pruning() and info['complete'] and info['limits'] == limits
    assert info['window'] == ((102,), (102,)) and info['unsatisfiable_leaf'] is None
    assert root.source().same_owner(source) and root.select_domains(0, 3) == tuple((0, i, (102,)) for i in range(3))
    packet = image.proof_archive(); decoded = decode_proof(packet)
    assert decoded['profile'] == b'rexgraph.finite-projection-proof.v2'
    assert decoded['source'] == source.archive() and decoded['window'] is None
    assert decoded['cells'] == view['cells'] and decoded['limits'] == limits and decoded['status'] == 0
    assert decoded['pruning']['window'] == info['window']
    retained = decoded['pruning']['propagations'][0]
    assert retained['leaf'] == 0 and retained['status'] == 0 and not retained['reused']
    assert retained['root']['domains'] == root.select_domains(0, 3)
    assert decoded['image'] == (((102, 102), (tuple((0, i, 102) for i in range(3)),)),)
    assert decoded['queries'][0]['tuple'] == (102, 102)
    archive = source.archive()
    del source, family, image, run; gc.collect()
    assert root.source().archive() == archive and root.continue_global_section()['status'] == 'satisfiable'


@pytest.mark.parametrize('mode,reason', [('empty', 'window'), ('unsatisfiable', 'source')])
def test_complete_empty_has_an_explicit_reason_not_a_vacuous_witness(mode, reason):
    source = make('unsatisfiable' if reason == 'source' else 'unique')
    view = projection(window=((), (101, 102, 103))) if mode == 'empty' else projection()
    run = NativeFiniteFamily.projection(source, view).materialize(); image = run['image']
    proof = decode_proof(image.proof_archive()); info = image.pruning()
    assert run['status'] == 'complete' and proof['image'] == () and info['complete']
    assert image.analyze_question(((0, 0),))['status'] == 'inconsistent_family'
    if reason == 'window':
        assert info['window'][0] == () and not info['propagations'] and info['unsatisfiable_leaf'] is None
    else:
        assert info['unsatisfiable_leaf'] == 0 and info['propagations'][0]['status'] == 'unsatisfiable'
        assert image.pruning_root(0) is None and proof['pruning']['propagations'][0]['root'] is None


@pytest.mark.parametrize('cancel', [False, True])
def test_interruption_has_no_image_or_complete_pruning_claim(cancel):
    source = make(); token = FiniteSearchCancellation()
    if cancel: token.cancel()
    run = NativeFiniteFamily.projection(source, projection()).materialize(
        family_limits=(65536, 64000000 if cancel else 0, 16777216, 65536, 65536), cancellation=token)
    assert run['image'] is None and run['status'] == ('cancelled' if cancel else 'budget_exceeded')
    proof = decode_proof(run['queries'].proof_archive())
    assert proof['image'] is None and (proof['pruning'] is None or not proof['pruning']['complete'])


def test_ambiguous_image_does_not_turn_propagation_domains_into_sections():
    source = make('ambiguous')
    run = NativeFiniteFamily.projection(source, projection()).materialize(); image = run['image']
    assert image.inspect() == dict(row_count=3, query_count=9)
    assert image.pruning()['window'] == ((101, 102, 103),) * 2
    assert image.analyze_question(((0, 0), (0, 2)))['answer_count'] == 3
    assert sum(q['status'] == 'satisfiable' for q in image.select_queries(0, 9)) == 3
    assert len(decode_proof(image.proof_archive())['image']) == 3


@pytest.mark.parametrize('profile', ['prepared', 'root'])
@pytest.mark.parametrize('steps', [0, 64000000])
def test_prepared_request_cannot_lose_its_origin_on_owner_roundtrip(profile, steps):
    source = make(); prepared = PreparedFiniteSearch.prepare(source)['prepared']
    owner = prepared if profile == 'prepared' else prepared.propagate()['snapshot']
    family = NativeFiniteFamily.projection(owner, projection())
    run = family.materialize(family_limits=(65536, steps, 16777216, 65536, 65536))
    for value in (run['image'], run['queries']):
        if value is None: continue
        with pytest.raises(ValueError, match='unprepared single-source'):
            value.proof_archive()
        nested = value.family().materialize()
        with pytest.raises(ValueError, match='unprepared single-source'):
            nested['image'].proof_archive()


def test_parent_origin_concurrent_readers_and_bounded_metadata():
    source = make(); raw = NativeFiniteFamily.projection(source, projection())
    prepared = NativeFiniteFamily.projection(PreparedFiniteSearch.prepare(source)['prepared'], projection())
    joined = NativeFiniteFamily.intersect_images(raw, prepared, dict(key=702, revision=1,
        right_positions=(0, 1), right_symbols=((101, 102, 103), (101, 102, 103))))
    image = joined.materialize()['image']
    with pytest.raises(ValueError, match='unprepared single-source'):
        image.proof_archive()
    assert decode_proof(image.family().parent().materialize()['image'].proof_archive())['status'] == 0
    with pytest.raises(ValueError, match='unprepared single-source'):
        image.family().parent(right=True).materialize()['image'].proof_archive()
    raw_image = raw.materialize()['image']; packet = raw_image.proof_archive()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: raw_image.proof_archive(), range(8))) == [packet] * 8
    with pytest.raises(RuntimeError, match='metadata bound exceeded'):
        raw_image.pruning(max_values=0)
    changed = raw_image.pruning(); changed['complete'] = False
    assert raw_image.pruning()['complete'] and raw_image.proof_archive() == packet

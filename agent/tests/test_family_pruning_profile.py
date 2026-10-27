"""Pruning evidence needs its actual provider and fresh source execution."""
from hashlib import sha256
from pathlib import Path

import pytest

@pytest.fixture
def setup(tmp_path, monkeypatch):
    pytest.importorskip('rexgraph_cpp')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'bindings/cpp/tests'))
    from agent import FamilyVerifier, RelationalSession
    from rcdb import open_store
    from rcql import Result
    from rexgraph_cpp import NativeFiniteSource
    from test_relational_finite_field import asset, declaration
    store = open_store('local://' + str(tmp_path / 'records'))
    session = RelationalSession(store); verifier = FamilyVerifier(session)
    def source(mode):
        assert mode == 'strong'
        d = declaration(); d['charts'] = (dict(d['charts'][0], allowed=((102, 102),)), d['charts'][1])
        session.save_result('source', Result((NativeFiniteSource(asset(), d).archive(),), aliases=('source',)))
        return verifier.pin('source')
    yield verifier, session, source
    store.close()


def question(verifier, source):
    return verifier.create_question('task', source=source, answer_cells=((0, 0),),
        observation_cells=((0, 2),), key=701)


@pytest.mark.parametrize('defect', ['old-proof-profile', 'mixed-engine'])
def test_incompatible_provider_is_rejected_before_saving_a_proof(setup, monkeypatch, defect):
    from rexgraph_cpp import _finite_search, _finite_field
    verifier, session, source = setup
    task = question(verifier, source('strong'))
    if defect == 'old-proof-profile':
        original = _finite_search.finite_search_build_info()
        monkeypatch.setattr(_finite_search, 'finite_search_build_info',
            lambda: dict(original, proof_profile='rexgraph.finite-projection-proof.v1'))
        error, match = NotImplementedError, 'pruning proof profile v2'
    else:
        original = _finite_field.finite_field_build_info()
        monkeypatch.setattr(_finite_field, 'finite_field_build_info',
            lambda: dict(original, engine_headers='0' * 64))
        error, match = ValueError, 'different engine snapshots'
    with pytest.raises(error, match=match):
        verifier.verify(task, proof_id='forbidden')
    assert session.store.get_record('forbidden') is None


@pytest.mark.parametrize('defect', ['packet', 'foreign-source'])
def test_persisted_or_foreign_packets_never_issue_authority(setup, defect):
    from rcql import Result
    from rexgraph_cpp import NativeFiniteFamily, NativeFiniteSource
    from test_prepared_finite_search import projection
    from test_relational_finite_field import asset, declaration
    verifier, session, source = setup
    task = question(verifier, source('strong'))
    issued = verifier.verify(task, proof_id='original')
    proof = dict(session.read('original').value.named_values['proof'])
    assert proof['provider']['search']['proof_profile'] == 'rexgraph.finite-projection-proof.v2'
    if defect == 'packet':
        raw = proof['native_proof']; packet = raw[:-1] + bytes([raw[-1] ^ 1])
    else:
        d = declaration(); d['charts'] = (dict(d['charts'][0], allowed=((102, 102),)), d['charts'][1])
        foreign = NativeFiniteSource(asset(namespace=900099), d)
        packet = NativeFiniteFamily.projection(foreign, projection()).materialize()['image'].proof_archive()
    proof.update(native_proof=packet, native_sha256=sha256(packet).hexdigest())
    session.save_result('forged', Result((proof,), aliases=('proof',)))
    pin = verifier.pin('forged')
    assert verifier.assess(pin).named_values['status'] == 'unissued'
    with pytest.raises(ValueError, match='replay differs'):
        verifier.replay(pin, proof_id='forbidden-replay')
    assert session.store.get_record('forbidden-replay') is None
    assert verifier.assess(issued).named_values['accepted'] is True


def test_owning_pruning_survives_release_but_needs_fresh_replay_for_a_grant(setup):
    verifier, session, source = setup
    task = question(verifier, source('strong'))
    pin = verifier.verify(task, proof_id='original')
    owner = verifier.proof_owner(pin)['image']; root = owner.pruning_root(0)
    assert owner.pruning()['complete'] and owner.inspect()['query_count'] == 1
    verifier.release(pin)
    assert verifier.assess(pin).named_values['status'] == 'unissued'
    assert root.select_domains(0, 3) == tuple((0, i, (102,)) for i in range(3))
    fresh = verifier.replay(pin, proof_id='replayed')
    assert verifier.assess(fresh).named_values['answer'] == (102,)
    assert session.read('original').value.named_values['proof']['native_proof'] == session.read('replayed').value.named_values['proof']['native_proof']

"""Owning native comparison proofs retain exact evidence and bank bytes."""
from concurrent.futures import ThreadPoolExecutor
import gc

import pytest

from rexgraph.cpp_stalk_pattern import (
    compile_stalk_context_pattern_bank, prepare_stalk_context_pattern_bank,
)
from rexgraph.cpp_stalk_parameter_relations import stalk_context_view_rules_query
from rexgraph.cpp_stalk_transport import _prepare
from test_stalk_context_view_rules import learn
from test_stalk_observed_patterns import observed_model


COORDINATES = {'weights': 101, 'parameters': 102, 'domains': (100, 103)}


def prepared_case(redundant=False):
    source, state, views = learn(redundant=redundant)
    query = stalk_context_view_rules_query(state, (0, 1), (), views, minimum_peers=1)
    return source, state, query, prepare_stalk_context_pattern_bank(state, source, query)


@pytest.mark.parametrize('redundant', [False, True])
def test_complete_proof_and_canonical_bank_match_existing_path(redundant):
    source, state, query, proof = prepared_case(redundant)
    full, _ = _prepare(state, source).contextual_view_rules(
        query['training'], (), query['frames'], minimum_peers=1)
    summary = proof.inspect()
    assert summary['rules'] == full['rules']
    assert summary['block_statuses'] == full['block_statuses']
    assert summary['block_ranks'] == full['block_ranks']
    assert summary['block_rows'] == tuple((b['origin'], b['target']) for b in full['blocks'])
    assert summary['equation_count'] == len(full['relations'])
    assert summary['archive_digest'] == full['archive_digest']
    assert summary['evidence_digest'] == state.coefficient_digest
    assert summary['proof_storage'] == 'owning-native'
    assert 'relations' not in summary
    expected = compile_stalk_context_pattern_bank(state, source, query, (0, 1),
        model_namespace=905, coordinate_ids=COORDINATES)
    actual = proof.compile((0, 1), model_namespace=905, coordinate_ids=COORDINATES)
    assert actual.archive() == expected.archive()
    assert actual.inspect() == expected.inspect()
    assert state.payload['fit']['free_columns']
    with pytest.raises(ValueError, match='outside'):
        actual.response({0: (0, 0, 1, 0)})


def test_summary_edits_and_parallel_compilation_cannot_change_evidence():
    source, state, query, proof = prepared_case()
    expected = proof.compile((0, 1), model_namespace=905, coordinate_ids=COORDINATES).archive()
    edited = proof.inspect()
    edited['rules'][0]['admitted'] = False
    edited['archive_digest'] = '0' * 64
    del source, state, query
    gc.collect()
    with ThreadPoolExecutor(max_workers=4) as pool:
        images = tuple(pool.map(lambda _: proof.compile((0, 1), model_namespace=905,
            coordinate_ids=COORDINATES).archive(), range(8)))
    assert images == (expected,) * 8
    assert all(r['admitted'] for r in proof.inspect()['rules'])


@pytest.mark.parametrize('options', [{}, {'partial': True}, {'conflict': True}])
def test_observed_input_image_and_peer_admissions_are_preserved(options):
    source, state, query = observed_model(**options)
    query = stalk_context_view_rules_query(state, query['training'], (), query['frames'],
        minimum_peers=1, observed_inputs=True)
    proof = prepare_stalk_context_pattern_bank(state, source, query)
    if options:
        assert not any(r['admitted'] for r in proof.inspect()['rules'])
        with pytest.raises(ValueError, match='admitted'):
            proof.compile((0, 1), model_namespace=907, coordinate_ids=COORDINATES)
    else:
        expected = compile_stalk_context_pattern_bank(state, source, query, (0, 1),
            model_namespace=907, coordinate_ids=COORDINATES)
        actual = proof.compile((0, 1), model_namespace=907, coordinate_ids=COORDINATES)
        assert actual.archive() == expected.archive()
        with pytest.raises(ValueError, match='outside'):
            actual.response({0: (1, 0, 0, 1)})


@pytest.mark.parametrize('origins', [(), (0, 0), (1, 0), (2,), (-1,), '0'])
def test_compile_requires_declared_admitted_training_origins(origins):
    *_, proof = prepared_case()
    with pytest.raises((ValueError, TypeError, OverflowError)):
        proof.compile(origins, model_namespace=905, coordinate_ids=COORDINATES)


def test_preparation_rejects_target_queries_and_uninitialized_owners():
    source, state, query, _ = prepared_case()
    held = stalk_context_view_rules_query(state, query['training'], (2,), query['frames'], minimum_peers=1)
    with pytest.raises(ValueError, match='without target'):
        prepare_stalk_context_pattern_bank(state, source, held)
    from rexgraph_cpp import PreparedStalkTransportPatternTraining
    empty = PreparedStalkTransportPatternTraining()
    with pytest.raises(RuntimeError, match='not initialized'):
        empty.inspect()
    with pytest.raises(RuntimeError, match='not initialized'):
        empty.compile((0,), model_namespace=905, coordinate_ids=COORDINATES)

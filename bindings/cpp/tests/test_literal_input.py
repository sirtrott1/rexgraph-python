"""Source derived literal fields, owning cold replay and actual grant checks."""
import gc

import pytest
from rexgraph_cpp import (
    PreparedFieldProgram, PreparedAccessionFieldProgram, NativeInputAccession,
    NativeLiteralInput, NativeValueOrigin, NativeAvailabilityContext,
)
from test_relational_field_program import source, tensor
from test_relational_field_evidence import learning_declaration, actual_asset, input_accession


def declaration(use=111):
    specs = (((1, 0), 4311, 'byte-equal', b'bank'),
             ((1, 0), 4312, 'byte-prefix', b'ba'),
             ((1, 1), 4411, 'byte-suffix', b'k'),
             ((1, 1), 4412, 'byte-punctuation', b''))
    return {'profile': 'rexgraph.relational-literal-input.v1', 'key': 8001, 'revision': 1, 'use': use,
            'coordinates': tuple({'key': i+1, 'cell': cell, 'direction': direction, 'use': use,
                                  'span_index': 0, 'span': (0, 4), 'kind': kind, 'operand': operand}
                                 for i, (cell, direction, kind, operand) in enumerate(specs))}


def prepared(raw=b'bank', use=111):
    owner = source()
    program = PreparedAccessionFieldProgram(PreparedFieldProgram(owner.archive, learning_declaration()).archive())
    asset = actual_asset(999993, use, raw)
    accession = NativeInputAccession(asset, program, input_accession())
    return owner, program, asset, accession


def test_native_source_derived_complete_boolean_field_and_lifetime():
    owner, program, asset, accession = prepared()
    value = NativeLiteralInput(accession, declaration())
    assert value.asset().same_owner(asset)
    assert value.program().archive() == program.archive()
    assert value.declaration() == declaration()
    native_input = value.input()
    result = program.execute((native_input,))
    assert result.field_archive(1) == tensor((1, 1, 1, 0))
    assert result.fields()[0]['supported'] == 4
    assert result.fields()[0]['nonzeros'] == 3
    before = value.archive()
    del owner, program, asset, accession, value
    gc.collect()
    assert native_input.inspect()['field'] == tensor((1, 1, 1, 0))
    cold = NativeLiteralInput.from_archive(before)
    assert cold.archive() == before
    assert cold.program().execute((cold.input(),)).field_archive(1) == tensor((1, 1, 1, 0))


def test_cold_owners_are_independent_and_canonical_packets_recompute():
    _, _, _, accession = prepared()
    value = NativeLiteralInput(accession, declaration()); cold = NativeLiteralInput.from_archive(value.archive())
    assert value.same_owner(value) and not cold.same_owner(value)
    assert cold.asset().archive() == value.asset().archive()
    assert not cold.asset().same_owner(value.asset())
    assert cold.declaration() == value.declaration()
    assert cold.input().inspect() == value.input().inspect()
    with pytest.raises(TypeError): value.__init__(accession, declaration())
    with pytest.raises((ValueError, RuntimeError)): value.declaration(max_records=3)


@pytest.mark.parametrize('kind,operand,span_index,expected', (
    ('byte-equal', b'', 1, 1), ('byte-prefix', b'', 1, 1), ('byte-suffix', b'', 1, 1),
    ('byte-punctuation', b'', 1, 0), ('byte-prefix', b'longer than span', 0, 0),
    ('byte-suffix', b'ank', 0, 1), ('byte-equal', b'ban', 2, 1),
))
def test_ordered_empty_and_overlapping_spans(kind, operand, span_index, expected):
    _, _, asset, accession = prepared(); d = declaration()
    row = d['coordinates'][0]; row.update(kind=kind, operand=operand, span_index=span_index,
                                         span=asset.uses()[0]['spans'][span_index])
    value = NativeLiteralInput(accession, d)
    assert value.input().inspect()['field'] == tensor((expected, 1, 1, 0))


@pytest.mark.parametrize('fault', ('missing', 'reordered', 'duplicate-key', 'wrong-cell', 'wrong-direction',
                                   'wrong-span', 'wrong-span-index', 'foreign-use', 'caller-truth',
                                   'unknown-kind', 'punctuation-operand', 'unknown-profile'))
def test_forged_incomplete_or_reordered_literal_declarations_refuse(fault):
    _, _, _, accession = prepared(); d = declaration()
    if fault == 'missing': d['coordinates'] = d['coordinates'][:-1]
    elif fault == 'reordered': d['coordinates'] = tuple(reversed(d['coordinates']))
    elif fault == 'duplicate-key': d['coordinates'][1]['key'] = 1
    elif fault == 'wrong-cell': d['coordinates'][0]['cell'] = (1, 1)
    elif fault == 'wrong-direction': d['coordinates'][0]['direction'] = 4411
    elif fault == 'wrong-span': d['coordinates'][0]['span'] = (0, 3)
    elif fault == 'wrong-span-index': d['coordinates'][0]['span_index'] = 99
    elif fault == 'foreign-use': d['coordinates'][0]['use'] = 99999
    elif fault == 'caller-truth': d['coordinates'][0]['result'] = True
    elif fault == 'unknown-kind': d['coordinates'][0]['kind'] = 'unicode-category'
    elif fault == 'punctuation-operand': d['coordinates'][-1]['operand'] = b'!'
    else: d['profile'] = 'untrusted-literal-v0'
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        NativeLiteralInput(accession, d)


def test_changed_literal_changes_actual_native_field_without_changing_program():
    _, first_program, _, first_accession = prepared()
    _, second_program, _, second_accession = prepared(b'bark')
    first = NativeLiteralInput(first_accession, declaration())
    second = NativeLiteralInput(second_accession, declaration())
    assert first_program.archive() == second_program.archive()
    assert first.input().inspect()['field'] != second.input().inspect()['field']
    assert first.archive() != second.archive()


def test_actual_literal_proof_origin_and_publication_cutoff_are_required():
    owner, _, _, accession = prepared(); literal = NativeLiteralInput(accession, declaration())
    proof = NativeValueOrigin(9001, owner.archive, owner.identity['sequence'], 'interface', literal.archive())
    grant = {'origin': proof, 'namespace': proof.inspect()['namespace'], 'sequence': proof.inspect()['sequence']}
    context = NativeAvailabilityContext((grant,))
    actual = context.require(9001, 'interface', literal.archive())
    assert actual.same_owner(proof)
    context.recheck(actual)
    clone = NativeValueOrigin(9001, owner.archive, owner.identity['sequence'], 'interface', literal.archive())
    assert clone.receipt() == proof.receipt() and not clone.same_owner(proof)
    with pytest.raises(ValueError): NativeAvailabilityContext(({**grant, 'origin': clone},)).recheck(proof)
    with pytest.raises(ValueError): NativeAvailabilityContext(()).require(9001, 'interface', literal.archive())
    with pytest.raises(ValueError): context.require(9001, 'evidence', literal.archive())
    with pytest.raises(ValueError): context.require(9001, 'interface', literal.archive()+b'changed')
    with pytest.raises(ValueError): NativeAvailabilityContext(({**grant, 'namespace': grant['namespace']+1},)).recheck(proof)
    with pytest.raises(ValueError): NativeAvailabilityContext(({**grant, 'sequence': grant['sequence']-1},)).recheck(proof)


def test_literal_archive_and_action_admission_are_bounded():
    _, _, _, accession = prepared(); value = NativeLiteralInput(accession, declaration())
    with pytest.raises((ValueError, RuntimeError)): NativeLiteralInput.from_archive(value.archive(), max_bytes=32)
    with pytest.raises((ValueError, RuntimeError)): NativeLiteralInput.from_archive(value.archive()+b'trailing')
    with pytest.raises((ValueError, RuntimeError)): NativeLiteralInput(accession, declaration(), limits=(1, 10, 1, 16))

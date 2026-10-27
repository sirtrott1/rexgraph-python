"""Rendered query code retains the complete typed query and execution contract."""
from dataclasses import replace
from fractions import Fraction
import subprocess
import sys

import pytest

from rcql import (Comparison, Executor, Literal, MutationQuery, OutputRef,
                  Parameter, Program, ProgramInput, ProgramStep, Query,
                  format_query, parse)
from rcql.executor import format_expr
from rcql.program_codec import dumps


@pytest.mark.parametrize('value', ['', 'ordinary', "a'b", 'a"b', 'a\\b\n\t\x00', 'α ℝ 𝄞'])
def test_expression_strings_use_the_parser_string_syntax(value):
    expression = format_expr(Literal(value))
    assert expression.startswith('"')
    assert parse('FROM $r RETURN ' + expression).returns == (Literal(value),)


def test_expression_negation_uses_prefix_syntax():
    query = parse('FROM $r RETURN NOT (1 = 2)')
    assert parse('FROM $r RETURN ' + format_expr(query.returns[0])) == query


@pytest.mark.parametrize('text', [
    'FROM $r RETURN 1, -3/7, 1/1, 1.0, -0.0, 1e-30, TRUE, FALSE, NONE',
    'FROM REX("a\\\\b") RETURN "quoted\\\"line\\n", [1,2/3,NONE]',
    'EXPLAIN FROM AT_TIME(REX("data"),1/3) RETURN COUNT(CELLS(1))',
    'FROM FILE("path with spaces.rcbf") RETURN COUNT(CELLS(0))',
    'FROM RCDB_VERSION(RCDB("data"),"record",2) RETURN STATE_HASH()',
    'FROM $r AS original LET x = original.CELLS(0) LET n = COUNT(x) RETURN n AS size',
    'FROM $r MATCH cell IN CELLS(0) WHERE NOT (cell.index < 1) RETURN cell AS selected ORDER BY cell.index DESC LIMIT 2 OFFSET 1',
    'FROM $r MATCH a IN CELLS(0), b IN CELLS(0) WHERE (a.index = 0 OR b.index >= 1) AND NOT (a.index = b.index) RETURN a,b ORDER BY a.index ASC, b.index DESC LIMIT 0',
    'FROM $r LET x = $input RETURN APPLY(DIRAC(),x,true)',
    'FROM $r RETURN MODEL_INFER($model,[$field,$contract])',
    'FROM $db MUTATE "row" SET state = $state, actor = "writer", expected_version = 0 COMMIT',
    'EXPLAIN FROM $db AS database LET x = $state MUTATE "row" SET actor = "a\\nb", valid_from = 1/3, valid_to = NONE, expected_version = 2, expected_hash = "abcd", state = x ADD $first REMOVE $second COMMIT',
])
def test_complete_query_roundtrip(text):
    original = parse(text)
    rendered = format_query(original)
    restored = parse(rendered)
    assert type(restored) is type(original) and restored == original
    if isinstance(original, Query):
        assert dumps(restored) == dumps(original)
    assert format_query(restored) == rendered


@pytest.mark.parametrize('value', [0, True, False, None, Fraction(1, 1), 1.0, -0.0])
def test_scalar_kinds_and_signed_zero_survive(value):
    original = Query(Parameter('r'), (Literal(value),))
    restored = parse(format_query(original)).returns[0].value
    assert type(restored) is type(value)
    assert restored.hex() == value.hex() if isinstance(value, float) else restored == value


def test_program_step_code_preserves_captures_links_and_contracts():
    original = Program('pipeline', (
        ProgramStep('first', parse('FROM $r RETURN $value')),
        ProgramStep('second', parse('FROM $r RETURN $previous'),
                    (('previous', OutputRef('first')),), (ProgramInput('previous'),))),
        (ProgramInput('value'),), (('answer', OutputRef('second')),)).specialize({'value': {'quoted': 'α\n"'}})
    restored = replace(original, steps=tuple(replace(s, query=parse(format_query(s.query))) for s in original.steps))
    assert restored.to_bytes() == original.to_bytes()
    from rexgraph import RexGraph
    source = RexGraph.from_graph([0], [1])
    before = Executor(sources={'r': source}, exactness='exact').execute_program(original)
    after = Executor(sources={'r': source}, exactness='exact').execute_program(restored)
    assert before.values == after.values == ({'quoted': 'α\n"'},)


@pytest.mark.parametrize('value', [b'raw', {'field': 1}, (1, 2), [1, 2], object()])
def test_values_without_literal_syntax_must_remain_parameters(value):
    with pytest.raises(TypeError, match='bind other values'):
        format_query(Query(Parameter('r'), (Literal(value),)))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_numbers_are_not_emitted_as_code(value):
    with pytest.raises(ValueError, match='finite'):
        format_query(Query(Parameter('r'), (Literal(value),)))


def test_roundtrip_check_refuses_nonrepresentable_unary_metadata():
    query = Query(Parameter('r'), (Comparison('NOT', Literal(True), Literal(0)),))
    with pytest.raises(ValueError, match='syntax tree'):
        format_query(query)


def test_mutation_remains_effectful_after_rendering():
    original = parse('FROM $db MUTATE "row" SET state = $state, expected_version = 0 COMMIT')
    restored = parse(format_query(original))
    assert isinstance(restored, MutationQuery)
    with pytest.raises(TypeError, match='ordinary typed queries'):
        ProgramStep('forbidden', restored)


def test_query_formatting_does_not_import_the_numeric_executor():
    code = ('import sys;from rcql import parse,format_query;'
            'format_query(parse(\'FROM REX("data") RETURN 1/3\'));'
            'assert "rcql.executor" not in sys.modules;'
            'assert "numpy" not in sys.modules;assert "rexgraph_cpp" not in sys.modules')
    subprocess.run([sys.executable, '-I', '-B', '-c', code], check=True, capture_output=True)

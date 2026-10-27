"""Native runtime operations expose closed syntax and explicit source policy."""
import pytest


def test_runtime_operator_named_arguments_are_declared_and_duplicate_names_rejected():
    from rcql import parse
    from rcql.arguments import EXPRESSION_ARGUMENTS
    from rcql.operators import get_operator
    for name in ("RELATIONAL_CREATE", "RELATIONAL_EXTEND", "RELATIONAL_INFO", "RELATIONAL_EVALUATE",
                 "RELATIONAL_PACKET", "RELATIONAL_FIELD_FUNCTION", "RELATIONAL_FIELD_INFO"):
        assert get_operator(name).name == name and name in EXPRESSION_ARGUMENTS
    parse('FROM $r RETURN RELATIONAL_INFO(packet=$p,observation="function",identity=1)')
    parse('FROM $r RETURN RELATIONAL_EVALUATE(packet=$p,parameter=1/3,identity=1,tensor=false)')
    with pytest.raises((ValueError, SyntaxError)):
        parse('FROM $r RETURN RELATIONAL_INFO(packet=$p,packet=$other)')


def test_missing_source_permission_precedes_native_executor_import():
    from rcql import BoundSource, Executor, SourcePolicy, parse
    from rexgraph import RexGraph
    executor = Executor(sources={"r": BoundSource(RexGraph.from_cells([2, [[0, 1]]]), SourcePolicy.allow())})
    with pytest.raises(PermissionError):
        executor.execute(parse('FROM $r RETURN RELATIONAL_CREATE(namespace=89007)'))

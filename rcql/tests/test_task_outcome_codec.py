"""Portable outcome content is distinct from native assessment authority."""
from dataclasses import FrozenInstanceError, replace

import pytest
from rexgraph.source_binding import CppSourceIdentity, CppSourceReference

from rcql import Result, TaskOutcome


@pytest.mark.parametrize('named,positional', (
    ('TASK_OUTCOME(binding=$binding,execution_context=$authority,asset=$asset,task=$task,output=$output)',
     'TASK_OUTCOME($output,$task,$asset,$binding,$authority)'),
    ('TASK_STATUS(execution_context=$authority,outcome=$outcome)', 'TASK_STATUS($outcome,$authority)'),
    ('TASK_VERDICT(execution_context=$authority,outcome=$outcome)', 'TASK_VERDICT($outcome,$authority)'),
    ('TASK_INFO(execution_context=$authority,outcome=$outcome)', 'TASK_INFO($outcome,$authority)'),
    ('TASK_CERTIFICATE(execution_context=$authority,outcome=$outcome)', 'TASK_CERTIFICATE($outcome,$authority)'),
    ('TASK_FIELD(occurrences=true,execution_context=$authority,node=3,outcome=$outcome)',
     'TASK_FIELD($outcome,3,$authority,true)'),
))
def test_task_named_arguments_preserve_declared_order(named, positional):
    from rcql import parse
    assert parse('FROM $r RETURN '+named) == parse('FROM $r RETURN '+positional)


def passive():
    source = CppSourceReference(None, native=CppSourceIdentity(8, 1, 1, "a" * 64, "b" * 64, (("10", "20"), ("100",))))
    provider = tuple(sorted({"engine": "rexgraph-cpp", "engine_version": "0.2.1", "engine_headers": "c" * 64,
        "binding": "d" * 64, "profile": "owning-relational-task-outcome-v1", "scalar_policy": "gmp-rational-exact"}.items()))
    return TaskOutcome(b"passive-unassessed-native-content!", provider, source, (source,))


def test_passive_outcome_and_result_roundtrip_without_native_execution(monkeypatch):
    import rcql.task_outcome as module
    value = passive()

    def unavailable():
        raise AssertionError("portable decoding must not restore native owners")
    monkeypatch.setattr(module, "_binding", unavailable)
    restored = TaskOutcome.from_record(value.to_record())
    assert restored.coefficient_digest == value.coefficient_digest and restored._owner is None
    packet = Result((value,), aliases=("outcome",)).to_bytes()
    cold = Result.from_bytes(packet)
    assert cold.to_bytes() == packet
    assert cold.values[0].to_record() == value.to_record()
    assert cold.values[0].source.source is None
    with pytest.raises(FrozenInstanceError):
        value.archive = b"changed"


def test_passive_archive_checksum_and_closed_schema_are_checked():
    value = passive()
    for record in ({**value.to_record(), "archive": b"changed"}, {**value.to_record(), "extra": 1}):
        with pytest.raises(ValueError):
            TaskOutcome.from_record(record)
    with pytest.raises(ValueError):
        replace(value, limits=(1, 2, 3, 0))
    with pytest.raises(ValueError):
        replace(value, provider=value.provider + value.provider[:1])


def test_current_caller_caps_cannot_be_replaced_by_stored_policy(monkeypatch):
    import rcql.task_outcome as module
    monkeypatch.setattr(module, "_binding", lambda: pytest.fail("caller cap must precede native restoration"))
    value = passive()
    with pytest.raises(ValueError, match="current caller"):
        value.native(object(), limits=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="current caller"):
        value.native(object(), max_bytes=1)


def test_missing_native_provider_refuses_assessment(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "rexgraph_cpp", None)
    with pytest.raises(ImportError, match="native task outcome"):
        passive().native(object())


def test_native_outcome_keeps_intrinsic_publications_during_strict_python_source_binding():
    from rexgraph.tensor_field import FieldSource

    from rcql.result_codec import bind_sources

    value = passive()
    assert bind_sources({"outcome": value}, {}, strict=True)["outcome"] is value
    assert value._owner is None and value.source.source is None
    assert len(value.dependencies) == 1 and value.dependencies[0].matches(value.source)
    with pytest.raises(ValueError, match="unbound source"):
        bind_sources(FieldSource(None, "missing", 1, "a" * 64), {}, strict=True)

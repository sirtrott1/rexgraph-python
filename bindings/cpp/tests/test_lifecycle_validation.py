"""Publication ownership, concurrent model reuse and checked application views."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from rexgraph_cpp import NativeSource
from test_construction import ROWS, declaration, model

from rexgraph.cpp_perceptron import cpp_perceptron_input
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import ModelBatch, model_input
from rexgraph.source_binding import CppSourceIdentity, CppSourceReference, source_from_record


@pytest.mark.parametrize("name", ["namespace", "sequence", "first_sequence", "archive_digest",
                                 "endpoint_digest", "codec", "coordinates"])
def test_bound_publications_reject_every_changed_identity_field(name):
    source = NativeSource.from_declaration(declaration())
    ref = CppSourceReference(source)
    altered = replace(ref.native)
    value = getattr(altered, name)
    value = (value + 1 if isinstance(value, int) else "0" * 64 if name.endswith("digest") else
             "unknown" if name == "codec" else tuple(reversed(value)))
    object.__setattr__(altered, name, value)
    with pytest.raises(ValueError, match="declared publication"):
        CppSourceReference(source, native=altered)
    object.__setattr__(ref, "native", altered)
    with pytest.raises(ValueError, match="identity changed"):
        ref.check()


def test_bound_identity_normalizes_equal_boolean_cutoff():
    source = NativeSource.from_declaration(declaration())
    ref = CppSourceReference(source)
    altered = replace(ref.native)
    assert altered.sequence == 0
    object.__setattr__(altered, "sequence", False)
    bound = CppSourceReference(source, native=altered)
    assert type(bound.native.sequence) is int and bound.native.sequence == 0


@pytest.mark.parametrize("fault", ["namespace", "coordinates", "extra"])
def test_source_subclasses_retain_full_record_validation(fault):
    class AlteredSource(NativeSource):
        @property
        def identity(self):
            record = super().identity
            if fault == "namespace": record["namespace"] = True
            elif fault == "coordinates": record["coordinates"] = (("01",),)
            else: record["unknown"] = 1
            return record
    source = NativeSource.from_declaration(declaration())
    altered = AlteredSource(source.archive)
    with pytest.raises(ValueError):
        CppSourceReference(altered, native=CppSourceReference(source).native)


def test_identity_records_and_caller_sequences_are_independent():
    source = NativeSource.from_declaration(declaration())
    ref = CppSourceReference(source)
    record = ref.as_record()
    record["native"]["namespace"] += 1
    ref.check()
    assert source.identity == ref.native.as_record()
    keys = [["10", "11"], []]
    value = CppSourceIdentity(1, 0, 0, "a" * 64, "b" * 64, keys)
    keys[0][0] = "12"
    assert value.coordinates == (("10", "11"), ())
    detached = source_from_record(ref.as_record(), allow_cpp=True)
    reloaded = NativeSource(source.archive)
    assert detached.bind(reloaded).matches(ref)
    assert reloaded.identity["coordinates"] == source.identity["coordinates"]
    assert reloaded.identity["coordinates"] is not source.identity["coordinates"]


@pytest.mark.parametrize("storage", ["values", "packed"])
def test_concurrent_core_calls_preserve_models_and_output_seals(storage):
    source = NativeSource.from_declaration(declaration())
    state = model(source, parameter_storage=storage)
    inputs = cpp_perceptron_input(state, source, ROWS, prepare=True, cpu_workers=2, rows_per_tile=1)
    before = state.coefficient_digest
    expected = infer_model(state, source, inputs)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: infer_model(state, source, inputs), range(16)))
    assert all(v.coefficient_digest == expected.coefficient_digest for v in results)
    assert all(v.execution_receipt == expected.execution_receipt for v in results)
    assert state.coefficient_digest == before
    assert unpack_model(pack_model(state, native=True)).coefficient_digest == before


@pytest.mark.parametrize("kind", ["model", "output", "batch", "input"])
def test_query_typing_and_system_views_reject_changed_declarations(kind):
    from rcql.binding import bind
    from rcql.capabilities import SourcePolicy
    from rcql.model_contracts import model_literal
    from system.serialize import json_value

    from rexgraph.io.model_record import model_record
    source = NativeSource.from_declaration(declaration())
    state = model(source)
    output = infer_model(state, source, cpp_perceptron_input(state, source, ROWS, prepare=True))
    value = (state if kind == "model" else output if kind == "output" else
             ModelBatch(state.source, state.space, [1, -1, 1], [True] * 3) if kind == "batch" else
             model_input(output, "float64"))
    binding = bind("record", model_record(source, state), SourcePolicy.allow("read"))
    assert model_literal(binding, value).declaration_digest == value.coefficient_digest
    assert json_value(value)["digest"] == value.coefficient_digest
    if kind == "model": object.__setattr__(value, "step", 1)
    elif kind == "batch": object.__setattr__(value, "targets", np.zeros(3, dtype=object))
    else:
        value.values.flags.writeable = True
        value.values[0] += 1
    with pytest.raises(ValueError): model_literal(binding, value)
    with pytest.raises(ValueError): json_value(value)

"""Separately supplied provider authority and unchanged portable model records."""
from fractions import Fraction as Q

import numpy as np
import pytest

import rexgraph.model_contract as contract
from rexgraph import RexGraph
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, invoke_model_provider, model_coordinates
from rexgraph.model_runtime import infer_model, native_model, train_model, transport_model
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState
from rexgraph.tensor_field import FieldSource
from rexgraph.value_codec import pack_value, unpack_value


@pytest.fixture(autouse=True)
def isolated_providers(monkeypatch):
    monkeypatch.setattr(contract, "_model_providers", {})


def fixture():
    source = RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2])
    state = ModelState("context_fixture", "1", FieldSource(source), model_coordinates(source), (),
                       {"execution_provider": {"name": "context_fixture", "version": "1"}}, {}, "rational")
    return source, state


def output(state):
    return ModelOutput([Q(1, 3)] * len(state.space.keys), state.space, (), state.source,
                       state.coefficient_digest, "rational", "context-fixture")


@pytest.mark.parametrize("operation", ["create", "prepare", "infer", "train", "transport", "export", "restore"])
def test_context_operations_reject_missing_authority_before_callback(operation):
    calls = []
    callbacks = {operation: lambda *a, **kw: calls.append(kw)}
    callbacks.setdefault("infer", lambda *a: None)
    provider = ModelProvider("context_fixture", "1", (("context_fixture", "1"),),
                             context_operations=(operation,), **callbacks)
    with pytest.raises(ValueError, match="separate execution context"):
        invoke_model_provider(provider, operation)
    assert calls == []


@pytest.mark.parametrize("declaration,error", [
    ("infer", TypeError), (("infer", "infer"), ValueError),
    (("unknown",), ValueError), (("prepare",), ValueError),
])
def test_context_contract_rejects_invalid_or_unimplemented_operations(declaration, error):
    with pytest.raises(error):
        ModelProvider("context_fixture", "1", (("context_fixture", "1"),), lambda *a: None,
                      context_operations=declaration)


def test_context_inference_is_explicit_rechecked_and_never_serialized():
    source, state = fixture()
    available, withheld = object(), object()
    calls = []

    def infer(selected, current, inputs, *, execution_context):
        calls.append(execution_context)
        if execution_context is not available:
            raise ValueError("authority withheld")
        return output(selected)

    contract.register_model_provider(ModelProvider("context_fixture", "1", ((state.adapter, "1"),), infer,
                                                   context_operations=("infer",)))
    saved = pack_value(pack_model(state, native=True))
    restored = unpack_model(unpack_value(saved))
    assert pack_value(pack_model(restored, native=True)) == saved
    with pytest.raises(ValueError, match="separate execution context"):
        infer_model(restored, source)
    result = infer_model(restored, source, execution_context=available)
    assert tuple(result.values) == (Q(1, 3), Q(1, 3))
    with pytest.raises(ValueError, match="authority withheld"):
        infer_model(restored, source, execution_context=withheld)
    assert calls == [available, withheld]
    assert pack_value(pack_model(restored, native=True)) == saved
    assert contract.model_provider_inventory()[0]["context_operations"] == ("infer",)


def test_context_training_retains_immutable_child_and_source_contract():
    source, state = fixture()
    authority = object()
    calls = []

    def update(selected, current, batch, *, steps, execution_context):
        assert execution_context is authority
        calls.append(steps)
        return selected.child(payload={"updates": steps}, step=selected.step + steps)

    provider = ModelProvider("context_fixture", "1", ((state.adapter, "1"),), lambda *a: output(a[0]),
                             train=update, context_operations=("train",))
    contract.register_model_provider(provider)
    batch = ModelBatch(state.source, state.space, np.array([0, 1]), np.array([True, False]))
    with pytest.raises(ValueError, match="separate execution context"):
        train_model(state, source, batch)
    child = train_model(state, source, batch, steps=2, execution_context=authority)
    assert calls == [2] and child.parent == state.coefficient_digest and child.step == 2
    assert child.source.matches(state.source) and child.space == state.space
    assert state.payload == {}


def test_context_transport_receives_original_destination_and_mapping():
    source, state = fixture()
    destination = FieldSource(source)
    mapping, authority = object(), object()
    calls = []

    def move(selected, current, target, actual_map, *, optimizer, execution_context):
        calls.append((target, actual_map, optimizer, execution_context))
        return selected.child(payload=selected.payload, source=target)

    contract.register_model_provider(ModelProvider("context_fixture", "1", ((state.adapter, "1"),),
                                                   lambda *a: output(a[0]), transport=move,
                                                   context_operations=("transport",)))
    with pytest.raises(ValueError, match="separate execution context"):
        transport_model(state, source, destination, mapping)
    child = transport_model(state, source, destination, mapping, execution_context=authority)
    assert calls == [(destination, mapping, "reset", authority)]
    assert child.parent == state.coefficient_digest and child.step == state.step


def test_legacy_provider_callback_and_inventory_remain_unchanged():
    source, state = fixture()
    calls = []

    def infer(selected, current, inputs):
        calls.append(inputs)
        return output(selected)

    provider = ModelProvider("context_fixture", "1", ((state.adapter, "1"),), infer)
    contract.register_model_provider(provider)
    with pytest.raises(TypeError, match="does not admit an execution context"):
        infer_model(state, source, execution_context=object())
    assert calls == []
    assert infer_model(state, source).model_digest == state.coefficient_digest
    assert calls == [None]
    assert set(contract.model_provider_inventory()[0]) == {"name", "version", "adapters", "arithmetic", "operations"}


def test_builtin_model_never_ignores_supplied_context():
    source, _ = fixture()
    state = native_model(source)
    with pytest.raises(TypeError, match="does not admit an execution context"):
        infer_model(state, source, execution_context=object())


def test_context_operation_declarations_copy_caller_metadata():
    names = ["infer"]
    provider = ModelProvider("context_fixture", "1", (("context_fixture", "1"),), lambda *a, **k: None,
                             context_operations=names)
    names.append("train")
    assert provider.context_operations == ("infer",)
    with pytest.raises(TypeError):
        invoke_model_provider(object(), "infer")
    with pytest.raises(ValueError, match="unknown"):
        invoke_model_provider(provider, "__dict__", execution_context=object())


@pytest.mark.parametrize("contextual", [False, True])
def test_provider_operation_configuration_does_not_replace_dispatch(contextual):
    calls = []
    authority = object()

    def create(source, *, operation, **kwargs):
        calls.append((source, operation, kwargs))
        return operation

    provider = ModelProvider("operation_fixture", "1", (("operation_fixture", "1"),),
                             lambda *a: None, create=create,
                             context_operations=("create",) if contextual else ())
    context = authority if contextual else None
    # This is the provider's mathematical operation, not a second dispatch selector.
    assert invoke_model_provider(provider, "create", "source", operation="restore",
                                 execution_context=context) == "restore"
    assert calls == [("source", "restore", {"execution_context": authority} if contextual else {})]


@pytest.mark.parametrize('contextual_prepare',[False,True])
def test_native_record_source_binding_and_preparation_context_are_separate(contextual_prepare, monkeypatch):
    from pathlib import Path
    native=pytest.importorskip('rexgraph_cpp')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'bindings/cpp/tests'))
    from test_construction import declaration
    from rexgraph.cpp_model import cpp_model
    from rexgraph.io.model_record import model_record,read_model_record
    from dataclasses import replace
    source=native.NativeSource.from_declaration(declaration())
    state=replace(cpp_model(source,(10,11,12)),adapter='record_context',
        configuration={'execution_provider':{'name':'record_context','version':'1'}})
    authority=object();calls=[]
    def prepare(selected,current,**kwargs):
        calls.append(kwargs)
        if contextual_prepare and kwargs.get('execution_context') is not authority:
            raise ValueError('wrong preparation authority')
    provider=ModelProvider('record_context','1',(('record_context','1'),),lambda *a,**k:None,prepare=prepare,
        context_operations=('infer','prepare') if contextual_prepare else ('infer',))
    contract.register_model_provider(provider)
    record=model_record(source,state)
    detached,owner=read_model_record(record,_prepare=False)
    assert detached.source.matches(state.source) and owner.identity==source.identity and not calls
    if contextual_prepare:
        with pytest.raises(ValueError,match='separate execution context'):read_model_record(record)
    restored,owner=read_model_record(record,execution_context=authority)
    assert restored.source.matches(state.source)
    assert calls==[{'execution_context':authority} if contextual_prepare else {}]

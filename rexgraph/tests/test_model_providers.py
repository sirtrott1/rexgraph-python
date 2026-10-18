"""Execution provider dispatch, source validation and checkpoint declarations."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

import rexgraph.model_contract as contract
from rexgraph import RexGraph
from rexgraph.model_contract import ModelProvider, model_coordinates, register_model_provider
from rexgraph.model_runtime import infer_model, train_model, transport_model
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState
from rexgraph.tensor_field import FieldSource, TensorField


@pytest.fixture(autouse=True)
def isolated_providers(monkeypatch):
    monkeypatch.setattr(contract, "_model_providers", {})


def source(ids=(11, 12)):
    return RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2], relation_ids=np.array(ids))


def model(rex, declaration=None, **kwargs):
    declaration = {"name": "contract_fixture", "version": "1"} if declaration is None else declaration
    return ModelState("provider_field", "1", FieldSource(rex), model_coordinates(rex), (),
                      {"execution_provider": declaration}, {"gain": Q(3, 2)},
                      kwargs.get("arithmetic", "rational"))


def inputs(state, rex):
    return TensorField(state.space, [Q(1, 3), Q(-2, 5)], source=FieldSource(rex),
                       grade=1, variance="chain")


def forward(state, rex, values):
    values = values.bind(rex)
    return ModelOutput(values.values * state.payload["gain"], values.space, values.axes,
                       state.source, state.coefficient_digest, "rational", "provider-field-fixture",
                       grade=values.grade, variance=values.variance)


def update(state, rex, batch, *, steps):
    return state.child(payload={"gain": state.payload["gain"] + steps}, step=state.step + steps)


def provider(**kwargs):
    return ModelProvider("contract_fixture", "1", (("provider_field", "1"),), forward, **kwargs)


def test_provider_inference_preserves_exact_values_and_identity():
    rex = source()
    state = model(rex)
    register_model_provider(provider())
    out = infer_model(state, rex, inputs(state, rex))
    assert tuple(out.values) == (Q(1, 2), Q(-3, 5))
    assert out.model_digest == state.coefficient_digest
    assert out.source.matches(state.source)
    assert out.arithmetic == "rational" and out.grade == 1 and out.variance == "chain"


def test_provider_selection_fails_before_legacy_fallback():
    rex = source()
    state = model(rex)
    with pytest.raises(LookupError, match="not registered"):
        infer_model(state, rex)
    register_model_provider(provider())
    changed = replace(state, configuration={"execution_provider": {"name": "contract_fixture", "version": "2"}})
    with pytest.raises(LookupError, match="version '2'"):
        infer_model(changed, rex)


@pytest.mark.parametrize("declaration", [False, "contract_fixture", {}, {"name": "contract_fixture"},
    {"name": "contract_fixture", "version": "1", "fallback": "torch"},
    {"name": 1, "version": "1"}, {"name": "contract_fixture", "version": ""}])
def test_malformed_provider_declarations_are_rejected(declaration):
    rex = source()
    state = model(rex, declaration)
    with pytest.raises(ValueError, match="execution provider"):
        infer_model(state, rex)


@pytest.mark.parametrize("change", [{"adapter_version": "2"}, {"arithmetic": "approximate"}])
def test_provider_rejects_unsupported_adapter_and_arithmetic(change):
    rex = source()
    state = replace(model(rex), **change)
    register_model_provider(provider())
    with pytest.raises(ValueError, match="not supported"):
        infer_model(state, rex)


def test_provider_registration_and_inventory_do_not_replace_implementations():
    first = provider()
    register_model_provider(first)
    register_model_provider(first)
    with pytest.raises(RuntimeError, match="already registered"):
        register_model_provider(provider())
    second = replace(first, version="2")
    register_model_provider(second)
    inventory = contract.model_provider_inventory()
    assert [v["version"] for v in inventory] == ["1", "2"]
    assert inventory[0]["operations"] == ("infer",)
    inventory[0]["name"] = "changed"
    assert contract.model_provider_inventory()[0]["name"] == "contract_fixture"


@pytest.mark.parametrize("changes, error", [
    ({"name": ""}, ValueError), ({"version": " 1"}, ValueError),
    ({"adapters": ()}, ValueError), ({"adapters": (("native_field", "1"),)}, ValueError),
    ({"adapters": (("online_flow", "1"),)}, ValueError),
    ({"adapters": (("provider_field", "1"), ("provider_field", "1"))}, ValueError),
    ({"adapters": ("provider_field",)}, TypeError), ({"arithmetic": "rational"}, TypeError),
    ({"arithmetic": ("rounded",)}, ValueError), ({"infer": None}, TypeError),
    ({"train": True}, TypeError)])
def test_invalid_provider_contracts_are_rejected(changes, error):
    with pytest.raises(error):
        replace(provider(), **changes)


def test_provider_contract_copies_caller_metadata():
    adapters = [["provider_field", "1"]]
    arithmetic = ["rational"]
    selected = ModelProvider("contract_fixture", "1", adapters, forward, arithmetic)
    adapters[0][1] = "changed"
    arithmetic.append("approximate")
    assert selected.adapters == (("provider_field", "1"),)
    assert selected.arithmetic == ("rational",)


def test_provider_training_returns_child_and_retains_parent():
    rex = source()
    state = model(rex)
    original = state.coefficient_digest
    batch = ModelBatch(state.source, state.space, np.array([0, 1]), np.array([True, False]))
    register_model_provider(provider(train=update))
    child = train_model(state, rex, batch, steps=3)
    assert child.parent == original and child.step == 3 and child.payload["gain"] == Q(9, 2)
    assert state.coefficient_digest == original and state.payload["gain"] == Q(3, 2)


def test_missing_training_and_transport_capabilities_fail_explicitly():
    rex = source()
    state = model(rex)
    batch = ModelBatch(state.source, state.space, np.array([0, 1]), np.array([True, False]))
    register_model_provider(provider())
    with pytest.raises(NotImplementedError, match="train"):
        train_model(state, rex, batch)
    with pytest.raises(NotImplementedError, match="transport"):
        transport_model(state, rex, FieldSource(source((21, 22))), None)


def test_source_and_batch_validation_precede_provider_execution():
    calls = []
    rex = source()
    other = source((21, 22))
    state = model(rex)
    selected = provider(train=lambda *a, **k: calls.append("train"))
    register_model_provider(selected)
    with pytest.raises(ValueError, match="source"):
        infer_model(state, other)
    batch = ModelBatch(FieldSource(other), state.space, np.array([0, 1]), np.array([True, False]))
    with pytest.raises(ValueError, match="another source"):
        train_model(state, rex, batch)
    assert calls == []


@pytest.mark.parametrize("fault", ["type", "model", "source", "arithmetic"])
def test_invalid_provider_outputs_are_rejected(fault):
    rex = source()
    state = model(rex)
    def invalid(*args):
        result = forward(*args)
        if fault == "type": return result.values
        if fault == "model": return replace(result, model_digest="0" * 64)
        if fault == "source": return replace(result, source=FieldSource(source((21, 22))))
        return replace(result, arithmetic="approximate", values=np.asarray(result.values, dtype=float))
    register_model_provider(replace(provider(), infer=invalid))
    with pytest.raises((TypeError, ValueError), match="model provider"):
        infer_model(state, rex, inputs(state, rex))


@pytest.mark.parametrize("phase", ["provider", "output_validation"])
def test_model_changes_during_execution_or_output_validation_are_rejected(phase):
    rex = source()
    state = model(rex)

    class ChangingOutput(ModelOutput):
        def check_state(self):
            super().check_state()
            object.__setattr__(state, "payload", {"gain": Q(9)})

    def invalid(*args):
        out = forward(*args)
        if phase == "provider":
            object.__setattr__(state, "payload", {"gain": Q(9)})
            return out
        return ChangingOutput(out.values, out.space, out.axes, out.source, out.model_digest,
                              out.arithmetic, out.method, grade=out.grade, variance=out.variance)

    register_model_provider(replace(provider(), infer=invalid))
    with pytest.raises(ValueError, match="changed after construction"):
        infer_model(state, rex, inputs(state, rex))


@pytest.mark.parametrize("fault", ["type", "parent", "step", "source", "adapter", "provider", "arithmetic"])
def test_invalid_provider_training_children_are_rejected(fault):
    rex = source()
    state = model(rex)
    batch = ModelBatch(state.source, state.space, np.array([0, 1]), np.array([True, False]))
    def invalid(*args, **kwargs):
        child = update(*args, **kwargs)
        if fault == "type": return None
        if fault == "parent": return replace(child, parent="0" * 64)
        if fault == "step": return replace(child, step=0)
        if fault == "source": return replace(child, source=FieldSource(source((21, 22))))
        if fault == "adapter": return replace(child, adapter_version="2")
        if fault == "provider":
            return replace(child, configuration={"execution_provider": {"name": "other", "version": "1"}})
        return replace(child, arithmetic="approximate")
    register_model_provider(provider(train=invalid))
    with pytest.raises((TypeError, ValueError), match="model provider"):
        train_model(state, rex, batch)


def test_provider_transport_retains_parent_and_destination():
    rex = source()
    destination = FieldSource(source((21, 22)))
    state = model(rex)
    calls = []
    def move(previous, current, target, mapping, *, optimizer):
        calls.append((current, mapping, optimizer))
        return previous.child(payload=previous.payload, source=target,
                              space=model_coordinates(target.source), dependencies=(previous.source,))
    register_model_provider(provider(transport=move))
    mapping = object()
    result = transport_model(state, rex, destination, mapping)
    assert result.parent == state.coefficient_digest and result.step == state.step
    assert result.source.matches(destination)
    assert calls == [(rex, mapping, "reset")]


def test_provider_declaration_survives_existing_model_record_codec():
    from rexgraph.io.model_record import model_record, read_model_record
    from rexgraph.state import from_state, to_state
    rex = source()
    state = model(rex)
    register_model_provider(provider())
    restored, restored_source = read_model_record(from_state(to_state(model_record(rex, state))))
    assert restored.coefficient_digest == state.coefficient_digest
    output = infer_model(restored, restored_source, inputs(restored, restored_source))
    assert tuple(output.values) == (Q(1, 2), Q(-3, 5))


def test_provider_dispatch_runs_with_torch_imports_blocked(tmp_path):
    script = textwrap.dedent('''
        import importlib.abc
        import sys
        class BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise AssertionError("provider dispatch attempted to import Torch")
        sys.meta_path.insert(0, BlockTorch())
        from fractions import Fraction
        import numpy as np
        from rexgraph import RexGraph
        from rexgraph.model_contract import ModelProvider, register_model_provider, model_coordinates
        from rexgraph.model_runtime import infer_model
        from rexgraph.model_state import ModelState, ModelOutput
        from rexgraph.tensor_field import FieldSource
        rex = RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2])
        state = ModelState("fixture", "1", FieldSource(rex), model_coordinates(rex), (),
            {"execution_provider": {"name": "fixture", "version": "1"}}, {}, "rational")
        def infer(selected, source, inputs):
            return ModelOutput([Fraction(1, 3)] * len(selected.space.keys), selected.space,
                (), selected.source, selected.coefficient_digest, "rational", "fixture")
        try:
            infer_model(state, rex)
        except LookupError:
            pass
        else:
            raise AssertionError("unregistered provider was accepted")
        register_model_provider(ModelProvider("fixture", "1", (("fixture", "1"),), infer))
        assert all(v == Fraction(1, 3) for v in infer_model(state, rex).values)
        assert not any(n == "torch" or n.startswith("torch.") for n in sys.modules)
        assert "rexgraph.nn" not in sys.modules
    ''')
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr

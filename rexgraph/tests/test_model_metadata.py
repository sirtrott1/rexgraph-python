"""Shared immutable declarations preserve model bytes and mutation detection."""
import gc
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pytest

from rexgraph.model_codec import model_payload, pack_model, unpack_model
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree, thaw_tree
from rexgraph.tensor_field import FieldSource
from rexgraph.type_accession import CoordinateSpace


def detached_tree(value):
    if isinstance(value, MappingProxyType):
        return MappingProxyType({k: detached_tree(v) for k, v in value.items()})
    if isinstance(value, tuple):
        return tuple(detached_tree(v) for v in value)
    return value


def state(config=None, payload=None):
    return ModelState("metadata-fixture", "1", FieldSource(None, state_digest="a" * 64),
                      CoordinateSpace("rows", ("a", "b")), (),
                      {} if config is None else config, {} if payload is None else payload, "rational")


def equal_tensors(left, right):
    assert set(left) == set(right)
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert left[name].shape == right[name].shape
        assert np.array_equal(left[name], right[name])


def test_repeated_declarations_preserve_each_ordered_observation():
    rows = tuple({"event_time": 1, "knowledge_time": 2, "partition": "training"} for _ in range(8192))
    frozen = freeze_tree(rows)
    assert len(frozen) == 8192 and frozen[0] is frozen[-1]
    rows[0]["event_time"] = 9
    assert frozen[0]["event_time"] == 1
    with pytest.raises(TypeError): frozen[0]["event_time"] = 9
    copied = thaw_tree(frozen)
    copied[0]["event_time"] = 9
    assert copied[-1]["event_time"] == 1 and frozen[-1]["event_time"] == 1


@pytest.mark.parametrize("pair", [({"value": 1}, {"value": True}), ({"value": 0.0}, {"value": -0.0}),
                                  ({1: "one"}, {"1": "one"}),
                                  ({"a": 1, "b": 2}, {"b": 2, "a": 1})])
def test_scalar_kind_zero_sign_key_kind_and_insertion_order_remain_distinct(pair):
    a, b = freeze_tree(pair)
    assert a is not b
    assert tuple(a) == tuple(pair[0]) and tuple(b) == tuple(pair[1])
    assert thaw_tree(a) == pair[0] and thaw_tree(b) == pair[1]


@pytest.mark.parametrize("values", [np.array([1, 2], np.int64), np.array([True, False]),
                                    np.array([Fraction(1, 3), Fraction(2, 5)], object),
                                    Fraction(1, 3), b"attachment", (1, 2), {"nested": 1}])
def test_mutable_or_tensor_bearing_declarations_are_not_interned(values):
    frozen = freeze_tree(({"value": values}, {"value": values}))
    assert frozen[0] is not frozen[1]
    if isinstance(values, np.ndarray):
        assert not np.shares_memory(frozen[0]["value"], frozen[1]["value"])
        assert not np.shares_memory(values, frozen[0]["value"])


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["model", "output", "batch"])
def test_shared_and_separate_declarations_have_identical_encodings(native, kind):
    declarations = tuple({"event_time": i % 3, "partition": "training", "zero": -0.0}
                         for i in range(4096))
    data = {"repeated": declarations, "mixed": {1: "one", "1": 1},
            "exact": Fraction(7, 13), "bytes": b"archive", "mask": np.array([True, False]),
            "scalar": np.array(2, np.int64), "parameters": np.array([3, -7], np.int64)}
    selected = state(data)
    if kind == "output":
        selected = ModelOutput([Fraction(1, 3), Fraction(2, 5)], selected.space, (), selected.source,
                               selected.coefficient_digest, "rational", "metadata-fixture",
                               execution_receipt=data)
        name = "execution_receipt"
    elif kind == "batch":
        selected = ModelBatch(selected.source, selected.space, [1, -1], [True, True], data)
        name = "inputs"
    else:
        name = "configuration"
    separate = replace(selected)
    object.__setattr__(separate, name, detached_tree(getattr(separate, name)))
    assert getattr(selected, name)["repeated"][0] is getattr(selected, name)["repeated"][3]
    assert getattr(separate, name)["repeated"][0] is not getattr(separate, name)["repeated"][3]
    separate.check_state()
    equal_tensors(model_payload(selected, native=native), model_payload(separate, native=native))
    equal_tensors(pack_model(selected, native=native), pack_model(separate, native=native))
    assert unpack_model(pack_model(selected, native=native)).coefficient_digest == selected.coefficient_digest


@pytest.mark.parametrize("change", ["configuration", "payload", "array", "fraction", "underlying"])
def test_each_seal_pass_detects_changes_after_shared_encoding(change):
    selected = state({"repeated": ({"value": 1},) * 4},
                     {"parameters": np.array([1, 2], np.int64), "exact": Fraction(1, 3)})
    original = selected.coefficient_digest
    selected.check_state(); selected.check_state()
    if change == "configuration":
        object.__setattr__(selected, "configuration", MappingProxyType({"repeated": ({"value": 9},) * 4}))
    elif change == "payload":
        object.__setattr__(selected, "payload", MappingProxyType({"parameters": np.array([9, 2], np.int64)}))
    elif change == "array":
        selected.payload["parameters"].flags.writeable = True
        selected.payload["parameters"][0] = 9
    elif change == "fraction":
        selected.payload["exact"]._numerator = 7
    else:
        underlying = next(v for v in gc.get_referents(selected.configuration["repeated"][0]) if isinstance(v, dict))
        underlying["value"] = 9
    with pytest.raises(ValueError, match="changed after construction"): selected.check_state()
    with pytest.raises(ValueError, match="changed after construction"): _ = selected.coefficient_digest
    assert selected._digest == original


def test_repeated_attachments_and_arrays_keep_distinct_tensor_names():
    selected = state(payload={"arrays": (np.array([1, 2]),) * 3, "attachments": ({"bytes": b"x"},) * 3})
    result = model_payload(selected)
    assert len([name for name in result if name.startswith("tensor/")]) == 6
    equal_tensors(result, model_payload(unpack_model(pack_model(selected))))


def test_unique_declarations_still_roundtrip_after_the_sharing_bound():
    data = tuple({"event_time": i, "partition": "training"} for i in range(4096))
    selected = state({"availability": data})
    restored = unpack_model(pack_model(selected, native=True))
    assert restored.coefficient_digest == selected.coefficient_digest
    assert tuple(v["event_time"] for v in restored.configuration["availability"]) == tuple(range(4096))


def test_metadata_sharing_does_not_cross_freeze_calls():
    a = freeze_tree(({"value": 1},) * 3)
    b = freeze_tree(({"value": 1},) * 3)
    assert a[0] is a[-1] and b[0] is b[-1] and a[0] is not b[0]


def test_model_seal_preserves_structural_and_exact_tensor_codecs():
    import json

    from rexgraph.state import encode_tensors, state_digest

    selected = state({"name": "\u03bb", "values": (True, None, -0.0)}, {
        "byte_coefficients": np.arange(256, dtype=np.uint8),
        "offsets": np.array([0, 1, 3, 6], dtype=np.int64),
        "exact": np.array([Fraction(1, 3), Fraction(1 << 512, 7)], dtype=object)})
    tensors = model_payload(selected)
    codecs = encode_tensors(tensors)
    assert "spec" not in codecs
    assert codecs["tensor/0"]["c"] == "arange"
    assert any(value["c"] == "exact" for value in codecs.values())
    assert any(value["c"] == "delta" for value in codecs.values())
    tensors["codec"] = np.frombuffer(json.dumps(codecs, sort_keys=True).encode(), dtype=np.uint8)
    assert state_digest(tensors) == selected.coefficient_digest
    assert unpack_model(pack_model(selected, native=True)).coefficient_digest == selected.coefficient_digest

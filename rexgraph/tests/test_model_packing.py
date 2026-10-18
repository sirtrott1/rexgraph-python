"""Complete model envelopes retain their canonical schema and tensor bytes."""
from fractions import Fraction

import numpy as np
import pytest

from rexgraph.model_codec import model_payload, pack_model, unpack_model
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, ModelTimeline, model_input
from rexgraph.tensor_field import FieldSource
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value


def legacy_pack(value, native):
    value.check_state()
    tensors = model_payload(value, native=native)
    if native:
        spec = unpack_value(tensors["spec"].tobytes())
        spec["tensor_names"] = sorted([*spec["tensor_names"], "model_digest"])
        tensors["spec"] = np.frombuffer(pack_value(spec), np.uint8).copy()
    tensors["model_digest"] = np.frombuffer(value.coefficient_digest.encode(), np.uint8).copy()
    return tensors


def envelope(kind, declarations):
    source = FieldSource(None, state_digest="a" * 64)
    space = CoordinateSpace("rows", ("first", "second"))
    data = {"availability": tuple({"time": i if declarations == "unique" else 1,
                                   "partition": "training"} for i in range(128)),
            "mixed": {1: "one", "1": 1}, "name": "\u03bb\u2603",
            "zero": -0.0, "exact": Fraction(1 << 300, 13),
            "bytes": b"\x00\xffarchive", "bool": np.array([True, False]),
            "scalar": np.array(3, np.int64), "parameters": np.arange(256, dtype=np.uint8)}
    if declarations == "empty":
        data = {}
    state = ModelState("packing-fixture", "1", source, space, (), data, {}, "rational")
    if kind == "model":
        return state
    output = ModelOutput([Fraction(1, 3), Fraction(-2, 7)], space, (), source,
                         state.coefficient_digest, "rational", "packing-fixture", execution_receipt=data)
    if kind == "output":
        return output
    if kind == "input":
        return model_input(output)
    if kind == "batch":
        return ModelBatch(source, space, output.values, [True, False], data)
    from rexgraph.graph import RexGraph, TemporalRex
    from rexgraph.temporal_state import to_temporal_state
    history = TemporalRex([])
    history.append_snapshot(RexGraph.from_hypergraph([0, 1], [0]), at=0)
    temporal = to_temporal_state(history)
    return ModelTimeline((Fraction(1, 3),), (state.coefficient_digest,), (source.state_digest,),
                         temporal.header, temporal.tensors)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["model", "output", "input", "batch", "timeline"])
@pytest.mark.parametrize("declarations", ["empty", "repeated", "unique"])
def test_complete_packing_matches_prior_bytes(native, kind, declarations):
    value = envelope(kind, declarations)
    actual, expected = pack_model(value, native=native), legacy_pack(value, native)
    assert set(actual) == set(expected)
    for name in actual:
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].shape == expected[name].shape
        if actual[name].dtype.hasobject:
            assert np.array_equal(actual[name], expected[name])
        else:
            assert actual[name].tobytes() == expected[name].tobytes()
    if native:
        declared = unpack_value(actual["spec"].tobytes())["tensor_names"]
        assert declared == sorted(set(actual) - {"spec"})
    assert unpack_model(actual).coefficient_digest == value.coefficient_digest


@pytest.mark.parametrize("kind", ["model", "output", "input", "batch", "timeline"])
@pytest.mark.parametrize("change", ["missing", "extra", "seal"])
def test_native_tensor_claims_and_current_seal_are_required(kind, change):
    tensors = pack_model(envelope(kind, "repeated"), native=True)
    if change == "missing":
        tensors.pop("model_digest")
    elif change == "extra":
        tensors["unclaimed"] = np.array([1], np.int64)
    else:
        tensors["model_digest"][0] ^= 1
    with pytest.raises(ValueError):
        unpack_model(tensors)


def test_packing_detects_model_mutation_during_specification_encoding(monkeypatch):
    import rexgraph.model_codec as codec
    value = envelope("model", "repeated")
    original = codec.pack_spec
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(value, "step", value.step + 1)
        return result
    monkeypatch.setattr(codec, "pack_spec", changed)
    with pytest.raises(ValueError, match="changed after construction"):
        pack_model(value, native=True)


def test_model_tensor_copies_are_independent_after_packing():
    value = envelope("model", "repeated")
    first, second = pack_model(value, native=True), pack_model(value, native=True)
    for name in first:
        assert not np.shares_memory(first[name], second[name])
    parameters = next(a for a in first.values() if a.dtype == np.uint8 and a.shape == (256,))
    parameters[0] = 99
    assert value.configuration["parameters"][0] == 0
    value.check_state()
    with pytest.raises(ValueError, match="model identity mismatch"):
        unpack_model(first)

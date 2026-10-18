"""Native source declarations and unchanged legacy model identities."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from rexgraph.field_codec import pack_field, unpack_field
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_state import ModelOutput, ModelState
from rexgraph.source_binding import CppSourceIdentity, CppSourceReference, source_from_record
from rexgraph.tensor_field import FieldSource
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import unpack_spec


def identity():
    return CppSourceIdentity(88, 2, 0, "a" * 64, "b" * 64, (("10", "11"), ("100",), ()))


def test_detached_source_retains_native_kind_and_record_identity():
    ref = CppSourceReference(None, "source", 3, native=identity())
    restored = source_from_record(ref.as_record(), allow_cpp=True)
    assert restored.matches(ref) and restored.detached().matches(ref)
    assert restored.source is None and restored.native.coordinates[-1] == ()
    legacy = FieldSource(None, "source", 3, ref.state_digest)
    assert not ref.matches(legacy) and not legacy.matches(ref)


@pytest.mark.parametrize("changes", [{"namespace": -1}, {"namespace": True},
    {"sequence": 1 << 64}, {"first_sequence": 3}, {"codec": "unknown"},
    {"archive_digest": "a" * 63}, {"archive_digest": " " * 64},
    {"endpoint_digest": "g" * 64}, {"endpoint_digest": "B" * 64},
    {"coordinates": ()}, {"coordinates": "100"}, {"coordinates": ("100",)},
    {"coordinates": (("100", "100"),)}, {"coordinates": (("01",),)},
    {"coordinates": ((str(1 << 64),),)}, {"coordinates": (("foo",),)}])
def test_native_identity_validation(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(identity(), **changes)


@pytest.mark.parametrize("change", ["unknown", "version", "boolean", "extra", "native_extra"])
def test_native_source_record_is_a_closed_schema(change):
    record = CppSourceReference(None, native=identity()).as_record()
    if change == "unknown": record["kind"] = "unknown"
    elif change == "version": record["schema_version"] = 2
    elif change == "boolean": record["schema_version"] = True
    elif change == "extra": record["extra"] = "unknown"
    else: record["native"]["extra"] = "unknown"
    with pytest.raises(ValueError):
        source_from_record(record, allow_cpp=True)


def test_legacy_model_and_output_seals_remain_unchanged():
    source = FieldSource(None, "data", 3, "a" * 64)
    space = CoordinateSpace("edge", ("100", "101"))
    state = ModelState("native_field", "1", source, space, (),
                       {"grade": 1, "parameter": Fraction(1, 3)},
                       {"weights": np.array([Fraction(2, 7), Fraction(0)], dtype=object)}, "rational")
    output = ModelOutput([Fraction(1, 3), Fraction(0)], space, (), source,
                         state.coefficient_digest, "rational", "legacy-fixture", grade=1, variance="chain")
    assert state.coefficient_digest == "af9638b02f66f8ac7617e0282bc481c670f643f5a84fd30a8fd77638fae04fa9"
    assert output.coefficient_digest == "9673193c7f399baf96ebde0732cd8ab9084d5b109c895a122ed7d9b3a02a7f43"
    for native in (False, True):
        for value in (state, output):
            encoded = pack_model(value, native=native)
            assert unpack_spec(encoded)["version"] == 1
            assert "execution_receipt" not in unpack_spec(encoded)
            assert unpack_model(encoded).coefficient_digest == value.coefficient_digest
        field = output.tensor()
        assert field.coefficient_digest == "33dcbcb108b05c920959765bcaa04aa35a4fced64188b199d75355b556f15ffe"
        encoded = pack_field(field, native=native)
        assert unpack_spec(encoded)["version"] == 1
        assert unpack_field(encoded).coefficient_digest == field.coefficient_digest


def test_execution_receipt_changes_output_identity_and_is_copied():
    source = FieldSource(None, state_digest="a" * 64)
    receipt = {"provider": "test", "input": 7}
    original = ModelOutput([Fraction(1)], CoordinateSpace("edge", ("100",)), (), source,
                           "b" * 64, "rational", "test", execution_receipt=receipt)
    receipt["input"] = 8
    assert original.execution_receipt["input"] == 7
    changed = replace(original, execution_receipt=receipt)
    assert original.coefficient_digest != changed.coefficient_digest
    assert unpack_model(pack_model(original)).execution_receipt == original.execution_receipt


def test_model_history_validates_state_type_before_source_kind():
    from rexgraph.io.model_record import model_history
    with pytest.raises(TypeError, match="ModelState entries"):
        model_history([None], [None], [Fraction(0)])


def test_native_model_history_requires_a_declared_timeline():
    from rexgraph.io.model_record import model_history
    state = ModelState("cpp_field", "1", CppSourceReference(None, native=identity()),
                       CoordinateSpace("edge", ("100",)), (), {}, {}, "rational")
    with pytest.raises(NotImplementedError, match="timeline profile"):
        model_history([state], [None], [Fraction(0)])

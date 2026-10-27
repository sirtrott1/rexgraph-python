"""Native relational children preserve structural lineage through ordinary I/O."""
from fractions import Fraction as Q
import importlib.util
import os
from pathlib import Path

import pytest


@pytest.fixture
def learned():
    pytest.importorskip("rexgraph_cpp._temporal_program")
    from rexgraph.relational_runtime import RelationalRuntime
    path = Path(os.environ.get("REXGRAPH_RELATIONAL_TEST_HELPER", str(
        Path(__file__).resolve().parents[2] / "bindings/cpp/tests/test_temporal_program.py")))
    if not path.is_file():pytest.skip("generated native relational test declarations required")
    spec = importlib.util.spec_from_file_location("relational_declarations", path)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper, helper.fit_fixture(RelationalRuntime.create(89001))


def test_whole_source_extension_and_cold_parent_correspondence(learned):
    from rexgraph.relational_runtime import RelationalRuntime
    helper, (parent, child, t, _, _, _, _, _) = learned
    assert child.extends(parent) and child.parent_digest == parent.digest
    assert child.fit()["forced"]
    before = child.to_bytes()
    fresh = RelationalRuntime.from_bytes(before)
    assert fresh.to_bytes() == before and fresh.extends(parent)
    with pytest.raises(AttributeError):fresh._packet = b"forged"
    family = child.selected
    new, cubic = helper.function(child, t, (2, -1, 3, 1), label="independent input")
    result = new.extend("apply_learned", dict(family=family, bindings=[(1, cubic)], label="complete transfer"))
    assert result.function()["pieces"][0]["numerator"] == (Q(16), Q(48), Q(24))
    assert result.evaluate(Q(1, 3)) == Q(104, 3)
    assert child.to_bytes() == before


def test_forged_parent_and_implementation_are_not_lineage_or_execution_authority(learned):
    from rexgraph.relational_runtime import RelationalRuntime
    from rexgraph.value_codec import pack_value, unpack_value
    _, (parent, child, *_) = learned
    data = unpack_value(child.to_bytes())
    data["parent"]["source"] = "0" * 64
    untrusted = RelationalRuntime.from_bytes(pack_value(data))
    assert not untrusted.extends(parent)
    data = unpack_value(child.to_bytes())
    data["implementation"]["binding"] = "0" * 64
    with pytest.raises(ValueError, match="implementation differs"):
        RelationalRuntime.from_bytes(pack_value(data))
    with pytest.raises(ValueError, match="restore admission"):
        RelationalRuntime.from_bytes(child.to_bytes(), restore_limits=(1,) * 7)


@pytest.mark.parametrize("format", ("rcbd", "json", "safetensors", "h5"))
def test_complete_native_source_endpoint_roundtrip_in_record_formats(learned, tmp_path, format):
    if format == "safetensors":pytest.importorskip("safetensors")
    if format == "h5":pytest.importorskip("h5py")
    from rexgraph.io import load, save
    from rexgraph.io.relational_record import read_relational_record, relational_record
    _, (_, runtime, *_) = learned
    path = tmp_path / ("program." + format)
    save(path, relational_record(runtime))
    cold = read_relational_record(load(path))
    assert cold.to_bytes() == runtime.to_bytes()
    assert cold.fit() == runtime.fit()


def test_import_of_core_runtime_module_does_not_load_optional_executor():
    # The module's import is also covered in a fresh Core-only process.
    from rexgraph.relational_runtime import RelationalRuntime
    with pytest.raises(TypeError):RelationalRuntime()


@pytest.mark.parametrize("format", ("rcbd", "json", "safetensors", "h5"))
def test_whole_field_function_capsule_roundtrips_with_its_source_and_pairing(tmp_path, format):
    pytest.importorskip("rexgraph_cpp._temporal_program")
    root = os.environ.get("REXGRAPH_FIELD_FUNCTION_FIXTURE")
    if root is None:pytest.skip("synthetic field-function envelope not supplied")
    if format == "safetensors":pytest.importorskip("safetensors")
    if format == "h5":pytest.importorskip("h5py")
    from rexgraph.io import load, save
    from rexgraph.io.relational_record import relational_record, read_relational_record
    from rexgraph.relational_runtime import RelationalFieldFunction
    value = RelationalFieldFunction.from_envelope((Path(root) / "whole-field.envelope").read_bytes(), 56000)
    raw = value.to_bytes()
    restored = RelationalFieldFunction.from_bytes(raw)
    assert restored.to_bytes() == raw
    assert restored.inspect() == value.inspect()
    assert restored.evaluate(Q(1, 3)).inspect() == value.evaluate(Q(1, 3)).inspect()
    path = tmp_path / ("field-function." + format)
    save(path, relational_record(value))
    assert read_relational_record(load(path)).to_bytes() == raw

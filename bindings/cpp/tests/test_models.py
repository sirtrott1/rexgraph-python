"""Exact native model records over the existing Core and RCDB contracts."""
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from rexgraph_cpp import NativeSource

from rexgraph.cpp_model import cpp_model, cpp_source_record, restore_cpp_source
from rexgraph.field_codec import pack_field, unpack_field
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import ModelBatch
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.state import from_state, to_state
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import unpack_spec


def model(source, **kwargs):
    return cpp_model(source, (10, 11, 12), metric_keys=(2000, 2001, 2002),
                     metric_selection="diagonal", **kwargs)


def test_exact_provider_and_output_receipt(native_source):
    state = model(native_source)
    output = infer_model(state, native_source, native_source.field(1001))
    assert state.arithmetic == output.arithmetic == "rational"
    assert tuple(output.values) == (Fraction(83, 30), Fraction(-17, 45), Fraction(-19, 20))
    assert output.space.keys == ("100", "101", "102")
    assert output.execution_receipt["input_field"] == 1001
    assert output.execution_receipt["input_carrier"] == output.execution_receipt["output_carrier"] == 11
    assert output.execution_receipt["engine"] == state.configuration["engine_receipt"]
    assert output.execution_receipt["metric_keys"] == (2000, 2001, 2002)
    assert output.execution_receipt["metric_selection"] == "diagonal"
    assert output.execution_receipt["operation"] == "hodge"
    with pytest.raises(TypeError):
        output.execution_receipt["input_field"] = 3001


def test_native_source_without_provider_rejects_before_legacy_dispatch(native_source):
    state = model(native_source)
    configuration = {key: value for key, value in state.configuration.items()
                     if key != "execution_provider"}
    undeclared = replace(state, configuration=configuration)
    with pytest.raises(LookupError, match="native source models require an explicit execution provider"):
        infer_model(undeclared, native_source, native_source.field(1001))


@pytest.mark.parametrize("operation,grade,input_key,output_grade", [("boundary", 1, 1001, 0),
                            ("adjoint", 2, 1001, 2), ("hodge_up", 2, 1002, 2),
                            ("green", 1, 1001, 1), ("harmonic", 1, 1001, 1)])
def test_model_output_coordinate_grade(native_source, operation, grade, input_key, output_grade):
    state = model(native_source, operation=operation, grade=grade,
                  profile="global" if operation in {"green", "harmonic"} else "finite")
    output = infer_model(state, native_source, native_source.field(input_key))
    assert output.grade == output_grade
    assert output.space.keys == native_source.identity["coordinates"][output_grade]
    assert output.model_digest == state.coefficient_digest


@pytest.mark.parametrize("native", [False, True])
def test_source_and_output_codecs_preserve_native_kind(native_source, native):
    ref = CppSourceReference(native_source, "native-data", 7)
    state = model(native_source, reference=ref)
    output = infer_model(state, native_source, native_source.field(1001))
    for value in (state, output):
        encoded = pack_model(value, native=native)
        assert unpack_spec(encoded)["version"] == 2
        restored = unpack_model(encoded)
        assert restored.coefficient_digest == value.coefficient_digest
        assert isinstance(restored.source, CppSourceReference)
        assert restored.source.source is None
        assert restored.source.native == ref.native
        assert restored.source.record_id == "native-data" and restored.source.version == 7
    field = output.tensor()
    encoded = pack_field(field, native=native)
    assert unpack_spec(encoded)["version"] == 2
    assert unpack_field(encoded).coefficient_digest == field.coefficient_digest


def test_native_source_reference_rejects_legacy_tag_and_wrong_identity(native_source):
    ref = CppSourceReference(native_source)
    with pytest.raises(ValueError, match="schema"):
        source_from_record(ref.as_record())
    assert source_from_record(ref.as_record(), allow_cpp=True).matches(ref)
    wrong = replace(ref.native, namespace=ref.native.namespace + 1)
    with pytest.raises(ValueError, match="declared publication"):
        CppSourceReference(native_source, native=wrong)


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_native_model_save_restore_with_temporal_source(native_fixtures, tmp_path, suffix):
    archive = (native_fixtures[0] / "diagonal-history.rfcxs").read_bytes()
    source = NativeSource(archive)
    state = model(source, operation="green", profile="global", parameter=Fraction(1, 2))
    expected = infer_model(state, source, source.field(1001))
    record = model_record(source, state)
    record.attach_metadata(1, 0, "prediction", expected)
    path = tmp_path / ("native" + suffix)
    save_model(path, record)
    record = load_model(path)
    restored, loaded = read_model_record(record)
    assert loaded.archive == archive and loaded.identity == source.identity
    assert loaded.at(0).archive == (native_fixtures[0] / "diagonal.rfcxs").read_bytes()
    assert restored.coefficient_digest == state.coefficient_digest
    assert infer_model(restored, loaded, loaded.field(1001)).coefficient_digest == expected.coefficient_digest
    prediction = record.get_metadata(1, 0, "prediction")
    assert prediction.coefficient_digest == expected.coefficient_digest
    assert prediction.execution_receipt == expected.execution_receipt


def test_existing_graph_state_carries_native_archive(native_source):
    state = model(native_source)
    record = from_state(to_state(model_record(native_source, state)))
    restored, source = read_model_record(record)
    assert source.archive == native_source.archive
    assert restored.coefficient_digest == state.coefficient_digest


@pytest.mark.parametrize("mode", ["bytes", "digest", "reference", "dtype", "shape", "bound", "unknown"])
def test_source_attachment_validation(native_source, mode):
    ref = CppSourceReference(native_source)
    record = cpp_source_record(native_source, ref)
    limit = 256 * 1024 * 1024
    if mode == "bytes": record["archive"][10] ^= 1
    elif mode == "digest":
        record["reference"]["state_digest"] = "0" * 64
    elif mode == "reference": record["reference"]["native"]["namespace"] += 1
    elif mode == "dtype": record["archive"] = record["archive"].astype(np.int16)
    elif mode == "shape": record["archive"] = record["archive"].reshape(1, -1)
    elif mode == "bound": limit = len(record["archive"]) - 1
    elif mode == "unknown": record["extra"] = "unclaimed"
    with pytest.raises(ValueError):
        restore_cpp_source(record, ref, max_bytes=limit)


@pytest.mark.parametrize("change", ["receipt", "space", "profile", "provider", "extra"])
def test_restore_rejects_incompatible_model_declaration(native_source, change):
    state = model(native_source)
    configuration = dict(state.configuration)
    if change == "receipt": configuration["engine_receipt"] = {}
    elif change == "profile": configuration["input_profile"] = "undocumented"
    elif change == "provider": configuration["execution_provider"] = {"name": "missing", "version": "1"}
    elif change == "extra": configuration["unclaimed"] = True
    invalid = replace(state, configuration=configuration,
                      space=CoordinateSpace("other", state.space.keys) if change == "space" else state.space)
    record = model_record(native_source, invalid)
    with pytest.raises(LookupError if change == "provider" else ValueError):
        read_model_record(record)


@pytest.mark.parametrize("kwargs", [{"parameter": 0.5}, {"parameter": True}, {"parameter": 1},
    {"grade": True}, {"grade": 5}, {"grade": 0, "operation": "boundary"},
    {"operation": "green"}, {"operation": "harmonic"}, {"operation": "unregistered"},
    {"profile": "global", "dimension_limit": 2}])
def test_model_declaration_rejects_unsupported_profile(native_source, kwargs):
    with pytest.raises((ValueError, TypeError, IndexError, RuntimeError)):
        model(native_source, **kwargs)


def test_native_model_does_not_fall_back_to_training_or_float_inputs(native_source):
    state = model(native_source)
    with pytest.raises(TypeError, match="NativeField"):
        infer_model(state, native_source, np.zeros(3))
    batch = ModelBatch(state.source, state.space, np.zeros(3, dtype=object), np.ones(3, dtype=bool))
    with pytest.raises(NotImplementedError, match="train"):
        train_model(state, native_source, batch)


NO_TORCH = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("native model imported Torch")
sys.meta_path.insert(0, NoTorch())
'''


@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
def test_rcdb_fresh_process_exact_restore(native_source, tmp_path, backend):
    from rcdb import open_store
    url = {"file": "file://" + str(tmp_path / "file"),
           "rex": "rex://" + str(tmp_path / "rex"),
           "sqlite": "sqlite:///" + str(tmp_path / "records.sqlite")}[backend]
    state = model(native_source)
    output = infer_model(state, native_source, native_source.field(1001))
    record = model_record(native_source, state)
    record.attach_metadata(1, 0, "prediction", output)
    options = {} if backend == "sqlite" else {"read_only": False}
    store = open_store(url, **options).configure_security(require_commits=True)
    store.commit_mutation("native-model", record, expected_version=0)
    assert store.verify_commits("native-model")
    store.close()
    script = NO_TORCH + '''
import json
from rcdb import open_store
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
store = open_store(sys.argv[1])
record = store.read_record("native-model", version=1).value
state, source = read_model_record(record)
output = infer_model(state, source, source.field(1001))
assert output.coefficient_digest == record.get_metadata(1, 0, "prediction").coefficient_digest
print(json.dumps({"model": state.coefficient_digest, "source": source.identity,
                  "output": output.coefficient_digest, "values": [str(v) for v in output.values]}))
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
store.close()
'''
    completed = subprocess.run([sys.executable, "-c", script, url], cwd=tmp_path,
                               check=True, capture_output=True, text=True)
    restored = json.loads(completed.stdout)
    assert restored["model"] == state.coefficient_digest
    assert restored["output"] == output.coefficient_digest
    assert restored["source"]["archive_digest"] == native_source.identity["archive_digest"]
    assert restored["values"] == [str(v) for v in output.values]


def test_ordinary_core_remains_usable_without_optional_bridge(tmp_path):
    script = NO_TORCH + '''
class NoBridge(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rexgraph_cpp" or fullname.startswith("rexgraph_cpp."):
            raise ModuleNotFoundError("optional bindings unavailable", name="rexgraph_cpp")
sys.meta_path.insert(0, NoBridge())
from fractions import Fraction
from rexgraph.graph import RexGraph
from rexgraph.model_runtime import native_model, infer_model
from rexgraph.tensor_field import TensorField
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.cpp_model import cpp_model
source = RexGraph.from_hypergraph([0, 1], [0])
model = native_model(source, operation="hodge")
field = TensorField(model.space, [Fraction(1, 3)], source=model.source, grade=1, variance="chain")
output = infer_model(model, source, field)
assert unpack_model(pack_model(model)).coefficient_digest == model.coefficient_digest
assert "rexgraph_cpp" not in sys.modules and "torch" not in sys.modules
try:
    cpp_model(source, ())
except ImportError as exc:
    assert "optional" in str(exc)
else:
    raise AssertionError("missing optional bindings accepted")
'''
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True,
                   capture_output=True, text=True)

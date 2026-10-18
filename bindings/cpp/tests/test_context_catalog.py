"""Fitted byte contexts retain labels, full maps and parameter availability."""
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction as Q

import pytest
from rexgraph_cpp import NativeSource, fit_context_counts

from rexgraph.cpp_context_catalog import (
    cpp_context_annotation, cpp_context_catalog, cpp_context_catalog_input,
    cpp_context_training_batch, cpp_learned_correspondence, prepare_context_catalog,
)
from rexgraph.cpp_correspondence import cpp_correspondence_input, prepare_correspondence
from rexgraph.io.model_record import model_record, load_model, save_model, read_model_record
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import thaw_tree

from test_correspondence import chart_parts, declaration, fixture

FEATURES = ((b"bank", "right", b"river"), (b"bank", "left", b"river"),
            (b"bank", "right", b"money"), (b"bank", "left", b"money"),
            (b"bank", "left", b"near"), (b"bank", "left", b"quiet"))
LABELS = ("river-margin", "financial-institution", "unresolved")
_annotations = None
_trained = None


def catalog_fixture(**options):
    initial, _, _, pattern = fixture(counts=(0, 0, 0))
    vertices, edges, faces = [], [], []
    for i in range(len(FEATURES)):
        v, e, f = chart_parts(100*i)
        vertices.extend(v); edges.extend(e); faces.append(f)
    source = NativeSource.from_declaration(declaration(901, 2,
        (4*len(FEATURES), 4*len(FEATURES), len(FEATURES)), [*edges, *faces], vertices))
    state = cpp_context_catalog(source, pattern, carrier=12, features=FEATURES, labels=LABELS,
                               unresolved=2, **options)
    return state, source, pattern


def occurrence(feature=0, index=0, *, grade=1, literal=None):
    mention, side, context = FEATURES[feature] if literal is None else literal
    left, right = (mention, context) if side == "left" else (context, mention)
    prefix = f"α中🙂 {index}: ".encode()
    raw = prefix + left + b" " + right + f" ; document {index}\x00".encode()
    a, b = len(prefix), len(prefix)+len(left)
    return fixture(raw, ((a, b), (b+1, b+1+len(right))), side,
                   counts=(0, 0, 0), grade=grade, chart_namespace=10000+index)


def annotations(n=32):
    global _annotations
    if _annotations is not None:
        return _annotations[:n]
    samples = []
    schedule = [(f, int(f >= 2)) for f in range(4) for _ in range(6)] + [(4, v) for v in (0,)*5+(1,)*3]
    for i, (feature, label) in enumerate(schedule):
        context, source, *_ = occurrence(feature, i)
        samples.append(cpp_context_annotation(f"training-{i}", context, source, feature=feature,
            label=label, event_time=10+i, knowledge_time=100+i, label_available_time=200+i))
    _annotations = tuple(samples)
    return _annotations[:n]


def fitted():
    global _trained
    if _trained is not None:
        return _trained
    state, owner, pattern = catalog_fixture()
    batch = cpp_context_training_batch(state, owner, annotations(), event_cutoff=50, knowledge_cutoff=250)
    _trained = train_model(state, owner, batch), owner, pattern
    return _trained


def learned(catalog, owner, feature=0, index=999, grade=1, literal=None):
    context, source, raw, pattern = occurrence(feature, index, grade=grade, literal=literal)
    declaration = {k: thaw_tree(context.configuration[k]) for k in
        ("carriers", "raw_carriers", "pattern_carriers", "scope", "templates", "labels", "unresolved", "grade", "limits", "max_bytes")}
    result = cpp_learned_correspondence(catalog, owner, source, raw, pattern,
        event_cutoff=60, knowledge_cutoff=260, **declaration)
    return result, source


def test_fit_replays_source_annotations_and_exact_ambiguous_counts():
    state, owner, pattern = fitted()
    assert state.payload["counts"] == ((6, 0, 0), (6, 0, 0), (0, 6, 0), (0, 6, 0), (5, 3, 0), (0, 0, 0))
    assert prepare_context_catalog(state, owner) == state.payload["counts"]
    output = infer_model(state, owner, cpp_context_catalog_input(state, owner, event_cutoff=60, knowledge_cutoff=260))
    assert tuple(output.values[4]) == (Q(5, 8), Q(3, 8), Q(0))
    assert tuple(output.values[5]) == (Q(0), Q(0), Q(1))
    assert output.tensor().source.matches(state.source)
    assert unpack_model(pack_model(output)).coefficient_digest == output.coefficient_digest


def test_sequential_batches_fit_once_and_repeated_annotations_reject():
    state, owner, _ = catalog_fixture()
    samples = annotations()
    first = train_model(state, owner, cpp_context_training_batch(state, owner, samples[:16],
        event_cutoff=50, knowledge_cutoff=250))
    second = train_model(first, owner, cpp_context_training_batch(first, owner, samples[16:],
        event_cutoff=60, knowledge_cutoff=260))
    assert second.payload["counts"] == fitted()[0].payload["counts"]
    assert state.step == 0 and first.step == 1 and second.step == 2
    assert second.parent == first.coefficient_digest
    with pytest.raises(ValueError, match="already fitted"):
        train_model(second, owner, cpp_context_training_batch(second, owner, samples[:1],
            event_cutoff=60, knowledge_cutoff=260))
    with pytest.raises(ValueError, match="exactly one"):
        train_model(state, owner, cpp_context_training_batch(state, owner, samples,
            event_cutoff=50, knowledge_cutoff=250), steps=2)


@pytest.mark.parametrize("feature", range(6))
@pytest.mark.parametrize("grade", range(3))
def test_heldout_occurrences_use_fitted_complete_maps(feature, grade):
    catalog, owner, _ = fitted()
    state, source = learned(catalog, owner, feature, 7000+10*feature+grade, grade)
    assert state.adapter_version == "2"
    values = tuple(range(1, (4, 4, 1)[grade]+1))
    rational = infer_model(state, source, cpp_correspondence_input(state, source, values))
    integer = infer_model(state, source, cpp_correspondence_input(state, source, values, integer=True))
    assert tuple(rational.values) == tuple(integer.values)
    assert len(integer.dependencies) == 3 and integer.dependencies[2].matches(catalog.source)
    assert integer.execution_receipt["parameter_origin"]["catalog_digest"] == catalog.coefficient_digest
    field = integer.tensor()
    assert field.source.matches(state.dependencies[1]) and any(v.matches(catalog.source) for v in field.dependencies)
    assert unpack_model(pack_model(integer)).coefficient_digest == integer.coefficient_digest
    n = (4, 4, 1)[grade]
    probabilities = (Q(1), Q(0), Q(0)) if feature < 2 else (Q(0), Q(1), Q(0)) if feature < 4 else (Q(5, 8), Q(3, 8), Q(0)) if feature == 4 else (Q(0), Q(0), Q(1))
    assert tuple(integer.values) == tuple(p*x for p in probabilities for x in values)
    assert len(integer.values) == n*3


def test_unknown_context_uses_declared_unresolved_template():
    catalog, owner, _ = fitted()
    state, source = learned(catalog, owner, literal=(b"bank", "left", b"stone"))
    assert state.payload["counts"] == (0, 0, 0) and state.configuration["parameter_origin"]["feature"] is None
    output = infer_model(state, source, cpp_correspondence_input(state, source, (1, 0, 0, 0), integer=True))
    assert tuple(output.values)[8] == 1


@pytest.mark.parametrize("fault", ["validation", "test", "event", "knowledge", "label_time", "feature", "label", "literal", "owner", "archive", "fitted", "key"])
def test_invalid_annotations_do_not_modify_parameters(fault):
    state, owner, _ = catalog_fixture()
    sample = thaw_tree(annotations(1)[0])
    if fault in {"validation", "test"}: sample["partition"] = fault
    elif fault == "event": sample["event_time"] = 51
    elif fault == "knowledge": sample["knowledge_time"] = 251
    elif fault == "label_time": sample["label_available_time"] = 251
    elif fault == "feature": sample["feature"] = 6
    elif fault == "label": sample["label"] = 3
    elif fault == "literal": sample["feature"] = 2
    elif fault == "owner": sample["chart_source"]["reference"]["native"]["namespace"] += 1
    elif fault == "archive": sample["chart_source"]["archive"][8] ^= 1
    elif fault == "key": sample["key"] = ""
    else:
        from rexgraph.value_codec import pack_value
        context, source, *_ = fixture(counts=(1, 0, 0))
        sample["context"] = pack_value(pack_model(context, native=True))
        sample["chart_source"] = model_record(source, context).get_metadata(1, 0, "model_source")
    digest = state.coefficient_digest
    with pytest.raises((ValueError, TypeError)):
        train_model(state, owner, cpp_context_training_batch(state, owner, (sample,), event_cutoff=50, knowledge_cutoff=250))
    assert state.coefficient_digest == digest and not any(sum(row) for row in state.payload["counts"])


@pytest.mark.parametrize("fault", ["counts", "history", "parent", "forged_parent", "cutoff", "feature", "bytes", "bound", "step"])
def test_portable_fitted_parameters_reject_inconsistent_history(fault):
    state, owner, _ = fitted()
    c, p = thaw_tree(state.configuration), thaw_tree(state.payload)
    changes = {}
    if fault == "counts": p["counts"] = ((7, 0, 0), *p["counts"][1:])
    elif fault == "history": p["history"] = ()
    elif fault == "parent": changes["parent"] = "1"*64
    elif fault == "forged_parent":
        changes["parent"] = "1"*64
        p["history"][0]["parent"] = "1"*64
    elif fault == "cutoff": p["history"][0]["knowledge_cutoff"] = 100
    elif fault == "feature": c["features"] = (FEATURES[2], *c["features"][1:])
    elif fault == "bytes": c["max_bytes"] = 1
    elif fault == "bound": c["max_samples"] = 31
    else: changes["step"] = 2
    with pytest.raises((ValueError, TypeError)):
        replace(state, configuration=c, payload=p, **changes)


@pytest.mark.parametrize("name", ["event_cutoff", "knowledge_cutoff"])
def test_fitted_parameters_cannot_appear_before_availability(name):
    state, owner, _ = fitted()
    kwargs = dict(event_cutoff=60, knowledge_cutoff=260); kwargs[name] = 0
    with pytest.raises(ValueError, match="unavailable"):
        infer_model(state, owner, cpp_context_catalog_input(state, owner, **kwargs))


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
@pytest.mark.parametrize("kind", ["catalog", "learned"])
def test_cold_checkpoint_restores_training_evidence_and_reuses_model(suffix, kind, tmp_path):
    catalog, owner, _ = fitted()
    state, source = (catalog, owner) if kind == "catalog" else learned(catalog, owner, 4)
    path = tmp_path/(kind+suffix)
    save_model(path, model_record(source, state))
    code = """
import json,sys
from rexgraph.io.model_record import load_model,read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_correspondence import cpp_correspondence_input
from rexgraph.cpp_context_catalog import cpp_context_catalog_input
s,o=read_model_record(load_model(sys.argv[1]))
x=cpp_context_catalog_input(s,o,event_cutoff=60,knowledge_cutoff=260) if s.adapter=='cpp_context_catalog' else cpp_correspondence_input(s,o,(1,0,0,0),integer=True)
y=infer_model(s,o,x)
print(json.dumps(dict(model=s.coefficient_digest,output=y.coefficient_digest)))
"""
    inputs = cpp_context_catalog_input(state, source, event_cutoff=60, knowledge_cutoff=260) if kind == "catalog" else cpp_correspondence_input(state, source, (1, 0, 0, 0), integer=True)
    output = infer_model(state, source, inputs)
    actual = json.loads(subprocess.check_output([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, text=True))
    assert actual == dict(model=state.coefficient_digest, output=output.coefficient_digest)


def test_rcql_training_inference_agent_and_system_share_fitted_catalog():
    from agent.hive import Hive
    from rcql import Executor, parse
    from system.serialize import json_value
    state, owner, pattern = catalog_fixture()
    batch = cpp_context_training_batch(state, owner, annotations(), event_cutoff=50, knowledge_cutoff=250)
    record = model_record(owner, state)
    trained = Executor(sources={"record": record}, params={"model": state, "batch": batch}).execute(parse(
        'FROM $record RETURN MODEL_TRAIN($model,$batch)')).values[0]
    assert trained.payload["counts"] == fitted()[0].payload["counts"]
    record = model_record(owner, trained)
    inputs = cpp_context_catalog_input(trained, owner, event_cutoff=60, knowledge_cutoff=260)
    output = Executor(sources={"record": record}, params={"model": trained, "input": inputs}).execute(parse(
        'FROM $record RETURN MODEL_INFER($model,$input)')).values[0]
    hive = Hive("fitted-context-catalog")
    hive.add_worker("interpret", lambda value: infer_model(trained, owner, value), capability="predict")
    assert hive.invoke("interpret", inputs, record=False).coefficient_digest == output.coefficient_digest
    assert json_value(output)["arithmetic"] == "rational"
    config = dict(pattern=pattern, carrier=12, features=FEATURES, labels=LABELS, unresolved=2)
    created = Executor(sources={"record": record}, params={"config": config}).execute(parse(
        'FROM $record RETURN MODEL_INIT("cpp_context_catalog",$config)')).values[0]
    assert created.adapter == "cpp_context_catalog" and created.step == 0


@pytest.mark.parametrize("backend", ["sqlite", "file", "rex"])
def test_rcdb_copy_restores_learned_counts_and_original_sources(backend, tmp_path):
    from rcdb import copy_record, open_store
    catalog, owner, _ = fitted()
    state, source = learned(catalog, owner, 4)
    record = model_record(source, state)
    stores = [open_store(f"{backend}:///{tmp_path/name}", **({} if backend == "sqlite" else {"read_only": False})) for name in ("first", "second")]
    try:
        stores[0].put("model", record)
        receipt = copy_record(stores[0], stores[1], stores[0].get_record("model"), return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
        restored, actual_source = read_model_record(stores[1].get_version("model", 1))
        assert restored.coefficient_digest == state.coefficient_digest
        assert prepare_correspondence(restored, actual_source)[3] == 8
    finally:
        for store in stores: store.close()


@pytest.mark.parametrize("fault", ["count", "index", "labels", "width", "limit", "bool"])
def test_checked_native_count_kernel_rejects_invalid_or_overflowing_inputs(fault):
    counts, features, labels, maximum = ((1, 0), (0, 0)), (0,), (1,), 100
    if fault == "count": counts = ((-1, 0),)
    elif fault == "index": features = (2,)
    elif fault == "labels": labels = (2,)
    elif fault == "width": counts = ((0,), (0, 0))
    elif fault == "limit": maximum = 1
    else: features = (True,)
    with pytest.raises((ValueError, TypeError, IndexError, RuntimeError)):
        fit_context_counts(counts, features, labels, maximum=maximum)


def test_native_counts_match_independent_integer_observations():
    features = tuple(i % 17 for i in range(10000))
    labels = tuple(i % 3 for i in range(10000))
    result = fit_context_counts(((0, 0, 0),)*17, features, labels)
    expected = [[0, 0, 0] for _ in range(17)]
    for feature, label in zip(features, labels, strict=True): expected[feature][label] += 1
    assert result == tuple(tuple(row) for row in expected)


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_restored_catalog_can_continue_training_on_a_new_annotation(suffix, tmp_path):
    state, owner, _ = fitted()
    path = tmp_path/("catalog"+suffix)
    save_model(path, model_record(owner, state))
    restored, actual_owner = read_model_record(load_model(path))
    context, chart, *_ = occurrence(0, 29999)
    sample = cpp_context_annotation("continuation-29999", context, chart, feature=0, label=0,
                                   event_time=61, knowledge_time=261, label_available_time=262)
    child = train_model(restored, actual_owner, cpp_context_training_batch(restored, actual_owner,
        (sample,), event_cutoff=65, knowledge_cutoff=280))
    assert child.step == 2 and child.parent == state.coefficient_digest
    assert child.payload["counts"][0] == (7, 0, 0)
    assert prepare_context_catalog(child, actual_owner) == child.payload["counts"]


def test_core_only_inspection_checks_counts_and_ancestry_without_loading_binding():
    state, owner, _ = fitted()
    from rexgraph.value_codec import pack_value
    wire = pack_value(pack_model(state, native=True))
    code = """
import importlib.abc,sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith('rexgraph_cpp'): raise ImportError('native binding disabled')
sys.meta_path.insert(0,Reject())
from rexgraph.value_codec import unpack_value
from rexgraph.model_codec import unpack_model
s=unpack_model(unpack_value(sys.stdin.buffer.read()))
assert s.payload['counts'][4]==(5,3,0)
assert not any(k.startswith('rexgraph_cpp') for k in sys.modules)
print(s.coefficient_digest)
"""
    actual = subprocess.check_output([sys.executable, "-I", "-c", code], input=wire, cwd="/tmp").decode().strip()
    assert actual == state.coefficient_digest


@pytest.mark.parametrize("fault", ["counts", "feature", "availability", "catalog_source"])
def test_learned_maps_reject_changed_parameter_origins(fault):
    catalog, owner, _ = fitted()
    state, source = learned(catalog, owner, 4)
    c, p = thaw_tree(state.configuration), thaw_tree(state.payload)
    if fault == "counts": p["counts"] = (6, 2, 0)
    elif fault == "feature": c["parameter_origin"]["feature"] = 0
    elif fault == "availability": c["parameter_origin"]["knowledge_cutoff"] = 249
    else: c["parameter_origin"]["catalog_source"]["reference"]["native"]["namespace"] += 1
    with pytest.raises(ValueError):
        replace(state, configuration=c, payload=p)


def test_rcql_field_of_learned_occurrence_binds_the_actual_pattern_owner():
    from rcql import Executor, parse
    catalog, owner, _ = fitted()
    state, source = learned(catalog, owner, 4)
    output = infer_model(state, source, cpp_correspondence_input(state, source, (1, 0, 0, 0), integer=True))
    field = Executor(sources={"record": model_record(source, state)}, params={"output": output}).execute(parse(
        'FROM $record RETURN MODEL_FIELD($output)')).values[0]
    assert field.source.matches(state.dependencies[1])
    assert any(v.matches(catalog.source) for v in field.dependencies)
    assert tuple(field.values) == tuple(output.values)


def test_prepared_cache_retains_seal_checks_before_reuse():
    catalog, owner, _ = fitted()
    state, source = learned(catalog, owner, 4, 29998)
    first = prepare_correspondence(state, source)
    assert prepare_correspondence(state, source) is first
    object.__setattr__(state, "payload", {"counts": (6, 2, 0)})
    with pytest.raises(ValueError, match="changed after construction"):
        prepare_correspondence(state, source)

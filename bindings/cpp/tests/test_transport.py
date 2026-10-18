"""Native structural bijections, parameter handoff and exact destination resume."""
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import pytest
from rexgraph_cpp import CancellationToken, ExecutionCancelled, ExecutionLimitError, NativeSource
from test_construction import LABELS, ROWS, batch, declaration, model

from rexgraph.cpp_learning import (
    cpp_integer_input,
    cpp_integer_model,
    cpp_trainable_model,
    cpp_training_batch,
    prepare_learning_model,
)
from rexgraph.cpp_model import cpp_model
from rexgraph.cpp_perceptron import cpp_perceptron_batch, cpp_perceptron_input
from rexgraph.cpp_transport import cpp_transport
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model, transport_model
from rexgraph.model_state import thaw_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.value_codec import pack_value

MAPS = ((1, 2, 0), (2, 0, 1), (1, 0))


def permute(values, indices):
    result = [None] * len(values)
    for i, j in enumerate(indices):
        result[j] = values[i]
    return tuple(result)


def relabel(d, parameters=None, feature_map=(1, 0)):
    target = deepcopy(d)
    target["namespace"] = 188
    target["witnesses"] = permute((10010, 10011, 10012), MAPS[0])
    relations = []
    for k in (1, 2):
        rows = []
        for r in d["relations"]:
            if r["grade"] != k: continue
            rows.append({**r, "key": r["key"]+10000, "occurrences": tuple({**o,
                "key": o["key"]+20000, "lower": o["lower"]+10000} for o in r["occurrences"])})
        relations.extend(permute(rows, MAPS[k]))
    target["relations"] = relations
    carriers = {c["key"]: c for c in d["carriers"]}
    target["carriers"] = [{**c, "key": c["key"]+100,
        "coordinates": c["coordinates"]+1000 if c["coordinates"] else 0} for c in d["carriers"]]
    fields = []
    for field in d["fields"]:
        c = carriers[field["carrier"]]
        p = MAPS[c["grade"]] if c["kind"] == "source_grade" else feature_map if c["key"] == 40 else (0,)
        values = field["entries"]
        if parameters is not None and field["key"] in {4000, 4001}:
            data = parameters[:-1] if field["key"] == 4000 else parameters[-1:]
            values = tuple(enumerate(data))
        fields.append({**field, "key": field["key"]+1000, "carrier": field["carrier"]+100,
                       "entries": tuple((p[i], value) for i, value in values),
                       "support": None if field["support"] is None else tuple(p[i] for i in field["support"])})
    target["fields"] = fields
    target["metrics"] = [{**metric, "key": metric["key"]+1000, "carrier": metric["carrier"]+100,
                          "diagonal": permute(metric["diagonal"], MAPS[carriers[metric["carrier"]]["grade"]])}
                         for metric in d["metrics"]]
    return target


def links(source, feature_map=(1, 0)):
    return tuple(dict(source=key, target=key+100,
                      indices=MAPS[grade] if kind == 1 else feature_map if key == 40 else (0,))
                 for key, grade, kind in source.carriers)


def tensor_links(source):
    return tuple(dict(source=key, target=key+1000) for key, _, _ in source.tensors)


def metric_declaration():
    return dict(grade=1, channels=("identity", "hodge"), model="transport-test", gains=4000, bias=4001,
                channel_coordinates=900, bias_coordinates=901, history_namespace=700,
                metadata_carrier=9900, metadata_tensor=9901, time_coordinates=9902, metadata_coordinates=9903,
                initial_parameter_knowledge_time=2, publication_cutoff=0,
                samples=(dict(input=1001, target=1001, event_time=3, knowledge_time=4, label_available_time=5,
                              partition="training"),
                         dict(input=1003, target=1001, event_time=3, knowledge_time=4, label_available_time=5,
                              partition="training")))


def family(adapter, *, metadata_capacity=None):
    d = declaration()
    d["fields"].append(dict(key=1003, carrier=11, entries=((0, Q(-1, 7)), (1, Q(2, 3))), support=None))
    source = NativeSource.from_declaration(d)
    if adapter == "cpp_perceptron":
        state = train_model(model(source), source, batch(model(source), source), steps=3)
    elif adapter == "cpp_metric":
        dc = metric_declaration()
        if metadata_capacity is not None:
            dc["metadata_capacity"] = metadata_capacity
        initial = cpp_trainable_model(source, (10, 11, 12), dc,
                                      metric_keys=(2000, 2001, 2002), metric_selection="diagonal")
        state = train_model(initial, source, cpp_training_batch(initial, source, rate=Q(1, 1000), event_cutoff=3, knowledge_cutoff=10))
    elif adapter == "cpp_integer":
        state = cpp_integer_model(source, 11, (dict(channels=("identity", "hodge"), gains=(2, -1), bias=3, activation="relu"),))
    else:
        state = cpp_model(source, (10, 11, 12), metric_keys=(2000, 2001, 2002), metric_selection="diagonal")
    target = NativeSource.from_declaration(relabel(d, state.payload.get("parameters")))
    c = thaw_tree(state.configuration)
    if adapter == "cpp_perceptron":
        config = dict(carrier=111, weights_key=5000, bias_key=5001, availability=permute(c["availability"], MAPS[1]),
                      feature_coordinates=1900, time_coordinates=999, initial_parameter_knowledge_time=20)
    elif adapter == "cpp_metric":
        dc = c["declaration"]
        dc.update(gains=5000, bias=5001, channel_coordinates=1900, bias_coordinates=1901,
                  channels=permute(dc["channels"], (1, 0)), history_namespace=800,
                  initial_parameter_knowledge_time=10)
        offset = state.step % len(dc["samples"])
        dc["samples"] = tuple({**s, "input": s["input"]+1000, "target": s["target"]+1000}
                              for s in (*dc["samples"][offset:], *dc["samples"][:offset]))
        config = dict(grade_carriers=(110, 111, 112), declaration=dc, metric_keys=(3000, 3001, 3002), metric_selection="diagonal")
    elif adapter == "cpp_integer":
        config = dict(carrier=111, layers=c["layers"])
    else:
        config = {k: v for k, v in c.items() if k not in {"execution_provider", "engine_receipt", "input_profile", "batch_input_profile"}}
        config.update(grade_carriers=(110, 111, 112), metric_keys=(3000, 3001, 3002))
    mapping = cpp_transport(source, target, MAPS, links(source), tensor_links(source), configuration=config)
    return source, state, target, mapping


def rows_after():
    return permute(tuple(tuple((1-i, value) for i, value in row) for row in ROWS), MAPS[1])


def inference(state, source, destination=False):
    if state.adapter == "cpp_perceptron":
        inputs = cpp_perceptron_input(state, source, rows_after() if destination else ROWS)
    elif state.adapter == "cpp_integer":
        inputs = cpp_integer_input(state, source, permute((1, -2, 3), MAPS[1]) if destination else (1, -2, 3))
    else:
        inputs = source.field(2001 if destination else 1001)
    return infer_model(state, source, inputs)


def next_batch(state, source):
    if state.adapter == "cpp_metric":
        return cpp_training_batch(state, source, rate=Q(1, 1000), event_cutoff=10, knowledge_cutoff=21)
    return cpp_perceptron_batch(state, source, rows_after(), permute(tuple(-v for v in LABELS), MAPS[1]),
                                rate=2, event_cutoff=10, knowledge_cutoff=21)


def test_native_bijection_preserves_fields_metrics_and_occurrence_order(native_fixtures):
    source = NativeSource.from_declaration(declaration())
    target = NativeSource.from_declaration(relabel(declaration()))
    native = source.transport(target, MAPS, links(source))
    oracle = json.loads((native_fixtures[0] / "transport.json").read_text())
    assert native.occurrence_maps == tuple(tuple(row) for row in oracle["occurrences"])
    for k, values in enumerate(oracle["fields"]):
        assert native.push_values(10+k, source.field(1000+k).values)["values"] == tuple(Q(v) for v in values)
    assert native.occurrence_maps == ((), (4, 5, 6, 7, 8, 0, 1, 2, 3), (0, 1))
    for item in tensor_links(source): native.check_tensor(item["source"], item["target"])
    result = native.push_values(11, (Q(1, 3), 0, Q(-2, 5)))
    assert result["values"] == (0, Q(-2, 5), Q(1, 3))
    assert result["output_carrier"] == 111
    assert result["grade"] == 1
    assert target.field(4002).support == (2,)
    assert target.field(2001).support == (0, 1, 2)
    del source, target
    assert native.push_values(40, (1, -2))["values"] == (-2, 1)


@pytest.mark.parametrize("change", ["order", "inactive", "type", "scale", "orientation"])
def test_structural_change_rejects_even_when_dimensions_agree(change):
    source = NativeSource.from_declaration(declaration())
    d = relabel(declaration())
    r = next(v for v in d["relations"] if v["key"] == 10100)
    if change == "order": r["occurrences"] = tuple(reversed(r["occurrences"]))
    elif change == "inactive": r["occurrences"][-1]["lower"] = 10011
    elif change == "type": d["types"][0]["name"] = "another-type"
    elif change == "scale": d["semantic_scale"] = 1
    else: r["occurrences"][-1]["negative"] = True
    target = NativeSource.from_declaration(d)
    with pytest.raises(ValueError): source.transport(target, MAPS, links(source))


@pytest.mark.parametrize("change", ["partial", "duplicate", "range", "carrier", "carrier_map", "unknown", "float", "bool"])
def test_invalid_map_rejects(change):
    source = NativeSource.from_declaration(declaration()); target = NativeSource.from_declaration(relabel(declaration()))
    maps, cs = list(MAPS), list(links(source))
    if change == "partial": maps.pop()
    elif change == "duplicate": maps[1] = (0, 0, 1)
    elif change == "range": maps[1] = (3, 0, 1)
    elif change == "carrier": cs[0]["target"] = 111
    elif change == "carrier_map": cs[1]["indices"] = (0, 1, 2)
    elif change == "unknown": cs[0]["extra"] = 1
    elif change == "float": maps[0] = (1.0, 2, 0)
    else: maps[0] = (True, 2, 0)
    with pytest.raises((ValueError, TypeError)): source.transport(target, maps, cs)


@pytest.mark.parametrize("kind", ["support", "zero", "metric"])
def test_tensor_equality_distinguishes_support_zero_and_metric(kind):
    source = NativeSource.from_declaration(declaration()); d = relabel(declaration())
    if kind == "support": d["fields"][1]["support"] = (1, 2)
    elif kind == "zero": d["fields"][1]["entries"] += ((0, 1),)
    else: d["metrics"][1]["diagonal"] = (9, 2, 3)
    target = NativeSource.from_declaration(d); native = source.transport(target, MAPS, links(source))
    with pytest.raises(ValueError): native.check_tensor(2001 if kind == "metric" else 1001, 3001 if kind == "metric" else 2001)


def test_limits_and_owned_cancellation():
    source = NativeSource.from_declaration(declaration()); target = NativeSource.from_declaration(relabel(declaration()))
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled): source.transport(target, MAPS, links(source), cancellation=token)
    with pytest.raises(ExecutionLimitError): source.transport(target, MAPS, links(source), limits=(1000, 1000, 1000, 1, 1000))
    native = source.transport(target, MAPS, links(source))
    with pytest.raises(ExecutionCancelled): native.push_values(11, (1, 2, 3), cancellation=token)
    with pytest.raises(ExecutionCancelled): native.check_tensor(1001, 2001, cancellation=token)
    with pytest.raises(TypeError): native.push_values(11, (1, 2.0, 3))


@pytest.mark.parametrize("adapter", ["cpp_field", "cpp_integer", "cpp_metric", "cpp_perceptron"])
def test_model_transport_preserves_predictions_and_complete_origin(adapter):
    source, state, target, mapping = family(adapter)
    child = transport_model(state, source, CppSourceReference(target), mapping)
    assert child.parent == state.coefficient_digest and child.step == state.step
    assert child.source.matches(CppSourceReference(target))
    assert tuple(inference(child, target, True).values) == permute(tuple(inference(state, source).values), MAPS[1])
    assert child.payload["transport"]["origin_archive"] == source.archive
    assert unpack_model(child.payload["transport"]["origin_model"]).coefficient_digest == state.coefficient_digest
    assert inference(child, target, True).execution_receipt["transport_base_step"] == state.step
    if adapter in {"cpp_metric", "cpp_perceptron"}:
        expected = (*reversed(state.payload["parameters"][:-1]), state.payload["parameters"][-1])
        assert child.payload["parameters"] == expected
        next_child = train_model(child, target, next_batch(child, target), steps=2)
        assert next_child.step == child.step+2
        assert pack_value(thaw_tree(next_child.payload["transport"])) == pack_value(thaw_tree(child.payload["transport"]))
    with pytest.raises(NotImplementedError): transport_model(child, target, CppSourceReference(target), mapping)


def test_rational_sample_cursor_and_gradient_continue_after_transport():
    source, state, target, mapping = family("cpp_metric")
    child = transport_model(state, source, CppSourceReference(target), mapping)
    before = prepare_learning_model(child, target)
    assert before.snapshot["step"] == 0 and before.parameter_knowledge_time == 10
    actual = train_model(child, target, next_batch(child, target), steps=3)
    expected = train_model(state, source, cpp_training_batch(state, source, rate=Q(1, 1000), event_cutoff=10, knowledge_cutoff=21), steps=3)
    assert actual.payload["parameters"] == (*reversed(expected.payload["parameters"][:-1]), expected.payload["parameters"][-1])
    output = inference(actual, target, True)
    assert output.execution_receipt["parameter_history_step"] == 3
    assert output.execution_receipt["parameter_step"] == 4
    assert output.execution_receipt["parameter_knowledge_time"] == 21


@pytest.mark.parametrize("adapter", ["cpp_metric", "cpp_perceptron"])
def test_future_parameters_stay_unavailable_after_transport(adapter):
    source, state, target, mapping = family(adapter)
    child = transport_model(state, source, CppSourceReference(target), mapping)
    if adapter == "cpp_metric":
        observations = cpp_training_batch(child, target, rate=Q(1, 1000), event_cutoff=10, knowledge_cutoff=9)
    else:
        observations = cpp_perceptron_batch(child, target, rows_after(), LABELS, event_cutoff=10, knowledge_cutoff=19)
    with pytest.raises(ValueError): train_model(child, target, observations)


@pytest.mark.parametrize("adapter", ["cpp_field", "cpp_integer", "cpp_metric", "cpp_perceptron"])
@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_transport_records_restore_and_resume(adapter, suffix, tmp_path):
    source, state, target, mapping = family(adapter)
    child = transport_model(state, source, CppSourceReference(target), mapping)
    path = tmp_path/f"transport{suffix}"
    save_model(path, model_record(target, child))
    restored, loaded = read_model_record(load_model(path))
    assert restored.coefficient_digest == child.coefficient_digest
    assert tuple(inference(restored, loaded, True).values) == tuple(inference(child, target, True).values)
    if adapter in {"cpp_metric", "cpp_perceptron"}:
        resumed = train_model(restored, loaded, next_batch(restored, loaded), steps=2)
        expected = train_model(child, target, next_batch(child, target), steps=2)
        assert resumed.coefficient_digest == expected.coefficient_digest
        save_model(path, model_record(loaded, resumed))
        assert read_model_record(load_model(path))[0].coefficient_digest == resumed.coefficient_digest


@pytest.mark.parametrize("change", ["parameters", "origin_digest", "base_step", "origin_archive", "origin_model", "mapping", "extra"])
def test_forged_transport_checkpoint_rejects(change):
    source, state, target, mapping = family("cpp_perceptron")
    child = transport_model(state, source, CppSourceReference(target), mapping)
    payload = thaw_tree(child.payload)
    proof = payload["transport"]
    if change == "parameters": payload["parameters"] = (9, 9, 9)
    elif change == "origin_digest": proof["origin_digest"] = "0"*64
    elif change == "base_step": proof["base_step"] += 1
    elif change == "origin_archive": proof["origin_archive"] = target.archive
    elif change == "origin_model":
        forged = replace(state, payload={**dict(state.payload), "parameters": (9, 9, 9)})
        proof["origin_model"] = pack_model(forged)
        proof["origin_digest"] = forged.coefficient_digest
    elif change == "mapping": proof["mapping"]["time_policy"] = "guess"
    else: proof["extra"] = 1
    broken = replace(child, payload=payload)
    with pytest.raises((ValueError, TypeError)): inference(broken, target, True)


@pytest.mark.parametrize("change", ["target_reference", "source_reference", "configuration", "tensor_link", "carry", "destination_parameters", "parent"])
def test_transport_rejection_leaves_parent_unchanged(change):
    source, state, target, mapping = family("cpp_perceptron")
    before = state.coefficient_digest
    record = thaw_tree(mapping)
    ref = CppSourceReference(target)
    if change == "target_reference": record["destination"]["record_id"] = "other-record"
    elif change == "source_reference": record["source"]["record_id"] = "other-record"
    elif change == "configuration": record["configuration"]["time_coordinates"] = 998
    elif change == "tensor_link": record["tensor_links"] = record["tensor_links"][:-1]
    elif change == "destination_parameters":
        d = relabel(declaration()); target = NativeSource.from_declaration(d); ref = CppSourceReference(target)
        record = thaw_tree(cpp_transport(source, target, MAPS, links(source), tensor_links(source), configuration=record["configuration"]))
    if change == "parent":
        child = transport_model(state, source, ref, mapping)
        with pytest.raises(ValueError): inference(replace(child, parent="0"*64), target, True)
    else:
        with pytest.raises((ValueError, TypeError)):
            transport_model(state, source, ref, record, optimizer="carry" if change == "carry" else "reset")
    assert state.coefficient_digest == before


def test_proof_limit_and_unknown_declaration_reject(monkeypatch):
    source, state, target, mapping = family("cpp_perceptron")
    record = {**dict(mapping), "extra": True}
    with pytest.raises(ValueError): transport_model(state, source, CppSourceReference(target), record)
    monkeypatch.setattr("rexgraph.cpp_transport._proof_limit", 1)
    with pytest.raises(ValueError, match="proof byte bound"):
        transport_model(state, source, CppSourceReference(target), mapping)


def test_cold_restore_replays_origin_but_live_updates_reuse_certification(monkeypatch):
    from rexgraph import cpp_transport as module
    source, state, target, mapping = family("cpp_metric")
    child = transport_model(state, source, CppSourceReference(target), mapping)
    original = module.unpack_model
    calls = []
    def counted(value):
        calls.append(1)
        return original(value)
    monkeypatch.setattr(module, "unpack_model", counted)
    module._cache.clear()
    inference(child, target, True)
    assert calls == [1]
    following = train_model(child, target, next_batch(child, target), steps=2)
    inference(following, target, True)
    assert calls == [1]


@pytest.mark.parametrize("operation", ["boundary", "adjoint", "hodge_down", "hodge_up", "green", "harmonic"])
def test_all_fixed_field_actions_transport_their_declared_input_and_output_grades(operation):
    source = NativeSource.from_declaration(declaration()); target = NativeSource.from_declaration(relabel(declaration()))
    profile = "global" if operation in {"green", "harmonic"} else "finite"
    state = cpp_model(source, (10, 11, 12), operation=operation, profile=profile,
                      metric_keys=(2000, 2001, 2002), metric_selection="diagonal")
    config = dict(grade_carriers=(110, 111, 112), operation=operation, profile=profile,
                  metric_keys=(3000, 3001, 3002), metric_selection="diagonal")
    mapping = cpp_transport(source, target, MAPS, links(source), tensor_links(source), configuration=config)
    child = transport_model(state, source, CppSourceReference(target), mapping)
    grade = 0 if operation == "adjoint" else 1
    old = infer_model(state, source, source.field(1000+grade))
    new = infer_model(child, target, target.field(2000+grade))
    assert tuple(new.values) == permute(tuple(old.values), MAPS[old.grade])


@pytest.mark.parametrize("adapter", ["cpp_field", "cpp_integer", "cpp_metric", "cpp_perceptron"])
def test_rcql_transport_uses_the_selected_native_destination(adapter):
    from rcql.binding import bind

    from rcql import Executor, SourcePolicy, parse, plan_query
    source, state, target, mapping = family(adapter)
    from rexgraph.cpp_model import _create
    base = _create(target, adapter=adapter, **mapping["configuration"])
    destination = bind("destination", model_record(target, base), SourcePolicy.allow("*"))
    executor = Executor(sources={"original": model_record(source, state)},
                        params={"destination": destination, "mapping": mapping})
    query = parse('FROM $original RETURN MODEL_TRANSPORT(MODEL_STATE(),$destination,$mapping)')
    result = executor.execute(query)
    child = result.values[0]
    origin = bind("original", model_record(source, state), SourcePolicy.allow("*"))
    plan = plan_query(origin, query, parameters=executor.params)
    assert plan.returns[0].result.source == destination.ref
    assert child.coefficient_digest == transport_model(state, source, CppSourceReference(target), mapping).coefficient_digest
    assert tuple(inference(child, target, True).values) == permute(tuple(inference(state, source).values), MAPS[1])


@pytest.mark.parametrize("adapter", ["cpp_metric", "cpp_perceptron"])
@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
def test_fresh_process_transport_resume_and_application_routes(adapter, backend, tmp_path):
    from test_learning import NO_TORCH

    from rcdb import open_store
    source, state, target, mapping = family(adapter)
    child = transport_model(state, source, CppSourceReference(target), mapping)
    expected = train_model(child, target, next_batch(child, target), steps=2)
    url = {"file": "file://"+str(tmp_path/"file"), "rex": "rex://"+str(tmp_path/"rex"),
           "sqlite": "sqlite:///"+str(tmp_path/"records.sqlite")}[backend]
    store = open_store(url, **({} if backend == "sqlite" else {"read_only": False})).configure_security(require_commits=True)
    store.commit_mutation("transported", model_record(target, child), expected_version=0)
    assert store.verify_commits("transported"); store.close()
    script = NO_TORCH + '''
from unittest.mock import patch
from fractions import Fraction as Q
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.io.model_record import read_model_record,model_record
from rexgraph.cpp_learning import cpp_training_batch
from rexgraph.cpp_perceptron import cpp_perceptron_batch,cpp_perceptron_input
from rexgraph.model_runtime import infer_model,train_model
store=open_store(sys.argv[1])
state,source=read_model_record(store.read_record("transported",version=1).value)
if state.adapter=="cpp_metric":
    batch=cpp_training_batch(state,source,rate=Q(1,1000),event_cutoff=10,knowledge_cutoff=21)
    inputs={"field_key":2001}
else:
    rows=(((1,-2),(0,-1)),((1,1),(1,1),(0,2)),((1,2),(0,1)))
    batch=cpp_perceptron_batch(state,source,rows,(1,-1,-1),rate=2,event_cutoff=10,knowledge_cutoff=21)
    inputs=cpp_perceptron_input(state,source,rows)
child=train_model(state,source,batch,steps=2)
out=infer_model(child,source,inputs)
preview=json_value(child)
assert preview["resumable"] and preview["transport"]["base_step"]==state.step
assert preview["parameter_history_step"]==2
record=model_record(source,child)
result=Executor(sources={"record":record},params={"input":inputs}).execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$input)'))
assert result.values[0].coefficient_digest==out.coefficient_digest
with patch("agent.activity.record"),patch("agent.activity.open_use",return_value=0),patch("agent.activity.close_use"):
    hive=Hive("private-transport")
    hive.add_worker("native-model",lambda value,**kw:infer_model(child,source,value),capability="predict")
    assert hive.invoke("native-model",inputs,record=False).coefficient_digest==out.coefficient_digest
assert not any(name=="torch" or name.startswith("torch.") for name in sys.modules)
print(child.coefficient_digest)
store.close()
'''
    result = subprocess.run([sys.executable, "-c", script, url], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected.coefficient_digest

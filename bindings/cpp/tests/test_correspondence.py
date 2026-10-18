"""Owned context maps preserve complete boundaries, bytes and exact parameters."""
import gc
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph_cpp import NativeSource

from rexgraph.cpp_correspondence import (
    cpp_correspondence_input, cpp_correspondence_model, prepare_correspondence,
)
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import ModelBatch, thaw_tree
from rexgraph.state import from_state, to_state


def declaration(namespace, scale, sizes, relations, witnesses, fields=()):
    return dict(profile="declared-exact-source-v1", namespace=namespace, semantic_scale=scale,
        max_grade=2, types=(dict(name="context-relation", min_arity=0, max_arity=4),),
        witnesses=witnesses, relations=relations, fields=fields, metrics=(),
        carriers=tuple(dict(key=10+k, kind="source_grade", grade=k, coordinates=0, size=n)
                       for k, n in enumerate(sizes)))


def relation(grade, key, lowers, signs, start):
    return dict(grade=grade, key=key, type=0, occurrences=tuple(
        dict(key=start+i, lower=lower, exists=True, negative=negative, magnitude=1)
        for i, (lower, negative) in enumerate(zip(lowers, signs, strict=True))))


def chart_parts(offset=0):
    vertices = tuple(1000 + offset + i for i in range(4))
    edges = tuple(2000 + offset + i for i in range(4))
    relations = [relation(1, edge, (vertices[a], vertices[b]), (True, False), offset+2*i)
                 for i, (edge, (a, b)) in enumerate(zip(edges, ((0, 1), (1, 2), (2, 3), (0, 3)), strict=True))]
    face = relation(2, 3000+offset, edges, (False, False, False, True), offset+8)
    return vertices, relations, face


def fixture(raw=b"river bank money", spans=((0, 5), (6, 10)), mention="right", counts=(5, 3, 0), grade=1, chart_namespace=102):
    n = len(raw)
    vertices = tuple(100+i for i in range(n+1))
    edges = [relation(1, 200+i, (vertices[i], vertices[i+1]), (True, False), 2*i) for i in range(n)]
    fields = (dict(key=1001, carrier=11, entries=tuple(enumerate(raw)), support=None),)
    byte_source = NativeSource.from_declaration(declaration(101, 0, (n+1, n, 0), edges, vertices, fields))
    v, e, f = chart_parts()
    source = NativeSource.from_declaration(declaration(chart_namespace, 1, (4, 4, 1), [*e, f], v))
    all_v, all_e, all_f = [], [], []
    for j in range(3):
        v, e, f = chart_parts(100*j)
        all_v.extend(v); all_e.extend(e); all_f.append(f)
    pattern = NativeSource.from_declaration(declaration(103, 0, (12, 12, 3), [*all_e, *all_f], all_v))
    templates = tuple(tuple(tuple(((j*size+i, Q(1)),) for i in range(size))
                            for size in (4, 4, 1)) for j in range(3))
    scope = dict(codec="utf-8", data_field=1001, left_span=spans[0], right_span=spans[1],
                 mention_side=mention, mention_bytes=raw[slice(*spans[int(mention == "right")])],
                 context_bytes=raw[slice(*spans[int(mention != "right")])])
    state = cpp_correspondence_model(source, byte_source, pattern, carriers=(10, 11, 12),
        raw_carriers=(10, 11, 12), pattern_carriers=(10, 11, 12), scope=scope, templates=templates,
        labels=("river-margin", "financial-institution", "unresolved"), unresolved=2, counts=counts, grade=grade)
    return state, source, byte_source, pattern


def predict(state, source, values, **kwargs):
    return infer_model(state, source, cpp_correspondence_input(state, source, values, **kwargs))


@pytest.mark.parametrize("grade", [0, 1, 2])
@pytest.mark.parametrize("counts", [(6, 0, 0), (0, 6, 0), (5, 3, 0), (0, 0, 0)])
def test_exact_integer_and_rational_maps_agree_at_every_grade(grade, counts):
    state, source, raw, pattern = fixture(counts=counts, grade=grade)
    n = (4, 4, 1)[grade]
    q = tuple((-1)**i*(i+1) for i in range(n))
    rational, integer = predict(state, source, q), predict(state, source, q, integer=True)
    assert tuple(rational.values) == tuple(integer.values)
    expected = tuple(Q(v, sum(counts)) for v in counts) if sum(counts) else (Q(0), Q(0), Q(1))
    assert tuple(rational.values) == tuple(p*x for p in expected for x in q)
    assert rational.grade == grade and rational.space.keys == pattern.identity["coordinates"][grade]
    assert all(a.matches(b) for a, b in zip(rational.dependencies, state.dependencies, strict=True))
    assert rational.execution_receipt["target"] == state.dependencies[1].as_record()
    assert integer.execution_receipt["denominator"] == (sum(counts) or 1)
    assert source.archive == prepare_correspondence(state, source)[0].source.archive
    restored = unpack_model(pack_model(integer))
    assert restored.coefficient_digest == integer.coefficient_digest


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
@pytest.mark.parametrize("counts", [(5, 3, 0), (0, 0, 0)])
def test_full_source_and_parameter_roundtrip_is_cold(suffix, counts, tmp_path):
    raw_bytes = "α中🙂 ".encode() + b"river bank money\x00"
    a = len("α中🙂 ".encode())
    state, owner, raw, pattern = fixture(raw_bytes, ((a, a+5), (a+6, a+10)), counts=counts)
    record = model_record(owner, state)
    record.attach_metadata(1, 0, "prediction", predict(state, owner, (0, 0, 1, 0), integer=True))
    path = tmp_path / ("correspondence" + suffix)
    save_model(path, record)
    code = """
import json, sys
from rexgraph.io.model_record import load_model, read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_correspondence import cpp_correspondence_input, prepare_correspondence
record = load_model(sys.argv[1]); state, source = read_model_record(record)
p = prepare_correspondence(state, source)
out = infer_model(state, source, cpp_correspondence_input(state, source, (0,0,1,0), integer=True))
assert out.coefficient_digest == record.get_metadata(1,0,'prediction').coefficient_digest
print(json.dumps(dict(model=state.coefficient_digest, raw=p[2].hex(), counts=state.payload['counts'],
                     archives=[source.archive.hex(),p[0].destination.archive.hex(),p[1].destination.archive.hex()])))
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd="/tmp", capture_output=True, text=True, check=True)
    observed = json.loads(result.stdout)
    assert observed["model"] == state.coefficient_digest
    assert bytes.fromhex(observed["raw"]) == raw_bytes
    assert observed["counts"] == list(counts)
    assert tuple(bytes.fromhex(a) for a in observed["archives"]) == (owner.archive, raw.archive, pattern.archive)
    loaded, restored = read_model_record(load_model(path))
    assert loaded.coefficient_digest == state.coefficient_digest
    assert restored is not owner and restored.identity == owner.identity


@pytest.mark.parametrize("side", ["raw", "pattern"])
def test_existing_record_state_roundtrip_and_concurrent_actions(side):
    state, source, *_ = fixture()
    state2, source2 = read_model_record(from_state(to_state(model_record(source, state))))
    expected = predict(state, source, (0, 0, 1, 0), side=side)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: predict(state2, source2, (0, 0, 1, 0), side=side), range(16)))
    assert all(out.coefficient_digest == expected.coefficient_digest for out in outputs)


def test_native_map_owns_endpoints_and_rejects_foreign_equal_publications():
    state, source, *_ = fixture()
    plan = prepare_correspondence(state, source)[1]
    equal = NativeSource(source.archive)
    with pytest.raises(ValueError, match="actual source owner"):
        plan.push(equal, 1, (0, 0, 1, 0))
    del state, source
    gc.collect()
    assert plan.push(plan.source, 2, (1,)) == (Q(5,8), Q(3,8), Q(0))
    assert plan.push(plan.source, 2, (1,), integer=True) == (5, 3, 0)


def test_same_bank_span_can_retain_two_declared_contexts():
    river, rs, *_ = fixture(counts=(6, 0, 0))
    money, ms, *_ = fixture(spans=((6, 10), (11, 16)), mention="left", counts=(0, 6, 0), chart_namespace=104)
    assert river.configuration["scope"]["mention_bytes"] == money.configuration["scope"]["mention_bytes"] == b"bank"
    assert river.dependencies[0].matches(money.dependencies[0])
    assert not river.source.matches(money.source)
    assert tuple(predict(river, rs, (0, 0, 1, 0)).values)[2] == 1
    assert tuple(predict(money, ms, (1, 0, 0, 0)).values)[4] == 1
    assert tuple(predict(river, rs, (0, 0, 1, 0), side="raw").values) == tuple(predict(money, ms, (1, 0, 0, 0), side="raw").values)


def test_adjacent_morphemes_keep_nonzero_pattern_connector_with_zero_byte_image():
    state, source, *_ = fixture(b"riverbank", ((0, 5), (5, 9)))
    assert not any(predict(state, source, (0, 1, 0, 0), side="raw").values)
    assert any(predict(state, source, (0, 1, 0, 0)).values)


@pytest.mark.parametrize("fault", ["missing_grade", "missing_face", "provenance", "literal", "counts", "target_owner", "archive", "float", "unknown", "carrier", "scope"])
def test_invalid_restored_contract_rejects_before_inference(fault):
    state, source, *_ = fixture()
    c, p = thaw_tree(state.configuration), thaw_tree(state.payload)
    if fault == "missing_grade": c["templates"] = tuple(t[:2] for t in c["templates"])
    elif fault == "missing_face":
        c["templates"] = ((c["templates"][0][0], c["templates"][0][1], ((),)), *c["templates"][1:])
    elif fault == "provenance": c["provenance"] = (c["provenance"][0], ((), *c["provenance"][1][1:]), c["provenance"][2])
    elif fault == "literal": c["scope"]["mention_bytes"] = b"loan"
    elif fault == "counts": p["counts"] = (-1, 0, 0)
    elif fault == "target_owner": c["pattern_source"]["reference"]["native"]["namespace"] += 1
    elif fault == "archive": c["raw_source"]["archive"][8] ^= 1
    elif fault == "float": c["templates"] = (((((0, 1.0),), *c["templates"][0][0][1:]), *c["templates"][0][1:]), *c["templates"][1:])
    elif fault == "unknown": c["unresolved"] = 3
    elif fault == "carrier": c["pattern_carriers"] = (10, 10, 12)
    else: c["scope"]["right_span"] = (0, 4)
    with pytest.raises((TypeError, ValueError), match="chain|correspondence|byte|boundary|exact|Fraction|uint8|integer|physical order"):
        changed = replace(state, configuration=c, payload=p)
        prepare_correspondence(changed, source)


@pytest.mark.parametrize("fault", ["float", "length", "bool_grade", "foreign", "integer_raw", "overflow"])
def test_invalid_inputs_and_integer_overflow_reject(fault):
    state, source, *_ = fixture(counts=(5, 3, 0))
    inputs = thaw_tree(cpp_correspondence_input(state, source, (0, 0, 1, 0)))
    if fault == "float": inputs["values"] = (0, 0, 1.0, 0)
    elif fault == "length": inputs["values"] = (0,)
    elif fault == "bool_grade": inputs["grade"] = True
    elif fault == "foreign": inputs["source"]["native"]["namespace"] += 1
    elif fault == "integer_raw": inputs.update(integer=True, side="raw")
    else: inputs.update(integer=True, values=(0, 0, (1 << 63)-1, 0))
    with pytest.raises((TypeError, ValueError, OverflowError)):
        infer_model(state, source, inputs)


def test_fixed_correspondence_profile_does_not_invent_a_training_callback():
    state, source, *_ = fixture()
    batch = ModelBatch(state.source, state.space, np.zeros(4, dtype=np.int64), np.ones(4, dtype=bool))
    with pytest.raises(NotImplementedError, match="does not support train"):
        train_model(state, source, batch)


def test_rcql_creation_inference_agent_and_system_share_owned_model_routes():
    from agent.hive import Hive
    from rcql import Executor, parse
    from system.serialize import json_value
    state, source, raw, pattern = fixture()
    config = {key: thaw_tree(state.configuration[key]) for key in
        ("carriers", "raw_carriers", "pattern_carriers", "scope", "templates", "labels", "unresolved", "grade", "limits", "max_bytes")}
    config.update(raw=raw, pattern=pattern, counts=state.payload["counts"])
    observed = cpp_correspondence_input(state, source, (0, 0, 1, 0), integer=True)
    record = model_record(source, state)
    created = Executor(sources={"record": record}, params={"config": config}).execute(parse(
        'FROM $record RETURN MODEL_INIT("cpp_correspondence",$config)')).values[0]
    assert created.adapter == "cpp_correspondence" and created.arithmetic == "rational"
    actual = Executor(sources={"record": record}, params={"model": state, "input": observed}).execute(parse(
        'FROM $record RETURN MODEL_INFER($model,$input)')).values[0]
    expected = infer_model(state, source, observed)
    assert actual.coefficient_digest == expected.coefficient_digest
    hive = Hive("context-correspondence")
    hive.add_worker("interpret", lambda value: infer_model(state, source, value), capability="predict")
    assert hive.invoke("interpret", observed, record=False).coefficient_digest == expected.coefficient_digest
    view = json_value(actual)
    assert view["arithmetic"] == "rational" and view["execution_receipt"]["denominator"] == 8
    assert json_value(created)["adapter"] == "cpp_correspondence"


@pytest.mark.parametrize("backend", ["sqlite", "file", "rex"])
def test_rcdb_copy_and_fresh_process_restore_preserve_all_three_sources(tmp_path, backend):
    from rcdb import copy_record, open_store
    state, source, *_ = fixture()
    observed = cpp_correspondence_input(state, source, (0, 0, 1, 0), integer=True)
    expected = infer_model(state, source, observed)
    record = model_record(source, state)
    record.attach_metadata(1, 0, "input", thaw_tree(observed))
    record.attach_metadata(1, 0, "prediction", expected)
    urls = [f"{backend}:///{tmp_path / name}" for name in ("first", "second")]
    stores = [open_store(url, **({} if backend == "sqlite" else {"read_only": False})) for url in urls]
    try:
        stores[0].put("context", record)
        receipt = copy_record(stores[0], stores[1], stores[0].get_record("context"), return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
    finally:
        for store in stores: store.close()
    code = """
import json, sys
from rcdb import open_store
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_correspondence import prepare_correspondence
store = open_store(sys.argv[1])
try:
    record = store.get_version('context', 1); state, source = read_model_record(record)
    output = infer_model(state, source, record.get_metadata(1,0,'input'))
    assert output.coefficient_digest == record.get_metadata(1,0,'prediction').coefficient_digest
    assert prepare_correspondence(state,source)[2] == b'river bank money'
    print(json.dumps(dict(model=state.coefficient_digest, output=output.coefficient_digest)))
finally: store.close()
"""
    observed = json.loads(subprocess.check_output([sys.executable, "-I", "-c", code, urls[1]], cwd=tmp_path, text=True))
    assert observed == {"model": state.coefficient_digest, "output": expected.coefficient_digest}


@pytest.mark.parametrize("fault", ["carrier", "target", "denominator", "integer", "grade", "dependencies"])
def test_portable_output_preserves_target_carrier_and_arithmetic(fault):
    state, source, *_ = fixture()
    output = predict(state, source, (0, 0, 1, 0), integer=True)
    r = thaw_tree(output.execution_receipt); changes = {}
    if fault == "carrier": r["target_carrier"] = 99
    elif fault == "target": r["target"]["native"]["namespace"] += 1
    elif fault == "denominator": r["denominator"] = 0
    elif fault == "integer": r["integer"] = 1
    elif fault == "grade": changes["grade"] = 9
    else: changes["dependencies"] = output.dependencies[:1]
    with pytest.raises(ValueError, match="correspondence|bounded"):
        replace(output, execution_receipt=r, **changes)


@pytest.mark.parametrize('fault', ['records', 'work', 'bits', 'missing', 'index', 'fractional_integer'])
def test_native_map_admission_and_complete_family(fault):
    state, source, _, pattern = fixture()
    family = state.configuration['templates'][0]
    limits, denominator = (1000000, 64000000, 16384), 0
    if fault == 'records': limits = (3, 64000000, 16384)
    elif fault == 'work': limits = (1000000, 1, 16384)
    elif fault == 'bits':
        family = tuple(tuple(tuple((i, q * (1 << 20)) for i, q in image) for image in row) for row in family)
        limits = (1000000, 64000000, 16)
    elif fault == 'missing': family = family[:2]
    elif fault == 'index': family = (((((999, Q(1)),), *family[0][1:])), *family[1:])
    else:
        family = tuple(tuple(tuple((i, q / 3) for i, q in image) for image in row) for row in family)
        denominator = 1
    with pytest.raises((ValueError, IndexError, RuntimeError), match='bound|grade|index|integral'):
        source.chain_map(pattern, family, limits=limits, denominator=denominator)


def test_integer_map_does_not_reinterpret_nonunit_lattice_scales():
    vertices = (10, 11)
    edge = relation(1, 100, vertices, (True, False), 0)
    for item in edge['occurrences']: item['magnitude'] = Q(1, 2)
    first = NativeSource.from_declaration(declaration(104, 1, (2, 1, 0), (edge,), vertices))
    second = NativeSource.from_declaration(declaration(105, 0, (2, 1, 0), (edge,), vertices))
    family = ((((0, Q(1)),), ((1, Q(1)),)), (((0, Q(1)),),), ())
    exact = first.chain_map(second, family)
    assert exact.push(first, 1, (Q(2, 3),)) == (Q(2, 3),)
    with pytest.raises(ValueError, match='unit retained lattice scales'):
        first.chain_map(second, family, denominator=1)


@pytest.mark.parametrize('grade', [0, 1, 2])
def test_fractional_template_coefficients_keep_one_complete_exact_denominator(grade, tmp_path):
    state, source, *_ = fixture(grade=grade)
    c = thaw_tree(state.configuration)
    scales = (Q(1,2), Q(1,3), Q(1,5))
    c['templates'] = tuple(tuple(tuple(tuple((i, q*s) for i, q in image) for image in row) for row in t)
                           for s,t in zip(scales,c['templates'],strict=True))
    model = replace(state, configuration=c)
    q = (1, 0, 0, 0) if grade < 2 else (1,)
    rational, integer = predict(model, source, q), predict(model, source, q, integer=True)
    assert tuple(rational.values) == tuple(integer.values)
    assert integer.execution_receipt['denominator'] == 16
    assert rational.values[0] == Q(5,16) and rational.values[len(q)] == Q(1,8)
    path = tmp_path / 'rational-templates.rcbd'
    save_model(path, model_record(source, model))
    restored, owner = read_model_record(load_model(path))
    assert predict(restored, owner, q, integer=True).coefficient_digest == integer.coefficient_digest


@pytest.mark.parametrize('grade', [0, 1, 2])
@pytest.mark.parametrize('side', ['raw', 'pattern'])
def test_tensor_field_and_rcql_field_use_actual_target_with_chart_provenance(grade, side):
    from rexgraph.field_codec import pack_field, unpack_field
    from rcql import Executor, parse
    state, source, raw, pattern = fixture(grade=grade)
    q = (1, 0, 0, 0) if grade < 2 else (1,)
    output = predict(state, source, q, side=side)
    field = output.tensor()
    target, index = (raw, 0) if side == 'raw' else (pattern, 1)
    assert field.source.matches(state.dependencies[index])
    assert field.dependencies[0].matches(state.source)
    assert tuple(field.values) == tuple(output.values)
    assert field.bind(target).source.source is target
    with pytest.raises(ValueError, match='declared publication'):
        field.bind(source)
    restored = unpack_field(pack_field(field))
    assert restored.coefficient_digest == field.coefficient_digest
    record = model_record(source, state)
    result = Executor(sources={'record':record}, params={'output':output}).execute(parse(
        'FROM $record RETURN MODEL_FIELD($output)')).values[0]
    assert result.coefficient_digest == field.coefficient_digest
    assert result.source.source is not None

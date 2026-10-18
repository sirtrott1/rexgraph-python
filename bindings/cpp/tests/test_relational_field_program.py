"""Complete native program fields, original flags and installed application routes."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import pytest

from rexgraph_cpp import NativeSource, PreparedFieldProgram, NativeFieldProgramResult, encode_stalk_column_tensor
from rexgraph.cpp_relational_field_program import (
    cpp_relational_field_program_model, field_program_query, read_field_program_result,
    field_program_stalks, clear_field_program_cache,
)
from rexgraph.model_codec import pack_model
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import ModelBatch, thaw_tree
from rexgraph.io.model_record import model_record, read_model_record, save_model, load_model
from rexgraph.value_codec import pack_value


def matrix(rows):
    return {"shape": (len(rows), len(rows[0]) if rows else 0),
            "entries": tuple((i, j, Q(value)) for i, row in enumerate(rows) for j, value in enumerate(row) if value)}


MAPS = (((1, 2), (0, 1)), ((1, 0), (0, 1), (1, 1)), ((1, 2), (0, 1)),
        ((7, 3), (2, 5), (1, 1)), ((1, 0), (0, 1)), ((2, 1), (1, 3)),
        ((0, 1), (1, 0), (2, -1)), ((-1, 1), (2, -2)))
COEFFICIENTS = (Q(1), Q(-1), Q(-1), Q(0), Q(0), Q(1), Q(1, 2), Q(1))
LOWER = (0, 1, 0, 1, 2, 0, 1, 2)
UPPER = (0, 0, 0, 0, 0, 1, 1, 1)
LATER = ((1, 0, 1, 0, 0, 0, 0), (0, 1, 0, 1, 0, 0, 0),
         (0, 0, 0, 0, 1, 1, 0), (1, 0, 0, 0, 0, 0, 1))


def source(namespace=992001):
    flags = ((10, 1, True), (20, -1, True), (10, -1, True), (20, 1, False),
             (30, 0, True), (10, 1, True), (20, Q(1, 2), True), (30, -1, True))
    relations = []
    for key, start, end in ((100, 0, 5), (101, 5, 8)):
        relations.append({"grade": 1, "key": key, "type": 0, "occurrences": tuple(
            {"key": i + 1, "lower": lower, "negative": coefficient < 0, "magnitude": abs(coefficient),
             "exists": active, "aspects": (0,)} for i, (lower, coefficient, active) in enumerate(flags[start:end], start))})
    return NativeSource.from_declaration({"profile": "declared-role-source-v1", "namespace": namespace,
        "semantic_scale": 3, "max_grade": 1, "role_aspects": ("boundary",),
        "types": ({"name": "relation", "min_arity": 0, "max_arity": 8},), "witnesses": (10, 20, 30),
        "relations": tuple(relations), "fields": (), "metrics": (),
        "carriers": ({"key": 10, "kind": "source_grade", "grade": 0, "coordinates": 0, "size": 3},
                     {"key": 11, "kind": "source_grade", "grade": 1, "coordinates": 0, "size": 2})})


def declaration():
    stalks = (((0, 0), 1001, (101, 102), ((2, 1), (1, 3))),
              ((0, 1), 1002, (201, 202, 203), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
              ((0, 2), 1001, (101, 102), ((1, 0), (0, 1))),
              ((1, 0), 1003, (301, 302), ((3, 1), (1, 2))),
              ((1, 1), 1003, (301, 302), ((1, 0), (0, 1))))
    return {"profile": "rexgraph.relational-field-program.v1", "key": 51, "revision": 2, "differential": True,
        "stalks": tuple({"cell": cell, "basis": basis, "directions": directions, "metric": matrix(metric)}
                        for cell, basis, directions, metric in stalks),
        "occurrences": tuple({"grade": 1, "slot": i,
            "role": "realized_incidence" if i == 7 else "coefficient_independent", "map": matrix(values)}
                              for i, values in enumerate(MAPS)),
        "nodes": ({"key": 1, "kind": "input", "grade": 1},
                  {"key": 2, "kind": "boundary", "grade": 0, "input": 1},
                  {"key": 3, "kind": "declared_map", "grade": 1, "input": 2, "map": matrix(LATER)},
                  {"key": 4, "kind": "boundary", "grade": 0, "input": 3})}


def tensor(values):
    return encode_stalk_column_tensor(len(values), tuple((i, Q(v)) for i, v in enumerate(values) if v))


def action(rows, values):
    return tuple(sum((Q(a) * b for a, b in zip(row, values, strict=True)), Q(0)) for row in rows)


def boundary(values):
    fields = [[Q(0)] * n for n in (2, 3, 2)]; occurrences = []
    for i, mapping in enumerate(MAPS):
        contribution = tuple(COEFFICIENTS[i] * v for v in action(mapping, values[2 * UPPER[i]:2 * UPPER[i] + 2]))
        occurrences.extend(contribution)
        for j, q in enumerate(contribution): fields[LOWER[i]][j] += q
    return tuple(v for row in fields for v in row), tuple(occurrences)


@pytest.fixture
def fixture():
    owner = source(); spec = declaration()
    return owner, spec, cpp_relational_field_program_model(owner, spec)


@pytest.mark.parametrize("trial", range(16))
def test_complete_native_intermediates_and_occurrences_match_fraction_reference(fixture, trial):
    owner, _, state = fixture
    values = (Q(trial - 4, 3), Q(trial % 7 - 2, 5), Q(trial % 11 - 3, 7), Q(trial % 13 - 6, 2))
    query = field_program_query(state, ({"node": 1, "use": 800 + trial, "field": tensor(values)},))
    output = infer_model(state, owner, query)
    result = read_field_program_result(output, state=state)
    lower, occurrence = boundary(values); next_input = action(LATER, lower); last, last_occurrence = boundary(next_input)
    for node, expected in ((1, values), (2, lower), (3, next_input), (4, last)):
        assert result.field_archive(node) == tensor(expected)
    assert result.field_archive(2, occurrences=True) == tensor(occurrence)
    assert result.field_archive(4, occurrences=True) == tensor(last_occurrence)
    assert result.fields()[1]["occurrence_dimension"] == 19
    selected = field_program_stalks(output, 2, ((0, 1), (0, 0)), coordinate_limit=5)
    assert selected[0]["values"] == lower[2:5] and selected[1]["values"] == lower[:2]
    assert selected[0]["basis"] == 1002 and selected[0]["directions"] == (201, 202, 203)
    assert output.values.size == 0 and output.field_archive["data"] == result.archive()
    cold = NativeFieldProgramResult.from_archive(result.archive())
    assert cold.field_archive(4) == result.field_archive(4)
    assert cold.input_uses()[0]["use"] == 800 + trial


def test_original_flags_basis_metrics_and_equal_value_uses_remain_distinct(fixture):
    owner, spec, state = fixture
    program = PreparedFieldProgram(owner.archive, spec)
    metadata = program.inspect()
    assert len(metadata["occurrences"]) == 8
    assert tuple(row["key"] for row in metadata["occurrences"]) == tuple(range(1, 9))
    assert metadata["occurrences"][3]["active"] is False
    assert metadata["occurrences"][4]["coefficient"] == 0
    assert metadata["stalks"][0]["metric"] != metadata["stalks"][2]["metric"]
    assert program.publication()["semantic_scale"] == 3
    assert PreparedFieldProgram.from_archive(program.archive()).archive() == program.archive()
    outputs = [infer_model(state, owner, field_program_query(state, ({"node": 1, "use": use, "field": tensor((1, 2, 3, 4))},)))
               for use in (91, 92)]
    assert read_field_program_result(outputs[0]).field_archive(4) == read_field_program_result(outputs[1]).field_archive(4)
    assert outputs[0].field_archive["data"] != outputs[1].field_archive["data"]


@pytest.mark.parametrize("kind", ["map", "role", "basis", "metric", "order", "node"])
def test_incomplete_or_undeclared_native_program_contracts_reject(kind):
    owner = source(); spec = deepcopy(declaration())
    if kind == "map": spec["occurrences"] = spec["occurrences"][:-1]
    elif kind == "role": spec["occurrences"][0]["role"] = "sheaf_restriction"
    elif kind == "basis": spec["stalks"][2]["directions"] = (102, 101)
    elif kind == "metric": spec["stalks"][0]["metric"] = matrix(((1, 2), (2, 1)))
    elif kind == "order": spec["nodes"] = tuple(reversed(spec["nodes"]))
    else: spec["nodes"][2]["kind"] = "learned_map"
    with pytest.raises((ValueError, RuntimeError)):
        PreparedFieldProgram(owner.archive, spec)


def test_owner_lifetime_bounds_and_concurrent_native_execution(fixture):
    owner, spec, _ = fixture
    program = PreparedFieldProgram(owner.archive, spec)
    inputs = ({"node": 1, "use": 99, "field": tensor((1, 2, 3, 4))},)
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(program.execute, [inputs]*8))
    del owner, program
    assert len({value.archive() for value in outputs}) == 1
    with pytest.raises((ValueError, RuntimeError)):
        outputs[0].select(2, ((1, 0),))
    with pytest.raises((ValueError, RuntimeError)):
        outputs[0].select(2, ((0, 0), (0, 0)))
    with pytest.raises(RuntimeError):
        outputs[0].select(2, ((0, 1),), coordinate_limit=2)
    with pytest.raises(RuntimeError):
        outputs[0].fields(max_records=1)
    with pytest.raises(RuntimeError):
        PreparedFieldProgram.__new__(PreparedFieldProgram).archive()
    with pytest.raises(RuntimeError):
        NativeFieldProgramResult.__new__(NativeFieldProgramResult).fields()


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_cold_model_source_result_and_standalone_field_replay(fixture, tmp_path, suffix):
    owner, _, state = fixture
    output = infer_model(state, owner, field_program_query(state, ({"node": 1, "use": 99, "field": tensor((1, 2, 3, 4))},)))
    record = model_record(owner, state); record.attach_metadata(1, 0, "response", output)
    path = tmp_path / ("program" + suffix); save_model(path, record)
    clear_field_program_cache()
    loaded = load_model(path); restored, source_ = read_model_record(loaded)
    response = loaded.get_metadata(1, 0, "response")
    assert restored.coefficient_digest == state.coefficient_digest and source_.archive == owner.archive
    assert read_field_program_result(response, state=restored).field_archive(4) == read_field_program_result(output).field_archive(4)
    packet = tmp_path / "response.value"; packet.write_bytes(pack_value(pack_model(output, native=True)))
    code = '''
from pathlib import Path
import sys
from rexgraph.value_codec import unpack_value
from rexgraph.model_codec import unpack_model
from rexgraph.cpp_relational_field_program import read_field_program_result
output=unpack_model(unpack_value(Path(sys.argv[1]).read_bytes()))
result=read_field_program_result(output)
assert result.input_uses()[0]['use']==99
assert len(result.fields())==4
assert result.select(2,[(0,1)],coordinate_limit=3)[0]['basis']==1002
'''
    subprocess.run([sys.executable, "-I", "-c", code, str(packet)], cwd=tmp_path, check=True)


@pytest.mark.parametrize('backend', ['file', 'rex', 'sqlite'])
def test_five_package_routes_and_explicit_unsupported_learning(fixture, tmp_path, backend):
    from rcql import Executor, parse
    from rcdb import open_store, copy_record
    from system.serialize import json_value
    from agent.hive import Hive
    owner, spec, state = fixture
    query = field_program_query(state, ({"node": 1, "use": 99, "field": tensor((1, 2, 3, 4))},))
    output = infer_model(state, owner, query)
    record = model_record(owner, state)
    parsed = parse('FROM $r RETURN MODEL_INIT("cpp_relational_field_program",$c),MODEL_INFER(MODEL_STATE(),$q),'
                   'MODEL_STALK($o,[0,1],node=2,max_coordinates=3),MODEL_VALUES($o),MODEL_INFO(MODEL_STATE())')
    values = Executor(sources={"r": record}, params={"c": {"declaration": spec}, "q": query, "o": output}).execute(parsed).values
    assert values[0].coefficient_digest == state.coefficient_digest
    assert values[1].field_archive["sha256"] == output.field_archive["sha256"]
    assert values[2]["basis"] == 1002 and len(values[2]["values"]) == 3
    assert len(values[3]["fields"]) == 4 and values[4]["trainable"] is False
    preview = json_value(output)
    assert preview["retains_whole_fields"] and "values" not in preview and "data" not in preview["field_archive"]
    def store(name):
        if backend == 'sqlite': return open_store('sqlite:///' + str(tmp_path / (name + '.sqlite')))
        return open_store(backend + '://' + str(tmp_path / name), read_only=False)
    stores = [store(name) for name in ("first", "second")]
    try:
        stored = stores[0].put("program", record)
        copy_record(stores[0], stores[1], stored)
        copied, actual = read_model_record(stores[1].get("program"))
        assert infer_model(copied, actual, query).field_archive["sha256"] == output.field_archive["sha256"]
    finally:
        for store in stores: store.close()
    hive = Hive()
    hive.add_worker("whole-fields", lambda value: infer_model(state, owner, value), capability="analyze")
    assert hive.dispatch_capability("analyze", query)["result"].field_archive["sha256"] == output.field_archive["sha256"]
    batch = ModelBatch(state.source, state.space, [0] * len(state.space.keys), [True] * len(state.space.keys))
    with pytest.raises(NotImplementedError, match="train"):
        train_model(state, owner, batch)
    with pytest.raises(TypeError, match="explicit native node"):
        output.tensor()


@pytest.mark.parametrize("operation", ['MODEL_STALK($o,[0,1],node=2)', 'MODEL_VALUES($o)'])
def test_rcql_program_inspection_checks_the_selected_source(fixture, operation):
    from rcql import Executor, parse
    owner, spec, state = fixture
    output = infer_model(state, owner, field_program_query(state, ({"node": 1, "use": 99, "field": tensor((1, 2, 3, 4))},)))
    other = source(992002)
    record = model_record(other, cpp_relational_field_program_model(other, spec))
    with pytest.raises(ValueError):
        Executor(sources={"r": record}, params={"o": output}).execute(parse('FROM $r RETURN ' + operation))


@pytest.mark.parametrize("kind", ['program', 'engine', 'bytes', 'source', 'receipt', 'model'])
def test_native_replay_rejects_false_executable_envelopes(fixture, kind):
    from hashlib import sha256
    from rexgraph.source_binding import CppSourceReference
    owner, _, state = fixture
    output = infer_model(state, owner, field_program_query(state, ({"node": 1, "use": 99, "field": tensor((1, 2, 3, 4))},)))
    archive = thaw_tree(output.field_archive); receipt = thaw_tree(output.execution_receipt)
    changes = {}
    if kind == 'program': archive['program_sha256'] = '0' * 64
    elif kind == 'engine': archive['engine_receipt']['engine_digest'] = '0' * 64
    elif kind == 'bytes':
        archive['data'] = archive['data'][:-1]
        archive['sha256'] = sha256(archive['data']).hexdigest()
    elif kind == 'source': changes['source'] = CppSourceReference(source(992002))
    elif kind == 'receipt': receipt['interpretation'] = 'identified-training-target'
    else: changes['model_digest'] = '0' * 64
    candidate = replace(output, field_archive=archive, execution_receipt=receipt, **changes)
    clear_field_program_cache()
    with pytest.raises((ValueError, RuntimeError)):
        read_field_program_result(candidate, state=state if kind == 'model' else None)


def test_rex_client_transfers_whole_program_records_with_existing_workspace_and_audit_contract(fixture, tmp_path, monkeypatch):
    from contextlib import closing
    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from rcdb import MemoryStore, record_packet
    from agent.client import RexClient
    from agent.rcdb import reset_default_store
    from agent.server import audit, auth
    from agent.server.routes.rex import router
    monkeypatch.setenv('REXGRAPH_CONFIG_DIR', str(tmp_path))
    monkeypatch.setenv('REXGRAPH_RCDB_URI', 'local://' + str(tmp_path / 'received'))
    monkeypatch.setenv('REXGRAPH_AUDIT_JOURNAL', str(tmp_path / 'audit.jsonl'))
    monkeypatch.setenv('REXGRAPH_FRAME_KEY', 'local-program-frame-key')
    monkeypatch.setattr(auth, '_CONFIG_DIR', tmp_path)
    reset_default_store(); auth.reset_auth_manager(); audit.reset_cache()
    app = FastAPI(); app.include_router(router)
    owner, _, state = fixture
    output = infer_model(state, owner, field_program_query(state, ({'node': 1, 'use': 99, 'field': tensor((1, 2, 3, 4))},)))
    record = model_record(owner, state); record.attach_metadata(1, 0, 'response', output)
    try:
        manager = auth.get_auth_manager(); manager.enable_auth()
        token = manager.create_token('program-reader', ['alpha'], role='user')
        other_token = manager.create_token('other-reader', ['beta'], role='user')
        with closing(MemoryStore()) as upstream:
            upstream.put('program', record, meta={'workspace': 'forged'})
            packet = record_packet(upstream, 'program')
        with TestClient(app) as transport:
            def strip(kw): return {k: v for k, v in kw.items() if k != 'timeout'}
            monkeypatch.setattr(httpx, 'stream', lambda method, url, **kw: transport.stream(method, url, **strip(kw)))
            monkeypatch.setattr(httpx, 'get', lambda url, **kw: transport.get(url, **strip(kw)))
            client = RexClient('http://testserver', api_key=token, workspace='alpha', frame_key='local-program-frame-key')
            receipt = client.rex_store_record(packet, source='program-source', courier='program-carrier')
            received = client.rex_fetch_record(receipt.destination_record_id, version=receipt.destination_version).snapshot()
            restored, actual = read_model_record(received.value)
            response = received.value.get_metadata(1, 0, 'response')
            assert restored.coefficient_digest == state.coefficient_digest and actual.archive == owner.archive
            assert read_field_program_result(response, state=restored).archive() == read_field_program_result(output).archive()
            assert received.record.meta['workspace'] == 'alpha' and received.record.meta['stored_by'] == 'program-reader'
            assert client.rex_audit()['integrity']['valid']
            other = RexClient('http://testserver', api_key=other_token, workspace='beta', frame_key='local-program-frame-key')
            with pytest.raises(httpx.HTTPStatusError) as denied:
                other.rex_fetch_record(receipt.destination_record_id)
            assert denied.value.response.status_code == 404
    finally:
        reset_default_store(); auth.reset_auth_manager(); audit.reset_cache()

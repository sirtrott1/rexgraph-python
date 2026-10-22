"""Direct Agent/RCDB/RCQL use and the same operations as scoped Hive tools."""
import base64
from fractions import Fraction as Q

import pytest
from agent.mcp_tools import Context, call, definitions

from agent import RelationalSession
from rcdb import MemoryStore, VersionConflictError, open_store
from rcql import Program, ProgramTransformation, Result
from rexgraph import RexGraph

DECLARATION = {"name": "Quote", "inputs": [
    {"name": "x", "kind": "ExactRational"}, {"name": "fee", "kind": "ExactRational"}],
    "steps": [{"name": "subtotal", "query": "FROM $data RETURN SUM([$x,$fee])"},
              {"name": "twice", "query": "FROM $data RETURN SUM([$subtotal,$subtotal])",
               "inputs": {"subtotal": {"step": "subtotal", "output": 0}}}]}
SOURCE = {"data": {"record_id": "data", "version": 1}}


@pytest.mark.parametrize("changes", [
    {"steps": [{"name": "bad", "query": "FROM $data RETURN $x", "inputs": []}]},
    {"steps": [{"name": "bad", "query": "FROM $data RETURN $x", "inputs": {"x": []}}]},
    {"inputs": {}},
    {"outputs": []},
])
def test_program_tool_schema_and_admission_reject_malformed_links_before_writes(changes):
    jsonschema = pytest.importorskip("jsonschema")
    from agent.mcp_tools import TOOLS

    declaration = {**DECLARATION, **changes}
    schema = TOOLS["rexgraph_rcql_publish_program"].definition()["input_schema"]
    jsonschema.validate({"record_id": "valid", "declaration": DECLARATION, "expected_version": 0}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"record_id": "bad", "declaration": declaration, "expected_version": 0}, schema)
    session = RelationalSession(MemoryStore())
    with pytest.raises(TypeError, match="require a"):
        session.publish_program("bad", declaration, expected_version=0)
    assert session.store.get_record("bad") is None
    assert session.records() == []


@pytest.fixture(params=["memory", "local", "sqlite"])
def session(request, tmp_path):
    if request.param == "memory": store = MemoryStore()
    elif request.param == "local": store = open_store("local://" + str(tmp_path / "records"))
    else: store = open_store("sqlite:///" + str(tmp_path / "records.sqlite"))
    value = RelationalSession(store)
    value.publish("data", RexGraph.from_cells([3, [[0, 1], [1, 2]]]), expected_version=0)
    yield value
    store.close()


def test_agent_query_and_saved_result_are_canonical(session):
    result = session.query('FROM RCDB_VERSION($db,"data",1) RETURN COUNT(CELLS(1)),SUM([1/3,2/7])', save_as="answer")
    assert result.values == (2, Q(13, 21))
    restored = session.read("answer").value
    assert restored.to_bytes() == result.to_bytes()
    assert session.read("answer").record.object_type == "RCQLResult"


def test_program_revision_retains_whole_steps_certificate_and_old_execution(session):
    session.publish_program("quote", DECLARATION, expected_version=0)
    first = session.run_program("quote", version=1, sources=SOURCE,
        parameters={"x": Q(3, 2), "fee": Q(1, 4)}, save_as="first")
    assert first.values == (Q(7, 2),)
    assert first.native_plan["step_results"][0]["values"] == (Q(7, 4),)
    restored = Result.from_bytes(first.to_bytes())
    assert restored.to_bytes() == first.to_bytes()
    assert restored.native_plan["program"].to_bytes() == Program.from_record(session.read("quote", 1).value).to_bytes()
    change = session.transform_program("quote", version=1, rule="specialize", arguments={"fee": Q(3, 4)},
        parameters={"x": Q(3, 2)}, sources=SOURCE)
    assert change["record"]["version"] == 2
    current = session.read("quote", 2).value
    certificate = ProgramTransformation.from_bytes(current.get_metadata(1, 0, "rcql_transformation"))
    certificate.verify(Program.from_record(session.read("quote", 1).value))
    assert session.run_program("quote", version=2, sources=SOURCE, parameters={"x": Q(3, 2)}).values == (Q(9, 2),)
    assert session.run_program("quote", version=1, sources=SOURCE,
        parameters={"x": Q(3, 2), "fee": Q(1, 4)}).values == first.values
    with pytest.raises(VersionConflictError):
        session.transform_program("quote", version=1, rule="specialize", arguments={"fee": Q(1, 2)},
            parameters={"x": Q(3, 2)}, sources=SOURCE)
    assert len(session.store.history("quote")) == 2


def test_mutations_are_explicit_guarded_and_never_saved_as_query_results(session):
    text = 'FROM $db MUTATE "copy" SET state=RCDB_GET("data"),expected_version=0 COMMIT'
    with pytest.raises(PermissionError): session.query(text)
    assert session.store.get_record("copy") is None
    with pytest.raises(ValueError, match="mutation receipts"):
        session.query(text, allow_write=True, save_as="bad")
    assert session.store.get_record("copy") is None
    with pytest.raises(ValueError, match="expected_version"):
        session.query('FROM $db MUTATE "copy" SET state=RCDB_GET("data") COMMIT', allow_write=True)
    result = session.query(text, allow_write=True)
    assert result.values[0].version == 1
    assert session.store.verify_commits("copy")


def test_source_bindings_and_parameters_do_not_open_ambient_resources(session):
    with pytest.raises(ValueError, match="reserved"):
        session.query("FROM $db RETURN RCDB_LIST()", sources={"db": SOURCE["data"]})
    with pytest.raises(KeyError): session.query('FROM REX("/tmp/unbound") RETURN BETTI(0)')
    with pytest.raises(ValueError, match="both"):
        session.query("FROM $data RETURN $x", sources=SOURCE, parameters={"x": 1},
            record_parameters={"x": {"record_id": "data"}})


def test_tools_and_hive_share_real_query_program_and_record_routes(tmp_path, monkeypatch):
    from agent.hive import Hive

    from agent import rcdb
    store = MemoryStore()
    monkeypatch.setattr(rcdb, "default_store", lambda: store)
    ctx = Context(workspace="alpha", identity="alice", is_admin=False, auth_enabled=True)
    hive = Hive("relational")
    names = [d["name"] for d in definitions(ctx) if d["name"].startswith(("rexgraph_rcdb_", "rexgraph_rcql_"))]
    assert len(hive.add_tools(context=ctx, names=names)) == 7
    hive.invoke("rexgraph_rcdb_publish", {"record_id": "data", "cells": [3, [[0, 1], [1, 2]]], "expected_version": 0})
    hive.invoke("rexgraph_rcql_publish_program", {"record_id": "quote", "declaration": DECLARATION, "expected_version": 0})
    out = hive.invoke("rexgraph_rcql_run_program", {"record_id": "quote", "version": 1, "sources": SOURCE,
        "parameters": {"x": {"numerator": 3, "denominator": 2}, "fee": {"numerator": 1, "denominator": 4}}, "save_as": "answer"})
    native = Result.from_bytes(base64.b64decode(out["result_base64"]))
    assert native.values == (Q(7, 2),)
    assert store.get_record("answer").meta["workspace"] == "alpha"
    assert store.commit_history("data")[-1].transition.actor == "alice"
    assert {row["id"] for row in hive.invoke("rexgraph_rcdb_list", {})["records"]} == {"data", "quote", "answer"}
    changed = hive.invoke("rexgraph_rcql_transform_program", {"record_id": "quote", "version": 1,
        "rule": "input", "arguments": ["x", "amount"], "sources": SOURCE,
        "parameters": {"amount": {"numerator": 3, "denominator": 2}, "fee": {"numerator": 1, "denominator": 4}}})
    assert changed["record"]["version"] == 2


def test_scoped_tools_hide_payloads_and_prevent_ownership_or_actor_spoofing(monkeypatch):
    from agent import rcdb
    store = MemoryStore()
    monkeypatch.setattr(rcdb, "default_store", lambda: store)
    a = Context(workspace="alpha", identity="alice", auth_enabled=True)
    b = Context(workspace="beta", identity="bob", auth_enabled=True)
    call("rexgraph_rcdb_publish", context=a, record_id="secret", cells=[2, [[0, 1]]], expected_version=0)
    assert call("rexgraph_rcdb_list", context=b)["records"] == []
    with pytest.raises(KeyError): call("rexgraph_rcdb_inspect", context=b, record_id="secret", history=True)
    with pytest.raises(KeyError):
        call("rexgraph_rcql_query", context=b, query='FROM RCDB_VERSION($db,"secret",1) RETURN BETTI(0)')
    with pytest.raises(PermissionError):
        call("rexgraph_rcdb_publish", context=b, record_id="secret", cells=[2, [[0, 1]]], expected_version=1)
    call("rexgraph_rcdb_publish", context=b, record_id="own", cells=[2, [[0, 1]]], expected_version=0,
        meta={"workspace": "alpha", "stored_by": "alice"})
    assert store.get_record("own").meta == {"workspace": "beta", "stored_by": "bob"}
    out = call("rexgraph_rcql_query", context=b, allow_write=True,
        query='FROM $db MUTATE "copy" SET state=RCDB_GET("own"),actor="alice",expected_version=0 COMMIT')
    assert out["kind"] == "mutation"
    assert store.commit_history("copy")[-1].transition.actor == "bob"


def test_registered_name_objects_can_be_used_directly_as_parameters(session):
    from rcql.ast import Call, ListExpr, Parameter

    from rcql import NameRelation
    step = NameRelation("Double", ("x",), Call("SUM", (ListExpr((Parameter("x"), Parameter("x"))),)))
    session.publish("method", step.to_record(), expected_version=0)
    result = session.query('FROM $data RETURN NAME_APPLY($method,[3/2])', sources=SOURCE,
        record_parameters={"method": {"record_id": "method", "version": 1, "kind": "name"}})
    assert result.values == (Q(3),)


def test_native_model_whole_output_survives_agent_program_and_result_storage():
    binding = pytest.importorskip("rexgraph_cpp")
    from rexgraph.cpp_model import cpp_model, cpp_rational_batch_input
    from rexgraph.io.model_record import model_record
    from rexgraph.model_runtime import infer_model
    from rexgraph.model_state import thaw_tree
    from rexgraph.type_accession import CoordinateSpace
    source = binding.NativeSource.from_declaration({
        "profile": "declared-exact-source-v1", "namespace": 91, "semantic_scale": 0, "max_grade": 1,
        "types": ({"name": "agent-fixture", "min_arity": 0, "max_arity": 2},), "witnesses": (),
        "relations": [{"grade": 1, "key": 100, "type": 0, "occurrences": [
            {"key": 0, "lower": 10, "exists": True, "negative": False, "magnitude": Q(1)},
            {"key": 1, "lower": 11, "exists": True, "negative": True, "magnitude": Q(1)}]}],
        "carriers": [{"key": 10, "kind": "source_grade", "grade": 0, "coordinates": 0, "size": 2},
                     {"key": 11, "kind": "source_grade", "grade": 1, "coordinates": 0, "size": 1}],
        "fields": [{"key": 1001, "carrier": 11, "entries": ((0, Q(1, 3)),), "support": None}], "metrics": []})
    state = cpp_model(source, (10, 11))
    inputs = cpp_rational_batch_input(state, source, (1001,), CoordinateSpace("samples", ("one",)))
    expected = infer_model(state, source, inputs)
    session = RelationalSession(MemoryStore())
    session.publish("model", model_record(source, state), expected_version=0)
    session.publish("inputs", thaw_tree(inputs), expected_version=0)
    session.publish_program("infer", {"name": "NativeInference", "inputs": [{"name": "inputs"}],
        "steps": [{"name": "infer", "query": "FROM $model RETURN MODEL_INFER(MODEL_STATE(),$inputs)"}]}, expected_version=0)
    result = session.run_program("infer", sources={"model": {"record_id": "model", "version": 1}},
        record_parameters={"inputs": {"record_id": "inputs", "version": 1}}, save_as="whole-output")
    assert result.values[0].coefficient_digest == expected.coefficient_digest
    assert result.values[0].axes == expected.axes and result.values[0].space == expected.space
    restored = session.read("whole-output").value
    assert restored.to_bytes() == result.to_bytes()
    assert restored.native_plan["step_results"][0]["values"][0].coefficient_digest == expected.coefficient_digest


def test_cold_default_agent_store_reopens_programs_and_complete_results(tmp_path):
    import json
    import os
    import subprocess
    import sys
    uri = "local://" + str(tmp_path / "records")
    store = open_store(uri)
    session = RelationalSession(store)
    session.publish("data", RexGraph.from_cells([3, [[0, 1], [1, 2]]]), expected_version=0)
    session.publish_program("quote", DECLARATION, expected_version=0)
    first = session.run_program("quote", version=1, sources=SOURCE,
        parameters={"x": Q(3, 2), "fee": Q(1, 4)}, save_as="answer")
    store.close()
    script = '''
import json
from fractions import Fraction as Q
from agent import RelationalSession
s=RelationalSession()
r=s.run_program("quote",version=1,sources={"data":{"record_id":"data","version":1}},parameters={"x":Q(3,2),"fee":Q(1,4)})
saved=s.read("answer").value
assert saved.values==r.values
print(json.dumps({"value":str(r.values[0]),"saved_bytes":saved.to_bytes().hex(),"steps":len(saved.native_plan["step_results"])}))
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
        env=dict(os.environ, REXGRAPH_RCDB_URI=uri, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    reply = json.loads(result.stdout)
    assert reply["value"] == "7/2" and reply["steps"] == 2
    assert bytes.fromhex(reply["saved_bytes"]) == first.to_bytes()


def test_tool_agents_create_objects_reuse_query_outputs_and_iterate_without_host_code(monkeypatch):
    from agent import rcdb
    store = MemoryStore()
    monkeypatch.setattr(rcdb, "default_store", lambda: store)
    call("rexgraph_rcdb_publish", record_id="data", cells=[2, [[0, 1]]], expected_version=0)
    call("rexgraph_rcql_query", sources=SOURCE, save_as="factory",
        query='FROM $data LET object=NAME_BIND(NAME("SUM"),"values",[1,2]) RETURN NAME_RECORD(object)')
    published = call("rexgraph_rcdb_publish", record_id="object", expected_version=0,
        value_record={"record_id": "factory", "version": 1, "output": 0})
    assert published["record"]["version"] == 1
    assert store.get_record("object").meta["agent_parent_record"]["record_id"] == "factory"
    result = call("rexgraph_rcql_query", sources=SOURCE, query='FROM $data RETURN NAME_APPLY($object,[])',
        record_parameters={"object": {"record_id": "object", "version": 1, "kind": "name"}})
    assert result["values"] == [3]
    call("rexgraph_rcql_query", sources=SOURCE, save_as="operation",
        query='FROM $data RETURN NAME_BIND(NAME("NAME_ALIAS"),"name","revised")')
    repeated = call("rexgraph_rcql_query", sources=SOURCE, save_as="iterated",
        query='FROM $data LET result=NAME_ITERATE($operation,$object,3,"operation") '
              'LET final=RECURSIVE_VALUE(result) RETURN NAME_RECORD(final),NAME_APPLY(final,[]),RECURSIVE_RESULT_RECORD(result)',
        record_parameters={"operation": {"record_id": "operation", "output": 0},
            "object": {"record_id": "object", "kind": "name"}})
    assert repeated["values"][1] == 3
    assert Result.from_bytes(base64.b64decode(repeated["result_base64"])).native_plan["agent_record_parameters"]["operation"]["output"] == 0
    with pytest.raises(ValueError, match="existing canonical Result"):
        call("rexgraph_rcdb_publish", record_id="bad", expected_version=0,
            value_record={"record_id": "data", "output": 0})
    assert store.get_record("bad") is None


@pytest.mark.parametrize("order", ["runtime-first", "module-first", "session-first"])
def test_root_runtime_shortcut_survives_all_import_orders(order, tmp_path):
    import os
    import subprocess
    import sys
    script = '''
import sys
if sys.argv[1]=="runtime-first":
 from agent import rcql_runtime
elif sys.argv[1]=="module-first":
 import agent.rcql_runtime
 from agent import rcql_runtime
else:
 from agent import RelationalSession
 from rcdb import MemoryStore
 RelationalSession(MemoryStore()).query("FROM $db RETURN RCDB_LIST()")
 from agent import rcql_runtime
from rcql import parse,SourcePolicy
from rexgraph import RexGraph
rcql_runtime.register("example",RexGraph.from_cells([2,[[0,1]]]),policy=SourcePolicy.allow("read"))
assert rcql_runtime.execute(parse("FROM $example RETURN COUNT(CELLS(1))")).values==(1,)
rcql_runtime.remove("example")
assert rcql_runtime.sources()==()
'''
    result = subprocess.run([sys.executable, "-c", script, order], cwd=tmp_path,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

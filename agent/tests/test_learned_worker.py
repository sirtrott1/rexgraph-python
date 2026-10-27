"""Whole-field learning selects and executes a real versioned RCQL Hive worker."""
from contextlib import closing
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    pytest.importorskip("rexgraph_cpp._assignment")
    helper = Path(__file__).resolve().parents[2] / "rexgraph/tests/test_assignment_provider.py"
    spec = importlib.util.spec_from_file_location("learned_worker_fixture", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, _, case, source, state = module.fixture()
    from agent import LearnedWorker, RelationalSession
    from agent.hive import Hive
    from rcdb import open_store
    from rcql import Result
    from rexgraph.io.model_record import model_record
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", "off")
    uri = "local://" + str(tmp_path / "records")
    store = open_store(uri)
    session = RelationalSession(store)
    session.publish("model", model_record(source, state), expected_version=0)
    for i in range(2):
        rows = module.rows(case, case.model(), i)
        session.save_result("observation-" + str(i), Result((rows,), aliases=("observations",)))
        session.save_result("input-" + str(i), Result((case.read(case.data["held"][i]["input"]),), aliases=("input",)))
        session.save_result("goal-" + str(i), Result((case.read("held-goal-" + str(i + 2) + ".bin"),), aliases=("goal",)))
    current = [case.context()]
    hive = Hive("learned-worker-lifecycle")
    worker = LearnedWorker(session, hive, "selection", authority=lambda: current[0])
    root = worker.records.pin("model", 1)
    worker.activate(root, expected_version=0)
    yield worker, session, hive, case, current, uri, root, module
    worker.close()
    hive.stop_all()
    store.close()


def learn(setup):
    worker, session, _, _, _, _, _, _ = setup
    for i in range(2):
        worker.learn(worker.records.pin("observation-" + str(i)), model_id="model",
            expected_model_version=i + 1, expected_selection_version=i + 1)
    assert session.read("selection").record.version == 3
    return worker.decide("decision")


def dispatch(setup, decision, ordinal=0, record_id="execution"):
    worker = setup[0]
    return worker.dispatch(decision, worker.records.pin("input-" + str(ordinal)),
        worker.records.pin("goal-" + str(ordinal)), record_id=record_id)


def test_native_learning_versions_actual_program_and_complete_held_fields(setup):
    from rexgraph.cpp_assignment import read_assignment_result
    from rexgraph.io.model_record import read_model_record
    worker, session, hive, case, current, _, root, _ = setup
    weak = worker.decide("weak")
    assert worker.assess_decision(weak).named_values["accepted"] is None
    assert not hive._bees
    with pytest.raises(ValueError, match="unresolved"):
        dispatch(setup, weak)
    decision = learn(setup)
    assert worker.assess_decision(decision).named_values["accepted"] is True
    chosen = session.read("decision").value.named_values["decision"]
    assert chosen["binding"]["worker"] == 11
    assert len(hive._bees) == 1
    before, _ = read_model_record(session.read("model", 1).value, execution_context=current[0])
    after, _ = read_model_record(session.read("model", 3).value, execution_context=current[0])
    assert before.space == after.space and before.source.matches(after.source)
    assert before.output_axes == after.output_axes
    assert after.step == 2 and before.step == 0
    fields = []
    for ordinal in range(2):
        receipt = dispatch(setup, decision, ordinal, "execution-" + str(ordinal))
        observed = worker.assess_execution(receipt).named_values
        assert observed["accepted"] is True and observed["status"] == "satisfied"
        packet = session.read(receipt["record_id"]).value.named_values["execution"]
        assert packet["worker_invoked"] is True and packet["elapsed_ns"] > 0
        result = session.read(packet["output"]["record_id"]).value
        output = result.named_values["response"]
        assert output.execution_receipt["executed_external_worker"] is False
        native = read_assignment_result(output, execution_context=current[0], state=after)
        assert native.archive(current[0]) == case.read(case.data["held"][ordinal]["response"])
        field = native.field_archive(3, current[0])
        assert field["free_parameter_count"] == 1
        fields.append(field["particular"])
        worker.publish(receipt, "answer-" + str(ordinal))
        queried = session.query('FROM $db RETURN RCDB_GET("answer-' + str(ordinal) + '")')
        assert queried.values[0].named_values["answer"]["output"] == packet["output"]
    assert fields[0] != fields[1]


def test_weak_strong_withdrawal_reroot_preserves_the_native_root(setup):
    worker, session, hive, case, current, _, root, _ = setup
    strong = learn(setup)
    executed = dispatch(setup, strong)
    root_bytes = root["state_digest"]
    current[0] = case.context(remove=(case.data["observations"][1]["origin"],))
    assert worker.assess_decision(strong).named_values["accepted"] is None
    assert worker.assess_execution(executed).named_values["accepted"] is None
    with pytest.raises(ValueError):
        worker.restore()
    # An explicit selection can still admit the historical root; no child is
    # silently treated as though its withdrawn measurement remained available.
    worker.activate(root, expected_version=3)
    weak = worker.decide("weak-again")
    assert worker.assess_decision(weak).named_values["accepted"] is None
    assert not hive._bees
    assert worker.records.pin("model", 1)["state_digest"] == root_bytes
    current[0] = case.context()
    worker.activate(worker.records.pin("model", 3), expected_version=4)
    again = worker.decide("strong-again")
    assert worker.assess_decision(again).named_values["accepted"] is True
    assert worker.assess_decision(strong).named_values["accepted"] is None


@pytest.mark.parametrize("changed", ["input", "goal", "program", "decision", "output", "execution", "selection"])
def test_current_dependencies_are_rechecked_before_answer_publication(setup, changed):
    worker, session, _, _, _, _, root, _ = setup
    selected = learn(setup)
    result = dispatch(setup, selected)
    receipt = session.read(result["record_id"]).value.named_values["execution"]
    if changed == "selection":
        worker.activate(root, expected_version=3)
    else:
        ref = {"input": receipt["input"], "goal": receipt["goal"],
               "decision": selected, "program": session.read("decision").value.named_values["decision"]["program"],
               "output": receipt["output"], "execution": result}[changed]
        snapshot = session.read(ref["record_id"], ref["version"])
        session.publish(ref["record_id"], snapshot.value, expected_version=ref["version"])
    assert worker.assess_execution(result).named_values["accepted"] is None
    with pytest.raises(ValueError, match="cannot publish"):
        worker.publish(result, "answer")
    assert session.store.get_record("answer") is None


def test_replaced_worker_and_caller_owned_authority_cannot_execute(setup):
    worker, _, hive, _, _, _, _, _ = setup
    selected = learn(setup)
    bee = worker._runtime["bee"]
    with pytest.raises(ValueError, match="registration-owned"):
        hive.invoke(bee.name, {"authority": None})
    with pytest.raises(ValueError, match="registration-owned"):
        hive.invoke(bee.name, {"owner": None})
    bee._handler = lambda _: True
    assert worker.assess_decision(selected).named_values["accepted"] is None
    with pytest.raises(ValueError, match="unresolved"):
        dispatch(setup, selected)


def test_wrong_input_goal_or_whole_response_cannot_publish(setup, monkeypatch):
    worker, session, hive, _, _, _, _, _ = setup
    selected = learn(setup)
    # A valid response to another input is not authority for this input.
    original = hive.invoke
    other = original(worker._runtime["bee"].name,
        {"input": session.read("input-1").value.named_values["input"]})
    monkeypatch.setattr(hive, "invoke", lambda *a, **k: other)
    with pytest.raises(ValueError, match="original input differs"):
        dispatch(setup, selected)
    assert session.store.get_record("execution") is None
    assert session.store.get_record("execution/output/1") is None


def test_whole_model_bytes_are_not_reloaded_by_current_decision_guard(setup, monkeypatch):
    worker, session, _, _, _, _, _, _ = setup
    selected = learn(setup)
    monkeypatch.setattr(session, "read", lambda *a, **k: pytest.fail("guard decoded stored field payload"))
    assert worker.assess_decision(selected).named_values["accepted"] is True


def test_released_and_cold_stored_decisions_need_fresh_execution(setup):
    from agent import LearnedWorker, RelationalSession
    from agent.hive import Hive
    from rcdb import open_store
    worker, session, _, _, current, uri, _, _ = setup
    selected = learn(setup)
    receipt = dispatch(setup, selected)
    worker.publish(receipt, "answer")
    worker.release()
    assert worker.assess_decision(selected).named_values["status"] == "unissued"
    assert worker.assess_execution(receipt).named_values["status"] == "unissued"
    with closing(open_store(uri)) as store:
        hive = Hive("cold-learned-worker")
        cold = LearnedWorker(RelationalSession(store), hive, "selection", authority=lambda: current[0])
        try:
            cold.restore()
            assert cold.assess_execution(receipt).named_values["accepted"] is None
            fresh = cold.decide("cold-decision")
            result = cold.dispatch(fresh, cold.records.pin("input-0"), cold.records.pin("goal-0"), record_id="cold-execution")
            assert cold.assess_execution(result).named_values["accepted"] is True
            first = session.read("execution/output/1").value.to_bytes()
            assert session.read("cold-execution/output/1").value.to_bytes() == first
        finally:
            cold.close()
            hive.stop_all()


def test_hive_workflow_passes_selected_record_refs_without_copying_whole_fields(setup):
    from agent.hive_workflow import WorkflowCheck as Check, WorkflowRef as Ref, WorkflowStep as Step
    worker, session, hive, _, _, _, _, _ = setup
    selected = learn(setup)
    hive.add_worker("select", lambda _: selected, capability="analyze")
    hive.add_worker("eligible", worker.assess_decision, capability="analyze")
    hive.add_worker("execute", lambda pin: dispatch(setup, pin), capability="transform")
    hive.add_worker("verify", worker.assess_execution, capability="analyze")
    hive.add_worker("publish", lambda pin: worker.publish(pin, "answer"), capability="transform")
    result = hive.run_workflow((Step("select", "select", checks=(Check("eligible", Ref("select")),)),
        Step("execute", "execute", Ref("select"), checks=(Check("verify", Ref("execute")),)),
        Step("publish", "publish", Ref("execute"), guards=(Check("verify", Ref("execute")),))))
    assert result.ok, result.steps
    packet = result.to_result().to_bytes()
    output = session.read("execution/output/1").value.to_bytes()
    assert output not in packet and len(packet) < 15000


def test_future_observation_and_duplicate_learning_leave_active_parent_unchanged(setup):
    worker, session, _, case, current, _, _, _ = setup
    before = worker.records.pin("model", 1)
    selected = worker.records.pin("selection", 1)
    origin = case.data["observations"][0]["origin"]
    current[0] = case.context(limits={origin: 0})
    with pytest.raises(ValueError):
        worker.learn(worker.records.pin("observation-0"), model_id="model", expected_model_version=1,
                     expected_selection_version=1)
    assert worker.records.pin("model") == before and worker.records.pin("selection") == selected
    current[0] = case.context()
    worker.learn(worker.records.pin("observation-0"), model_id="model", expected_model_version=1,
                 expected_selection_version=1)
    parent = worker.records.pin("model")
    with pytest.raises(ValueError):
        worker.learn(worker.records.pin("observation-0"), model_id="model", expected_model_version=2,
                     expected_selection_version=2)
    assert worker.records.pin("model") == parent


def test_separate_process_restores_program_executes_and_checks_whole_response(setup):
    import json
    from hashlib import sha256
    worker, session, _, _, _, uri, _, _ = setup
    selected = learn(setup)
    dispatch(setup, selected)
    original = session.read("execution/output/1").value.to_bytes()
    code = '''
import sys,json,importlib.util
from pathlib import Path
from contextlib import closing
from hashlib import sha256
from agent import LearnedWorker,RelationalSession
from agent.hive import Hive
from rcdb import open_store
spec=importlib.util.spec_from_file_location('native_assignment_fixture',Path(sys.argv[2]))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
case=module.Fixture()
with closing(open_store(sys.argv[1])) as store:
    session=RelationalSession(store);hive=Hive('process-worker')
    worker=LearnedWorker(session,hive,'selection',authority=case.context)
    try:
        old=worker.records.pin('execution')
        assert worker.assess_execution(old).named_values['status']=='unissued'
        worker.restore()
        decision=worker.decide('process-decision')
        receipt=worker.dispatch(decision,worker.records.pin('input-0'),worker.records.pin('goal-0'),record_id='process-execution')
        assert worker.assess_execution(receipt).named_values['accepted'] is True
        worker.publish(receipt,'process-answer')
        output=session.read('process-execution/output/1').value.to_bytes()
        print(json.dumps({'output_sha256':sha256(output).hexdigest(),'worker_invoked':True}))
    finally:
        worker.close();hive.stop_all()
'''
    helper = Path(__file__).resolve().parents[2] / "bindings/cpp/tests/test_assignment_evidence.py"
    run = subprocess.run([sys.executable, "-I", "-B", "-c", code, uri, str(helper)],
        cwd=Path(uri.removeprefix("local://")).parent, env=dict(os.environ), text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    data = json.loads(run.stdout)
    assert data["worker_invoked"] and data["output_sha256"] == sha256(original).hexdigest()


def test_caller_mutation_cannot_relabel_issued_input_or_goal_references(setup):
    worker, session, _, _, _, _, _, _ = setup
    selected = learn(setup)
    original_input = worker.records.pin("input-0")
    original_goal = worker.records.pin("goal-0")
    receipt = worker.dispatch(selected, original_input, original_goal, record_id="execution")
    original_input.update(worker.records.pin("input-1"))
    original_goal.update(worker.records.pin("goal-1"))
    value = session.read("input-0").value
    session.save_result("input-0", value, expected_version=1)
    assert worker.assess_execution(receipt).named_values["accepted"] is None

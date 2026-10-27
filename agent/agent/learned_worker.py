"""Measured native assignment learning and versioned RCQL workers in a Hive."""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter_ns

from .record_access import PinnedRecords, record_pin_key, copy_record_pin
from .relational import RelationalSession, _integer


def _result(value, alias):
    from rcql import Result
    return Result((value,), aliases=(alias,))


class LearnedWorker:
    """Learn over complete returned fields, then execute the identified program.

    Assignment candidates and conditional parameter families belong to the
    native model. Python publishes immutable model/program revisions and
    orchestrates existing Hive capabilities. Explicit activation may select a
    historical model; withdrawal never silently substitutes an older model.
    Elapsed time is recorded as an observation, not a worker scoring rule.
    """

    def __init__(self, session, hive, record_id, *, authority):
        from .hive import Hive
        if not isinstance(session, RelationalSession) or not isinstance(hive, Hive):
            raise TypeError("learned workers require an existing RelationalSession and Hive")
        if not callable(authority) or not isinstance(record_id, str) or not record_id:
            raise TypeError("provide a selection record identity and current authority resolver")
        self.session, self.hive, self.record_id = session, hive, record_id
        self._authority = authority
        self.records = PinnedRecords(session)
        self._runtime = None
        self._decisions = {}
        self._executions = {}

    def _model(self, pin):
        from rexgraph.io.model_record import read_model_record
        from rexgraph.cpp_assignment import prepare_assignment_model
        context = self._authority()
        state, source = read_model_record(self.records.read(pin), execution_context=context)
        native = prepare_assignment_model(state, source, execution_context=context)
        info = native.inspect(context, max_records=state.configuration["limits"][1])
        return state, source, native, info

    @staticmethod
    def _description(model, state, info):
        selected = info["identified_candidate"]
        status = info["status"]
        if state.step == 0 or info["remainder_count"]:
            selected = None
            status = "untrained" if state.step == 0 else "unresolved"
        return dict(profile="agent.learned-worker.v1", model=model, model_digest=state.coefficient_digest,
                    status=status, candidate=selected, program=None, binding=None)

    def activate(self, model, *, expected_version):
        """Publish an explicit temporal selection and its learned program child.

        A historical model pin is intentional. The active selection record, its
        program, and future dispatch inputs must remain current. Program/model
        publications are independent records, not one multi-record transaction.
        """
        from rcql.assignment import prepare_assignment_program
        _integer(expected_version, "expected_version")
        model = copy_record_pin(model)
        state, source, native, info = self._model(model)
        description = self._description(model, state, info)
        if description["candidate"] is not None:
            prepared = prepare_assignment_program(state, source, execution_context=self._authority())
            record = prepared.program.to_record()
            record.attach_metadata(1, 0, "rcql_transformation", prepared.transformation.to_bytes().decode("utf8"))
            record._agent_meta = {"rcql_evidence": [
                {key: model[key] for key in ("record_id", "version", "state_digest")} ]}
            program_id = self.record_id + "/program/" + state.coefficient_digest
            existing = self.session.store.get_record(program_id)
            if existing is None:
                self.session.publish(program_id, record, expected_version=0)
            else:
                from rcql import Program
                stored = self.session.read(program_id, 1).value
                if (existing.version != 1 or Program.from_record(stored).to_bytes() != prepared.program.to_bytes()
                        or stored.get_metadata(1, 0, "rcql_transformation") != prepared.transformation.to_bytes().decode("utf8")):
                    raise ValueError("learned program record differs from its immutable specialization")
            from rexgraph.model_state import thaw_tree
            description.update(program=self.records.pin(program_id, 1), binding=thaw_tree(prepared.binding))
        self.session.save_result(self.record_id, _result(description, "selection"), expected_version=expected_version)
        pin = self.records.pin(self.record_id)
        self.restore()
        return pin

    def restore(self):
        """Re-admit the selected native model and register its actual RCQL worker."""
        from rcql import Program
        from rcql.assignment import restore_assignment_program
        pin = self.records.pin(self.record_id)
        if self._runtime is not None and self._runtime["selection"] == pin:
            self._check_runtime(self._runtime)
            return pin
        value = self.records.value(pin, "selection", current=True)
        if not isinstance(value, dict) or set(value) != {
                "profile", "model", "model_digest", "status", "candidate", "program", "binding"}:
            raise ValueError("unsupported learned worker selection")
        state, source, native, info = self._model(value["model"])
        expected = self._description(value["model"], state, info)
        for name in ("profile", "model_digest", "status", "candidate"):
            if value[name] != expected[name]:
                raise ValueError("selected native assignment differs from its published description")
        runtime = dict(selection=pin, description=value, state=state, source=source, owner=native, bee=None)
        if value["candidate"] is not None:
            self.records.current(value["program"])
            program = Program.from_record(self.records.read(value["program"]))
            prepared = restore_assignment_program(program, source, execution_context=self._authority())
            from rexgraph.model_state import thaw_tree
            if (prepared.state.coefficient_digest != state.coefficient_digest
                    or thaw_tree(prepared.binding) != value["binding"] or prepared.candidate != value["candidate"]):
                raise ValueError("worker program uses another model, candidate or whole field interface")
            name = "assignment/" + sha256((self.session.store.store_id + self.record_id).encode()).hexdigest()[:16]
            name += "/" + str(pin["version"])
            bee = self.hive.add_program(name, self.session, value["program"]["record_id"],
                version=value["program"]["version"],
                sources={"r": {k: value["model"][k] for k in ("record_id", "version")}},
                runtime_parameters=prepared.runtime_parameters(self._authority))
            runtime.update(bee=bee, owner=prepared.owner,
                           registration=(bee._handler, bee._context, bee.capability, bee.worker_type, bee.model))
        elif value["program"] is not None or value["binding"] is not None:
            raise ValueError("unresolved selection cannot carry an executable worker")
        previous = self._runtime
        self._runtime = runtime
        if previous is not None and previous["bee"] is not None:
            old = previous["bee"]
            if self.hive._bees.get(old.name) is old:
                self.hive.remove(old.name)
        return pin

    def _check_runtime(self, runtime):
        self.records.current(runtime["selection"])
        runtime["state"].check_state()
        runtime["owner"].admit(self._authority())
        bee = runtime["bee"]
        if bee is not None:
            self.records.current(runtime["description"]["program"])
            if (self.hive._bees.get(bee.name) is not bee or
                    (bee._handler, bee._context, bee.capability, bee.worker_type, bee.model) != runtime["registration"]):
                raise ValueError("registered worker changed after selection")

    def learn(self, observations, *, model_id, expected_model_version, expected_selection_version):
        """Fit one whole-field observation batch and activate the immutable child.

        The observations record contains canonical assignment observation rows.
        Source identities, axes, arithmetic and output ports remain governed by
        ordinary train_model. A structural extension requires a separate API.
        """
        from rexgraph.cpp_assignment import assignment_batch
        from rexgraph.io.model_record import model_record
        from rexgraph.model_runtime import train_model
        _integer(expected_model_version, "expected_model_version")
        _integer(expected_selection_version, "expected_selection_version", 1)
        observations = copy_record_pin(observations)
        self.restore()
        runtime = self._runtime
        if runtime["selection"]["version"] != expected_selection_version:
            raise ValueError("learning selection version differs")
        self._check_runtime(runtime)
        rows = self.records.value(observations, "observations", current=True)
        context = self._authority()
        batch = assignment_batch(runtime["state"], runtime["source"], rows, execution_context=context)
        child = train_model(runtime["state"], runtime["source"], batch, execution_context=context)
        self.records.current(observations)
        self._check_runtime(runtime)
        self.session.publish(model_id, model_record(runtime["source"], child), expected_version=expected_model_version,
            meta={"parent_model": runtime["description"]["model"], "observations": observations})
        model = self.records.pin(model_id)
        return self.activate(model, expected_version=expected_selection_version)

    def decide(self, record_id, *, expected_version=0):
        """Retain the native family status; only an identified assignment can dispatch."""
        _integer(expected_version, "expected_version")
        self.restore()
        runtime = self._runtime
        self._check_runtime(runtime)
        value = dict(selection=runtime["selection"], status=runtime["description"]["status"],
                     model=runtime["description"]["model"], program=runtime["description"]["program"],
                     binding=runtime["description"]["binding"])
        self.session.save_result(record_id, _result(value, "decision"), expected_version=expected_version)
        pin = self.records.pin(record_id)
        self._decisions[record_pin_key(pin)] = runtime
        return pin

    def assess_decision(self, pin):
        accepted, status = None, "unissued"
        try:
            pin = copy_record_pin(pin)
            runtime = self._decisions.get(record_pin_key(pin))
            if runtime is not None:
                self.records.current(pin)
                self._check_runtime(runtime)
                status = runtime["description"]["status"]
                accepted = True if runtime["bee"] is not None else False if status == "inconsistent" else None
        except (ValueError, KeyError, ImportError, OSError):
            status = "stale_or_unavailable"
        from rcql import Result
        return Result((accepted, status, pin), aliases=("accepted", "status", "decision"))

    def dispatch(self, decision, input, goal, *, record_id, expected_version=0):
        """Invoke the chosen Hive program, checking its whole response and task goal."""
        from rcql import Result
        from rexgraph.cpp_assignment import read_assignment_result
        _integer(expected_version, "expected_version")
        decision, input, goal = (copy_record_pin(ref) for ref in (decision, input, goal))
        if self.assess_decision(decision).named_values["accepted"] is not True:
            raise ValueError("worker assignment is unresolved, stale or unavailable")
        runtime = self._decisions[record_pin_key(decision)]
        original_input = self.records.value(input, "input", current=True)
        goal_archive = self.records.value(goal, "goal", current=True)
        if not isinstance(original_input, bytes) or not isinstance(goal_archive, bytes):
            raise TypeError("dispatch requires native input and prospective goal archive records")
        self._check_runtime(runtime)
        self.records.current(input)
        self.records.current(goal)
        started = perf_counter_ns()
        result = self.hive.invoke(runtime["bee"].name, {"input": original_input}, sender="assignment:" + self.record_id)
        elapsed = perf_counter_ns() - started
        if not isinstance(result, Result) or tuple(result.aliases) != ("response",):
            raise ValueError("selected worker did not return its declared complete response")
        output = result.named_values["response"]
        context = self._authority()
        actual = read_assignment_result(output, execution_context=context, state=runtime["state"])
        actual.admit(context)
        if (not actual.model(context).same_owner(runtime["owner"])
                or actual.inspect(context)["candidate"] != runtime["description"]["candidate"]
                or actual.original(context).input().archive() != original_input):
            raise ValueError("returned worker model, binding or original input differs")
        postcondition = actual.restore_goal(goal_archive, context)
        certificate = actual.postcondition(postcondition, context)
        self._check_runtime(runtime)
        self.records.current(input)
        self.records.current(goal)
        # The native receipt still truthfully describes its own internal family
        # execution. This outer receipt records the actual Hive invocation.
        output_id = record_id + "/output/" + str(expected_version + 1)
        self.session.save_result(output_id, result, expected_version=0)
        output_pin = self.records.pin(output_id)
        receipt = dict(profile="agent.learned-worker-execution.v1", decision=decision,
                       input=input, goal=goal, output=output_pin, worker=runtime["bee"].name,
                       worker_invoked=True, binding=runtime["description"]["binding"],
                       postcondition=certificate, elapsed_ns=elapsed,
                       retained_native_entries=runtime["owner"].inspect(context)["retained_entries"])
        self.session.save_result(record_id, _result(receipt, "execution"), expected_version=expected_version)
        pin = self.records.pin(record_id)
        self._executions[record_pin_key(pin)] = (runtime, receipt, actual, postcondition)
        return pin

    def assess_execution(self, pin):
        from rcql import Result
        accepted, status = None, "unissued"
        try:
            pin = copy_record_pin(pin)
            issued = self._executions.get(record_pin_key(pin))
            if issued is not None:
                runtime, receipt, actual, goal = issued
                self.records.current(pin)
                for ref in (receipt["decision"], receipt["input"], receipt["goal"], receipt["output"]):
                    self.records.current(ref)
                self._check_runtime(runtime)
                status = actual.postcondition(goal, self._authority())["status"]
                accepted = True if status == "satisfied" else False if status == "refuted" else None
        except (ValueError, KeyError, ImportError, OSError):
            status = "stale_or_unavailable"
        return Result((accepted, status, pin), aliases=("accepted", "status", "execution"))

    def publish(self, execution, record_id, *, expected_version=0):
        """Recheck current native evidence before publishing small result references."""
        _integer(expected_version, "expected_version")
        execution = copy_record_pin(execution)
        verdict = self.assess_execution(execution)
        if verdict.named_values["accepted"] is not True:
            raise ValueError("worker execution cannot publish: " + verdict.named_values["status"])
        receipt = self._executions[record_pin_key(execution)][1]
        value = dict(execution=execution, output=receipt["output"], decision=receipt["decision"])
        self.session.save_result(record_id, _result(value, "answer"), expected_version=expected_version)
        return self.records.pin(record_id)

    def release(self):
        """Release live decisions and output owners; stored records remain inspectable."""
        self._decisions.clear()
        self._executions.clear()

    def close(self):
        """Detach this instance's worker and release its live native owners."""
        self.release()
        if self._runtime is not None and self._runtime["bee"] is not None:
            bee = self._runtime["bee"]
            if self.hive._bees.get(bee.name) is bee:
                self.hive.remove(bee.name)
        self._runtime = None

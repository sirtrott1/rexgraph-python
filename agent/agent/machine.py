"""Persistent, versioned RCQL program transitions executed as Hive capabilities."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


def _text(value, name):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(name + " requires nonempty text of at most 512 characters")
    return value


def _version(value):
    if type(value) is not int or value < 1:
        raise ValueError("a machine requires explicit positive record versions")
    return value


def _packet(value):
    from rcql import Result
    return Result((value,)).to_bytes()


@dataclass(frozen=True)
class MachineTransition:
    """A committed transition, with its complete result and actual publication."""
    record: object
    result: object
    job_id: str
    replayed: bool = False


@dataclass(frozen=True)
class MachineCheckpoint:
    """Decoded machine state alongside its actual committed complex carrier."""
    record: object
    value: object
    state_digest: str
    carrier: object


class RelationalMachine:
    """One pinned pure program with explicit message and state-feedback ports.

    Each successful job publishes its state and complete result in one guarded
    RCDB record revision. Cache publications are independent derived records.
    Failed jobs leave machine state unchanged. No external effects are retried.
    """

    def __init__(self, session, record_id):
        from rcql.prepared import PreparedQueryCache

        from .relational import RelationalSession
        if not isinstance(session, RelationalSession):
            raise TypeError("a machine requires an existing RelationalSession")
        self.session = session
        self.record_id = _text(record_id, "machine record")
        namespace = "machine-cache/" + hashlib.sha256(record_id.encode()).hexdigest()
        self.cache = PreparedQueryCache(session.store, namespace=namespace)
        self._preparation = None
        self._definition = None

    @classmethod
    def create(cls, session, record_id, *, program_id, program_version, sources,
               state, message_ports=(), feedback=None):
        """Create a machine at version one, pinning every program and source."""
        from rcql.program_codec import dumps

        from rcql import Program, Result
        machine = cls(session, record_id)
        program_snapshot = session.read(_text(program_id, "program"), _version(program_version))
        program = Program.from_record(program_snapshot.value)
        selected = {}
        for alias, selector in sources.items():
            if not isinstance(selector, dict) or set(selector) != {"record_id", "version"}:
                raise ValueError("machine sources require record_id and explicit version")
            snapshot = session.read(selector["record_id"], _version(selector["version"]))
            selected[alias] = {**selector, "state_digest": snapshot.state_digest}
        ports = tuple(message_ports)
        expected = {spec.name for spec in program.inputs}
        feedback = dict(feedback or {})
        if (not isinstance(state, dict) or set(state) != expected or len(set(ports)) != len(ports)
                or not set(ports) <= expected or not set(feedback) <= expected
                or not set(feedback.values()) <= {name for name, _ in program.outputs}):
            raise ValueError("machine state, message and feedback ports must match the program")
        definition = dumps({"schema": "agent.machine", "version": 1,
            "program": {"record_id": program_id, "version": program_version,
                        "state_digest": program_snapshot.state_digest,
                        "program_digest": program.coefficient_digest},
            "sources": selected, "message_ports": ports, "feedback": feedback})
        machine._admit(definition, state)
        checkpoint = Result(({"schema": "agent.machine-state.v1", "definition": definition,
                              "state": dict(state), "last_job": None},), aliases=("machine",))
        machine._publish(checkpoint, expected_version=0)
        return machine

    def _publish(self, checkpoint, *, expected_version, meta=None):
        from rcql.program_codec import loads

        from rexgraph import RexGraph
        carrier = RexGraph.from_cells([1, [[0]]])
        carrier.attach_metadata(1, 0, "agent_machine_state", checkpoint.to_bytes())
        declaration = loads(checkpoint.values[0]['definition'])
        references = (declaration['program'], *declaration['sources'].values())
        carrier._agent_meta = {'rcql_evidence': [
            {key: ref[key] for key in ('record_id', 'version', 'state_digest')} for ref in references]}
        return self.session.publish(self.record_id, carrier, expected_version=expected_version,
            meta={"machine_schema": "agent.machine-state.v1", **(meta or {})})

    def _preparation_for(self, definition):
        from rcql.program_codec import loads

        value = loads(definition)
        if (not isinstance(value, dict) or set(value) != {
                "schema", "version", "program", "sources", "message_ports", "feedback"}
                or value["schema"] != "agent.machine" or type(value["version"]) is not int
                or value["version"] != 1):
            raise ValueError("unsupported machine declaration")
        reference = value["program"]
        sources = {}
        for alias, selector in value["sources"].items():
            _version(selector["version"])
            sources[alias] = {key: selector[key] for key in ("record_id", "version")}
        if definition != self._definition:
            prepared = self.session.prepare_program(reference["record_id"],
                version=_version(reference["version"]), sources=sources)
            if (prepared.snapshot.state_digest != reference["state_digest"]
                    or prepared.program.coefficient_digest != reference["program_digest"]):
                raise ValueError("machine program differs from its pinned declaration")
            for alias, selector in value["sources"].items():
                if prepared._bound[alias].ref.state_digest != selector["state_digest"]:
                    raise ValueError("machine source differs from its pinned declaration")
            self._preparation, self._definition = prepared, definition
        prepared = self._preparation
        return value, prepared

    def _admit(self, definition, state):
        value, prepared = self._preparation_for(definition)
        sources = value["sources"]
        program = prepared.program
        if not isinstance(state, dict) or set(state) != {spec.name for spec in program.inputs}:
            raise ValueError("machine state must bind every declared input exactly")
        if (len(set(value["message_ports"])) != len(value["message_ports"])
                or not set(value["message_ports"]) <= set(state)
                or not set(value["feedback"]) <= set(state)
                or not set(value["feedback"].values()) <= {name for name, _ in program.outputs}):
            raise ValueError("machine declaration has invalid ports")
        from rcql.ast import Parameter
        if any(not isinstance(step.query.source, Parameter) or step.query.source.name not in sources
               for step in program.steps):
            raise ValueError("machine calculations require directly bound pinned sources")
        explanation = prepared.execute(state, explain=True, exactness="exact")
        outputs = dict(zip((name for name, _ in program.outputs), explanation['output_types'], strict=True))
        inputs = {spec.name: spec for spec in program.inputs}
        for port, alias in value['feedback'].items():
            inputs[port].validate(None, outputs[alias])
        return value, prepared

    def inspect(self, version=None):
        """Read a complete historical checkpoint without executing its program."""
        from rcql import Result
        snapshot = self.session.read(self.record_id, version)
        from rexgraph import RexGraph
        if not isinstance(snapshot.value, RexGraph):
            raise ValueError("record is not a machine checkpoint")
        packet = snapshot.value.get_metadata(1, 0, 'agent_machine_state')
        if not isinstance(packet, bytes):
            raise ValueError("record has no canonical machine state")
        result = Result.from_bytes(packet)
        if len(result.values) != 1:
            raise ValueError("record is not a machine checkpoint")
        value = result.values[0]
        if (not isinstance(value, dict) or set(value) != {"schema", "definition", "state", "last_job"}
                or value["schema"] != "agent.machine-state.v1"):
            raise ValueError("unsupported machine checkpoint")
        self._admit(value["definition"], value["state"])
        return MachineCheckpoint(snapshot.record, result, snapshot.state_digest, snapshot.value)

    def advance(self, *, job_id, expected_version, parameters=None, optimized=True):
        """Consume one explicit input patch and atomically publish the transition.

        Repeating a committed job with its original parent version and identical
        message returns that historical receipt. A different message under the
        same job identity is refused. Callers explicitly retry after a conflict.
        """
        from rcdb import VersionConflictError
        from rcql import Result
        _text(job_id, "job identity")
        _version(expected_version)
        if type(optimized) is not bool or parameters is not None and not isinstance(parameters, dict):
            raise TypeError("optimized must be boolean and parameters must be a port mapping")
        message = dict(parameters or {})
        request = hashlib.sha256(_packet({"parent_version": expected_version, "message": message})).hexdigest()
        # History metadata selects the one committed receipt; no mutable in-memory
        # job table is authoritative, and old jobs survive later transitions.
        for record in self.session.store.history(self.record_id):
            if (record.meta or {}).get("machine_job") == job_id:
                if record.meta.get("machine_request") != request:
                    raise ValueError("job identity was already used for a different request")
                snapshot = self.inspect(record.version)
                job = snapshot.value.values[0]["last_job"]
                if job["id"] != job_id or job["request_digest"] != request:
                    raise ValueError("machine job receipt differs from its publication")
                return MachineTransition(snapshot.record, Result.from_bytes(job["result"]), job_id, True)
        snapshot = self.inspect()
        if snapshot.record.version != expected_version:
            raise VersionConflictError("machine parent version differs from current state")
        checkpoint = snapshot.value.values[0]
        declaration, preparation = self._preparation_for(checkpoint["definition"])
        if not set(message) <= set(declaration["message_ports"]):
            raise ValueError("message attempts to modify an undeclared input port")
        supplied = {**checkpoint["state"], **message}
        # inspect admitted the parent. Execution checks the patched input; the
        # following admission checks the complete prospective feedback state.
        result = preparation.execute(supplied, exactness="exact",
            cache=self.cache if optimized else None)
        named = dict(zip(result.aliases, result.values, strict=True))
        state = {**supplied, **{port: named[alias] for port, alias in declaration["feedback"].items()}}
        self._admit(checkpoint["definition"], state)
        job = {"id": job_id, "request_digest": request, "parent_version": expected_version,
               "message": message, "optimized": optimized, "result": result.to_bytes()}
        next_checkpoint = Result(({**checkpoint, "state": state, "last_job": job},), aliases=("machine",),
                                 provenance=result.provenance, execution=result.execution)
        published = self._publish(next_checkpoint, expected_version=expected_version,
            meta={"machine_job": job_id, "machine_request": request})
        return MachineTransition(published, result, job_id)

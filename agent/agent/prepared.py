"""Owned program and record snapshots for repeated in-process execution."""
from threading import RLock


class PreparedRelationalProgram:
    """Pin explicit record revisions and the native preparations they own.

    Revisions are read through the session's existing store scope once. Later
    publications do not retarget this preparation. Inputs, policies and the
    actual pinned owners are still checked on every application.
    """

    def __init__(self, session, record_id, *, version, sources):
        from rcql import Program
        from rexgraph.tensor_field import FieldSource

        from .relational import _integer
        _integer(version, "version", 1)
        for selector in sources.values():
            if not isinstance(selector, dict) or set(selector) != {"record_id", "version"}:
                raise ValueError("preparation requires explicit source record revisions")
            _integer(selector["version"], "version", 1)
        self._session = session
        self.snapshot = session.read(record_id, version)
        self.program = Program.from_record(self.snapshot.value)
        self._runtime, self._bound = session._runtime(sources)
        self._references = (FieldSource(self.snapshot.value, record_id, version, self.snapshot.state_digest),
            *(FieldSource(v.value, v.ref.record_id, v.ref.record_version, v.ref.state_digest)
              for v in self._bound.values() if v.ref is not None))
        self._resources = {}
        self._lock = RLock()
        self.check()

    def check(self):
        for reference in self._references:
            reference.check()
        for bound in self._bound.values():
            bound.require("read")
        _ = self.program.coefficient_digest

    def execute(self, parameters, *, cache=None, explain=False, exactness="declared"):
        from rcql.execution_trace import query_resource_scope
        with self._lock, query_resource_scope(resources=self._resources):
            self.check()
            supplied = self._session._bind_record_sources(parameters, self._bound)
            output = self._runtime.execute_program(self.program, params=supplied,
                cache=cache, explain=explain, exactness=exactness)
            self.check()
            return output if explain else self._session._program_result(self.snapshot, self.program, output, {})

    def clear(self):
        """Release derived native owners; the original source pins remain."""
        with self._lock:
            self._resources.clear()

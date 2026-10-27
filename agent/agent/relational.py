"""Agent access to the existing RCQL runtime and governed RCDB record store.

Python callers receive native values. The tool handlers below use canonical RCQL
result bytes for complete transport and a bounded JSON preview for inspection.
"""
from __future__ import annotations

import base64
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} requires an integer >= {minimum}")
    return value


def _flag(value, name):
    if type(value) is not bool:
        raise TypeError(f"{name} requires a boolean")
    return value


def _literal(value):
    """Read finite JSON parameters, including the existing exact fraction view."""
    if isinstance(value, dict):
        if set(value) == {"numerator", "denominator"}:
            if type(value["numerator"]) is not int or type(value["denominator"]) is not int:
                raise TypeError("fraction components require integers")
            return Fraction(value["numerator"], value["denominator"])
        return {k: _literal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_literal(v) for v in value]
    return value


def _preview(value, depth=0):
    """Bound metadata inspection without expanding native fields or object arrays."""
    if depth > 12:
        return {"kind": type(value).__name__, "preview": "omitted"}
    if value is None or type(value) in (bool, str, float):
        return value
    if type(value) is int:
        return value if abs(value) <= 2**53 - 1 else {"integer": str(Decimal(value))}
    if isinstance(value, Fraction):
        return {"numerator": _preview(value.numerator), "denominator": _preview(value.denominator)}
    if isinstance(value, (tuple, list)):
        sample = [_preview(v, depth + 1) for v in value[:32]]
        return sample if len(value) <= 32 else {"sample": sample, "count": len(value), "truncated": True}
    if isinstance(value, dict):
        sample = {str(k): _preview(v, depth + 1) for k, v in list(value.items())[:32]}
        return sample if len(value) <= 32 else {"sample": sample, "count": len(value), "truncated": True}
    return {"kind": type(value).__name__, "preview": "use the canonical result payload or stored record"}


def _program(declaration):
    from rcql import OutputRef, Program, ProgramInput, ProgramStep, parse
    if not isinstance(declaration, dict) or set(declaration) - {"name", "steps", "inputs", "outputs"}:
        raise ValueError("program declaration accepts name, steps, inputs and outputs")
    if not {"name", "steps"} <= declaration.keys():
        raise ValueError("program declaration requires name and steps")
    if not isinstance(declaration["steps"], (tuple, list)) or not declaration["steps"]:
        raise TypeError("program steps require a nonempty sequence")
    inputs = declaration.get("inputs", ())
    if not isinstance(inputs, (tuple, list)) or any(not isinstance(spec, dict) for spec in inputs):
        raise TypeError("program inputs require a sequence of input contract mappings")

    def references(value, name):
        if not isinstance(value, dict) or any(not isinstance(ref, dict) for ref in value.values()):
            raise TypeError(f"{name} require a mapping from parameter names to output reference mappings")
        return tuple((key, OutputRef(**ref)) for key, ref in value.items())

    steps = []
    for step in declaration["steps"]:
        if not isinstance(step, dict) or set(step) - {"name", "query", "inputs"}:
            raise ValueError("program step accepts name, query and input links")
        if not {"name", "query"} <= step.keys():
            raise ValueError("program step requires name and query")
        links = references(step.get("inputs", {}), "program step inputs")
        steps.append(ProgramStep(step["name"], parse(step["query"]), links))
    outputs = references(declaration.get("outputs", {}), "program outputs")
    return Program(declaration["name"], tuple(steps),
        tuple(ProgramInput(**spec) for spec in inputs), outputs)


class RelationalSession:
    """Use Agent's store and typed runtime under one existing caller context.

    Omitting a context is the trusted local operator, as in mcp_tools.call.
    Socket and Hive tool callers reuse ScopedStore's workspace and actor rules.
    No store URL or filesystem path is accepted from tool arguments.
    """

    def __init__(self, store=None, *, context=None):
        from agent.rcdb import default_store
        from agent.record_access import require_store
        from agent.server.scope import ScopedStore
        from rcql import register_result_storage_codec
        register_result_storage_codec()
        store = require_store(default_store() if store is None else store)
        if context is not None and context.auth_enabled:
            if isinstance(store, ScopedStore):
                if (store._workspace, store._caller) != (context.workspace, context.identity):
                    raise PermissionError("store is already bound to another caller")
            else:
                store = ScopedStore(store, context.workspace, context.identity)
        self.store = store
        self.context = context

    def read(self, record_id, version=None):
        if version is not None:
            _integer(version, "version", 1)
        value = self.store.read_record(record_id, version=version)
        if value is None:
            raise KeyError(f"no record {record_id!r}")
        return value

    def records(self, *, limit=100, offset=0):
        _integer(limit, "limit", 1)
        _integer(offset, "offset")
        return self.store.list(limit=min(limit, 1000), offset=offset)

    def publish(self, record_id, value, *, expected_version, meta=None):
        """Publish through the existing checked record engine; never infer a version."""
        from rcql import Result
        from rexgraph import RexGraph
        _integer(expected_version, "expected_version")
        if isinstance(value, Result):
            return self.save_result(record_id, value, expected_version=expected_version, meta=meta)
        if isinstance(value, RexGraph):
            metadata = dict(getattr(value, "_agent_meta", {}) or {})
            metadata.update(meta or {})
            return self.store.commit_mutation(record_id, value, expected_version=expected_version,
                meta=metadata, actor="agent", analytics=False)
        return self.store.put_record(record_id, value, expected_version=expected_version, meta=meta)

    def _runtime(self, sources=None, *, allow_write=False):
        from rcql.types import SourceRef

        from agent.rcql_runtime import RCQLRuntime
        from rcql import BoundSource, SourcePolicy
        _flag(allow_write, "allow_write")
        permissions = ["read", "records", "identity", "history", "search"]
        if allow_write:
            permissions.extend(("mutate", "train"))
        policy = SourcePolicy.allow(*permissions)
        bound = {"db": BoundSource(self.store, policy)}
        for alias, selector in (sources or {}).items():
            if alias == "db":
                raise ValueError("db is reserved for the caller's record store")
            if not isinstance(selector, dict) or set(selector) - {"record_id", "version"}:
                raise ValueError("source bindings accept record_id and version")
            snapshot = self.read(selector["record_id"], selector.get("version"))
            ref = SourceRef(alias, state_digest=snapshot.state_digest, policy_digest=policy.digest,
                record_id=snapshot.record.id, record_version=snapshot.record.version)
            bound[alias] = BoundSource(snapshot.value, policy, ref=ref)
        runtime = RCQLRuntime()
        for name, value in bound.items():
            runtime.register(name, value)
        return runtime, bound

    def _parameters(self, parameters=None, record_parameters=None):
        result = dict(parameters or {})
        references = {}
        for name, selector in (record_parameters or {}).items():
            if name in result:
                raise ValueError("a parameter was supplied both as a value and a record")
            result[name], references[name] = self._record_value(selector)
        return result, references

    @staticmethod
    def _bind_record_sources(parameters, bound):
        """Bind explicit portable endpoint references to selected owning snapshots."""
        from rcql.result_codec import bind_sources
        from rcql.source_context import field_references
        for reference in field_references(parameters):
            reference.check()
        references = {(value.ref.state_digest, value.ref.record_id, value.ref.record_version): value.value
                      for value in bound.values() if value.ref is not None
                      and value.ref.state_digest is not None and value.ref.record_id is not None
                      and value.ref.record_version is not None}
        from rexgraph.source_binding import CppSourceReference, CppAlgebraicSourceReference
        native_types = (CppSourceReference, CppAlgebraicSourceReference)
        needed = {(ref.state_digest, ref.record_id, ref.version)
                  for ref in field_references(parameters) if isinstance(ref, native_types)}
        if needed:
            from rexgraph import RexGraph
            from rexgraph.model_state import ModelState
            from rcql.model_operators import source_for
            from rexgraph.io.model_record import read_model_record
            for selected in bound.values():
                graph = selected.value
                if not isinstance(graph, RexGraph) or not graph.nE:
                    continue
                state = graph.get_metadata(1, 0, "model")
                if not isinstance(state, ModelState) or not isinstance(state.source, native_types):
                    continue
                key = (state.source.state_digest, state.source.record_id, state.source.version)
                if key in needed:
                    native = (read_model_record(graph, _prepare=False)[1]
                              if isinstance(state.source, CppSourceReference) else source_for(graph, state))
                    state.source.bind(native).check()
                    references[key] = native
        return bind_sources(parameters, references, strict=True)

    def _record_value(self, selector):
        from rcql import NameRelation, Program, RecursiveProgram, Result
        decoders = {"value": lambda v: v, "program": Program.from_record,
                    "name": NameRelation.from_record, "recursive": RecursiveProgram.from_record}
        if not isinstance(selector, dict) or set(selector) - {"record_id", "version", "kind", "output"}:
            raise ValueError("record parameters accept record_id, version, kind and optional result output")
        snapshot = self.read(selector["record_id"], selector.get("version"))
        value = snapshot.value
        if "output" in selector:
            index = _integer(selector["output"], "output")
            if not isinstance(value, Result) or index >= len(value.values):
                raise ValueError("output must select an existing canonical Result value")
            value = value.values[index]
        kind = selector.get("kind", "value")
        if kind not in decoders:
            raise ValueError("record parameter kind must be value, program, name or recursive")
        reference = {"record_id": snapshot.record.id, "version": snapshot.record.version,
            "state_digest": snapshot.state_digest, "kind": kind}
        if "output" in selector:
            reference["output"] = selector["output"]
        return decoders[kind](value), reference

    def query(self, query, *, parameters=None, sources=None, record_parameters=None,
              allow_write=False, save_as=None, expected_version=0):
        from rcql.ast import Literal

        from rcql import MutationQuery, parse
        query = parse(query) if isinstance(query, str) else query
        if isinstance(query, MutationQuery) and save_as is not None:
            raise ValueError("mutation receipts cannot be saved as query result records")
        if isinstance(query, MutationQuery) and query.expected_version == Literal(None):
            raise ValueError("Agent mutations require an explicit expected_version")
        runtime, _ = self._runtime(sources, allow_write=allow_write)
        supplied, references = self._parameters(parameters, record_parameters)
        result = runtime.execute(query, params=supplied)
        if references:
            result = replace(result, native_plan={**(result.native_plan or {}), "agent_record_parameters": references})
        if save_as is not None:
            self.save_result(save_as, result, expected_version=expected_version)
        return result

    def save_result(self, record_id, result, *, expected_version=0, meta=None, tags=None):
        from rcdb import QUERY_RESULT_CODEC
        from rcql import register_result_storage_codec
        register_result_storage_codec()
        return self.store.put_record(record_id, result, codec=QUERY_RESULT_CODEC,
            expected_version=_integer(expected_version, "expected_version"), meta=meta, tags=tags)

    def publish_program(self, record_id, program, *, expected_version, meta=None):
        from rcql import Program
        if isinstance(program, dict):
            program = _program(program)
        if not isinstance(program, Program):
            raise TypeError("publish_program requires a finite Program or its step declaration")
        return self.publish(record_id, program.to_record(), expected_version=expected_version, meta=meta)

    def run_program(self, record_id, *, version=None, parameters=None, sources=None,
                    record_parameters=None, allow_write=False, save_as=None, expected_version=0,
                    cache=None, exactness="declared", bind_record_sources=False):
        from rcql import Program
        snapshot = self.read(record_id, version)
        program = Program.from_record(snapshot.value)
        runtime, bound = self._runtime(sources, allow_write=allow_write)
        supplied, references = self._parameters(parameters, record_parameters)
        _flag(bind_record_sources, "bind_record_sources")
        if bind_record_sources:
            supplied = self._bind_record_sources(supplied, bound)
        output = runtime.execute_program(program, params=supplied, cache=cache, exactness=exactness)
        result = self._program_result(snapshot, program, output, references)
        if save_as is not None:
            self.save_result(save_as, result, expected_version=expected_version)
        return result

    def prepare_program(self, record_id, *, version, sources):
        """Retain explicit owning snapshots for repeated finite program execution."""
        from .prepared import PreparedRelationalProgram
        return PreparedRelationalProgram(self, record_id, version=version, sources=sources)

    def run_program_bundle(self, record_id, *, version, bundle, source_outputs,
                           parameter_outputs, parameters=None, save_as=None, expected_version=0):
        """Execute with actual sources and fields selected from one owning Result.

        Output indices are explicit ports within the pinned bundle. Its record
        identity is retained separately from the native sources it contains.
        Historical execution does not grant current publication authority.
        """
        from rcql import BoundSource, Program, Result
        from .record_access import PinnedRecords, copy_record_pin
        _integer(version, "version", 1)
        bundle = copy_record_pin(bundle)
        value = PinnedRecords(self).read(bundle)
        if not isinstance(value, Result):
            raise TypeError("program bundle requires a canonical Result record")

        def selected(mapping, reserved=()):
            if not isinstance(mapping, dict):
                raise TypeError("bundle ports require name to output-index mappings")
            result = {}
            for name, index in mapping.items():
                if not isinstance(name, str) or not name or name in reserved:
                    raise ValueError("invalid or reserved program bundle port")
                _integer(index, "bundle output")
                if index >= len(value.values):
                    raise ValueError("bundle output is outside the complete Result")
                result[name] = index
            return result

        sources = selected(source_outputs, ("db",))
        inputs = selected(parameter_outputs)
        supplied = dict(parameters or {})
        if supplied.keys() & inputs.keys():
            raise ValueError("caller parameters overlap bundle-owned inputs")
        supplied.update((name, value.values[index]) for name, index in inputs.items())
        snapshot = self.read(record_id, version)
        program = Program.from_record(snapshot.value)
        runtime, bound = self._runtime()
        for name, index in sources.items():
            runtime.register(name, BoundSource(value.values[index], bound["db"].policy))
        output = runtime.execute_program(program, params=supplied, exactness="exact")
        result = self._program_result(snapshot, program, output, {})
        result = replace(result, native_plan={**result.native_plan,
            "agent_bundle": {"profile": "agent.program-bundle.v1", "record": bundle,
                "source_outputs": sources, "parameter_outputs": inputs},
            "bundle_values": value.values})
        if save_as is not None:
            self.save_result(save_as, result, expected_version=expected_version)
        return result

    @staticmethod
    def _program_result(snapshot, program, output, references):
        from rcql import Result
        steps = dict(output.steps)
        result = Result(output.values, plan=("PROGRAM_RUN",), aliases=output.aliases,
            exactness=tuple(steps[ref.step].exactness[ref.output] for _, ref in program.outputs),
            native_plan={"program_digest": output.program_digest, "record_id": snapshot.record.id,
                "version": snapshot.record.version, "program": program,
                "record_parameters": references,
                "dependencies": tuple({"reference": ref.as_record(), "source": ref.source}
                    for ref in output.dependencies),
                "step_results": tuple({"name": n, "values": r.values, "aliases": r.aliases,
                    "native_plan": r.native_plan, "provenance": r.provenance,
                    "execution": r.execution} for n, r in output.steps)},
            provenance=tuple(p for _, step in output.steps for p in step.provenance),
            execution=tuple(p for _, step in output.steps for p in step.execution))
        return result

    def transform_program(self, record_id, *, version, rule, arguments,
                          parameters=None, sources=None, record_parameters=None):
        from rcql import Program, ProgramTransformation, bind
        snapshot = self.read(record_id, version)
        original = Program.from_record(snapshot.value)
        change = ProgramTransformation.program(original, rule, arguments)
        _, bound = self._runtime(sources)
        bindings = {n: bind(n, v.value, v.policy, source_ref=v.ref) for n, v in bound.items()}
        supplied, references = self._parameters(parameters, record_parameters)
        candidate = change.compile_program(original, bindings, supplied)
        # The canonical certificate is part of the same record publication.
        record = candidate.to_record()
        record.attach_metadata(1, 0, "rcql_transformation", change.to_bytes().decode("utf8"))
        published = self.publish(record_id, record, expected_version=version,
            meta={"agent_validation_records": references} if references else None)
        return {"record": published.to_wire_dict(), "program_digest": candidate.coefficient_digest,
                "transformation_digest": change.coefficient_digest, "verification": change.verify(original)}


def _session(context):
    return RelationalSession(context=context)


def _response(result, *, include_payload=True):
    _flag(include_payload, "include_payload")
    if any(p.get("logical_operator") == "COMMIT" for p in result.provenance):
        return {"kind": "mutation", "records": [v.to_wire_dict() for v in result.values]}
    response = {"kind": "query", "values": _preview(result.values),
        "aliases": list(result.aliases), "provenance": _preview(result.provenance)}
    if include_payload:
        response["result_base64"] = base64.b64encode(result.to_bytes()).decode("ascii")
        response["encoding"] = "rcql.Result.to_bytes"
    return response


def records(limit=100, offset=0, _ctx=None):
    return {"records": [r.to_wire_dict() for r in _session(_ctx).records(limit=limit, offset=offset)]}


def import_file(record_id, expected_version, files, format, _ctx=None):
    """Publish one resolved file through the installed Core format and record codecs."""
    from agent.native_files import load_native_file as load
    from agent.server.budget import check_size
    if not isinstance(files, (list, tuple)) or len(files) != 1:
        raise ValueError("file publication requires exactly one resolved input")
    if not isinstance(format, str) or not format:
        raise TypeError("file publication requires an explicit installed format")
    _integer(expected_version, "expected_version")
    value = load(files[0], format=format)
    from rexgraph import RexGraph
    if isinstance(value, RexGraph):
        check_size({"grade0": value.nV, "grade1": value.nE,
                    "participations": len(value.boundary_idx)})
    record = _session(_ctx).publish(record_id, value, expected_version=expected_version,
                                   meta={"agent_file_format": format})
    return {"record": record.to_wire_dict()}


def inspect(record_id, version=None, history=False, _ctx=None):
    session = _session(_ctx)
    snapshot = session.read(record_id, version)
    result = {"record": snapshot.record.to_wire_dict(), "state_digest": snapshot.state_digest}
    if _flag(history, "history"):
        result["history"] = [r.to_wire_dict() for r in session.store.history(record_id)]
    return result


def publish(record_id, expected_version, cells=None, value_record=None, meta=None, _ctx=None):
    from agent.server.budget import check_size
    from rexgraph import RexGraph
    _integer(expected_version, "expected_version")
    session = _session(_ctx)
    if (cells is None) == (value_record is None):
        raise ValueError("publish requires exactly one of cells or value_record")
    if value_record is not None:
        value, reference = session._record_value(value_record)
        metadata = dict(_literal(meta) or {}, agent_parent_record=reference)
        record = session.publish(record_id, value, expected_version=expected_version, meta=metadata)
        return {"record": record.to_wire_dict()}
    if not isinstance(cells, (tuple, list)) or not cells:
        raise TypeError("cells requires Core's nonempty graded cell declaration")
    _integer(cells[0], "vertex count")
    counts = {"grade0": cells[0]}
    participants = 0
    for grade, declarations in enumerate(cells[1:], 1):
        if not isinstance(declarations, (tuple, list)):
            raise TypeError("each positive grade requires a sequence of cell declarations")
        counts["grade" + str(grade)] = len(declarations)
        participants += sum(len(cell) for cell in declarations)
    check_size({**counts, "participations": participants})
    record = session.publish(record_id, RexGraph.from_cells(cells),
        expected_version=expected_version, meta=_literal(meta))
    return {"record": record.to_wire_dict()}


def query(query, parameters=None, sources=None, record_parameters=None, allow_write=False,
          save_as=None, expected_version=0, include_payload=True, _ctx=None):
    _flag(include_payload, "include_payload")
    result = _session(_ctx).query(query, parameters=_literal(parameters), sources=sources,
        record_parameters=record_parameters, allow_write=allow_write,
        save_as=save_as, expected_version=expected_version)
    return _response(result, include_payload=include_payload)


def publish_program(record_id, declaration, expected_version, _ctx=None):
    record = _session(_ctx).publish_program(record_id, declaration, expected_version=expected_version)
    return {"record": record.to_wire_dict()}


def run_program(record_id, version=None, parameters=None, sources=None, record_parameters=None,
                allow_write=False, save_as=None, expected_version=0, include_payload=True, _ctx=None):
    _flag(include_payload, "include_payload")
    result = _session(_ctx).run_program(record_id, version=version, parameters=_literal(parameters),
        sources=sources, record_parameters=record_parameters, allow_write=allow_write,
        save_as=save_as, expected_version=expected_version)
    return _response(result, include_payload=include_payload)


def transform_program(record_id, version, rule, arguments, parameters=None, sources=None,
                      record_parameters=None, _ctx=None):
    return _session(_ctx).transform_program(record_id, version=version, rule=rule,
        arguments=_literal(arguments), parameters=_literal(parameters), sources=sources,
        record_parameters=record_parameters)

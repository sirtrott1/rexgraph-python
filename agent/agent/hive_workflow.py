"""Declared worker dependencies executed by Hive's existing coordinator."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Condition
from types import MappingProxyType
from uuid import uuid4


def _name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("workflow step and worker names require nonempty text")
    return value


@dataclass(frozen=True)
class WorkflowRef:
    """Select a whole output, mapping/sequence path, or named RCQL Result value."""

    step: str
    path: tuple = ()

    def __post_init__(self):
        _name(self.step)
        path = tuple(self.path)
        if any(type(key) not in (str, int) or (type(key) is int and key < 0) for key in path):
            raise ValueError("workflow paths require literal keys or nonnegative indices")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class WorkflowCheck:
    """Run an explicit observer returning a named boolean or unresolved value.

    The worker must return an RCQL Result. Only a literal True accepts, False
    rejects and None leaves the obligation unresolved. Other result values and
    the complete observer result remain available as evidence. Pinned RCQL
    program workers provide pure observers with retained source declarations.
    """

    worker: str
    data: object = None
    verdict: str = "accepted"

    def __post_init__(self):
        _name(self.worker)
        _name(self.verdict)


@dataclass(frozen=True)
class WorkflowStep:
    """Invoke a registered capability worker after declared dependencies complete."""

    id: str
    worker: str
    data: object = None
    needs: tuple[str, ...] = ()
    guards: tuple[WorkflowCheck, ...] = ()
    checks: tuple[WorkflowCheck, ...] = ()

    def __post_init__(self):
        _name(self.id)
        _name(self.worker)
        needs = tuple(self.needs)
        if any(not isinstance(name, str) or not name for name in needs) or len(set(needs)) != len(needs):
            raise ValueError("workflow dependencies require distinct step names")
        object.__setattr__(self, "needs", needs)
        for name in ("guards", "checks"):
            declarations = tuple(getattr(self, name))
            if len(declarations) > 10000 or any(not isinstance(v, WorkflowCheck) for v in declarations):
                raise TypeError("workflow guards and checks require finite WorkflowCheck declarations")
            object.__setattr__(self, name, declarations)


@dataclass(frozen=True)
class WorkflowResult:
    """In-process outputs retaining their actual owners and per-step outcomes.

    RCQL Result/RCDB codecs remain the persistence contracts for stored values.
    A workflow is not a multi-record transaction or an automatic retry policy.
    """

    run_id: str
    outputs: Mapping
    steps: tuple

    @property
    def ok(self):
        return all(step["state"] == "completed" for step in self.steps)

    @property
    def verified(self):
        """Every task completed with at least one accepted result check."""
        return bool(self.steps) and self.ok and all(step.get("validation") == "accepted" for step in self.steps)

    def to_result(self):
        """Snapshot complete outcomes through the existing RCQL Result codec.

        Nested Results use their canonical bytes, including their provenance.
        Other native values keep their existing portable codecs. Unsupported
        handles are refused by the codec; callable workers are not serialized.
        Saving is explicit through RelationalSession.save_result, with its
        existing workspace and expected-version rules.
        """
        from rcql import Result

        active = set()

        def portable(value):
            if isinstance(value, Result):
                return {"encoding": "rcql.Result.to_bytes", "payload": value.to_bytes()}
            if not isinstance(value, (Mapping, tuple, list)):
                return value
            if id(value) in active or len(active) > 64:
                raise ValueError("workflow outcomes must be acyclic and finitely nested")
            active.add(id(value))
            try:
                if isinstance(value, Mapping):
                    return {key: portable(item) for key, item in value.items()}
                items = [portable(item) for item in value]
                return tuple(items) if isinstance(value, tuple) else items
            finally:
                active.remove(id(value))

        return Result(({"schema": "agent.workflow-outcome.v1", "run_id": self.run_id,
            "outputs": portable(self.outputs), "steps": portable(self.steps)},),
            plan=("HIVE_WORKFLOW",), aliases=("workflow",))


def _tree(value, reference, active=None):
    if isinstance(value, WorkflowRef):
        return reference(value)
    if not isinstance(value, (dict, list, tuple)):
        return value                       # native owners/fields remain whole
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError("workflow input containers must be acyclic")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            return {key: _tree(item, reference, active) for key, item in value.items()}
        items = [_tree(item, reference, active) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    finally:
        active.remove(id(value))


def run_workflow(hive, declarations, *, sender="user"):
    """Execute finite dependency waves through existing Hive capability workers.

    All names, workers and cycles are checked before any task starts. A failed
    task or unaccepted check blocks its dependents; independent tasks may
    complete. Guards run before a task, checks after its return. After-checks
    cannot roll back a mutation; guard mutations before invoking them.
    Coordinator failures do not retry already attempted mutations.
    """
    from . import activity

    steps = tuple(s if isinstance(s, WorkflowStep) else WorkflowStep(**s) for s in declarations)
    if not 0 < len(steps) <= 10000 or len({s.id for s in steps}) != len(steps):
        raise ValueError("workflow requires distinct steps in a finite nonempty declaration")
    names = {s.id for s in steps}
    dependencies, workers = {}, {}

    def register(name):
        bee = hive.get(name)
        if bee is None or not callable(bee._handler):
            raise ValueError(f"{name!r} requires a registered callable capability worker")
        workers[name] = (bee, bee._handler, bee._context, bee.capability, bee.worker_type, bee.model)

    for step in steps:
        register(step.worker)
        refs = []
        _tree(step.data, lambda ref, found=refs: found.append(ref.step))
        for before, phase in ((True, step.guards), (False, step.checks)):
            for check in phase:
                register(check.worker)
                observed = []
                _tree(check.data, lambda ref, found=observed: found.append(ref.step))
                # Only an after-check can inspect this task's own return.
                refs.extend(name for name in observed if before or name != step.id)
        dependencies[step.id] = tuple(dict.fromkeys((*step.needs, *refs)))
        if set(dependencies[step.id]) - names:
            raise ValueError(f"unknown dependency for workflow step {step.id!r}")
    remaining, known = set(names), set()
    while remaining:
        ready = {name for name in remaining if set(dependencies[name]) <= known}
        if not ready:
            raise ValueError("workflow dependencies contain a cycle")
        remaining -= ready
        known |= ready

    run_id = uuid4().hex
    outcomes, outputs, pending = {}, {}, {s.id: s for s in steps}
    gate = Condition()
    active, aborted = 0, False

    def resolve(ref, own=None):
        from rcql import Result
        value = own[1] if own is not None and ref.step == own[0] else outputs[ref.step]
        for key in ref.path:
            if isinstance(value, Result) and type(key) is str:
                if len(value.aliases) != len(value.values) or value.aliases.count(key) != 1:
                    raise ValueError("workflow Result paths require a unique declared output alias")
                value = value.values[value.aliases.index(key)]
            elif isinstance(value, Mapping) or isinstance(value, (tuple, list)) and type(key) is int:
                value = value[key]
            else:
                raise TypeError("workflow path requires a mapping key or sequence index")
        return value

    def invoke(name, data):
        bee, handler, context, capability, worker_type, model = workers[name]
        if (hive.get(name) is not bee or bee._handler is not handler or bee._context is not context
                or (bee.capability, bee.worker_type, bee.model) != (capability, worker_type, model)):
            raise RuntimeError("workflow worker changed; declare a new workflow")
        return hive.invoke(name, data, sender=sender)

    def observe(declarations, *, own=None):
        from rcql import Result

        observations = []
        for check in declarations:
            observation = {"worker": check.worker, "verdict": check.verdict}
            try:
                data = _tree(check.data, lambda ref: resolve(ref, own))
                observation["input"] = data
                result = invoke(check.worker, data)
                observation["result"] = result
                if (not isinstance(result, Result) or len(result.aliases) != len(result.values)
                        or result.aliases.count(check.verdict) != 1):
                    raise TypeError("workflow checks require a Result with a unique named verdict")
                verdict = result.values[result.aliases.index(check.verdict)]
                if verdict is not None and type(verdict) is not bool:
                    raise TypeError("workflow check verdict must be boolean or None")
                observation["state"] = "unresolved" if verdict is None else "accepted" if verdict else "rejected"
            except Exception as exc:
                observation.update(state="failed", error={"type": type(exc).__name__, "message": str(exc)})
            observations.append(observation)
        state = next((state for state in ("failed", "rejected", "unresolved")
                      if any(item["state"] == state for item in observations)), "accepted")
        return state, tuple(observations)

    def task(step):
        def execute():
            nonlocal active
            with gate:
                if aborted:
                    return None
                active += 1
            outcome = {"id": step.id, "worker": step.worker,
                "capability": workers[step.worker][3], "worker_type": workers[step.worker][4],
                "model": workers[step.worker][5],
                "dependencies": dependencies[step.id]}
            try:
                data = _tree(step.data, resolve)
                outcome["input"] = data
                if step.guards:
                    state, observations = observe(step.guards)
                    outcome["guards"] = observations
                    if state != "accepted":
                        outcome.update(state=state, validation="not_run")
                        return outcome
                value = invoke(step.worker, data)
                outcome.update(state="completed", value=value)
                if step.checks:
                    state, observations = observe(step.checks, own=(step.id, value))
                    outcome.update(checks=observations, validation=state,
                                   state="completed" if state == "accepted" else state)
            except Exception as exc:
                outcome.update(state="failed", error={"type": type(exc).__name__, "message": str(exc)})
            finally:
                with gate:
                    outcomes[step.id] = outcome
                    active -= 1
                    gate.notify_all()
            return outcome
        return {"id": step.id, "kind": "invoke", "fn": execute,
                "weight": hive._task_weight("invoke", step.worker)}

    while pending:
        ready = [s for s in steps if s.id in pending and set(dependencies[s.id]) <= outcomes.keys()]
        runnable = []
        for step in ready:
            pending.pop(step.id)
            blocked = tuple(name for name in dependencies[step.id] if outcomes[name]["state"] != "completed")
            if blocked:
                outcomes[step.id] = {"id": step.id, "worker": step.worker, "state": "skipped",
                    "dependencies": dependencies[step.id], "blocked_by": blocked}
            else:
                runnable.append(step)
        if not runnable:
            continue
        try:
            hive._run_wave([task(step) for step in runnable], serial_fallback=False)
        except Exception as exc:
            with gate:
                aborted = True
                gate.wait_for(lambda: active == 0)
            error = {"type": type(exc).__name__, "message": str(exc)}
            for step in (*runnable, *pending.values()):
                if step.id not in outcomes:
                    outcomes[step.id] = {"id": step.id, "worker": step.worker, "state": "skipped",
                        "dependencies": dependencies[step.id], "scheduler_error": error}
            pending.clear()
        for step in runnable:
            if step.id not in outcomes:
                outcomes[step.id] = {"id": step.id, "worker": step.worker, "state": "failed",
                    "dependencies": dependencies[step.id],
                    "error": {"type": "CoordinatorError", "message": "worker produced no outcome"}}
        outputs.update({step.id: outcomes[step.id]["value"] for step in runnable
                        if "value" in outcomes[step.id]})

    trace = tuple(MappingProxyType(outcomes[step.id]) for step in steps)
    activity.record("workflow:" + run_id, "complete", scope=hive.name,
                    detail={"steps": len(steps), "produced": len(outputs),
                            "completed": sum(s["state"] == "completed" for s in trace)})
    return WorkflowResult(run_id, MappingProxyType(outputs), trace)

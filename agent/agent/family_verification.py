"""Native finite-family checks with pinned RCDB proofs and guarded publication.

Verdicts describe the explicitly declared source and view. A finite symbol or
source label does not establish the truth of an external observation. Saved
proof packets are replay evidence; only execution in this verifier issues a
current publication grant. In-process callers share Agent's trusted operator.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from .relational import RelationalSession, _integer
from .record_access import PinnedRecords, copy_record_pin, record_pin_key as _pin_key

_PROFILE = "finite-equality-task-v1"
_QUESTION = "finite-source-question-v1"
_FAMILY_LIMITS = (65536, 64000000, 16777216, 65536, 65536)
def _result(value, alias):
    from rcql import Result
    return Result((value,), aliases=(alias,))


def _validate_task(spec):
    if isinstance(spec, dict) and spec.get("profile") == _QUESTION:
        return _validate_question(spec)
    expected = {"profile", "source", "projection", "claims", "dependencies",
                "family_limits", "max_checks", "max_proof_bytes"}
    if not isinstance(spec, dict) or set(spec) != expected or spec["profile"] != _PROFILE:
        raise ValueError("unsupported finite task record")
    _pin_key(spec["source"])
    if not isinstance(spec["dependencies"], (tuple, list)):
        raise TypeError("task dependencies require pinned record references")
    for pin in spec["dependencies"]:
        _pin_key(pin)
    projection, claims = spec["projection"], spec["claims"]
    family_limits = spec["family_limits"]
    max_checks, max_proof_bytes = spec["max_checks"], spec["max_proof_bytes"]
    if not isinstance(projection, dict) or set(projection) != {
            "key", "revision", "cells", "window", "fixed", "distinct_relations"}:
        raise ValueError("provide the complete finite projection declaration")
    _integer(projection["key"], "projection key", 1)
    _integer(projection["revision"], "projection revision", 1)
    if not isinstance(claims, (tuple, list)) or not claims:
        raise ValueError("provide a nonempty sequence of named equality claims")
    names = set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"name", "terms"}:
            raise ValueError("claims require name and terms")
        name = claim["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("claim names must be nonempty and unique")
        names.add(name)
        if not isinstance(claim["terms"], (tuple, list)):
            raise TypeError("claim terms require grade/cell/symbol triples")
        for term in claim["terms"]:
            if not isinstance(term, (tuple, list)) or len(term) != 3:
                raise ValueError("claim terms require grade/cell/symbol triples")
            for value in term:
                _integer(value, "finite claim coordinate or symbol")
    if not isinstance(family_limits, (tuple, list)) or len(family_limits) != 5:
        raise ValueError("finite family limits require five nonnegative integers")
    for value in family_limits:
        _integer(value, "finite family limit")
    _integer(max_checks, "max_checks")
    _integer(max_proof_bytes, "max_proof_bytes")
    def rows(values, width):
        if not isinstance(values, (tuple, list)):
            raise TypeError("finite projection coordinates require sequences")
        for row in values:
            if not isinstance(row, (tuple, list)) or len(row) != width:
                raise ValueError("finite projection coordinate arity differs")
            for value in row:
                _integer(value, "finite projection coordinate")
    rows(projection["cells"], 2)
    rows(projection["fixed"], 3)
    rows(projection["distinct_relations"], 2)
    if projection["window"] is not None:
        if not isinstance(projection["window"], (tuple, list)) or len(projection["window"]) != len(projection["cells"]):
            raise ValueError("finite projection window must match its cells")
        for values in projection["window"]:
            if not isinstance(values, (tuple, list)):
                raise TypeError("finite projection window requires symbol sequences")
            for value in values:
                _integer(value, "finite projection symbol")


def _question_cells(values, *, nonempty=False):
    if not isinstance(values, (tuple, list)) or nonempty and not values:
        raise ValueError("provide explicit question/observation cell sequences")
    result = []
    for cell in values:
        if not isinstance(cell, (tuple, list)) or len(cell) != 2:
            raise ValueError("question cells require grade/index pairs")
        for value in cell:
            _integer(value, "question coordinate")
        value = tuple(cell)
        if value in result:
            raise ValueError("question map repeats a cell")
        result.append(value)
    return tuple(result)


def _validate_question(spec):
    expected = {"profile", "source", "projection", "answer_cells", "observation_cells", "dependencies",
                "family_limits", "max_checks", "max_proof_bytes"}
    if set(spec) != expected:
        raise ValueError("unsupported finite question record")
    base = {k: v for k, v in spec.items() if k not in ("answer_cells", "observation_cells")}
    base.update(profile=_PROFILE, claims=({"name": "admission", "terms": ()},))
    _validate_task(base)
    q = _question_cells(spec["answer_cells"], nonempty=True)
    m = _question_cells(spec["observation_cells"])
    cells = tuple(dict.fromkeys((*q, *m)))
    d = spec["projection"]
    if (tuple(tuple(c) for c in d["cells"]) != cells or d["window"] is not None or d["fixed"]
            or d["distinct_relations"]):
        raise ValueError("questions require the complete original carriers and source constraints")


class FamilyVerifier:
    """Execute symbolic claims and answer projections with the native provider.

    Task, source and additional input/model/field dependencies are selected RCDB
    records. Publication requires all claims to be entailed, or a question to
    have one answer, in a nonempty complete joint image. Workers exchange small
    proof references; the full native source,
    joint tuples, whole witnesses and query ledger are saved once per execution.
    Dependency checks occur again inside publication, followed by an expected
    version write. This is not a transaction across dependency records.
    """

    def __init__(self, session):
        if not isinstance(session, RelationalSession):
            raise TypeError("family verification requires a RelationalSession")
        self.session = session
        self._records = PinnedRecords(session)
        self._epoch = 0
        self._available = True
        self._issued = {}

    def withdraw(self):
        """Withdraw this executor and invalidate all its earlier grants."""
        self._available = False
        self._epoch += 1

    def resume(self):
        """Admit new executions; withdrawn grants require fresh execution."""
        self._available = True
        self._epoch += 1

    def release(self, pin):
        """Release a live grant and its native owners; the RCDB proof remains replayable."""
        self._issued.pop(_pin_key(copy_record_pin(pin)), None)

    def pin(self, record_id, version=None):
        return self._records.pin(record_id, version)

    def _read(self, pin, *, current=False):
        return self._records.read(pin, current=current)

    def _current(self, pin):
        return self._records.current(pin)

    def _value(self, pin, alias, *, current=False):
        return self._records.value(pin, alias, current=current)

    @staticmethod
    def _provider():
        # An absent optional provider raises explicitly; no Python solver fallback.
        import rexgraph_cpp._finite_field as field_module
        import rexgraph_cpp._finite_search as module
        from rexgraph_cpp import NativeFiniteFamily, NativeFiniteSource, NativeFiniteImage, NativeFiniteImageQueries
        if not all(callable(getattr(owner, name, None)) for owner, name in (
                (NativeFiniteImage, "classify_equalities"), (NativeFiniteImage, "proof_archive"),
                (NativeFiniteImage, "pruning"), (NativeFiniteImageQueries, "proof_archive"),
                (NativeFiniteImageQueries, "pruning"))):
            raise NotImplementedError("native provider requires canonical pruning proof profile v2")
        search = module.finite_search_build_info()
        if search.get("proof_profile") != "rexgraph.finite-projection-proof.v2":
            raise NotImplementedError("native provider requires canonical pruning proof profile v2")
        source = field_module.finite_field_build_info()
        if any(not search.get(key) or search[key] != source.get(key)
               for key in ("engine", "engine_version", "engine_headers")):
            raise ValueError("native provider modules use different engine snapshots")
        receipt = dict(search=search, source=source,
                       modules={m.__name__: sha256(Path(m.__file__).read_bytes()).hexdigest()
                                for m in (module, field_module)})
        return receipt, NativeFiniteSource, NativeFiniteFamily

    def create_task(self, record_id, *, source, projection, claims, dependencies=(),
                    family_limits=_FAMILY_LIMITS, max_checks=64000000,
                    max_proof_bytes=33554432, expected_version=0):
        """Save an explicit finite view and named grade/cell/symbol conjunctions.

        Windows and fixed assignments are task assumptions, retained in the
        proof. An empty image never establishes a claim vacuously. Every claim
        participates in publication; there is no caller-selected easier subset.
        """
        from rcql import Result
        source = copy_record_pin(source)
        dependencies = tuple(copy_record_pin(pin) for pin in dependencies)
        self._read(source, current=True)
        for pin in dependencies:
            self._read(pin, current=True)
        spec = dict(profile=_PROFILE, source=source, projection=projection, claims=tuple(claims),
                    dependencies=dependencies, family_limits=tuple(family_limits),
                    max_checks=max_checks, max_proof_bytes=max_proof_bytes)
        _validate_task(spec)
        # Canonical serialization detaches mutable caller declarations.
        packet = Result.from_bytes(_result(spec, "task").to_bytes())
        self.session.save_result(record_id, packet, expected_version=expected_version)
        return self.pin(record_id)

    def create_question(self, record_id, *, source, answer_cells, observation_cells=(), key, revision=1,
                        dependencies=(), family_limits=_FAMILY_LIMITS, max_checks=64000000,
                        max_proof_bytes=33554432, expected_version=0):
        """Declare an answer projection and optional observable clarification map.

        No answer values are supplied. All constraints come from the selected
        native source; a question cannot clip carriers or fix its own answer.
        A unique result is exact relative to this source, not a claim that the
        model class includes every external interpretation.
        """
        from rcql import Result
        source = copy_record_pin(source)
        dependencies = tuple(copy_record_pin(ref) for ref in dependencies)
        self._read(source, current=True)
        for ref in dependencies:
            self._read(ref, current=True)
        q = _question_cells(answer_cells, nonempty=True)
        m = _question_cells(observation_cells)
        projection = dict(key=key, revision=revision, cells=tuple(dict.fromkeys((*q, *m))),
                          window=None, fixed=(), distinct_relations=())
        spec = dict(profile=_QUESTION, source=source, projection=projection, answer_cells=q, observation_cells=m,
                    dependencies=dependencies, family_limits=tuple(family_limits), max_checks=max_checks,
                    max_proof_bytes=max_proof_bytes)
        _validate_task(spec)
        self.session.save_result(record_id, Result.from_bytes(_result(spec, "task").to_bytes()),
                                 expected_version=expected_version)
        return self.pin(record_id)

    def _task(self, pin):
        spec = self._value(pin, "task", current=True)
        _validate_task(spec)
        self._current(spec["source"])
        for dependency in spec["dependencies"]:
            self._current(dependency)
        return spec

    def _execute(self, task, cancellation):
        _pin_key(task)
        task = dict(task)
        if not self._available:
            raise RuntimeError("finite verification provider is unavailable")
        provider, source_type, family_type = self._provider()
        epoch = self._epoch
        spec = self._task(task)
        archive = self._value(spec["source"], "source", current=True)
        if not isinstance(archive, bytes):
            raise TypeError("finite source record requires native source archive bytes")
        source = source_type.from_archive(archive)
        family = family_type.projection(source, spec["projection"])
        execution = family.materialize(family_limits=spec["family_limits"], cancellation=cancellation)
        image = execution["image"]
        question = spec["profile"] == _QUESTION
        analysis = None
        if image is None:
            verdicts = () if question else tuple(dict(status="incomplete", satisfying=0, refuting=0) for _ in spec["claims"])
            if question:
                analysis = dict(status="incomplete", answer=None, answer_count=None,
                    observation_count=None, max_remaining_answers=None, observation_determines_answer=None,
                    obstruction=None, checks=0, retained_values=0)
            native_proof = execution["queries"].proof_archive(max_bytes=spec["max_proof_bytes"])
        else:
            if question:
                if not callable(getattr(image, "analyze_question", None)):
                    raise NotImplementedError("native provider lacks complete question/refinement analysis")
                analysis = image.analyze_question(spec["answer_cells"], spec["observation_cells"],
                    max_checks=spec["max_checks"], cancellation=cancellation)
                verdicts = ()
            else:
                verdicts = image.classify_equalities(tuple(c["terms"] for c in spec["claims"]), max_checks=spec["max_checks"])
            native_proof = image.proof_archive(max_bytes=spec["max_proof_bytes"])
        inputs = (spec["source"], *spec["dependencies"])
        proof = dict(profile=spec["profile"], task=task, inputs=inputs, provider=provider, epoch=epoch,
                     status=execution["status"], verdicts=verdicts, work=execution["work"],
                     image_rows=None if image is None else image.inspect()["row_count"],
                     native_proof=native_proof, native_sha256=sha256(native_proof).hexdigest())
        if question:
            proof["question"] = analysis
        return proof, execution

    def _save(self, proof, execution, proof_id, expected_version):
        record = self.session.save_result(proof_id, _result(proof, "proof"), expected_version=expected_version)
        pin = (self.pin(proof_id) if record.envelope is None else
               dict(store_id=self.session.store.store_id, record_id=record.id,
                    version=record.version, state_digest=record.envelope.object_digest))
        grant = {key: proof[key] for key in ("task", "inputs", "provider", "epoch", "status", "verdicts", "image_rows")}
        if "question" in proof:
            grant["question"] = proof["question"]
        self._issued[_pin_key(pin)] = (grant, execution)
        return pin

    def verify(self, task, *, proof_id, expected_version=0, cancellation=None):
        """Run native execution and save its complete proof once, returning its pin."""
        proof, execution = self._execute(task, cancellation)
        return self._save(proof, execution, proof_id, expected_version)

    def replay(self, old_proof, *, proof_id, expected_version=0, cancellation=None):
        """Recompute a saved recipe and issue fresh authority only after exact comparison."""
        previous = self._value(old_proof, "proof")
        proof, execution = self._execute(previous["task"], cancellation)
        for key in ("profile", "provider", "status", "image_rows", "verdicts", "work", "native_sha256", "native_proof"):
            if proof[key] != previous[key]:
                raise ValueError(f"finite proof replay differs in {key}")
        if proof.get("question") != previous.get("question"):
            raise ValueError("finite question replay differs")
        return self._save(proof, execution, proof_id, expected_version)

    def assess(self, pin):
        """Return a strict bool/None Hive verdict without embedding the proof payload."""
        from rcql import Result
        accepted, status, issued = None, "unissued", None
        try:
            pin = copy_record_pin(pin)
            issued = self._issued.get(_pin_key(pin))
            if issued is not None:
                proof, _ = issued
                self._current(pin)
                if not self._available or proof["epoch"] != self._epoch:
                    status = "provider_unavailable"
                elif self._provider()[0] != proof["provider"]:
                    status = "provider_changed"
                else:
                    self._current(proof["task"])
                    for dependency in proof["inputs"]:
                        self._current(dependency)
                    statuses = tuple(v["status"] for v in proof["verdicts"])
                    if "question" in proof:
                        status = proof["question"]["status"]
                        accepted = True if status == "unique" else False if status == "inconsistent_family" else None
                        if status in ("budget_exceeded", "cancelled"):
                            status = "incomplete"
                    elif proof["image_rows"] == 0:
                        accepted, status = False, "inconsistent_family"
                    elif not statuses:
                        status = "unsupported"
                    elif all(s == "entailed" for s in statuses):
                        accepted, status = True, "entailed"
                    elif any(s in ("refuted", "inconsistent_family") for s in statuses):
                        accepted = False
                        status = "inconsistent_family" if "inconsistent_family" in statuses else "refuted"
                    else:
                        status = "incomplete" if "incomplete" in statuses else (
                            "unsupported" if "unsupported" in statuses else "alternatives")
        except (ValueError, KeyError, OSError, ImportError, NotImplementedError):
            accepted, status = None, "stale_or_unavailable"
        if issued is not None and "question" in issued[0]:
            report = dict(issued[0]["question"])
            answer = report["answer"] if accepted is True else None
            report["answer"] = answer
            return Result((accepted, status, pin, answer, report), aliases=("accepted", "status", "proof", "answer", "question"))
        return Result((accepted, status, pin), aliases=("accepted", "status", "proof"))

    def proof_owner(self, pin):
        """Inspect whole native intermediary owners from a current execution."""
        pin = copy_record_pin(pin)
        if self.assess(pin).named_values["status"] in {
                "unissued", "provider_unavailable", "provider_changed", "stale_or_unavailable"}:
            raise ValueError("proof has no current native execution")
        return self._issued[_pin_key(pin)][1]

    def publish(self, pin, record_id, *, expected_version=0):
        """Recheck actual evidence inside the expected-version publication worker."""
        pin = copy_record_pin(pin)
        verdict = self.assess(pin)
        if verdict.named_values["accepted"] is not True:
            raise ValueError(f"finite proof cannot publish: {verdict.named_values['status']}")
        proof = self._issued[_pin_key(pin)][0]
        value = dict(proof=pin, task=proof["task"], status="entailed")
        if "question" in proof:
            value.update(status="source_unique", answer=proof["question"]["answer"], scope="declared-source")
        self.session.save_result(record_id, _result(value, "answer"), expected_version=expected_version)
        return self.pin(record_id)

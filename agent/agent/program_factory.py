"""Owning finite program alternatives, explicit interpretation and Hive execution."""
from hashlib import sha256
from inspect import getsourcefile, isfunction, ismethod
from pathlib import Path

from .family_verification import FamilyVerifier
from .record_access import PinnedRecords, copy_record_pin, record_pin_key
from .relational import _integer


def _result(value, alias):
    from rcql import Result
    return Result((value,), aliases=(alias,))


class FiniteProgramFactory:
    """Interpret actual native sections into prospective executable programs.

    The trusted local operator supplies an explicit forward/reverse map and its
    pinned dependencies. Selection retains the complete unresolved native family;
    an admitted row is not a uniquely learned answer. Saved records grant no live
    authority until native replay and interpretation checks succeed again.
    Groups in distinct_sources require separate actual owners at registration.
    """

    def __init__(self, verifier, *, interpretation, version, assemble, reverse, dependencies=(),
                 helpers=(), distinct_sources=(), max_records=4096):
        if not isinstance(verifier, FamilyVerifier):
            raise TypeError("a finite program factory requires a FamilyVerifier")
        if not isinstance(interpretation, str) or not interpretation:
            raise ValueError("provide the declared interpretation identity")
        _integer(version, "interpretation version", 1)
        _integer(max_records, "complete witness admission", 1)
        self.verifier, self.session = verifier, verifier.session
        self.records = PinnedRecords(self.session)
        self._name, self._version = interpretation, version
        self._assemble, self._reverse = assemble, reverse
        self._helpers = tuple(helpers)
        self._dependencies = tuple(copy_record_pin(p) for p in dependencies)
        if not isinstance(distinct_sources, (tuple, list)):
            raise TypeError("distinct source groups require explicit port sequences")
        groups = []
        for group in distinct_sources:
            if (not isinstance(group, (tuple, list)) or len(group) < 2
                    or any(not isinstance(name, str) or not name for name in group)
                    or len(set(group)) != len(group)):
                raise ValueError("distinct source groups require at least two different port names")
            groups.append(tuple(group))
        self._distinct_sources = tuple(groups)
        self._maximum = max_records
        self._live, self._workers, self._executions = {}, {}, {}
        self._callbacks = self._callback_tokens()

    def _callback_tokens(self):
        tokens = []
        for callback in (self._assemble, self._reverse, *self._helpers):
            if not (isfunction(callback) or ismethod(callback)) or getsourcefile(callback) is None:
                raise TypeError("interpretation maps require source-backed Python functions")
            function = getattr(callback, "__func__", callback)
            tokens.append((callback, function.__code__))
        return tuple(tokens)

    def _interpretation(self):
        if self._callback_tokens() != self._callbacks:
            raise ValueError("registered program interpretation changed")
        callbacks = []
        for callback in (self._assemble, self._reverse, *self._helpers):
            path = Path(getsourcefile(callback))
            callbacks.append(dict(module=callback.__module__, name=callback.__qualname__,
                                  sha256=sha256(path.read_bytes()).hexdigest()))
        for pin in self._dependencies:
            self.records.current(pin)
        from . import hive, relational
        from rcql import program, executor, result_codec
        implementations = {module.__name__: sha256(Path(module.__file__).read_bytes()).hexdigest()
                           for module in (hive, relational, program, executor, result_codec)}
        implementations[__name__] = sha256(Path(__file__).read_bytes()).hexdigest()
        return dict(profile="agent.finite-program-interpretation.v1", name=self._name,
                    version=self._version, callbacks=tuple(callbacks),
                    implementation=implementations,
                    distinct_sources=self._distinct_sources,
                    dependencies=tuple(dict(p) for p in self._dependencies))

    def _section(self, proof, row):
        from rcql import Program
        _integer(row, "program row")
        execution = self.verifier.proof_owner(proof)
        image = execution["image"]
        if execution["status"] != "complete" or image is None:
            raise ValueError("program selection requires a complete native image")
        field = image.row(row)
        if field.inspect()["leaf_count"] != 1:
            raise ValueError("program interpretation requires one explicit source")
        witness = field.witness(count=field.field_size(), max_records=self._maximum)
        source = field.family().source()
        program = self._assemble(witness, source)
        if not isinstance(program, Program):
            raise TypeError("program interpretation must return a canonical Program")
        recovered = tuple(tuple(v) for v in self._reverse(program, source))
        if recovered != witness:
            raise ValueError("program reverse map differs from the complete native section")
        restored = Program.from_bytes(program.to_bytes())
        if tuple(tuple(v) for v in self._reverse(restored, source)) != witness:
            raise ValueError("canonical Program bytes changed the admitted image")
        return program, witness

    def select(self, proof, row, *, program_id, selection_id):
        """Publish an explicit prospective choice from the actual complete family."""
        proof = copy_record_pin(proof)
        interpretation = self._interpretation()
        program, witness = self._section(proof, row)
        evidence = self.records.value(proof, "proof", current=True)
        self.session.publish_program(program_id, program, expected_version=0)
        program_pin = self.records.pin(program_id)
        value = dict(profile="agent.finite-program-selection.v1", proof=proof, row=row,
                     native_sha256=evidence["native_sha256"], provider=evidence["provider"],
                     interpretation=interpretation, witness=witness, program=program_pin,
                     program_digest=program.coefficient_digest, choice="prospective_admitted_section")
        self.session.save_result(selection_id, _result(value, "program_selection"))
        pin = self.records.pin(selection_id)
        self._live[record_pin_key(pin)] = proof
        self._check(pin)
        return pin

    def _check(self, selection):
        from rcql import Program
        selection = copy_record_pin(selection)
        value = self.records.value(selection, "program_selection", current=True)
        if value["profile"] != "agent.finite-program-selection.v1" or value["choice"] != "prospective_admitted_section":
            raise ValueError("unsupported native program selection")
        if value["interpretation"] != self._interpretation():
            raise ValueError("program interpretation or its dependencies changed")
        proof = self._live.get(record_pin_key(selection))
        if proof is None:
            raise ValueError("program selection needs fresh native replay")
        self.records.current(value["proof"])
        evidence = self.records.value(proof, "proof", current=True)
        if (evidence["native_sha256"], evidence["provider"]) != (value["native_sha256"], value["provider"]):
            raise ValueError("program uses another native family or provider")
        program, witness = self._section(proof, value["row"])
        stored = Program.from_record(self.records.read(value["program"], current=True))
        if (witness != value["witness"] or program.coefficient_digest != value["program_digest"]
                or stored.to_bytes() != program.to_bytes()):
            raise ValueError("program record differs from its actual native section")
        return value

    def restore(self, selection, *, proof_id=None, proof=None):
        """Recompute the native family before adopting a saved program selection."""
        selection = copy_record_pin(selection)
        value = self.records.value(selection, "program_selection", current=True)
        if value["interpretation"] != self._interpretation():
            raise ValueError("program interpretation or its dependencies changed")
        if (proof_id is None) == (proof is None):
            raise ValueError("provide either a fresh proof reference or a replay record identity")
        fresh = (self.verifier.replay(value["proof"], proof_id=proof_id) if proof is None
                 else copy_record_pin(proof))
        key = record_pin_key(selection)
        previous = self._live.get(key)
        self._live[key] = fresh
        try:
            self._check(selection)
        except Exception:
            if previous is None:
                self._live.pop(key, None)
            else:
                self._live[key] = previous
            if proof_id is not None:
                self.verifier.release(fresh)
            raise
        return selection

    def assess(self, selection):
        try:
            self._check(selection)
            accepted, status = True, "admitted_program"
        except (ValueError, KeyError, OSError, ImportError, NotImplementedError):
            accepted, status = None, "unissued_or_stale"
        from rcql import Result
        return Result((accepted, status), aliases=("accepted", "status"))

    def register(self, hive, name, selection, *, bundle, source_outputs, parameter_outputs):
        """Expose the selected canonical Program through the existing Hive path."""
        selection, bundle = copy_record_pin(selection), copy_record_pin(bundle)
        value = self._check(selection)
        self.records.current(bundle)
        self._check_source_interface(bundle, source_outputs)
        bee = hive.add_program_bundle(name, self.session, value["program"]["record_id"],
            version=value["program"]["version"], bundle=bundle,
            source_outputs=source_outputs, parameter_outputs=parameter_outputs)
        self._workers[name] = dict(hive=hive, bee=bee, selection=selection, bundle=bundle,
            registration=(bee._handler, bee._context, bee.capability, bee.worker_type, bee.model))
        return bee

    def _check_source_interface(self, bundle, source_outputs):
        """Check declared source inequalities on the actual owning bundle ports."""
        if not self._distinct_sources:
            return
        from rcql import Result
        value = self.records.read(bundle, current=True)
        if not isinstance(value, Result) or not isinstance(source_outputs, dict):
            raise TypeError("source interface requires an owning Result and explicit output ports")
        for group in self._distinct_sources:
            owners = []
            for alias in group:
                if alias not in source_outputs:
                    raise ValueError("declared distinct source port is not bound")
                index = _integer(source_outputs[alias], "source output")
                if index >= len(value.values):
                    raise ValueError("source output is outside the complete Result")
                selected = value.values[index]
                for previous in owners:
                    same = selected is previous
                    compare = getattr(selected, "same_owner", None)
                    if not same and type(selected) is type(previous) and callable(compare):
                        same = bool(compare(previous))
                    if same:
                        raise ValueError("declared distinct source ports share an actual owner")
                owners.append(selected)

    def _worker(self, name):
        worker = self._workers[name]
        self._check(worker["selection"])
        self.records.current(worker["bundle"])
        bee = worker["bee"]
        if (worker["hive"]._bees.get(name) is not bee or
                (bee._handler, bee._context, bee.capability, bee.worker_type, bee.model) != worker["registration"]):
            raise ValueError("registered program worker changed")
        return worker

    def execute(self, name, *, result_id, receipt_id):
        """Run the actual Hive worker and retain its whole Result once in RCDB."""
        worker = self._worker(name)
        result = worker["hive"].invoke(name, {})
        self._worker(name)
        self.session.save_result(result_id, result)
        value = dict(profile="agent.finite-program-execution.v1", worker=name,
                     selection=worker["selection"], bundle=worker["bundle"],
                     result=self.records.pin(result_id))
        self.session.save_result(receipt_id, _result(value, "program_execution"))
        pin = self.records.pin(receipt_id)
        self._executions[record_pin_key(pin)] = worker
        return pin

    def assess_execution(self, receipt):
        try:
            value = self.records.value(receipt, "program_execution", current=True)
            worker = self._executions.get(record_pin_key(receipt))
            if worker is None or self._worker(value["worker"]) is not worker:
                raise ValueError("execution needs a fresh registered worker run")
            if value["selection"] != worker["selection"] or value["bundle"] != worker["bundle"]:
                raise ValueError("execution receipt has another selection or input bundle")
            self.records.current(value["result"])
            accepted, status = True, "current_execution"
        except (ValueError, KeyError, OSError, ImportError, NotImplementedError):
            accepted, status = None, "unissued_or_stale"
        from rcql import Result
        return Result((accepted, status), aliases=("accepted", "status"))

    def publish(self, receipt, record_id):
        """Publish only a current locally executed program-result reference."""
        receipt = copy_record_pin(receipt)
        if self.assess_execution(receipt).named_values["accepted"] is not True:
            raise ValueError("program execution cannot publish without current evidence")
        value = self.records.value(receipt, "program_execution", current=True)
        self.session.save_result(record_id, _result(dict(receipt=receipt, result=value["result"]), "program_result"))
        return self.records.pin(record_id)

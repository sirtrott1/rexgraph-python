"""Immutable native action recipes and explicit current evidence requests."""
from hashlib import sha256
from importlib.resources import files

from .value_codec import pack_value, unpack_value

_LIMITS = (256, 256, 65536, 4096, 256, 65536, 1048576, 4096, 16000000, 64000000, 4096)
_MAX_BYTES = 132 << 20
_WRAPPER = sha256(files("rexgraph").joinpath("action_program.py").read_bytes()).hexdigest()


def _backend():
    try:
        from rexgraph_cpp import _action_program
    except ImportError as exc:
        raise ImportError("native action programs require rexgraph_cpp with RexTemporal support") from exc
    implementation = _action_program.action_program_build_info()
    implementation["core_wrapper"] = _WRAPPER
    return _action_program, implementation


def _read(packet, profile, keys):
    if type(packet) is not bytes or len(packet) > _MAX_BYTES:
        raise ValueError("bounded immutable action packet bytes required")
    data = unpack_value(packet)
    if type(data) is not dict or set(data) != keys | {"profile", "implementation"} or data["profile"] != profile:
        raise ValueError("unsupported action packet")
    if data["implementation"] != _backend()[1]:
        raise ValueError("action implementation differs; use its original runtime or an explicit migration")
    if pack_value(data) != packet:
        raise ValueError("noncanonical action packet")
    return data


class _Immutable:
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("use the declared creation or replay method")

    def __setattr__(self, key, value):
        raise AttributeError("action values are immutable")


class ActionEvidence(_Immutable):
    """Own a complete native evidence graph and a declared current request.

    Construction retains the request. Each action execution performs its native
    admission again; neither this packet nor a previous receipt grants admission.
    """
    __slots__ = ("_owner", "_packet")

    @classmethod
    def create(cls, envelope, specification):
        native, implementation = _backend()
        owner = native.NativeActionEvidence.from_envelope(envelope, specification)
        packet = pack_value({"profile": "rexgraph.action-evidence.v1", "implementation": implementation,
                             "envelope": owner.archive(), "specification": specification})
        if len(packet) > _MAX_BYTES:
            raise ValueError("action evidence packet exceeds admission")
        out = object.__new__(cls)
        object.__setattr__(out, "_owner", owner)
        object.__setattr__(out, "_packet", packet)
        return out

    @classmethod
    def from_bytes(cls, packet):
        data = _read(packet, "rexgraph.action-evidence.v1", {"envelope", "specification"})
        return cls.create(data["envelope"], data["specification"])

    def to_bytes(self):
        return self._packet

    def inspect(self):
        data = unpack_value(self._packet)
        return {"envelope_sha256": sha256(data["envelope"]).hexdigest(), "specification": data["specification"]}


class ActionProgram(_Immutable):
    """Own an exact action DAG over one actual native source publication."""
    __slots__ = ("_owner", "_packet")

    @staticmethod
    def description(recipe):
        return _backend()[0].action_description(recipe)

    @classmethod
    def create(cls, source, endpoint, recipe, publication, availability, *, limits=_LIMITS):
        native, implementation = _backend()
        owner = native.NativeActionProgram.prepare(source, endpoint, recipe, publication, availability, limits=limits)
        packet = pack_value({"profile": "rexgraph.action-program.v1", "implementation": implementation,
            "source": source, "endpoint": endpoint, "recipe": recipe, "publication": publication, "limits": tuple(limits)})
        if len(packet) > _MAX_BYTES:
            raise ValueError("action program packet exceeds admission")
        out = object.__new__(cls)
        object.__setattr__(out, "_owner", owner)
        object.__setattr__(out, "_packet", packet)
        return out

    @classmethod
    def from_bytes(cls, packet, *, availability, restore_limits=_LIMITS):
        data = _read(packet, "rexgraph.action-program.v1", {"source", "endpoint", "recipe", "publication", "limits"})
        bounds = data["limits"]
        if (type(bounds) is not tuple or len(bounds) != 11 or len(restore_limits) != 11
                or any(type(v) is not int or v < 0 for v in (*bounds, *restore_limits))
                or any(v > cap for v, cap in zip(bounds, restore_limits, strict=True))):
            raise ValueError("action restore bounds exceed admission")
        return cls.create(data["source"], data["endpoint"], data["recipe"], data["publication"], availability, limits=bounds)

    def to_bytes(self):
        return self._packet

    def inspect(self):
        data = unpack_value(self._packet)
        return {"recipe": data["recipe"], "publication": data["publication"], "limits": data["limits"],
                "source_sha256": sha256(data["source"]).hexdigest(), "endpoint_sha256": sha256(data["endpoint"]).hexdigest()}

    def execute(self, evidence, availability, *, input_selector=None, pairs=(), lease=None):
        if not isinstance(evidence, ActionEvidence):
            raise TypeError("current owning ActionEvidence is required, not a receipt")
        if input_selector is None:
            input_selector = unpack_value(self._packet)["publication"]
        availability = tuple(tuple(v) for v in availability)
        pairs = tuple(tuple(pair) for pair in pairs)
        result = self._owner.execute(evidence._owner, availability, input_selector, pairs=pairs, lease=lease)
        out = object.__new__(ActionOutput)
        for name, value in (("_owner", result), ("_program", self), ("_evidence", evidence),
                            ("_availability", tuple(tuple(v) for v in availability)),
                            ("_input", input_selector), ("_pairs", pairs)):
            object.__setattr__(out, name, value)
        return out


class ActionOutput(_Immutable):
    """Retain complete native component fields, aggregate and signed moments."""
    __slots__ = ("_owner", "_program", "_evidence", "_availability", "_input", "_pairs")

    def inspect(self, *, max_entries=65536):
        return self._owner.inspect(max_entries=max_entries)

    def to_bytes(self, *, max_entries=65536):
        packet = pack_value({"profile": "rexgraph.action-output.v1", "implementation": _backend()[1],
            "program": self._program.to_bytes(), "evidence": self._evidence.to_bytes(),
            "availability": self._availability, "input": self._input, "pairs": self._pairs,
            "snapshot": self.inspect(max_entries=max_entries)})
        if len(packet) > _MAX_BYTES:
            raise ValueError("action output packet exceeds admission")
        return packet

    @classmethod
    def replay(cls, packet, *, evidence, availability, max_entries=65536):
        data = _read(packet, "rexgraph.action-output.v1", {"program", "evidence", "availability", "input", "pairs", "snapshot"})
        if not isinstance(evidence, ActionEvidence):
            raise TypeError("fresh owning evidence request is required for action replay")
        previous = ActionEvidence.from_bytes(data["evidence"])
        if previous._owner.archive() != evidence._owner.archive():
            raise ValueError("action replay requires the retained evidence graph or an explicit transport")
        program = ActionProgram.from_bytes(data["program"], availability=availability)
        result = program.execute(evidence, availability, input_selector=data["input"], pairs=data["pairs"])
        actual = result.inspect(max_entries=max_entries)
        # A fresh request owns its fresh admission; the retained receipt is data.
        for key in ("output", "fields", "components", "interactions", "totals", "matches_reference"):
            if actual[key] != data["snapshot"][key]:
                raise ValueError("stored action fields differ from native semantic replay")
        return result

"""Versioned references to owning C++ source publications."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from rexgraph.coordinate_map import _identity
from rexgraph.tensor_field import FieldSource


def _uint(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 1 << 64:
        raise ValueError("native publication keys require unsigned 64 bit integers")
    return int(value)


def _sha(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("native publication digest requires SHA256")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("native publication digest is not hexadecimal") from exc
    if decoded.hex() != value:
        raise ValueError("native publication digest requires canonical hexadecimal")
    return value


@dataclass(frozen=True)
class CppSourceIdentity:
    """Native namespace, retained cutoff, exact archive and ordered primary keys."""
    namespace: int
    sequence: int
    first_sequence: int
    archive_digest: str
    endpoint_digest: str
    coordinates: tuple[tuple[str, ...], ...]
    codec: str = "rfcxs-v1"

    def __post_init__(self):
        for name in ("namespace", "sequence", "first_sequence"):
            object.__setattr__(self, name, _uint(getattr(self, name)))
        if self.first_sequence > self.sequence or self.codec != "rfcxs-v1":
            raise ValueError("invalid native cutoff or archive codec")
        _sha(self.archive_digest)
        _sha(self.endpoint_digest)
        if isinstance(self.coordinates, (str, bytes)):
            raise TypeError("native coordinates require one ordered sequence per grade")
        coordinates = []
        for keys in self.coordinates:
            if isinstance(keys, (str, bytes)):
                raise TypeError("native grade coordinates require a sequence")
            keys = tuple(keys)
            if (any(not isinstance(key, str) or not key or len(key) > 20
                    or not key.isascii() or not key.isdecimal() or str(int(key)) != key
                    or int(key) >= 1 << 64 for key in keys) or len(set(keys)) != len(keys)):
                raise ValueError("native primary coordinates must be distinct declared keys")
            coordinates.append(keys)
        if not coordinates:
            raise ValueError("native source requires its retained grades")
        original = self.coordinates
        retained = (original if type(original) is tuple and
                    all(a is b for a, b in zip(coordinates, original, strict=True)) else tuple(coordinates))
        object.__setattr__(self, "coordinates", retained)

    def as_record(self):
        return {"namespace": self.namespace, "sequence": self.sequence,
                "first_sequence": self.first_sequence, "codec": self.codec,
                "archive_digest": self.archive_digest, "endpoint_digest": self.endpoint_digest,
                "coordinates": self.coordinates}

    @classmethod
    def from_record(cls, record):
        from collections.abc import Mapping
        names = {"namespace", "sequence", "first_sequence", "codec", "archive_digest",
                 "endpoint_digest", "coordinates"}
        if not isinstance(record, Mapping) or set(record) != names:
            raise ValueError("invalid native source identity declaration")
        return cls(**record)


def _source_publication(source, declared=None):
    from rexgraph_cpp import NativeSource
    record = source.identity
    if type(source) is NativeSource and type(declared) is CppSourceIdentity:
        known = declared.as_record()
        if (declared.coordinates is record["coordinates"] and
                all(type(value) is type(record[key]) and value == record[key]
                    for key, value in known.items())):
            return declared
    return CppSourceIdentity.from_record(record)


@dataclass(frozen=True, eq=False)
class CppSourceReference(FieldSource):
    """A durable native publication reference with an optional owning handle."""
    native: CppSourceIdentity | None = None

    def __post_init__(self):
        native = self.native
        if self.source is not None:
            from rexgraph_cpp import NativeSource
            if not isinstance(self.source, NativeSource):
                raise TypeError("C++ source binding requires an owning NativeSource")
            current = _source_publication(self.source, native)
            if native is not None and native != current:
                raise ValueError("C++ source differs from the declared publication")
            native = current
        if not isinstance(native, CppSourceIdentity):
            raise TypeError("detached C++ source requires its publication identity")
        if self.state_digest is not None and self.state_digest != native.archive_digest:
            raise ValueError("native archive digest differs from its source reference")
        validated = FieldSource(None, self.record_id, self.version, native.archive_digest)
        object.__setattr__(self, "native", native)
        object.__setattr__(self, "version", validated.version)
        object.__setattr__(self, "state_digest", validated.state_digest)

    @property
    def coefficient_digest(self):
        n = self.native
        return _identity(("cpp_source_reference_v1", self.record_id, self.version,
                          n.namespace, n.sequence, n.first_sequence, n.codec,
                          n.archive_digest, n.endpoint_digest, n.coordinates))

    def check(self):
        if self.source is not None and _source_publication(self.source, self.native) != self.native:
            raise ValueError("native publication identity changed")

    def bind(self, source):
        return CppSourceReference(source, self.record_id, self.version, self.state_digest, self.native)

    def detached(self):
        return CppSourceReference(None, self.record_id, self.version, self.state_digest, self.native)

    def as_record(self):
        return {"kind": "cpp_source", "schema_version": 1, "record_id": self.record_id,
                "version": self.version, "state_digest": self.state_digest, "native": self.native.as_record()}


def source_from_record(record, *, allow_cpp=False):
    """Decode a legacy source or a declared C++ source reference."""
    from collections.abc import Mapping
    if not isinstance(record, Mapping):
        raise ValueError("invalid source reference record")
    if "kind" not in record:
        if set(record) != {"record_id", "version", "state_digest"}:
            raise ValueError("invalid legacy source reference")
        return FieldSource(None, record["record_id"], record["version"], record["state_digest"])
    if (not allow_cpp or set(record) != {"kind", "schema_version", "record_id", "version", "state_digest", "native"}
            or record["kind"] not in {"cpp_source", "cpp_algebraic_source", "cpp_algebraic_graph_source"} or type(record["schema_version"]) is not int
            or record["schema_version"] != 1):
        raise ValueError("unsupported source reference schema")
    if record["kind"] in {"cpp_algebraic_source", "cpp_algebraic_graph_source"}:
        native = CppAlgebraicIdentity.from_record(record["native"])
        if (record["kind"] == "cpp_algebraic_graph_source") != (native.graph_digest is not None):
            raise ValueError("algebraic source kind differs from its archive codec")
        return CppAlgebraicSourceReference(None, record["record_id"], record["version"], record["state_digest"],
                                           native)
    return CppSourceReference(None, record["record_id"], record["version"], record["state_digest"],
                              CppSourceIdentity.from_record(record["native"]))


@dataclass(frozen=True)
class CppAlgebraicIdentity:
    """Exact archive/source identities, retained grades and polynomial coefficient basis."""
    archive_digest: str
    source_digest: str
    source_sizes: tuple[int, ...]
    polynomial: tuple
    codec: str = "rfcxa-v1"
    graph_digest: str | None = None
    archive_key: int | None = None

    def __post_init__(self):
        from rexgraph.exact_value import exact_fraction
        _sha(self.archive_digest); _sha(self.source_digest)
        if self.codec not in {"rfcxa-v1", "rfcxg-archive-v1"} or isinstance(self.source_sizes, (str, bytes)):
            raise ValueError("invalid algebraic source codec or grades")
        if self.codec == "rfcxg-archive-v1":
            _sha(self.graph_digest)
            key = _uint(self.archive_key)
            if key == 0:
                raise ValueError("algebraic graph archive key must be nonzero")
            object.__setattr__(self, "archive_key", key)
        elif self.graph_digest is not None or self.archive_key is not None:
            raise ValueError("standalone algebraic archive has no graph reference")
        sizes = tuple(_uint(v) for v in self.source_sizes)
        if not sizes:
            raise ValueError("algebraic source requires all retained grades")
        if isinstance(self.polynomial, (str, bytes)):
            raise TypeError("algebraic coefficient basis requires an exact polynomial")
        polynomial = tuple(exact_fraction(v) for v in self.polynomial)
        if len(polynomial) < 2 or polynomial[-1] != 1:
            raise ValueError("algebraic coefficient basis requires a monic nonconstant polynomial")
        object.__setattr__(self, "source_sizes", sizes)
        object.__setattr__(self, "polynomial", polynomial)

    @property
    def field_degree(self):
        return len(self.polynomial) - 1

    @property
    def coordinate_scope(self):
        return self.archive_digest if self.graph_digest is None else f"{self.graph_digest}:archive:{self.archive_key}"

    def as_record(self):
        result = {"archive_digest": self.archive_digest, "source_digest": self.source_digest,
                "source_sizes": self.source_sizes, "codec": self.codec,
                "polynomial": tuple((format(q.numerator, "x"), format(q.denominator, "x")) for q in self.polynomial)}
        if self.graph_digest is not None:
            result.update(graph_digest=self.graph_digest, archive_key=self.archive_key)
        return result

    @classmethod
    def from_record(cls, record):
        from collections.abc import Mapping
        from fractions import Fraction
        names = {"archive_digest", "source_digest", "source_sizes", "codec", "polynomial"}
        if isinstance(record, Mapping) and record.get("codec") == "rfcxg-archive-v1":
            names |= {"graph_digest", "archive_key"}
        if not isinstance(record, Mapping) or set(record) != names:
            raise ValueError("invalid algebraic source identity declaration")
        try:
            polynomial = tuple(Fraction(int(n, 16), int(d, 16)) for n, d in record["polynomial"])
            result = cls(record["archive_digest"], record["source_digest"], record["source_sizes"], polynomial, record["codec"],
                         record.get("graph_digest"), record.get("archive_key"))
            if result.as_record()["polynomial"] != tuple(tuple(v) for v in record["polynomial"]):
                raise ValueError("algebraic polynomial coordinates require canonical hexadecimal")
            return result
        except (TypeError, ZeroDivisionError) as exc:
            raise ValueError("invalid algebraic polynomial coordinates") from exc


@dataclass(frozen=True, eq=False)
class CppAlgebraicSourceReference(FieldSource):
    """A durable standalone or graph archive reference with an optional immutable native owner."""
    native: CppAlgebraicIdentity | None = None

    def __post_init__(self):
        native = self.native
        if self.source is not None:
            from rexgraph_cpp import NativeAlgebraicArchive
            if type(self.source) is not NativeAlgebraicArchive:
                raise TypeError("algebraic source binding requires an owning NativeAlgebraicArchive")
            m = self.source.manifest
            current = CppAlgebraicIdentity(m["archive_digest"], m["source_digest"], m["source_sizes"], m["minimal_polynomial"], m["codec"],
                                            m.get("graph_digest"), m.get("archive_key"))
            if native is not None and native != current:
                raise ValueError("algebraic source differs from the declared archive")
            native = current
        if not isinstance(native, CppAlgebraicIdentity):
            raise TypeError("detached algebraic source requires its archive identity")
        if self.state_digest is not None and self.state_digest != native.archive_digest:
            raise ValueError("algebraic archive digest differs from its source reference")
        validated = FieldSource(None, self.record_id, self.version, native.archive_digest)
        object.__setattr__(self, "native", native)
        object.__setattr__(self, "version", validated.version)
        object.__setattr__(self, "state_digest", validated.state_digest)

    @property
    def coefficient_digest(self):
        n = self.native
        values = ("cpp_algebraic_source_reference_v1", self.record_id, self.version,
                          n.archive_digest, n.source_digest, n.source_sizes,
                          n.as_record()["polynomial"], n.codec)
        if n.graph_digest is not None:
            values += (n.graph_digest, n.archive_key)
        return _identity(values)

    def check(self):
        if self.source is not None:
            self.bind(self.source)

    def bind(self, source):
        return CppAlgebraicSourceReference(source, self.record_id, self.version, self.state_digest, self.native)

    def detached(self):
        return CppAlgebraicSourceReference(None, self.record_id, self.version, self.state_digest, self.native)

    def as_record(self):
        kind = "cpp_algebraic_source" if self.native.graph_digest is None else "cpp_algebraic_graph_source"
        return {"kind": kind, "schema_version": 1, "record_id": self.record_id,
                "version": self.version, "state_digest": self.state_digest, "native": self.native.as_record()}

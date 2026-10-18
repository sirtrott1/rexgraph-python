# distutils: language=c++
"""Owning native field programs, immutable archives and bounded stalk reads."""
from collections.abc import Mapping
from hashlib import sha256
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._stalk_fields import _natural, _scalar, _values

cdef extern from "field_program_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass ProgramEntry:
        size_t row, column
        string value
    cdef cppclass ProgramMatrix:
        size_t rows, columns
        vector[ProgramEntry] entries
    cdef cppclass ProgramStalk:
        size_t grade, cell
        uint64_t basis
        vector[uint64_t] directions
        ProgramMatrix metric
    cdef cppclass ProgramOccurrence:
        size_t grade, slot
        unsigned role
        ProgramMatrix map
    cdef cppclass ProgramNode:
        uint64_t key, input
        unsigned kind
        size_t grade
        cppbool has_input
        ProgramMatrix map
    cdef cppclass ProgramInput:
        uint64_t node, use
        string field
    cdef cppclass ProgramSourceInfo:
        uint64_t name_space, sequence
        size_t semantic_scale
        string endpoint, archive
        vector[vector[string]] coordinates
    cdef cppclass ProgramStalkInfo:
        size_t grade, cell
        uint64_t basis
        vector[uint64_t] directions
        string metric
    cdef cppclass ProgramOccurrenceInfo:
        size_t grade, slot, upper, lower, dimension
        uint64_t key
        unsigned role
        cppbool active
        string coefficient, map
    cdef cppclass ProgramFieldInfo:
        uint64_t node
        size_t grade, dimension, supported, nonzeros, occurrence_dimension
        cppbool has_occurrences
    cdef cppclass ProgramSelected:
        ProgramStalkInfo stalk
        string field
        vector[string] values
    cdef cppclass FieldProgramResultBridge
    cdef cppclass FieldProgramBridge:
        FieldProgramBridge(const string&, uint64_t, uint64_t, const vector[ProgramStalk]&,
            const vector[ProgramOccurrence]&, const vector[ProgramNode]&, cppbool,
            const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[FieldProgramBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        ProgramSourceInfo source() except +
        uint64_t key() except +
        uint64_t revision() except +
        cppbool differential() except +
        vector[ProgramNode] nodes(size_t) except +
        vector[ProgramStalkInfo] stalks(size_t) except +
        vector[ProgramOccurrenceInfo] occurrences(size_t) except +
        shared_ptr[FieldProgramResultBridge] execute(const vector[ProgramInput]&) except +
    cdef cppclass FieldProgramResultBridge:
        @staticmethod
        shared_ptr[FieldProgramResultBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[FieldProgramBridge] program() except +
        vector[ProgramInput] inputs(size_t) except +
        vector[ProgramFieldInfo] fields(size_t) except +
        string field(uint64_t, cppbool) except +
        vector[ProgramSelected] select(uint64_t, const vector[vector[size_t]]&, size_t) except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FIELD_PROGRAM_BINDING_DIGEST

_LIMITS = (4096, 16000000, 64000000, 16384)
_SCHEMA = "rexgraph.relational-field-program.v1"
_KINDS = {"input": 0, "boundary": 1, "declared_map": 2}
_ROLES = {"realized_incidence": 0, "coefficient_independent": 1}


def field_program_build_info():
    return {"engine": "rexgraph-cpp", "engine_version": RFC_ENGINE_VERSION.decode(),
            "engine_headers": RFC_ENGINE_DIGEST.decode(), "binding": RFC_FIELD_PROGRAM_BINDING_DIGEST.decode(),
            "profile": "owning-relational-field-program-v1", "scalar_policy": "gmp-rational-exact"}


def _closed(value, keys):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError("closed field program declaration required")


cdef vector[size_t] _bounds(value):
    cdef vector[size_t] result
    values = tuple(value)
    if len(values) != 4 or any(not _natural(v) for v in values):
        raise ValueError("four positive field program action bounds required")
    for item in values:
        result.push_back(_natural(item))
    return result


cdef ProgramMatrix _matrix(value, size_t bits):
    _closed(value, ("shape", "entries"))
    cdef ProgramMatrix result
    cdef ProgramEntry entry
    shape = tuple(value["shape"])
    if len(shape) != 2:
        raise ValueError("field program map requires an explicit rectangular shape")
    result.rows = _natural(shape[0]); result.columns = _natural(shape[1])
    for row, column, coefficient in value["entries"]:
        entry.row = _natural(row); entry.column = _natural(column)
        entry.value = _scalar(coefficient, bits)
        result.entries.push_back(entry)
    return result


cdef object _stalk(ProgramStalkInfo value):
    return {"cell": (int(value.grade), int(value.cell)), "basis": int(value.basis),
            "directions": tuple(int(v) for v in value.directions), "metric": <bytes>value.metric}


cdef class PreparedFieldProgram:
    cdef shared_ptr[FieldProgramBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum

    cdef shared_ptr[FieldProgramBridge] _pin(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native field program handle is not initialized")
        return self._handle

    def __init__(self, source_archive, declaration, *, limits=_LIMITS, max_bytes=256 * 1024 * 1024):
        if self._handle.get() != NULL:
            raise TypeError("native field program owners are immutable")
        if type(source_archive) is not bytes:
            raise TypeError("field program source requires immutable canonical archive bytes")
        _closed(declaration, ("profile", "key", "revision", "stalks", "occurrences", "nodes", "differential"))
        if declaration["profile"] != _SCHEMA or type(declaration["differential"]) is not bool:
            raise ValueError("declared field program schema and differential boolean required")
        self._limits = _bounds(limits); self._maximum = _natural(max_bytes)
        cdef vector[ProgramStalk] stalks
        cdef vector[ProgramOccurrence] occurrences
        cdef vector[ProgramNode] nodes
        cdef ProgramStalk stalk
        cdef ProgramOccurrence occurrence
        cdef ProgramNode node
        cdef ProgramMatrix empty
        cdef string source = source_archive
        cdef uint64_t key = _natural(declaration["key"])
        cdef uint64_t revision = _natural(declaration["revision"])
        cdef cppbool differential = declaration["differential"]
        empty.rows = 0; empty.columns = 0
        for row in declaration["stalks"]:
            _closed(row, ("cell", "basis", "directions", "metric"))
            cell = tuple(row["cell"])
            if len(cell) != 2:
                raise ValueError("field program cell requires grade and original index")
            stalk.grade = _natural(cell[0]); stalk.cell = _natural(cell[1]); stalk.basis = _natural(row["basis"])
            stalk.directions.clear()
            for direction in row["directions"]:
                stalk.directions.push_back(_natural(direction))
            stalk.metric = _matrix(row["metric"], self._limits[3]); stalks.push_back(stalk)
        for row in declaration["occurrences"]:
            _closed(row, ("grade", "slot", "role", "map"))
            if row["role"] not in _ROLES:
                raise ValueError("unsupported field program occurrence map role")
            occurrence.grade = _natural(row["grade"]); occurrence.slot = _natural(row["slot"])
            occurrence.role = _ROLES[row["role"]]; occurrence.map = _matrix(row["map"], self._limits[3])
            occurrences.push_back(occurrence)
        for row in declaration["nodes"]:
            if not isinstance(row, Mapping) or row.get("kind") not in _KINDS:
                raise ValueError("unsupported field program node kind")
            required = ("key", "kind", "grade")
            if row["kind"] != "input": required += ("input",)
            if row["kind"] == "declared_map": required += ("map",)
            _closed(row, required)
            node.key = _natural(row["key"]); node.kind = _KINDS[row["kind"]]; node.grade = _natural(row["grade"])
            node.has_input = row["kind"] != "input"; node.input = _natural(row["input"]) if node.has_input else 0
            node.map = _matrix(row["map"], self._limits[3]) if row["kind"] == "declared_map" else empty
            nodes.push_back(node)
        cdef shared_ptr[FieldProgramBridge] owner
        with nogil:
            owner.reset(new FieldProgramBridge(source, key, revision, stalks, occurrences, nodes,
                                               differential, self._limits, self._maximum))
        if self._handle.get() != NULL:
            raise TypeError("native field program owners are immutable")
        self._handle = owner

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256 * 1024 * 1024):
        if type(archive) is not bytes:
            raise TypeError("field program requires immutable native program bytes")
        cdef PreparedFieldProgram result = cls.__new__(cls)
        result._limits = _bounds(limits); result._maximum = _natural(max_bytes)
        cdef string data = archive
        with nogil:
            result._handle = FieldProgramBridge.restore(data, result._limits, result._maximum)
        return result

    def archive(self):
        cdef shared_ptr[FieldProgramBridge] pin = self._pin()
        cdef string data
        with nogil: data = pin.get().archive()
        return <bytes>data

    def identity(self):
        cdef shared_ptr[FieldProgramBridge] pin = self._pin()
        cdef uint64_t key, revision
        cdef cppbool differential
        with nogil:
            key = pin.get().key(); revision = pin.get().revision(); differential = pin.get().differential()
        return {"key": int(key), "revision": int(revision), "differential": bool(differential)}

    def publication(self):
        cdef shared_ptr[FieldProgramBridge] pin = self._pin()
        cdef ProgramSourceInfo value
        with nogil: value = pin.get().source()
        return {"namespace": int(value.name_space), "sequence": int(value.sequence),
                "endpoint_digest": sha256(<bytes>value.endpoint).hexdigest(),
                "coordinates": tuple(tuple(word.decode() for word in row) for row in value.coordinates),
                "semantic_scale": int(value.semantic_scale), "snapshot_archive": <bytes>value.archive}

    def inspect(self, *, max_records=256):
        cdef shared_ptr[FieldProgramBridge] pin = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[ProgramStalkInfo] stalks
        cdef vector[ProgramOccurrenceInfo] occurrences
        cdef vector[ProgramNode] nodes
        cdef uint64_t key, revision
        cdef cppbool differential
        with nogil:
            stalks = pin.get().stalks(maximum); occurrences = pin.get().occurrences(maximum)
            nodes = pin.get().nodes(maximum); key = pin.get().key()
            revision = pin.get().revision(); differential = pin.get().differential()
        return {"key": int(key), "revision": int(revision), "differential": bool(differential),
                "stalks": tuple(_stalk(value) for value in stalks),
                "occurrences": tuple({"grade": int(value.grade), "slot": int(value.slot), "key": int(value.key),
                    "upper": int(value.upper), "lower": int(value.lower), "active": bool(value.active),
                    "dimension": int(value.dimension), "role": "realized_incidence" if value.role == 0 else "coefficient_independent",
                    "coefficient": _values((<bytes>value.coefficient,))[0], "map": <bytes>value.map} for value in occurrences),
                "nodes": tuple({"key": int(value.key), "kind": ("input", "boundary", "declared_map")[value.kind],
                                "grade": int(value.grade), "input": int(value.input) if value.has_input else None} for value in nodes)}

    def execute(self, inputs):
        cdef shared_ptr[FieldProgramBridge] pin = self._pin()
        cdef vector[ProgramInput] values
        cdef ProgramInput row
        for value in inputs:
            _closed(value, ("node", "use", "field"))
            if type(value["field"]) is not bytes:
                raise TypeError("field program input requires immutable encoded sparse tensor bytes")
            row.node = _natural(value["node"]); row.use = _natural(value["use"]); row.field = value["field"]
            values.push_back(row)
        cdef NativeFieldProgramResult result = NativeFieldProgramResult.__new__(NativeFieldProgramResult)
        result._limits = self._limits; result._maximum = self._maximum
        with nogil: result._handle = pin.get().execute(values)
        return result


cdef class NativeFieldProgramResult:
    cdef shared_ptr[FieldProgramResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum

    cdef shared_ptr[FieldProgramResultBridge] _pin(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native field program result handle is not initialized")
        return self._handle

    def __init__(self):
        raise TypeError("field program results are produced by execution or native archive replay")

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256 * 1024 * 1024):
        if type(archive) is not bytes:
            raise TypeError("field program result requires immutable native bytes")
        cdef NativeFieldProgramResult result = cls.__new__(cls)
        result._limits = _bounds(limits); result._maximum = _natural(max_bytes)
        cdef string data = archive
        with nogil: result._handle = FieldProgramResultBridge.restore(data, result._limits, result._maximum)
        return result

    def archive(self):
        cdef shared_ptr[FieldProgramResultBridge] pin = self._pin()
        cdef string data
        with nogil: data = pin.get().archive()
        return <bytes>data

    def program(self):
        cdef shared_ptr[FieldProgramResultBridge] pin = self._pin()
        cdef PreparedFieldProgram result = PreparedFieldProgram.__new__(PreparedFieldProgram)
        result._limits = self._limits; result._maximum = self._maximum
        with nogil: result._handle = pin.get().program()
        return result

    def fields(self, *, max_records=256):
        cdef shared_ptr[FieldProgramResultBridge] pin = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[ProgramFieldInfo] records
        with nogil: records = pin.get().fields(maximum)
        return tuple({"node": int(v.node), "grade": int(v.grade), "dimension": int(v.dimension),
                      "supported": int(v.supported), "nonzeros": int(v.nonzeros),
                      "occurrence_dimension": int(v.occurrence_dimension) if v.has_occurrences else None} for v in records)

    def input_uses(self, *, max_records=256):
        cdef shared_ptr[FieldProgramResultBridge] pin = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[ProgramInput] records
        with nogil: records = pin.get().inputs(maximum)
        return tuple({"node": int(v.node), "use": int(v.use), "field": <bytes>v.field} for v in records)

    def field_archive(self, node, *, occurrences=False):
        cdef shared_ptr[FieldProgramResultBridge] pin = self._pin()
        if type(occurrences) is not bool:
            raise TypeError("original occurrence selection must be boolean")
        cdef uint64_t key = _natural(node)
        cdef cppbool original = occurrences
        cdef string data
        with nogil: data = pin.get().field(key, original)
        return <bytes>data

    def select(self, node, cells, *, coordinate_limit=256):
        cdef shared_ptr[FieldProgramResultBridge] pin = self._pin()
        cdef uint64_t key = _natural(node)
        cdef size_t maximum = _natural(coordinate_limit)
        cdef vector[vector[size_t]] requested
        cdef vector[size_t] cell
        cdef vector[ProgramSelected] records
        for value in cells:
            value = tuple(value)
            if len(value) != 2:
                raise ValueError("field program cell requires grade and original index")
            cell.clear(); cell.push_back(_natural(value[0])); cell.push_back(_natural(value[1])); requested.push_back(cell)
        with nogil: records = pin.get().select(key, requested, maximum)
        return tuple(dict(_stalk(value.stalk), field=<bytes>value.field,
                          values=_values(tuple(<bytes>q for q in value.values))) for value in records)

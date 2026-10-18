# distutils: language=c++
"""Exact scalar channel actions on retained native relations."""
from fractions import Fraction
from numbers import Integral
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t

cdef extern from "channel_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass ScalarChannelResult:
        vector[vector[string]] fields
        vector[string] energies
        size_t steps
    cdef cppclass ScalarChannels:
        ScalarChannels(const string&, const vector[string]&, bint,
                       const vector[size_t]&, size_t) except +
        size_t relations() except +
        size_t boundaries() except +
        size_t occurrences() except +
        size_t retained_entries() except +
        size_t preparation_steps() except +
        vector[string] incidence() except +
        vector[vector[size_t]] occurrence_cells() except +
        vector[uint64_t] occurrence_keys() except +
        vector[size_t] occurrence_activity() except +
        vector[string] traces() except +
        vector[vector[string]] diagonals() except +
        ScalarChannelResult apply(const vector[string]&, bint) except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_CHANNEL_BINDING_DIGEST

def channels_build_info():
    return {"engine": "rexgraph-cpp", "engine_version": RFC_ENGINE_VERSION.decode(),
            "engine_headers": RFC_ENGINE_DIGEST.decode(), "binding": RFC_CHANNEL_BINDING_DIGEST.decode(),
            "profile": "owning-rational-relational-channels-v1"}

def _natural(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < (1 << 64):
        raise ValueError("relational bound requires an unsigned integer")
    return int(value)

def _scalar(value, integer_bits):
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
        raise TypeError("relational scalar requires an integer or Fraction")
    q = Fraction(int(value)) if isinstance(value, Integral) else Fraction(value)
    if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > integer_bits:
        raise ValueError("relational scalar exceeds integer bit admission")
    return (format(q.numerator, "x") + "/" + format(q.denominator, "x")).encode()

def _values(encoded):
    result = []
    for value in encoded:
        parts = value.decode("ascii").split("/")
        result.append(Fraction(int(parts[0], 16), int(parts[1], 16) if len(parts) == 2 else 1))
    return tuple(result)

cdef class PreparedRelationalChannels:
    """Own a native endpoint and its rational T, raw G, F and C actions."""
    cdef shared_ptr[ScalarChannels] _handle
    cdef object _bounds, _archive_digest, _weights, _reading
    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("relational channels are not initialized")

    def __init__(self, archive, weights, *, coparticipation="share",
                 limits=(4096, 16000000, 64000000, 1 << 24), max_bytes=256*1024*1024):
        from hashlib import sha256
        if self._handle.get() != NULL:
            raise RuntimeError("relational channel owner is immutable")
        if not isinstance(archive, bytes):
            raise TypeError("relational channels require encoded source bytes")
        admitted_bytes = _natural(max_bytes)
        if len(archive) > admitted_bytes:
            raise ValueError("relational archive exceeds byte admission")
        checked = tuple(_natural(i) for i in limits)
        if len(checked) != 4 or any(not i for i in checked):
            raise ValueError("four positive relational action limits required")
        if coparticipation not in ("share", "count"):
            raise ValueError("coparticipation requires share or count")
        cdef vector[string] encoded
        retained = []
        for value in weights:
            if len(retained) >= checked[0]:
                raise ValueError("relation weight carrier exceeds admission")
            encoded.push_back(_scalar(value, checked[3]))
            retained.append(Fraction(int(value)) if isinstance(value, Integral) else Fraction(value))
        cdef string data = archive
        cdef vector[size_t] bounds = checked
        cdef bint count = coparticipation == "count"
        cdef size_t byte_limit = admitted_bytes
        with nogil:
            self._handle.reset(new ScalarChannels(data, encoded, count, bounds, byte_limit))
        self._bounds = checked; self._weights = tuple(retained); self._reading = coparticipation
        self._archive_digest = sha256(archive).hexdigest()

    @property
    def dimensions(self):
        self._require()
        return self._handle.get().boundaries(), self._handle.get().relations()

    @property
    def receipt(self):
        self._require()
        return dict(build=channels_build_info(), source_archive=self._archive_digest,
                    grade=1, field_metric="unit", geometry="raw", coparticipation=self._reading,
                    weights=self._weights, dimensions=self.dimensions,
                    occurrences=self._handle.get().occurrences(), limits=self._bounds,
                    retained_entries=self._handle.get().retained_entries(),
                    preparation_steps=self._handle.get().preparation_steps(),
                    traces=_values(self._handle.get().traces()),
                    diagonals=tuple(_values(row) for row in self._handle.get().diagonals()),
                    signed_incidence=_values(self._handle.get().incidence()),
                    occurrence_cells=tuple(tuple(row) for row in self._handle.get().occurrence_cells()),
                    occurrence_keys=tuple(self._handle.get().occurrence_keys()),
                    occurrence_activity=tuple(bool(v) for v in self._handle.get().occurrence_activity()),
                    channels=("T", "G", "F", "C"))

    def apply(self, field, *, trace_normalized=False):
        self._require()
        if type(trace_normalized) is not bool:
            raise TypeError("trace normalization requires a boolean")
        cdef vector[string] encoded
        width = self._handle.get().relations()
        for value in field:
            if encoded.size() >= width:
                raise ValueError("field exceeds original relation carrier")
            encoded.push_back(_scalar(value, self._bounds[3]))
        if encoded.size() != width:
            raise ValueError("complete original relation field required")
        cdef bint normalized = trace_normalized
        cdef ScalarChannelResult result
        with nogil:
            result = self._handle.get().apply(encoded, normalized)
        return dict(profile="rational-relational-channel-response-v1", source_archive=self._archive_digest,
                    fields=tuple(_values(row) for row in result.fields), energies=_values(result.energies),
                    trace_normalized=trace_normalized, steps=result.steps, channels=("T", "G", "F", "C"))

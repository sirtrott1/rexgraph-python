# distutils: language = c++
"""Owning packed Boolean input planes and canonical truth gate fitting."""
from numbers import Integral

cimport cython
from libc.stddef cimport size_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from ._bridge import CancellationToken, ExecutionCancelled

cdef extern from "binary_bridge.hpp" nogil:
    cdef cppclass CppBinaryInput "rexgraph::python::binary::Input":
        CppBinaryInput(const string&, const string&, size_t, size_t) except +
        size_t lanes() except +
        size_t words() except +
        size_t max_lanes() except +
        string first() except +
        string second() except +
    cdef cppclass BinaryFit "rexgraph::python::binary::Fit":
        unsigned truth_table
        size_t hamming_error
    string cpp_forward "rexgraph::python::binary::forward"(unsigned, const shared_ptr[CppBinaryInput]&) except +
    BinaryFit cpp_fit "rexgraph::python::binary::fit"(const shared_ptr[CppBinaryInput]&, const string&) except +
    string cpp_version "rexgraph::python::binary::engine_version"() except +
    string cpp_engine "rexgraph::python::binary::engine_digest"() except +
    string cpp_binding "rexgraph::python::binary::binding_digest"() except +

def _uint(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 1 << 64:
        raise ValueError("packed Boolean sizes require unsigned 64 bit integers")
    return int(value)

def _cancel(token):
    if token is not None:
        if not isinstance(token, CancellationToken):
            raise TypeError("packed Boolean cancellation requires CancellationToken")
        if token.cancelled:
            raise ExecutionCancelled("packed Boolean execution cancelled")

def binary_build_info():
    """Identify the canonical SDK and the separate packed Boolean binding."""
    return {"engine": "rexgraph-cpp", "version": (<bytes>cpp_version()).decode("ascii"),
            "engine_digest": (<bytes>cpp_engine()).decode("ascii"),
            "binding_digest": (<bytes>cpp_binding()).decode("ascii"),
            "profile": "binary-truth-gate-v1"}

@cython.final
cdef class NativeBinaryInput:
    cdef shared_ptr[CppBinaryInput] _pin
    cdef bint _building

    def __init__(self, first, second, lanes, *, max_lanes=1048576, cancellation=None):
        if self._building or self._pin.get() != NULL:
            raise RuntimeError("packed Boolean input is already initialized")
        _cancel(cancellation)
        if type(first) is not bytes or type(second) is not bytes:
            raise TypeError("packed Boolean planes require immutable bytes")
        cdef size_t count = _uint(lanes)
        cdef size_t maximum = _uint(max_lanes)
        if not 0 < maximum <= 64000000 or count > maximum:
            raise ValueError("packed Boolean lane admission exceeded")
        if len(first) != (count + 7) // 8 or len(second) != (count + 7) // 8:
            raise ValueError("packed Boolean plane differs from its declared lane count")
        if count % 8 and (first[-1] >> (count % 8) or second[-1] >> (count % 8)):
            raise ValueError("packed Boolean plane has nonzero tail padding")
        cdef string a = first
        cdef string b = second
        cdef shared_ptr[CppBinaryInput] pin
        self._building = True
        try:
            with nogil:
                pin = shared_ptr[CppBinaryInput](new CppBinaryInput(a, b, count, maximum))
            _cancel(cancellation)
            self._pin = pin
        finally:
            self._building = False

    cdef void _check(self) except *:
        if self._pin.get() == NULL:
            raise RuntimeError("uninitialized packed Boolean input")

    @property
    def lanes(self):
        self._check()
        return self._pin.get().lanes()
    @property
    def word_count(self):
        self._check()
        return self._pin.get().words()
    @property
    def max_lanes(self):
        self._check()
        return self._pin.get().max_lanes()
    @property
    def first(self):
        self._check()
        return <bytes>self._pin.get().first()
    @property
    def second(self):
        self._check()
        return <bytes>self._pin.get().second()

@cython.final
cdef class NativeBinaryGate:
    cdef unsigned _table
    cdef bint _ready

    def __init__(self, truth_table=0):
        if self._ready:
            raise RuntimeError("binary truth gate is already initialized")
        table = _uint(truth_table)
        if table > 15:
            raise ValueError("binary truth table must fit four bits")
        self._table = table
        self._ready = True

    cdef void _check(self) except *:
        if not self._ready:
            raise RuntimeError("uninitialized binary truth gate")

    @property
    def truth_table(self):
        self._check()
        return self._table

    def forward(self, NativeBinaryInput inputs not None, *, cancellation=None):
        self._check()
        inputs._check()
        _cancel(cancellation)
        cdef string raw
        cdef shared_ptr[CppBinaryInput] pin = inputs._pin
        with nogil:
            raw = cpp_forward(self._table, pin)
        _cancel(cancellation)
        return <bytes>raw

def fit_binary_gate(NativeBinaryInput inputs not None, targets, *, cancellation=None):
    """Fit all sixteen truth tables, choosing the lowest table on equal Hamming loss."""
    inputs._check()
    _cancel(cancellation)
    if type(targets) is not bytes:
        raise TypeError("packed Boolean targets require immutable bytes")
    if len(targets) != (inputs.lanes + 7) // 8:
        raise ValueError("packed Boolean targets differ from their declared lane count")
    if inputs.lanes % 8 and targets[-1] >> (inputs.lanes % 8):
        raise ValueError("packed Boolean target has nonzero tail padding")
    cdef string raw = targets
    cdef BinaryFit result
    cdef shared_ptr[CppBinaryInput] pin = inputs._pin
    with nogil:
        result = cpp_fit(pin, raw)
    _cancel(cancellation)
    return NativeBinaryGate(result.truth_table), result.hamming_error

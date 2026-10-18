from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport InputUseAssetBridge, LiteralInputBridge
from ._finite_field cimport FiniteSourceBridge, FiniteTarget, FiniteMapBridge, FiniteResultBridge

cdef extern from "literal_finite_field_bridge.hpp" namespace "rexgraph::python" nogil:
    string literal_use_value(shared_ptr[InputUseAssetBridge], uint64_t, const vector[size_t]&, size_t) except +
    cdef cppclass LiteralFiniteRule:
        uint64_t key
        vector[uint64_t] bindings, symbols
        cppbool has_target
        size_t target
        string unsupported
        vector[shared_ptr[LiteralInputBridge]] inputs
    cdef cppclass LiteralFiniteDeclaration:
        uint64_t key, revision
        vector[LiteralFiniteRule] rules
    cdef cppclass LiteralFiniteResultBridge
    cdef cppclass LiteralFiniteMapBridge:
        LiteralFiniteMapBridge(shared_ptr[FiniteSourceBridge], const vector[FiniteTarget]&, const LiteralFiniteDeclaration&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[LiteralFiniteMapBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[FiniteSourceBridge] source() except +
        shared_ptr[FiniteMapBridge] raw_map() except +
        cppbool same_owner(const LiteralFiniteMapBridge&) except +
        LiteralFiniteDeclaration info(size_t) except +
        shared_ptr[LiteralFiniteResultBridge] execute() except +
    cdef cppclass LiteralFiniteResultBridge:
        @staticmethod
        shared_ptr[LiteralFiniteResultBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[LiteralFiniteMapBridge] map() except +
        shared_ptr[FiniteResultBridge] family() except +

cdef class PreparedLiteralFiniteFieldMap:
    cdef shared_ptr[LiteralFiniteMapBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[LiteralFiniteMapBridge] _pin(self) except *

cdef class NativeLiteralFiniteFieldResult:
    cdef shared_ptr[LiteralFiniteResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[LiteralFiniteResultBridge] _pin(self) except *

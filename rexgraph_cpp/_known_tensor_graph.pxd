from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport LiteralInputBridge, AccessionInputBridge, InputAccessionBridge, FieldProgramBridge, ProgramStalkInfo

ctypedef int (*GraphCheckFunction)(void*) noexcept nogil
cdef extern from "known_tensor_graph_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass GraphCheck:
        GraphCheck() noexcept
        GraphCheckFunction run
        void* context
    cdef cppclass GraphBlock:
        size_t grade, cell
        uint64_t source_basis, lifted_basis
        vector[uint64_t] left, right, products
    cdef cppclass GraphDeclaration:
        uint64_t key, revision
        vector[GraphBlock] blocks
    cdef cppclass KnownGraphInputBridge
    cdef cppclass KnownGraphBridge:
        KnownGraphBridge(shared_ptr[LiteralInputBridge], shared_ptr[AccessionInputBridge], const GraphDeclaration&, const vector[size_t]&, size_t, GraphCheck) except +
        @staticmethod
        shared_ptr[KnownGraphBridge] restore(const string&, const vector[size_t]&, size_t, GraphCheck) except +
        string archive(GraphCheck) except +
        GraphDeclaration declaration(size_t) except +
        vector[ProgramStalkInfo] stalks(size_t) except +
        string field() except +
        string product(size_t, size_t) except +
        string matrix(unsigned int) except +
        size_t retained_entries() except +
        shared_ptr[AccessionInputBridge] parent() except +
        shared_ptr[LiteralInputBridge] literal() except +
        cppbool same_owner(const KnownGraphBridge&) except +
        void require_image(const string&, GraphCheck) except +
        shared_ptr[KnownGraphInputBridge] admit(shared_ptr[InputAccessionBridge], GraphCheck) except +
    cdef cppclass KnownGraphInputBridge:
        @staticmethod
        shared_ptr[KnownGraphInputBridge] restore(const string&, const vector[size_t]&, size_t, GraphCheck) except +
        string archive(GraphCheck) except +
        shared_ptr[KnownGraphBridge] graph() except +
        shared_ptr[AccessionInputBridge] input() except +
        shared_ptr[FieldProgramBridge] program() except +

cdef class _Check:
    cdef object callback, error
cdef _Check _check(value)
cdef GraphCheck _callback(_Check state)
cdef void _raise_check(_Check state) except *

cdef class NativeKnownTensorGraph:
    cdef shared_ptr[KnownGraphBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[KnownGraphBridge] _pin(self) except *

cdef class NativeKnownTensorGraphInput:
    cdef shared_ptr[KnownGraphInputBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[KnownGraphInputBridge] _pin(self) except *

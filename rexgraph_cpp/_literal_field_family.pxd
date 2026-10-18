from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from ._known_tensor_graph cimport GraphCheck, KnownGraphInputBridge
from ._field_interface cimport LiteralInputBridge, AccessionResultBridge
from ._field_evidence cimport FieldFamilyBridge, JointResultBridge, JointEvaluationBridge

cdef extern from "literal_field_family_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass LiteralFamilyInput:
        shared_ptr[LiteralInputBridge] literal
        shared_ptr[KnownGraphInputBridge] graph
    cdef cppclass LiteralJointEvaluationBridge
    cdef cppclass LiteralJointResultBridge:
        LiteralJointResultBridge(shared_ptr[FieldFamilyBridge], size_t, const vector[LiteralFamilyInput]&, const vector[size_t]&, size_t, GraphCheck) except +
        @staticmethod
        shared_ptr[LiteralJointResultBridge] restore(const string&, const vector[size_t]&, size_t, GraphCheck) except +
        string archive(GraphCheck) except +
        shared_ptr[JointResultBridge] raw() except +
        shared_ptr[FieldFamilyBridge] family() except +
        size_t member() except +
        vector[LiteralFamilyInput] inputs(size_t) except +
        shared_ptr[LiteralJointEvaluationBridge] evaluate(const vector[string]&, GraphCheck) except +
    cdef cppclass LiteralJointEvaluationBridge:
        @staticmethod
        shared_ptr[LiteralJointEvaluationBridge] restore(const string&, const vector[size_t]&, size_t, GraphCheck) except +
        string archive(GraphCheck) except +
        shared_ptr[LiteralJointResultBridge] parent() except +
        shared_ptr[JointEvaluationBridge] raw() except +
        shared_ptr[AccessionResultBridge] result() except +

cdef class NativeLiteralJointFieldResult:
    cdef shared_ptr[LiteralJointResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[LiteralJointResultBridge] _pin(self) except *

cdef class NativeLiteralJointFieldEvaluation:
    cdef shared_ptr[LiteralJointEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[LiteralJointEvaluationBridge] _pin(self) except *

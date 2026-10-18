from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._finite_field cimport FiniteDeclaration, FiniteSourceBridge, FiniteMapBridge, FiniteResultBridge, FiniteBranchBridge, FiniteEvaluationBridge
from ._field_interface cimport InputUseAssetBridge
from ._field_interface cimport ProgramStalkInfo

cdef extern from "finite_operator_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass FiniteOperatorPort:
        uint64_t key, rule
        size_t input
    cdef cppclass FiniteOperatorDeclaration:
        uint64_t key, revision
        vector[FiniteOperatorPort] ports
    cdef cppclass FiniteUseCorrespondence:
        uint64_t reference, input
    cdef cppclass FiniteTransportDeclaration:
        uint64_t key, revision
        vector[vector[size_t]] cells, occurrences
        vector[FiniteUseCorrespondence] uses
    cdef cppclass FiniteVariableCorrespondence:
        size_t reference, input, grade, reference_cell, input_cell
    cdef cppclass FiniteSchemaInfo:
        uint64_t key, node
        size_t grade
        cppbool isometric
        vector[ProgramStalkInfo] stalks
        vector[vector[size_t]] cells, occurrences
        string accession, reconstruction
    cdef cppclass FiniteInputSchemaBridge:
        uint64_t port() except +
        shared_ptr[InputUseAssetBridge] asset() except +
        FiniteSchemaInfo info(size_t) except +
    cdef cppclass FiniteOperatorInput:
        uint64_t port, use
        shared_ptr[FiniteInputSchemaBridge] schema
        string field
    cdef cppclass FiniteTransportBridge:
        FiniteTransportBridge(shared_ptr[FiniteSourceBridge], shared_ptr[FiniteSourceBridge], const FiniteTransportDeclaration&, const vector[size_t]&, size_t) except +
        shared_ptr[FiniteSourceBridge] reference() except +
        shared_ptr[FiniteSourceBridge] source() except +
        FiniteTransportDeclaration info(size_t) except +
        cppbool same_owner(const FiniteTransportBridge&) except +
        vector[FiniteVariableCorrespondence] variables(size_t,size_t) except +
        vector[FiniteUseCorrespondence] uses(size_t,size_t) except +
    cdef cppclass FiniteOperatorInstanceBridge
    cdef cppclass FiniteOperatorResultBridge
    cdef cppclass FiniteOperatorBranchBridge
    cdef cppclass FiniteOperatorEvaluationBridge
    cdef cppclass FiniteOperatorBridge:
        FiniteOperatorBridge(shared_ptr[FiniteMapBridge], const FiniteOperatorDeclaration&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[FiniteOperatorBridge] restore(const string&,const vector[size_t]&,size_t) except +
        string archive() except +
        shared_ptr[FiniteMapBridge] reference() except +
        uint64_t key() except +
        uint64_t revision() except +
        size_t size() except +
        cppbool same_owner(const FiniteOperatorBridge&) except +
        vector[FiniteOperatorPort] ports(size_t,size_t) except +
        FiniteDeclaration source_declaration(shared_ptr[InputUseAssetBridge],const FiniteTransportDeclaration&,size_t) except +
        shared_ptr[FiniteInputSchemaBridge] input_schema(shared_ptr[InputUseAssetBridge],const FiniteTransportDeclaration&,uint64_t) except +
        shared_ptr[FiniteOperatorInstanceBridge] instantiate(shared_ptr[FiniteSourceBridge],uint64_t,uint64_t,const FiniteTransportDeclaration&,const vector[FiniteOperatorInput]&) except +
    cdef cppclass FiniteOperatorInstanceBridge:
        @staticmethod
        shared_ptr[FiniteOperatorInstanceBridge] restore(const string&,const vector[size_t]&,size_t) except +
        string archive() except +
        uint64_t key() except +
        uint64_t revision() except +
        size_t size() except +
        cppbool same_owner(const FiniteOperatorInstanceBridge&) except +
        shared_ptr[FiniteOperatorBridge] operator_program() except +
        shared_ptr[FiniteTransportBridge] transport() except +
        shared_ptr[FiniteSourceBridge] source() except +
        shared_ptr[FiniteMapBridge] model() except +
        vector[FiniteOperatorInput] inputs(size_t,size_t) except +
        shared_ptr[FiniteOperatorResultBridge] execute() except +
    cdef cppclass FiniteOperatorResultBridge:
        @staticmethod
        shared_ptr[FiniteOperatorResultBridge] restore(const string&,const vector[size_t]&,size_t) except +
        string archive() except +
        shared_ptr[FiniteOperatorInstanceBridge] instance() except +
        shared_ptr[FiniteResultBridge] raw() except +
        size_t size() except +
        vector[shared_ptr[FiniteOperatorBranchBridge]] select(size_t,size_t) except +
        shared_ptr[FiniteOperatorEvaluationBridge] evaluate(size_t,const vector[string]&) except +
    cdef cppclass FiniteOperatorBranchBridge:
        shared_ptr[FiniteOperatorResultBridge] parent() except +
        size_t ordinal() except +
        shared_ptr[FiniteBranchBridge] raw() except +
    cdef cppclass FiniteOperatorEvaluationBridge:
        @staticmethod
        shared_ptr[FiniteOperatorEvaluationBridge] restore(const string&,const vector[size_t]&,size_t) except +
        string archive() except +
        shared_ptr[FiniteOperatorResultBridge] family() except +
        size_t ordinal() except +
        shared_ptr[FiniteEvaluationBridge] raw() except +

cdef class PreparedFiniteFieldOperator:
    cdef shared_ptr[FiniteOperatorBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FiniteOperatorBridge] _pin(self) except *

cdef class NativeFiniteSourceTransport:
    cdef shared_ptr[FiniteTransportBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FiniteTransportBridge] _pin(self) except *

cdef class NativeFiniteInputSchema:
    cdef shared_ptr[FiniteInputSchemaBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteInputSchemaBridge] _pin(self) except *

cdef class NativeFiniteOperatorInstance:
    cdef shared_ptr[FiniteOperatorInstanceBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteOperatorInstanceBridge] _pin(self) except *

cdef class NativeFiniteOperatorResult:
    cdef shared_ptr[FiniteOperatorResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteOperatorResultBridge] _pin(self) except *

cdef class NativeFiniteOperatorBranch:
    cdef shared_ptr[FiniteOperatorBranchBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteOperatorBranchBridge] _pin(self) except *

cdef class NativeFiniteOperatorEvaluation:
    cdef shared_ptr[FiniteOperatorEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteOperatorEvaluationBridge] _pin(self) except *

cdef FiniteTransportDeclaration _transport(value,vector[size_t] bounds)
cdef NativeFiniteInputSchema _wrap_schema(shared_ptr[FiniteInputSchemaBridge] handle,vector[size_t] bounds,size_t maximum)
cdef PreparedFiniteFieldOperator _wrap_operator(shared_ptr[FiniteOperatorBridge] handle,vector[size_t] bounds,size_t maximum)
cdef NativeFiniteOperatorInstance _wrap_instance(shared_ptr[FiniteOperatorInstanceBridge] handle,vector[size_t] bounds,size_t maximum)
cdef NativeFiniteOperatorResult _wrap_result(shared_ptr[FiniteOperatorResultBridge] handle,vector[size_t] bounds,size_t maximum)
cdef NativeFiniteOperatorBranch _wrap_branch(shared_ptr[FiniteOperatorBranchBridge] handle,vector[size_t] bounds,size_t maximum)
cdef NativeFiniteOperatorEvaluation _wrap_evaluation(shared_ptr[FiniteOperatorEvaluationBridge] handle,vector[size_t] bounds,size_t maximum)

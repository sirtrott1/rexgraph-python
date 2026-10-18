from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_admission cimport AvailabilityContextBridge
from ._field_interface cimport InputUseAssetBridge
from ._finite_field cimport FiniteDeclaration
from ._finite_field_admission cimport AvailableFiniteSourceBridge, AvailableFiniteMapBridge
from ._finite_operator cimport (FiniteOperatorDeclaration, FiniteOperatorPort, FiniteTransportDeclaration,
    FiniteInputSchemaBridge, FiniteOperatorBridge, FiniteOperatorInstanceBridge, FiniteOperatorResultBridge,
    FiniteOperatorBranchBridge, FiniteOperatorEvaluationBridge)

cdef extern from "finite_operator_admission_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass AvailableFiniteOperatorInput:
        uint64_t port,use
        shared_ptr[FiniteInputSchemaBridge] schema
        string field
        vector[uint64_t] origins
    cdef cppclass AvailableFiniteOperatorRuleInfo:
        uint64_t key
        vector[uint64_t] bindings,symbols
        cppbool has_target
        size_t target,input_count
        string unsupported
    cdef cppclass AvailableFiniteOperatorChildInfo:
        uint64_t key,revision
        vector[AvailableFiniteOperatorRuleInfo] rules
    cdef cppclass AvailableFiniteOperatorInstanceBridge
    cdef cppclass AvailableFiniteOperatorResultBridge
    cdef cppclass AvailableFiniteOperatorBranchBridge
    cdef cppclass AvailableFiniteOperatorEvaluationBridge
    cdef cppclass AvailableFiniteOperatorBridge:
        AvailableFiniteOperatorBridge(shared_ptr[AvailableFiniteMapBridge],const FiniteOperatorDeclaration&,unsigned,const vector[uint64_t]&,shared_ptr[AvailabilityContextBridge],const vector[size_t]&,size_t) except +
        @staticmethod
        shared_ptr[AvailableFiniteOperatorBridge] restore(const string&,shared_ptr[AvailabilityContextBridge],const vector[size_t]&,size_t) except +
        @staticmethod
        string payload(shared_ptr[AvailableFiniteMapBridge],const FiniteOperatorDeclaration&,unsigned,shared_ptr[AvailabilityContextBridge],size_t) except +
        @staticmethod
        vector[string] input_payloads(shared_ptr[FiniteInputSchemaBridge],uint64_t,const string&,const vector[size_t]&,size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FiniteOperatorBridge] raw() except +
        shared_ptr[AvailableFiniteMapBridge] reference() except +
        cppbool same_owner(const AvailableFiniteOperatorBridge&) except +
        unsigned status() except +
        vector[uint64_t] origins() except +
        vector[string] receipts(size_t) except +
        vector[string] dependency_archives(size_t) except +
        FiniteDeclaration source_declaration(shared_ptr[InputUseAssetBridge],const FiniteTransportDeclaration&,size_t) except +
        shared_ptr[FiniteInputSchemaBridge] input_schema(shared_ptr[InputUseAssetBridge],const FiniteTransportDeclaration&,uint64_t) except +
        string transport_payload(shared_ptr[AvailableFiniteSourceBridge],const FiniteTransportDeclaration&,unsigned,shared_ptr[AvailabilityContextBridge]) except +
        AvailableFiniteOperatorChildInfo child_declaration(uint64_t,uint64_t,const FiniteTransportDeclaration&,const vector[AvailableFiniteOperatorInput]&,size_t) except +
        string child_payload(shared_ptr[AvailableFiniteSourceBridge],uint64_t,uint64_t,const FiniteTransportDeclaration&,const vector[AvailableFiniteOperatorInput]&,unsigned,shared_ptr[AvailabilityContextBridge]) except +
        string instance_payload(shared_ptr[AvailableFiniteSourceBridge],uint64_t,uint64_t,const FiniteTransportDeclaration&,const vector[AvailableFiniteOperatorInput]&,unsigned,const vector[uint64_t]&,shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableFiniteOperatorInstanceBridge] instantiate(shared_ptr[AvailableFiniteSourceBridge],uint64_t,uint64_t,const FiniteTransportDeclaration&,const vector[AvailableFiniteOperatorInput]&,unsigned,const vector[uint64_t]&,shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableFiniteOperatorInstanceBridge:
        @staticmethod
        shared_ptr[AvailableFiniteOperatorInstanceBridge] restore(const string&,shared_ptr[AvailabilityContextBridge],const vector[size_t]&,size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FiniteOperatorInstanceBridge] raw() except +
        shared_ptr[AvailableFiniteOperatorBridge] operator_program() except +
        shared_ptr[AvailableFiniteSourceBridge] source() except +
        shared_ptr[AvailableFiniteMapBridge] field_map() except +
        cppbool same_owner(const AvailableFiniteOperatorInstanceBridge&) except +
        unsigned status() except +
        vector[uint64_t] origins() except +
        vector[string] receipts(size_t) except +
        vector[string] dependency_archives(size_t) except +
        shared_ptr[AvailableFiniteOperatorResultBridge] execute(shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableFiniteOperatorResultBridge:
        @staticmethod
        shared_ptr[AvailableFiniteOperatorResultBridge] restore(const string&,shared_ptr[AvailabilityContextBridge],const vector[size_t]&,size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FiniteOperatorResultBridge] raw() except +
        shared_ptr[AvailableFiniteOperatorInstanceBridge] instance() except +
        size_t size() except +
        vector[shared_ptr[AvailableFiniteOperatorBranchBridge]] select(size_t,size_t,shared_ptr[AvailabilityContextBridge]) except +
        string completion_payload(size_t,const vector[string]&,uint64_t,shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableFiniteOperatorEvaluationBridge] evaluate(size_t,const vector[string]&,uint64_t,uint64_t,shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableFiniteOperatorBranchBridge:
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableFiniteOperatorResultBridge] family() except +
        shared_ptr[FiniteOperatorBranchBridge] raw() except +
        size_t ordinal() except +
    cdef cppclass AvailableFiniteOperatorEvaluationBridge:
        @staticmethod
        shared_ptr[AvailableFiniteOperatorEvaluationBridge] restore(const string&,shared_ptr[AvailabilityContextBridge],const vector[size_t]&,size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableFiniteOperatorResultBridge] family() except +
        shared_ptr[FiniteOperatorEvaluationBridge] raw() except +
        size_t ordinal() except +
        string setting_receipt() except +
        string choice_receipt() except +
        vector[string] dependency_archives(size_t) except +

cdef class PreparedAvailableFiniteFieldOperator:
    cdef shared_ptr[AvailableFiniteOperatorBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableFiniteOperatorBridge] _pin(self) except *

cdef class NativeAvailableFiniteOperatorInstance:
    cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] _pin(self) except *

cdef class NativeAvailableFiniteOperatorResult:
    cdef shared_ptr[AvailableFiniteOperatorResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableFiniteOperatorResultBridge] _pin(self) except *

cdef class NativeAvailableFiniteOperatorBranch:
    cdef shared_ptr[AvailableFiniteOperatorBranchBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableFiniteOperatorBranchBridge] _pin(self) except *

cdef class NativeAvailableFiniteOperatorEvaluation:
    cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] _pin(self) except *

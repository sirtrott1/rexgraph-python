# distutils: language=c++
"""Bounded native graph gluing, selected operation words and complete joint families."""
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as cppbool
from libc.stdint cimport uint64_t
from libc.stddef cimport size_t
cdef extern from "demand_gluing_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass DemandPatchData:
        DemandPatchData() except +
        size_t index
        vector[uint64_t] cells,occurrences
        uint64_t available
    string demand_cover(const string&,const vector[uint64_t]&,const vector[DemandPatchData]&) except +
    cdef cppclass DemandArchiveInput:
        DemandArchiveInput() except +
        uint64_t name_space,source,revision,payload
        string bytes
    string demand_publish(const string&,const vector[DemandArchiveInput]&) except +
    cdef cppclass DemandCancel:
        DemandCancel() except +
        void cancel()
    cdef cppclass DemandBridge:
        @staticmethod
        shared_ptr[DemandBridge] open(const string&,const string&,const string&,const vector[size_t]&,const shared_ptr[DemandCancel]&) except +
        string execute(const string&) except +
        cppbool evict(size_t) except +
cdef extern from *:
    """
    static const char* demand_engine(){return RFC_ENGINE_DIGEST;}
    static const char* demand_binding(){return RFC_DEMAND_BINDING_DIGEST;}
    """
    const char* demand_engine()
    const char* demand_binding()
def build_info():
    return dict(engine_digest=(<bytes>demand_engine()).decode(),binding_digest=(<bytes>demand_binding()).decode(),profile='demand-graph-identity-v1')
def cover(bytes manifest,metadata,patches):
    cdef vector[uint64_t] m=metadata
    cdef vector[DemandPatchData] rows
    cdef DemandPatchData p
    for row in patches:
        p.index=row['index'];p.cells=row['cells'];p.occurrences=row['occurrences'];p.available=row['available_at'];rows.push_back(p)
    cdef string raw=manifest
    cdef string out=demand_cover(raw,m,rows)
    return <bytes>out
cdef class NativeDemand:
    cdef shared_ptr[DemandBridge] _owner
    cdef shared_ptr[DemandCancel] _cancel
    @classmethod
    def open(cls,path,bytes manifest,bytes cover,limits):
        cdef NativeDemand value=cls.__new__(cls)
        cdef string p=str(path).encode(),m=manifest,c=cover
        cdef vector[size_t] l=limits
        value._cancel=shared_ptr[DemandCancel](new DemandCancel())
        with nogil:value._owner=DemandBridge.open(p,m,c,l,value._cancel)
        return value
    def execute(self,bytes request):
        cdef string r=request
        cdef string out
        with nogil:out=self._owner.get().execute(r)
        return <bytes>out
    def cancel(self):self._cancel.get().cancel()
    def evict(self,index):return self._owner.get().evict(index)

def publish(path,payloads):
    cdef vector[DemandArchiveInput] rows
    cdef DemandArchiveInput value
    for row in payloads:
        value.name_space=row['namespace'];value.source=row['source'];value.revision=row['revision'];value.payload=row['payload'];value.bytes=row['bytes'];rows.push_back(value)
    cdef string p=str(path).encode()
    cdef string result=demand_publish(p,rows)
    return <bytes>result

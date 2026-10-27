# distutils: language=c++
"""Fresh owning replay of the closed C declaration profiles."""
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector as cppvector
from libcpp.string cimport string
from libcpp cimport bool as cppbool
from libc.stdint cimport uint64_t
from libc.stddef cimport size_t
from fractions import Fraction

cdef extern from "c_workflow_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass CCancel:
        CCancel() except +
        void cancel()
    cdef cppclass CPayloadBridge:
        CPayloadBridge(size_t) except +
        void integer(uint64_t) except +
        void flag(cppbool) except +
        void rational(const string&) except +
        void vector(const cppvector[string]&) except +
        void matrix(size_t,size_t,const cppvector[string]&) except +
        void blob(const string&) except +
        void text(const string&) except +
        string take() except +
    cdef cppclass CRecordData:
        CRecordData() except +
        uint64_t kind,key,revision
        cppvector[size_t] dependencies
        string payload
    cdef cppclass CMetaData:
        uint64_t producer_namespace,producer,revision,evidence_namespace,evidence_cutoff
        string purpose
    cdef cppclass CItem:
        string name,kind,bytes
        size_t rows,cols
        cppvector[string] values
        cppvector[uint64_t] integers
    cdef cppclass CInspection:
        string type
        cppvector[CItem] items
    cdef cppclass CWorkflowBridge:
        @staticmethod
        shared_ptr[CWorkflowBridge] decode(const string&,const cppvector[size_t]&,const cppvector[size_t]&,const shared_ptr[CCancel]&) except +
        @staticmethod
        shared_ptr[CWorkflowBridge] prepare(const CMetaData&,const cppvector[CRecordData]&,const cppvector[size_t]&,const cppvector[size_t]&,const cppvector[size_t]&,const shared_ptr[CCancel]&) except +
        string encode() except +
        cppvector[CRecordData] records() except +
        cppvector[size_t] exports() except +
        CMetaData meta() except +
        cppbool same(const shared_ptr[CWorkflowBridge]&) except +
        CInspection inspect(size_t,size_t) except +
        CInspection evaluate(size_t,const string&,const string&,size_t,size_t,size_t) except +

cdef extern from *:
    """
    static const char* c_engine_digest(){return RFC_ENGINE_DIGEST;}
    static const char* c_binding_digest(){return RFC_C_WORKFLOW_BINDING_DIGEST;}
    """
    const char* c_engine_digest()
    const char* c_binding_digest()

LIMITS=(32<<20,128,32,2000000,256,2048)
EXACT=(4096,16000000,64000000,1<<24)

def build_info():
    return {"engine_digest":(<bytes>c_engine_digest()).decode(),
            "binding_digest":(<bytes>c_binding_digest()).decode(),
            "profile":"closed-C-declarations-v1","constructors":66}

def _natural(x):
    if type(x) is not int or x<0 or x>=2**64:
        raise ValueError("a nonnegative exact 64-bit integer is required")
    return x

def _rational(x):
    if isinstance(x,bool) or isinstance(x,float):
        raise TypeError("exact rational input is required")
    if type(x) not in (int,str,Fraction):
        raise TypeError("use an integer, Fraction or exact rational text")
    q=Fraction(x)
    return (format(q.numerator,'x')+'/'+format(q.denominator,'x')).encode('ascii')

cdef object _text(string value):
    cdef bytes raw=value
    return raw.decode()

cdef object _fraction_text(string value):
    text=_text(value)
    n,separator,d=text.partition('/')
    return Fraction(int(n,16),int(d,16) if separator else 1)

cdef object _inspection(CInspection result):
    cdef CItem item
    cdef string scalar
    out={};shapes={}
    for item in result.items:
        name=_text(item.name);kind=_text(item.kind)
        if kind=="rational": value=_fraction_text(item.values[0])
        elif kind=="boolean": value=bool(item.integers[0])
        elif kind=="integers": value=tuple(item.integers)
        elif kind in ("matrix","vector"):
            value=[]
            for scalar in item.values:value.append(_fraction_text(scalar))
            value=tuple(value)
            if kind=="matrix":
                shapes[name]=(int(item.rows),int(item.cols))
                value=tuple(value[i*item.cols:(i+1)*item.cols] for i in range(item.rows))
            else:shapes[name]=(len(value),)
        elif kind=="text": value=_text(item.bytes)
        elif kind=="bytes": value=<bytes>item.bytes
        else: raise ValueError("unknown native C observation")
        out[name]=value
    return {"type":_text(result.type),"fields":out,"shapes":shapes}

cdef class Cancellation:
    cdef shared_ptr[CCancel] _ptr
    def __cinit__(self):self._ptr=shared_ptr[CCancel](new CCancel())
    def cancel(self):self._ptr.get().cancel()

cdef class Payload:
    cdef CPayloadBridge* _ptr
    def __cinit__(self, maximum=32<<20):self._ptr=new CPayloadBridge(_natural(maximum))
    def __dealloc__(self):del self._ptr
    def u64(self,value):self._ptr.integer(_natural(value));return self
    def flag(self,value):
        if type(value) is not bool:raise TypeError("a declared boolean is required")
        self._ptr.flag(value);return self
    def rational(self,value):self._ptr.rational(_rational(value));return self
    def vector(self,values):
        cdef cppvector[string] rows
        for x in values:rows.push_back(_rational(x))
        self._ptr.vector(rows);return self
    def vectors(self,values):
        values=tuple(values);self.u64(len(values))
        for x in values:self.vector(x)
        return self
    def matrix(self,values,shape=None):
        cdef cppvector[string] flat
        rows=tuple(tuple(row) for row in values)
        r=len(rows);c=len(rows[0]) if r else 0
        if shape is not None:
            if len(shape)!=2:raise ValueError("matrix shape needs two dimensions")
            r,c=map(_natural,shape)
        if len(rows)!=r or any(len(row)!=c for row in rows):raise ValueError("matrix has an incomplete declared carrier")
        for row in rows:
            for x in row:flat.push_back(_rational(x))
        self._ptr.matrix(r,c,flat);return self
    def matrices(self,values):
        values=tuple(values);self.u64(len(values))
        for x in values:self.matrix(x)
        return self
    def indices(self,values):
        values=tuple(values);self.u64(len(values))
        for x in values:self.u64(x)
        return self
    def blob(self,value):
        if type(value) is not bytes:raise TypeError("native source bytes are required")
        self._ptr.blob(value);return self
    def string(self,value):
        if type(value) is not str:raise TypeError("declared text is required")
        self._ptr.text(value.encode());return self
    def take(self):return <bytes>self._ptr.take()

cdef class NativeArchive:
    cdef shared_ptr[CWorkflowBridge] _ptr
    def __init__(self):
        if not self._ptr:raise TypeError("use NativeArchive.decode or prepare")
    cdef void _require(self) except *:
        if not self._ptr:raise RuntimeError("native C archive is not initialized")
    @staticmethod
    def decode(bytes archive,limits=LIMITS,exact=EXACT,Cancellation cancel=None):
        if archive is None or len(limits)!=6 or len(exact)!=4:raise ValueError("complete C receiving admission required")
        if len(archive)>_natural(limits[0]):raise ValueError("native C archive exceeds receiving bytes")
        cdef string data=archive
        cdef cppvector[size_t] l=[_natural(v) for v in limits]
        cdef cppvector[size_t] e=[_natural(v) for v in exact]
        cdef NativeArchive result=NativeArchive.__new__(NativeArchive)
        if cancel is None:cancel=Cancellation()
        with nogil:result._ptr=CWorkflowBridge.decode(data,l,e,cancel._ptr)
        return result
    @staticmethod
    def prepare(meta,records,exports=(),limits=LIMITS,exact=EXACT,Cancellation cancel=None):
        cdef CMetaData m
        cdef CRecordData row
        cdef cppvector[CRecordData] rows
        cdef cppvector[size_t] l=[_natural(v) for v in limits]
        cdef cppvector[size_t] e=[_natural(v) for v in exact]
        cdef cppvector[size_t] out=[_natural(v) for v in exports]
        if set(meta)!={"producer_namespace","producer","revision","evidence_namespace","evidence_cutoff","purpose"}:raise ValueError("complete original producer/evidence metadata required")
        m.producer_namespace=_natural(meta['producer_namespace']);m.producer=_natural(meta['producer']);m.revision=_natural(meta['revision']);m.evidence_namespace=_natural(meta['evidence_namespace']);m.evidence_cutoff=_natural(meta['evidence_cutoff']);m.purpose=meta['purpose'].encode()
        from itertools import islice
        if len(limits)!=6 or len(exact)!=4:raise ValueError("complete C receiving admission required")
        for value in islice(records,limits[1]+1):
            if rows.size()>=l[1]:raise ValueError("C declaration record admission")
            if set(value)!={"kind","key","revision","dependencies","payload"}:raise ValueError("closed native C declaration required")
            row=CRecordData();row.kind=_natural(value['kind']);row.key=_natural(value['key']);row.revision=_natural(value['revision']);row.dependencies=[_natural(v) for v in value['dependencies']]
            if type(value['payload']) is not bytes:raise TypeError("declaration payload must be bytes")
            row.payload=value['payload'];rows.push_back(row)
        cdef NativeArchive result=NativeArchive.__new__(NativeArchive)
        if cancel is None:cancel=Cancellation()
        with nogil:result._ptr=CWorkflowBridge.prepare(m,rows,out,l,e,cancel._ptr)
        return result
    def to_bytes(self):
        self._require()
        return <bytes>self._ptr.get().encode()
    def records(self):
        self._require()
        cdef CRecordData row
        return tuple(dict(kind=row.kind,key=row.key,revision=row.revision,dependencies=tuple(row.dependencies),payload=<bytes>row.payload) for row in self._ptr.get().records())
    @property
    def exports(self):
        self._require()
        return tuple(self._ptr.get().exports())
    @property
    def metadata(self):
        self._require()
        cdef CMetaData m=self._ptr.get().meta()
        return dict(producer_namespace=m.producer_namespace,producer=m.producer,revision=m.revision,evidence_namespace=m.evidence_namespace,evidence_cutoff=m.evidence_cutoff,purpose=_text(m.purpose))
    def same_owner(self,NativeArchive other not None):
        self._require();other._require()
        return self._ptr.get().same(other._ptr)
    def inspect(self,index,maximum=1000000):
        self._require()
        return _inspection(self._ptr.get().inspect(_natural(index),_natural(maximum)))
    def evaluate(self,index,operation,parameter=0,grade=0,power=0,maximum=1000000):
        self._require()
        cdef CInspection result
        cdef string op=operation.encode(),q=_rational(parameter)
        cdef size_t at=_natural(index),k=_natural(grade),n=_natural(power),cap=_natural(maximum)
        with nogil:result=self._ptr.get().evaluate(at,op,q,k,n,cap)
        return _inspection(result)

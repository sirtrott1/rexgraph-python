# distutils: language=c++
"""Whole native joint fields retaining actual literal and known input owners."""
from ._field_interface cimport NativeLiteralInput, NativeAccessionFieldResult, _bounds, _maximum, _image
from ._field_evidence cimport PreparedFieldFamily, NativeJointFieldResult, NativeJointFieldEvaluation
from ._known_tensor_graph cimport NativeKnownTensorGraphInput, _Check, _check, _callback, _raise_check
from ._stalk_fields import _natural, _scalar

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_LITERAL_FIELD_FAMILY_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)

def literal_field_family_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_LITERAL_FIELD_FAMILY_BINDING_DIGEST.decode(),
            'profile':'owning-literal-joint-field-v1','scalar_policy':'gmp-rational-exact'}

cdef class NativeLiteralJointFieldResult:
    cdef shared_ptr[LiteralJointResultBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('literal joint field result is not initialized')
        return self._handle

    def __init__(self, PreparedFieldFamily family not None, member, inputs, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        if self._loading or self._handle.get()!=NULL:raise TypeError('literal joint field result is immutable')
        self._loading=True
        cdef vector[size_t] bounds
        cdef size_t maximum,index
        cdef vector[LiteralFamilyInput] values
        cdef LiteralFamilyInput row
        cdef NativeLiteralInput literal
        cdef NativeKnownTensorGraphInput graph
        cdef shared_ptr[FieldFamilyBridge] actual
        cdef shared_ptr[LiteralJointResultBridge] loaded
        cdef _Check state
        cdef GraphCheck callback
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);index=_natural(member);actual=family._pin()
            for value in inputs:
                row.literal.reset();row.graph.reset()
                if isinstance(value,NativeLiteralInput):literal=value;row.literal=literal._pin()
                elif isinstance(value,NativeKnownTensorGraphInput):graph=value;row.graph=graph._pin()
                else:raise TypeError('actual literal or known graph input owner required')
                values.push_back(row)
                if values.size()>bounds[0]:raise ValueError('complete literal input count exceeds admission')
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=shared_ptr[LiteralJointResultBridge](new LiteralJointResultBridge(actual,index,values,bounds,maximum,callback))
            except BaseException:_raise_check(state);raise
            self._limits=bounds;self._maximum=maximum;self._handle=loaded
        finally:self._loading=False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        cdef NativeLiteralJointFieldResult out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef _Check state
        cdef GraphCheck callback
        cdef shared_ptr[LiteralJointResultBridge] loaded
        if out._loading or out._handle.get()!=NULL:raise TypeError('native cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum)
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=LiteralJointResultBridge.restore(image,bounds,maximum,callback)
            except BaseException:_raise_check(state);raise
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def archive(self, *, check=None):
        cdef shared_ptr[LiteralJointResultBridge] pin=self._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef string image
        try:
            with nogil:image=pin.get().archive(callback)
        except BaseException:_raise_check(state);raise
        return <bytes>image

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    @property
    def member_index(self):
        cdef shared_ptr[LiteralJointResultBridge] pin=self._pin()
        cdef size_t member
        with nogil:member=pin.get().member()
        return int(member)

    def family(self):
        cdef shared_ptr[LiteralJointResultBridge] pin=self._pin()
        cdef shared_ptr[FieldFamilyBridge] loaded
        cdef PreparedFieldFamily out=PreparedFieldFamily.__new__(PreparedFieldFamily)
        with nogil:loaded=pin.get().family()
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def raw_result(self):
        cdef shared_ptr[LiteralJointResultBridge] pin=self._pin()
        cdef shared_ptr[JointResultBridge] loaded
        cdef NativeJointFieldResult out=NativeJointFieldResult.__new__(NativeJointFieldResult)
        with nogil:loaded=pin.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def inputs(self, *, max_records=256):
        cdef shared_ptr[LiteralJointResultBridge] pin=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[LiteralFamilyInput] values
        cdef NativeLiteralInput literal
        cdef NativeKnownTensorGraphInput graph
        with nogil:values=pin.get().inputs(maximum)
        out=[]
        for value in values:
            if value.literal.get()!=NULL:
                literal=NativeLiteralInput.__new__(NativeLiteralInput)
                literal._limits=self._limits;literal._maximum=self._maximum;literal._handle=value.literal;out.append(literal)
            else:
                graph=NativeKnownTensorGraphInput.__new__(NativeKnownTensorGraphInput)
                graph._limits=self._limits;graph._maximum=self._maximum;graph._handle=value.graph;out.append(graph)
        return tuple(out)

    def evaluate(self, free_values, *, check=None):
        cdef shared_ptr[LiteralJointResultBridge] pin=self._pin()
        cdef vector[string] values
        for value in free_values:
            values.push_back(_scalar(value,self._limits[3]))
            if values.size()>self._limits[0]:raise ValueError('complete free setting exceeds admission')
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef shared_ptr[LiteralJointEvaluationBridge] loaded
        cdef NativeLiteralJointFieldEvaluation out
        try:
            with nogil:loaded=pin.get().evaluate(values,callback)
        except BaseException:_raise_check(state);raise
        out=NativeLiteralJointFieldEvaluation.__new__(NativeLiteralJointFieldEvaluation)
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def program(self):return self.raw_result().program()
    def input_uses(self, *, max_records=256):return self.raw_result().input_uses(max_records=max_records)
    def fields(self, *, max_records=256):return self.raw_result().fields(max_records=max_records)
    def field_archive(self,node, *, occurrences=False):return self.raw_result().field_archive(node,occurrences=occurrences)
    def select(self,node,cells, *, coordinate_limit=256):return self.raw_result().select(node,cells,coordinate_limit=coordinate_limit)
    def identified(self,node,coordinate):return self.raw_result().identified(node,coordinate)

cdef class NativeLiteralJointFieldEvaluation:
    cdef shared_ptr[LiteralJointEvaluationBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('literal joint evaluation is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        cdef NativeLiteralJointFieldEvaluation out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef _Check state
        cdef GraphCheck callback
        cdef shared_ptr[LiteralJointEvaluationBridge] loaded
        if out._loading or out._handle.get()!=NULL:raise TypeError('native cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum)
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=LiteralJointEvaluationBridge.restore(image,bounds,maximum,callback)
            except BaseException:_raise_check(state);raise
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def archive(self, *, check=None):
        cdef shared_ptr[LiteralJointEvaluationBridge] pin=self._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef string image
        try:
            with nogil:image=pin.get().archive(callback)
        except BaseException:_raise_check(state);raise
        return <bytes>image

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def parent(self):
        cdef shared_ptr[LiteralJointEvaluationBridge] pin=self._pin()
        cdef shared_ptr[LiteralJointResultBridge] loaded
        cdef NativeLiteralJointFieldResult out=NativeLiteralJointFieldResult.__new__(NativeLiteralJointFieldResult)
        with nogil:loaded=pin.get().parent()
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def raw_evaluation(self):
        cdef shared_ptr[LiteralJointEvaluationBridge] pin=self._pin()
        cdef shared_ptr[JointEvaluationBridge] loaded
        cdef NativeJointFieldEvaluation out=NativeJointFieldEvaluation.__new__(NativeJointFieldEvaluation)
        with nogil:loaded=pin.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def result(self):
        cdef shared_ptr[LiteralJointEvaluationBridge] pin=self._pin()
        cdef shared_ptr[AccessionResultBridge] loaded
        cdef NativeAccessionFieldResult out=NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil:loaded=pin.get().result()
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def free_values_archive(self):return self.raw_evaluation().free_values_archive()
    def program(self):return self.result().program()
    def fields(self, *, max_records=256):return self.result().fields(max_records=max_records)
    def field_archive(self,node, *, occurrences=False):return self.result().field_archive(node,occurrences=occurrences)
    def select(self,node,cells, *, coordinate_limit=256):return self.result().select(node,cells,coordinate_limit=coordinate_limit)
    def input_uses(self, *, max_records=256):return self.result().input_uses(max_records=max_records)

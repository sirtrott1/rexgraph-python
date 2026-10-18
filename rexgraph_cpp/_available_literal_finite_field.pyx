# distutils: language=c++
"""Complete native reader availability, original publishers and prospective choices."""
from ._field_interface cimport _bounds, _maximum, _image
from ._field_admission cimport NativeAvailabilityContext
from ._finite_field_admission cimport PreparedAvailableFiniteFieldMap, _free
from ._literal_finite_field cimport PreparedLiteralFiniteFieldMap
from ._finite_field cimport NativeFiniteFieldResult, NativeFiniteFieldBranch, NativeFiniteFieldEvaluation
from ._field_program import _closed
from ._stalk_fields import _natural

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_AVAILABLE_LITERAL_FINITE_FIELD_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)

def available_literal_finite_field_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_AVAILABLE_LITERAL_FINITE_FIELD_BINDING_DIGEST.decode(),
            'profile':'available-literal-finite-field-v1','scalar_policy':'gmp-rational-exact',
            'authority':'complete-native-reader-and-independent-publisher-context'}

cdef vector[vector[uint64_t]] _rows(values, vector[size_t] bounds):
    cdef vector[vector[uint64_t]] out
    cdef vector[uint64_t] row
    cdef size_t count=0
    for keys in values:
        count+=1
        if count>bounds[1]:raise ValueError('reader origin row count exceeds admission')
        row.clear()
        for key in keys:
            row.push_back(_natural(key));count+=1
            if count>bounds[1]:raise ValueError('reader origin dependency count exceeds admission')
        out.push_back(row)
    return out

def literal_finite_interface_origin_payload(PreparedAvailableFiniteFieldMap base not None,
        PreparedLiteralFiniteFieldMap reader not None, input_origins,
        NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    if base.admission()!={'limits':tuple(bounds),'max_bytes':maximum}:raise ValueError('reader base admission differs')
    cdef vector[vector[uint64_t]] keys=_rows(input_origins,bounds)
    cdef shared_ptr[AvailableFiniteMapBridge] actual=base._pin()
    cdef shared_ptr[LiteralFiniteMapBridge] recipe=reader._pin()
    cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
    cdef string image
    with nogil:image=AvailableLiteralMapBridge.interface_payload(actual,recipe,keys,authority,maximum)
    return <bytes>image


cdef class PreparedAvailableLiteralFiniteFieldMap:
    cdef shared_ptr[AvailableLiteralMapBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('available literal finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralMapBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef PreparedAvailableLiteralFiniteFieldMap out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[AvailableLiteralMapBridge] loaded
        if out._loading or out._handle.get()!=NULL:raise TypeError('available cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum)
            authority=execution_context._pin()
            with nogil:loaded=AvailableLiteralMapBridge.restore(image,authority,bounds,maximum)
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def archive(self):
        cdef shared_ptr[AvailableLiteralMapBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def __init__(self, PreparedAvailableFiniteFieldMap base not None, PreparedLiteralFiniteFieldMap reader not None,
                 origins, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('available literal finite owner is immutable')
        self._loading=True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef vector[vector[uint64_t]] rows
        cdef uint64_t interface
        cdef shared_ptr[AvailableFiniteMapBridge] actual
        cdef shared_ptr[LiteralFiniteMapBridge] recipe
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[AvailableLiteralMapBridge] loaded
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes)
            if base.admission()!={'limits':tuple(bounds),'max_bytes':maximum}:raise ValueError('reader base admission differs')
            _closed(origins,('inputs','interface'));rows=_rows(origins['inputs'],bounds);interface=_natural(origins['interface'])
            actual=base._pin();recipe=reader._pin();authority=execution_context._pin()
            with nogil:loaded=shared_ptr[AvailableLiteralMapBridge](new AvailableLiteralMapBridge(actual,recipe,rows,interface,authority,bounds,maximum))
            self._limits=bounds;self._maximum=maximum;self._handle=loaded
        finally:self._loading=False

    def same_owner(self, PreparedAvailableLiteralFiniteFieldMap other not None):
        cdef shared_ptr[AvailableLiteralMapBridge] actual=self._pin(), compared=other._pin()
        cdef cppbool same
        with nogil:same=actual.get().same_owner(compared.get()[0])
        return bool(same)

    def origins(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableLiteralMapBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[vector[uint64_t]] rows
        cdef uint64_t interface
        with nogil:rows=actual.get().inputs(maximum);interface=actual.get().interface()
        return {'inputs':tuple(tuple(row) for row in rows),'interface':interface}

    def execute(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralMapBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableLiteralFiniteFieldResult out=NativeAvailableLiteralFiniteFieldResult.__new__(NativeAvailableLiteralFiniteFieldResult)
        with nogil:out._handle=actual.get().execute(authority)
        out._limits=self._limits;out._maximum=self._maximum;return out

    def base(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralMapBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef PreparedAvailableFiniteFieldMap out=PreparedAvailableFiniteFieldMap.__new__(PreparedAvailableFiniteFieldMap)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().base()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def reader(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralMapBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef PreparedLiteralFiniteFieldMap out=PreparedLiteralFiniteFieldMap.__new__(PreparedLiteralFiniteFieldMap)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().reader()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def dependency_archives(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableLiteralMapBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] rows
        with nogil:rows=actual.get().dependencies(maximum)
        return tuple(<bytes>row for row in rows)


cdef class NativeAvailableLiteralFiniteFieldResult:
    cdef shared_ptr[AvailableLiteralResultBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('available literal finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeAvailableLiteralFiniteFieldResult out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[AvailableLiteralResultBridge] loaded
        if out._loading or out._handle.get()!=NULL:raise TypeError('available cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum)
            authority=execution_context._pin()
            with nogil:loaded=AvailableLiteralResultBridge.restore(image,authority,bounds,maximum)
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def archive(self):
        cdef shared_ptr[AvailableLiteralResultBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def inspect(self, NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        cdef shared_ptr[AvailableLiteralResultBridge] actual=self._pin()
        cdef size_t count
        with nogil:count=actual.get().size()
        return {'branch_count':count}

    def select(self, begin, count, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t first=_natural(begin),length=_natural(count)
        cdef vector[shared_ptr[AvailableLiteralBranchBridge]] rows
        cdef shared_ptr[AvailableLiteralBranchBridge] row
        cdef NativeAvailableLiteralFiniteFieldBranch out
        with nogil:rows=actual.get().select(first,length,authority)
        selected=[]
        for row in rows:
            out=NativeAvailableLiteralFiniteFieldBranch.__new__(NativeAvailableLiteralFiniteFieldBranch)
            out._handle=row;out._limits=self._limits;out._maximum=self._maximum;selected.append(out)
        return tuple(selected)

    def completion_payload(self, ordinal, free_values, parameter_setting, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t index=_natural(ordinal)
        cdef uint64_t setting=_natural(parameter_setting)
        cdef vector[string] values=_free(free_values,self._limits)
        cdef string image
        with nogil:image=actual.get().completion_payload(index,values,setting,authority)
        return <bytes>image

    def evaluate(self, ordinal, free_values, parameter_setting, choice_origin, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t index=_natural(ordinal)
        cdef uint64_t setting=_natural(parameter_setting),choice=_natural(choice_origin)
        cdef vector[string] values=_free(free_values,self._limits)
        cdef NativeAvailableLiteralFiniteFieldEvaluation out=NativeAvailableLiteralFiniteFieldEvaluation.__new__(NativeAvailableLiteralFiniteFieldEvaluation)
        with nogil:out._handle=actual.get().evaluate(index,values,setting,choice,authority)
        out._limits=self._limits;out._maximum=self._maximum;return out

    def model(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef PreparedAvailableLiteralFiniteFieldMap out=PreparedAvailableLiteralFiniteFieldMap.__new__(PreparedAvailableLiteralFiniteFieldMap)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().model()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def raw(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeFiniteFieldResult out=NativeFiniteFieldResult.__new__(NativeFiniteFieldResult)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeAvailableLiteralFiniteFieldBranch:
    cdef shared_ptr[AvailableLiteralBranchBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('available literal finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralBranchBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    def inspect(self, NativeAvailabilityContext execution_context not None):
        return self.raw(execution_context).inspect()

    def family(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralBranchBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableLiteralFiniteFieldResult out=NativeAvailableLiteralFiniteFieldResult.__new__(NativeAvailableLiteralFiniteFieldResult)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().family()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def raw(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralBranchBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeFiniteFieldBranch out=NativeFiniteFieldBranch.__new__(NativeFiniteFieldBranch)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeAvailableLiteralFiniteFieldEvaluation:
    cdef shared_ptr[AvailableLiteralEvaluationBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('available literal finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralEvaluationBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeAvailableLiteralFiniteFieldEvaluation out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[AvailableLiteralEvaluationBridge] loaded
        if out._loading or out._handle.get()!=NULL:raise TypeError('available cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum)
            authority=execution_context._pin()
            with nogil:loaded=AvailableLiteralEvaluationBridge.restore(image,authority,bounds,maximum)
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def archive(self):
        cdef shared_ptr[AvailableLiteralEvaluationBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def inspect(self, NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        cdef shared_ptr[AvailableLiteralEvaluationBridge] actual=self._pin()
        cdef string setting,choice
        with nogil:setting=actual.get().setting_receipt();choice=actual.get().choice_receipt()
        return {'ordinal':self.raw(execution_context).ordinal(),'parameter_setting_receipt':<bytes>setting if setting.size() else None,'choice_receipt':<bytes>choice}

    def family(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralEvaluationBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableLiteralFiniteFieldResult out=NativeAvailableLiteralFiniteFieldResult.__new__(NativeAvailableLiteralFiniteFieldResult)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().family()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def raw(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableLiteralEvaluationBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeFiniteFieldEvaluation out=NativeFiniteFieldEvaluation.__new__(NativeFiniteFieldEvaluation)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def dependency_archives(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableLiteralEvaluationBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] rows
        with nogil:rows=actual.get().dependencies(maximum)
        return tuple(<bytes>row for row in rows)

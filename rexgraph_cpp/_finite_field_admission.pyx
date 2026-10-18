# distutils: language=c++
"""Available finite carriers and whole field alternatives with actual native admission."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport NativeInputUseAsset, _bounds, _maximum, _image, _accession_declaration
from ._field_admission cimport NativeAvailabilityContext, PreparedAvailableFieldProgram, _keys
from ._field_evidence_admission cimport PreparedAvailableFieldEvidence
from ._finite_field cimport (NativeFiniteSource, NativeFiniteSections, PreparedFiniteFieldMap,
    NativeFiniteFieldResult, NativeFiniteFieldBranch, NativeFiniteFieldEvaluation, _declaration)
from ._field_program import _closed
from ._stalk_fields import _natural, _scalar

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FINITE_FIELD_ADMISSION_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)
_STATUSES=('observed','prospective')


def finite_field_admission_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
        'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_FINITE_FIELD_ADMISSION_BINDING_DIGEST.decode(),
        'profile':'owning-available-relational-finite-field-v1','scalar_policy':'gmp-rational-exact'}


cdef unsigned _status(value) except *:
    if value not in _STATUSES:raise ValueError('finite relationship requires explicit observed or prospective status')
    return _STATUSES.index(value)


cdef vector[uint64_t] _ids(value, size_t maximum):
    cdef vector[uint64_t] out
    for key in value:
        out.push_back(_natural(key))
        if out.size()>maximum:raise ValueError('finite key count exceeds admission')
    return out


cdef vector[string] _free(value, vector[size_t] bounds):
    cdef vector[string] out
    for q in value:
        out.push_back(_scalar(q,bounds[3]))
        if out.size()>bounds[0]:raise ValueError('complete finite choice dimension exceeds admission')
    return out


cdef vector[AvailableFiniteTarget] _targets(value, vector[size_t] bounds):
    cdef vector[AvailableFiniteTarget] out
    cdef AvailableFiniteTarget target
    for row in value:
        target.fixed.reset();target.affine.reset()
        if isinstance(row,PreparedAvailableFieldProgram):target.fixed=(<PreparedAvailableFieldProgram>row)._pin()
        elif isinstance(row,PreparedAvailableFieldEvidence):target.affine=(<PreparedAvailableFieldEvidence>row)._pin()
        else:raise TypeError('actual admitted fixed program or complete evidence target required')
        out.push_back(target)
        if out.size()>bounds[0]:raise ValueError('finite target count exceeds admission')
    return out


cdef object _target(AvailableFiniteTarget value, vector[size_t] bounds, size_t maximum):
    cdef PreparedAvailableFieldProgram fixed
    cdef PreparedAvailableFieldEvidence affine
    if value.fixed.get()!=NULL:
        fixed=PreparedAvailableFieldProgram.__new__(PreparedAvailableFieldProgram);fixed._handle=value.fixed;fixed._limits=bounds;fixed._maximum=maximum;return fixed
    affine=PreparedAvailableFieldEvidence.__new__(PreparedAvailableFieldEvidence);affine._handle=value.affine;affine._limits=bounds;affine._maximum=maximum;return affine


cdef AvailableFiniteMapDeclaration _map(value, vector[size_t] bounds, size_t maximum):
    _closed(value,('profile','key','revision','rules'))
    if value['profile']!='rexgraph.available-relational-finite-field-map.v1':raise ValueError('unknown available finite map profile')
    cdef AvailableFiniteMapDeclaration out
    cdef AvailableFiniteRule rule
    cdef AvailableFiniteInput item
    out.key=_natural(value['key']);out.revision=_natural(value['revision'])
    for row in value['rules']:
        _closed(row,('key','bindings','symbols','target','inputs','unsupported'))
        rule.key=_natural(row['key']);rule.bindings=_ids(row['bindings'],bounds[0]);rule.symbols=_ids(row['symbols'],bounds[0]);rule.has_target=row['target'] is not None
        rule.target=0 if row['target'] is None else _natural(row['target']);rule.inputs.clear()
        if not isinstance(row['unsupported'],str):raise TypeError('finite unsupported reason requires text')
        rule.unsupported=row['unsupported'].encode()
        for submission in row['inputs']:
            _closed(submission,('asset','accession','use','field','origins'))
            if not isinstance(submission['asset'],NativeInputUseAsset):raise TypeError('actual finite asset owner required; archive bytes are passive')
            item.asset=(<NativeInputUseAsset>submission['asset'])._pin();item.accession=_accession_declaration(submission['accession'],bounds[3]);item.use=_natural(submission['use'])
            item.field=_image(submission['field'],maximum);item.origins=_keys(submission['origins'],('asset','value','accession','reconstruction'));rule.inputs.push_back(item)
            if rule.inputs.size()>bounds[0]:raise ValueError('finite input submission count exceeds admission')
        out.rules.push_back(rule)
        if out.rules.size()>bounds[1]:raise ValueError('finite rule count exceeds admission')
    return out


def finite_constraint_origin_payload(NativeInputUseAsset asset not None, declaration, status, *, limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef FiniteDeclaration d=_declaration(declaration,bounds,maximum)
    cdef unsigned kind=_status(status)
    cdef shared_ptr[InputUseAssetBridge] actual=asset._pin()
    cdef string image
    with nogil:image=AvailableFiniteSourceBridge.constraint_payload(actual,d,kind,maximum)
    return <bytes>image


def finite_interpretation_origin_payload(PreparedAvailableFiniteSource source not None, targets, declaration, status,
        NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef vector[AvailableFiniteTarget] rows=_targets(targets,bounds)
    cdef AvailableFiniteMapDeclaration d=_map(declaration,bounds,maximum)
    cdef unsigned kind=_status(status)
    cdef shared_ptr[AvailableFiniteSourceBridge] actual=source._pin()
    cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
    cdef string image
    with nogil:image=AvailableFiniteMapBridge.interpretation_payload(actual,rows,d,kind,authority,bounds,maximum)
    return <bytes>image


cdef class PreparedAvailableFiniteSource:
    cdef shared_ptr[AvailableFiniteSourceBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)
    def __init__(self, NativeInputUseAsset asset not None, declaration, status, origins, NativeAvailabilityContext execution_context not None,
            *, limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('available finite source is immutable')
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef FiniteDeclaration d=_declaration(declaration,bounds,maximum)
        cdef unsigned kind=_status(status)
        cdef vector[uint64_t] keys=_keys(origins,('asset','constraints'))
        cdef shared_ptr[InputUseAssetBridge] owner=asset._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        self._loading=True
        try:
            with nogil:self._handle=shared_ptr[AvailableFiniteSourceBridge](new AvailableFiniteSourceBridge(owner,d,kind,keys,authority,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef PreparedAvailableFiniteSource out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:out._handle=AvailableFiniteSourceBridge.restore(image,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def raw(self):
        """Return the whole mathematical owner as an explicitly unadmitted escape."""
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=self._pin()
        cdef NativeFiniteSource out=NativeFiniteSource.__new__(NativeFiniteSource)
        with nogil:out._handle=actual.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def inspect(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=self._pin()
        cdef unsigned kind
        cdef vector[uint64_t] keys
        with nogil:kind=actual.get().status();keys=actual.get().origins()
        return dict(self.raw().inspect(max_records=max_records),status=_STATUSES[kind],origins=dict(zip(('asset', 'constraints'),tuple(keys),strict=True)))

    def same_owner(self, PreparedAvailableFiniteSource other not None):
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=self._pin(),second=other._pin()
        cdef bint equal
        with nogil:equal=actual.get().same_owner(second.get()[0])
        return bool(equal)

    def receipts(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] rows
        with nogil:rows=actual.get().receipts(maximum)
        return tuple(<bytes>row for row in rows)

    def dependency_archives(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] rows
        with nogil:rows=actual.get().dependency_archives(maximum)
        return tuple(<bytes>row for row in rows)

    def sections(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableFiniteSections out=NativeAvailableFiniteSections.__new__(NativeAvailableFiniteSections)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().sections(authority)
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeAvailableFiniteSections:
    cdef shared_ptr[AvailableFiniteSectionsBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteSectionsBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeAvailableFiniteSections out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:out._handle=AvailableFiniteSectionsBridge.restore(image,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[AvailableFiniteSectionsBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def raw(self):
        """Return the whole mathematical owner as an explicitly unadmitted escape."""
        cdef shared_ptr[AvailableFiniteSectionsBridge] actual=self._pin()
        cdef NativeFiniteSections out=NativeFiniteSections.__new__(NativeFiniteSections)
        with nogil:out._handle=actual.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def source(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteSectionsBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef PreparedAvailableFiniteSource out=PreparedAvailableFiniteSource.__new__(PreparedAvailableFiniteSource)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().source()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def select(self, begin, count, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteSectionsBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t first=_natural(begin),length=_natural(count)
        cdef vector[shared_ptr[AvailableFiniteSectionBridge]] rows
        cdef shared_ptr[AvailableFiniteSectionBridge] row
        cdef NativeAvailableFiniteSection out
        with nogil:rows=actual.get().select(first,length,authority)
        selected=[]
        for row in rows:
            out=NativeAvailableFiniteSection.__new__(NativeAvailableFiniteSection);out._handle=row;out._limits=self._limits;out._maximum=self._maximum;selected.append(out)
        return tuple(selected)

    def inspect(self, NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        return self.raw().inspect()


cdef class NativeAvailableFiniteSection:
    cdef shared_ptr[AvailableFiniteSectionBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteSectionBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    def family(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteSectionBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableFiniteSections out=NativeAvailableFiniteSections.__new__(NativeAvailableFiniteSections)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().family()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def inspect(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteSectionBridge] actual=self._pin()
        cdef size_t ordinal,maximum=_natural(max_records)
        cdef vector[uint64_t] assignment
        with nogil:ordinal=actual.get().ordinal();assignment=actual.get().assignment(maximum)
        return {'ordinal':ordinal,'assignment':tuple(assignment)}


cdef class PreparedAvailableFiniteFieldMap:
    cdef shared_ptr[AvailableFiniteMapBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)
    def __init__(self, PreparedAvailableFiniteSource source not None, targets, declaration, status, origins,
            NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('available finite map is immutable')
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef vector[AvailableFiniteTarget] rows=_targets(targets,bounds)
        cdef AvailableFiniteMapDeclaration d=_map(declaration,bounds,maximum)
        cdef unsigned kind=_status(status)
        cdef vector[uint64_t] keys=_keys(origins,('maps','relationship'))
        cdef shared_ptr[AvailableFiniteSourceBridge] owner=source._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        self._loading=True
        try:
            with nogil:self._handle=shared_ptr[AvailableFiniteMapBridge](new AvailableFiniteMapBridge(owner,rows,d,kind,keys,authority,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef PreparedAvailableFiniteFieldMap out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:out._handle=AvailableFiniteMapBridge.restore(image,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def raw(self):
        """Return the whole mathematical owner as an explicitly unadmitted escape."""
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin()
        cdef PreparedFiniteFieldMap out=PreparedFiniteFieldMap.__new__(PreparedFiniteFieldMap)
        with nogil:out._handle=actual.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def inspect(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin()
        cdef unsigned kind
        cdef vector[uint64_t] keys
        with nogil:kind=actual.get().status();keys=actual.get().origins()
        return dict(self.raw().inspect(max_records=max_records),status=_STATUSES[kind],origins=dict(zip(('maps', 'relationship'),tuple(keys),strict=True)))

    def same_owner(self, PreparedAvailableFiniteFieldMap other not None):
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin(),second=other._pin()
        cdef bint equal
        with nogil:equal=actual.get().same_owner(second.get()[0])
        return bool(equal)

    def receipts(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] rows
        with nogil:rows=actual.get().receipts(maximum)
        return tuple(<bytes>row for row in rows)

    def dependency_archives(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] rows
        with nogil:rows=actual.get().dependency_archives(maximum)
        return tuple(<bytes>row for row in rows)

    def source(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef PreparedAvailableFiniteSource out=PreparedAvailableFiniteSource.__new__(PreparedAvailableFiniteSource)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().source()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def targets(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[AvailableFiniteTarget] rows
        with nogil:rows=actual.get().targets(maximum)
        return tuple(_target(row,self._limits,self._maximum) for row in rows)

    def execute(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteMapBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableFiniteFieldResult out=NativeAvailableFiniteFieldResult.__new__(NativeAvailableFiniteFieldResult)
        with nogil:out._handle=actual.get().execute(authority)
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeAvailableFiniteFieldResult:
    cdef shared_ptr[AvailableFiniteResultBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeAvailableFiniteFieldResult out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:out._handle=AvailableFiniteResultBridge.restore(image,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[AvailableFiniteResultBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def raw(self):
        """Return the whole mathematical owner as an explicitly unadmitted escape."""
        cdef shared_ptr[AvailableFiniteResultBridge] actual=self._pin()
        cdef NativeFiniteFieldResult out=NativeFiniteFieldResult.__new__(NativeFiniteFieldResult)
        with nogil:out._handle=actual.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def model(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef PreparedAvailableFiniteFieldMap out=PreparedAvailableFiniteFieldMap.__new__(PreparedAvailableFiniteFieldMap)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().model()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def select(self, begin, count, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t first=_natural(begin),length=_natural(count)
        cdef vector[shared_ptr[AvailableFiniteBranchBridge]] rows
        cdef shared_ptr[AvailableFiniteBranchBridge] row
        cdef NativeAvailableFiniteFieldBranch out
        with nogil:rows=actual.get().select(first,length,authority)
        selected=[]
        for row in rows:
            out=NativeAvailableFiniteFieldBranch.__new__(NativeAvailableFiniteFieldBranch);out._handle=row;out._limits=self._limits;out._maximum=self._maximum;selected.append(out)
        return tuple(selected)

    def inspect(self, NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteResultBridge] actual=self._pin()
        cdef size_t count
        with nogil:count=actual.get().size()
        return {'branch_count':count}

    def completion_payload(self, ordinal, free_values, parameter_setting, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t index=_natural(ordinal)
        cdef uint64_t setting=_natural(parameter_setting)
        cdef vector[string] values=_free(free_values,self._limits)
        cdef string image
        with nogil:image=actual.get().completion_payload(index,values,setting,authority)
        return <bytes>image

    def evaluate(self, ordinal, free_values, parameter_setting, choice_origin, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t index=_natural(ordinal)
        cdef uint64_t setting=_natural(parameter_setting),choice=_natural(choice_origin)
        cdef vector[string] values=_free(free_values,self._limits)
        cdef NativeAvailableFiniteFieldEvaluation out=NativeAvailableFiniteFieldEvaluation.__new__(NativeAvailableFiniteFieldEvaluation)
        with nogil:out._handle=actual.get().evaluate(index,values,setting,choice,authority)
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeAvailableFiniteFieldBranch:
    cdef shared_ptr[AvailableFiniteBranchBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteBranchBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    def raw(self):
        """Return the whole mathematical owner as an explicitly unadmitted escape."""
        cdef shared_ptr[AvailableFiniteBranchBridge] actual=self._pin()
        cdef NativeFiniteFieldBranch out=NativeFiniteFieldBranch.__new__(NativeFiniteFieldBranch)
        with nogil:out._handle=actual.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def family(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteBranchBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableFiniteFieldResult out=NativeAvailableFiniteFieldResult.__new__(NativeAvailableFiniteFieldResult)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().family()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def inspect(self, NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        return self.raw().inspect()


cdef class NativeAvailableFiniteFieldEvaluation:
    cdef shared_ptr[AvailableFiniteEvaluationBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteEvaluationBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeAvailableFiniteFieldEvaluation out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:out._handle=AvailableFiniteEvaluationBridge.restore(image,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[AvailableFiniteEvaluationBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def raw(self):
        """Return the whole mathematical owner as an explicitly unadmitted escape."""
        cdef shared_ptr[AvailableFiniteEvaluationBridge] actual=self._pin()
        cdef NativeFiniteFieldEvaluation out=NativeFiniteFieldEvaluation.__new__(NativeFiniteFieldEvaluation)
        with nogil:out._handle=actual.get().raw()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def dependency_archives(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteEvaluationBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] rows
        with nogil:rows=actual.get().dependency_archives(maximum)
        return tuple(<bytes>row for row in rows)

    def family(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteEvaluationBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableFiniteFieldResult out=NativeAvailableFiniteFieldResult.__new__(NativeAvailableFiniteFieldResult)
        with nogil:
            actual.get().admit(authority)
            out._handle=actual.get().family()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def inspect(self, NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteEvaluationBridge] actual=self._pin()
        cdef size_t ordinal
        cdef string setting,choice
        with nogil:ordinal=actual.get().ordinal();setting=actual.get().setting_receipt();choice=actual.get().choice_receipt()
        return {'ordinal':ordinal,'parameter_setting_receipt':<bytes>setting if setting.size() else None,'choice_receipt':<bytes>choice}

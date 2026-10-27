# distutils: language=c++
"""Native whole-field task postconditions with explicit current authority."""
from cpython.bytes cimport PyBytes_FromStringAndSize
from ._field_interface cimport (
    NativeInputUseAsset, InterfaceNodeCell,
    _bounds, _maximum, _matrix, _image, _publication,
)
from ._field_admission cimport NativeAvailableFieldResult, NativeAvailabilityContext, NativeValueOrigin
from ._field_evidence_admission cimport NativeAvailableAffineResult
from ._known_tensor_graph cimport _Check, _check, _callback, _raise_check
from ._stalk_fields import _natural
from ._field_program import _closed

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_TASK_OUTCOME_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)
_STATUS=('satisfied','open','refuted')

def task_outcome_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
        'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_TASK_OUTCOME_BINDING_DIGEST.decode(),
        'profile':'owning-relational-task-outcome-v1','scalar_policy':'gmp-rational-exact'}

cdef TaskDeclaration _task(value, vector[size_t] bounds, size_t maximum):
    _closed(value,('profile','key','revision','use','origin','basis','directions','metric','target'))
    if value['profile']!='rexgraph.relational-task-postcondition.v1':raise ValueError('unknown task profile')
    cdef TaskDeclaration out
    out.key=_natural(value['key']);out.revision=_natural(value['revision']);out.use=_natural(value['use'])
    out.origin=_natural(value['origin']);out.basis=_natural(value['basis'])
    for key in value['directions']:
        out.directions.push_back(_natural(key))
        if out.directions.size()>bounds[0]:raise ValueError('task direction admission exceeded')
    out.metric=_matrix(value['metric'],bounds[3]);out.target=_image(value['target'],maximum)
    return out

cdef TaskBinding _binding(value,vector[size_t] bounds):
    _closed(value,('profile','key','revision','worker','machine','operation','origin','interface','view'))
    if value['profile']!='rexgraph.relational-worker-binding.v1':raise ValueError('unknown worker binding profile')
    cdef TaskBinding out
    cdef InterfaceNodeCell cell
    out.key=_natural(value['key']);out.revision=_natural(value['revision']);out.worker=_natural(value['worker'])
    out.machine=_natural(value['machine']);out.operation=_natural(value['operation']);out.origin=_natural(value['origin'])
    patch=value['interface'];_closed(patch,('key','branch','cells'))
    out.interface.key=_natural(patch['key']);out.interface.branch=_natural(patch['branch'])
    for node,coordinate in patch['cells']:
        if len(coordinate)!=2:raise ValueError('task patch requires grade and cell')
        cell.node=_natural(node);cell.grade=_natural(coordinate[0]);cell.cell=_natural(coordinate[1]);out.interface.cells.push_back(cell)
        if out.interface.cells.size()>bounds[0]:raise ValueError('task patch admission exceeded')
    view=value['view'];_closed(view,('profile','key','basis','directions','accession','reconstruction','physical'))
    if view['profile']!='rexgraph.relational-field-view.v1':raise ValueError('unknown task view profile')
    out.view.key=_natural(view['key']);out.view.basis=_natural(view['basis'])
    for key in view['directions']:
        out.view.directions.push_back(_natural(key))
        if out.view.directions.size()>bounds[0]:raise ValueError('task view admission exceeded')
    out.view.accession=_matrix(view['accession'],bounds[3]);out.view.reconstruction=_matrix(view['reconstruction'],bounds[3])
    out.view.has_physical=view['physical'] is not None
    if out.view.has_physical:out.view.physical=_matrix(view['physical'],bounds[3])
    return out

cdef void _response(value,shared_ptr[AvailableResultBridge]& field,shared_ptr[AvailableAffineBridge]& family) except *:
    cdef NativeAvailableFieldResult actual
    cdef NativeAvailableAffineResult affine
    if isinstance(value,NativeAvailableFieldResult):actual=value;field=actual._pin()
    elif isinstance(value,NativeAvailableAffineResult):affine=value;family=affine._pin()
    else:raise TypeError('task outcome requires an actual available execution or affine result owner')

def task_goal_origin_payload(task,NativeInputUseAsset asset not None,response,*,limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef TaskDeclaration d=_task(task,bounds,maximum)
    cdef shared_ptr[InputUseAssetBridge] original=asset._pin()
    cdef shared_ptr[AvailableResultBridge] field
    cdef shared_ptr[AvailableAffineBridge] family
    cdef string image
    _response(response,field,family)
    with nogil:image=TaskOutcomeBridge.goal_payload(d,original,field,family,bounds,maximum)
    return <bytes>image

def task_binding_origin_payload(binding,response,*,limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef TaskBinding d=_binding(binding,bounds)
    cdef shared_ptr[AvailableResultBridge] field
    cdef shared_ptr[AvailableAffineBridge] family
    cdef string image
    _response(response,field,family)
    with nogil:image=TaskOutcomeBridge.binding_payload(d,field,family,bounds,maximum)
    return <bytes>image

cdef class _Sink:
    pass

cdef int _write(void* context,const char* data,size_t count) noexcept with gil:
    cdef _Sink sink=<_Sink><object>context
    try:
        returned=sink.write(PyBytes_FromStringAndSize(data,count))
        if returned is not None and (type(returned) is not int or returned!=count):
            raise OSError('task packet sink performed a short write')
    except BaseException as error:sink.error=error;return 0
    return 1

cdef class NativeTaskOutcome:
    cdef shared_ptr[TaskOutcomeBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('native task outcome is not initialized')
        return self._handle

    @property
    def limits(self):
        self._pin()
        return tuple(self._limits)

    @property
    def max_bytes(self):
        self._pin()
        return int(self._maximum)

    def __init__(self,task,NativeInputUseAsset asset not None,binding,response,
                 NativeAvailabilityContext execution_context not None,*,limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        if self._loading or self._handle.get()!=NULL:raise TypeError('native task outcome owners are immutable')
        self._loading=True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef TaskDeclaration t
        cdef TaskBinding b
        cdef shared_ptr[InputUseAssetBridge] original
        cdef shared_ptr[AvailableResultBridge] field
        cdef shared_ptr[AvailableAffineBridge] family
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[TaskOutcomeBridge] loaded
        cdef _Check state
        cdef GraphCheck callback
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);t=_task(task,bounds,maximum);b=_binding(binding,bounds)
            original=asset._pin();authority=execution_context._pin();_response(response,field,family)
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=shared_ptr[TaskOutcomeBridge](new TaskOutcomeBridge(t,original,b,field,family,authority,bounds,maximum,callback))
            except BaseException:_raise_check(state);raise
            self._handle=loaded;self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls,archive,NativeAvailabilityContext execution_context not None,*,limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        cdef NativeTaskOutcome out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:out._handle=TaskOutcomeBridge.restore(image,authority,bounds,maximum,callback)
        except BaseException:_raise_check(state);raise
        out._limits=bounds;out._maximum=maximum
        return out

    def admit(self,NativeAvailabilityContext execution_context not None,*,check=None):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:pin.get().admit(authority,callback)
        except BaseException:_raise_check(state);raise

    def archive(self,NativeAvailabilityContext execution_context not None,*,check=None):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef string image
        try:
            with nogil:image=pin.get().archive(authority,callback)
        except BaseException:_raise_check(state);raise
        return <bytes>image

    def stream(self,consume,NativeAvailabilityContext execution_context not None,*,check=None):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef _Sink sink=_Sink.__new__(_Sink)
        sink.write=consume;sink.error=None
        if not callable(consume):raise TypeError('task stream consumer must be callable')
        cdef TaskSink target
        target.write=_write;target.context=<void*>sink
        cdef size_t count
        try:
            with nogil:count=pin.get().stream(target,authority,callback)
        except BaseException:
            if sink.error is not None:raise sink.error
            _raise_check(state);raise
        return int(count)

    def inspect(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef GraphCheck check
        cdef vector[uint64_t] values
        cdef unsigned status
        cdef cppbool executed
        with nogil:
            values=pin.get().identity(authority,check);status=pin.get().status(authority,check);executed=pin.get().executed()
        return dict(zip(('key','revision','use','goal_origin','basis','binding','binding_revision','worker','machine','operation','binding_origin'),
                        tuple(int(v) for v in values)),status=_STATUS[status],executed_response=bool(executed))

    def certificate(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef GraphCheck check
        cdef TaskCertificate p
        with nogil:p=pin.get().certificate(authority,check)
        return {'status':_STATUS[p.status],'rank':int(p.rank),'residual':<bytes>p.residual,
            'directions':<bytes>p.directions,'feasible_shift':<bytes>p.feasible_shift,
            'feasible_kernel':<bytes>p.feasible_kernel,'contradiction':<bytes>p.contradiction,
            'encoding':'rfc-native-exact-field-matrix-vector-v1'}

    def response(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef GraphCheck check
        cdef shared_ptr[AvailableResultBridge] field
        cdef shared_ptr[AvailableAffineBridge] family
        cdef NativeAvailableFieldResult raw
        cdef NativeAvailableAffineResult affine
        with nogil:field=pin.get().field(authority,check)
        if field.get()!=NULL:
            raw=NativeAvailableFieldResult.__new__(NativeAvailableFieldResult);raw._handle=field
            raw._limits=self._limits;raw._maximum=self._maximum;return raw
        with nogil:family=pin.get().family(authority,check)
        affine=NativeAvailableAffineResult.__new__(NativeAvailableAffineResult);affine._handle=family
        affine._limits=self._limits;affine._maximum=self._maximum;return affine

    def source_archive(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef GraphCheck check
        cdef ProgramSourceInfo info
        with nogil:info=pin.get().source(authority,check)
        return <bytes>info.archive

    def definition(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef GraphCheck check
        cdef TaskDetails d
        with nogil:d=pin.get().details(authority,check)
        return {'interface':int(d.interface_key),'branch':int(d.branch),'view':int(d.view),
            'directions':tuple(int(k) for k in d.directions),
            'cells':tuple((int(c.node),int(c.grade),int(c.cell)) for c in d.cells),
            'target':<bytes>d.target,'metric':<bytes>d.metric,'accession':<bytes>d.accession,
            'reconstruction':<bytes>d.reconstruction,'encoding':'rfc-native-exact-field-matrix-vector-v1'}

    def dependency_archives(self,NativeAvailabilityContext execution_context not None,*,max_records=65536):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef GraphCheck check
        cdef size_t maximum=_maximum(max_records)
        cdef vector[string] values
        with nogil:values=pin.get().dependencies(authority,maximum,check)
        return tuple(<bytes>value for value in values)

    def origins(self,NativeAvailabilityContext execution_context not None,*,max_records=65536):
        cdef shared_ptr[TaskOutcomeBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef GraphCheck check
        cdef size_t maximum=_maximum(max_records)
        cdef vector[shared_ptr[ValueOriginBridge]] values
        cdef NativeValueOrigin origin
        with nogil:values=pin.get().origins(authority,maximum,check)
        out=[]
        for value in values:
            origin=NativeValueOrigin.__new__(NativeValueOrigin);origin._handle=value;out.append(origin)
        return tuple(out)

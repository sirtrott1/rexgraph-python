# distutils: language=c++
"""Whole joint field owners with explicit original publishers and current grants."""
from cpython.bytes cimport PyBytes_FromStringAndSize
from ._field_interface cimport NativeLiteralInput, PreparedAccessionFieldProgram, _bounds, _maximum, _image
from ._field_evidence cimport PreparedFieldFamily
from ._field_evidence_admission cimport PreparedAvailableFieldEvidence
from ._field_admission cimport NativeAvailabilityContext, NativeValueOrigin, _keys
from ._field_admission import _ROLES
from ._known_tensor_graph cimport NativeKnownTensorGraphInput, _Check, _check, _callback, _raise_check
from ._literal_field_family cimport NativeLiteralJointFieldResult, NativeLiteralJointFieldEvaluation
from ._field_program import _closed
from ._stalk_fields import _natural, _scalar

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_AVAILABLE_LITERAL_FIELD_FAMILY_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)
_INPUT_KEYS=('asset','value','accession','reconstruction')
_PROGRAM_KEYS=('program','maps','metrics')

def available_literal_field_family_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_AVAILABLE_LITERAL_FIELD_FAMILY_BINDING_DIGEST.decode(),
            'profile':'available-literal-joint-field-family-v1','scalar_policy':'gmp-rational-exact'}

cdef vector[string] _free(values, vector[size_t] bounds):
    cdef vector[string] out
    for value in values:
        out.push_back(_scalar(value,bounds[3]))
        if out.size()>bounds[0]:raise ValueError('complete joint free setting exceeds admission')
    return out

cdef JointReaderInput _reader(value):
    cdef JointReaderInput out
    cdef NativeLiteralInput literal
    cdef NativeKnownTensorGraphInput graph
    cdef vector[uint64_t] keys
    _closed(value,('input','origins'))
    obj=value['input']
    if isinstance(obj,NativeLiteralInput):literal=obj;out.input.literal=literal._pin()
    elif isinstance(obj,NativeKnownTensorGraphInput):graph=obj;out.input.graph=graph._pin()
    else:raise TypeError('actual literal or known-graph input owner required')
    origins=value['origins'];_closed(origins,('input','original_program','original_input','reader'))
    out.origins=_keys(origins['input'],_INPUT_KEYS)
    keys=_keys(origins['original_program'],_PROGRAM_KEYS) if origins['original_program'] is not None else [0,0,0]
    for key in keys:out.origins.push_back(key)
    keys=_keys(origins['original_input'],_INPUT_KEYS) if origins['original_input'] is not None else [0,0,0,0]
    for key in keys:out.origins.push_back(key)
    out.origins.push_back(_natural(origins['reader']))
    return out

cdef vector[JointReaderInput] _inputs(values, vector[size_t] bounds):
    cdef vector[JointReaderInput] out
    for value in values:
        out.push_back(_reader(value))
        if out.size()>bounds[0]:raise ValueError('complete joint reader input count exceeds admission')
    return out

cdef vector[vector[JointObservationReaders]] _ledger(values, vector[size_t] bounds):
    cdef vector[vector[JointObservationReaders]] out
    cdef vector[JointObservationReaders] member
    cdef JointObservationReaders row
    cdef size_t count=0
    for rows in values:
        member.clear();count+=1
        for value in rows:
            _closed(value,('observation','inputs'))
            row.observation=_natural(value['observation']);row.inputs=_inputs(value['inputs'],bounds)
            count+=1+row.inputs.size()
            if count>bounds[1]:raise ValueError('complete training reader ledger exceeds admission')
            member.push_back(row)
        out.push_back(member)
        if out.size()>bounds[0]:raise ValueError('joint family member count exceeds admission')
    return out

cdef vector[shared_ptr[AvailableEvidenceBridge]] _members(values, vector[size_t] bounds):
    cdef vector[shared_ptr[AvailableEvidenceBridge]] out
    cdef PreparedAvailableFieldEvidence member
    for value in values:
        if not isinstance(value,PreparedAvailableFieldEvidence):raise TypeError('actual available evidence member required')
        member=value
        if member._limits!=bounds:raise ValueError('available member exact action policy differs')
        out.push_back(member._pin())
        if out.size()>bounds[0]:raise ValueError('available member count exceeds admission')
    return out

cdef vector[uint64_t] _interfaces(values, vector[size_t] bounds):
    cdef vector[uint64_t] out
    for value in values:
        out.push_back(_natural(value))
        if out.size()>bounds[0]:raise ValueError('complete family interface count exceeds admission')
    return out

cdef object _wrap_reader(JointReaderInput row, vector[size_t] bounds, size_t maximum):
    cdef NativeLiteralInput literal
    cdef NativeKnownTensorGraphInput graph
    if row.input.literal.get()!=NULL:
        literal=NativeLiteralInput.__new__(NativeLiteralInput)
        literal._limits=bounds;literal._maximum=maximum;literal._handle=row.input.literal;value=literal
    else:
        graph=NativeKnownTensorGraphInput.__new__(NativeKnownTensorGraphInput)
        graph._limits=bounds;graph._maximum=maximum;graph._handle=row.input.graph;value=graph
    keys=tuple(int(k) for k in row.origins)
    return {'input':value,'origins':{'input':dict(zip(_INPUT_KEYS,keys[:4],strict=True)),
        'original_program':dict(zip(_PROGRAM_KEYS,keys[4:7],strict=True)) if any(keys[4:7]) else None,
        'original_input':dict(zip(_INPUT_KEYS,keys[7:11],strict=True)) if any(keys[7:11]) else None,'reader':keys[11]}}

cdef object _origins(vector[shared_ptr[ValueOriginBridge]] values):
    cdef NativeValueOrigin out
    result=[]
    for value in values:
        out=NativeValueOrigin.__new__(NativeValueOrigin);out._handle=value;result.append(out)
    return tuple(result)

cdef NativeJointFieldPacket _packet(shared_ptr[AvailableJointPacketBridge] handle):
    cdef NativeJointFieldPacket out=NativeJointFieldPacket.__new__(NativeJointFieldPacket)
    out._handle=handle;return out

cdef class _Sink:
    pass

cdef int _write(void* context,const char* data,size_t count) noexcept with gil:
    cdef _Sink sink=<_Sink><object>context
    try:
        result=sink.write(PyBytes_FromStringAndSize(data,count))
        if result is not None and result!=count:raise OSError('joint packet sink performed a short write')
    except BaseException as error:
        sink.error=error;return 0
    return 1

cdef class NativeJointFieldPacket:
    """An immutable native packet whose byte access rechecks retained origins."""
    def __init__(self):raise TypeError('joint packets are returned by native preparation')
    cdef shared_ptr[AvailableJointPacketBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('joint packet is not initialized')
        return self._handle

    @property
    def size(self):
        cdef shared_ptr[AvailableJointPacketBridge] pin=self._pin()
        cdef size_t count
        with nogil:count=pin.get().size()
        return int(count)

    def to_bytes(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointPacketBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef string image
        try:
            with nogil:image=pin.get().bytes(authority,callback)
        except BaseException:_raise_check(state);raise
        return <bytes>image

    def stream(self, write, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointPacketBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        if not callable(write):raise TypeError('joint packet write must be callable')
        cdef _Sink sink=_Sink.__new__(_Sink)
        sink.write=write;sink.error=None
        cdef JointSink target
        target.write=_write;target.context=<void*>sink
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef size_t count
        try:
            with nogil:count=pin.get().stream(target,authority,callback)
        except BaseException:
            if sink.error is not None:raise sink.error
            _raise_check(state);raise
        return int(count)

def joint_packet_value_origin(key, history, cutoff, role, NativeJointFieldPacket packet not None,
        NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024,
        admission_limits=(4096,64*1024*1024,256*1024*1024),check=None):
    """Publish a packet only at the explicitly supplied publisher and cutoff."""
    if role not in _ROLES:raise ValueError('explicit native dependency role required')
    cdef uint64_t original=_natural(key),sequence=_natural(cutoff)
    cdef unsigned selected=_ROLES.index(role)
    cdef vector[size_t] bounds=_bounds(limits),admission
    for value in admission_limits:admission.push_back(_natural(value))
    if admission.size()!=3:raise ValueError('three origin admission bounds required')
    cdef size_t maximum=_maximum(max_bytes)
    cdef string source=_image(history,maximum)
    cdef shared_ptr[AvailableJointPacketBridge] pin=packet._pin()
    cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
    cdef _Check state=_check(check)
    cdef GraphCheck callback=_callback(state)
    cdef shared_ptr[ValueOriginBridge] loaded
    cdef NativeValueOrigin out
    try:
        with nogil:loaded=pin.get().publish(original,source,sequence,selected,authority,bounds,maximum,admission,callback)
    except BaseException:_raise_check(state);raise
    out=NativeValueOrigin.__new__(NativeValueOrigin);out._handle=loaded;return out

def joint_family_interface_payload(PreparedFieldFamily family not None, interface, *, max_bytes=256*1024*1024,check=None):
    cdef shared_ptr[FieldFamilyBridge] actual=family._pin()
    cdef size_t index=_natural(interface),maximum=_maximum(max_bytes)
    cdef _Check state=_check(check)
    cdef GraphCheck callback=_callback(state)
    cdef shared_ptr[AvailableJointPacketBridge] loaded
    try:
        with nogil:loaded=AvailableJointFamilyBridge.interface_payload(actual,index,maximum,callback)
    except BaseException:_raise_check(state);raise
    return _packet(loaded)

def joint_family_origin_payload(PreparedFieldFamily family not None, members, training_readers, interface_origins,
        NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef shared_ptr[FieldFamilyBridge] actual=family._pin()
    cdef vector[shared_ptr[AvailableEvidenceBridge]] values=_members(members,bounds)
    cdef vector[vector[JointObservationReaders]] rows=_ledger(training_readers,bounds)
    cdef vector[uint64_t] keys=_interfaces(interface_origins,bounds)
    cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
    cdef _Check state=_check(check)
    cdef GraphCheck callback=_callback(state)
    cdef shared_ptr[AvailableJointPacketBridge] loaded
    try:
        with nogil:loaded=AvailableJointFamilyBridge.model_payload(actual,values,rows,keys,authority,bounds,maximum,callback)
    except BaseException:_raise_check(state);raise
    return _packet(loaded)

def replay_available_joint_family(PreparedFieldFamily blueprint not None, member_origins, training_readers,
        NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
    """Rebuild available evidence with its original append batches and parents."""
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef shared_ptr[FieldFamilyBridge] original=blueprint._pin()
    cdef vector[JointMemberOrigins] recipes
    cdef JointMemberOrigins row
    cdef vector[uint64_t] target
    for value in member_origins:
        _closed(value,('program','root','targets'))
        row.program=_keys(value['program'],_PROGRAM_KEYS);row.root=_natural(value['root']);row.targets.clear()
        for observed in value['targets']:
            target=_keys(observed,('asset','value','port'));row.targets.push_back(target)
            if row.targets.size()>bounds[0]:raise ValueError('complete original target ledger exceeds admission')
        recipes.push_back(row)
        if recipes.size()>bounds[0]:raise ValueError('complete member replay recipe exceeds admission')
    cdef vector[vector[JointObservationReaders]] rows=_ledger(training_readers,bounds)
    cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
    cdef _Check state=_check(check)
    cdef GraphCheck callback=_callback(state)
    cdef JointAvailableReplay loaded
    cdef PreparedFieldFamily family
    cdef PreparedAvailableFieldEvidence member
    try:
        with nogil:loaded=rebuild_available_field_family(original,recipes,rows,authority,bounds,maximum,callback)
    except BaseException:_raise_check(state);raise
    family=PreparedFieldFamily.__new__(PreparedFieldFamily);family._limits=bounds;family._maximum=maximum;family._handle=loaded.family
    result=[]
    for handle in loaded.members:
        member=PreparedAvailableFieldEvidence.__new__(PreparedAvailableFieldEvidence)
        member._limits=bounds;member._maximum=maximum;member._handle=handle;result.append(member)
    return family,tuple(result)

def replay_joint_reader_input(value, PreparedAccessionFieldProgram program not None, *, limits=_LIMITS,
        max_bytes=256*1024*1024,check=None):
    """Replay a retained accession declaration on the same complete program image."""
    cdef LiteralFamilyInput original,loaded
    cdef NativeLiteralInput literal
    cdef NativeKnownTensorGraphInput graph
    if isinstance(value,NativeLiteralInput):literal=value;original.literal=literal._pin()
    elif isinstance(value,NativeKnownTensorGraphInput):graph=value;original.graph=graph._pin()
    else:raise TypeError('actual literal or known-graph input owner required')
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef shared_ptr[FieldProgramBridge] target=program._pin()
    cdef _Check state=_check(check)
    cdef GraphCheck callback=_callback(state)
    try:
        with nogil:loaded=_replay_joint_reader_input(original,target,bounds,maximum,callback)
    except BaseException:_raise_check(state);raise
    if loaded.literal.get()!=NULL:
        literal=NativeLiteralInput.__new__(NativeLiteralInput);literal._limits=bounds;literal._maximum=maximum;literal._handle=loaded.literal;return literal
    graph=NativeKnownTensorGraphInput.__new__(NativeKnownTensorGraphInput);graph._limits=bounds;graph._maximum=maximum;graph._handle=loaded.graph;return graph

cdef class PreparedAvailableLiteralFieldFamily:
    cdef shared_ptr[AvailableJointFamilyBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('available joint owner is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        cdef PreparedAvailableLiteralFieldFamily out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[AvailableJointFamilyBridge] loaded
        cdef _Check state
        cdef GraphCheck callback
        if out._loading or out._handle.get()!=NULL:raise TypeError('available cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum);authority=execution_context._pin()
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=AvailableJointFamilyBridge.restore(image,authority,bounds,maximum,callback)
            except BaseException:_raise_check(state);raise
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:pin.get().admit(authority,callback)
        except BaseException:_raise_check(state);raise

    def prepare(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableJointPacketBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:loaded=pin.get().prepare(authority,callback)
        except BaseException:_raise_check(state);raise
        return _packet(loaded)

    def archive(self, NativeAvailabilityContext execution_context not None, *, check=None):
        return self.prepare(execution_context,check=check).to_bytes(execution_context,check=check)

    def __init__(self, PreparedFieldFamily family not None, members, training_readers, origins,
            NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        if self._loading or self._handle.get()!=NULL:raise TypeError('available joint model is immutable')
        self._loading=True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef shared_ptr[FieldFamilyBridge] original
        cdef vector[shared_ptr[AvailableEvidenceBridge]] values
        cdef vector[vector[JointObservationReaders]] rows
        cdef vector[uint64_t] interfaces
        cdef uint64_t origin
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[AvailableJointFamilyBridge] loaded
        cdef _Check state
        cdef GraphCheck callback
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);original=family._pin()
            values=_members(members,bounds);rows=_ledger(training_readers,bounds)
            _closed(origins,('interfaces','family'));interfaces=_interfaces(origins['interfaces'],bounds);origin=_natural(origins['family'])
            authority=execution_context._pin();state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=shared_ptr[AvailableJointFamilyBridge](new AvailableJointFamilyBridge(original,values,rows,interfaces,origin,authority,bounds,maximum,callback))
            except BaseException:_raise_check(state);raise
            self._limits=bounds;self._maximum=maximum;self._handle=loaded
        finally:self._loading=False

    def same_owner(self, PreparedAvailableLiteralFieldFamily other not None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin(),actual=other._pin()
        cdef cppbool same
        with nogil:same=pin.get().same_owner(actual.get()[0])
        return bool(same)

    def family(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[FieldFamilyBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef PreparedFieldFamily out
        try:
            with nogil:loaded=pin.get().family(authority,callback)
        except BaseException:_raise_check(state);raise
        out=PreparedFieldFamily.__new__(PreparedFieldFamily);out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def members(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef vector[shared_ptr[AvailableEvidenceBridge]] loaded
        cdef PreparedAvailableFieldEvidence out
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:loaded=pin.get().members(authority,callback)
        except BaseException:_raise_check(state);raise
        result=[]
        for value in loaded:
            out=PreparedAvailableFieldEvidence.__new__(PreparedAvailableFieldEvidence)
            out._limits=self._limits;out._maximum=self._maximum;out._handle=value;result.append(out)
        return tuple(result)

    def training_readers(self, NativeAvailabilityContext execution_context not None, *, max_records=4096,check=None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[vector[JointObservationReaders]] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:loaded=pin.get().training_readers(authority,maximum,callback)
        except BaseException:_raise_check(state);raise
        return tuple(tuple({'observation':int(row.observation),'inputs':tuple(_wrap_reader(v,self._limits,self._maximum) for v in row.inputs)} for row in member) for member in loaded)

    def origins(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef vector[uint64_t] keys
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:keys=pin.get().origin_keys(authority,callback)
        except BaseException:_raise_check(state);raise
        return {'interfaces':tuple(int(keys[i]) for i in range(keys.size()-1)),'family':int(keys.back())}

    def obligations(self, NativeAvailabilityContext execution_context not None, *, max_records=4096,check=None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[shared_ptr[ValueOriginBridge]] values
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:values=pin.get().obligations(authority,maximum,callback)
        except BaseException:_raise_check(state);raise
        return _origins(values)

    def setting_payload(self, free_values, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef vector[string] values=_free(free_values,self._limits)
        cdef shared_ptr[AvailableJointPacketBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:loaded=pin.get().setting(values,authority,callback)
        except BaseException:_raise_check(state);raise
        return _packet(loaded)

    def execute(self, member, inputs, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointFamilyBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t index=_natural(member)
        cdef vector[JointReaderInput] rows=_inputs(inputs,self._limits)
        cdef shared_ptr[AvailableJointResultBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef NativeAvailableLiteralJointFieldResult out
        try:
            with nogil:loaded=pin.get().execute(index,rows,authority,callback)
        except BaseException:_raise_check(state);raise
        out=NativeAvailableLiteralJointFieldResult.__new__(NativeAvailableLiteralJointFieldResult)
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

cdef class NativeAvailableLiteralJointFieldResult:
    cdef shared_ptr[AvailableJointResultBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('available joint owner is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        cdef NativeAvailableLiteralJointFieldResult out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[AvailableJointResultBridge] loaded
        cdef _Check state
        cdef GraphCheck callback
        if out._loading or out._handle.get()!=NULL:raise TypeError('available cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum);authority=execution_context._pin()
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=AvailableJointResultBridge.restore(image,authority,bounds,maximum,callback)
            except BaseException:_raise_check(state);raise
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointResultBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:pin.get().admit(authority,callback)
        except BaseException:_raise_check(state);raise

    def prepare(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointResultBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableJointPacketBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:loaded=pin.get().prepare(authority,callback)
        except BaseException:_raise_check(state);raise
        return _packet(loaded)

    def archive(self, NativeAvailabilityContext execution_context not None, *, check=None):
        return self.prepare(execution_context,check=check).to_bytes(execution_context,check=check)

    def __init__(self):raise TypeError('available joint results come from native execution or cold restoration')

    def model(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointResultBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableJointFamilyBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef PreparedAvailableLiteralFieldFamily out
        try:
            with nogil:loaded=pin.get().model(authority,callback)
        except BaseException:_raise_check(state);raise
        out=PreparedAvailableLiteralFieldFamily.__new__(PreparedAvailableLiteralFieldFamily)
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def raw_result(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointResultBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[LiteralJointResultBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef NativeLiteralJointFieldResult out
        try:
            with nogil:loaded=pin.get().raw(authority,callback)
        except BaseException:_raise_check(state);raise
        out=NativeLiteralJointFieldResult.__new__(NativeLiteralJointFieldResult)
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def inputs(self, NativeAvailabilityContext execution_context not None, *, max_records=256,check=None):
        cdef shared_ptr[AvailableJointResultBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[JointReaderInput] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:loaded=pin.get().inputs(authority,maximum,callback)
        except BaseException:_raise_check(state);raise
        return tuple(_wrap_reader(v,self._limits,self._maximum) for v in loaded)

    def obligations(self, NativeAvailabilityContext execution_context not None, *, max_records=4096,check=None):
        cdef shared_ptr[AvailableJointResultBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[shared_ptr[ValueOriginBridge]] values
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:values=pin.get().obligations(authority,maximum,callback)
        except BaseException:_raise_check(state);raise
        return _origins(values)

    def completion_payload(self, free_values, setting, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointResultBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef vector[string] values=_free(free_values,self._limits)
        cdef uint64_t key=_natural(setting)
        cdef shared_ptr[AvailableJointPacketBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:loaded=pin.get().completion(values,key,authority,callback)
        except BaseException:_raise_check(state);raise
        return _packet(loaded)

    def evaluate(self, free_values, setting, completion, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointResultBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef vector[string] values=_free(free_values,self._limits)
        cdef uint64_t selected=_natural(setting),decision=_natural(completion)
        cdef shared_ptr[AvailableJointEvaluationBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef NativeAvailableLiteralJointFieldEvaluation out
        try:
            with nogil:loaded=pin.get().evaluate(values,selected,decision,authority,callback)
        except BaseException:_raise_check(state);raise
        out=NativeAvailableLiteralJointFieldEvaluation.__new__(NativeAvailableLiteralJointFieldEvaluation)
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def fields(self, execution_context, *, max_records=256,check=None):return self.raw_result(execution_context,check=check).fields(max_records=max_records)
    def field_archive(self,node,execution_context, *, occurrences=False,check=None):return self.raw_result(execution_context,check=check).field_archive(node,occurrences=occurrences)
    def select(self,node,cells,execution_context, *, coordinate_limit=256,check=None):return self.raw_result(execution_context,check=check).select(node,cells,coordinate_limit=coordinate_limit)

cdef class NativeAvailableLiteralJointFieldEvaluation:
    cdef shared_ptr[AvailableJointEvaluationBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('available joint owner is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        cdef NativeAvailableLiteralJointFieldEvaluation out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[AvailableJointEvaluationBridge] loaded
        cdef _Check state
        cdef GraphCheck callback
        if out._loading or out._handle.get()!=NULL:raise TypeError('available cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum);authority=execution_context._pin()
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=AvailableJointEvaluationBridge.restore(image,authority,bounds,maximum,callback)
            except BaseException:_raise_check(state);raise
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointEvaluationBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:pin.get().admit(authority,callback)
        except BaseException:_raise_check(state);raise

    def prepare(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointEvaluationBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableJointPacketBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:loaded=pin.get().prepare(authority,callback)
        except BaseException:_raise_check(state);raise
        return _packet(loaded)

    def archive(self, NativeAvailabilityContext execution_context not None, *, check=None):
        return self.prepare(execution_context,check=check).to_bytes(execution_context,check=check)

    def __init__(self):raise TypeError('available joint evaluations come from native decisions or cold restoration')

    def parent(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointEvaluationBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableJointResultBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef NativeAvailableLiteralJointFieldResult out
        try:
            with nogil:loaded=pin.get().parent(authority,callback)
        except BaseException:_raise_check(state);raise
        out=NativeAvailableLiteralJointFieldResult.__new__(NativeAvailableLiteralJointFieldResult)
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def raw_evaluation(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointEvaluationBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[LiteralJointEvaluationBridge] loaded
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef NativeLiteralJointFieldEvaluation out
        try:
            with nogil:loaded=pin.get().raw(authority,callback)
        except BaseException:_raise_check(state);raise
        out=NativeLiteralJointFieldEvaluation.__new__(NativeLiteralJointFieldEvaluation)
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def decision_origins(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AvailableJointEvaluationBridge] pin=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef vector[shared_ptr[ValueOriginBridge]] values
        cdef shared_ptr[ValueOriginBridge] value
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef unsigned i
        try:
            for i in range(2):
                with nogil:value=pin.get().origin(i,authority,callback)
                values.push_back(value)
        except BaseException:_raise_check(state);raise
        return _origins(values)

    def fields(self, execution_context, *, max_records=256,check=None):return self.raw_evaluation(execution_context,check=check).fields(max_records=max_records)
    def field_archive(self,node,execution_context, *, occurrences=False,check=None):return self.raw_evaluation(execution_context,check=check).field_archive(node,occurrences=occurrences)
    def select(self,node,cells,execution_context, *, coordinate_limit=256,check=None):return self.raw_evaluation(execution_context,check=check).select(node,cells,coordinate_limit=coordinate_limit)

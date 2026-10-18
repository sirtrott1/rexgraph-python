# distutils: language=c++
"""Admitted native operators with full independent input and choice ancestry."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport NativeInputUseAsset, _bounds, _maximum, _image
from ._field_admission cimport NativeAvailabilityContext, _keys
from ._finite_field cimport _declaration_info
from ._finite_field_admission cimport (PreparedAvailableFiniteSource, PreparedAvailableFiniteFieldMap, _status, _free)
from ._finite_operator cimport (NativeFiniteInputSchema, _transport, _wrap_schema, _wrap_operator,
    _wrap_instance, _wrap_result, _wrap_branch, _wrap_evaluation)
from ._field_program import _closed
from ._stalk_fields import _natural

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FINITE_OPERATOR_ADMISSION_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)
_STATUSES=('observed','prospective')
_INSTANCE_KEYS=('transport_maps','transport_relationship','maps','relationship','instance')


def finite_operator_admission_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_FINITE_OPERATOR_ADMISSION_BINDING_DIGEST.decode(),
            'profile':'owning-available-relational-finite-operator-v1','scalar_policy':'gmp-rational-exact'}


cdef FiniteOperatorDeclaration _declaration(value, vector[size_t] bounds):
    _closed(value,('profile','key','revision','ports'))
    if value['profile']!='rexgraph.relational-finite-operator.v1':raise ValueError('unknown finite operator profile')
    cdef FiniteOperatorDeclaration d
    cdef FiniteOperatorPort port
    d.key=_natural(value['key']);d.revision=_natural(value['revision'])
    for row in value['ports']:
        _closed(row,('key','rule','input'));port.key=_natural(row['key']);port.rule=_natural(row['rule']);port.input=_natural(row['input'])
        d.ports.push_back(port)
        if d.ports.size()>bounds[1]:raise ValueError('finite operator ports exceed admission')
    return d


cdef vector[AvailableFiniteOperatorInput] _rows(value,vector[size_t] bounds,size_t maximum):
    cdef vector[AvailableFiniteOperatorInput] rows
    cdef AvailableFiniteOperatorInput row
    for item in value:
        _closed(item,('port','schema','use','field','origins'))
        if not isinstance(item['schema'],NativeFiniteInputSchema):raise TypeError('actual passive native finite input schema required')
        row.port=_natural(item['port']);row.use=_natural(item['use']);row.schema=(<NativeFiniteInputSchema>item['schema'])._pin()
        row.field=_image(item['field'],maximum);row.origins=_keys(item['origins'],('asset','value','accession','reconstruction'));rows.push_back(row)
        if rows.size()>bounds[1]:raise ValueError('complete operator inputs exceed admission')
    return rows


def finite_operator_origin_payload(PreparedAvailableFiniteFieldMap reference not None,declaration,status,
        NativeAvailabilityContext execution_context not None,*,limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef FiniteOperatorDeclaration d=_declaration(declaration,bounds)
    cdef unsigned kind=_status(status)
    cdef shared_ptr[AvailableFiniteMapBridge] parent=reference._pin()
    cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
    cdef string image
    with nogil:image=AvailableFiniteOperatorBridge.payload(parent,d,kind,authority,maximum)
    return <bytes>image


def finite_operator_input_origin_payloads(NativeFiniteInputSchema schema not None,use,field,*,limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef uint64_t key=_natural(use)
    cdef string image=_image(field,maximum)
    cdef shared_ptr[FiniteInputSchemaBridge] actual=schema._pin()
    cdef vector[string] rows
    with nogil:rows=AvailableFiniteOperatorBridge.input_payloads(actual,key,image,bounds,maximum)
    return dict(zip(('asset','value','accession','reconstruction'),tuple(<bytes>r for r in rows)))


cdef PreparedAvailableFiniteFieldOperator _adopt_operator(shared_ptr[AvailableFiniteOperatorBridge] handle,vector[size_t] bounds,size_t maximum):
    cdef PreparedAvailableFiniteFieldOperator out=PreparedAvailableFiniteFieldOperator.__new__(PreparedAvailableFiniteFieldOperator)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeAvailableFiniteOperatorInstance _adopt_instance(shared_ptr[AvailableFiniteOperatorInstanceBridge] handle,vector[size_t] bounds,size_t maximum):
    cdef NativeAvailableFiniteOperatorInstance out=NativeAvailableFiniteOperatorInstance.__new__(NativeAvailableFiniteOperatorInstance)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeAvailableFiniteOperatorResult _adopt_result(shared_ptr[AvailableFiniteOperatorResultBridge] handle,vector[size_t] bounds,size_t maximum):
    cdef NativeAvailableFiniteOperatorResult out=NativeAvailableFiniteOperatorResult.__new__(NativeAvailableFiniteOperatorResult)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeAvailableFiniteOperatorBranch _adopt_branch(shared_ptr[AvailableFiniteOperatorBranchBridge] handle,vector[size_t] bounds,size_t maximum):
    cdef NativeAvailableFiniteOperatorBranch out=NativeAvailableFiniteOperatorBranch.__new__(NativeAvailableFiniteOperatorBranch)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeAvailableFiniteOperatorEvaluation _adopt_evaluation(shared_ptr[AvailableFiniteOperatorEvaluationBridge] handle,vector[size_t] bounds,size_t maximum):
    cdef NativeAvailableFiniteOperatorEvaluation out=NativeAvailableFiniteOperatorEvaluation.__new__(NativeAvailableFiniteOperatorEvaluation)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef PreparedAvailableFiniteSource _adopt_source(shared_ptr[AvailableFiniteSourceBridge] handle,vector[size_t] bounds,size_t maximum):
    cdef PreparedAvailableFiniteSource out=PreparedAvailableFiniteSource.__new__(PreparedAvailableFiniteSource)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef PreparedAvailableFiniteFieldMap _adopt_map(shared_ptr[AvailableFiniteMapBridge] handle,vector[size_t] bounds,size_t maximum):
    cdef PreparedAvailableFiniteFieldMap out=PreparedAvailableFiniteFieldMap.__new__(PreparedAvailableFiniteFieldMap)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef class PreparedAvailableFiniteFieldOperator:
    cdef shared_ptr[AvailableFiniteOperatorBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite operator owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority)

    def __init__(self,PreparedAvailableFiniteFieldMap reference not None,declaration,status,origins,
            NativeAvailabilityContext execution_context not None,*,limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('available finite operator is immutable')
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef FiniteOperatorDeclaration d
        cdef unsigned kind
        cdef vector[uint64_t] keys
        cdef shared_ptr[AvailableFiniteMapBridge] parent
        cdef shared_ptr[AvailabilityContextBridge] authority
        self._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);d=_declaration(declaration,bounds);kind=_status(status)
            keys=_keys(origins,('maps','relationship'));parent=reference._pin();authority=execution_context._pin()
            with nogil:self._handle=shared_ptr[AvailableFiniteOperatorBridge](new AvailableFiniteOperatorBridge(parent,d,kind,keys,authority,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls,archive,NativeAvailabilityContext execution_context not None,*,limits=_LIMITS,max_bytes=256*1024*1024):
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableFiniteOperatorBridge] handle
        with nogil:handle=AvailableFiniteOperatorBridge.restore(image,authority,bounds,maximum)
        return _adopt_operator(handle,bounds,maximum)

    def archive(self):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def raw(self):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorBridge] handle
        with nogil:handle=owner.get().raw()
        return _wrap_operator(handle,self._limits,self._maximum)

    def reference(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteMapBridge] handle
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority);handle=owner.get().reference()
        return _adopt_map(handle,self._limits,self._maximum)

    def receipts(self,NativeAvailabilityContext execution_context not None,*,max_records=4096):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef vector[string] rows
        with nogil:owner.get().admit(authority);rows=owner.get().receipts(count)
        return tuple(<bytes>r for r in rows)

    def dependency_archives(self,NativeAvailabilityContext execution_context not None,*,max_records=4096):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef vector[string] rows
        with nogil:owner.get().admit(authority);rows=owner.get().dependency_archives(count)
        return tuple(<bytes>r for r in rows)

    def same_owner(self,PreparedAvailableFiniteFieldOperator other not None):
        cdef shared_ptr[AvailableFiniteOperatorBridge] left=self._pin(),right=other._pin()
        return bool(left.get().same_owner(right.get()[0]))

    def inspect(self,NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        return dict(self.raw().inspect(),status=_STATUSES[owner.get().status()],origins=dict(zip(('maps','relationship'),owner.get().origins())),authority='current-native-context-admitted')

    def select_ports(self,begin,count,NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        return self.raw().select_ports(begin,count)

    def source_declaration(self,NativeInputUseAsset asset not None,transport,*,max_records=4096):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[InputUseAssetBridge] actual=asset._pin()
        cdef FiniteTransportDeclaration d=_transport(transport,self._limits)
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef FiniteDeclaration value
        with nogil:value=owner.get().source_declaration(actual,d,count)
        return _declaration_info(value)

    def input_schema(self,NativeInputUseAsset asset not None,transport,port):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[InputUseAssetBridge] actual=asset._pin()
        cdef FiniteTransportDeclaration d=_transport(transport,self._limits)
        cdef uint64_t key=_natural(port)
        cdef shared_ptr[FiniteInputSchemaBridge] value
        with nogil:value=owner.get().input_schema(actual,d,key)
        return _wrap_schema(value,self._limits,self._maximum)

    def transport_payload(self,PreparedAvailableFiniteSource source not None,transport,status,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=source._pin()
        cdef FiniteTransportDeclaration d=_transport(transport,self._limits)
        cdef unsigned kind=_status(status)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef string image
        with nogil:image=owner.get().transport_payload(actual,d,kind,authority)
        return <bytes>image

    def child_map_declaration(self,declaration,inputs,*,max_records=4096):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        _closed(declaration,('profile','key','revision','transport'))
        if declaration['profile']!='rexgraph.relational-finite-operator-instance.v1':raise ValueError('unknown finite instance profile')
        cdef uint64_t key=_natural(declaration['key']),revision=_natural(declaration['revision'])
        cdef FiniteTransportDeclaration d=_transport(declaration['transport'],self._limits)
        cdef vector[AvailableFiniteOperatorInput] rows=_rows(inputs,self._limits,self._maximum)
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef AvailableFiniteOperatorChildInfo value
        with nogil:value=owner.get().child_declaration(key,revision,d,rows,count)
        return {'profile':'rexgraph.available-finite-operator-child-declaration-view.v1','key':value.key,'revision':value.revision,
                'rules':tuple({'key':r.key,'bindings':tuple(r.bindings),'symbols':tuple(r.symbols),'target':r.target if r.has_target else None,
                               'input_count':r.input_count,'unsupported':(<bytes>r.unsupported).decode()} for r in value.rules),'authority':'raw-none'}

    def child_map_payload(self,PreparedAvailableFiniteSource source not None,declaration,inputs,status,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=source._pin()
        _closed(declaration,('profile','key','revision','transport'))
        if declaration['profile']!='rexgraph.relational-finite-operator-instance.v1':raise ValueError('unknown finite instance profile')
        cdef uint64_t key=_natural(declaration['key']),revision=_natural(declaration['revision'])
        cdef FiniteTransportDeclaration d=_transport(declaration['transport'],self._limits)
        cdef vector[AvailableFiniteOperatorInput] rows=_rows(inputs,self._limits,self._maximum)
        cdef unsigned kind=_status(status)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef string value
        with nogil:value=owner.get().child_payload(actual,key,revision,d,rows,kind,authority)
        return <bytes>value

    def instance_payload(self,PreparedAvailableFiniteSource source not None,declaration,inputs,status,origins,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=source._pin()
        _closed(declaration,('profile','key','revision','transport'))
        if declaration['profile']!='rexgraph.relational-finite-operator-instance.v1':raise ValueError('unknown finite instance profile')
        cdef uint64_t key=_natural(declaration['key']),revision=_natural(declaration['revision'])
        cdef FiniteTransportDeclaration d=_transport(declaration['transport'],self._limits)
        cdef vector[AvailableFiniteOperatorInput] rows=_rows(inputs,self._limits,self._maximum)
        cdef unsigned kind=_status(status)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef vector[uint64_t] keys=_keys(origins,_INSTANCE_KEYS)
        cdef string value
        with nogil:value=owner.get().instance_payload(actual,key,revision,d,rows,kind,keys,authority)
        return <bytes>value

    def instantiate(self,PreparedAvailableFiniteSource source not None,declaration,inputs,status,origins,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteSourceBridge] actual=source._pin()
        _closed(declaration,('profile','key','revision','transport'))
        if declaration['profile']!='rexgraph.relational-finite-operator-instance.v1':raise ValueError('unknown finite instance profile')
        cdef uint64_t key=_natural(declaration['key']),revision=_natural(declaration['revision'])
        cdef FiniteTransportDeclaration d=_transport(declaration['transport'],self._limits)
        cdef vector[AvailableFiniteOperatorInput] rows=_rows(inputs,self._limits,self._maximum)
        cdef unsigned kind=_status(status)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef vector[uint64_t] keys=_keys(origins,_INSTANCE_KEYS)
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] value
        with nogil:value=owner.get().instantiate(actual,key,revision,d,rows,kind,keys,authority)
        return _adopt_instance(value,self._limits,self._maximum)


cdef class NativeAvailableFiniteOperatorInstance:
    cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite operator owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority)

    @classmethod
    def from_archive(cls,archive,NativeAvailabilityContext execution_context not None,*,limits=_LIMITS,max_bytes=256*1024*1024):
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] handle
        with nogil:handle=AvailableFiniteOperatorInstanceBridge.restore(image,authority,bounds,maximum)
        return _adopt_instance(handle,bounds,maximum)

    def archive(self):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def raw(self):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorInstanceBridge] handle
        with nogil:handle=owner.get().raw()
        return _wrap_instance(handle,self._limits,self._maximum)

    def operator_program(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteOperatorBridge] handle
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority);handle=owner.get().operator_program()
        return _adopt_operator(handle,self._limits,self._maximum)

    def source(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteSourceBridge] handle
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority);handle=owner.get().source()
        return _adopt_source(handle,self._limits,self._maximum)

    def field_map(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteMapBridge] handle
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority);handle=owner.get().field_map()
        return _adopt_map(handle,self._limits,self._maximum)

    def receipts(self,NativeAvailabilityContext execution_context not None,*,max_records=4096):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef vector[string] rows
        with nogil:owner.get().admit(authority);rows=owner.get().receipts(count)
        return tuple(<bytes>r for r in rows)

    def dependency_archives(self,NativeAvailabilityContext execution_context not None,*,max_records=4096):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef vector[string] rows
        with nogil:owner.get().admit(authority);rows=owner.get().dependency_archives(count)
        return tuple(<bytes>r for r in rows)

    def same_owner(self,NativeAvailableFiniteOperatorInstance other not None):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] left=self._pin(),right=other._pin()
        return bool(left.get().same_owner(right.get()[0]))

    def inspect(self,NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        return dict(self.raw().inspect(),status=_STATUSES[owner.get().status()],origins=dict(zip(_INSTANCE_KEYS,owner.get().origins())),authority='current-native-context-admitted')

    def execute(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] value
        with nogil:value=owner.get().execute(authority)
        return _adopt_result(value,self._limits,self._maximum)

    def variable_positions(self,begin,count,NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        return self.raw().source_transport().variable_positions(begin,count)

    def use_correspondences(self,begin,count,NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        return self.raw().source_transport().use_correspondences(begin,count)

    def select_inputs(self,begin,count,NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        return self.raw().select_inputs(begin,count)


cdef class NativeAvailableFiniteOperatorResult:
    cdef shared_ptr[AvailableFiniteOperatorResultBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite operator owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority)

    @classmethod
    def from_archive(cls,archive,NativeAvailabilityContext execution_context not None,*,limits=_LIMITS,max_bytes=256*1024*1024):
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] handle
        with nogil:handle=AvailableFiniteOperatorResultBridge.restore(image,authority,bounds,maximum)
        return _adopt_result(handle,bounds,maximum)

    def archive(self):
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def raw(self):
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorResultBridge] handle
        with nogil:handle=owner.get().raw()
        return _wrap_result(handle,self._limits,self._maximum)

    def instance(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteOperatorInstanceBridge] handle
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority);handle=owner.get().instance()
        return _adopt_instance(handle,self._limits,self._maximum)

    def inspect(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] owner=self._pin()
        self.admit(execution_context)
        return {'branch_count':owner.get().size(),'authority':'current-native-context-admitted'}

    def select(self,begin,count,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t first=_natural(begin),size=_natural(count)
        cdef vector[shared_ptr[AvailableFiniteOperatorBranchBridge]] rows
        with nogil:rows=owner.get().select(first,size,authority)
        return tuple(_adopt_branch(r,self._limits,self._maximum) for r in rows)

    def completion_payload(self,ordinal,values,parameter_setting,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t index=_natural(ordinal)
        cdef uint64_t setting=_natural(parameter_setting)
        cdef vector[string] free=_free(values,self._limits)
        cdef string image
        with nogil:image=owner.get().completion_payload(index,free,setting,authority)
        return <bytes>image

    def evaluate(self,ordinal,values,parameter_setting,choice_origin,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t index=_natural(ordinal)
        cdef uint64_t setting=_natural(parameter_setting),choice=_natural(choice_origin)
        cdef vector[string] free=_free(values,self._limits)
        cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] value
        with nogil:value=owner.get().evaluate(index,free,setting,choice,authority)
        return _adopt_evaluation(value,self._limits,self._maximum)


cdef class NativeAvailableFiniteOperatorBranch:
    cdef shared_ptr[AvailableFiniteOperatorBranchBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite operator owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorBranchBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority)

    def raw(self):
        cdef shared_ptr[AvailableFiniteOperatorBranchBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorBranchBridge] handle
        with nogil:handle=owner.get().raw()
        return _wrap_branch(handle,self._limits,self._maximum)

    def family(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorBranchBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] handle
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority);handle=owner.get().family()
        return _adopt_result(handle,self._limits,self._maximum)

    def inspect(self,NativeAvailabilityContext execution_context not None):
        self.admit(execution_context)
        return dict(self.raw().raw().inspect(),authority='current-native-context-admitted')


cdef class NativeAvailableFiniteOperatorEvaluation:
    cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('available finite operator owner is not initialized')
        return self._handle

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def admit(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority)

    @classmethod
    def from_archive(cls,archive,NativeAvailabilityContext execution_context not None,*,limits=_LIMITS,max_bytes=256*1024*1024):
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] handle
        with nogil:handle=AvailableFiniteOperatorEvaluationBridge.restore(image,authority,bounds,maximum)
        return _adopt_evaluation(handle,bounds,maximum)

    def archive(self):
        cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def raw(self):
        cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorEvaluationBridge] handle
        with nogil:handle=owner.get().raw()
        return _wrap_evaluation(handle,self._limits,self._maximum)

    def family(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] owner=self._pin()
        cdef shared_ptr[AvailableFiniteOperatorResultBridge] handle
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:owner.get().admit(authority);handle=owner.get().family()
        return _adopt_result(handle,self._limits,self._maximum)

    def inspect(self,NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef string setting,choice
        with nogil:owner.get().admit(authority);setting=owner.get().setting_receipt();choice=owner.get().choice_receipt()
        return {'ordinal':owner.get().ordinal(),'parameter_setting_receipt':<bytes>setting if setting.size() else None,
                'choice_receipt':<bytes>choice,'authority':'current-native-context-admitted'}

    def dependency_archives(self,NativeAvailabilityContext execution_context not None,*,max_records=4096):
        cdef shared_ptr[AvailableFiniteOperatorEvaluationBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef vector[string] rows
        with nogil:owner.get().admit(authority);rows=owner.get().dependency_archives(count)
        return tuple(<bytes>r for r in rows)

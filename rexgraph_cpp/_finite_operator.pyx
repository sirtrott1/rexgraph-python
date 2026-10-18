# distutils: language=c++
"""Raw native finite operators with explicit independent input ancestry."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._finite_field cimport (NativeFiniteSource, PreparedFiniteFieldMap, NativeFiniteFieldResult,
    NativeFiniteFieldBranch, NativeFiniteFieldEvaluation, _declaration_info)
from ._field_interface cimport NativeInputUseAsset, _bounds, _maximum, _image
from ._field_program import _closed
from ._stalk_fields import _natural, _scalar

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FINITE_OPERATOR_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)


def finite_operator_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_FINITE_OPERATOR_BINDING_DIGEST.decode(),
            'profile':'raw-owning-relational-finite-operator-v1','scalar_policy':'gmp-rational-exact','authority':'raw-none'}


cdef vector[vector[size_t]] _routing(value, size_t maximum):
    cdef vector[vector[size_t]] out
    cdef vector[size_t] row
    cdef size_t count=0
    for items in value:
        row.clear();count+=1
        if count>maximum:raise ValueError('finite transport routing exceeds admission')
        for item in items:
            count+=1
            if count>maximum:raise ValueError('finite transport routing exceeds admission')
            row.push_back(_natural(item))
        out.push_back(row)
    return out


cdef FiniteTransportDeclaration _transport(value, vector[size_t] bounds):
    _closed(value,('profile','key','revision','source_map','uses'))
    if value['profile']!='rexgraph.relational-finite-source-transport.v1':raise ValueError('unknown finite transport profile')
    _closed(value['source_map'],('cells','occurrences'))
    cdef FiniteTransportDeclaration out
    cdef FiniteUseCorrespondence use
    out.key=_natural(value['key']);out.revision=_natural(value['revision'])
    out.cells=_routing(value['source_map']['cells'],bounds[1]);out.occurrences=_routing(value['source_map']['occurrences'],bounds[1])
    for row in value['uses']:
        _closed(row,('reference','input'));use.reference=_natural(row['reference']);use.input=_natural(row['input'])
        out.uses.push_back(use)
        if out.uses.size()>bounds[1]:raise ValueError('finite use correspondences exceed admission')
    return out


cdef object _transport_info(FiniteTransportDeclaration value):
    return {'profile':'rexgraph.relational-finite-source-transport.v1','key':value.key,'revision':value.revision,
            'source_map':{'cells':tuple(tuple(r) for r in value.cells),'occurrences':tuple(tuple(r) for r in value.occurrences)},
            'uses':tuple({'reference':u.reference,'input':u.input} for u in value.uses)}


cdef PreparedFiniteFieldOperator _wrap_operator(shared_ptr[FiniteOperatorBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef PreparedFiniteFieldOperator out=PreparedFiniteFieldOperator.__new__(PreparedFiniteFieldOperator)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteSourceTransport _wrap_transport(shared_ptr[FiniteTransportBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteSourceTransport out=NativeFiniteSourceTransport.__new__(NativeFiniteSourceTransport)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteInputSchema _wrap_schema(shared_ptr[FiniteInputSchemaBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteInputSchema out=NativeFiniteInputSchema.__new__(NativeFiniteInputSchema)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteOperatorInstance _wrap_instance(shared_ptr[FiniteOperatorInstanceBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteOperatorInstance out=NativeFiniteOperatorInstance.__new__(NativeFiniteOperatorInstance)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteOperatorResult _wrap_result(shared_ptr[FiniteOperatorResultBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteOperatorResult out=NativeFiniteOperatorResult.__new__(NativeFiniteOperatorResult)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteOperatorBranch _wrap_branch(shared_ptr[FiniteOperatorBranchBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteOperatorBranch out=NativeFiniteOperatorBranch.__new__(NativeFiniteOperatorBranch)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteOperatorEvaluation _wrap_evaluation(shared_ptr[FiniteOperatorEvaluationBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteOperatorEvaluation out=NativeFiniteOperatorEvaluation.__new__(NativeFiniteOperatorEvaluation)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteSource _wrap_source(shared_ptr[FiniteSourceBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteSource out=NativeFiniteSource.__new__(NativeFiniteSource)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef PreparedFiniteFieldMap _wrap_map(shared_ptr[FiniteMapBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef PreparedFiniteFieldMap out=PreparedFiniteFieldMap.__new__(PreparedFiniteFieldMap)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteFieldResult _wrap_raw_result(shared_ptr[FiniteResultBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteFieldResult out=NativeFiniteFieldResult.__new__(NativeFiniteFieldResult)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteFieldBranch _wrap_raw_branch(shared_ptr[FiniteBranchBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteFieldBranch out=NativeFiniteFieldBranch.__new__(NativeFiniteFieldBranch)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeFiniteFieldEvaluation _wrap_raw_evaluation(shared_ptr[FiniteEvaluationBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeFiniteFieldEvaluation out=NativeFiniteFieldEvaluation.__new__(NativeFiniteFieldEvaluation)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef NativeInputUseAsset _wrap_asset(shared_ptr[InputUseAssetBridge] handle, vector[size_t] bounds, size_t maximum):
    cdef NativeInputUseAsset out=NativeInputUseAsset.__new__(NativeInputUseAsset)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef class PreparedFiniteFieldOperator:
    cdef shared_ptr[FiniteOperatorBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite operator owner is not initialized')
        return self._handle

    def __init__(self, PreparedFiniteFieldMap reference not None, declaration, *, limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('finite operator owner is immutable')
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef FiniteOperatorDeclaration d
        cdef FiniteOperatorPort port
        cdef shared_ptr[FiniteMapBridge] parent
        self._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes)
            _closed(declaration,('profile','key','revision','ports'))
            if declaration['profile']!='rexgraph.relational-finite-operator.v1':raise ValueError('unknown finite operator profile')
            d.key=_natural(declaration['key']);d.revision=_natural(declaration['revision'])
            for row in declaration['ports']:
                _closed(row,('key','rule','input'));port.key=_natural(row['key']);port.rule=_natural(row['rule']);port.input=_natural(row['input'])
                d.ports.push_back(port)
                if d.ports.size()>bounds[1]:raise ValueError('finite operator ports exceed admission')
            parent=reference._pin()
            with nogil:self._handle=shared_ptr[FiniteOperatorBridge](new FiniteOperatorBridge(parent,d,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[FiniteOperatorBridge] handle
        with nogil:handle=FiniteOperatorBridge.restore(image,bounds,maximum)
        return _wrap_operator(handle,bounds,maximum)

    def archive(self):
        cdef shared_ptr[FiniteOperatorBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def inspect(self):
        cdef shared_ptr[FiniteOperatorBridge] owner=self._pin()
        return {'key':owner.get().key(),'revision':owner.get().revision(),'port_count':owner.get().size(),'authority':'raw-none'}

    def same_owner(self, PreparedFiniteFieldOperator other not None):
        cdef shared_ptr[FiniteOperatorBridge] left=self._pin(), right=other._pin()
        return bool(left.get().same_owner(right.get()[0]))

    def reference(self):
        cdef shared_ptr[FiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[FiniteMapBridge] handle
        with nogil:handle=owner.get().reference()
        return _wrap_map(handle,self._limits,self._maximum)

    def select_ports(self, begin, count):
        cdef shared_ptr[FiniteOperatorBridge] owner=self._pin()
        cdef size_t first=_natural(begin), size=_natural(count)
        cdef vector[FiniteOperatorPort] rows
        with nogil:rows=owner.get().ports(first,size)
        return tuple({'key':p.key,'rule':p.rule,'input':p.input} for p in rows)

    def source_declaration(self, NativeInputUseAsset asset not None, transport, *, max_records=4096):
        cdef shared_ptr[FiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[InputUseAssetBridge] actual=asset._pin()
        cdef FiniteTransportDeclaration d=_transport(transport,self._limits)
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef FiniteDeclaration value
        with nogil:value=owner.get().source_declaration(actual,d,count)
        return _declaration_info(value)

    def input_schema(self, NativeInputUseAsset asset not None, transport, port):
        cdef shared_ptr[FiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[InputUseAssetBridge] actual=asset._pin()
        cdef FiniteTransportDeclaration d=_transport(transport,self._limits)
        cdef uint64_t key=_natural(port)
        cdef shared_ptr[FiniteInputSchemaBridge] handle
        with nogil:handle=owner.get().input_schema(actual,d,key)
        return _wrap_schema(handle,self._limits,self._maximum)

    def instantiate(self, NativeFiniteSource source not None, declaration, inputs):
        cdef shared_ptr[FiniteOperatorBridge] owner=self._pin()
        cdef shared_ptr[FiniteSourceBridge] actual=source._pin()
        _closed(declaration,('profile','key','revision','transport'))
        if declaration['profile']!='rexgraph.relational-finite-operator-instance.v1':raise ValueError('unknown finite instance profile')
        cdef uint64_t key=_natural(declaration['key']), revision=_natural(declaration['revision'])
        cdef FiniteTransportDeclaration d=_transport(declaration['transport'],self._limits)
        cdef vector[FiniteOperatorInput] rows
        cdef FiniteOperatorInput row
        cdef shared_ptr[FiniteOperatorInstanceBridge] handle
        for item in inputs:
            _closed(item,('port','schema','use','field'))
            if not isinstance(item['schema'],NativeFiniteInputSchema):raise TypeError('native passive finite port schema required')
            row.port=_natural(item['port']);row.schema=(<NativeFiniteInputSchema>item['schema'])._pin()
            row.use=_natural(item['use']);row.field=_image(item['field'],self._maximum);rows.push_back(row)
            if rows.size()>self._limits[1]:raise ValueError('finite instance inputs exceed admission')
        with nogil:handle=owner.get().instantiate(actual,key,revision,d,rows)
        return _wrap_instance(handle,self._limits,self._maximum)


cdef class NativeFiniteInputSchema:
    cdef shared_ptr[FiniteInputSchemaBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite operator owner is not initialized')
        return self._handle

    def asset(self):
        cdef shared_ptr[FiniteInputSchemaBridge] owner=self._pin()
        cdef shared_ptr[InputUseAssetBridge] handle
        with nogil:handle=owner.get().asset()
        return _wrap_asset(handle,self._limits,self._maximum)

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[FiniteInputSchemaBridge] owner=self._pin()
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef FiniteSchemaInfo value
        with nogil:value=owner.get().info(count)
        return {'profile':'rexgraph.relational-finite-input-schema-view.v1','port':owner.get().port(),
            'key':value.key,'node':value.node,'grade':value.grade,'isometric':bool(value.isometric),
            'stalks':tuple({'cell':(s.grade,s.cell),'basis':s.basis,'directions':tuple(s.directions),'metric_archive':<bytes>s.metric} for s in value.stalks),
            'instances':{'cells':tuple(tuple(r) for r in value.cells),'occurrences':tuple(tuple(r) for r in value.occurrences)},
            'accession_archive':<bytes>value.accession,'reconstruction_archive':<bytes>value.reconstruction,'authority':'raw-none'}


cdef class NativeFiniteSourceTransport:
    cdef shared_ptr[FiniteTransportBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite operator owner is not initialized')
        return self._handle

    def __init__(self, NativeFiniteSource reference not None, NativeFiniteSource source not None, declaration, *, limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('finite transport owner is immutable')
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef FiniteTransportDeclaration d
        cdef shared_ptr[FiniteSourceBridge] old, actual
        self._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);d=_transport(declaration,bounds)
            old=reference._pin();actual=source._pin()
            with nogil:self._handle=shared_ptr[FiniteTransportBridge](new FiniteTransportBridge(old,actual,d,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[FiniteTransportBridge] owner=self._pin()
        cdef size_t count=min(_natural(max_records),self._limits[1])
        cdef FiniteTransportDeclaration value
        with nogil:value=owner.get().info(count)
        return _transport_info(value)

    def same_owner(self, NativeFiniteSourceTransport other not None):
        cdef shared_ptr[FiniteTransportBridge] left=self._pin(), right=other._pin()
        return bool(left.get().same_owner(right.get()[0]))

    def variable_positions(self, begin, count):
        cdef shared_ptr[FiniteTransportBridge] owner=self._pin()
        cdef size_t first=_natural(begin), size=_natural(count)
        cdef vector[FiniteVariableCorrespondence] rows
        with nogil:rows=owner.get().variables(first,size)
        return tuple({'reference':v.reference,'input':v.input,'reference_cell':(v.grade,v.reference_cell),'input_cell':(v.grade,v.input_cell)} for v in rows)

    def use_correspondences(self, begin, count):
        cdef shared_ptr[FiniteTransportBridge] owner=self._pin()
        cdef size_t first=_natural(begin), size=_natural(count)
        cdef vector[FiniteUseCorrespondence] rows
        with nogil:rows=owner.get().uses(first,size)
        return tuple({'reference':u.reference,'input':u.input} for u in rows)

    def reference(self):
        cdef shared_ptr[FiniteTransportBridge] owner=self._pin()
        cdef shared_ptr[FiniteSourceBridge] handle
        with nogil:handle=owner.get().reference()
        return _wrap_source(handle,self._limits,self._maximum)

    def source(self):
        cdef shared_ptr[FiniteTransportBridge] owner=self._pin()
        cdef shared_ptr[FiniteSourceBridge] handle
        with nogil:handle=owner.get().source()
        return _wrap_source(handle,self._limits,self._maximum)


cdef class NativeFiniteOperatorInstance:
    cdef shared_ptr[FiniteOperatorInstanceBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite operator owner is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[FiniteOperatorInstanceBridge] handle
        with nogil:handle=FiniteOperatorInstanceBridge.restore(image,bounds,maximum)
        return _wrap_instance(handle,bounds,maximum)

    def archive(self):
        cdef shared_ptr[FiniteOperatorInstanceBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def inspect(self):
        cdef shared_ptr[FiniteOperatorInstanceBridge] owner=self._pin()
        return {'key':owner.get().key(),'revision':owner.get().revision(),'input_count':owner.get().size(),'authority':'raw-none'}

    def same_owner(self, NativeFiniteOperatorInstance other not None):
        cdef shared_ptr[FiniteOperatorInstanceBridge] left=self._pin(), right=other._pin()
        return bool(left.get().same_owner(right.get()[0]))

    def select_inputs(self, begin, count):
        cdef shared_ptr[FiniteOperatorInstanceBridge] owner=self._pin()
        cdef size_t first=_natural(begin), size=_natural(count)
        cdef vector[FiniteOperatorInput] rows
        with nogil:rows=owner.get().inputs(first,size)
        return tuple({'port':r.port,'use':r.use,'schema':_wrap_schema(r.schema,self._limits,self._maximum),'field':<bytes>r.field} for r in rows)

    def operator_program(self):
        cdef shared_ptr[FiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorBridge] handle
        with nogil:handle=owner.get().operator_program()
        return _wrap_operator(handle,self._limits,self._maximum)

    def source_transport(self):
        cdef shared_ptr[FiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[FiniteTransportBridge] handle
        with nogil:handle=owner.get().transport()
        return _wrap_transport(handle,self._limits,self._maximum)

    def source(self):
        cdef shared_ptr[FiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[FiniteSourceBridge] handle
        with nogil:handle=owner.get().source()
        return _wrap_source(handle,self._limits,self._maximum)

    def map(self):
        cdef shared_ptr[FiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[FiniteMapBridge] handle
        with nogil:handle=owner.get().model()
        return _wrap_map(handle,self._limits,self._maximum)

    def execute(self):
        cdef shared_ptr[FiniteOperatorInstanceBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorResultBridge] handle
        with nogil:handle=owner.get().execute()
        return _wrap_result(handle,self._limits,self._maximum)


cdef class NativeFiniteOperatorResult:
    cdef shared_ptr[FiniteOperatorResultBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite operator owner is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[FiniteOperatorResultBridge] handle
        with nogil:handle=FiniteOperatorResultBridge.restore(image,bounds,maximum)
        return _wrap_result(handle,bounds,maximum)

    def archive(self):
        cdef shared_ptr[FiniteOperatorResultBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def inspect(self):
        cdef shared_ptr[FiniteOperatorResultBridge] owner=self._pin()
        return {'branch_count':owner.get().size(),'authority':'raw-none'}

    def select(self, begin, count):
        cdef shared_ptr[FiniteOperatorResultBridge] owner=self._pin()
        cdef size_t first=_natural(begin), size=_natural(count)
        cdef vector[shared_ptr[FiniteOperatorBranchBridge]] rows
        with nogil:rows=owner.get().select(first,size)
        return tuple(_wrap_branch(r,self._limits,self._maximum) for r in rows)

    def evaluate(self, ordinal, values):
        cdef shared_ptr[FiniteOperatorResultBridge] owner=self._pin()
        cdef size_t index=_natural(ordinal)
        cdef vector[string] free
        cdef shared_ptr[FiniteOperatorEvaluationBridge] handle
        for value in values:
            free.push_back(_scalar(value,self._limits[3]))
            if free.size()>self._limits[0]:raise ValueError('complete finite operator choice exceeds admission')
        with nogil:handle=owner.get().evaluate(index,free)
        return _wrap_evaluation(handle,self._limits,self._maximum)

    def instance(self):
        cdef shared_ptr[FiniteOperatorResultBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorInstanceBridge] handle
        with nogil:handle=owner.get().instance()
        return _wrap_instance(handle,self._limits,self._maximum)

    def raw(self):
        cdef shared_ptr[FiniteOperatorResultBridge] owner=self._pin()
        cdef shared_ptr[FiniteResultBridge] handle
        with nogil:handle=owner.get().raw()
        return _wrap_raw_result(handle,self._limits,self._maximum)


cdef class NativeFiniteOperatorBranch:
    cdef shared_ptr[FiniteOperatorBranchBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite operator owner is not initialized')
        return self._handle

    def ordinal(self):
        cdef shared_ptr[FiniteOperatorBranchBridge] owner=self._pin()
        return owner.get().ordinal()

    def family(self):
        cdef shared_ptr[FiniteOperatorBranchBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorResultBridge] handle
        with nogil:handle=owner.get().parent()
        return _wrap_result(handle,self._limits,self._maximum)

    def raw(self):
        cdef shared_ptr[FiniteOperatorBranchBridge] owner=self._pin()
        cdef shared_ptr[FiniteBranchBridge] handle
        with nogil:handle=owner.get().raw()
        return _wrap_raw_branch(handle,self._limits,self._maximum)


cdef class NativeFiniteOperatorEvaluation:
    cdef shared_ptr[FiniteOperatorEvaluationBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite operator owner is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[FiniteOperatorEvaluationBridge] handle
        with nogil:handle=FiniteOperatorEvaluationBridge.restore(image,bounds,maximum)
        return _wrap_evaluation(handle,bounds,maximum)

    def archive(self):
        cdef shared_ptr[FiniteOperatorEvaluationBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def ordinal(self):
        cdef shared_ptr[FiniteOperatorEvaluationBridge] owner=self._pin()
        return owner.get().ordinal()

    def family(self):
        cdef shared_ptr[FiniteOperatorEvaluationBridge] owner=self._pin()
        cdef shared_ptr[FiniteOperatorResultBridge] handle
        with nogil:handle=owner.get().family()
        return _wrap_result(handle,self._limits,self._maximum)

    def raw(self):
        cdef shared_ptr[FiniteOperatorEvaluationBridge] owner=self._pin()
        cdef shared_ptr[FiniteEvaluationBridge] handle
        with nogil:handle=owner.get().raw()
        return _wrap_raw_evaluation(handle,self._limits,self._maximum)

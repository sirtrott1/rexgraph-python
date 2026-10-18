# distutils: language=c++
"""Owning native byte derived finite readings and complete field results."""
from ._field_interface cimport NativeInputUseAsset, NativeLiteralInput, PreparedAccessionFieldProgram, _bounds, _maximum, _image
from ._field_evidence cimport PreparedFieldEvidence
from ._finite_field cimport NativeFiniteSource, PreparedFiniteFieldMap, NativeFiniteFieldResult
from ._field_program import _closed
from ._stalk_fields import _natural

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_LITERAL_FINITE_FIELD_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)


def literal_finite_field_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_LITERAL_FINITE_FIELD_BINDING_DIGEST.decode(),
            'profile':'raw-owning-literal-finite-field-v1','scalar_policy':'gmp-rational-exact','authority':'raw-none'}


def encode_literal_use_value(NativeInputUseAsset asset not None, use, *, limits=_LIMITS, max_bytes=256*1024*1024):
    """Encode the native canonical ordered span bytes with their actual owner."""
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef uint64_t key=_natural(use)
    cdef shared_ptr[InputUseAssetBridge] held=asset._pin()
    cdef string result
    with nogil:result=literal_use_value(held,key,bounds,maximum)
    return <bytes>result


cdef vector[uint64_t] _keys(values, size_t maximum):
    cdef vector[uint64_t] out
    for value in values:
        out.push_back(_natural(value))
        if out.size()>maximum:raise ValueError('literal finite dependency keys exceed admission')
    return out


cdef LiteralFiniteDeclaration _declaration(value, vector[size_t] bounds):
    _closed(value,('profile','key','revision','rules'))
    if value['profile']!='rexgraph.relational-literal-finite-field-map.v1':raise ValueError('unknown literal finite map profile')
    cdef LiteralFiniteDeclaration out
    cdef LiteralFiniteRule rule
    out.key=_natural(value['key']);out.revision=_natural(value['revision'])
    for row in value['rules']:
        _closed(row,('key','bindings','symbols','target','inputs','unsupported'))
        if not isinstance(row['unsupported'],str):raise TypeError('unsupported reason must be text')
        rule.key=_natural(row['key']);rule.bindings=_keys(row['bindings'],bounds[1]);rule.symbols=_keys(row['symbols'],bounds[1])
        rule.has_target=row['target'] is not None;rule.target=0 if row['target'] is None else _natural(row['target'])
        rule.unsupported=row['unsupported'].encode();rule.inputs.clear()
        for literal in row['inputs']:
            if not isinstance(literal,NativeLiteralInput):raise TypeError('actual native literal derivation required')
            rule.inputs.push_back((<NativeLiteralInput>literal)._pin())
            if rule.inputs.size()>bounds[0]:raise ValueError('literal input count exceeds admission')
        out.rules.push_back(rule)
        if out.rules.size()>bounds[0]:raise ValueError('literal rule count exceeds admission')
    return out


cdef class PreparedLiteralFiniteFieldMap:
    cdef shared_ptr[LiteralFiniteMapBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('native literal finite map is not initialized')
        return self._handle

    def __init__(self, NativeFiniteSource source not None, targets, declaration, *, limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('native literal finite map owner is immutable')
        self._loading=True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef vector[FiniteTarget] rows
        cdef FiniteTarget target
        cdef LiteralFiniteDeclaration d
        cdef shared_ptr[FiniteSourceBridge] actual
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);actual=source._pin()
            for value in targets:
                target.fixed.reset();target.affine.reset()
                if isinstance(value,PreparedAccessionFieldProgram):target.fixed=(<PreparedAccessionFieldProgram>value)._pin()
                elif isinstance(value,PreparedFieldEvidence):target.affine=(<PreparedFieldEvidence>value)._pin()
                else:raise TypeError('actual raw program or complete field evidence target required')
                rows.push_back(target)
                if rows.size()>bounds[0]:raise ValueError('literal numerical targets exceed admission')
            d=_declaration(declaration,bounds)
            with nogil:self._handle=shared_ptr[LiteralFiniteMapBridge](new LiteralFiniteMapBridge(actual,rows,d,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef PreparedLiteralFiniteFieldMap out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        with nogil:out._handle=LiteralFiniteMapBridge.restore(image,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[LiteralFiniteMapBridge] held=self._pin()
        cdef string result
        with nogil:result=held.get().archive()
        return <bytes>result

    def source(self):
        cdef shared_ptr[LiteralFiniteMapBridge] held=self._pin()
        cdef NativeFiniteSource out=NativeFiniteSource.__new__(NativeFiniteSource)
        with nogil:out._handle=held.get().source()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def raw_map(self):
        """Return the generic map; retain this composite for reader provenance."""
        cdef shared_ptr[LiteralFiniteMapBridge] held=self._pin()
        cdef PreparedFiniteFieldMap out=PreparedFiniteFieldMap.__new__(PreparedFiniteFieldMap)
        with nogil:out._handle=held.get().raw_map()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def same_owner(self, PreparedLiteralFiniteFieldMap other not None):
        cdef shared_ptr[LiteralFiniteMapBridge] left=self._pin(),right=other._pin()
        return bool(left.get().same_owner(right.get()[0]))

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[LiteralFiniteMapBridge] held=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef LiteralFiniteDeclaration d
        cdef NativeLiteralInput literal
        with nogil:d=held.get().info(maximum)
        rules=[]
        for r in d.rules:
            inputs=[]
            for value in r.inputs:
                literal=NativeLiteralInput.__new__(NativeLiteralInput);literal._handle=value
                literal._limits=self._limits;literal._maximum=self._maximum;inputs.append(literal)
            rules.append({'key':r.key,'bindings':tuple(r.bindings),'symbols':tuple(r.symbols),
                          'target':r.target if r.has_target else None,'inputs':tuple(inputs),'unsupported':(<bytes>r.unsupported).decode()})
        return {'profile':'rexgraph.relational-literal-finite-field-map.v1','key':d.key,'revision':d.revision,'rules':tuple(rules)}

    def execute(self):
        cdef shared_ptr[LiteralFiniteMapBridge] held=self._pin()
        cdef NativeLiteralFiniteFieldResult out=NativeLiteralFiniteFieldResult.__new__(NativeLiteralFiniteFieldResult)
        with nogil:out._handle=held.get().execute()
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeLiteralFiniteFieldResult:
    cdef shared_ptr[LiteralFiniteResultBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('native literal finite result is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeLiteralFiniteFieldResult out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        with nogil:out._handle=LiteralFiniteResultBridge.restore(image,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[LiteralFiniteResultBridge] held=self._pin()
        cdef string result
        with nogil:result=held.get().archive()
        return <bytes>result

    def map(self):
        cdef shared_ptr[LiteralFiniteResultBridge] held=self._pin()
        cdef PreparedLiteralFiniteFieldMap out=PreparedLiteralFiniteFieldMap.__new__(PreparedLiteralFiniteFieldMap)
        with nogil:out._handle=held.get().map()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def family(self):
        cdef shared_ptr[LiteralFiniteResultBridge] held=self._pin()
        cdef NativeFiniteFieldResult out=NativeFiniteFieldResult.__new__(NativeFiniteFieldResult)
        with nogil:out._handle=held.get().family()
        out._limits=self._limits;out._maximum=self._maximum;return out

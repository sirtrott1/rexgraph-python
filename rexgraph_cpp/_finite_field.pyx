# distutils: language=c++
"""Raw owning finite symbols, correlated sections and whole field choices."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport (NativeInputUseAsset, NativeAccessionInput, PreparedAccessionFieldProgram,
    NativeAccessionFieldResult, _bounds, _maximum, _image, _indices)
from ._field_evidence cimport PreparedFieldEvidence, NativeAffineFieldResult, NativeAffineEvaluation
from ._field_program import _closed
from ._stalk_fields import _natural, _scalar

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FINITE_FIELD_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)
_STATUSES=('unmapped','unsupported','interpreted')


def finite_field_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_FINITE_FIELD_BINDING_DIGEST.decode(),
            'profile':'raw-owning-relational-finite-field-v1','scalar_policy':'gmp-rational-exact','authority':'raw-none'}


cdef vector[uint64_t] _keys(value, size_t maximum):
    cdef vector[uint64_t] out
    for v in value:
        out.push_back(_natural(v))
        if out.size()>maximum:raise ValueError('finite keys exceed admission')
    return out


cdef FiniteDeclaration _declaration(value, vector[size_t] bounds, size_t maximum):
    _closed(value,('profile','key','revision','symbols','variables','charts','bindings'))
    if value['profile']!='rexgraph.relational-finite-source.v1':raise ValueError('unknown finite source profile')
    cdef FiniteDeclaration out
    cdef FiniteSymbol symbol
    cdef FiniteVariable variable
    cdef FiniteChart chart
    cdef FiniteBinding binding
    out.key=_natural(value['key']);out.revision=_natural(value['revision'])
    for row in value['symbols']:
        _closed(row,('key','definition'));symbol.key=_natural(row['key']);symbol.definition=_image(row['definition'],maximum)
        out.symbols.push_back(symbol)
        if out.symbols.size()>bounds[0]:raise ValueError('finite symbol count exceeds admission')
    for row in value['variables']:
        _closed(row,('cell','domain'))
        if len(row['cell'])!=2:raise ValueError('finite variable requires original grade/index')
        variable.grade=_natural(row['cell'][0]);variable.cell=_natural(row['cell'][1]);variable.domain=_keys(row['domain'],bounds[1]);out.variables.push_back(variable)
        if out.variables.size()>bounds[0]:raise ValueError('finite variable count exceeds admission')
    for row in value['charts']:
        _closed(row,('key','name','patch','support','allowed'))
        if not isinstance(row['name'],str):raise TypeError('finite chart name must be text')
        chart.key=_natural(row['key']);chart.name=row['name'].encode();chart.patch=_indices(row['patch']);chart.support=_indices(row['support']);chart.allowed.clear()
        for item in row['allowed']:
            chart.allowed.push_back(_keys(item,bounds[1]))
            if chart.allowed.size()>bounds[1]:raise ValueError('finite chart table exceeds admission')
        out.charts.push_back(chart)
        if out.charts.size()>bounds[1]:raise ValueError('finite chart count exceeds admission')
    for row in value['bindings']:
        _closed(row,('key','use','cell'))
        if len(row['cell'])!=2:raise ValueError('finite binding requires original grade/index')
        binding.key=_natural(row['key']);binding.use=_natural(row['use']);binding.grade=_natural(row['cell'][0]);binding.cell=_natural(row['cell'][1]);out.bindings.push_back(binding)
        if out.bindings.size()>bounds[1]:raise ValueError('finite binding count exceeds admission')
    return out


cdef object _declaration_info(FiniteDeclaration value):
    return {'profile':'rexgraph.relational-finite-source.v1','key':value.key,'revision':value.revision,
        'symbols':tuple({'key':v.key,'definition':<bytes>v.definition} for v in value.symbols),
        'variables':tuple({'cell':(v.grade,v.cell),'domain':tuple(v.domain)} for v in value.variables),
        'charts':tuple({'key':v.key,'name':(<bytes>v.name).decode(),'patch':tuple(tuple(r) for r in v.patch),
                        'support':tuple(tuple(r) for r in v.support),'allowed':tuple(tuple(r) for r in v.allowed)} for v in value.charts),
        'bindings':tuple({'key':v.key,'use':v.use,'cell':(v.grade,v.cell)} for v in value.bindings)}


cdef FiniteTarget _target(value):
    cdef FiniteTarget out
    if isinstance(value,PreparedAccessionFieldProgram):out.fixed=(<PreparedAccessionFieldProgram>value)._pin()
    elif isinstance(value,PreparedFieldEvidence):out.affine=(<PreparedFieldEvidence>value)._pin()
    else:raise TypeError('actual raw accession program or complete field evidence target required')
    return out


cdef object _target_info(FiniteTarget value, vector[size_t] bounds, size_t maximum):
    cdef PreparedAccessionFieldProgram fixed
    cdef PreparedFieldEvidence affine
    if value.fixed.get()!=NULL:
        fixed=PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram);fixed._handle=value.fixed;fixed._limits=bounds;fixed._maximum=maximum;return fixed
    affine=PreparedFieldEvidence.__new__(PreparedFieldEvidence);affine._handle=value.affine;affine._limits=bounds;affine._maximum=maximum;return affine


cdef FiniteMapDeclaration _map_declaration(value, vector[size_t] bounds):
    _closed(value,('profile','key','revision','rules'))
    if value['profile']!='rexgraph.relational-finite-field-map.v1':raise ValueError('unknown finite field map profile')
    cdef FiniteMapDeclaration out
    cdef FiniteRule rule
    out.key=_natural(value['key']);out.revision=_natural(value['revision'])
    for row in value['rules']:
        _closed(row,('key','bindings','symbols','target','inputs','unsupported'))
        if not isinstance(row['unsupported'],str):raise TypeError('finite unsupported reason must be text')
        rule.key=_natural(row['key']);rule.bindings=_keys(row['bindings'],bounds[1]);rule.symbols=_keys(row['symbols'],bounds[1]);rule.has_target=row['target'] is not None
        rule.target=0 if row['target'] is None else _natural(row['target']);rule.unsupported=row['unsupported'].encode();rule.inputs.clear()
        for item in row['inputs']:
            if not isinstance(item,NativeAccessionInput):raise TypeError('actual admitted raw source input required')
            rule.inputs.push_back((<NativeAccessionInput>item)._pin())
            if rule.inputs.size()>bounds[0]:raise ValueError('finite rule inputs exceed admission')
        out.rules.push_back(rule)
        if out.rules.size()>bounds[1]:raise ValueError('finite rule count exceeds admission')
    return out


cdef object _branch_info(FiniteBranchInfo value):
    return {'ordinal':value.ordinal,'section':value.section,'rule':value.rule if value.has_rule else None,
            'status':_STATUSES[value.status],'target':value.target if value.has_target else None,
            'affine':bool(value.affine),'assignment':tuple(value.assignment)}


cdef class NativeFiniteSource:
    cdef shared_ptr[FiniteSourceBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite source is not initialized')
        return self._handle

    def __init__(self, NativeInputUseAsset asset not None, declaration, *, limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('raw finite source owner is immutable')
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef FiniteDeclaration d=_declaration(declaration,bounds,maximum)
        cdef shared_ptr[InputUseAssetBridge] owner=asset._pin()
        self._loading=True
        try:
            with nogil:self._handle=shared_ptr[FiniteSourceBridge](new FiniteSourceBridge(owner,d,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeFiniteSource out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        with nogil:out._handle=FiniteSourceBridge.restore(image,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[FiniteSourceBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[FiniteSourceBridge] owner=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef FiniteDeclaration value
        with nogil:value=owner.get().info(maximum)
        return _declaration_info(value)

    def asset(self):
        cdef shared_ptr[FiniteSourceBridge] owner=self._pin()
        cdef NativeInputUseAsset out=NativeInputUseAsset.__new__(NativeInputUseAsset)
        with nogil:out._handle=owner.get().asset()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def same_owner(self, NativeFiniteSource other not None):
        cdef shared_ptr[FiniteSourceBridge] left=self._pin(),right=other._pin()
        return bool(left.get().same_owner(right.get()[0]))

    def sections(self):
        cdef shared_ptr[FiniteSourceBridge] owner=self._pin()
        cdef NativeFiniteSections out=NativeFiniteSections.__new__(NativeFiniteSections)
        with nogil:out._handle=owner.get().sections()
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeFiniteSections:
    cdef shared_ptr[FiniteSectionsBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite sections are not initialized')
        return self._handle

    def source(self):
        cdef shared_ptr[FiniteSectionsBridge] owner=self._pin()
        cdef NativeFiniteSource out=NativeFiniteSource.__new__(NativeFiniteSource)
        with nogil:out._handle=owner.get().source()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def inspect(self):
        cdef shared_ptr[FiniteSectionsBridge] owner=self._pin()
        return {'candidate_count':owner.get().candidates(),'section_count':owner.get().size()}

    def select(self, begin, count):
        cdef shared_ptr[FiniteSectionsBridge] owner=self._pin()
        cdef size_t start=_natural(begin),length=_natural(count)
        cdef vector[vector[uint64_t]] rows
        with nogil:rows=owner.get().select(start,length)
        return tuple(tuple(row) for row in rows)


cdef class PreparedFiniteFieldMap:
    cdef shared_ptr[FiniteMapBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite field map is not initialized')
        return self._handle

    def __init__(self, NativeFiniteSource source not None, targets, declaration, *, limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('raw finite map owner is immutable')
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef vector[FiniteTarget] rows
        for value in targets:
            rows.push_back(_target(value))
            if rows.size()>bounds[0]:raise ValueError('finite target count exceeds admission')
        cdef FiniteMapDeclaration d=_map_declaration(declaration,bounds)
        cdef shared_ptr[FiniteSourceBridge] owner=source._pin()
        self._loading=True
        try:
            with nogil:self._handle=shared_ptr[FiniteMapBridge](new FiniteMapBridge(owner,rows,d,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef PreparedFiniteFieldMap out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        with nogil:out._handle=FiniteMapBridge.restore(image,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[FiniteMapBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def source(self):
        cdef shared_ptr[FiniteMapBridge] owner=self._pin()
        cdef NativeFiniteSource out=NativeFiniteSource.__new__(NativeFiniteSource)
        with nogil:out._handle=owner.get().source()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def targets(self, *, max_records=256):
        cdef shared_ptr[FiniteMapBridge] owner=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[FiniteTarget] rows
        with nogil:rows=owner.get().targets(maximum)
        return tuple(_target_info(row,self._limits,self._maximum) for row in rows)

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[FiniteMapBridge] owner=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef FiniteMapDeclaration value
        cdef NativeAccessionInput inp
        cdef shared_ptr[AccessionInputBridge] handle
        with nogil:value=owner.get().info(maximum)
        rows=[]
        for row in value.rules:
            inputs=[]
            for handle in row.inputs:
                inp=NativeAccessionInput.__new__(NativeAccessionInput);inp._handle=handle;inputs.append(inp)
            rows.append({'key':row.key,'bindings':tuple(row.bindings),'symbols':tuple(row.symbols),
                         'target':row.target if row.has_target else None,'inputs':tuple(inputs),'unsupported':(<bytes>row.unsupported).decode()})
        return {'profile':'rexgraph.relational-finite-field-map.v1','key':value.key,'revision':value.revision,'rules':tuple(rows)}

    def execute(self):
        cdef shared_ptr[FiniteMapBridge] owner=self._pin()
        cdef NativeFiniteFieldResult out=NativeFiniteFieldResult.__new__(NativeFiniteFieldResult)
        with nogil:out._handle=owner.get().execute()
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeFiniteFieldResult:
    cdef shared_ptr[FiniteResultBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite field result is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeFiniteFieldResult out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        with nogil:out._handle=FiniteResultBridge.restore(image,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[FiniteResultBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def model(self):
        cdef shared_ptr[FiniteResultBridge] owner=self._pin()
        cdef PreparedFiniteFieldMap out=PreparedFiniteFieldMap.__new__(PreparedFiniteFieldMap)
        with nogil:out._handle=owner.get().model()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def sections(self):
        cdef shared_ptr[FiniteResultBridge] owner=self._pin()
        cdef NativeFiniteSections out=NativeFiniteSections.__new__(NativeFiniteSections)
        with nogil:out._handle=owner.get().sections()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def inspect(self):
        cdef shared_ptr[FiniteResultBridge] owner=self._pin()
        return {'branch_count':owner.get().size()}

    def select(self, begin, count):
        cdef shared_ptr[FiniteResultBridge] owner=self._pin()
        cdef size_t start=_natural(begin),length=_natural(count)
        cdef vector[shared_ptr[FiniteBranchBridge]] rows
        cdef shared_ptr[FiniteBranchBridge] handle
        cdef NativeFiniteFieldBranch branch
        with nogil:rows=owner.get().select(start,length)
        out=[]
        for handle in rows:
            branch=NativeFiniteFieldBranch.__new__(NativeFiniteFieldBranch);branch._handle=handle;branch._limits=self._limits;branch._maximum=self._maximum;out.append(branch)
        return tuple(out)

    def evaluate(self, ordinal, values):
        cdef shared_ptr[FiniteResultBridge] owner=self._pin()
        cdef size_t position=_natural(ordinal)
        cdef vector[string] free
        for value in values:
            free.push_back(_scalar(value,self._limits[3]))
            if free.size()>self._limits[0]:raise ValueError('finite free values exceed admission')
        cdef NativeFiniteFieldEvaluation out=NativeFiniteFieldEvaluation.__new__(NativeFiniteFieldEvaluation)
        with nogil:out._handle=owner.get().evaluate(position,free)
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeFiniteFieldBranch:
    cdef shared_ptr[FiniteBranchBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite field branch is not initialized')
        return self._handle

    def family(self):
        cdef shared_ptr[FiniteBranchBridge] owner=self._pin()
        cdef NativeFiniteFieldResult out=NativeFiniteFieldResult.__new__(NativeFiniteFieldResult)
        with nogil:out._handle=owner.get().family()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def inspect(self):
        cdef shared_ptr[FiniteBranchBridge] owner=self._pin()
        cdef FiniteBranchInfo value
        with nogil:value=owner.get().info()
        return _branch_info(value)

    def result(self):
        cdef shared_ptr[FiniteBranchBridge] owner=self._pin()
        cdef shared_ptr[AccessionResultBridge] fixed
        cdef shared_ptr[AffineResultBridge] affine
        cdef NativeAccessionFieldResult f
        cdef NativeAffineFieldResult a
        with nogil:fixed=owner.get().fixed();affine=owner.get().affine()
        if fixed.get()!=NULL:
            f=NativeAccessionFieldResult.__new__(NativeAccessionFieldResult);f._handle=fixed;f._limits=self._limits;f._maximum=self._maximum;return f
        if affine.get()!=NULL:
            a=NativeAffineFieldResult.__new__(NativeAffineFieldResult);a._handle=affine;a._limits=self._limits;a._maximum=self._maximum;return a
        return None


cdef class NativeFiniteFieldEvaluation:
    cdef shared_ptr[FiniteEvaluationBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('raw finite field evaluation is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeFiniteFieldEvaluation out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        with nogil:out._handle=FiniteEvaluationBridge.restore(image,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[FiniteEvaluationBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().archive()
        return <bytes>image

    def family(self):
        cdef shared_ptr[FiniteEvaluationBridge] owner=self._pin()
        cdef NativeFiniteFieldResult out=NativeFiniteFieldResult.__new__(NativeFiniteFieldResult)
        with nogil:out._handle=owner.get().family()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def ordinal(self):
        cdef shared_ptr[FiniteEvaluationBridge] owner=self._pin()
        return owner.get().ordinal()

    def result(self):
        cdef shared_ptr[FiniteEvaluationBridge] owner=self._pin()
        cdef NativeAccessionFieldResult out=NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil:out._handle=owner.get().result()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def affine_evaluation(self):
        cdef shared_ptr[FiniteEvaluationBridge] owner=self._pin()
        cdef shared_ptr[AffineEvaluationBridge] actual
        cdef NativeAffineEvaluation out
        with nogil:actual=owner.get().affine()
        if actual.get()==NULL:return None
        out=NativeAffineEvaluation.__new__(NativeAffineEvaluation);out._handle=actual;out._limits=self._limits;out._maximum=self._maximum;return out

    def free_values_archive(self):
        cdef shared_ptr[FiniteEvaluationBridge] owner=self._pin()
        cdef string image
        with nogil:image=owner.get().free_values()
        return <bytes>image

# distutils: language=c++
"""Owning native product evidence, source lineage and complete field families."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_admission cimport (PreparedAvailableFieldProgram, NativeAvailabilityContext, AvailableProgramBridge,
    AvailabilityContextBridge, AdmissionInput, _input, _keys)
from ._field_interface cimport (PreparedAccessionFieldProgram, NativeInputUseAsset, FieldProgramBridge, InputUseAssetBridge,
    _bounds, _maximum, _image, _indices)
from ._known_tensor_square cimport NativeAvailableKnownTensorSquareResult, AvailableKnownSquareResultBridge
from ._field_evidence cimport PreparedFieldEvidence, NativeAffineFieldResult, EvidenceDeclaration, _evidence_declaration
from ._field_evidence_admission cimport AvailableEvidenceBridge
from ._stalk_fields import _natural
from ._field_program import _closed

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_KNOWN_PRODUCT_LEARNING_BINDING_DIGEST

_LIMITS=(70000,16000000,64000000,16384)
_SHARED=(256*1024*1024,256*1024*1024,1000000,2*1024*1024*1024)


def known_product_learning_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),'engine_headers':RFC_ENGINE_DIGEST.decode(),
            'binding':RFC_KNOWN_PRODUCT_LEARNING_BINDING_DIGEST.decode(),'profile':'owning-available-known-product-learning-v1','scalar_policy':'gmp-rational-exact'}


cdef vector[size_t] _sharing(value):
    value=tuple(value)
    if len(value)!=4 or any(type(v) is not int or v<=0 for v in value):raise ValueError('four positive native shared storage bounds required')
    cdef vector[size_t] out
    for v in value:out.push_back(_natural(v))
    return out


cdef vector[vector[uint64_t]] _uses(value):
    cdef vector[vector[uint64_t]] out
    cdef vector[uint64_t] pair
    for row in value:
        if len(row)!=2:raise ValueError('complete original/initial use correspondence required')
        pair.clear();pair.push_back(_natural(row[0]));pair.push_back(_natural(row[1]));out.push_back(pair)
    return out


cdef KnownProductObservation _observation(value,vector[size_t] bounds,size_t maximum):
    _closed(value,('key','input','lineage','target_asset','target_use','target','observed','origins','interface_origin','evidence_origin'))
    cdef NativeAvailableKnownTensorSquareResult input_owner=value['input']
    cdef NativeInputUseAsset target=value['target_asset']
    cdef NativeAvailableInitialInputLineage original
    cdef KnownProductObservation out
    out.input=input_owner._pin();out.target_asset=target._pin()
    if value['lineage'] is not None:
        original=value['lineage'];out.lineage=original._pin()
    out.key=_natural(value['key']);out.use=_natural(value['target_use'])
    out.target=_image(value['target'],maximum);out.observed=_indices(value['observed'])
    keys=_keys(value['origins'],('asset','value'));out.asset_origin=keys[0];out.value_origin=keys[1]
    out.interface_origin=_natural(value['interface_origin']);out.evidence_origin=_natural(value['evidence_origin'])
    if len(value['observed'])>bounds[1]:raise ValueError('whole observation mask exceeds admission')
    return out


cdef PreparedAvailableKnownProductEvidence _adopt_evidence(shared_ptr[KnownProductEvidenceBridge] handle,
    vector[size_t] bounds,size_t maximum,vector[size_t] sharing,bint teacher,string expected):
    cdef PreparedAvailableKnownProductEvidence out=PreparedAvailableKnownProductEvidence.__new__(PreparedAvailableKnownProductEvidence)
    out._handle=handle;out._limits=bounds;out._maximum=maximum;out._shared=sharing;out._teacher=teacher;out._expected=expected
    return out


cdef string _expected_payload(value,teacher,size_t maximum):
    if type(teacher) is not bool:raise TypeError('explicit declared grammar profile required')
    if teacher:return _image(value,maximum)
    if type(value) is not bytes or value:raise ValueError('generic rational model has no teacher moment root payload')
    return string()


cdef NativeAvailableKnownTensorSquareResult _adopt_input(shared_ptr[AvailableKnownSquareResultBridge] handle,
    vector[size_t] bounds,size_t maximum):
    cdef NativeAvailableKnownTensorSquareResult out=NativeAvailableKnownTensorSquareResult.__new__(NativeAvailableKnownTensorSquareResult)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


def require_four_use_grammar_image(PreparedAvailableFieldProgram program not None,submission,NativeAvailabilityContext context not None,
                                  *,limits=_LIMITS,max_bytes=256*1024*1024):
    cdef shared_ptr[AvailableProgramBridge] parent=program._pin()
    cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef AdmissionInput row=_input(submission,bounds,maximum,True)
    with nogil:require_four_use_grammar(parent,row,authority,bounds,maximum)


def shared_known_product_input(NativeAvailableKnownTensorSquareResult input not None,*,shared_limits=_SHARED):
    cdef shared_ptr[AvailableKnownSquareResultBridge] owner=input._pin()
    cdef vector[size_t] sharing=_sharing(shared_limits)
    cdef string image
    with nogil:image=shared_product_input(owner,sharing)
    return <bytes>image


def initial_input_lineage_payload(NativeInputUseAsset original not None,NativeAvailableKnownTensorSquareResult input not None,declaration,
                                  *,limits=_LIMITS,max_bytes=256*1024*1024):
    _closed(declaration,('profile','key','revision','uses'))
    if declaration['profile']!='rexgraph.initial-input-lineage.v1':raise ValueError('native initial lineage declaration required')
    cdef shared_ptr[InputUseAssetBridge] asset=original._pin()
    cdef shared_ptr[AvailableKnownSquareResultBridge] result=input._pin()
    cdef vector[vector[uint64_t]] uses=_uses(declaration['uses'])
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef uint64_t key=_natural(declaration['key']),revision=_natural(declaration['revision'])
    cdef string image
    with nogil:image=InitialInputLineageBridge.payload(asset,result,key,revision,uses,bounds,maximum)
    return <bytes>image


cdef class NativeAvailableInitialInputLineage:
    cdef shared_ptr[InitialInputLineageBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('native initial lineage is not initialized')
        return self._handle

    def __init__(self,NativeInputUseAsset original not None,NativeAvailableKnownTensorSquareResult input not None,declaration,origins,
                 NativeAvailabilityContext context not None,*,limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('native initial lineage owners are immutable')
        _closed(declaration,('profile','key','revision','uses'))
        if declaration['profile']!='rexgraph.initial-input-lineage.v1':raise ValueError('native initial lineage declaration required')
        self._loading=True
        cdef shared_ptr[InputUseAssetBridge] asset=original._pin()
        cdef shared_ptr[AvailableKnownSquareResultBridge] result=input._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef vector[vector[uint64_t]] uses=_uses(declaration['uses'])
        cdef vector[uint64_t] keys=_keys(origins,('original','initial','interface'))
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef uint64_t key=_natural(declaration['key']),revision=_natural(declaration['revision'])
        try:
            with nogil:self._handle=shared_ptr[InitialInputLineageBridge](new InitialInputLineageBridge(asset,result,key,revision,uses,keys,authority,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls,image,NativeAvailableKnownTensorSquareResult input not None,NativeAvailabilityContext context not None,
                     *,limits=_LIMITS,max_bytes=256*1024*1024):
        cdef shared_ptr[AvailableKnownSquareResultBridge] result=input._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string packet=_image(image,maximum)
        cdef NativeAvailableInitialInputLineage out=cls.__new__(cls)
        with nogil:out._handle=InitialInputLineageBridge.restore(packet,result,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum
        return out

    def admit(self,NativeAvailabilityContext context not None):
        cdef shared_ptr[InitialInputLineageBridge] handle=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        with nogil:handle.get().admit(authority)

    def archive(self):
        cdef shared_ptr[InitialInputLineageBridge] handle=self._pin()
        cdef string image
        with nogil:image=handle.get().archive()
        return <bytes>image

    def asset_archive(self,kind='original'):
        if kind not in ('original','initial'):raise ValueError('original or initial source branch required')
        cdef shared_ptr[InitialInputLineageBridge] handle=self._pin()
        cdef bint original=kind=='original'
        cdef string image
        with nogil:image=handle.get().asset(original)
        return <bytes>image

    def dependency_archives(self,*,max_records=16000000):
        cdef shared_ptr[InitialInputLineageBridge] handle=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] images
        with nogil:images=handle.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)


cdef class NativeFourUseMomentBasis:
    cdef shared_ptr[FourUseMomentBasisBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('native moment basis is not initialized')
        return self._handle

    def __init__(self,PreparedAccessionFieldProgram program not None,NativeAvailableKnownTensorSquareResult reference not None,declaration,
                 *,limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('native moment bases are immutable')
        _closed(declaration,('profile','key','revision','node','parameters','pairs'))
        if declaration['profile']!='rexgraph.four-use-moment-basis.v1':raise ValueError('actual native moment declaration required')
        self._loading=True
        cdef shared_ptr[FieldProgramBridge] parent=program._pin()
        cdef shared_ptr[AvailableKnownSquareResultBridge] source=reference._pin()
        cdef uint64_t key=_natural(declaration['key']),revision=_natural(declaration['revision']),node=_natural(declaration['node'])
        cdef vector[uint64_t] parameters
        cdef vector[MomentPair] pairs
        cdef MomentPair pair
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        try:
            if len(declaration['parameters'])>bounds[1] or len(declaration['pairs'])>bounds[1]:raise ValueError('native basis declaration exceeds metadata admission')
            for name in declaration['parameters']:parameters.push_back(_natural(name))
            for row in declaration['pairs']:
                _closed(row,('relation','head','dependent','head_flag','dependent_flag','rows'))
                if len(row['rows'])!=2:raise ValueError('two actual incoming flag role rows required')
                pair.relation=_natural(row['relation']);pair.head=_natural(row['head']);pair.dependent=_natural(row['dependent'])
                pair.head_flag=_natural(row['head_flag']);pair.dependent_flag=_natural(row['dependent_flag'])
                pair.row0=_natural(row['rows'][0]);pair.row1=_natural(row['rows'][1]);pairs.push_back(pair)
            with nogil:self._handle=shared_ptr[FourUseMomentBasisBridge](new FourUseMomentBasisBridge(parent,source,key,revision,node,parameters,pairs,bounds,maximum))
        finally:self._loading=False

    def inspect(self):
        cdef shared_ptr[FourUseMomentBasisBridge] handle=self._pin()
        cdef vector[size_t] counts
        with nogil:counts=handle.get().counts()
        return dict(zip(('variable_count','parameter_count','basis_nonzeros','pairs'),tuple(int(x) for x in counts),strict=True))

    def matrix_archive(self,pair,kind='reading'):
        if kind not in ('reading','base_reconstruction','base_metric'):raise ValueError('declared native comparison matrix required')
        cdef shared_ptr[FourUseMomentBasisBridge] handle=self._pin()
        cdef size_t selected=_natural(pair)
        cdef unsigned which=('reading','base_reconstruction','base_metric').index(kind)
        cdef string image
        with nogil:image=handle.get().matrix(selected,which)
        return <bytes>image


def known_product_root_payload(PreparedAccessionFieldProgram program not None,parameters,*,limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef string image
    cdef shared_ptr[FourUseMomentBasisBridge] basis
    cdef shared_ptr[FieldProgramBridge] parent=program._pin()
    cdef EvidenceDeclaration declaration
    if isinstance(parameters,NativeFourUseMomentBasis):
        basis=(<NativeFourUseMomentBasis>parameters)._pin()
        with nogil:image=KnownProductEvidenceBridge.basis_root_payload(basis,bounds,maximum)
    else:
        declaration=_evidence_declaration(parameters,bounds)
        with nogil:image=AvailableEvidenceBridge.root_payload(parent,declaration,bounds,maximum)
    return <bytes>image


cdef class PreparedAvailableKnownProductEvidence:
    cdef shared_ptr[KnownProductEvidenceBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('native complete product evidence is not initialized')
        return self._handle

    def __init__(self,PreparedAvailableFieldProgram program not None,parameters,origin,NativeAvailabilityContext context not None,
                 *,limits=_LIMITS,max_bytes=256*1024*1024,shared_limits=_SHARED,teacher_grammar=False):
        if self._loading or self._handle.get()!=NULL:raise TypeError('native complete product evidence is immutable')
        if type(teacher_grammar) is not bool:raise TypeError('explicit declared grammar profile required')
        if teacher_grammar and not isinstance(parameters,NativeFourUseMomentBasis):raise TypeError('teacher role profile requires actual native moment basis')
        self._loading=True
        cdef shared_ptr[AvailableProgramBridge] parent=program._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef shared_ptr[FourUseMomentBasisBridge] basis
        cdef EvidenceDeclaration declaration
        cdef vector[size_t] bounds=_bounds(limits),sharing=_sharing(shared_limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef uint64_t key=_natural(origin)
        try:
            if isinstance(parameters,NativeFourUseMomentBasis):
                basis=(<NativeFourUseMomentBasis>parameters)._pin()
                with nogil:self._handle=KnownProductEvidenceBridge.from_basis(parent,basis,key,authority,bounds,maximum,sharing)
                if teacher_grammar:
                    with nogil:self._expected=KnownProductEvidenceBridge.basis_root_payload(basis,bounds,maximum)
            else:
                declaration=_evidence_declaration(parameters,bounds)
                with nogil:self._handle=shared_ptr[KnownProductEvidenceBridge](new KnownProductEvidenceBridge(parent,declaration,key,authority,bounds,maximum,sharing))
            self._limits=bounds;self._maximum=maximum;self._shared=sharing;self._teacher=teacher_grammar
        finally:self._loading=False

    @classmethod
    def from_archive(cls,image,NativeAvailabilityContext context not None,*,limits=_LIMITS,max_bytes=256*1024*1024,shared_limits=_SHARED,teacher_grammar=False,expected_root_payload=b''):
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef vector[size_t] bounds=_bounds(limits),sharing=_sharing(shared_limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string packet=_image(image,sharing[0])
        cdef string expected=_expected_payload(expected_root_payload,teacher_grammar,maximum)
        cdef bint teacher=teacher_grammar
        cdef shared_ptr[KnownProductEvidenceBridge] handle
        with nogil:handle=KnownProductEvidenceBridge.restore(packet,authority,bounds,maximum,sharing,teacher,expected)
        return _adopt_evidence(handle,bounds,maximum,sharing,teacher,expected)

    def admission(self):
        self._pin()
        return {'limits':tuple(int(x) for x in self._limits),'max_bytes':int(self._maximum),
                'shared_limits':tuple(int(x) for x in self._shared),'teacher_grammar':bool(self._teacher),
                'expected_root_payload':<bytes>self._expected}

    def archive(self):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef string image
        with nogil:image=handle.get().archive()
        return <bytes>image

    def parameter_schema_archive(self):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef string image
        with nogil:image=handle.get().parameter_schema()
        return <bytes>image

    def admit(self,NativeAvailabilityContext context not None):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        with nogil:handle.get().admit(authority)

    def evidence(self):
        """Whole parameter field inspection; retain this compound for execution."""
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef PreparedFieldEvidence out=PreparedFieldEvidence.__new__(PreparedFieldEvidence)
        with nogil:out._handle=handle.get().evidence()
        out._limits=self._limits;out._maximum=self._maximum
        return out

    def program(self):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef PreparedAvailableFieldProgram out=PreparedAvailableFieldProgram.__new__(PreparedAvailableFieldProgram)
        with nogil:out._handle=handle.get().program()
        out._limits=self._limits;out._maximum=self._maximum
        return out

    def parent(self):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin(),parent
        with nogil:parent=handle.get().parent()
        return None if parent.get()==NULL else _adopt_evidence(parent,self._limits,self._maximum,self._shared,self._teacher,self._expected)

    def origin_count(self):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef size_t count
        with nogil:count=handle.get().origin_count()
        return int(count)

    def dependency_archives(self,*,max_records=16000000):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] images
        with nogil:images=handle.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)

    def observation_payload(self,row,NativeAvailabilityContext context not None):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef KnownProductObservation observation=_observation(row,self._limits,self._maximum)
        cdef string image
        with nogil:image=handle.get().descriptor(observation,authority)
        return <bytes>image

    def observation_dependencies(self,rows,NativeAvailabilityContext context not None,*,max_records=16000000):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef vector[KnownProductObservation] observations
        cdef size_t maximum=_natural(max_records)
        cdef KnownProductBatchInfo info
        rows=tuple(rows)
        if len(rows)>self._limits[0]:raise ValueError('complete native observation suffix exceeds admission')
        for row in rows:observations.push_back(_observation(row,self._limits,self._maximum))
        with nogil:info=handle.get().batch_info(observations,authority,maximum)
        return {'origin_count':int(info.origin_count),'dependencies':tuple(<bytes>image for image in info.dependencies)}

    def append(self,rows,NativeAvailabilityContext context not None):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin(),child
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef vector[KnownProductObservation] observations
        cdef bint teacher=self._teacher
        rows=tuple(rows)
        if len(rows)>self._limits[0]:raise ValueError('native complete observation suffix exceeds admission')
        for row in rows:observations.push_back(_observation(row,self._limits,self._maximum))
        with nogil:child=handle.get().append(observations,authority,teacher)
        return _adopt_evidence(child,self._limits,self._maximum,self._shared,teacher,self._expected)

    def restore_input(self,image,NativeAvailabilityContext context not None):
        if self._teacher:raise ValueError('teacher cold input requires atomic restore_query with original lineage')
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef string packet=_image(image,self._shared[0])
        cdef shared_ptr[AvailableKnownSquareResultBridge] input
        with nogil:input=handle.get().restore_input(packet,authority)
        return _adopt_input(input,self._limits,self._maximum)

    def restore_query(self,image,lineage,NativeAvailabilityContext context not None):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef string packet=_image(image,self._shared[0])
        cdef string original=string() if lineage is None else _image(lineage,self._maximum)
        cdef bint teacher=self._teacher
        cdef KnownProductQueryOwners owners
        cdef NativeAvailableInitialInputLineage restored
        with nogil:owners=handle.get().restore_query(packet,original,authority,teacher)
        if owners.lineage.get()==NULL:return _adopt_input(owners.input,self._limits,self._maximum),None
        restored=NativeAvailableInitialInputLineage.__new__(NativeAvailableInitialInputLineage)
        restored._handle=owners.lineage;restored._limits=self._limits;restored._maximum=self._maximum
        return _adopt_input(owners.input,self._limits,self._maximum),restored

    def execute(self,NativeAvailableKnownTensorSquareResult input not None,NativeAvailabilityContext context not None,*,lineage=None):
        cdef shared_ptr[KnownProductEvidenceBridge] handle=self._pin()
        cdef shared_ptr[AvailableKnownSquareResultBridge] source=input._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef shared_ptr[InitialInputLineageBridge] original
        cdef bint teacher=self._teacher
        cdef NativeKnownProductFamily out=NativeKnownProductFamily.__new__(NativeKnownProductFamily)
        if lineage is not None:original=(<NativeAvailableInitialInputLineage?>lineage)._pin()
        with nogil:out._handle=handle.get().execute(source,original,authority,teacher)
        out._limits=self._limits;out._maximum=self._maximum;out._shared=self._shared;out._teacher=teacher;out._expected=self._expected
        return out


cdef class NativeKnownProductFamily:
    cdef shared_ptr[KnownProductFamilyBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('native complete product family is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls,image,NativeAvailabilityContext context not None,*,limits=_LIMITS,max_bytes=256*1024*1024,shared_limits=_SHARED,teacher_grammar=False,expected_root_payload=b''):
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef vector[size_t] bounds=_bounds(limits),sharing=_sharing(shared_limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string packet=_image(image,sharing[0])
        cdef string expected=_expected_payload(expected_root_payload,teacher_grammar,maximum)
        cdef bint teacher=teacher_grammar
        cdef NativeKnownProductFamily out=cls.__new__(cls)
        with nogil:out._handle=KnownProductFamilyBridge.restore(packet,authority,bounds,maximum,sharing,teacher,expected)
        out._limits=bounds;out._maximum=maximum;out._shared=sharing;out._teacher=teacher;out._expected=expected
        return out

    def archive(self):
        cdef shared_ptr[KnownProductFamilyBridge] handle=self._pin()
        cdef string image
        with nogil:image=handle.get().archive()
        return <bytes>image

    def admit(self,NativeAvailabilityContext context not None):
        cdef shared_ptr[KnownProductFamilyBridge] handle=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        with nogil:handle.get().admit(authority)

    def evidence(self):
        cdef shared_ptr[KnownProductFamilyBridge] handle=self._pin()
        cdef shared_ptr[KnownProductEvidenceBridge] evidence
        with nogil:evidence=handle.get().evidence()
        return _adopt_evidence(evidence,self._limits,self._maximum,self._shared,self._teacher,self._expected)

    def input(self):
        cdef shared_ptr[KnownProductFamilyBridge] handle=self._pin()
        cdef shared_ptr[AvailableKnownSquareResultBridge] input
        with nogil:input=handle.get().input()
        return _adopt_input(input,self._limits,self._maximum)

    def lineage(self):
        cdef shared_ptr[KnownProductFamilyBridge] handle=self._pin()
        cdef NativeAvailableInitialInputLineage out=NativeAvailableInitialInputLineage.__new__(NativeAvailableInitialInputLineage)
        with nogil:out._handle=handle.get().lineage()
        if out._handle.get()==NULL:return None
        out._limits=self._limits;out._maximum=self._maximum
        return out

    def result(self):
        """Complete field inspection; execution and replay retain this compound."""
        cdef shared_ptr[KnownProductFamilyBridge] handle=self._pin()
        cdef NativeAffineFieldResult out=NativeAffineFieldResult.__new__(NativeAffineFieldResult)
        with nogil:out._handle=handle.get().result()
        out._limits=self._limits;out._maximum=self._maximum
        return out

    def dependency_archives(self,*,max_records=16000000):
        cdef shared_ptr[KnownProductFamilyBridge] handle=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] images
        with nogil:images=handle.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)


def four_use_role_programs(NativeInputUseAsset initial not None,model_source,*,limits=_LIMITS,max_bytes=256*1024*1024):
    cdef shared_ptr[InputUseAssetBridge] asset=initial._pin()
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef string image=_image(model_source,maximum)
    cdef vector[shared_ptr[FieldProgramBridge]] owners
    cdef PreparedAccessionFieldProgram parent=PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
    cdef PreparedAccessionFieldProgram target=PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
    with nogil:owners=role_programs_native(asset,image,bounds,maximum)
    parent._handle=owners[0];parent._limits=bounds;parent._maximum=maximum
    target._handle=owners[1];target._limits=bounds;target._maximum=maximum
    return parent,target


def four_use_role_interface_payload(PreparedAccessionFieldProgram parent not None,*,max_bytes=256*1024*1024):
    cdef shared_ptr[FieldProgramBridge] owner=parent._pin()
    cdef size_t maximum=_maximum(max_bytes)
    cdef string image
    with nogil:image=FourUseRoleInputBridge.interface_payload(owner,maximum)
    return <bytes>image


def four_use_role_observed_tensor(image,*,limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef string packet=_image(image,maximum)
    cdef string result
    with nogil:result=role_observed_tensor_native(packet,bounds,maximum)
    return <bytes>result


cdef class PreparedFourUseRoleInput:
    cdef shared_ptr[FourUseRoleInputBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('four-use role input builder is not initialized')
        return self._handle

    def __init__(self,PreparedAvailableFieldProgram parent not None,PreparedAvailableFieldProgram target not None,
        interface_origin,NativeAvailabilityContext context not None,*,limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('four-use role input owners are immutable')
        self._loading=True
        cdef shared_ptr[AvailableProgramBridge] original=parent._pin(),model=target._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef uint64_t origin=_natural(interface_origin)
        try:
            with nogil:self._handle=shared_ptr[FourUseRoleInputBridge](new FourUseRoleInputBridge(original,model,origin,authority,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    def input_origin_payloads(self,NativeInputUseAsset asset not None,field,use=100):
        cdef shared_ptr[FourUseRoleInputBridge] owner=self._pin()
        cdef shared_ptr[InputUseAssetBridge] input_asset=asset._pin()
        cdef string packet=_image(field,self._maximum)
        cdef uint64_t use_key=_natural(use)
        cdef vector[string] payloads
        with nogil:payloads=owner.get().input_payloads(input_asset,packet,use_key)
        return dict(zip(('asset','value','accession','reconstruction'),tuple(<bytes>p for p in payloads),strict=True))

    def lift(self,NativeInputUseAsset asset not None,field,origins,NativeAvailabilityContext context not None,*,use=100):
        cdef shared_ptr[FourUseRoleInputBridge] owner=self._pin()
        cdef shared_ptr[InputUseAssetBridge] input_asset=asset._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef vector[uint64_t] keys=_keys(origins,('asset','value','accession','reconstruction'))
        cdef string packet=_image(field,self._maximum)
        cdef uint64_t use_key=_natural(use)
        cdef NativeFourUseRoleLift out=NativeFourUseRoleLift.__new__(NativeFourUseRoleLift)
        with nogil:out._handle=owner.get().lift(input_asset,packet,use_key,keys,authority)
        out._limits=self._limits;out._maximum=self._maximum
        return out


cdef class NativeFourUseRoleLift:
    cdef shared_ptr[FourUseRoleLiftBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('four-use role lift is not initialized')
        return self._handle

    def input_origin_payloads(self):
        cdef shared_ptr[FourUseRoleLiftBridge] owner=self._pin()
        cdef vector[string] payloads
        with nogil:payloads=owner.get().input_payloads()
        return dict(zip(('asset','value','accession','reconstruction'),tuple(<bytes>p for p in payloads),strict=True))

    def admit_input(self,origins,NativeAvailabilityContext context not None):
        cdef shared_ptr[FourUseRoleLiftBridge] owner=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=context._pin()
        cdef vector[uint64_t] keys=_keys(origins,('asset','value','accession','reconstruction'))
        cdef shared_ptr[AvailableKnownSquareResultBridge] result
        with nogil:result=owner.get().admit_input(keys,authority)
        return _adopt_input(result,self._limits,self._maximum)


def four_use_role_publisher_history(source,namespace,*,stages=5,limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef string image=_image(source,maximum)
    cdef uint64_t name_space=_natural(namespace),count=_natural(stages)
    cdef string result
    with nogil:result=role_publisher_history_native(image,name_space,count,bounds,maximum)
    return <bytes>result

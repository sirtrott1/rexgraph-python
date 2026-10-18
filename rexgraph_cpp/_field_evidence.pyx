# distutils: language=c++
"""Native whole field evidence, correlated responses and prospective choices."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport (
    ProgramMatrix, ProgramFieldInfo, FieldProgramBridge, AccessionResultBridge,
    InputUseAssetBridge, ProgramSelected, PreparedAccessionFieldProgram,
    NativeAccessionFieldResult, NativeInputUseAsset, _bounds, _maximum, _image,
    _matrix, _indices, _selected,
)
from ._stalk_fields import _natural, _scalar
from ._field_program import _closed


cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FIELD_EVIDENCE_BINDING_DIGEST

_LIMITS = (4096, 16000000, 64000000, 16384)


def field_evidence_build_info():
    return {'engine': 'rexgraph-cpp', 'engine_version': RFC_ENGINE_VERSION.decode(),
            'engine_headers': RFC_ENGINE_DIGEST.decode(), 'binding': RFC_FIELD_EVIDENCE_BINDING_DIGEST.decode(),
            'profile': 'owning-relational-field-evidence-v1', 'scalar_policy': 'gmp-rational-exact'}


cdef object _field(AffineFieldInfo value):
    return {'particular': <bytes>value.particular, 'directions': <bytes>value.directions,
            'dimension': int(value.dimension), 'supported': int(value.supported), 'nonzeros': int(value.nonzeros),
            'free_parameter_count': int(value.free_parameters), 'direction_nonzeros': int(value.direction_nonzeros)}


cdef object _rows(vector[vector[size_t]] value):
    return tuple(tuple(int(i) for i in row) for row in value)


cdef object _wrap_port(shared_ptr[OccurrenceTargetBridge] handle, vector[size_t] bounds, size_t maximum):
    if handle.get() == NULL: return None
    cdef NativeOccurrenceTargetPort out = NativeOccurrenceTargetPort.__new__(NativeOccurrenceTargetPort)
    out._handle=handle;out._limits=bounds;out._maximum=maximum
    return out


cdef class NativeOccurrenceTargetPort:
    cdef shared_ptr[OccurrenceTargetBridge] _pin(self) except *:
        if self._handle.get()==NULL: raise RuntimeError('native target port is not initialized')
        return self._handle

    def __init__(self, PreparedAccessionFieldProgram program not None, declaration, *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL: raise TypeError('native target port is immutable')
        self._loading=True
        cdef OccurrenceTargetDeclaration d
        cdef vector[uint64_t] coordinate
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef shared_ptr[FieldProgramBridge] actual
        try:
            _closed(declaration,('profile','key','revision','node','frame','coordinates','accession','reconstruction'))
            if declaration['profile']!='rexgraph.relational-occurrence-target-port.v1': raise ValueError('unknown target port profile')
            bounds=_bounds(limits);maximum=_maximum(max_bytes);actual=program._pin()
            d.key=_natural(declaration['key']);d.revision=_natural(declaration['revision']);d.node=_natural(declaration['node']);d.frame=_natural(declaration['frame'])
            for row in declaration['coordinates']:
                _closed(row,('key','occurrence','basis','direction'))
                grade,flag=row['occurrence'];coordinate.clear()
                for v in (row['key'],grade,flag,row['basis'],row['direction']):coordinate.push_back(_natural(v))
                d.coordinates.push_back(coordinate)
                if d.coordinates.size()>bounds[0]:raise ValueError('target coordinate admission exceeded')
            d.accession=_matrix(declaration['accession'],bounds[3]);d.reconstruction=_matrix(declaration['reconstruction'],bounds[3])
            with nogil:self._handle=shared_ptr[OccurrenceTargetBridge](new OccurrenceTargetBridge(actual,d,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[OccurrenceTargetBridge] handle
        with nogil:handle=OccurrenceTargetBridge.restore(image,bounds,maximum)
        return _wrap_port(handle,bounds,maximum)

    def archive(self):
        cdef shared_ptr[OccurrenceTargetBridge] handle=self._pin()
        cdef string image
        with nogil:image=handle.get().archive()
        return <bytes>image

    def inspect(self, *, max_records=256):
        cdef shared_ptr[OccurrenceTargetBridge] handle=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef OccurrenceTargetInfo v
        with nogil:v=handle.get().info(maximum)
        return {'key':int(v.key),'revision':int(v.revision),'node':int(v.node),'frame':int(v.frame),
                'coordinates':tuple({'key':int(c[0]),'occurrence':(int(c[1]),int(c[2])),'basis':int(c[3]),'direction':int(c[4])} for c in v.coordinates),
                'accession':<bytes>v.accession,'reconstruction':<bytes>v.reconstruction,'selection':<bytes>v.selection,'action':<bytes>v.action,
                'coordinate_metric':<bytes>v.coordinate_metric,'represented_metric':<bytes>v.represented_metric,'retained_entries':int(v.retained_entries)}

    def program(self):
        cdef shared_ptr[OccurrenceTargetBridge] handle=self._pin()
        cdef PreparedAccessionFieldProgram out=PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
        with nogil:out._handle=handle.get().program()
        out._limits=self._limits;out._maximum=self._maximum
        return out

    def rebind(self, PreparedAccessionFieldProgram program not None):
        cdef shared_ptr[OccurrenceTargetBridge] handle=self._pin(), result
        cdef shared_ptr[FieldProgramBridge] actual=program._pin()
        with nogil:result=handle.get().rebind(actual)
        return _wrap_port(result,self._limits,self._maximum)

    def same_owner(self, NativeOccurrenceTargetPort other not None):
        cdef shared_ptr[OccurrenceTargetBridge] handle=self._pin(), actual=other._pin()
        return bool(handle.get().same_owner(actual))


cdef EvidenceDeclaration _evidence_declaration(value, vector[size_t] bounds):
    _closed(value, ('profile', 'key', 'revision', 'node', 'variables', 'basis', 'parameters'))
    if value['profile'] != 'rexgraph.relational-field-parameters.v1': raise ValueError('unknown whole-field parameter profile')
    cdef EvidenceDeclaration out
    out.key = _natural(value['key']); out.revision = _natural(value['revision'])
    out.node = _natural(value['node']); out.variables = _indices(value['variables'])
    out.basis = _matrix(value['basis'], bounds[3])
    for key in value['parameters']: out.parameters.push_back(_natural(key))
    return out


cdef class PreparedFieldEvidence:

    cdef shared_ptr[FieldEvidenceBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native field evidence is not initialized')
        return self._handle

    def __init__(self, PreparedAccessionFieldProgram program not None, declaration,
                 *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native evidence owners are immutable')
        self._loading = True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef EvidenceDeclaration value
        cdef shared_ptr[FieldProgramBridge] actual
        cdef shared_ptr[FieldEvidenceBridge] handle
        try:
            bounds = _bounds(limits); maximum = _maximum(max_bytes); actual = program._pin()
            value = _evidence_declaration(declaration, bounds)
            with nogil: handle = shared_ptr[FieldEvidenceBridge](new FieldEvidenceBridge(actual, value, bounds, maximum))
            self._handle = handle; self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef PreparedFieldEvidence out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        with nogil: out._handle = FieldEvidenceBridge.restore(image, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef string value
        with nogil: value = handle.get().archive()
        return <bytes>value

    def identity(self):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef EvidenceIdentity value
        with nogil: value = handle.get().identity()
        return {'key': int(value.key), 'revision': int(value.revision), 'node': int(value.node), 'grade': int(value.grade),
                'parameter_count': int(value.parameters), 'free_parameter_count': int(value.free_parameters),
                'observation_count': int(value.observations), 'has_parent': bool(value.has_parent)}

    def inspect(self, *, max_records=256, typed=False):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef EvidenceDetails value
        if handle.get().has_occurrence_observations() and not typed:
            raise ValueError('occurrence history requires typed=True inspection')
        with nogil: value = handle.get().details(maximum)
        return {'variables': _rows(value.variables), 'parameters': tuple(int(k) for k in value.parameters),
                'free_parameters': tuple(int(k) for k in value.free_parameters), 'basis': <bytes>value.basis,
                'particular': <bytes>value.particular, 'kernel': <bytes>value.kernel, 'chain': <bytes>value.chain,
                'chain_coordinates': _rows(value.chain_coordinates), 'design': <bytes>value.design,
                'augmented': <bytes>value.augmented, 'target': <bytes>value.target,
                'observation_coordinates': _rows(value.typed_coordinates if typed else value.observation_coordinates),
                'parent_relation': {'shift': <bytes>value.parent_shift, 'map': <bytes>value.parent_map} if value.has_parent else None}

    def program(self):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef PreparedAccessionFieldProgram out = PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
        with nogil: out._handle = handle.get().program()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def parent(self):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin(), parent
        cdef PreparedFieldEvidence out
        with nogil: parent = handle.get().parent()
        if parent.get() == NULL: return None
        out = PreparedFieldEvidence.__new__(PreparedFieldEvidence)
        out._handle = parent; out._limits = self._limits; out._maximum = self._maximum
        return out

    def observation(self, index, *, max_records=256):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef size_t original = _natural(index), maximum = _natural(max_records)
        cdef EvidenceObservationInfo value
        with nogil: value = handle.get().observation(original, maximum)
        result = {'key': int(value.key), 'input': <bytes>value.input, 'target_asset': <bytes>value.target_asset,
                'target_use': int(value.target_use), 'target': <bytes>value.target, 'observed': _rows(value.observed)}
        if value.port.get()!=NULL:
            result.update(domain='original-occurrence',port=_wrap_port(value.port,self._limits,self._maximum),
                          observed_coordinates=tuple(int(k) for k in value.observed_coordinates))
        return result

    def update(self):
        cdef shared_ptr[FieldEvidenceBridge] handle=self._pin()
        cdef vector[size_t] value
        with nogil:value=handle.get().update()
        return dict(zip(('chain_reused','reused_observations','assembled_observations','execution_steps'),map(int,value)))

    def dependency_archives(self, *, max_records=256):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[string] values
        with nogil: values = handle.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in values)

    def append(self, observations):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef vector[EvidenceObservation] values
        cdef EvidenceObservation item
        cdef NativeAccessionFieldResult actual_input
        cdef NativeInputUseAsset target_asset
        cdef NativeOccurrenceTargetPort target_port
        cdef PreparedFieldEvidence out = PreparedFieldEvidence.__new__(PreparedFieldEvidence)
        for value in observations:
            item.port.reset();item.observed.clear();item.observed_coordinates.clear()
            if 'port' in value:
                _closed(value, ('key','input','target_asset','target_use','target','port','observed_coordinates'))
                if not isinstance(value['port'],NativeOccurrenceTargetPort):raise TypeError('actual native target port required')
                target_port=value['port'];item.port=target_port._pin()
                for k in value['observed_coordinates']:
                    item.observed_coordinates.push_back(_natural(k))
                    if item.observed_coordinates.size()>self._limits[0]:raise ValueError('target mask admission exceeded')
            else:
                _closed(value, ('key', 'input', 'target_asset', 'target_use', 'target', 'observed'))
                item.observed = _indices(value['observed'])
            if not isinstance(value['input'], NativeAccessionFieldResult) or not isinstance(value['target_asset'], NativeInputUseAsset):
                raise TypeError('actual native source input and target asset owners required')
            actual_input = value['input']; target_asset = value['target_asset']
            item.key = _natural(value['key']); item.input = actual_input._pin(); item.target_asset = target_asset._pin()
            item.target_use = _natural(value['target_use']); item.target = _image(value['target'], self._maximum)
            values.push_back(item)
            if values.size() > self._limits[0]: raise ValueError('evidence observation count exceeds admission')
        with nogil: out._handle = handle.get().append(values)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def execute(self, NativeAccessionFieldResult input not None):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef shared_ptr[AccessionResultBridge] actual = input._pin()
        cdef NativeAffineFieldResult out = NativeAffineFieldResult.__new__(NativeAffineFieldResult)
        with nogil: out._handle = handle.get().execute(actual)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def residual(self, observation):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef size_t index = _natural(observation)
        cdef NativeFieldResidual out = NativeFieldResidual.__new__(NativeFieldResidual)
        with nogil: out._handle = handle.get().residual(index)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def occurrence_map(self, grade, occurrence):
        cdef shared_ptr[FieldEvidenceBridge] handle = self._pin()
        cdef size_t selected_grade = _natural(grade), original = _natural(occurrence)
        cdef NativeOccurrenceMapFamily out = NativeOccurrenceMapFamily.__new__(NativeOccurrenceMapFamily)
        with nogil: out._handle = handle.get().occurrence_map(selected_grade, original)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeAffineFieldResult:

    cdef shared_ptr[AffineResultBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native affine field result is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeAffineFieldResult out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        with nogil: out._handle = AffineResultBridge.restore(image, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AffineResultBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    def evidence(self):
        cdef shared_ptr[AffineResultBridge] handle = self._pin()
        cdef PreparedFieldEvidence out = PreparedFieldEvidence.__new__(PreparedFieldEvidence)
        with nogil: out._handle = handle.get().evidence()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def input(self):
        cdef shared_ptr[AffineResultBridge] handle = self._pin()
        cdef NativeAccessionFieldResult out = NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil: out._handle = handle.get().input()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def program(self):
        return self.evidence().program()

    def input_uses(self, *, max_records=256):
        return self.input().input_uses(max_records=max_records)

    def fields(self, *, max_records=256):
        cdef shared_ptr[AffineResultBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[ProgramFieldInfo] values
        with nogil: values = handle.get().fields(maximum)
        return tuple({'node': int(v.node), 'grade': int(v.grade), 'dimension': int(v.dimension), 'supported': int(v.supported),
                      'nonzeros': int(v.nonzeros), 'occurrence_dimension': int(v.occurrence_dimension),
                      'has_occurrences': bool(v.has_occurrences)} for v in values)

    def field_archive(self, node, *, occurrences=False):
        if type(occurrences) is not bool: raise TypeError('explicit boolean occurrence selection required')
        cdef shared_ptr[AffineResultBridge] handle = self._pin()
        cdef uint64_t key = _natural(node)
        cdef cppbool flags = occurrences
        cdef AffineFieldInfo value
        with nogil: value = handle.get().field(key, flags)
        return _field(value)

    def identified(self, node, coordinate):
        cdef shared_ptr[AffineResultBridge] handle = self._pin()
        cdef uint64_t key = _natural(node)
        cdef size_t row = _natural(coordinate)
        cdef cppbool value
        with nogil: value = handle.get().identified(key, row)
        return bool(value)

    def select(self, node, cells, *, coordinate_limit=256):
        cdef shared_ptr[AffineResultBridge] handle = self._pin()
        cdef uint64_t key = _natural(node)
        cdef size_t maximum = _natural(coordinate_limit)
        cdef vector[vector[size_t]] requested = _indices(cells)
        cdef vector[AffineSelected] values
        with nogil: values = handle.get().select(key, requested, maximum)
        return tuple({'particular': _selected(v.particular), 'directions': <bytes>v.directions,
                      'identified': tuple(bool(b) for b in v.identified)} for v in values)

    def evaluate(self, free_values):
        cdef shared_ptr[AffineResultBridge] handle = self._pin()
        cdef vector[string] values
        cdef NativeAffineEvaluation out = NativeAffineEvaluation.__new__(NativeAffineEvaluation)
        for q in free_values:
            values.push_back(_scalar(q, self._limits[3]))
            if values.size() > self._limits[0]: raise ValueError('free value count exceeds admission')
        with nogil: out._handle = handle.get().evaluate(values)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeAffineEvaluation:

    cdef shared_ptr[AffineEvaluationBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native prospective evaluation is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeAffineEvaluation out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        with nogil: out._handle = AffineEvaluationBridge.restore(image, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AffineEvaluationBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    def family(self):
        cdef shared_ptr[AffineEvaluationBridge] handle = self._pin()
        cdef NativeAffineFieldResult out = NativeAffineFieldResult.__new__(NativeAffineFieldResult)
        with nogil: out._handle = handle.get().family()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def result(self):
        cdef shared_ptr[AffineEvaluationBridge] handle = self._pin()
        cdef NativeAccessionFieldResult out = NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil: out._handle = handle.get().result()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def free_values_archive(self):
        cdef shared_ptr[AffineEvaluationBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().free_values()
        return <bytes>image

    def program(self):
        return self.result().program()

    def fields(self, *, max_records=256):
        return self.result().fields(max_records=max_records)

    def field_archive(self, node, *, occurrences=False):
        return self.result().field_archive(node, occurrences=occurrences)

    def select(self, node, cells, *, coordinate_limit=256):
        return self.result().select(node, cells, coordinate_limit=coordinate_limit)

    def input_uses(self, *, max_records=256):
        return self.result().input_uses(max_records=max_records)


cdef class NativeFieldResidual:

    cdef shared_ptr[EvidenceResidualBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native whole-field residual is not initialized')
        return self._handle

    def field_archive(self):
        cdef shared_ptr[EvidenceResidualBridge] handle = self._pin()
        cdef AffineFieldInfo value
        with nogil: value = handle.get().field()
        return _field(value)

    def target_port(self):
        cdef shared_ptr[EvidenceResidualBridge] handle=self._pin()
        cdef shared_ptr[OccurrenceTargetBridge] value
        with nogil:value=handle.get().target_port()
        return _wrap_port(value,self._limits,self._maximum)

    def response(self):
        cdef shared_ptr[EvidenceResidualBridge] handle = self._pin()
        cdef NativeAffineFieldResult out = NativeAffineFieldResult.__new__(NativeAffineFieldResult)
        with nogil: out._handle = handle.get().response()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    @property
    def observation(self):
        cdef shared_ptr[EvidenceResidualBridge] handle = self._pin()
        cdef size_t value
        with nogil: value = handle.get().observation()
        return int(value)


cdef class NativeOccurrenceMapFamily:

    cdef shared_ptr[EvidenceMapBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native original occurrence map family is not initialized')
        return self._handle

    def inspect(self, *, max_records=256):
        cdef shared_ptr[EvidenceMapBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef AffineMapInfo value
        with nogil: value = handle.get().info(maximum)
        return {'grade': int(value.grade), 'occurrence': int(value.occurrence), 'particular': <bytes>value.particular,
                'directions': tuple(<bytes>m for m in value.directions)}

    def evidence(self):
        cdef shared_ptr[EvidenceMapBridge] handle = self._pin()
        cdef PreparedFieldEvidence out = PreparedFieldEvidence.__new__(PreparedFieldEvidence)
        with nogil: out._handle = handle.get().evidence()
        out._limits = self._limits; out._maximum = self._maximum
        return out


def field_family_build_info():
    return {'engine': 'rexgraph-cpp', 'engine_version': RFC_ENGINE_VERSION.decode(),
            'engine_headers': RFC_ENGINE_DIGEST.decode(), 'binding': RFC_FIELD_EVIDENCE_BINDING_DIGEST.decode(),
            'profile': 'owning-relational-field-family-v1', 'scalar_policy': 'gmp-rational-exact'}


cdef vector[vector[FamilyView]] _family_views(value, vector[size_t] bounds):
    cdef vector[vector[FamilyView]] out
    cdef vector[FamilyView] row
    cdef FamilyView item
    for cells in value:
        row.clear()
        for v in cells:
            _closed(v, ('frame', 'accession', 'reconstruction'))
            item.frame = _natural(v['frame'])
            item.accession = _matrix(v['accession'], bounds[3]); item.reconstruction = _matrix(v['reconstruction'], bounds[3])
            row.push_back(item)
            if row.size() > bounds[0]: raise ValueError('family view count exceeds admission')
        out.push_back(row)
        if out.size() > bounds[0]: raise ValueError('family grade count exceeds admission')
    return out


cdef FamilyDeclaration _family_declaration(value, vector[size_t] bounds, size_t maximum):
    _closed(value, ('profile', 'key', 'revision', 'interfaces'))
    if value['profile'] != 'rexgraph.relational-field-family.v1': raise ValueError('unknown native field family declaration')
    cdef FamilyDeclaration out
    cdef FamilyInterface item
    out.key = _natural(value['key']); out.revision = _natural(value['revision'])
    for v in value['interfaces']:
        _closed(v, ('key', 'revision', 'left', 'right', 'common', 'common_cutoff', 'common_left', 'common_right', 'left_views', 'right_views'))
        _closed(v['common_left'], ('cells', 'occurrences')); _closed(v['common_right'], ('cells', 'occurrences'))
        item.key = _natural(v['key']); item.revision = _natural(v['revision'])
        item.left = _natural(v['left']); item.right = _natural(v['right'])
        item.common = _image(v['common'], maximum); item.cutoff = _natural(v['common_cutoff'])
        item.left_cells = _indices(v['common_left']['cells']); item.left_occurrences = _indices(v['common_left']['occurrences'])
        item.right_cells = _indices(v['common_right']['cells']); item.right_occurrences = _indices(v['common_right']['occurrences'])
        item.left_views = _family_views(v['left_views'], bounds); item.right_views = _family_views(v['right_views'], bounds)
        out.interfaces.push_back(item)
        if out.interfaces.size() > bounds[0]: raise ValueError('family interface count exceeds admission')
    return out


cdef class PreparedFieldFamily:

    cdef shared_ptr[FieldFamilyBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native PreparedFieldFamily is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef PreparedFieldFamily out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        with nogil: out._handle = FieldFamilyBridge.restore(image, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    def __init__(self, declaration, members, *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native field family owners are immutable')
        self._loading = True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef FamilyDeclaration d
        cdef vector[shared_ptr[FieldEvidenceBridge]] actual
        cdef PreparedFieldEvidence member
        cdef shared_ptr[FieldFamilyBridge] handle
        try:
            bounds = _bounds(limits); maximum = _maximum(max_bytes)
            d = _family_declaration(declaration, bounds, maximum)
            for value in members:
                if not isinstance(value, PreparedFieldEvidence): raise TypeError('actual native evidence owners required')
                member = value; actual.push_back(member._pin())
                if actual.size() > bounds[0]: raise ValueError('family member count exceeds admission')
            with nogil: handle = shared_ptr[FieldFamilyBridge](new FieldFamilyBridge(d, actual, bounds, maximum))
            self._handle = handle; self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    def identity(self):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin()
        cdef FamilyIdentity v
        with nogil: v = handle.get().identity()
        return {'key': int(v.key), 'revision': int(v.revision), 'member_count': int(v.members),
                'interface_count': int(v.interfaces), 'free_parameter_count': int(v.free_parameters),
                'rank': int(v.rank), 'retained_entries': int(v.retained_entries), 'consistent': bool(v.consistent),
                'limits': tuple(int(i) for i in v.limits)}

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef FamilyProof v
        with nogil: v = handle.get().proof(maximum)
        return {'particular': <bytes>v.particular, 'kernel': <bytes>v.kernel,
                'coordinate_origin': <bytes>v.coordinate_origin, 'coordinate_kernel': <bytes>v.coordinate_kernel,
                'residual': <bytes>v.residual, 'equations': <bytes>v.equations, 'contradiction': <bytes>v.contradiction,
                'parameter_offsets': tuple(int(i) for i in v.parameter_offsets),
                'member_free_offsets': tuple(int(i) for i in v.member_free_offsets),
                'free_columns': tuple(int(i) for i in v.free_columns), 'statuses': tuple(int(i) for i in v.statuses),
                'equation_coordinates': _rows(v.equation_coordinates)}

    def witness(self, index, *, max_records=4096):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin()
        cdef size_t i = _natural(index), maximum = _natural(max_records)
        cdef FamilyWitness v
        with nogil: v = handle.get().witness(i, maximum)
        return {'interface': int(v.interface), 'grade': int(v.grade), 'occurrence': int(v.occurrence),
                'output_dimension': int(v.output_dimension), 'common_input': <bytes>v.common_input,
                'left_input': <bytes>v.left_input, 'right_input': <bytes>v.right_input,
                'response_shapes': _rows(v.response_shapes), 'equation_coordinates': _rows(v.equation_coordinates)}

    @property
    def witness_count(self):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin()
        cdef size_t count
        with nogil: count = handle.get().witnesses()
        return int(count)

    def member_coordinates(self, index):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin()
        cdef size_t i = _natural(index)
        cdef FamilyCoordinates v
        with nogil: v = handle.get().coordinates(i)
        return {'shift': <bytes>v.shift, 'map': <bytes>v.map}

    def member(self, index):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin()
        cdef size_t i = _natural(index)
        cdef PreparedFieldEvidence out = PreparedFieldEvidence.__new__(PreparedFieldEvidence)
        with nogil: out._handle = handle.get().member(i)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def same_owner(self, PreparedFieldFamily other not None):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin(), other_handle = other._pin()
        cdef cppbool value
        with nogil: value = handle.get().same_owner(other_handle)
        return bool(value)

    def execute(self, member, NativeAccessionFieldResult input not None):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin()
        cdef size_t i = _natural(member)
        cdef shared_ptr[AccessionResultBridge] actual = input._pin()
        cdef NativeJointFieldResult out = NativeJointFieldResult.__new__(NativeJointFieldResult)
        with nogil: out._handle = handle.get().execute(i, actual)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def occurrence_map(self, member, grade, occurrence):
        cdef shared_ptr[FieldFamilyBridge] handle = self._pin()
        cdef size_t i = _natural(member), k = _natural(grade), p = _natural(occurrence)
        cdef NativeJointOccurrenceMap out = NativeJointOccurrenceMap.__new__(NativeJointOccurrenceMap)
        with nogil: out._handle = handle.get().occurrence_map(i, k, p)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeJointFieldResult:

    cdef shared_ptr[JointResultBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native NativeJointFieldResult is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeJointFieldResult out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        with nogil: out._handle = JointResultBridge.restore(image, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    @property
    def member_index(self):
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef size_t i
        with nogil: i = handle.get().member()
        return int(i)

    def program(self):
        return self.original().program()

    def input_uses(self, *, max_records=256):
        return self.input().input_uses(max_records=max_records)

    def fields(self, *, max_records=256):
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[ProgramFieldInfo] values
        with nogil: values = handle.get().fields(maximum)
        return tuple({'node': int(v.node), 'grade': int(v.grade), 'dimension': int(v.dimension), 'supported': int(v.supported),
                      'nonzeros': int(v.nonzeros), 'occurrence_dimension': int(v.occurrence_dimension),
                      'has_occurrences': bool(v.has_occurrences)} for v in values)

    def field_archive(self, node, *, occurrences=False):
        if type(occurrences) is not bool: raise TypeError('explicit boolean occurrence selection required')
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef uint64_t key = _natural(node)
        cdef cppbool flags = occurrences
        cdef AffineFieldInfo v
        with nogil: v = handle.get().field(key, flags)
        return _field(v)

    def select(self, node, cells, *, coordinate_limit=256):
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef uint64_t key = _natural(node)
        cdef size_t maximum = _natural(coordinate_limit)
        cdef vector[vector[size_t]] requested = _indices(cells)
        cdef vector[AffineSelected] values
        with nogil: values = handle.get().select(key, requested, maximum)
        return tuple({'particular': _selected(v.particular), 'directions': <bytes>v.directions,
                      'identified': tuple(bool(b) for b in v.identified)} for v in values)

    def identified(self, node, coordinate):
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef uint64_t key = _natural(node)
        cdef size_t row = _natural(coordinate)
        cdef cppbool value
        with nogil: value = handle.get().identified(key, row)
        return bool(value)

    def evaluate(self, free_values):
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef vector[string] values
        cdef NativeJointFieldEvaluation out = NativeJointFieldEvaluation.__new__(NativeJointFieldEvaluation)
        for q in free_values:
            values.push_back(_scalar(q, self._limits[3]))
            if values.size() > self._limits[0]: raise ValueError('free value count exceeds admission')
        with nogil: out._handle = handle.get().evaluate(values)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def family(self):
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef PreparedFieldFamily out = PreparedFieldFamily.__new__(PreparedFieldFamily)
        with nogil: out._handle = handle.get().family()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def original(self):
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef NativeAffineFieldResult out = NativeAffineFieldResult.__new__(NativeAffineFieldResult)
        with nogil: out._handle = handle.get().original()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def input(self):
        cdef shared_ptr[JointResultBridge] handle = self._pin()
        cdef NativeAccessionFieldResult out = NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil: out._handle = handle.get().input()
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeJointFieldEvaluation:

    cdef shared_ptr[JointEvaluationBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native NativeJointFieldEvaluation is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeJointFieldEvaluation out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        with nogil: out._handle = JointEvaluationBridge.restore(image, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[JointEvaluationBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    def free_values_archive(self):
        cdef shared_ptr[JointEvaluationBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().free_values()
        return <bytes>image

    def program(self): return self.result().program()
    def fields(self, *, max_records=256): return self.result().fields(max_records=max_records)
    def field_archive(self, node, *, occurrences=False): return self.result().field_archive(node, occurrences=occurrences)
    def select(self, node, cells, *, coordinate_limit=256): return self.result().select(node, cells, coordinate_limit=coordinate_limit)
    def input_uses(self, *, max_records=256): return self.result().input_uses(max_records=max_records)

    def family(self):
        cdef shared_ptr[JointEvaluationBridge] handle = self._pin()
        cdef NativeJointFieldResult out = NativeJointFieldResult.__new__(NativeJointFieldResult)
        with nogil: out._handle = handle.get().family()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def result(self):
        cdef shared_ptr[JointEvaluationBridge] handle = self._pin()
        cdef NativeAccessionFieldResult out = NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil: out._handle = handle.get().result()
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeJointOccurrenceMap:

    cdef shared_ptr[JointMapBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native NativeJointOccurrenceMap is not initialized')
        return self._handle

    def inspect(self, *, max_records=256):
        cdef shared_ptr[JointMapBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef AffineMapInfo v
        with nogil: v = handle.get().info(maximum)
        return {'grade': int(v.grade), 'occurrence': int(v.occurrence), 'particular': <bytes>v.particular,
                'directions': tuple(<bytes>m for m in v.directions)}

    @property
    def member_index(self):
        cdef shared_ptr[JointMapBridge] handle = self._pin()
        cdef size_t value
        with nogil: value = handle.get().member()
        return int(value)

    def family(self):
        cdef shared_ptr[JointMapBridge] handle = self._pin()
        cdef PreparedFieldFamily out = PreparedFieldFamily.__new__(PreparedFieldFamily)
        with nogil: out._handle = handle.get().family()
        out._limits = self._limits; out._maximum = self._maximum
        return out

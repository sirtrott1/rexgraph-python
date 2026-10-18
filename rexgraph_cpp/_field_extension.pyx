# distutils: language=c++
"""Owned faithful source extensions and complete field completion families."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport (
    PreparedAccessionFieldProgram, NativeInputAccession, NativeAccessionFieldResult,
    _bounds, _maximum, _image, _matrix, _indices, _selected, ProgramSelected,
)
from ._field_admission cimport PreparedAvailableFieldProgram, NativeAvailabilityContext, _input
from ._field_evidence_admission cimport NativeAvailableAffineEvaluation
from ._field_program import _closed
from ._stalk_fields import _natural, _scalar

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FIELD_EXTENSION_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)


def field_extension_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_FIELD_EXTENSION_BINDING_DIGEST.decode(),
            'profile':'owning-relational-field-extension-v1','scalar_policy':'gmp-rational-exact'}


cdef unsigned _status(value) except *:
    if value=='observed':return 0
    if value=='prospective':return 1
    raise ValueError('extension relationship requires explicit observed or prospective status')


cdef ExtensionDeclaration _declaration(value, vector[size_t] bounds):
    _closed(value,('key','revision','free_carrier','free_basis','free_directions','source_map','stalks','isometric'))
    cdef ExtensionDeclaration out
    cdef ExtensionStalk stalk
    out.key=_natural(value['key']);out.revision=_natural(value['revision'])
    out.free_carrier=_natural(value['free_carrier']);out.free_basis=_natural(value['free_basis'])
    for direction in value['free_directions']:
        out.free_directions.push_back(_natural(direction))
        if out.free_directions.size()>bounds[0]:raise ValueError('completion carrier exceeds dimension admission')
    if type(value['isometric']) is not bool:raise TypeError('explicit metric pullback declaration required')
    out.isometric=value['isometric'];mapping=value['source_map'];_closed(mapping,('cells','occurrences'))
    out.cells=_indices(mapping['cells']);out.occurrences=_indices(mapping['occurrences'])
    for item in value['stalks']:
        _closed(item,('cell','source_basis','target_basis','inclusion','reconstruction'))
        if len(item['cell'])!=2:raise ValueError('original stalk requires grade and cell index')
        stalk.grade=_natural(item['cell'][0]);stalk.cell=_natural(item['cell'][1])
        stalk.source_basis=_natural(item['source_basis']);stalk.target_basis=_natural(item['target_basis'])
        stalk.inclusion=_matrix(item['inclusion'],bounds[3]);stalk.reconstruction=_matrix(item['reconstruction'],bounds[3])
        out.stalks.push_back(stalk)
        if out.stalks.size()>bounds[0]:raise ValueError('extension stalk count exceeds admission')
    return out


cdef void _parent(value, shared_ptr[AvailableProgramBridge]& fixed, shared_ptr[AvailableEvaluationBridge]& selected) except *:
    if isinstance(value,PreparedAvailableFieldProgram):fixed=(<PreparedAvailableFieldProgram>value)._pin()
    elif isinstance(value,NativeAvailableAffineEvaluation):selected=(<NativeAvailableAffineEvaluation>value)._pin()
    else:raise TypeError('actual fixed program or explicitly selected available affine parent required')


cdef vector[string] _free(values, vector[size_t] bounds):
    cdef vector[string] out
    for value in values:
        out.push_back(_scalar(value,bounds[3]))
        if out.size()>bounds[0]:raise ValueError('completion dimension exceeds admission')
    return out


def field_extension_origin_payload(parent, PreparedAvailableFieldProgram target not None, declaration, status,
                                   *, limits=_LIMITS,max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef ExtensionDeclaration d=_declaration(declaration,bounds)
    cdef unsigned kind=_status(status)
    cdef shared_ptr[AvailableProgramBridge] fixed, next=target._pin()
    cdef shared_ptr[AvailableEvaluationBridge] selected
    cdef string image
    _parent(parent,fixed,selected)
    with nogil:image=ExtensionModelBridge.root_payload(fixed,selected,next,d,kind,bounds,maximum)
    return <bytes>image


cdef class PreparedAvailableFieldExtension:

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef PreparedAvailableFieldExtension out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:out._handle=ExtensionModelBridge.restore(image,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    def dependency_archives(self, *, max_records=4096):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] images
        with nogil:images=actual.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)
    cdef shared_ptr[ExtensionModelBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('native extension is not initialized')
        return self._handle

    def __init__(self, parent, PreparedAvailableFieldProgram target not None, declaration, status, origin,
                 NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL:raise TypeError('native extension owner is immutable')
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef ExtensionDeclaration d=_declaration(declaration,bounds)
        cdef unsigned kind=_status(status)
        cdef uint64_t key=_natural(origin)
        cdef shared_ptr[AvailableProgramBridge] fixed, next=target._pin()
        cdef shared_ptr[AvailableEvaluationBridge] selected
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        _parent(parent,fixed,selected);self._loading=True
        try:
            with nogil:self._handle=shared_ptr[ExtensionModelBridge](new ExtensionModelBridge(fixed,selected,next,d,kind,key,authority,bounds,maximum))
            self._limits=bounds;self._maximum=maximum
        finally:self._loading=False

    def admission(self):return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def source(self):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef PreparedAccessionFieldProgram out=PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
        with nogil:out._handle=actual.get().source()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def target(self):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef PreparedAvailableFieldProgram out=PreparedAvailableFieldProgram.__new__(PreparedAvailableFieldProgram)
        with nogil:out._handle=actual.get().target()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def parent(self):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin(), previous
        cdef PreparedAvailableFieldExtension out
        with nogil:previous=actual.get().parent()
        if previous.get()==NULL:return None
        out=PreparedAvailableFieldExtension.__new__(PreparedAvailableFieldExtension);out._handle=previous
        out._limits=self._limits;out._maximum=self._maximum;return out

    def root(self):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef shared_ptr[AvailableProgramBridge] fixed
        cdef shared_ptr[AvailableEvaluationBridge] selected
        cdef PreparedAvailableFieldProgram f
        cdef NativeAvailableAffineEvaluation s
        with nogil:fixed=actual.get().fixed_parent();selected=actual.get().selected_parent()
        if fixed.get()!=NULL:
            f=PreparedAvailableFieldProgram.__new__(PreparedAvailableFieldProgram);f._handle=fixed
            f._limits=self._limits;f._maximum=self._maximum;return f
        if selected.get()!=NULL:
            s=NativeAvailableAffineEvaluation.__new__(NativeAvailableAffineEvaluation);s._handle=selected
            s._limits=self._limits;s._maximum=self._maximum;return s
        return None

    def info(self):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef ExtensionInfo info
        cdef vector[string] row
        cdef string inclusion,reconstruction,kernel
        with nogil:info=actual.get().info()
        maps=[]
        for row in info.grade_maps:
            inclusion=row[0];reconstruction=row[1];kernel=row[2]
            maps.append({'inclusion':<bytes>inclusion,'reconstruction':<bytes>reconstruction,'kernel':<bytes>kernel})
        return {'key':info.key,'revision':info.revision,'free_carrier':info.free_carrier,'free_basis':info.free_basis,
                'free_directions':tuple(info.free_directions),'status':'observed' if info.status==0 else 'prospective',
                'isometric':bool(info.isometric),'has_parent':bool(info.has_parent),'grade_maps':tuple(maps),
                'source_map':{'cells':tuple(tuple(row) for row in info.cells),'occurrences':tuple(tuple(row) for row in info.occurrences)}}

    def extension_payload(self, PreparedAvailableFieldProgram target not None, declaration, status):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef shared_ptr[AvailableProgramBridge] next=target._pin()
        cdef ExtensionDeclaration d=_declaration(declaration,self._limits)
        cdef unsigned kind=_status(status)
        cdef string image
        with nogil:image=actual.get().extension_payload(next,d,kind)
        return <bytes>image

    def extend(self, PreparedAvailableFieldProgram target not None, declaration, status, origin,
               NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef shared_ptr[AvailableProgramBridge] next=target._pin()
        cdef ExtensionDeclaration d=_declaration(declaration,self._limits)
        cdef unsigned kind=_status(status)
        cdef uint64_t key=_natural(origin)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef PreparedAvailableFieldExtension out=PreparedAvailableFieldExtension.__new__(PreparedAvailableFieldExtension)
        with nogil:out._handle=actual.get().extend(next,d,kind,key,authority)
        out._limits=self._limits;out._maximum=self._maximum;return out

    def transport_accession(self, NativeInputAccession original not None, key, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef shared_ptr[InputAccessionBridge] owner=original._pin()
        cdef uint64_t identity=_natural(key)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeInputAccession out=NativeInputAccession.__new__(NativeInputAccession)
        with nogil:out._handle=actual.get().transport(owner,identity,authority)
        out._limits=self._limits;out._maximum=self._maximum;return out

    def execute_inputs(self, values, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[ExtensionModelBridge] actual=self._pin()
        cdef vector[AdmissionInput] rows
        for value in values:
            rows.push_back(_input(value,self._limits,self._maximum,True))
            if rows.size()>self._limits[0]:raise ValueError('input count exceeds admission')
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableExtensionFamily out=NativeAvailableExtensionFamily.__new__(NativeAvailableExtensionFamily)
        with nogil:out._handle=actual.get().execute(rows,authority)
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeAvailableExtensionFamily:

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeAvailableExtensionFamily out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:out._handle=ExtensionFamilyBridge.restore(image,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[ExtensionFamilyBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[ExtensionFamilyBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    def dependency_archives(self, *, max_records=4096):
        cdef shared_ptr[ExtensionFamilyBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] images
        with nogil:images=actual.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)

    def fields(self, *, max_records=256):
        cdef shared_ptr[ExtensionFamilyBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[ProgramFieldInfo] values
        with nogil:values=actual.get().fields(maximum)
        return tuple({'node':value.node,'grade':value.grade,'dimension':value.dimension,'supported':value.supported,
                      'nonzeros':value.nonzeros,'has_occurrences':bool(value.has_occurrences),
                      'occurrence_dimension':value.occurrence_dimension} for value in values)
    cdef shared_ptr[ExtensionFamilyBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('native extension family is not initialized')
        return self._handle

    def model(self):
        cdef shared_ptr[ExtensionFamilyBridge] actual=self._pin()
        cdef PreparedAvailableFieldExtension out=PreparedAvailableFieldExtension.__new__(PreparedAvailableFieldExtension)
        with nogil:out._handle=actual.get().model()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def input(self):
        cdef shared_ptr[ExtensionFamilyBridge] actual=self._pin()
        cdef NativeAccessionFieldResult out=NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil:out._handle=actual.get().input()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def field_archive(self, node, *, occurrences=False):
        if type(occurrences) is not bool:raise TypeError('explicit occurrence selection required')
        cdef shared_ptr[ExtensionFamilyBridge] actual=self._pin()
        cdef uint64_t key=_natural(node)
        cdef cppbool flags=occurrences
        cdef AffineFieldInfo info
        with nogil:info=actual.get().field(key,flags)
        return {'particular':<bytes>info.particular,'directions':<bytes>info.directions,'dimension':info.dimension,
                'supported':info.supported,'nonzeros':info.nonzeros,'free_parameter_count':info.free_parameters,
                'direction_nonzeros':info.direction_nonzeros}

    def select(self, node, cells, *, coordinate_limit=256):
        cdef shared_ptr[ExtensionFamilyBridge] handle=self._pin()
        cdef uint64_t key=_natural(node)
        cdef size_t maximum=_natural(coordinate_limit)
        cdef vector[vector[size_t]] requested=_indices(cells)
        cdef vector[AffineSelected] values
        with nogil:values=handle.get().select(key,requested,maximum)
        return tuple({'particular':_selected(v.particular),'directions':<bytes>v.directions,
                      'identified':tuple(bool(b) for b in v.identified)} for v in values)

    def completion_payload(self, values):
        cdef shared_ptr[ExtensionFamilyBridge] actual=self._pin()
        cdef vector[string] free=_free(values,self._limits)
        cdef string image
        with nogil:image=actual.get().completion_payload(free)
        return <bytes>image

    def evaluate(self, values, origin, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[ExtensionFamilyBridge] actual=self._pin()
        cdef vector[string] free=_free(values,self._limits)
        cdef uint64_t key=_natural(origin)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeAvailableExtensionEvaluation out=NativeAvailableExtensionEvaluation.__new__(NativeAvailableExtensionEvaluation)
        with nogil:out._handle=actual.get().evaluate(free,key,authority)
        out._limits=self._limits;out._maximum=self._maximum;return out


cdef class NativeAvailableExtensionEvaluation:

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeAvailableExtensionEvaluation out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:out._handle=ExtensionEvaluationBridge.restore(image,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[ExtensionEvaluationBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().archive()
        return <bytes>image

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[ExtensionEvaluationBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil:actual.get().admit(authority)

    def dependency_archives(self, *, max_records=4096):
        cdef shared_ptr[ExtensionEvaluationBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] images
        with nogil:images=actual.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)

    def fields(self, *, max_records=256):
        cdef shared_ptr[ExtensionEvaluationBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[ProgramFieldInfo] values
        with nogil:values=actual.get().result().get().fields(maximum)
        return tuple({'node':value.node,'grade':value.grade,'dimension':value.dimension,'supported':value.supported,
                      'nonzeros':value.nonzeros,'has_occurrences':bool(value.has_occurrences),
                      'occurrence_dimension':value.occurrence_dimension} for value in values)
    cdef shared_ptr[ExtensionEvaluationBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('native extension evaluation is not initialized')
        return self._handle

    def family(self):
        cdef shared_ptr[ExtensionEvaluationBridge] actual=self._pin()
        cdef NativeAvailableExtensionFamily out=NativeAvailableExtensionFamily.__new__(NativeAvailableExtensionFamily)
        with nogil:out._handle=actual.get().family()
        out._limits=self._limits;out._maximum=self._maximum;return out

    def field_archive(self, node, *, occurrences=False):
        if type(occurrences) is not bool:raise TypeError('explicit occurrence selection required')
        cdef shared_ptr[ExtensionEvaluationBridge] actual=self._pin()
        cdef uint64_t key=_natural(node)
        cdef cppbool flags=occurrences
        cdef string image
        with nogil:image=actual.get().result().get().field(key,flags)
        return <bytes>image

    def select(self,node,cells,*,coordinate_limit=256):
        cdef shared_ptr[ExtensionEvaluationBridge] actual=self._pin()
        cdef uint64_t key=_natural(node)
        cdef vector[vector[size_t]] requested=_indices(cells)
        cdef size_t maximum=_natural(coordinate_limit)
        cdef vector[ProgramSelected] values
        with nogil:values=actual.get().result().get().select(key,requested,maximum)
        return tuple(_selected(value) for value in values)

    def free_values_archive(self):
        cdef shared_ptr[ExtensionEvaluationBridge] actual=self._pin()
        cdef string image
        with nogil:image=actual.get().free_values()
        return <bytes>image

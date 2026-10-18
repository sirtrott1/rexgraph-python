# distutils: language=c++
"""Owned exact local solves, standalone packets and joined native work."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport ProgramMatrix, ProgramSourceInfo, ProgramSelected, InterfaceView, _bounds, _maximum, _matrix, _indices, _image, _publication, _selected
from ._field_admission cimport AvailableResultBridge, AvailabilityContextBridge, NativeAvailableFieldResult, NativeAvailabilityContext, _keys
from ._field_evidence_admission cimport AvailableEvaluationBridge, NativeAvailableAffineEvaluation
from ._field_extension cimport ExtensionEvaluationBridge, NativeAvailableExtensionEvaluation
from ._field_program import _closed
from ._stalk_fields import _natural, _scalar

cdef extern from "field_local_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass LocalDeclaration:
        uint64_t key, revision, coordinate_carrier, coordinate_basis
        size_t grade
        unsigned kind
        string lambda_ "lambda"
        cppbool has_frame
        ProgramMatrix frame
        vector[size_t] interior
    cdef cppclass LocalReadout:
        vector[vector[size_t]] patch
        InterfaceView view
    cdef cppclass LocalInfo:
        uint64_t key, revision, node, coordinate_carrier, coordinate_basis
        size_t grade
        unsigned kind
        string lambda_ "lambda"
        vector[size_t] interior, interface, layout
        vector[vector[size_t]] patch
        uint64_t readout_key, readout_basis
        vector[uint64_t] readout_directions
        vector[vector[string]] matrices
    cdef cppclass LocalResultBridge:
        @staticmethod
        shared_ptr[LocalResultBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        vector[string] payloads(shared_ptr[AvailableResultBridge], shared_ptr[AvailableEvaluationBridge], shared_ptr[ExtensionEvaluationBridge], uint64_t,
            const LocalDeclaration&, const LocalReadout&, const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        string field(const string&) except +
        LocalInfo info() except +
        shared_ptr[AvailableResultBridge] fixed_rhs() except +
        shared_ptr[AvailableEvaluationBridge] selected_rhs() except +
        shared_ptr[ExtensionEvaluationBridge] extension_rhs() except +
        ProgramSourceInfo source() except +
        vector[string] dependency_archives(size_t) except +
        vector[ProgramSelected] select(uint64_t, const vector[vector[size_t]]&, size_t) except +
    cdef cppclass LocalWorkBridge:
        @staticmethod
        shared_ptr[LocalWorkBridge] start(shared_ptr[AvailableResultBridge], shared_ptr[AvailableEvaluationBridge], shared_ptr[ExtensionEvaluationBridge], uint64_t,
            const LocalDeclaration&, const LocalReadout&, const vector[uint64_t]&, shared_ptr[AvailabilityContextBridge],
            const vector[size_t]&, size_t, const vector[size_t]&, int) except +
        @staticmethod
        shared_ptr[LocalWorkBridge] replay(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t,
            const vector[size_t]&, int) except +
        void resume() except +
        void cancel() except +
        int phase() except +
        shared_ptr[LocalResultBridge] wait(shared_ptr[AvailabilityContextBridge]) except +
        vector[uint64_t] receipt() except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FIELD_LOCAL_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)
_WORK=(1,1,1,256*1024*1024,256*1024*1024)
_PHASES=('preparation','action','replay')

def field_local_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_FIELD_LOCAL_BINDING_DIGEST.decode(),
            'profile':'owning-relational-local-solve-v1','scalar_policy':'gmp-rational-exact'}

cdef LocalDeclaration _declaration(value, vector[size_t] bounds):
    _closed(value,('key','revision','coordinate_carrier','coordinate_basis','grade','kind','lambda','harmonic_frame','interior'))
    cdef LocalDeclaration out
    out.key=_natural(value['key']);out.revision=_natural(value['revision'])
    out.coordinate_carrier=_natural(value['coordinate_carrier']);out.coordinate_basis=_natural(value['coordinate_basis'])
    out.grade=_natural(value['grade']);out.lambda_=_scalar(value['lambda'],bounds[3])
    if value['kind'] not in ('green','harmonic-inverse'): raise ValueError('unknown exact solve kind')
    out.kind=0 if value['kind']=='green' else 1
    out.has_frame=value['harmonic_frame'] is not None
    if out.has_frame: out.frame=_matrix(value['harmonic_frame'],bounds[3])
    for i in value['interior']:
        out.interior.push_back(_natural(i))
        if out.interior.size()>bounds[0]: raise ValueError('interior exceeds dimension admission')
    return out

cdef LocalReadout _readout(value, vector[size_t] bounds):
    _closed(value,('patch','view'))
    cdef LocalReadout out
    cdef InterfaceView view
    out.patch=_indices(value['patch'])
    item=value['view'];_closed(item,('key','basis','directions','accession','reconstruction','physical'))
    view.key=_natural(item['key']);view.basis=_natural(item['basis'])
    for key in item['directions']: view.directions.push_back(_natural(key))
    view.accession=_matrix(item['accession'],bounds[3]);view.reconstruction=_matrix(item['reconstruction'],bounds[3])
    view.has_physical=item['physical'] is not None
    if view.has_physical: view.physical=_matrix(item['physical'],bounds[3])
    out.view=view;return out

cdef vector[size_t] _work(value):
    if len(value)!=5: raise ValueError('work requires key, generation, workers, buffer bytes and result bytes')
    cdef vector[size_t] out
    for item in value: out.push_back(_natural(item))
    return out

cdef int _pause(value) except -2:
    if value is None: return -1
    if value not in _PHASES: raise ValueError('unknown native phase')
    return _PHASES.index(value)

cdef void _rhs(value, shared_ptr[AvailableResultBridge]& fixed, shared_ptr[AvailableEvaluationBridge]& selected, shared_ptr[ExtensionEvaluationBridge]& extended) except *:
    if isinstance(value,NativeAvailableFieldResult): fixed=(<NativeAvailableFieldResult>value)._pin()
    elif isinstance(value,NativeAvailableAffineEvaluation): selected=(<NativeAvailableAffineEvaluation>value)._pin()
    elif isinstance(value,NativeAvailableExtensionEvaluation): extended=(<NativeAvailableExtensionEvaluation>value)._pin()
    else: raise TypeError('actual admitted fixed, selected affine or completed extension whole-field RHS required')

def local_field_origin_payloads(rhs, node, declaration, readout, *, limits=_LIMITS, max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef uint64_t key=_natural(node)
    cdef LocalDeclaration d=_declaration(declaration,bounds)
    cdef LocalReadout r=_readout(readout,bounds)
    cdef shared_ptr[AvailableResultBridge] fixed
    cdef shared_ptr[AvailableEvaluationBridge] selected
    cdef shared_ptr[ExtensionEvaluationBridge] extended
    cdef vector[string] images
    cdef string problem, reconstruction
    _rhs(rhs,fixed,selected,extended)
    with nogil: images=LocalResultBridge.payloads(fixed,selected,extended,key,d,r,bounds,maximum)
    problem=images[0];reconstruction=images[1]
    return {'problem':<bytes>problem,'interface':<bytes>problem,
            'reconstruction':<bytes>reconstruction,'gauge':<bytes>problem}

def submit_local_field_solve(rhs, node, declaration, readout, origins, NativeAvailabilityContext execution_context not None,
                           *, limits=_LIMITS, max_bytes=256*1024*1024, work=_WORK, pause_at=None):
    cdef vector[size_t] bounds=_bounds(limits), demand=_work(work)
    cdef size_t maximum=_maximum(max_bytes)
    cdef uint64_t key=_natural(node)
    cdef LocalDeclaration d=_declaration(declaration,bounds)
    cdef LocalReadout r=_readout(readout,bounds)
    _closed(origins,('problem','interface','reconstruction','gauge'))
    cdef vector[uint64_t] keys=_keys({name:origins[name] for name in ('problem','interface','reconstruction')},('problem','interface','reconstruction'))
    keys.push_back(_natural(origins['gauge']))
    cdef int pause=_pause(pause_at)
    cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
    cdef shared_ptr[AvailableResultBridge] fixed
    cdef shared_ptr[AvailableEvaluationBridge] selected
    cdef shared_ptr[ExtensionEvaluationBridge] extended
    cdef NativeLocalFieldWork out=NativeLocalFieldWork.__new__(NativeLocalFieldWork)
    _rhs(rhs,fixed,selected,extended)
    with nogil: out._handle=LocalWorkBridge.start(fixed,selected,extended,key,d,r,keys,authority,bounds,maximum,demand,pause)
    out._limits=bounds;out._maximum=maximum;return out

cdef class NativeLocalFieldResult:
    cdef shared_ptr[LocalResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum

    cdef shared_ptr[LocalResultBridge] _pin(self) except *:
        if self._handle.get()==NULL: raise RuntimeError('native local result is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024):
        cdef NativeLocalFieldResult out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil: out._handle=LocalResultBridge.restore(image,authority,bounds,maximum)
        out._limits=bounds;out._maximum=maximum;return out

    def archive(self):
        cdef shared_ptr[LocalResultBridge] actual=self._pin()
        cdef string image
        with nogil: image=actual.get().archive()
        return <bytes>image

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[LocalResultBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil: actual.get().admit(authority)

    def field_archive(self, name='global'):
        cdef shared_ptr[LocalResultBridge] actual=self._pin()
        cdef string key=name.encode('ascii'), image
        with nogil: image=actual.get().field(key)
        return <bytes>image

    def info(self):
        cdef shared_ptr[LocalResultBridge] actual=self._pin()
        cdef LocalInfo info
        cdef vector[string] row
        cdef string name, image
        with nogil: info=actual.get().info()
        matrices={}
        for row in info.matrices:
            name=row[0];image=row[1];matrices[(<bytes>name).decode('ascii')]=<bytes>image
        return {'key':info.key,'revision':info.revision,'node':info.node,'grade':info.grade,
                'coordinate_carrier':info.coordinate_carrier,'coordinate_basis':info.coordinate_basis,
                'kind':'green' if info.kind==0 else 'harmonic-inverse','lambda':(<bytes>info.lambda_).decode('ascii'),
                'interior':tuple(info.interior),'interface':tuple(info.interface),'layout':tuple(info.layout),
                'patch':tuple(tuple(row) for row in info.patch),'readout_key':info.readout_key,
                'readout_basis':info.readout_basis,'readout_directions':tuple(info.readout_directions),
                'matrices':matrices}

    def select(self, node, cells, *, coordinate_limit=256):
        cdef shared_ptr[LocalResultBridge] actual=self._pin()
        cdef uint64_t key=_natural(node)
        cdef vector[vector[size_t]] requested=_indices(cells)
        cdef size_t maximum=_natural(coordinate_limit)
        cdef vector[ProgramSelected] values
        with nogil: values=actual.get().select(key,requested,maximum)
        return tuple(_selected(value) for value in values)

    def source(self):
        cdef shared_ptr[LocalResultBridge] actual=self._pin()
        cdef ProgramSourceInfo info
        with nogil: info=actual.get().source()
        return _publication(info)

    def rhs(self):
        cdef shared_ptr[LocalResultBridge] actual=self._pin()
        cdef shared_ptr[AvailableResultBridge] fixed
        cdef NativeAvailableFieldResult f
        cdef NativeAvailableAffineEvaluation s
        cdef NativeAvailableExtensionEvaluation e
        cdef shared_ptr[AvailableEvaluationBridge] selected
        with nogil: fixed=actual.get().fixed_rhs()
        if fixed.get()!=NULL:
            f=NativeAvailableFieldResult.__new__(NativeAvailableFieldResult);f._handle=fixed
            f._limits=self._limits;f._maximum=self._maximum;return f
        with nogil: selected=actual.get().selected_rhs()
        if selected.get()!=NULL:
            s=NativeAvailableAffineEvaluation.__new__(NativeAvailableAffineEvaluation);s._handle=selected
            s._limits=self._limits;s._maximum=self._maximum;return s
        e=NativeAvailableExtensionEvaluation.__new__(NativeAvailableExtensionEvaluation)
        with nogil: e._handle=actual.get().extension_rhs()
        e._limits=self._limits;e._maximum=self._maximum;return e

    def dependency_archives(self, *, max_records=4096):
        cdef shared_ptr[LocalResultBridge] actual=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[string] images
        with nogil: images=actual.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)

cdef class NativeLocalFieldWork:
    cdef shared_ptr[LocalWorkBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum

    cdef shared_ptr[LocalWorkBridge] _pin(self) except *:
        if self._handle.get()==NULL: raise RuntimeError('native local work is not initialized')
        return self._handle

    @classmethod
    def replay(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS,max_bytes=256*1024*1024,work=_WORK,pause_at=None):
        cdef NativeLocalFieldWork out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits), demand=_work(work)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        cdef int pause=_pause(pause_at)
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        with nogil: out._handle=LocalWorkBridge.replay(image,authority,bounds,maximum,demand,pause)
        out._limits=bounds;out._maximum=maximum;return out

    def phase(self):
        cdef shared_ptr[LocalWorkBridge] actual=self._pin()
        cdef int value
        with nogil: value=actual.get().phase()
        return None if value<0 else _PHASES[value]

    def resume(self):
        cdef shared_ptr[LocalWorkBridge] actual=self._pin()
        with nogil: actual.get().resume()

    def cancel(self):
        cdef shared_ptr[LocalWorkBridge] actual=self._pin()
        with nogil: actual.get().cancel()

    def wait(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[LocalWorkBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef NativeLocalFieldResult out=NativeLocalFieldResult.__new__(NativeLocalFieldResult)
        with nogil: out._handle=actual.get().wait(authority)
        out._limits=self._limits;out._maximum=self._maximum;return out

    def receipt(self):
        cdef shared_ptr[LocalWorkBridge] actual=self._pin()
        cdef vector[uint64_t] values
        with nogil: values=actual.get().receipt()
        return {'key':values[0],'generation':values[1],
                'status':('queued','running','succeeded','failed','cancelled','blocked')[values[2]],
                'queue_ns':values[3],'execution_ns':values[4],'joined':bool(values[5])}

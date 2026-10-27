# distutils: language=c++
"""Owning native action recipes, source evidence requests and whole results."""
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as cppbool
from libc.stdint cimport uint64_t
from libc.stddef cimport size_t
from ._stalk_fields import _natural, _scalar
from ._field_program import _closed
from ._temporal_program import _rational

cdef extern from "action_program_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass ActionNode:
        uint64_t id, opcode, selector
        vector[uint64_t] operands
        string scale
    cdef cppclass ActionRecipe:
        uint64_t version, name_space, sequence, output, result, metric, reference
        cppbool has_reference
        string identity, meaning
        vector[ActionNode] nodes
        vector[string] labels
        vector[uint64_t] channels
    cdef cppclass ActionPolicy:
        vector[uint64_t] request, observation, coordinates, selection, law
        vector[vector[uint64_t]] selected, fields, rules, program_rules
        vector[string] identities, languages, descriptions, frontier
        unsigned purpose
    cdef cppclass ActionEvidenceBridge:
        @staticmethod
        shared_ptr[ActionEvidenceBridge] decode(const string&,const ActionPolicy&) except +
        string archive() except +
    cdef cppclass ActionLeaseBridge:
        ActionLeaseBridge(const vector[size_t]&) except +
        void cancel() except +
    cdef cppclass TemporalTensorData:
        vector[size_t] shape, support, indices
        vector[string] values
        size_t support_count
        cppbool full_support
    cdef cppclass ActionFieldData:
        string label, archive
        TemporalTensorData tensor
        vector[uint64_t] carrier
        vector[vector[uint64_t]] dependencies
    cdef cppclass ActionSnapshot:
        cppbool moments, has_reference, matches_reference
        vector[ActionFieldData] fields
        vector[vector[string]] components, interactions
        vector[string] totals
        vector[vector[uint64_t]] admitted
        size_t steps, retained_entries
    cdef cppclass ActionResultBridge:
        ActionSnapshot inspect(size_t) except +
    cdef cppclass ActionProgramBridge:
        @staticmethod
        shared_ptr[ActionProgramBridge] prepare(const string&,const string&,const ActionRecipe&,uint64_t,
            const vector[vector[uint64_t]]&,const vector[size_t]&) except +
        @staticmethod
        string description(const ActionRecipe&) except +
        shared_ptr[ActionResultBridge] execute(uint64_t,const shared_ptr[ActionEvidenceBridge]&,
            const vector[vector[uint64_t]]&,const vector[vector[uint64_t]]&,const shared_ptr[ActionLeaseBridge]&) except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_ACTION_PROGRAM_BINDING_DIGEST
    const char* RFC_TEMPORAL_SDK_DIGEST

ACTION_LIMITS=(256,256,65536,4096,256,65536,1048576,4096,16000000,64000000,4096)
LEASE_LIMITS=(1048576,16000000,16000000,64000000,4096)
_OPCODES={'tensor':1,'identity':2,'scale':3,'sum':4,'then':5}
_PURPOSES={'prediction':0,'training':1,'evaluation':2}


def action_program_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
        'engine_headers':RFC_ENGINE_DIGEST.decode(),'extension_sdk':RFC_TEMPORAL_SDK_DIGEST.decode(),
        'binding':RFC_ACTION_PROGRAM_BINDING_DIGEST.decode(),'profile':'owned-native-action-program-v1',
        'arithmetic':'gmp-rational-exact','evidence_scope':'current-request-and-native-availability'}


cdef string _text(value):
    if type(value) is not str:raise TypeError('action text requires str')
    raw=value.encode('utf-8')
    if len(raw)>65536:raise ValueError('action text admission exceeded')
    return raw


cdef bytes _bytes(const string& value):
    return value


cdef string _image(value):
    if type(value) is not bytes:raise TypeError('immutable native archive bytes required')
    if len(value)>64<<20:raise ValueError('native archive admission exceeded')
    return value


cdef vector[uint64_t] _row(value, size_t count):
    if len(value)!=count:raise ValueError('action identity row arity differs')
    cdef vector[uint64_t] out
    for item in value:out.push_back(_natural(item))
    return out


cdef vector[vector[uint64_t]] _rows(value, size_t width, size_t maximum=65536):
    cdef vector[vector[uint64_t]] out
    for row in value:
        if out.size()>=maximum:raise ValueError('action identity rows exceed admission')
        out.push_back(_row(row,width))
    return out


cdef vector[size_t] _limits(value, size_t count):
    if len(value)!=count:raise ValueError('action limit count differs')
    cdef vector[size_t] out
    for item in value:out.push_back(_natural(item))
    return out


cdef ActionRecipe _recipe(spec):
    _closed(spec,('version','namespace','sequence','identity','meaning','nodes','output','result','metric','channels','reference'))
    cdef ActionRecipe r
    cdef ActionNode n
    r.version=_natural(spec['version']);r.name_space=_natural(spec['namespace']);r.sequence=_natural(spec['sequence'])
    r.identity=_text(spec['identity']);r.meaning=_text(spec['meaning'])
    if spec['output'] not in ('field','moments'):raise ValueError('field or moments output required')
    r.output=1 if spec['output']=='field' else 2;r.result=_natural(spec['result']);r.metric=_natural(spec['metric'])
    r.has_reference=spec['reference'] is not None
    if r.has_reference:r.reference=_natural(spec['reference'])
    for node in spec['nodes']:
        if r.nodes.size()>=4096:raise ValueError('action node declaration admission exceeded')
        _closed(node,('id','opcode','selector','operands','scale'))
        if node['opcode'] not in _OPCODES:raise ValueError('unsupported native action opcode')
        n.id=_natural(node['id']);n.opcode=_OPCODES[node['opcode']];n.selector=_natural(node['selector'])
        if len(node['operands'])>2:raise ValueError('action operand arity admission exceeded')
        n.operands=_row(node['operands'],len(node['operands']));n.scale=_scalar(node['scale'],4096);r.nodes.push_back(n)
    for channel in spec['channels']:
        if r.channels.size()>=4096:raise ValueError('action channel declaration admission exceeded')
        _closed(channel,('label','action'));r.labels.push_back(_text(channel['label']));r.channels.push_back(_natural(channel['action']))
    return r


cdef ActionPolicy _policy(spec):
    _closed(spec,('request','observation','frontier','purpose','selected','fields','coordinates','selection','rules','law','programs'))
    cdef ActionPolicy p
    p.request=_row(spec['request'],3);p.observation=_row(spec['observation'],3)
    if len(spec['frontier'])!=3:raise ValueError('event, known and validity frontier required')
    for value in spec['frontier']:p.frontier.push_back(_scalar(value,4096))
    if spec['purpose'] not in _PURPOSES:raise ValueError('declared evidence purpose required')
    p.purpose=_PURPOSES[spec['purpose']]
    p.selected=_rows(spec['selected'],4);p.fields=_rows(spec['fields'],4);p.rules=_rows(spec['rules'],4)
    p.coordinates=_row(spec['coordinates'],4);p.selection=_row(spec['selection'],4)
    if spec['law'] is not None:p.law=_row(spec['law'],4)
    for program in spec['programs']:
        if p.identities.size()>=4096:raise ValueError('action producer declaration admission exceeded')
        _closed(program,('identity','language','source','rule'))
        p.identities.push_back(_text(program['identity']));p.languages.push_back(_text(program['language']))
        p.descriptions.push_back(_text(program['source']));p.program_rules.push_back(_row(program['rule'],4))
    return p


def action_description(recipe):
    cdef ActionRecipe r=_recipe(recipe)
    cdef string description
    with nogil:description=ActionProgramBridge.description(r)
    return (<bytes>description).decode('utf-8')


cdef class NativeActionEvidence:
    cdef shared_ptr[ActionEvidenceBridge] _handle
    cdef shared_ptr[ActionEvidenceBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('action evidence owner is not initialized')
        return self._handle

    @classmethod
    def from_envelope(cls, archive, specification):
        cdef string data=_image(archive)
        cdef ActionPolicy p=_policy(specification)
        cdef NativeActionEvidence out=cls.__new__(cls)
        with nogil:out._handle=ActionEvidenceBridge.decode(data,p)
        return out

    def archive(self):return _bytes(self._pin().get().archive())


cdef class NativeActionLease:
    cdef shared_ptr[ActionLeaseBridge] _handle
    cdef shared_ptr[ActionLeaseBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('action lease is not initialized')
        return self._handle

    @classmethod
    def create(cls, *, limits=LEASE_LIMITS):
        cdef vector[size_t] bounds=_limits(limits,5)
        cdef NativeActionLease out=cls.__new__(cls)
        with nogil:out._handle=shared_ptr[ActionLeaseBridge](new ActionLeaseBridge(bounds))
        return out

    def cancel(self):
        cdef shared_ptr[ActionLeaseBridge] owner=self._pin()
        with nogil:owner.get().cancel()


cdef class NativeActionResult:
    cdef shared_ptr[ActionResultBridge] _handle

    def inspect(self, *, max_entries=65536):
        if self._handle.get()==NULL:raise RuntimeError('action result is not initialized')
        cdef size_t maximum=_natural(max_entries)
        cdef ActionSnapshot r
        with nogil:r=self._handle.get().inspect(maximum)
        return {'output':'moments' if r.moments else 'field',
            'fields':tuple({'label':(<bytes>f.label).decode(),'archive':<bytes>f.archive,'carrier':tuple(f.carrier),
                'dependencies':tuple(tuple(v) for v in f.dependencies),'shape':tuple(f.tensor.shape),
                'support_count':f.tensor.support_count,'full_support':bool(f.tensor.full_support),
                'entries':tuple((index,_rational(f.tensor.values[i])) for i,index in enumerate(f.tensor.indices))} for f in r.fields),
            'components':tuple({'label':_bytes(c[0]).decode(),'self':_rational(c[1]),'with_previous':_rational(c[2]),
                'increment':_rational(c[3]),'prefix_quadrance':_rational(c[4])} for c in r.components),
            'interactions':tuple({'left':int(_bytes(p[0])),'right':int(_bytes(p[1])),'pairing':_rational(p[2]),'doubled':_rational(p[3])} for p in r.interactions),
            'totals':tuple(_rational(v) for v in r.totals),'matches_reference':bool(r.matches_reference) if r.has_reference else None,
            'admitted_dependencies':tuple(tuple(v) for v in r.admitted),
            'steps':r.steps if r.moments else None,'retained_sparse_entries':r.retained_entries if r.moments else None}


cdef class NativeActionProgram:
    cdef shared_ptr[ActionProgramBridge] _handle
    cdef shared_ptr[ActionProgramBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('action program owner is not initialized')
        return self._handle

    @classmethod
    def prepare(cls, source, endpoint, recipe, publication, availability, *, limits=ACTION_LIMITS):
        cdef string s=_image(source),e=_image(endpoint)
        cdef ActionRecipe r=_recipe(recipe)
        cdef uint64_t field=_natural(publication)
        cdef vector[vector[uint64_t]] available=_rows(availability,2,4096)
        cdef vector[size_t] bounds=_limits(limits,11)
        cdef NativeActionProgram out=cls.__new__(cls)
        with nogil:out._handle=ActionProgramBridge.prepare(s,e,r,field,available,bounds)
        return out

    def execute(self, NativeActionEvidence evidence not None, availability, input_selector, *, pairs=(), NativeActionLease lease=None):
        cdef shared_ptr[ActionProgramBridge] owner=self._pin()
        cdef shared_ptr[ActionEvidenceBridge] current=evidence._pin()
        cdef vector[vector[uint64_t]] available=_rows(availability,2,4096),selected=_rows(pairs,2)
        cdef uint64_t input_id=_natural(input_selector)
        if lease is None:lease=NativeActionLease.create()
        cdef shared_ptr[ActionLeaseBridge] execution=lease._pin()
        cdef NativeActionResult out=NativeActionResult.__new__(NativeActionResult)
        with nogil:out._handle=owner.get().execute(input_id,current,available,selected,execution)
        return out

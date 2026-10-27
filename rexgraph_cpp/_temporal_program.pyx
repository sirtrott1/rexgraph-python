# distutils: language=c++
"""Owning immutable snapshots of the native relational function runtime."""
from fractions import Fraction
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as cppbool
from libc.stdint cimport uint64_t
from libc.stddef cimport size_t
from ._stalk_fields import _natural, _scalar
from ._field_program import _closed

cdef extern from "temporal_program_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass TemporalDomain:
        cppbool has_lower, has_upper, lower_closed, upper_closed
        string lower, upper
    cdef cppclass TemporalPiece:
        uint64_t identity
        TemporalDomain domain
        vector[string] numerator, denominator
        string provenance
    cdef cppclass TemporalFunction:
        uint64_t identity, coordinate
        string name, provenance
        vector[TemporalPiece] pieces
    cdef cppclass TemporalTransition:
        uint64_t left, right
        cppbool same_program, has_pairs
        vector[vector[uint64_t]] pairs, participation
    cdef cppclass TemporalRequest:
        string operation
        vector[uint64_t] ids
        vector[vector[uint64_t]] rows
        vector[string] texts, scalars
        vector[TemporalDomain] domains
        TemporalFunction function
        vector[TemporalTransition] transitions
    cdef cppclass TemporalFit:
        uint64_t family, context, function_set
        cppbool complete, forced
        size_t examined
        vector[uint64_t] programs
        vector[vector[vector[uint64_t]]] inputs
    cdef cppclass TemporalEvolution:
        cppbool complete, forced
        unsigned status
        size_t steps
        vector[uint64_t] contexts
        vector[vector[size_t]] possible
        vector[vector[vector[uint64_t]]] witnesses
    cdef cppclass TemporalSelection:
        uint64_t premise, evolution, context, local_family, program
        vector[vector[uint64_t]] inputs
    cdef cppclass TemporalInfo:
        uint64_t name_space, sequence, selected
        vector[vector[uint64_t]] objects, applications, ports, references
        vector[uint64_t] verified_layers, structural_interfaces
        vector[string] labels, application_labels
    cdef cppclass TemporalProgramBridge:
        @staticmethod
        shared_ptr[TemporalProgramBridge] create(uint64_t, const vector[size_t]&) except +
        @staticmethod
        shared_ptr[TemporalProgramBridge] decode(const string&, const string&, uint64_t, const vector[size_t]&) except +
        void validate() except +
        vector[string] encode() except +
        cppbool same_owner(const TemporalProgramBridge&) except +
        cppbool extends(const TemporalProgramBridge&) except +
        uint64_t selected() except +
        vector[uint64_t] identity() except +
        TemporalInfo inspect(size_t) except +
        TemporalFunction function(uint64_t) except +
        string evaluate(uint64_t, const string&) except +
        string evaluate_tensor(uint64_t, const string&) except +
        vector[TemporalDomain] support(uint64_t) except +
        vector[vector[uint64_t]] memberships(uint64_t) except +
        TemporalRequest program(uint64_t) except +
        TemporalFit fit(uint64_t) except +
        TemporalEvolution evolution(uint64_t) except +
        TemporalSelection selection(uint64_t) except +
        TemporalRequest layer(uint64_t) except +
        shared_ptr[TemporalProgramBridge] extend(const TemporalRequest&) except +
    cdef cppclass TemporalTensorData:
        vector[size_t] shape, support, indices
        vector[string] values
        size_t support_count
        cppbool full_support
    cdef cppclass TemporalWholeFieldBridge:
        TemporalTensorData inspect(size_t) except +
        cppbool same_owner(const TemporalWholeFieldBridge&) except +
        string quadrance() except +
    cdef cppclass TemporalFieldFunctionBridge:
        @staticmethod
        shared_ptr[TemporalFieldFunctionBridge] decode(const string&,uint64_t,cppbool) except +
        string archive() except +
        vector[vector[uint64_t]] ports() except +
        TemporalTensorData coefficients(size_t) except +
        vector[size_t] powers() except +
        shared_ptr[TemporalWholeFieldBridge] evaluate(const string&) except +
        cppbool same_owner(const TemporalFieldFunctionBridge&) except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_TEMPORAL_PROGRAM_BINDING_DIGEST
    const char* RFC_TEMPORAL_SDK_DIGEST

DEFAULT_LIMITS=(65536,128,256,1000000,65536,4194304,16777216)
_OPCODES={'input':1,'constant':2,'derivative':3,'pullback':4,'add':5,
          'subtract':6,'multiply':7,'divide':8,'restrict':9}
_KINDS=('function','tensor','program','support','temporal_relation','probability',
        'scalar','family','context','coordinate','reference','function_set','interface')


def temporal_program_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(), 'extension_sdk':RFC_TEMPORAL_SDK_DIGEST.decode(),
            'binding':RFC_TEMPORAL_PROGRAM_BINDING_DIGEST.decode(),
            'profile':'owned-relational-function-runtime-v1','arithmetic':'gmp-rational-exact',
            'evidence_scope':'declared-inputs'}


cdef string _text(value):
    if not isinstance(value,str): raise TypeError('runtime text requires str')
    raw=value.encode('utf-8')
    if len(raw)>65536: raise ValueError('runtime text exceeds admission')
    return raw


cdef bytes _raw(const string& value):
    return value


cdef vector[uint64_t] _ids(value, size_t maximum=65536):
    cdef vector[uint64_t] out
    for x in value:
        if out.size()>=maximum:raise ValueError('runtime identity count exceeds admission')
        out.push_back(_natural(x))
    return out


cdef vector[size_t] _limits(value):
    if len(value)!=7: raise ValueError('seven temporal learning bounds required')
    cdef vector[size_t] out
    for v in value: out.push_back(_natural(v))
    return out


cdef TemporalDomain _domain(value):
    if len(value)!=4:raise ValueError('domain requires lower, upper and both closed flags')
    if type(value[2]) is not bool or type(value[3]) is not bool:raise TypeError('domain endpoint flags require bool')
    cdef TemporalDomain out
    out.has_lower=value[0] is not None;out.has_upper=value[1] is not None
    out.lower_closed=value[2];out.upper_closed=value[3]
    if out.has_lower:out.lower=_scalar(value[0],16384)
    if out.has_upper:out.upper=_scalar(value[1],16384)
    return out


def _rational(word):
    parts=(<bytes>word).decode('ascii').split('/')
    return Fraction(int(parts[0],16),int(parts[1],16) if len(parts)==2 else 1)


cdef object _domain_info(TemporalDomain d):
    return (_rational(d.lower) if d.has_lower else None,
            _rational(d.upper) if d.has_upper else None,bool(d.lower_closed),bool(d.upper_closed))


cdef object _tensor_info(TemporalTensorData t):
    return {'shape':tuple(t.shape),'full_support':bool(t.full_support),'support_count':t.support_count,
            'support':None if t.full_support else tuple(t.support),
            'entries':tuple((i,_rational(t.values[n])) for n,i in enumerate(t.indices))}


cdef TemporalRequest _request(operation,spec):
    cdef TemporalRequest q
    cdef TemporalPiece piece
    cdef TemporalTransition transition
    cdef vector[uint64_t] row
    q.operation=_text(operation)
    if operation=='coordinate':
        _closed(spec,('name','clock','unit','origin'))
        for k in ('name','clock','unit','origin'):q.texts.push_back(_text(spec[k]))
    elif operation=='function':
        _closed(spec,('coordinate','name','provenance','pieces'))
        q.function.coordinate=_natural(spec['coordinate']);q.function.name=_text(spec['name']);q.function.provenance=_text(spec['provenance'])
        for p in spec['pieces']:
            if q.function.pieces.size()>=4096:raise ValueError('function piece count exceeds admission')
            _closed(p,('identity','domain','numerator','denominator','provenance'))
            piece.identity=_natural(p['identity']);piece.domain=_domain(p['domain']);piece.provenance=_text(p['provenance'])
            piece.numerator.clear();piece.denominator.clear()
            for v in p['numerator']:
                if piece.numerator.size()>=4096:raise ValueError('numerator coefficient count exceeds admission')
                piece.numerator.push_back(_scalar(v,16384))
            for v in p['denominator']:
                if piece.denominator.size()>=4096:raise ValueError('denominator coefficient count exceeds admission')
                piece.denominator.push_back(_scalar(v,16384))
            q.function.pieces.push_back(piece)
    elif operation=='program':
        _closed(spec,('opcode','role','constant','operands','scale','shift','domain','coordinate','meaning'))
        if spec['opcode'] not in _OPCODES:raise ValueError('unsupported function program opcode')
        q.ids=_ids((_OPCODES[spec['opcode']],spec['role'],spec['constant'],spec['coordinate']))
        q.rows.push_back(_ids(spec['operands'],64));q.scalars.push_back(_scalar(spec['scale'],16384));q.scalars.push_back(_scalar(spec['shift'],16384));q.texts.push_back(_text(spec['meaning']))
        if spec['domain'] is not None:q.domains.push_back(_domain(spec['domain']))
    elif operation=='support':
        _closed(spec,('coordinate','domains','label'));q.ids=_ids((spec['coordinate'],));q.texts.push_back(_text(spec['label']))
        for d in spec['domains']:
            if q.domains.size()>=65536:raise ValueError('support component count exceeds admission')
            q.domains.push_back(_domain(d))
    elif operation=='context':
        _closed(spec,('support','meaning'));q.ids=_ids((spec['support'],));q.texts.push_back(_text(spec['meaning']))
    elif operation=='function_set':
        _closed(spec,('coordinate','members','label'));q.ids=_ids((spec['coordinate'],));q.texts.push_back(_text(spec['label']))
        for m in spec['members']:
            if len(m)!=5 or type(m[4]) is not bool:raise TypeError('membership requires identity,role,function,support,exists')
            if q.rows.size()>=65536:raise ValueError('membership count exceeds admission')
            q.rows.push_back(_ids((*m[:4],int(m[4]))))
    elif operation=='select_evolution':
        _closed(spec,('family','context'));q.ids=_ids((spec['family'],spec['context']))
    elif operation in ('apply','apply_learned','apply_evolution'):
        _closed(spec,('program' if operation=='apply' else 'family','bindings','label')+ (('context',) if operation=='apply_evolution' else ()))
        q.ids=_ids((spec['program' if operation=='apply' else 'family'],))
        if operation=='apply_evolution':q.ids.push_back(_natural(spec['context']))
        q.texts.push_back(_text(spec['label']))
        for p in spec['bindings']:
            if len(p)!=2:raise ValueError('function binding requires role and actual function identity')
            if q.rows.size()>=64:raise ValueError('input role count exceeds admission')
            q.rows.push_back(_ids(p))
    elif operation=='fit':
        _closed(spec,('function_set','context','candidates','expected','observed_domain'))
        q.ids=_ids((spec['function_set'],spec['context'],spec['expected']));q.rows.push_back(_ids(spec['candidates']))
        if spec['observed_domain'] is not None:q.domains.push_back(_domain(spec['observed_domain']))
    elif operation=='join':
        _closed(spec,('families','transitions'));q.ids=_ids(spec['families'],256)
        for t in spec['transitions']:
            if q.transitions.size()>=65536:raise ValueError('transition count exceeds admission')
            _closed(t,('left','right','same_program','allowed_pairs','participation'))
            if type(t['same_program']) is not bool:raise TypeError('same_program requires bool')
            transition.left=_natural(t['left']);transition.right=_natural(t['right']);transition.same_program=t['same_program']
            transition.has_pairs=t['allowed_pairs'] is not None;transition.pairs.clear();transition.participation.clear()
            if transition.has_pairs:
                for p in t['allowed_pairs']:
                    if len(p)!=2:raise ValueError('allowed program pair requires two identities')
                    if transition.pairs.size()>=1000000:raise ValueError('transition pair count exceeds admission')
                    transition.pairs.push_back(_ids(p))
            for p in t['participation']:
                if len(p)!=3 or p[2] not in (1,2,3):raise ValueError('participant coupling requires roles and a declared rule')
                if transition.participation.size()>=64:raise ValueError('participant coupling count exceeds admission')
                transition.participation.push_back(_ids(p))
            q.transitions.push_back(transition)
    elif operation=='glue':
        _closed(spec,('sections','label'));q.texts.push_back(_text(spec['label']))
        for p in spec['sections']:
            if len(p)!=4:raise ValueError('local section requires function,support,chart scale and shift')
            if q.rows.size()>=65536:raise ValueError('section count exceeds admission')
            q.rows.push_back(_ids(p[:2]));q.scalars.push_back(_scalar(p[2],16384));q.scalars.push_back(_scalar(p[3],16384))
    elif operation=='tensor':
        _closed(spec,('function','piece','label'));q.ids=_ids((spec['function'],spec['piece']));q.texts.push_back(_text(spec['label']))
    elif operation=='interval':
        _closed(spec,('start','end','event_coordinate','domain','label'));q.ids=_ids((spec['start'],spec['end'],spec['event_coordinate']));q.domains.push_back(_domain(spec['domain']));q.texts.push_back(_text(spec['label']))
    elif operation in ('allen','support_union','support_intersection','support_difference'):
        _closed(spec,('left','right','label'));q.ids=_ids((spec['left'],spec['right']));q.texts.push_back(_text(spec['label']))
    elif operation=='atom_support':
        _closed(spec,('relation','atoms','label'));q.ids=_ids((spec['relation'],));q.rows.push_back(_ids(spec['atoms'],13));q.texts.push_back(_text(spec['label']))
    else:raise ValueError('unsupported relational runtime extension')
    return q


cdef class NativeRelationalRuntime:
    cdef shared_ptr[TemporalProgramBridge] _handle
    cdef vector[size_t] _bounds

    cdef shared_ptr[TemporalProgramBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('relational runtime owner is not initialized')
        return self._handle

    @classmethod
    def create(cls, namespace, *, limits=DEFAULT_LIMITS):
        cdef uint64_t ns=_natural(namespace)
        cdef vector[size_t] bounds=_limits(limits)
        cdef NativeRelationalRuntime out=cls.__new__(cls)
        with nogil:out._handle=TemporalProgramBridge.create(ns,bounds)
        out._bounds=bounds
        return out

    @classmethod
    def decode(cls, source, endpoint, *, selected=0, limits=DEFAULT_LIMITS):
        if type(source) is not bytes or type(endpoint) is not bytes:raise TypeError('canonical source and endpoint bytes required')
        if len(source)>64*1024*1024 or len(endpoint)>64*1024*1024:raise ValueError('runtime archive exceeds admission')
        cdef string s=source,e=endpoint
        cdef uint64_t selection=_natural(selected)
        cdef vector[size_t] bounds=_limits(limits)
        cdef NativeRelationalRuntime out=cls.__new__(cls)
        with nogil:out._handle=TemporalProgramBridge.decode(s,e,selection,bounds)
        out._bounds=bounds
        return out

    @property
    def selected(self):return self._pin().get().selected()

    @property
    def limits(self):return tuple(self._bounds)

    def identity(self):return tuple(self._pin().get().identity())

    def extend(self, operation, specification):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef TemporalRequest q=_request(operation,specification)
        cdef NativeRelationalRuntime out=NativeRelationalRuntime.__new__(NativeRelationalRuntime)
        with nogil:out._handle=owner.get().extend(q)
        out._bounds=self._bounds
        return out

    def encode(self):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef vector[string] raw
        with nogil:raw=owner.get().encode()
        return (_raw(raw[0]),_raw(raw[1]))

    def validate(self):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        with nogil:owner.get().validate()

    def same_owner(self, NativeRelationalRuntime other not None):
        return bool(self._pin().get().same_owner(other._pin().get()[0]))

    def extends(self, NativeRelationalRuntime parent not None):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin(),before=parent._pin()
        cdef cppbool good
        with nogil:good=owner.get().extends(before.get()[0])
        return bool(good)

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef TemporalInfo info
        with nogil:info=owner.get().inspect(maximum)
        return {'namespace':info.name_space,'sequence':info.sequence,'selected':info.selected,
            'objects':tuple({'id':o[0],'kind':_KINDS[o[1]-1],'coordinate':o[2],'context':o[3],
                'producer':o[4],'status':o[5],'references':tuple(info.references[i]),'label':_raw(info.labels[i]).decode()} for i,o in enumerate(info.objects)),
            'applications':tuple({'id':a[0],'operation':a[1],'operation_occurrence':a[2],
                'support':a[3],'support_occurrence':a[4],'executable':bool(a[5]),'label':_raw(info.application_labels[i]).decode()} for i,a in enumerate(info.applications)),
            'ports':tuple(tuple(p) for p in info.ports),'evidence_scope':'declared-inputs',
            'verified_layer_diagrams':tuple(info.verified_layers),
            'structural_only_interfaces':tuple(info.structural_interfaces)}

    def function(self, identity):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef uint64_t key=_natural(identity)
        cdef TemporalFunction f
        with nogil:f=owner.get().function(key)
        return {'identity':f.identity,'coordinate':f.coordinate,'name':(<bytes>f.name).decode(),
            'provenance':(<bytes>f.provenance).decode(),'pieces':tuple({'identity':p.identity,
                'domain':_domain_info(p.domain),'numerator':tuple(_rational(v) for v in p.numerator),
                'denominator':tuple(_rational(v) for v in p.denominator),'provenance':(<bytes>p.provenance).decode()} for p in f.pieces)}

    def evaluate(self, identity, parameter, *, tensor=False):
        if type(tensor) is not bool:raise TypeError('tensor selector requires bool')
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef uint64_t key=_natural(identity)
        cdef string value=_scalar(parameter,16384),result
        if tensor:
            with nogil:result=owner.get().evaluate_tensor(key,value)
        else:
            with nogil:result=owner.get().evaluate(key,value)
        return _rational(result)

    def fit(self, identity):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef uint64_t key=_natural(identity)
        cdef TemporalFit f
        with nogil:f=owner.get().fit(key)
        return {'family':f.family,'context':f.context,'function_set':f.function_set,'complete':bool(f.complete),
            'forced':bool(f.forced),'examined':f.examined,'matches':tuple({'program':p,
                'inputs':tuple(tuple(v) for v in f.inputs[i])} for i,p in enumerate(f.programs))}

    def support(self, identity):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef uint64_t key=_natural(identity)
        cdef vector[TemporalDomain] domains
        with nogil:domains=owner.get().support(key)
        return tuple(_domain_info(d) for d in domains)

    def selection(self, identity):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef uint64_t key=_natural(identity)
        cdef TemporalSelection s
        with nogil:s=owner.get().selection(key)
        return {'premise':s.premise,'evolution':s.evolution,'context':s.context,'local_family':s.local_family,
                'match':{'program':s.program,'inputs':tuple(tuple(v) for v in s.inputs)}}

    def layer(self, identity):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef uint64_t key=_natural(identity)
        cdef TemporalRequest q
        with nogil:q=owner.get().layer(key)
        return {'diagram':key,'profile':('functions','families')[q.ids[0]-1],'output':q.ids[1],
            'previous':q.ids[2],'compatible':bool(q.ids[3]),'name':_raw(q.texts[0]).decode(),
            'status':_raw(q.texts[1]).decode(),'verified':True,
            'layers':tuple({'identity':r[0],'object':r[1],'support':r[2],
                'chart_scale':_rational(q.scalars[2*i]),'chart_shift':_rational(q.scalars[2*i+1]),
                'role':_raw(q.texts[2+i]).decode()} for i,r in enumerate(q.rows))}

    def memberships(self, identity):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef uint64_t key=_natural(identity)
        cdef vector[vector[uint64_t]] rows
        with nogil:rows=owner.get().memberships(key)
        return tuple((r[0],r[1],r[2],r[3],bool(r[4])) for r in rows)

    def program(self, identity):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef uint64_t key=_natural(identity)
        cdef TemporalRequest p
        with nogil:p=owner.get().program(key)
        names=('input','constant','derivative','pullback','add','subtract','multiply','divide','restrict',
               'fit_family','join_families','glue_sections','tensor_view','select_evolution')
        return {'opcode':names[p.ids[0]-1],'role':p.ids[1],'constant':p.ids[2],'coordinate':p.ids[3],
            'operands':tuple(p.rows[0]),'scale':_rational(p.scalars[0]),'shift':_rational(p.scalars[1]),
            'domain':_domain_info(p.domains[0]) if p.domains.size() else None,'meaning':_raw(p.texts[0]).decode()}

    def evolution(self, identity):
        cdef shared_ptr[TemporalProgramBridge] owner=self._pin()
        cdef uint64_t key=_natural(identity)
        cdef TemporalEvolution e
        with nogil:e=owner.get().evolution(key)
        return {'complete':bool(e.complete),'forced':bool(e.forced),'status':('satisfiable','unsatisfiable','budget_exceeded','cancelled')[e.status],
            'search_steps':e.steps,'possible_matches':{c:tuple(e.possible[i]) for i,c in enumerate(e.contexts)},
            'witnesses':tuple(tuple(tuple(p) for p in w) for w in e.witnesses),
            'witness_scope':'one complete witness per feasible projection; retained factors define the joint family'}


cdef class NativeRelationalWholeField:
    cdef shared_ptr[TemporalWholeFieldBridge] _handle

    cdef shared_ptr[TemporalWholeFieldBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('whole field owner is not initialized')
        return self._handle

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[TemporalWholeFieldBridge] owner=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef TemporalTensorData t
        with nogil:t=owner.get().inspect(maximum)
        return _tensor_info(t)

    def quadrance(self):
        cdef shared_ptr[TemporalWholeFieldBridge] owner=self._pin()
        cdef string q
        with nogil:q=owner.get().quadrance()
        return _rational(q)

    def same_owner(self, NativeRelationalWholeField other not None):
        return bool(self._pin().get().same_owner(other._pin().get()[0]))


cdef class NativeRelationalFieldFunction:
    cdef shared_ptr[TemporalFieldFunctionBridge] _handle

    cdef shared_ptr[TemporalFieldFunctionBridge] _pin(self) except *:
        if self._handle.get()==NULL:raise RuntimeError('field function owner is not initialized')
        return self._handle

    @classmethod
    def from_envelope(cls, archive, record, *, polynomial=True):
        if type(archive) is not bytes:raise TypeError('native model envelope bytes required')
        if len(archive)>32<<20:raise ValueError('field function envelope exceeds admission')
        if type(polynomial) is not bool:raise TypeError('polynomial selector requires bool')
        cdef string raw=archive
        cdef uint64_t key=_natural(record)
        cdef cppbool poly=polynomial
        cdef NativeRelationalFieldFunction out=cls.__new__(cls)
        with nogil:out._handle=TemporalFieldFunctionBridge.decode(raw,key,poly)
        return out

    def archive(self):
        cdef shared_ptr[TemporalFieldFunctionBridge] owner=self._pin()
        cdef string raw
        with nogil:raw=owner.get().archive()
        return _raw(raw)

    def ports(self):
        cdef shared_ptr[TemporalFieldFunctionBridge] owner=self._pin()
        cdef vector[vector[uint64_t]] rows
        with nogil:rows=owner.get().ports()
        return tuple({'port':tuple(r[:3]),'grade':r[3],'source_cell_key':r[4],
                      'offset':r[5],'dimension':r[6],'coordinates':tuple(r[7:])} for r in rows)

    def coefficients(self, *, max_records=4096):
        cdef shared_ptr[TemporalFieldFunctionBridge] owner=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef TemporalTensorData t
        cdef vector[size_t] powers
        with nogil:
            t=owner.get().coefficients(maximum)
            powers=owner.get().powers()
        return {'tensor':_tensor_info(t),'power_axis':tuple(powers)}

    def evaluate(self, parameter):
        cdef shared_ptr[TemporalFieldFunctionBridge] owner=self._pin()
        cdef string q=_scalar(parameter,16384)
        cdef NativeRelationalWholeField out=NativeRelationalWholeField.__new__(NativeRelationalWholeField)
        with nogil:out._handle=owner.get().evaluate(q)
        return out

    def same_owner(self, NativeRelationalFieldFunction other not None):
        return bool(self._pin().get().same_owner(other._pin().get()[0]))

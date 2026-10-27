# cython: language_level=3
"""Frozen native temporal sources; mutations always return a new owner."""
from fractions import Fraction
from numbers import Integral
from collections.abc import Mapping
from libc.stdint cimport uint64_t
from libc.stddef cimport size_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "temporal_bridge.hpp" namespace "rexgraph::temporal_python" nogil:
    cdef cppclass Domain:
        string lower, upper
        bint lower_closed, upper_closed
    cdef cppclass Piece:
        Piece() except +
        uint64_t identity
        Domain domain
        vector[string] numerator, denominator
        string provenance
    cdef cppclass Function:
        uint64_t identity, coordinate
        string name, provenance
        vector[Piece] pieces
    cdef cppclass Node:
        uint64_t opcode, role, constant, coordinate
        vector[uint64_t] operands
        string scale, shift, meaning
        bint has_domain
        Domain domain
    cdef cppclass Member:
        Member() except +
        uint64_t identity, role, function, support
        bint exists
    cdef cppclass Coupling:
        Coupling() except +
        uint64_t left_role, right_role, rule
    cdef cppclass Transition:
        Transition() except +
        uint64_t left, right
        bint same, restricted
        vector[uint64_t] pairs
        vector[Coupling] couplings
    cdef cppclass Layer:
        Layer() except +
        uint64_t identity, object, support
        string scale, shift, role
    cdef cppclass Fit:
        uint64_t family, context, function_set
        bint complete, forced
        size_t examined
        vector[uint64_t] programs
        vector[vector[uint64_t]] bindings
    cdef cppclass Joint:
        uint64_t family
        bint complete, forced
        string status
        size_t steps
        vector[uint64_t] contexts
        vector[vector[uint64_t]] projections, witnesses
    cdef cppclass Product:
        uint64_t diagram, output
        bint complete, compatible
        string status
    cdef cppclass Object:
        uint64_t identity, kind, coordinate, producer
        string label
        vector[uint64_t] references
    cdef cppclass Temporal:
        @staticmethod
        shared_ptr[Temporal] create(uint64_t, const vector[size_t]&) except +
        @staticmethod
        shared_ptr[Temporal] load(const string&, const string&, const vector[size_t]&) except +
        shared_ptr[Temporal] fork() except +
        string source_bytes() except +
        string endpoint_bytes() except +
        uint64_t name_space() except +
        vector[Object] objects() except +
        uint64_t coordinate(const string&, const string&, const string&, const string&) except +
        uint64_t function(uint64_t, const vector[Piece]&, const string&) except +
        Function function_info(uint64_t) except +
        string value(uint64_t, const string&) except +
        bint equal(uint64_t, uint64_t) except +
        uint64_t program(const Node&) except +
        Node program_info(uint64_t) except +
        uint64_t apply(uint64_t, const vector[uint64_t]&, uint64_t) except +
        uint64_t support(uint64_t, const vector[Domain]&, const string&) except +
        vector[Domain] support_info(uint64_t) except +
        uint64_t context(uint64_t, const string&) except +
        uint64_t function_set(uint64_t, const vector[Member]&, const string&) except +
        uint64_t fit(uint64_t, uint64_t, const vector[uint64_t]&, uint64_t, bint, const Domain&) except +
        Fit fit_info(uint64_t) except +
        uint64_t joint(const vector[uint64_t]&, const vector[Transition]&) except +
        Joint joint_info(uint64_t) except +
        Product product(bint, const vector[Layer]&, const vector[Transition]&, const string&, uint64_t) except +
        Product product_info(uint64_t) except +
        uint64_t temporal(uint64_t, const vector[uint64_t]&, const vector[uint64_t]&, const vector[string]&, const Domain&, const vector[string]&, const string&) except +
        void verify() except +
    string build_digest() except +


def _uint(value, zero=False):
    if isinstance(value, bool) or not isinstance(value, Integral) or not (0 if zero else 1) <= value < 2**64:
        raise ValueError("native identities require unsigned integers")
    return int(value)


def _sequence(value, maximum=65536):
    if not isinstance(value, (tuple, list)) or len(value) > maximum:
        raise ValueError("bounded tuple/list declaration required")
    return value


cdef string _text(object value):
    if not isinstance(value, str) or len(value.encode('utf8')) > 65536:
        raise ValueError("bounded text required")
    return value.encode('utf8')


cdef string _rational(object value):
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction, str)):
        raise TypeError("exact integer or rational required; floating inputs are not admitted")
    if isinstance(value, str) and len(value) > 1048576:
        raise ValueError("rational text exceeds its bound")
    q = Fraction(value)
    return str(q).encode('ascii')


cdef vector[uint64_t] _ids(object values, bint zero=False):
    cdef vector[uint64_t] result
    for value in _sequence(values): result.push_back(_uint(value, zero))
    return result


cdef Domain _domain(object value):
    cdef Domain result
    if value is None: value = (None, None, False, False)
    value = _sequence(value, 4)
    if len(value) != 4 or type(value[2]) is not bool or type(value[3]) is not bool:
        raise ValueError("domain requires lower, upper, lower_closed and upper_closed")
    if value[0] is not None: result.lower = _rational(value[0])
    if value[1] is not None: result.upper = _rational(value[1])
    result.lower_closed = value[2]; result.upper_closed = value[3]
    return result


cdef object _domain_value(Domain value):
    return (Fraction((<bytes>value.lower).decode()) if value.lower.size() else None,
            Fraction((<bytes>value.upper).decode()) if value.upper.size() else None,
            bool(value.lower_closed), bool(value.upper_closed))


cdef vector[Transition] _transitions(object values):
    cdef vector[Transition] result
    cdef Transition item
    cdef Coupling coupling
    for value in _sequence(values, 4096):
        if not isinstance(value, Mapping) or set(value) - {'left', 'right', 'same_program', 'allowed_pairs', 'couplings'}:
            raise ValueError("typed transition declaration required")
        item = Transition()
        item.left = _uint(value['left']); item.right = _uint(value['right'])
        same = value.get('same_program', False)
        if type(same) is not bool: raise TypeError("same_program requires Boolean")
        item.same = same
        pairs = value.get('allowed_pairs')
        item.restricted = pairs is not None
        if pairs is not None:
            for pair in _sequence(pairs):
                if len(_sequence(pair, 2)) != 2: raise ValueError("ordered program pair required")
                item.pairs.push_back(_uint(pair[0])); item.pairs.push_back(_uint(pair[1]))
        for row in _sequence(value.get('couplings', ()), 4096):
            if len(_sequence(row, 3)) != 3: raise ValueError("participation coupling requires two roles and a rule")
            coupling = Coupling(); coupling.left_role = _uint(row[0]); coupling.right_role = _uint(row[1]); coupling.rule = _uint(row[2])
            item.couplings.push_back(coupling)
        result.push_back(item)
    return result


def build_info():
    cdef string value = build_digest()
    return {'profile': 'retained-temporal-v1', 'temporal_digest': (<bytes>value).decode()}


cdef class NativeTemporal:
    cdef shared_ptr[Temporal] _handle
    cdef object _work

    cdef Temporal* _get(self) except NULL nogil:
        if self._handle.get() == NULL:
            with gil:
                raise RuntimeError("uninitialized native temporal source")
        return self._handle.get()

    def __init__(self, source=None, endpoint=None, *, name_space=1, limits=()):
        if self._handle.get() != NULL: raise RuntimeError("native temporal source is immutable")
        self._work = tuple(_sequence(limits, 9))
        cdef vector[size_t] work
        cdef string a, b
        cdef uint64_t namespace_value
        for value in self._work: work.push_back(_uint(value, True))
        if source is None and endpoint is None:
            namespace_value = _uint(name_space)
            with nogil: self._handle = Temporal.create(namespace_value, work)
        else:
            if type(source) is not bytes or type(endpoint) is not bytes or max(len(source), len(endpoint)) > 64*1024*1024:
                raise ValueError("two bounded native archives required")
            a = source; b = endpoint
            with nogil: self._handle = Temporal.load(a, b, work)

    cdef NativeTemporal _fork(self):
        cdef NativeTemporal result = NativeTemporal.__new__(NativeTemporal)
        result._work = self._work
        with nogil: result._handle = self._get().fork()
        return result

    @property
    def source_bytes(self):
        cdef string value = self._get().source_bytes()
        return <bytes>value

    @property
    def endpoint_bytes(self):
        cdef string value = self._get().endpoint_bytes()
        return <bytes>value

    @property
    def name_space(self): return self._get().name_space()

    @property
    def limits(self): return self._work

    def objects(self):
        cdef vector[Object] values = self._get().objects()
        return tuple({'id': x.identity, 'kind': x.kind, 'coordinate': x.coordinate,
                      'producer': x.producer, 'name': (<bytes>x.label).decode(), 'references': tuple(x.references)} for x in values)

    def coordinate(self, name, clock, unit, origin):
        cdef NativeTemporal out = self._fork()
        cdef string n=_text(name), c=_text(clock), u=_text(unit), o=_text(origin)
        cdef uint64_t result
        with nogil: result=out._handle.get().coordinate(n,c,u,o)
        return out,result

    def function(self, coordinate, pieces, name='function'):
        cdef NativeTemporal out=self._fork()
        cdef vector[Piece] data
        cdef Piece piece
        cdef string n=_text(name)
        cdef uint64_t c=_uint(coordinate), result
        for index, value in enumerate(_sequence(pieces,4096)):
            if not isinstance(value,Mapping) or set(value)-{'identity','domain','numerator','denominator','provenance'}:
                raise ValueError("function piece declaration required")
            piece=Piece();piece.identity=_uint(value.get('identity',index+1));piece.domain=_domain(value.get('domain'))
            piece.provenance=_text(value.get('provenance','supplied piece'))
            for q in _sequence(value['numerator'],4096): piece.numerator.push_back(_rational(q))
            for q in _sequence(value.get('denominator',(1,)),4096): piece.denominator.push_back(_rational(q))
            data.push_back(piece)
        with nogil: result=out._handle.get().function(c,data,n)
        return out,result

    def function_info(self, identity):
        cdef Function value=self._get().function_info(_uint(identity))
        return {'id':value.identity,'coordinate':value.coordinate,'name':(<bytes>value.name).decode(),
                'provenance':(<bytes>value.provenance).decode(),
                'pieces':tuple({'identity':p.identity,'domain':_domain_value(p.domain),
                    'numerator':tuple(Fraction((<bytes>q).decode()) for q in p.numerator),
                    'denominator':tuple(Fraction((<bytes>q).decode()) for q in p.denominator),
                    'provenance':(<bytes>p.provenance).decode()} for p in value.pieces)}

    def value(self, identity, at):
        cdef uint64_t i=_uint(identity)
        cdef string t=_rational(at),value
        with nogil: value=self._get().value(i,t)
        return Fraction((<bytes>value).decode()) if value.size() else None

    def equal(self, left, right):
        cdef uint64_t a=_uint(left),b=_uint(right)
        cdef bint result
        with nogil: result=self._get().equal(a,b)
        return bool(result)

    def program(self, opcode, operands=(), *, role=1, constant=0, scale=1, shift=0, domain=None, coordinate=0, name='program'):
        cdef NativeTemporal out=self._fork()
        cdef Node node
        node.opcode=_uint(opcode);node.operands=_ids(operands);node.role=_uint(role);node.constant=_uint(constant,True)
        node.coordinate=_uint(coordinate,True);node.scale=_rational(scale);node.shift=_rational(shift);node.meaning=_text(name)
        node.has_domain=domain is not None;node.domain=_domain(domain)
        cdef uint64_t result
        with nogil: result=out._handle.get().program(node)
        return out,result

    def program_info(self, identity):
        cdef Node node=self._get().program_info(_uint(identity))
        return {'opcode':node.opcode,'operands':tuple(node.operands),'role':node.role,'constant':node.constant,
            'scale':Fraction((<bytes>node.scale).decode()),'shift':Fraction((<bytes>node.shift).decode()),
            'domain':_domain_value(node.domain) if node.has_domain else None,'coordinate':node.coordinate,'name':(<bytes>node.meaning).decode()}

    def apply(self, program, bindings, learned_family=0):
        if not isinstance(bindings,Mapping) or len(bindings)>64: raise ValueError("bounded role bindings required")
        cdef NativeTemporal out=self._fork()
        cdef vector[uint64_t] data
        for role,function in sorted(bindings.items()): data.push_back(_uint(role));data.push_back(_uint(function))
        cdef uint64_t p=_uint(program,True), f=_uint(learned_family,True),result
        with nogil: result=out._handle.get().apply(p,data,f)
        return out,result

    def support(self, coordinate, components, name='support'):
        cdef NativeTemporal out=self._fork()
        cdef vector[Domain] data
        for value in _sequence(components,4096): data.push_back(_domain(value))
        cdef uint64_t c=_uint(coordinate), result
        cdef string n=_text(name)
        with nogil: result=out._handle.get().support(c,data,n)
        return out,result

    def support_info(self, identity):
        cdef vector[Domain] values=self._get().support_info(_uint(identity))
        return tuple(_domain_value(value) for value in values)

    def context(self, support, name='context'):
        cdef NativeTemporal out=self._fork()
        cdef uint64_t s=_uint(support),result
        cdef string n=_text(name)
        with nogil: result=out._handle.get().context(s,n)
        return out,result

    def function_set(self, coordinate, members, name='function set'):
        cdef NativeTemporal out=self._fork()
        cdef vector[Member] data
        cdef Member item
        for row in _sequence(members):
            if len(_sequence(row,5))!=5 or type(row[4]) is not bool: raise ValueError("membership requires identity, role, function, support and existence")
            item=Member();item.identity=_uint(row[0]);item.role=_uint(row[1]);item.function=_uint(row[2]);item.support=_uint(row[3]);item.exists=row[4];data.push_back(item)
        cdef uint64_t c=_uint(coordinate),result
        cdef string n=_text(name)
        with nogil: result=out._handle.get().function_set(c,data,n)
        return out,result

    def fit(self, function_set, context, candidates, expected, observation_domain=None):
        cdef NativeTemporal out=self._fork()
        cdef uint64_t s=_uint(function_set),c=_uint(context),e=_uint(expected),result
        cdef vector[uint64_t] programs=_ids(candidates)
        cdef bint observed=observation_domain is not None
        cdef Domain d=_domain(observation_domain)
        with nogil: result=out._handle.get().fit(s,c,programs,e,observed,d)
        return out,result

    def fit_info(self, identity):
        cdef Fit value=self._get().fit_info(_uint(identity))
        return {'family':value.family,'context':value.context,'function_set':value.function_set,
            'complete':bool(value.complete),'forced':bool(value.forced),'examined':value.examined,
            'matches':tuple({'program':value.programs[i], 'inputs':tuple(tuple(row[j:j+3]) for j in range(0,len(row),3))}
                            for i,row in enumerate(value.bindings))}

    def joint(self, families, transitions=()):
        cdef NativeTemporal out=self._fork()
        cdef vector[uint64_t] values=_ids(families)
        cdef vector[Transition] conditions=_transitions(transitions)
        cdef uint64_t result
        with nogil: result=out._handle.get().joint(values,conditions)
        return out,result

    def joint_info(self, identity):
        cdef Joint value=self._get().joint_info(_uint(identity))
        return {'family':value.family,'complete':bool(value.complete),'forced':bool(value.forced),
            'status':(<bytes>value.status).decode(),'steps':value.steps,
            'projections':tuple((value.contexts[i],tuple(row)) for i,row in enumerate(value.projections)),
            'witnesses':tuple(tuple(tuple(row[j:j+2]) for j in range(0,len(row),2)) for row in value.witnesses),
            'witness_semantics':'one global witness per surviving projection; not exhaustive enumeration'}

    def product(self, functions, layers, transitions=(), name='layer product', previous=0):
        if type(functions) is not bool: raise TypeError("explicit function/family profile required")
        cdef NativeTemporal out=self._fork()
        cdef vector[Layer] values
        cdef Layer item
        for row in _sequence(layers,4096):
            if len(_sequence(row,6))!=6: raise ValueError("layer requires identity, object, support, scale, shift and role")
            item=Layer();item.identity=_uint(row[0]);item.object=_uint(row[1]);item.support=_uint(row[2],True)
            item.scale=_rational(row[3]);item.shift=_rational(row[4]);item.role=_text(row[5]);values.push_back(item)
        cdef vector[Transition] conditions=_transitions(transitions)
        cdef uint64_t prior=_uint(previous,True)
        cdef string n=_text(name)
        cdef bint profile=functions
        cdef Product result
        with nogil: result=out._handle.get().product(profile,values,conditions,n,prior)
        return out,{'diagram':result.diagram,'output':result.output,'complete':bool(result.complete),
                    'compatible':bool(result.compatible),'status':(<bytes>result.status).decode()}

    def temporal(self, operation, inputs, *, integers=(), rationals=(), domain=None, atoms=(), name='temporal construction'):
        cdef NativeTemporal out=self._fork()
        cdef vector[uint64_t] data=_ids(inputs),numbers=_ids(integers,True)
        cdef vector[string] qs,points
        for value in _sequence(rationals,4096): qs.push_back(_rational(value))
        for row in _sequence(atoms,4096):
            if len(_sequence(row,3))!=3: raise ValueError("atom requires location, mass and identity")
            points.push_back(_rational(row[0]));points.push_back(_rational(row[1]));points.push_back(_text(row[2]))
        cdef uint64_t op=_uint(operation),result
        cdef Domain d=_domain(domain)
        cdef string n=_text(name)
        with nogil: result=out._handle.get().temporal(op,data,numbers,qs,d,points,n)
        return out,result

    def product_info(self, identity):
        cdef Product result=self._get().product_info(_uint(identity))
        return {'diagram':result.diagram,'output':result.output,'complete':bool(result.complete),
                'compatible':bool(result.compatible),'status':(<bytes>result.status).decode()}

    def verify(self):
        with nogil: self._get().verify()
        return True

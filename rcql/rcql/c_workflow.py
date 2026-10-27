"""Retained closed native declarations and prospective C workflow values."""
from dataclasses import dataclass, field
from hashlib import sha256
from threading import RLock
from rexgraph.value_codec import pack_value, unpack_value
from rexgraph.source_binding import source_from_record

KINDS=tuple('''Source AlgebraicArchive CyclicChain Characters Branch BranchModes PeriodConstruction PeriodLattice Symplectic Surface SurfacePeriods SurfaceSymplectic Normalization EllipticFactors Coordinate Domain Expression Carrier UnivariateFamily JetPoint JetDirection JetBase JetReframe Jet RectangularJet Response GreenResponse DerivativeField ResponseApply HodgeAction MapAction InverseAction DerivativeWord WordApply CrossPairing SurfaceCocycle SurfaceHolonomy SurfaceInheritance RotatingSystem RotatingCoefficient RefinementLevel RefinementReframe RefinementMap RefinementCompose RefinementRetraction RefinementDiagram RefinementField RefinementEnergy RefinementSplit RefinementResponse RefinementResponseApply RefinementAccumulate LimitingContract NativeEndpoint NativeOperatorFamily NativeField Krylov CayleyOrbit CovariantFamily FamilyJet FamilyMap FamilyMapJet FamilyAction FamilyResponse LayerFamily LayerFamilyJet'''.split())
LIMITS=(32<<20,128,32,2000000,256,2048)
EXACT=(4096,16000000,64000000,1<<24)
_MAGIC=b'RGCW\x01'
_MAX=64<<20

def _backend():
    from rexgraph_cpp import _c_workflow
    return _c_workflow

def _pack(kind,body):
    payload=pack_value(dict(version=1,kind=kind,value=body))
    raw=_MAGIC+sha256(payload).digest()+payload
    if len(raw)>_MAX:raise ValueError('C workflow packet exceeds its byte limit')
    return raw

def _unpack(raw,kind):
    if type(raw) is not bytes or not 37<=len(raw)<=_MAX or raw[:5]!=_MAGIC or sha256(raw[37:]).digest()!=raw[5:37]:
        raise ValueError('invalid C workflow packet')
    d=unpack_value(raw[37:])
    if type(d) is not dict or set(d)!={'version','kind','value'} or type(d['version']) is not int or d['version']!=1 or d['kind']!=kind:
        raise ValueError('unknown C workflow packet version or kind')
    if _pack(kind,d['value'])!=raw:raise ValueError('noncanonical C workflow packet')
    return d['value']

def _natural(x):
    if type(x) is not int or not 0<=x<2**64:raise ValueError('an exact nonnegative integer is required')
    return x

def _record(row):
    if type(row) is not dict or set(row)!={'kind','key','revision','dependencies','payload'}:raise ValueError('complete closed C declaration required')
    row=dict(row)
    if isinstance(row['kind'],str):row['kind']=KINDS.index(row['kind'])+1
    if not 1<=_natural(row['kind'])<=len(KINDS):raise ValueError('unknown C constructor')
    for k in ('key','revision'):
        if not _natural(row[k]):raise ValueError('C declaration needs nonzero identity/revision')
    row['dependencies']=tuple(_natural(v) for v in row['dependencies'])
    if type(row['payload']) is not bytes:raise TypeError('native declaration payload bytes required')
    return row

@dataclass(frozen=True,eq=False)
class CWorkflow:
    data: bytes
    _owner: object=field(default=None,init=False,compare=False,repr=False)
    _lock: object=field(default_factory=RLock,init=False,compare=False,repr=False)
    _receiving:tuple=field(default=LIMITS,init=False,compare=False,repr=False)
    _arithmetic:tuple=field(default=EXACT,init=False,compare=False,repr=False)
    _kind='c_workflow'
    def __post_init__(self):
        d=self.declaration
        if type(d) is not dict or set(d)!={'implementation','metadata','archive','records','exports','limits','exact','dependencies','parent'}:raise ValueError('invalid C workflow fields')
        if type(d['metadata']) is not dict or set(d['metadata'])!={'producer_namespace','producer','revision','evidence_namespace','evidence_cutoff','purpose'}:raise ValueError('complete original C metadata required')
        if len(d['limits'])!=6 or len(d['exact'])!=4 or any(type(v) is not int or v<0 for v in (*d['limits'],*d['exact'])):raise ValueError('explicit receiving limits required')
        if type(d['archive']) is not bytes or len(d['archive'])>d['limits'][0]:raise ValueError('invalid native C archive')
        if len(d['records'])>d['limits'][1]:raise ValueError('C declaration count exceeds admission')
        names=set();depths=[]
        for i,raw in enumerate(d['records']):
            row=_record(raw)
            if row!=raw or any(j>=i for j in row['dependencies']):raise ValueError('noncanonical or foreign C dependency')
            name=(row['key'],row['revision'])
            if name in names:raise ValueError('duplicate C declaration identity/revision')
            names.add(name);depth=1+max((depths[j] for j in row['dependencies']),default=0);depths.append(depth)
            if depth>d['limits'][2]:raise ValueError('C declaration depth admission')
        if any(type(i) is not int or not 0<=i<len(d['records']) for i in d['exports']):raise ValueError('unknown C export')
        for ref in self.dependencies:ref.check()
        parent=d['parent'];depth=0
        while parent is not None:
            depth+=1
            if depth>128:raise ValueError('C workflow revision depth limit')
            old=_unpack(parent,self._kind)
            if any(v not in d['dependencies'] for v in old['dependencies']):raise ValueError('C revision discarded an original source dependency')
            parent=old['parent']
    @property
    def declaration(self):return _unpack(self.data,self._kind)
    @property
    def coefficient_digest(self):return sha256(self.data).hexdigest()
    @property
    def dependencies(self):return tuple(source_from_record(v,allow_cpp=True) for v in self.declaration['dependencies'])
    @property
    def exports(self):return tuple(self.value(i) for i in self.declaration['exports'])
    @classmethod
    def _wrap(cls,native,limits,exact,dependencies=(),parent=None):
        refs={v.coefficient_digest:v.as_record() for v in dependencies}
        d=dict(implementation=_backend().build_info(),metadata=native.metadata,archive=native.to_bytes(),records=native.records(),exports=native.exports,
               limits=tuple(limits),exact=tuple(exact),dependencies=tuple(refs.values()),parent=parent)
        result=cls(_pack(cls._kind,d))
        object.__setattr__(result,'_owner',native)
        return result
    @classmethod
    def decode(cls,archive,*,limits=LIMITS,exact=EXACT,dependencies=(),cancel=None):
        return cls._wrap(_backend().NativeArchive.decode(archive,limits,exact,cancel),limits,exact,dependencies)
    @classmethod
    def create(cls,metadata,records,exports=(),*,limits=LIMITS,exact=EXACT,dependencies=(),cancel=None):
        from itertools import islice
        if len(limits)!=6 or any(type(v) is not int or v<0 for v in limits):raise ValueError('six receiving limits required')
        rows=tuple(_record(v) for v in islice(records,limits[1]+1))
        if len(rows)>limits[1]:raise ValueError('C declaration count exceeds receiving limits')
        return cls._wrap(_backend().NativeArchive.prepare(metadata,rows,exports,limits,exact,cancel),limits,exact,dependencies)
    @classmethod
    def from_bytes(cls,raw,*,limits=LIMITS,exact=EXACT):
        if len(limits)!=6 or len(exact)!=4 or any(type(v) is not int or v<0 for v in (*limits,*exact)):
            raise ValueError('explicit current C receiving admission required')
        result=cls(raw)
        object.__setattr__(result,'_receiving',tuple(limits))
        object.__setattr__(result,'_arithmetic',tuple(exact))
        return result
    def to_bytes(self):return self.data
    def native(self):
        d=self.declaration;backend=_backend()
        if d['implementation']!=backend.build_info():raise ValueError('C implementation cohort differs; an explicit migration is required')
        with self._lock:
            if self._owner is None:
                owner=backend.NativeArchive.decode(d['archive'],tuple(map(min,zip(d['limits'],self._receiving,strict=True))),tuple(map(min,zip(d['exact'],self._arithmetic,strict=True))))
                if owner.records()!=d['records'] or owner.exports!=d['exports'] or owner.metadata!=d['metadata']:raise ValueError('prospective C descriptors differ from reconstructed native declarations')
                object.__setattr__(self,'_owner',owner)
        return self._owner
    def require_scope(self,scope=None):
        if scope is not None and scope!=self.declaration['metadata']:
            raise ValueError('C producer/evidence cutoff/purpose differs from the requested scope')
    def replay(self,expected_implementation,*,limits=LIMITS,exact=EXACT,sources=None,cancel=None):
        """Explicitly reconstruct original data under the current native code.

        The expected receipt selects the intended producer; it grants no old
        certificate authority. Ordinary native() and cold result reads remain
        strict about the exact implementation cohort.
        """
        d=self.declaration
        if type(expected_implementation) is not dict or expected_implementation!=d['implementation']:
            raise ValueError('explicit original C implementation receipt required')
        current=_backend().build_info()
        if (set(expected_implementation)!=set(current) or
            expected_implementation['profile']!=current['profile'] or
            type(expected_implementation['constructors']) is not int or
            expected_implementation['constructors']!=current['constructors']):
            raise ValueError('C replay requires the supported closed declaration profile')
        for key in ('engine_digest','binding_digest'):
            value=expected_implementation[key]
            if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('C implementation receipt requires exact source digests')
        if len(limits)!=6 or len(exact)!=4 or any(type(v) is not int or v<0 for v in (*limits,*exact)):
            raise ValueError('explicit current C receiving admission required')
        self.require_sources(sources)
        admitted=tuple(map(min,zip(d['limits'],self._receiving,limits,strict=True)))
        arithmetic=tuple(map(min,zip(d['exact'],self._arithmetic,exact,strict=True)))
        owner=_backend().NativeArchive.decode(d['archive'],admitted,arithmetic,cancel)
        if (owner.records()!=d['records'] or owner.exports!=d['exports'] or
            owner.metadata!=d['metadata'] or owner.to_bytes()!=d['archive']):
            raise ValueError('C replay changed original declarations or metadata')
        return self._wrap(owner,admitted,arithmetic,self.dependencies,self.data)

    def value(self,index):return CValue(self,_natural(index))
    def modify(self,replacements=(),*,exports=None,cancel=None):
        d=self.declaration;records=[dict(r) for r in d['records']]
        seen=set()
        for index,row in replacements:
            index=_natural(index)
            if index in seen or index>=len(records):raise ValueError('C modification needs distinct existing declaration positions')
            seen.add(index);records[index]=_record(row)
        metadata=dict(self.native().metadata)
        metadata['revision']=_natural(metadata['revision']+1)
        native=_backend().NativeArchive.prepare(metadata,records,d['exports'] if exports is None else exports,d['limits'],d['exact'],cancel)
        return self._wrap(native,d['limits'],d['exact'],self.dependencies,self.data)
    def require_sources(self,sources=None):
        from .layer_product import _execution_sources
        return _execution_sources(self,{} if sources is None else sources)

@dataclass(frozen=True,eq=False)
class CValue:
    workflow:CWorkflow
    index:int
    _kind='c_value'
    def __post_init__(self):
        _natural(self.index)
        if type(self.workflow) is not CWorkflow or self.index>=len(self.workflow.declaration['records']):raise ValueError('C value requires its actual declaration graph')
    @property
    def constructor(self):return KINDS[self.workflow.declaration['records'][self.index]['kind']-1]
    @property
    def coefficient_digest(self):return sha256(pack_value((self.workflow.coefficient_digest,self.index))).hexdigest()
    @property
    def dependencies(self):return self.workflow.dependencies
    def same_owner(self,other):return type(other) is CValue and self.workflow is other.workflow and self.index==other.index
    def inspect(self,maximum=1000000,sources=None):self.workflow.require_sources(sources);return self.workflow.native().inspect(self.index,maximum)
    def evaluate(self,operation,parameter=0,*,grade=0,power=0,maximum=1000000,sources=None):
        self.workflow.require_sources(sources);return self.workflow.native().evaluate(self.index,operation,parameter,grade,power,maximum)

@dataclass(frozen=True)
class CInterface:
    data:bytes
    _c_owner:object=field(default=None,init=False,compare=False,repr=False)
    _kind='c_interface'
    @classmethod
    def declare(cls,key,proof):
        if type(proof) is not CValue or proof.constructor not in {'RefinementDiagram','LayerFamily'}:raise TypeError('a complete native refinement diagram or coupled rational layer family is required')
        result=cls(_pack(cls._kind,dict(key=key,workflow=proof.workflow.data,proof=proof.index)))
        object.__setattr__(result,'_c_owner',proof.workflow)
        return result
    def __post_init__(self):
        d=self.declaration
        if set(d)!={'key','workflow','proof'} or type(d['key']) is not str or not d['key']:raise ValueError('invalid native C interface')
        value=CWorkflow.from_bytes(d['workflow']).value(d['proof'])
        if value.constructor not in {'RefinementDiagram','LayerFamily'}:raise ValueError('unqualified C compatibility interface')
    @property
    def declaration(self):return _unpack(self.data,self._kind)
    @property
    def coefficient_digest(self):return sha256(self.data).hexdigest()
    @property
    def dependencies(self):return CWorkflow.from_bytes(self.declaration['workflow']).dependencies

def _proof_participants(workflow,proof):
    row=workflow.declaration['records'][proof]
    kind=KINDS[row['kind']-1]
    if kind=='RefinementDiagram':
        # Only declared level legs participate; maps/retractions retain their own roles.
        return tuple(i for i in row['dependencies'] if KINDS[workflow.declaration['records'][i]['kind']-1] in {'RefinementLevel','RefinementReframe'})
    if kind=='LayerFamily':
        first,second=(workflow.declaration['records'][i] for i in row['dependencies'][:2])
        return first['dependencies'][0],second['dependencies'][0]
    raise ValueError('unsupported native joint interface')

def shared_owner(layers,interfaces):
    """Live assembly requires actual owner identity; portable replay is explicit."""
    values=(*layers,*interfaces)
    owners=tuple(getattr(v,'_c_owner',None) for v in values)
    if not owners or owners[0] is None or any(v is not owners[0] for v in owners):
        raise ValueError('C layers and interfaces require their actual shared workflow owner')
    return owners[0]

def validate_product(product,*,execute=False,sources=None):
    layers=product.layers;d=product.declaration
    graphs={v.declaration['native'] for v in layers}
    if len(graphs)!=1:raise ValueError('C layers require one explicitly shared complete declaration graph')
    graph=getattr(product,'_c_owner',None) or CWorkflow.from_bytes(next(iter(graphs)))
    if d['links'] or d['transitions']:raise ValueError('C links and transitions require explicit native constructors')
    participants=tuple(v.declaration['object'] for v in layers)
    interfaces=tuple(CInterface(raw) for raw in d['interfaces'])
    if not 1<=len(interfaces)<=4096 or len({v.declaration['key'] for v in interfaces})!=len(interfaces):raise ValueError('native C layers require distinct complete compatibility interfaces')
    for interface in interfaces:
        x=interface.declaration
        if x['workflow']!=graph.data or _proof_participants(graph,x['proof'])!=participants:
            raise ValueError('C interface does not certify these actual ordered participants')
    if execute:
        graph.require_sources(sources);graph.native()
        object.__setattr__(product,'_c_owner',graph)
    return graph

# This ledger deliberately keeps different mathematical objects in separate profiles.
def primitive_ledger():
    declaration_only={'AlgebraicArchive','BranchModes','DerivativeWord','FamilyAction','HodgeAction','InverseAction','JetDirection','JetPoint','LayerFamilyJet','MapAction','NativeEndpoint','NativeOperatorFamily','RefinementRetraction','UnivariateFamily'}
    rows=[dict(name=name,constructor=i+1,status='registered',profile='closed-C-declarations-v1',
               inspection='declaration-only' if name in declaration_only else 'native-observation') for i,name in enumerate(KINDS)]
    rows.extend(dict(name=name,status='registered',profile='native-C-observation-v1') for name in ('rank','expression','moment','rotation','half_turn','domain_contains','layer_at'))
    rows.extend(dict(name=name,status='native-qualified',profile='native-action-program-v1') for name in ('Tensor','Identity','Scale','Sum','Then','Field','ActionMoments'))
    rows.extend(dict(name=name,status='native-unregistered',profile=profile) for name,profile in (
        ('PreparedGradeFieldAction','native grade action'),
        ('FactoredAction','retained operation factors'),('NativeFactoredAction','source-owned operation factors'),
        ('moment_kernel','operation-valued pairing kernel'),('channel_field_gram','typed channel pairing'),
        ('common_target_gram','common-target channel pairing'),('PreparedNativeMoments','deterministic operator moments'),
        ('PreparedNativeEnsemble','ensemble law contractions'),('PreparedNativeMultitimeObservations','transported multitime fields and moments'),
        ('PreparedNativeMomentChange','exact finite change'),('changing_dimension_cross_moment','rectangular endpoint moment change'),
        ('oriented_action_value','variational action functional'),('ExactChainStrainOperator','oriented exact strain'),
        ('exact_composite_stationary_field','variational stationary solve')))
    return tuple(rows)

"""Prospective retained demand programs; native execution requires an explicit archive."""
from dataclasses import dataclass
from hashlib import sha256
from fractions import Fraction
import struct
from rexgraph.value_codec import pack_value,unpack_value
from rexgraph.source_binding import source_from_record

from .demand_gluing_profile import META,LIMITS

MAGIC=b'RGDG\x01'

def _backend():
    from rexgraph_cpp import _demand_gluing
    return _demand_gluing

def _nat(v):
    if type(v) is not int or not 0<=v<2**64:raise ValueError('exact unsigned integer required')
    return v

def _q(v):
    if type(v) not in (int,str,Fraction):raise TypeError('exact rational required')
    return Fraction(v)

def _pack(value):
    p=pack_value(value);raw=MAGIC+sha256(p).digest()+p
    if len(raw)>64<<20:raise ValueError('demand program admission')
    return raw

def _unpack(raw):
    if type(raw) is not bytes or not 37<=len(raw)<=64<<20 or raw[:5]!=MAGIC or sha256(raw[37:]).digest()!=raw[5:37]:raise ValueError('invalid demand program packet')
    value=unpack_value(raw[37:])
    if _pack(value)!=raw:raise ValueError('noncanonical demand declaration')
    return value

class Reader:
    def __init__(self,raw):self.raw=raw;self.p=0
    def n(self,fmt='<Q'):
        size=struct.calcsize(fmt);v=struct.unpack_from(fmt,self.raw,self.p);self.p+=size;return v[0] if len(v)==1 else v
    def rational(self):
        n=self.n();value=self.raw[self.p:self.p+n].decode('ascii');self.p+=n;return Fraction(value)
    def vector(self):return tuple(self.rational() for _ in range(self.n()))
    def matrix(self):
        rows,cols,count=self.n('<QQQ');entries=tuple((self.n(),self.n(),self.rational()) for _ in range(count))
        return dict(rows=rows,cols=cols,entries=entries)

def _result(raw):
    r=Reader(raw)
    if r.n()!=0x3152454458434652:raise ValueError('unknown native demand result')
    out=dict(grade=r.n('<H'),width=r.n(),moment=r.rational())
    out['field']=tuple((r.n(),r.vector()) for _ in range(r.n()));out['channels']=tuple((r.n(),r.rational()) for _ in range(r.n()))
    out['variables']=tuple(r.n() for _ in range(r.n()));out['edges']=tuple(r.n() for _ in range(r.n()))
    out['particular']=r.vector();out['contradiction']=r.vector();out['equations']=r.matrix();out['kernel']=r.matrix()
    out['boolean_members']=tuple(r.vector() for _ in range(r.n()));out['affine_consistent']=bool(r.n('<B'));out['boolean_complete']=bool(r.n('<B'));out['global']=bool(r.n('<B'))
    out['loaded']=tuple(tuple(r.n() for _ in range(6)) for _ in range(r.n()))
    names=('patch_visits','occurrence_visits','scalar_steps','work_envelope_bytes','output_envelope_bytes','fill_entries','output_coefficient_bits','cover_index_bytes','cache_metadata_bytes','resident_decoded_bytes','resident_coefficient_bits','peak_preparation_scratch_bytes','admitted_payload_io_bytes','resident_raw_bytes','metadata_io_bytes','observed_payload_io_bytes','archive_bytes','index_preparation_payload_io_bytes','index_preparation_patches','index_preparation_work_envelope_bytes','total_observed_payload_io_bytes','total_admitted_payload_io_bytes')
    out['resources']={key:r.n() for key in names}
    if r.p!=len(raw):raise ValueError('trailing native demand result')
    return out

def _request(spec):
    if type(spec) is not dict:raise TypeError('closed demand query record required')
    b=bytearray()
    def n(value,fmt='<Q'):b.extend(struct.pack(fmt,_nat(value)))
    def text(s):raw=s.encode('ascii');n(len(raw));b.extend(raw)
    def rational(v):text(str(_q(v)))
    if spec.get('kind')=='word':
        if set(spec)!={'kind','identity','grade','width','word','channels','field'}:raise ValueError('complete word query declaration required')
        if type(spec['channels']) is not bool or type(spec['word']) is not str:raise TypeError('word and channel declarations required')
        if any(x not in 'DUL' for x in spec['word']) or len(spec['word'])>4096:raise ValueError('unknown/excessive operation word')
        n(1,'<B');n(spec['identity']);n(spec['grade'],'<H');n(spec['width']);text(spec['word']);n(int(spec['channels']),'<B')
        rows=tuple(spec['field']);n(len(rows))
        for key,values in rows:
            n(key)
            if len(values)!=spec['width']:raise ValueError('complete field coordinate vector required')
            for q in values:rational(q)
    elif spec.get('kind')=='family':
        if set(spec)!={'kind','seeds','rhs','global','boolean'} or type(spec['global']) is not bool or type(spec['boolean']) is not bool:raise ValueError('complete joint-family declaration required')
        n(2,'<B');n(int(spec['global']),'<B');n(int(spec['boolean']),'<B');n(len(spec['seeds']))
        for v in spec['seeds']:n(v)
        n(len(spec['rhs']))
        for key,q in spec['rhs']:n(key);rational(q)
    else:raise ValueError('unknown demand query kind')
    if len(b)>16<<20:raise ValueError('demand query byte admission')
    return bytes(b)

@dataclass(frozen=True,eq=False)
class DemandProgram:
    data:bytes
    def __post_init__(self):
        d=self.declaration
        if type(d) is not dict or set(d)!={'version','implementation','manifest','cover','metadata','patches','dependencies','parent'} or d['version']!=1:raise ValueError('invalid demand program declaration')
        if type(d['manifest']) is not bytes or type(d['cover']) is not bytes or len(d['manifest'])>8<<20 or len(d['cover'])>16<<20:raise ValueError('demand original metadata byte admission')
        if set(d['metadata'])!=set(META):raise ValueError('complete producer/clock identities required')
        for v in d['metadata'].values():_nat(v)
        for ref in self.dependencies:ref.check()
        parent=d['parent'];depth=0
        while parent is not None:
            depth+=1
            if depth>128:raise ValueError('demand program parent depth admission')
            old=_unpack(parent)
            if any(ref not in d['dependencies'] for ref in old['dependencies']):raise ValueError('demand revision discarded a source dependency')
            parent=old['parent']
    @property
    def declaration(self):return _unpack(self.data)
    @property
    def coefficient_digest(self):return sha256(self.data).hexdigest()
    @property
    def dependencies(self):return tuple(source_from_record(ref,allow_cpp=True) for ref in self.declaration['dependencies'])
    @classmethod
    def create(cls,manifest,metadata,patches,*,dependencies=(),parent=None):
        if set(metadata)!=set(META):raise ValueError('complete producer/clock identities required')
        metadata={k:_nat(metadata[k]) for k in META};rows=[]
        for p in patches:
            if set(p)!={'index','cells','occurrences','available_at'}:raise ValueError('complete original patch map required')
            rows.append(dict(index=_nat(p['index']),cells=tuple(_nat(v) for v in p['cells']),occurrences=tuple(_nat(v) for v in p['occurrences']),available_at=_nat(p['available_at'])))
        cover=_backend().cover(manifest,tuple(metadata[k] for k in META),rows)
        return cls(_pack(dict(version=1,implementation=_backend().build_info(),manifest=manifest,cover=cover,metadata=metadata,patches=tuple(rows),dependencies=tuple(ref.as_record() for ref in dependencies),parent=parent)))
    @classmethod
    def from_bytes(cls,raw):return cls(raw)
    def to_bytes(self):return self.data
    def require_sources(self,sources=None):
        from .layer_product import _execution_sources
        return {} if not self.dependencies and sources is None else _execution_sources(self,sources)
    def revise(self,*,metadata=None,patches=None):
        d=self.declaration
        if d['implementation']!=_backend().build_info():raise ValueError('explicit demand replay required before revision')
        m=d['metadata'] if metadata is None else metadata
        if m['namespace']!=d['metadata']['namespace'] or m['key']!=d['metadata']['key'] or m['revision']<=d['metadata']['revision']:raise ValueError('demand revision must advance its original producer')
        return type(self).create(d['manifest'],m,d['patches'] if patches is None else patches,dependencies=self.dependencies,parent=self.data)
    def open(self,path,*,limits=LIMITS,sources=None):
        self.require_sources(sources);d=self.declaration
        if d['implementation']!=_backend().build_info():raise ValueError('original demand implementation differs; explicit replay required')
        if len(limits)!=19 or any(type(v) is not int or v<0 for v in limits):raise ValueError('nineteen receiving resource limits required')
        expected=_backend().cover(d['manifest'],tuple(d['metadata'][k] for k in META),d['patches'])
        if expected!=d['cover']:raise ValueError('demand original quotient declaration differs')
        return _backend().NativeDemand.open(path,d['manifest'],d['cover'],limits)
    def run(self,path,request,*,limits=LIMITS,sources=None):
        # Freeze query before opening a provider or calling native cancellation code.
        request=unpack_value(pack_value(request));raw=_request(request);owner=self.open(path,limits=limits,sources=sources)
        result=_result(owner.execute(raw))
        result['domain']='boolean' if request.get('boolean') else 'rational'
        result['consistent']=bool(result['boolean_members']) if request.get('boolean') else result['affine_consistent']
        return dict(profile='demand-graph-identity-v1',scope=('whole-declared-cover' if request.get('global') else 'connected-joint-family') if request['kind']=='family' else 'requested-operation-word',program=self.data,request=request,implementation=_backend().build_info(),result=result,parent=None)
    @classmethod
    def replay(cls,old,path,expected_implementation,*,limits=LIMITS,sources=None):
        if type(old) is not dict or old.get('profile')!='demand-graph-identity-v1' or old['implementation']!=expected_implementation:raise ValueError('explicit exact original demand receipt required')
        program=cls.from_bytes(old['program']);d=program.declaration
        if d['implementation']!=expected_implementation:raise ValueError('demand producer/receipt mismatch')
        current=cls.create(d['manifest'],d['metadata'],d['patches'],dependencies=program.dependencies,parent=program.data)
        result=current.run(path,old['request'],limits=limits,sources=sources)
        semantic=('grade','width','moment','field','channels','variables','edges','particular','contradiction','equations','kernel','boolean_members','consistent','affine_consistent','domain','boolean_complete','global')
        if any(result['result'][k]!=old['result'][k] for k in semantic):raise ValueError('historical demand observation differs under current implementation')
        result['parent']=pack_value(old);return result

def publish_archive(path,payloads):
    """Exclusively publish original canonical RFCX sources; never replace a file."""
    rows=[];total=0
    for row in payloads:
        if len(rows)==256:raise ValueError('archive producer count admission')
        if set(row)!={'namespace','source','revision','payload','bytes'} or type(row['bytes']) is not bytes:raise ValueError('exact original source producer required')
        total+=len(row['bytes'])
        if len(row['bytes'])>16<<20 or total>256<<20:raise ValueError('source receiving byte admission')
        rows.append({k:(v if k=='bytes' else _nat(v)) for k,v in row.items()})
    return _backend().publish(path,rows)

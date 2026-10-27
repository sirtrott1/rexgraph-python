from pathlib import Path
import os,struct
import pytest
pytest.importorskip('rexgraph_cpp._demand_gluing')
from rcql import DemandProgram,Executor,Result,parse
from rexgraph import RexGraph
from rcql.demand_gluing import LIMITS,publish_archive

@pytest.fixture
def fixture():
    root=os.environ.get('REXGRAPH_DEMAND_FIXTURES')
    if not root:pytest.skip('explicit native synthetic demand fixtures required')
    root=Path(root);meta=dict(namespace=901,key=44,revision=1,evidence_namespace=902,cutoff=10,parameter_revision=2,arrangement_revision=3)
    patches=tuple(dict(index=i,cells=(*pair,101+i),occurrences=tuple((101+i)*10+j for j in range(3)),available_at=5) for i,pair in enumerate([(1,2),(3,4),(2,3)]))
    return root,meta,patches

def word(operations='LLL',*,channels=False,width=1,field=None):
    return dict(kind='word',identity=70,grade=0,width=width,word=operations,channels=channels,field=((1,(1,)*width),) if field is None else field)

def test_original_program_cold_selected_moments(fixture):
    root,meta,patches=fixture;p=DemandProgram.create((root/'manifest.bin').read_bytes(),meta,patches)
    assert DemandProgram.from_bytes(p.data).data==p.data
    assert [p.run(root/'source.archive',word(w))['result']['moment'] for w in ('','L','LL','LLL')]==[1,1,2,5]
    result=p.run(root/'source.archive',word('L',channels=True))['result']
    assert dict(result['channels'])=={101:4,103:1}
    assert len(result['loaded'])==2 and result['resources']['observed_payload_io_bytes']<result['resources']['archive_bytes']
    assert p.run(root/'source.archive',word(width=2))['result']['moment']==10

def test_program_revision_retains_original_sources_and_runtime(fixture):
    root,meta,patches=fixture;p=DemandProgram.create((root/'manifest.bin').read_bytes(),meta,patches[:2]);old=p.run(root/'source.archive',word())
    new=p.revise(metadata=dict(meta,revision=2,parameter_revision=3,arrangement_revision=4),patches=patches)
    assert old['result']['moment']==4 and new.run(root/'source.archive',word())['result']['moment']==5
    assert new.declaration['parent']==p.data and p.declaration['metadata']==meta
    assert p.run(root/'source.archive',word())['result']['moment']==4

def test_complete_correlated_family_and_boolean_contradiction(fixture):
    root,meta,patches=fixture;raw=(root/'correlated.archive').read_bytes();n=struct.unpack_from('<Q',raw,24)[0]
    maps=tuple(dict(index=j,cells=(*pair,101+j),occurrences=tuple((101+j)*10+k for k in range(3)),available_at=5) for j,pair in enumerate([(1,2),(2,3)]))
    p=DemandProgram.create(raw[40:40+n],meta,maps)
    request=dict(kind='family',seeds=(1,),rhs=((101,1),(102,2)),boolean=False,**{'global':False})
    r=p.run(root/'correlated.archive',request)['result']
    assert r['consistent'] and r['variables']==(1,2,3) and r['kernel']['cols']==1
    assert r['particular'][2]-r['particular'][0]==1 and len(r['loaded'])==2
    boolean=p.run(root/'correlated.archive',dict(request,rhs=((101,1),(102,1)),boolean=True,**{'global':True}))['result']
    assert boolean['boolean_complete'] and set(boolean['boolean_members'])=={(0,1,0),(1,0,1)}

def test_native_session_eviction_and_cancellation(fixture):
    from rcql.demand_gluing import _request,_result
    root,meta,patches=fixture;p=DemandProgram.create((root/'manifest.bin').read_bytes(),meta,patches);limits=list(LIMITS);limits[11]=1
    owner=p.open(root/'source.archive',limits=limits);first=_result(owner.execute(_request(word())))
    for n in range(3):owner.evict(n)
    again=_result(owner.execute(_request(word())))
    assert first['field']==again['field'] and again['moment']==5 and first['loaded']!=again['loaded']
    owner.cancel()
    with pytest.raises(RuntimeError):owner.execute(_request(word()))

def test_cutoff_receiving_budget_and_original_scope(fixture):
    root,meta,patches=fixture;p=DemandProgram.create((root/'manifest.bin').read_bytes(),meta,patches)
    unavailable=p.revise(metadata=dict(meta,revision=2,cutoff=4))
    with pytest.raises(ValueError):unavailable.run(root/'source.archive',word())
    for index in (0,1,2,3,11,12,13,14,15,16):
        limits=list(LIMITS);limits[index]=0
        with pytest.raises((ValueError,RuntimeError,IndexError,MemoryError)):p.run(root/'source.archive',word(),limits=limits)
    with pytest.raises(TypeError):p.run(root/'source.archive',word(field=((1,(0.5,)),)))
    with pytest.raises(ValueError):p.run(root/'source.archive',dict(word(),unknown=True))

def test_explicit_historical_replay_and_plain_result_codec(fixture):
    root,meta,patches=fixture;p=DemandProgram.create((root/'manifest.bin').read_bytes(),meta,patches);first=p.run(root/'source.archive',word());cold=Result.from_bytes(Result((first,)).to_bytes()).values[0]
    assert cold['result']==first['result']
    newer=DemandProgram.replay(cold,root/'source.archive',cold['implementation'])
    assert newer['result']['moment']==5 and newer['parent'] is not None
    with pytest.raises(ValueError):DemandProgram.replay(cold,root/'source.archive',dict(cold['implementation'],engine_digest='0'*64))

def test_rcql_build_modify_execute_replay_and_cold_result(fixture):
    root,meta,patches=fixture;e=Executor(sources={'r':RexGraph.from_cells([1,[]])},params={'manifest':(root/'manifest.bin').read_bytes(),'meta':meta,'maps':patches,'path':str(root/'source.archive'),'request':word()})
    result=e.execute(parse('FROM $r LET p=DEMAND_PROGRAM($manifest,$meta,$maps) LET a=DEMAND_RUN(p,$path,$request) RETURN p,a'))
    assert result.values[1]['result']['moment']==5
    cold=Result.from_bytes(result.to_bytes());assert cold.values[0]==result.values[0] and cold.values[1]['result']['moment']==5
    e.params.update(result=cold.values[1],implementation=cold.values[1]['implementation'])
    replay=e.execute(parse('FROM $r RETURN DEMAND_REPLAY($result,$path,$implementation)'))
    assert replay.values[0]['result']['moment']==5

def test_archive_producer_is_exclusive_and_preserves_original_rfcx(fixture,tmp_path):
    root,meta,patches=fixture;raw=(root/'source.archive').read_bytes();n=struct.unpack_from('<Q',raw,24)[0];count=struct.unpack_from('<Q',raw,32)[0];payloads=[]
    for i in range(count):
        offset,size=struct.unpack_from('<QQ',raw,40+n+i*16);payloads.append(dict(namespace=990,source=i+1,revision=7,payload=i+1,bytes=raw[offset:offset+size]))
    path=tmp_path/'archive';manifest=publish_archive(path,payloads);assert manifest==(root/'manifest.bin').read_bytes()
    p=DemandProgram.create(manifest,meta,patches);assert p.run(path,word())['result']['moment']==5
    original=path.read_bytes()
    with pytest.raises(ValueError):publish_archive(path,payloads)
    assert path.read_bytes()==original

@pytest.mark.parametrize('permissions', [('read',),('read','files'),('read','file_read'),('read','files','file_read')])
def test_archive_execution_requires_existing_file_permissions(fixture,permissions):
    from rcql.capabilities import BoundSource,SourcePolicy
    root,meta,patches=fixture;p=DemandProgram.create((root/'manifest.bin').read_bytes(),meta,patches)
    e=Executor(sources={'r':BoundSource(RexGraph.from_cells([1,[]]),SourcePolicy.allow(*permissions))},params={'program':p.data,'path':str(root/'source.archive'),'request':word()})
    query=parse('FROM $r RETURN DEMAND_RUN($program,$path,$request)')
    if len(permissions)==3:assert e.execute(query).values[0]['result']['moment']==5
    else:
        with pytest.raises(PermissionError):e.execute(query)

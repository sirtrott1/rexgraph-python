"""Joint native parameter families, source identities and compatible sections."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
from hashlib import sha256
import subprocess
import sys

import pytest

from rexgraph.cpp_stalk_pattern import compile_stalk_context_pattern_bank
from rexgraph.cpp_stalk_parameter_relations import stalk_context_view_rules_query
from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model, stalk_transport_observation_batch
from rexgraph.model_runtime import train_model
from rexgraph_cpp import NativeSource, PreparedStalkTransportPatternBank
from test_stalk_context_view_rules import learn, query


def compile_bank(redundant=False):
    source, state, views = learn(redundant=redundant)
    bank = compile_stalk_context_pattern_bank(state, source, query(state, views), (0, 1),
        model_namespace=905, coordinate_ids={'weights':101, 'parameters':102, 'domains':(100,103)})
    return source, state, bank


def correlated_bank():
    declaration = dict(profile='declared-role-source-v1', namespace=99051, semantic_scale=0, max_grade=1,
        role_aspects=('boundary',), types=(dict(name='context', min_arity=1, max_arity=1),
        dict(name='anchor', min_arity=1, max_arity=1)), witnesses=(10,),
        relations=tuple(dict(grade=1,key=i+1,type=int(i>=3),occurrences=(
            dict(key=i+1,lower=10,negative=False,magnitude=1,exists=True,aspects=(0,)),)) for i in range(6)),
        fields=(), metrics=(), carriers=(dict(key=10,kind='source_grade',grade=0,coordinates=0,size=1),
            dict(key=11,kind='source_grade',grade=1,coordinates=0,size=6)))
    source = NativeSource.from_declaration(declaration)
    axes = ((('value',),), (('signal',),)*6)
    state = cpp_stalk_transport_model(source, axes, ((), ((),)*6), tuple((i,0,0) for i in range(6)), (0,))
    probes = (((-1,1,0,0,0,0),0), ((-1,0,1,0,0,0),0),
        ((0,0,0,Q(1,2),0,0),1), ((0,0,0,0,Q(1,2),0),1), ((0,0,0,0,0,Q(1,2)),1))
    for field, target in probes:
        state = train_model(state,source,stalk_transport_observation_batch(state,
            tuple((v,) for v in field), ((0,(target,)),)))
    identity = ((0,0,1),)
    views = ((('reading',('value',),identity,identity),),
        (('context',('signal',),identity,identity),)*6)
    q = stalk_context_view_rules_query(state,tuple(range(6)),(),views,minimum_peers=2)
    bank = compile_stalk_context_pattern_bank(state,source,q,(0,1,3),model_namespace=905,
        coordinate_ids={'weights':101,'parameters':102,'domains':(100,103,104)})
    return source,state,bank


def comparison(name, left, right, gain=1):
    return dict(name=name,left=left,right=right,left_frame=1,right_frame=1,interface_frame=1,
        left_map={'shape':(1,1),'entries':((0,0,1),)},
        right_map={'shape':(1,1),'entries':((0,0,gain),)})


@pytest.mark.parametrize('redundant',[False,True])
def test_joint_archive_keeps_named_carriers_and_original_source_provenance(redundant):
    source,state,bank=compile_bank(redundant)
    inspected=bank.inspect()
    assert len(inspected['weight_origin'])==16
    assert inspected['parameter_directions']['shape']==(16,0)
    assert all(inspected['identified'])
    assert inspected['source_sha256']==sha256(source.archive).hexdigest()
    assert inspected['evidence_sha256']==state.coefficient_digest
    assert inspected['observations']==4 and inspected['peers']==2
    assert len(inspected['occurrence_keys'])==4 and len(inspected['pattern_keys'])==2
    assert {name:axes for _,name,axes in inspected['common_frames']}=={
        'participant':('x','y'),'clause':('a','b','context','redundant')}
    x=(Q(2,3),Q(-3,7),0,Q(2,3) if redundant else 0)
    expected=(Q(149,105),Q(-10,7),Q(298,105),Q(-20,7))
    assert bank.response({0:x,1:x})['origin']==expected+expected
    assert bank.glue({0:x,1:x})['forced']
    with pytest.raises(ValueError,match='outside'):
        bank.response({0:(0,0,1,0)})
    image=bank.archive(); loaded=PreparedStalkTransportPatternBank(image,model_namespace=905)
    assert loaded.archive()==image and loaded.inspect()==inspected
    inspected['weight_origin']=(999,)
    assert bank.inspect()['weight_origin']!=(999,)


def test_learned_correlations_make_a_difference_known_without_filling_unknown_weights():
    _,_,bank=correlated_bank(); inspected=bank.inspect()
    assert inspected['weight_origin']==(0,0,2)
    assert inspected['parameter_directions']=={'shape':(3,1),'entries':((0,0,Q(1)),(1,0,Q(1)))}
    assert inspected['identified']==(False,False,True)
    assert bank.response({0:(1,)})['directions']['shape']==(1,1)
    cancelled=bank.response({0:(1,),1:(-1,)})
    assert cancelled['identified'] and cancelled['origin']==(0,)
    readings=bank.response({0:(1,),1:(1,),2:(1,)},shared=False)
    assert readings['origin']==(0,0,2) and readings['directions']==inspected['parameter_directions']


def test_gluing_identifies_shared_cells_once_and_returns_complete_conditioning():
    _,state,bank=correlated_bank(); digest=state.coefficient_digest
    forced=bank.glue({0:(1,),1:(1,)})
    assert forced['forced'] and forced['consistent']
    assert forced['section']['origin']==(0,) and forced['section_cells']==(0,)
    assert forced['section']['directions']['entries']==((0,0,Q(1)),)
    assert forced['boundary_response']['directions']['entries']==((0,0,Q(2)),)
    conditioned=bank.glue({0:(1,),2:(1,)})
    assert conditioned['consistent'] and not conditioned['forced']
    assert conditioned['parameter_origin']==(2,) and conditioned['parameter_directions']['shape']==(1,0)
    assert conditioned['section']['origin']==(2,) and conditioned['section']['identified']
    assert conditioned['boundary_response']['origin']==(4,)
    assert bank.response({0:(1,)})['directions']['shape']==(1,1)
    assert state.coefficient_digest==digest


def test_incompatible_restrictions_return_an_exact_witness_without_a_section():
    _,_,bank=correlated_bank()
    result=bank.glue({0:(1,),2:(1,)},(comparison('first',0,2),comparison('second',0,2,Q(1,2))))
    assert not result['consistent'] and not result['forced']
    assert result['section'] is None and result['boundary_response'] is None
    assert result['defect_origin']==(-2,-2,-1)
    witness=result['contradiction']
    assert sum(witness)==0
    assert sum(q*x for q,x in zip(witness,result['defect_origin'],strict=True))!=0
    assert result['defect_directions']['entries']==((0,0,Q(1)),(1,0,Q(1)),(2,0,Q(1)))


def test_cold_native_load_preserves_correlations_without_training_or_python_learning(tmp_path):
    _,_,bank=correlated_bank(); path=tmp_path/'bank.rfcxs'; path.write_bytes(bank.archive())
    code='''
from pathlib import Path
from rexgraph_cpp import PreparedStalkTransportPatternBank
import sys
image=Path(sys.argv[1]).read_bytes()
b=PreparedStalkTransportPatternBank(image,model_namespace=905)
assert b.response({0:(1,),1:(-1,)})['identified']
assert not b.response({0:(1,)})['identified']
g=b.glue({0:(1,),2:(1,)})
assert g['section']['origin']==(2,) and g['boundary_response']['origin']==(4,)
assert b.archive()==image
assert 'rexgraph.model_runtime' not in sys.modules
assert 'torch' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(path)],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    loaded=PreparedStalkTransportPatternBank(path.read_bytes(),model_namespace=905)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=tuple(pool.map(lambda _:loaded.glue({0:(1,),2:(1,)}),range(8)))
    assert all(r==results[0] for r in results)


@pytest.mark.parametrize('case',['namespace','truncated','corrupt','dimension','entries','steps','bytes'])
def test_archive_admission_is_bounded_and_canonical(case):
    _,_,bank=compile_bank(); image=bank.archive(); ns=905
    limits=(4096,16000000,64000000,16384); maximum=len(image)
    if case=='namespace':ns+=1
    if case=='truncated':image=image[:-1]
    if case=='corrupt':image=image[:-1]+bytes([image[-1]^1])
    if case=='dimension':limits=(1,16000000,64000000,16384)
    if case=='entries':limits=(4096,1,64000000,16384)
    if case=='steps':limits=(4096,16000000,1,16384)
    if case=='bytes':maximum-=1
    with pytest.raises((ValueError,RuntimeError,OverflowError)):
        PreparedStalkTransportPatternBank(image,model_namespace=ns,limits=limits,max_bytes=maximum)


@pytest.mark.parametrize('case',['order','duplicate','unobserved','conflict','partial','coordinates','domains','namespace'])
def test_only_admitted_patterns_and_distinct_declared_coordinates_are_exported(case):
    source,state,views=learn(conflict=case=='conflict',partial=case=='partial')
    origins={'order':(1,0),'duplicate':(0,0),'unobserved':(0,2)}.get(case,(0,1))
    coords={'weights':101,'parameters':102,'domains':(100,103)}
    if case=='coordinates':coords['weights']=1
    if case=='domains':coords['domains']=(100,100)
    with pytest.raises((ValueError,RuntimeError,OverflowError)):
        compile_stalk_context_pattern_bank(state,source,query(state,views),origins,
            model_namespace=0 if case=='namespace' else 905,coordinate_ids=coords)


@pytest.mark.parametrize('case',['left_frame','right_frame','interface_frame','missing','shape','duplicate','entry'])
def test_interface_comparisons_require_exact_declared_carriers_and_maps(case):
    _,_,bank=correlated_bank(); c=comparison('test',0,2)
    if case in ('left_frame','right_frame','interface_frame'):c[case]=99 if case!='interface_frame' else 0
    if case=='missing':c['right']=1
    if case=='shape':c['right_map']['shape']=(1,2)
    if case=='duplicate':c['left_map']['entries']=((0,0,1),(0,0,1))
    if case=='entry':c['left_map']['entries']=((1,0,1),)
    with pytest.raises((ValueError,RuntimeError,OverflowError)):
        bank.glue({0:(1,),2:(1,)},(c,))

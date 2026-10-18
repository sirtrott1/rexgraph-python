"""Complete native parameter families and explicit relation intersections."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import pytest

from rexgraph_cpp import NativeSource, PreparedStalkTransport, PreparedStalkParameterRelations
from rexgraph.cpp_stalk_parameter_relations import stalk_parameter_relations_query, stalk_parameter_relation
from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record
from test_stalk_objective import model, selected, AXES
from test_stalk_transport import declaration

RELATIONS=(('observed',((0,1),(1,-1)),1),('excluded',((0,1),(1,-1)),2),
           ('F-agreement',((0,1),(1,1)),0),('first-zero',((0,1),),0),('first-one',((0,1),),1))


def query(state, chosen=()):
    return stalk_parameter_relations_query(state,RELATIONS,selected=chosen)


def test_original_observations_classify_the_complete_family():
    source,state=model();digest=state.coefficient_digest
    output=infer_model(state,source,query(state,('F-agreement',)))
    certificate=output.execution_receipt['certificate']
    assert certificate['statuses']==('forced','excluded','open','open','open')
    assert certificate['evidence_scope']=='independent-original-entries'
    assert output.values.tolist()==[[Q(0),Q(0)],[Q(-1),Q(0)],[Q(1),Q(2)],[Q(1),Q(1)],[Q(0),Q(1)]]
    assert output.axes[0].keys==('origin','free:1')
    conditional=output.execution_receipt['conditional_family']
    assert conditional['consistent'] and conditional['particular']==(Q(1,2),Q(-1,2))
    assert conditional['kernel_shape']==(2,0) and state.coefficient_digest==digest
    assert certificate['evidence_fit']==state.payload['fit']
    relation=stalk_parameter_relation(output,'F-agreement')
    assert relation['status']=='open' and relation['residual_family']==(Q(1),Q(2))
    with pytest.raises(TypeError,match='explicit relation'):output.tensor()


def test_selected_F_coincidence_is_not_observed_parameter_sharing():
    source,state=model();objective=selected(source,state)
    coefficients=objective.payload['selection']['particular']
    assert sum(coefficients)==0
    output=infer_model(state,source,query(state))
    assert stalk_parameter_relation(output,'F-agreement')['status']=='open'
    with pytest.raises(ValueError):stalk_parameter_relations_query(objective,RELATIONS)


@pytest.mark.parametrize('chosen,consistent',[(('excluded',),False),(('first-zero','first-one'),False),
                                           (('observed','first-one'),True),(('F-agreement',),True)])
def test_joint_consistency_is_distinct_from_individual_feasibility(chosen,consistent):
    source,state=model();output=infer_model(state,source,query(state,chosen))
    family=output.execution_receipt['conditional_family']
    assert family['consistent'] is consistent and family['selected']==chosen
    if not consistent:assert family['particular']==() and family['kernel_shape']==(2,0)
    assert output.execution_receipt['certificate']['statuses']==('forced','excluded','open','open','open')


def test_repeated_inactive_and_supported_zero_entries_keep_free_directions():
    source=NativeSource.from_declaration(declaration(((10,1,True),(10,1,True),(20,7,False)),namespace=929801))
    axes=((('meaning',),('unused',)),(('reading',),))
    state=cpp_stalk_transport_model(source,axes,((),((),(),())),((0,0,0),(1,0,0),(2,0,0)))
    relations=(('tie',((0,1),(1,-1)),0),('inactive',((2,1),),0),('zero',((2,0),),0))
    output=infer_model(state,source,stalk_parameter_relations_query(state,relations))
    assert output.axes[0].keys==('origin','free:0','free:1','free:2')
    assert output.values.tolist()==[[Q(0),Q(1),Q(-1),Q(0)],[Q(0),Q(0),Q(0),Q(1)],[Q(0)]*4]
    receipt=output.execution_receipt['certificate']
    assert receipt['statuses']==('open','open','forced') and receipt['relations'][2][1]==((2,Q(0)),)
    assert receipt['evidence_fit']['occurrence_keys']==((),(1,2,3))


def test_declared_templates_are_explicitly_conditional_evidence():
    source,_=model()
    state=cpp_stalk_transport_model(source,AXES,((),((),())),((0,0,0),(1,0,0)),
        parameter_space={'name':'declared-gains','keys':('a','b')},
        parameter_basis=((0,0,1),(0,1,1),(1,0,1),(1,1,1)))
    output=infer_model(state,source,stalk_parameter_relations_query(state,(('tie',((0,1),(1,-1)),0),)))
    r=output.execution_receipt['certificate']
    assert r['statuses']==('forced',) and r['evidence_scope']=='conditional-on-declared-parameter-basis'
    assert output.axes[0].keys==('origin','free:a','free:b') and output.values.tolist()==[[Q(0)]*3]


def test_parameter_owner_pins_original_samples_and_returns_detached_receipts():
    source,state=model()
    transport=PreparedStalkTransport(source.archive,AXES,((),((),())),((0,0,0),(1,0,0)),
        (dict(input=((1,),(1,)),target=((1,),),observed=(True,)),))
    owner=PreparedStalkParameterRelations(transport,RELATIONS)
    del transport,source,state
    receipt=owner.receipt;receipt['evidence_fit']['particular']=(99,99);receipt['statuses']=()
    assert owner.receipt['evidence_fit']['particular']==(Q(1),Q(0))
    expected=owner.condition(('F-agreement',))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(result==expected for result in pool.map(lambda _:owner.condition(('F-agreement',)),range(8)))
    with pytest.raises(RuntimeError):owner.__init__(None,RELATIONS)


def test_uninitialized_or_missing_transport_owners_reject():
    owner=PreparedStalkParameterRelations.__new__(PreparedStalkParameterRelations)
    with pytest.raises(RuntimeError):owner.condition(())
    with pytest.raises(RuntimeError):_ = owner.receipt
    with pytest.raises(TypeError):PreparedStalkParameterRelations(None,RELATIONS)


@pytest.mark.parametrize('limits',[(4096,128,64000000,16384),(4096,16000000,200,16384)])
def test_native_parameter_relation_admissions_cover_declarations_and_replay(limits):
    source,_=model()
    transport=PreparedStalkTransport(source.archive,AXES,((),((),())),((0,0,0),(1,0,0)),limits=limits)
    with pytest.raises((ValueError,RuntimeError,OverflowError)):
        PreparedStalkParameterRelations(transport,tuple(('declaration:'+str(i),((0,1),(1,-1)),0) for i in range(30)))


@pytest.mark.parametrize('relations',[
    (), (('x',(),0),), (('x',((2,1),),0),), (('x',((0,1),(0,0)),0),),
    (('x',((0,1),),0),('x',((1,1),),0)), (('',((0,1),),0),),
    (('x',((True,1),),0),), (('x',((0,1.0),),0),), (('x',((0,1),),True),),
])
def test_malformed_entry_hypotheses_reject(relations):
    source,state=model()
    with pytest.raises((ValueError,TypeError,OverflowError)):
        q=stalk_parameter_relations_query(state,relations);infer_model(state,source,q)


@pytest.mark.parametrize('chosen',[('missing',),('observed','observed'),'observed',(True,)])
def test_malformed_selected_equations_reject(chosen):
    _,state=model()
    with pytest.raises((ValueError,TypeError)):query(state,chosen)


@pytest.mark.parametrize('field',['statuses','residual_origin','residual_directions','evidence_scope'])
def test_changed_family_certificates_reject(field):
    source,state=model();output=infer_model(state,source,query(state,('F-agreement',)))
    receipt=thaw_tree(output.execution_receipt)
    altered={'statuses':('forced',)*5,'residual_origin':(Q(99),)*5,
             'residual_directions':(),'evidence_scope':'independent-other-entries'}
    receipt['certificate'][field]=altered[field]
    with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=receipt)


def test_changed_intersection_values_axes_and_source_reject():
    source,state=model();output=infer_model(state,source,query(state,('F-agreement',)))
    receipt=thaw_tree(output.execution_receipt);receipt['conditional_family']['consistent']=False
    with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=receipt)
    values=output.values.copy();values[2,1]=Q(0)
    with pytest.raises(ValueError,match='replay'):replace(output,values=values)
    with pytest.raises(ValueError):replace(output,grade=0)
    q=query(state);q['extra']=0
    with pytest.raises(ValueError):infer_model(state,source,q)
    other=NativeSource.from_declaration(declaration(namespace=929802))
    with pytest.raises(ValueError):infer_model(state,other,query(state))


@pytest.mark.parametrize('value',[True,Q(1),1.0])
def test_source_reference_scalar_kinds_are_not_coerced(value):
    source,state=model();q=query(state);q['source_reference']['schema_version']=value
    with pytest.raises(ValueError,match='actual transport source'):infer_model(state,source,q)


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_complete_family_model_and_output_roundtrip(tmp_path,suffix):
    source,state=model();q=query(state,('F-agreement',));output=infer_model(state,source,q)
    record=model_record(source,state);record.attach_metadata(1,0,'query',q);record.attach_metadata(1,0,'output',output)
    path=tmp_path/('relations'+suffix);save_model(path,record)
    loaded=load_model(path);restored,retained=read_model_record(loaded)
    assert restored.coefficient_digest==state.coefficient_digest
    assert infer_model(restored,retained,q).coefficient_digest==output.coefficient_digest
    assert loaded.get_metadata(1,0,'output').coefficient_digest==output.coefficient_digest
    assert unpack_model(pack_model(output)).coefficient_digest==output.coefficient_digest


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_cold_five_package_relations_preserve_evidence_and_conditional_families(tmp_path,backend):
    from rcdb import open_store
    source,state=model();q=query(state,('F-agreement',));output=infer_model(state,source,q)
    record=model_record(source,state);record.attach_metadata(1,0,'query',q);record.attach_metadata(1,0,'output',output)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('relations',record,expected_version=0);assert store.verify_commits('relations');store.close()
    script='''
import sys
from unittest.mock import patch
from fractions import Fraction as Q
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import read_model_record
store=open_store(sys.argv[1]);record=store.read_record('relations',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
output=result.values[0]
assert output.coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert output.space.name=='transport-parameter-relation' and result.values[1]['parameter_relations']
assert output.execution_receipt['certificate']['statuses']==('forced','excluded','open','open','open')
assert output.execution_receipt['conditional_family']['particular']==(Q(1,2),Q(-1,2))
assert json_value(state)['parameter_relations']
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('relations');hive.add_worker('relations',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('relations',query,record=False).coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    result=subprocess.run([sys.executable,'-c',script,url,output.coefficient_digest],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr

"""Whole field constraints preserve correlated families and certify contradictions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import pytest

from rexgraph_cpp import PreparedStalkTransport, PreparedStalkParameterRelations
from rexgraph.cpp_stalk_parameter_relations import (
    stalk_context_rule_restriction, validate_stalk_context_restriction,
    stalk_context_view_rules_query, stalk_context_hypotheses, stalk_context_rule_response,
    PreparedStalkContextRules,
)
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.value_codec import pack_value, unpack_value
from test_stalk_objective import model, AXES
from test_stalk_parameter_relations import RELATIONS
from test_stalk_context_view_rules import learn, query


def owner():
    source,_=model()
    transport=PreparedStalkTransport(source.archive,AXES,((),((),())),((0,0,0),(1,0,0)),
        (dict(input=((1,),(1,)),target=((1,),),observed=(True,)),))
    return PreparedStalkParameterRelations(transport,RELATIONS)


def observation(field=((1,),(0,)), target=2):
    return dict(input=field,target=((target,),),observed=(True,))


def test_exact_intersection_keeps_original_coordinate_family_and_source():
    native=owner();before=native.receipt
    result=native.restrict_fields((observation(),))
    assert result['status']=='open' and result['additional_rank']==1
    family=result['family']
    assert family['particular']==(Q(2),Q(1)) and family['coordinate_particular']==(Q(1),)
    assert family['kernel_shape']==(2,0) and family['coordinate_kernel_shape']==(1,0)
    assert result['constraints']==((0,0,0),) and result['contradiction']==()
    assert result['archive_digest']==before['archive_digest'] and native.receipt==before


def test_selected_equations_and_observed_fields_condition_one_joint_family():
    native=owner()
    forced=native.restrict_fields((observation(target=Q(1,2)),),('F-agreement',))
    assert forced['status']=='forced' and forced['base_rank']==1 and forced['additional_rank']==0
    assert forced['family']['particular']==(Q(1,2),Q(-1,2))
    excluded=native.restrict_fields((observation(target=0),),('F-agreement',))
    assert excluded['status']=='excluded' and not excluded['family']['consistent']
    assert excluded['contradiction']==(Q(1),) and excluded['contradiction_value']==Q(-1,2)
    bad=native.restrict_fields((observation(),),('first-zero','first-one'))
    assert bad['status']=='inconsistent-base' and not bad['base_consistent'] and bad['contradiction']==()


def test_individually_open_fields_can_be_jointly_inconsistent():
    native=owner();left=observation(target=0);right=observation(target=1)
    assert native.restrict_fields((left,))['status']=='open'
    assert native.restrict_fields((right,))['status']=='open'
    joint=native.restrict_fields((left,right))
    assert joint['status']=='excluded' and joint['family']['particular']==()
    assert joint['constraints']==((0,0,0),(1,0,0))
    assert joint['contradiction']==(Q(-1),Q(1)) and joint['contradiction_value']==1
    assert native.restrict_fields((observation(),))['family']['particular']==(Q(2),Q(1))


def test_unobserved_field_directions_are_retained_after_restriction():
    source,_=model()
    transport=PreparedStalkTransport(source.archive,AXES,((),((),())),((0,0,0),(1,0,0)))
    native=PreparedStalkParameterRelations(transport,(('unrestricted',(),0),),allow_constants=True)
    result=native.restrict_fields((observation(),))
    assert result['family']['particular']==(Q(2),Q(0))
    assert result['family']['kernel']==((1,0,Q(1)),) and result['family']['free_columns']==(1,)


def test_declared_parameter_basis_is_applied_before_whole_field_conditioning():
    source,_=model()
    transport=PreparedStalkTransport(source.archive,AXES,((),((),())),((0,0,0),(1,0,0)),
        parameter_space=dict(name='declared',keys=('shared',)),parameter_basis=((0,0,2),(1,0,-1)))
    native=PreparedStalkParameterRelations(transport,(('entry',((0,1),),0),))
    result=native.restrict_fields((observation(target=4),))
    assert result['family']['particular']==(Q(2),) and result['family']['kernel_shape']==(1,0)
    assert result['status']=='open'


def test_whole_stalk_observation_includes_every_axis_without_scalar_projection():
    source,state,views=learn();output=infer_model(state,source,query(state,views))
    field=((0,0),(0,0),(Q(2,3),Q(-3,7),0))
    target=((0,0),)*4+((Q(149,105),Q(-10,7)),(0,0))
    observation=dict(input=field,target=target,observed=(False,)*4+(True,False))
    result=stalk_context_rule_restriction(output,(observation,))
    assert result['status']=='forced' and result['additional_rank']==0
    assert result['constraints']==((0,4,0),(0,4,1))
    changed=dict(observation,target=((0,0),)*4+((Q(149,105),0),(0,0)))
    rejected=stalk_context_rule_restriction(output,(changed,))
    assert rejected['status']=='excluded' and rejected['contradiction_value']!=0


def test_context_restriction_roundtrips_and_rejects_changed_receipts():
    source,state,views=learn();output=infer_model(state,source,query(state,views))
    sample=dict(input=((0,0),(0,0),(0,0,1)),target=((0,0),)*4+((1,0),(0,0)),observed=(False,)*4+(True,False))
    result=stalk_context_rule_restriction(output,(sample,))
    assert result['status']=='open' and result['family']['kernel_shape'][1]==2
    restored=unpack_value(pack_value(thaw_tree(result)))
    assert validate_stalk_context_restriction(output,restored)
    for key,value in (('status','forced'),('model_digest','0'*64),('constraints',()),('contradiction',(Q(1),))):
        bad=thaw_tree(restored);bad[key]=value
        with pytest.raises(ValueError,match='replay'):validate_stalk_context_restriction(output,bad)


def test_concurrent_restrictions_keep_detached_original_owners():
    native=owner();expected=native.restrict_fields((observation(),))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(r==expected for r in pool.map(lambda _:native.restrict_fields((observation(),)),range(8)))
    expected['family']['particular']=(99,99)
    assert native.restrict_fields((observation(),))['family']['particular']==(Q(2),Q(1))


@pytest.mark.parametrize('sample',[
    {},dict(input=((1,),),target=((2,),),observed=(True,)),
    dict(input=((1,),(0,)),target=((2,),),observed=(1,)),
    dict(input=((1,),(0,)),target=((2,),),observed=(False,)),
    dict(input=((1.0,),(0,)),target=((2,),),observed=(True,)),
    dict(input=((1,),(0,)),target=((True,),),observed=(True,)),
    dict(input=((1,),(0,)),target=((2,3),),observed=(True,)),
])
def test_malformed_fields_reject_without_changing_native_evidence(sample):
    native=owner();before=native.receipt
    with pytest.raises((ValueError,TypeError,RuntimeError,OverflowError)):native.restrict_fields((sample,))
    assert native.receipt==before


@pytest.mark.parametrize('selected',['observed',('missing',),('observed','observed'),(True,)])
def test_malformed_selected_equations_reject(selected):
    with pytest.raises((ValueError,TypeError)):owner().restrict_fields((observation(),),selected)


def test_empty_and_uninitialized_restrictions_reject():
    with pytest.raises((ValueError,RuntimeError)):owner().restrict_fields(())
    empty=PreparedStalkParameterRelations.__new__(PreparedStalkParameterRelations)
    with pytest.raises(RuntimeError):empty.restrict_fields((observation(),))


def test_observed_input_spans_transfer_only_the_independently_measured_direction():
    source,state,views=learn(partial=True)
    full=infer_model(state,source,query(state,views))
    assert not stalk_context_hypotheses(full)['selected_blocks']
    declared=stalk_context_view_rules_query(state,(0,1),(2,),views,minimum_peers=1,observed_inputs=True)
    output=infer_model(state,source,declared);acquisition=stalk_context_hypotheses(output)
    assert acquisition['selected_blocks']==((1,2),) and acquisition['input_scope']=='exact-observed-field-spans'
    assert acquisition['interpretation']=='learned-observed-context-view-rules-v1'
    known=stalk_context_rule_response(output,((0,0),(0,0),(1,0,0)))
    assert len(known['boundary_family'])==1 and known['boundary_family'][0][4:]==((Q(2),Q(3,7)),(Q(4),Q(6,7)))
    for field in ((0,1,0),(0,0,1)):
        assert len(stalk_context_rule_response(output,((0,0),(0,0),field))['boundary_family'])>1
    assert state.configuration['stalk_axes'][1][2]==('a','b','context')
    assert known['transfer_scope']=='conditional-on-covered-observed-field-invariance'


@pytest.mark.parametrize('scope',[1,0,None,'observed',Q(1)])
def test_observed_input_scope_requires_an_explicit_boolean(scope):
    source,state,views=learn()
    with pytest.raises(TypeError):stalk_context_view_rules_query(state,(0,1),(2,),views,observed_inputs=scope)


def test_native_work_admission_and_failed_calls_preserve_the_family():
    source,_=model()
    transport=PreparedStalkTransport(source.archive,AXES,((),((),())),((0,0,0),(1,0,0)),
        limits=(4096,16000000,2000,16384))
    native=PreparedStalkParameterRelations(transport,(('unrestricted',(),0),),allow_constants=True)
    before=native.receipt
    with pytest.raises((ValueError,RuntimeError,OverflowError)):native.restrict_fields((observation(),)*1000)
    assert native.receipt==before and native.restrict_fields((observation(),))['status']=='open'


def test_unobserved_nonzero_stalk_data_rejects_even_with_another_observed_stalk():
    source,state,views=learn();output=infer_model(state,source,query(state,views))
    sample=dict(input=((0,0),(0,0),(0,0,1)),target=((1,0),)+((0,0),)*5,
        observed=(False,)*4+(True,False))
    with pytest.raises(ValueError,match='unobserved'):stalk_context_rule_restriction(output,(sample,))


def test_prepared_executor_owns_the_verified_native_family_without_repeated_replay(monkeypatch):
    import rexgraph.cpp_stalk_parameter_relations as module
    source,state,views=learn();output=infer_model(state,source,query(state,views))
    executor=PreparedStalkContextRules(output);field=((0,0),(0,0),(1,0,0))
    sample=dict(input=field,target=((0,0),)*4+((2,Q(3,7)),(0,0)),observed=(False,)*4+(True,False))
    expected=stalk_context_rule_response(output,field)
    restriction=stalk_context_rule_restriction(output,(sample,))
    digest=output.coefficient_digest
    del source,state,output
    def forbidden(*args,**kwargs):raise AssertionError('repeated model replay')
    monkeypatch.setattr(module,'validate_stalk_parameter_relations_output',forbidden)
    with module._lock:module._cache.clear()
    assert executor.apply(field)==expected and executor.restrict_fields((sample,))==restriction
    assert executor.receipt['output_digest']==digest
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(r==restriction for r in pool.map(lambda _:executor.restrict_fields((sample,)),range(8)))
    with pytest.raises(FrozenInstanceError):executor._selected=()
    with pytest.raises(RuntimeError):executor.__init__(None)
    with pytest.raises(TypeError):executor.receipt['selected']=()


def test_prepared_executor_rejects_uninitialized_and_wrong_model_owners():
    with pytest.raises(TypeError):PreparedStalkContextRules(None)
    _,state=model()
    with pytest.raises(TypeError):PreparedStalkContextRules(state)
    empty=PreparedStalkContextRules.__new__(PreparedStalkContextRules)
    with pytest.raises(RuntimeError):empty.apply(())
    with pytest.raises(RuntimeError):empty.restrict_fields(())
    with pytest.raises(RuntimeError):_ = empty.receipt

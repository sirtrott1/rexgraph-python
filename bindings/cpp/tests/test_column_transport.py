"""Original column acquisition, exact affine families and native checkpoint replay."""
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys
import numpy as np
import pytest
from rexgraph_cpp import NativeSource,PreparedStalkTransport,encode_stalk_column_tensor
from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model,stalk_transport_query,_cache,replay_stalk_transport_model
from rexgraph.cpp_column_transport import column_transport_observation,column_transport_observation_group
from rexgraph.model_runtime import train_model,infer_model
from rexgraph.model_codec import pack_model,unpack_model
from rexgraph.io.model_record import model_record,save_model
from rexgraph.model_state import thaw_tree
from test_stalk_transport import AXES,SLOTS,TRUTH,declaration,truth,acquire


def models():
    source=NativeSource.from_declaration(declaration())
    maps=((),((),()))
    return source,cpp_stalk_transport_model(source,AXES,maps,SLOTS,(0,1)),cpp_stalk_transport_model(source,AXES,maps,SLOTS,(0,1),column_fields=True)


def learn(source,state,fields):
    packets=[]
    for field in fields:
        targets=truth(field)
        packets.append(column_transport_observation(state,{0:tuple((i,q) for i,q in enumerate(field[0]) if q)},
            {cell:tuple((i,q) for i,q in enumerate(values) if q) for cell,values in enumerate(targets)}))
    batch=column_transport_observation_group(state,packets)
    return train_model(state,source,batch,steps=len(fields)),batch


def test_column_fit_matches_complete_dense_joint_family_and_original_flags():
    source,dense,sparse=models();fields=(((1,0,0),),((0,1,0),),((0,0,1),))
    for field in fields:dense=acquire(dense,source,field)
    sparse,batch=learn(source,sparse,fields)
    for key in ('particular','kernel','kernel_shape','rank','pivots','free_columns','design','target','constraints','occurrence_keys','occurrence_coefficients','occurrence_activity'):
        assert sparse.payload['fit'][key]==dense.payload['fit'][key]
    assert sparse.payload['fit']['particular']==TRUTH and sparse.adapter_version=='3'
    assert batch.targets['profile']=='encoded-original-stalk-targets-v1' and len(batch.targets['stalks'])==2
    assert sum(o['batch_checkpoint'] is not None for o in sparse.payload['observations'])==1
    assert all(type(tensor) is bytes for o in sparse.payload['observations'] for fields in (o['input'],o['target']) for _,tensor in fields)
    query=stalk_transport_query(sparse,((Q(7,3),-2,Q(5,7)),))
    result=infer_model(sparse,source,query)
    assert result.execution_receipt['response']['boundary_family']==(truth(query['field']),)
    unpack_model(pack_model(result)).check_state()
    _cache.clear();restored=unpack_model(pack_model(sparse));result2=infer_model(restored,source,query)
    assert result2.execution_receipt['response']==result.execution_receipt['response']


def test_partial_learning_preserves_unknown_directions_and_observed_zero_stalks():
    source,_,sparse=models();sparse,_=learn(source,sparse,(((1,0,0),),))
    assert sparse.payload['fit']['kernel_shape']==(9,6)
    missing=infer_model(sparse,source,stalk_transport_query(sparse,((0,1,0),)))
    assert missing.execution_receipt['boundary_identified']==((False,False),(False,))
    zero=column_transport_observation(sparse,{}, {0:(),1:()})
    child=train_model(sparse,source,column_transport_observation_group(sparse,(zero,)),steps=1)
    assert child.payload['fit']['kernel']==sparse.payload['fit']['kernel']
    assert child.payload['observations'][-1]['target'] and sparse.step==1
    _cache.clear();infer_model(unpack_model(pack_model(child)),source,stalk_transport_query(child))
    replayed=replay_stalk_transport_model(child,source)
    assert replayed.payload['fit']['particular']==child.payload['fit']['particular']


@pytest.mark.parametrize('kind',['tensor','target','source','order','checkpoint'])
def test_foreign_mutated_or_incomplete_column_witnesses_are_rejected(kind):
    source,_,sparse=models();state,batch=learn(source,sparse,(((1,0,0),),))
    if kind=='checkpoint':
        payload=thaw_tree(state.payload);payload['observations'][0]['batch_sample']=1
        with pytest.raises(ValueError):replace(state,payload=payload)
        return
    inputs=thaw_tree(batch.inputs);targets=thaw_tree(batch.targets)
    if kind=='tensor':
        values=list(inputs['observations'][0]['input']);values[0]=(0,values[0][1]+b'x');inputs['observations'][0]['input']=tuple(values)
    elif kind=='target':
        rows=list(targets['stalks']);rows[0]=(rows[0][0],((0,encode_stalk_column_tensor(2,())),));targets['stalks']=tuple(rows)
    elif kind=='source':inputs['source_reference']={'foreign':True}
    else:inputs['observations'][0]['input']=inputs['observations'][0]['input']*2
    changed=replace(batch,inputs=inputs,targets=targets)
    with pytest.raises((ValueError,RuntimeError,KeyError)):train_model(sparse,source,changed,steps=1)
    assert sparse.step==0


def test_encoded_tensor_uses_original_stalk_shape_and_canonical_sparse_coordinates():
    source,_,state=models()
    for entries in (((1,1),(0,1)),((0,1),(0,1)),((0,0),),((3,1),)):
        with pytest.raises(ValueError):encode_stalk_column_tensor(3,entries)
    wrong=column_transport_observation(state,{0:encode_stalk_column_tensor(2,((0,1),))},{0:()})
    with pytest.raises(ValueError):train_model(state,source,column_transport_observation_group(state,(wrong,)),steps=1)
    fields=PreparedStalkTransport(source.archive,AXES,((),((),())),SLOTS,column_fields=True)
    partial=bytearray(encode_stalk_column_tensor(3,((0,1),)))
    with pytest.raises(TypeError):fields.append_columns((dict(input=((0,partial),),target=((0,encode_stalk_column_tensor(2,())),)),))


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_column_model_and_generation_inputs_roundtrip_in_cold_process(tmp_path,suffix):
    source,_,sparse=models();sparse,_=learn(source,sparse,(((1,0,0),),((0,1,0),),((0,0,1),)))
    query=stalk_transport_query(sparse,((2,-3,5),));record=model_record(source,sparse);record.attach_metadata(1,0,'query',query)
    path=tmp_path/('columns'+suffix);save_model(path,record)
    script="""
import sys
from rexgraph.io.model_record import load_model,read_model_record
from rexgraph.model_runtime import infer_model
record=load_model(sys.argv[1]);state,source=read_model_record(record)
result=infer_model(state,source,record.get_metadata(1,0,'query'));result.check_state()
assert state.adapter_version=='3' and state.step==3 and state.payload['fit']['rank']==9
assert sum(o['batch_checkpoint'] is not None for o in state.payload['observations'])==1
print('cold column model passed')
"""
    completed=subprocess.run([sys.executable,'-c',script,str(path)],cwd='/tmp',text=True,capture_output=True)
    assert completed.returncode==0,completed.stdout+completed.stderr


def test_encoded_target_contract_binds_source_rows_samples_and_provider():
    source,dense,sparse=models();_,batch=learn(source,sparse,(((1,0,0),),))
    with pytest.raises(ValueError,match='encoded target profile'):train_model(dense,source,batch)
    for field in ('space','source_reference','samples','stalks'):
        targets=thaw_tree(batch.targets)
        if field=='space':targets[field]=('foreign',dense.space.keys)
        elif field=='source_reference':targets[field]={'foreign':True}
        elif field=='samples':targets[field]=0
        else:targets[field]=targets[field]*2
        with pytest.raises(ValueError):replace(batch,targets=targets)
    with pytest.raises(ValueError):replace(batch,observed=np.zeros(len(batch.space.keys),dtype=bool))
    from system.serialize import json_value
    summary=json_value(batch)
    assert summary['targets']['storage']=='encoded-original-stalk-targets-v1'
    assert summary['targets']['samples']==1 and summary['targets']['tensors']==2


def test_rcql_trains_encoded_column_groups_and_reports_exact_parameters():
    from rcql import Executor,parse
    from system.serialize import json_value
    source,_,state=models();_,batch=learn(source,state,(((1,0,0),),))
    query=stalk_transport_query(state,((1,0,0),));record=model_record(source,state)
    result=Executor(sources={'r':record},params={'b':batch,'q':query}).execute(parse(
        'FROM $r RETURN MODEL_INFER(MODEL_TRAIN(MODEL_STATE(),$b),$q)'))
    assert result.values[0].execution_receipt['response']['boundary_family']==(truth(query['field']),)
    assert result.exactness[0].value=='rational'
    assert json_value(state)['observation_group_profile']=='column-stalk-transport-observation-group-v1'

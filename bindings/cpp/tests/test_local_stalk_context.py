"""Local context compilation and detached exact pattern serving."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
import subprocess
import sys

import pytest

from rexgraph.cpp_column_transport import column_transport_observation, column_transport_observation_group
from rexgraph.cpp_stalk_pattern import compile_stalk_context_pattern, compile_stalk_context_pattern_bank
from rexgraph.cpp_stalk_pattern_model import cpp_stalk_pattern_bank_model, stalk_pattern_bank_sparse_query, stalk_pattern_bank_result
from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model, _prepare
from rexgraph.cpp_stalk_parameter_relations import stalk_context_view_rules_query, stalk_context_hypotheses
from rexgraph.io.model_record import model_record, save_model
from rexgraph.model_runtime import infer_model, train_model
from rexgraph_cpp import NativeSource, PreparedStalkTransportPatternBank
from test_stalk_context_view_rules import learn, query


def column_model(**kwargs):
    source,dense,views=learn(**kwargs);c=dense.configuration
    state=cpp_stalk_transport_model(source,c['stalk_axes'],c['fixed_maps'],c['parameter_slots'],c['training_rows'],column_fields=True)
    packets=[]
    for record in dense.payload['observations']:
        inputs={cell:tuple((i,q) for i,q in enumerate(field) if q)
                for cell,field in enumerate(record['input']) if any(field)}
        targets={cell:tuple((i,q) for i,q in enumerate(record['target'][cell]) if q)
                 for cell,observed in enumerate(record['observed']) if observed}
        packets.append(column_transport_observation(state,inputs,targets))
    state=train_model(state,source,column_transport_observation_group(state,packets),steps=len(packets))
    return source,dense,state,views


@pytest.mark.parametrize('redundant',[False,True])
@pytest.mark.parametrize('observed',[False,True])
def test_native_context_queries_and_exports_never_expand_column_history(redundant,observed):
    source,dense,state,views=column_model(redundant=redundant);owner=_prepare(state,source)
    q=stalk_context_view_rules_query(state,(0,1),(2,),views,minimum_peers=1,observed_inputs=observed)
    legacy=stalk_context_view_rules_query(dense,(0,1),(2,),views,minimum_peers=1,observed_inputs=observed)
    a=stalk_context_hypotheses(infer_model(dense,source,legacy));b=stalk_context_hypotheses(infer_model(state,source,q))
    assert a['selected_blocks']==b['selected_blocks'] and a['interpretation']==b['interpretation']
    result=owner.contextual_view_rules((0,1),(2,),views,minimum_peers=1,observed_inputs=observed)[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        repeated=tuple(pool.map(lambda _:owner.contextual_view_rules((0,1),(2,),views,minimum_peers=1,observed_inputs=observed)[0],range(8)))
    assert all(item==result for item in repeated)
    single=compile_stalk_context_pattern(state,source,q,0,model_namespace=905,coordinate_ids=(100,101,102))
    bank=compile_stalk_context_pattern_bank(state,source,q,(0,),model_namespace=906,
        coordinate_ids={'weights':101,'parameters':102,'domains':(100,)})
    inspected=bank.inspect()
    assert len(inspected['pattern_keys'])==1
    assert owner.execution_stats['column_observations']==state.step
    assert owner.execution_stats['materialized_field_entries']==0
    assert single.inspect()['observations']==state.step and inspected['observations']==state.step


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_one_star_cold_serving_reads_only_compact_source_and_local_sparse_input(tmp_path,suffix):
    source,_,state,views=column_model();q=query(state,views)
    bank=compile_stalk_context_pattern_bank(state,source,q,(0,),model_namespace=906,
        coordinate_ids={'weights':101,'parameters':102,'domains':(100,)})
    image=bank.archive();local=NativeSource(image);serving=cpp_stalk_pattern_bank_model(local)
    request=stalk_pattern_bank_sparse_query(serving,{0:{'profile':'complete-sparse-stalk-field-v1','frame':2,'size':4,'entries':((0,Q(2,3)),(1,Q(-3,7)))}})
    output=infer_model(serving,local,request);expected=stalk_pattern_bank_result(output)
    assert serving.payload=={} and serving.step==0
    assert len(local.identity['coordinates'][1])==1
    assert len(local.identity['coordinates'][0])==2
    assert _prepare(state,source).execution_stats['materialized_field_entries']==0
    record=model_record(local,serving);record.attach_metadata(1,0,'query',request)
    path=tmp_path/('one-star'+suffix);save_model(path,record)
    script='''
import sys
from rexgraph.io.model_record import load_model,read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_stalk_pattern_model import stalk_pattern_bank_result
import rexgraph.cpp_stalk_transport as training
def forbidden(*a,**k):raise AssertionError('training source or evidence loaded for local inference')
training._prepare=forbidden
record=load_model(sys.argv[1]);state,source=read_model_record(record)
assert state.payload=={} and state.step==0 and len(source.identity['coordinates'][1])==1
output=infer_model(state,source,record.get_metadata(1,0,'query'))
print(output.coefficient_digest)
'''
    completed=subprocess.run([sys.executable,'-c',script,str(path)],cwd='/tmp',text=True,capture_output=True)
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert completed.stdout.strip()==output.coefficient_digest
    loaded=PreparedStalkTransportPatternBank(image,model_namespace=906)
    assert loaded.response_sparse({0:request['inputs'][0][1]})==bank.response_sparse({0:request['inputs'][0][1]})
    assert expected


def test_local_pattern_readout_is_integrated_with_rcql():
    from rcql import Executor,parse
    source,_,state,views=column_model();bank=compile_stalk_context_pattern_bank(state,source,query(state,views),(0,),
        model_namespace=906,coordinate_ids={'weights':101,'parameters':102,'domains':(100,)})
    local=NativeSource(bank.archive());serving=cpp_stalk_pattern_bank_model(local)
    q=stalk_pattern_bank_sparse_query(serving,{0:{'profile':'complete-sparse-stalk-field-v1','frame':2,'size':4,'entries':((0,1),)}})
    result=Executor(sources={'r':model_record(local,serving)},params={'q':q}).execute(parse('FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q)'))
    assert result.values[0].coefficient_digest==infer_model(serving,local,q).coefficient_digest

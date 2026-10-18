"""Compact serving on observed spans of complete named carriers."""
from fractions import Fraction as Q
from hashlib import sha256
import subprocess
import sys

import pytest

from rexgraph.cpp_stalk_parameter_relations import stalk_context_view_rules_query
from rexgraph.cpp_stalk_pattern import compile_stalk_context_pattern, compile_stalk_context_pattern_bank
from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model, stalk_transport_observation_batch, _prepare
from rexgraph.model_runtime import train_model
from rexgraph_cpp import PreparedStalkTransportPatternBank
from test_stalk_context_view_rules import learn
from test_stalk_pattern_bank import correlated_bank


def observed_model(*, repetitions=1, conflict=False, partial=False):
    source, old, views = learn(redundant=True)
    c = old.configuration
    state = cpp_stalk_transport_model(source, c['stalk_axes'], c['fixed_maps'], c['parameter_slots'], tuple(range(6)))
    for _ in range(repetitions):
        for row in (0, 1):
            field = tuple((1,1) if i == row else (0,)*len(axes) for i,axes in enumerate(c['stalk_axes'][1]))
            value = (Q(9,5)+int(conflict and row == 1), Q(31,7))
            target = ((2*row,value),) if partial else ((2*row,value),(2*row+1,tuple(2*v for v in value)))
            state = train_model(state, source, stalk_transport_observation_batch(state,field,target))
    query = stalk_context_view_rules_query(state,(0,1),(2,),views,minimum_peers=1,observed_inputs=True)
    return source,state,query


def compile_observed(bank=False, **options):
    source,state,query = observed_model(**options)
    if bank:
        result = compile_stalk_context_pattern_bank(state,source,query,(0,1),model_namespace=907,
            coordinate_ids={'weights':101,'parameters':102,'domains':(100,103)})
    else:
        result = compile_stalk_context_pattern(state,source,query,0,model_namespace=906,coordinate_ids=(100,101,102))
    return source,state,result


@pytest.mark.parametrize('bank',[False,True])
def test_observed_span_is_explicit_and_complete_carriers_survive(bank):
    source,state,model = compile_observed(bank)
    info = model.inspect()
    images = info['input_images'] if bank else (info['input_image'],)
    assert all(image == {'shape':(4,1),'entries':((0,0,Q(1)),(1,0,Q(1)),(3,0,Q(1)))} for image in images)
    assert dict((name,axes) for _,name,axes in info['common_frames'])['clause'] == ('a','b','context','redundant')
    assert info['parameter_directions']['shape'][1] == 0 and all(info['identified'])
    assert info['source_sha256'] == sha256(source.archive).hexdigest()
    assert info['evidence_sha256'] == state.coefficient_digest
    assert state.payload['fit']['kernel_shape'][1] > 0
    x = (Q(2,3),Q(2,3),0,Q(2,3))
    result = model.response({0:x} if bank else x)
    assert result['identified']
    expected = (Q(6,5),Q(62,21),Q(12,5),Q(124,21))
    assert result['origin'][:4] == expected
    assert not any(result['origin'][4:])
    for x in ((1,0,0,1),(0,0,1,0),(1,1,0,0)):
        with pytest.raises(ValueError,match='outside'):
            model.response({0:x} if bank else x)


@pytest.mark.parametrize('bank',[False,True])
def test_observed_mode_does_not_change_complete_carrier_admission(bank):
    source,state,q = observed_model()
    query = stalk_context_view_rules_query(state,q['training'],q['targets'],q['frames'],minimum_peers=1)
    with pytest.raises(ValueError,match='admitted'):
        if bank:
            compile_stalk_context_pattern_bank(state,source,query,(0,1),model_namespace=907,
                coordinate_ids={'weights':101,'parameters':102,'domains':(100,103)})
        else:
            compile_stalk_context_pattern(state,source,query,0,model_namespace=906,coordinate_ids=(100,101,102))


@pytest.mark.parametrize('bank',[False,True])
@pytest.mark.parametrize('options',[{'conflict':True},{'partial':True}])
def test_open_or_conflicting_peers_do_not_export(options,bank):
    with pytest.raises(ValueError,match='admitted'):
        compile_observed(bank,**options)


@pytest.mark.parametrize('bank',[False,True])
def test_serving_size_does_not_grow_with_repeated_fields(bank):
    _,_,first = compile_observed(bank)
    _,state,larger = compile_observed(bank,repetitions=30)
    assert larger.inspect()['observations'] == 60
    assert larger.inspect()['weight_origin'] == first.inspect()['weight_origin']
    assert larger.inspect()['parameter_directions'] == first.inspect()['parameter_directions']
    assert len(larger.archive())-larger.inspect()['provenance_bytes'] == len(first.archive())-first.inspect()['provenance_bytes']
    assert len(state.payload['observations']) == 60


@pytest.mark.parametrize('bank',[False,True])
def test_cold_native_serving_requires_no_training_checkpoint(tmp_path,bank):
    _,_,model = compile_observed(bank)
    path = tmp_path/'observed.rfcxs';path.write_bytes(model.archive())
    script = '''
from pathlib import Path
from fractions import Fraction as Q
from rexgraph_cpp import PreparedStalkTransportPattern, PreparedStalkTransportPatternBank
import sys
bank = sys.argv[2]=='True'
cls = PreparedStalkTransportPatternBank if bank else PreparedStalkTransportPattern
p = cls(Path(sys.argv[1]).read_bytes(),model_namespace=907 if bank else 906)
x = (Q(2,3),Q(2,3),0,Q(2,3))
r = p.response({0:x} if bank else x)
assert r['identified'] and r['origin'][:4] == (Q(6,5),Q(62,21),Q(12,5),Q(124,21))
assert p.archive()==Path(sys.argv[1]).read_bytes()
try:p.response({0:(1,0,0,1)} if bank else (1,0,0,1))
except ValueError:pass
else:raise AssertionError('unobserved input accepted')
assert 'rexgraph.model_runtime' not in sys.modules and 'torch' not in sys.modules
'''
    child = subprocess.run([sys.executable,'-I','-c',script,str(path),str(bank)],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert child.returncode == 0,child.stderr


@pytest.mark.parametrize('value',[None,0,1,'observed',1.0])
@pytest.mark.parametrize('bank',[False,True])
def test_observed_selection_requires_a_boolean(value,bank):
    source,state,q = observed_model();owner = _prepare(state,source)
    with pytest.raises(TypeError,match='boolean'):
        if bank:
            owner.learned_pattern_bank(q['training'],(0,1),q['frames'],model_namespace=907,
                coordinate_ids={'weights':101,'parameters':102,'domains':(100,103)},
                evidence_digest=state.coefficient_digest,minimum_peers=1,observed_inputs=value)
        else:
            owner.learned_pattern(q['training'],0,q['frames'],model_namespace=906,
                coordinate_ids=(100,101,102),evidence_digest=state.coefficient_digest,minimum_peers=1,observed_inputs=value)


def test_observable_bank_keeps_joint_unknowns_and_exact_gluing():
    source,old,_ = correlated_bank()
    axes = ((('value',),),(('signal','context'),)*6)
    state = cpp_stalk_transport_model(source,axes,((),((),)*6),tuple((p,0,j) for p in range(6) for j in range(2)),(0,))
    for sample in old.payload['observations']:
        field = tuple((value[0],value[0]) for value in sample['input'])
        state = train_model(state,source,stalk_transport_observation_batch(state,field,((0,sample['target'][0]),)))
    identity = ((0,0,1),)
    two = ((0,0,1),(1,1,1))
    views = ((('reading',('value',),identity,identity),), (('context',('signal','context'),two,two),)*6)
    query = stalk_context_view_rules_query(state,tuple(range(6)),(),views,minimum_peers=2,observed_inputs=True)
    bank = compile_stalk_context_pattern_bank(state,source,query,(0,1,3),model_namespace=908,
        coordinate_ids={'weights':101,'parameters':102,'domains':(100,103,104)})
    assert bank.inspect()['parameter_directions']['shape'] == (3,1)
    assert bank.response({2:(1,1)})['origin'] == (Q(2),)
    assert not bank.response({0:(1,1)})['identified']
    assert bank.response({0:(1,1),1:(-1,-1)})['identified']
    glued = bank.glue({0:(1,1),2:(1,1)})
    assert glued['section']['origin'] == (Q(2),)
    assert glued['boundary_response']['origin'] == (Q(4),)
    loaded = PreparedStalkTransportPatternBank(bank.archive(),model_namespace=908)
    assert loaded.inspect() == bank.inspect()
    assert loaded.response({0:(1,1),1:(-1,-1)})['identified']

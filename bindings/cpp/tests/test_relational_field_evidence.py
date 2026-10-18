"""Installed whole field updates, independent source uses and correlated families."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from hashlib import sha256
import subprocess
import sys

import pytest
from rexgraph_cpp import (NativeInputUseAsset, NativeInputAccession,
                         PreparedFieldEvidence, NativeAffineFieldResult, NativeAffineEvaluation, NativeFieldResidual, NativeOccurrenceMapFamily)
from rexgraph.cpp_relational_field_learning import (cpp_field_learning_model, field_program_batch, field_learning_query,
    read_field_learning_result, evaluate_field_program, inspect_field_learning_model, clear_field_learning_cache)
from rexgraph.cpp_relational_field_program import read_field_program_result
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.value_codec import pack_value
from rexgraph.io.model_record import model_record, read_model_record, save_model, load_model
from test_relational_field_program import source, declaration, matrix, tensor, action, MAPS, COEFFICIENTS, LOWER, UPPER, LATER
from test_relational_field_interface import accession_declaration


HELD_INPUT = (Q(2), Q(3), Q(5), Q(7))


class Reader:
    def __init__(self, image): self.image, self.offset = image, 0
    def take(self, n):
        value = self.image[self.offset:self.offset+n]; self.offset += n; assert len(value) == n; return value
    def u64(self): return int.from_bytes(self.take(8), 'little')
    def integer(self):
        negative = self.take(1)[0]; size = int.from_bytes(self.take(4), 'little')
        value = int.from_bytes(self.take(size), 'big'); return -value if negative else value
    def rational(self): return Q(self.integer(), self.integer())


def read_matrix(image):
    r = Reader(image); rows, cols, count = r.u64(), r.u64(), r.u64(); out = [[Q(0)]*cols for _ in range(rows)]
    for _ in range(count):
        i, j = r.u64(), r.u64(); out[i][j] = r.rational()
    assert r.offset == len(image)
    return tuple(tuple(row) for row in out)


def read_vector(image):
    r = Reader(image); values = tuple(r.rational() for _ in range(r.u64())); assert r.offset == len(image); return values


def learning_declaration():
    value = deepcopy(declaration()); value['nodes'] = value['nodes'][:3]
    value['occurrences'][7]['role'] = 'coefficient_independent'
    value['occurrences'][7]['map'] = matrix(tuple(tuple(-q for q in row) for row in MAPS[7]))
    return value


def parameters():
    coordinates = ((0,0,0),(0,0,1),(0,1,0),(0,1,1),
                   (6,0,0),(6,0,1),(6,1,0),(6,1,1),(6,2,0),(6,2,1),(3,0,0),(4,0,0))
    columns = (0,1,1,2,2,0,0,1,1,2,3,4)
    return {'profile': 'rexgraph.relational-field-parameters.v1', 'key': 6101, 'revision': 1, 'node': 2,
            'variables': coordinates, 'parameters': (7001,7002,7003,7004,7005),
            'basis': {'shape': (12,5), 'entries': tuple((i,j,Q(1)) for i,j in enumerate(columns))}}


def actual_asset(namespace, use, raw):
    original = source(namespace); spec = {'profile':'rexgraph.relational-input-use-asset.v1','key':namespace+1,'channel':namespace+2,
        'encoding':'UTF-8','bytes':raw,'uses':({'key':use,'cells':((1,100),(1,101)),
            'occurrences':tuple((1,i) for i in range(1,9)), 'spans':((0,len(raw)),(0,0),(0,min(3,len(raw))))},)}
    return NativeInputUseAsset(original.archive,spec)


def input_accession():
    d = accession_declaration(); d['stalks'] = (
        {'cell':(1,0),'basis':4301,'directions':(4311,4312),'metric':matrix(((3,1),(1,2)))},
        {'cell':(1,1),'basis':4401,'directions':(4411,4412),'metric':matrix(((1,0),(0,1)))})
    identity = matrix(tuple(tuple(int(i==j) for j in range(4)) for i in range(4)))
    d['accession'] = identity; d['reconstruction'] = identity; d['isometric'] = True
    return d


def input_row(asset, use, values):
    return {'asset':asset.archive(),'accession':input_accession(),'use':use,'field':tensor(values)}


def native_input(evidence, asset, use, values):
    program = evidence.program(); accession = NativeInputAccession(asset,program,input_accession())
    return program.execute((accession.admit(use,tensor(values)),))


def reference(values, theta):
    a,b,c,d,e = theta; maps = [[list(map(Q,row)) for row in m] for m in MAPS]
    for i,j,v in ((0,0,a),(0,1,b),(1,0,b),(1,1,c)): maps[0][i][j] += v
    for i,j,v in ((0,0,c),(0,1,a),(1,0,a),(1,1,b),(2,0,b),(2,1,c)): maps[6][i][j] += v
    maps[3][0][0] += d; maps[4][0][0] += e
    lower = [[Q(0)]*n for n in (2,3,2)]; flags = []
    for slot,m in enumerate(maps):
        contribution = tuple(COEFFICIENTS[slot]*q for q in action(m,values[2*UPPER[slot]:2*UPPER[slot]+2])); flags.extend(contribution)
        for i,q in enumerate(contribution): lower[LOWER[slot]][i] += q
    field = tuple(q for row in lower for q in row)
    return field,tuple(flags),action(LATER,field)


def observation(key, asset, use, values, target, mask=((0,0),(0,1),(0,2))):
    measurement = actual_asset(910000+key,9000+key,b'observed complete field')
    return {'key':key,'inputs':(input_row(asset,use,values),),'target_asset':measurement.archive(),
            'target_use':9000+key,'target':tensor(target),'observed':mask}


@pytest.fixture
def model():
    owner = source(); state = cpp_field_learning_model(owner,learning_declaration(),parameters=parameters())
    return owner,state


@pytest.mark.parametrize('trial',range(16))
def test_learned_maps_change_all_complete_held_intermediates_before_decoding(model,trial):
    owner,parent = model
    held_asset = actual_asset(999991,801,b'independent held input')
    before = infer_model(parent,owner,field_learning_query(parent,(input_row(held_asset,801,HELD_INPUT),)))
    theta = (Q(trial-4,3),Q(trial%7-2,5),Q(trial%13+1,7),Q(0),Q(0))
    training = actual_asset(900111,111,b'first source use'); values = (Q(1),Q(2),Q(3),Q(4))
    target = reference(values,theta)[0]
    batch = field_program_batch(parent,(observation(1,training,111,values,target),))
    child = train_model(parent,owner,batch)
    assert child.parent == parent.coefficient_digest and child.step == 1
    assert child.source.matches(parent.source) and child.space == parent.space and child.output_axes == parent.output_axes
    query = field_learning_query(child,(input_row(held_asset,801,HELD_INPUT),)); output = infer_model(child,owner,query)
    complete = read_field_learning_result(output,state=child); lower,flags,later = reference(HELD_INPUT,theta)
    for node,expected in ((1,HELD_INPUT),(2,lower),(3,later)):
        field = complete.field_archive(node); assert field['particular'] == tensor(expected)
        assert field['supported'] == len(expected) and field['free_parameter_count'] == 2
        assert field['direction_nonzeros'] == 0
    assert complete.field_archive(2,occurrences=True)['particular'] == tensor(flags)
    assert read_field_learning_result(before).field_archive(2)['particular'] != complete.field_archive(2)['particular']
    assert len(output.dependencies) == 3 and output.dependencies[-1].native.namespace == 999991
    evidence = complete.evidence(); info = evidence.inspect()
    assert info['free_parameters'] == (7004,7005) and read_vector(info['particular']) == theta
    assert evidence.observation(0)['target'] == tensor(target)
    assert evidence.residual(0).field_archive()['particular'] == tensor((0,)*7)
    inactive = evidence.occurrence_map(1,3).inspect(); zero = evidence.occurrence_map(1,4).inspect()
    assert read_matrix(inactive['directions'][0])[0][0] == 1 and read_matrix(zero['directions'][1])[0][0] == 1
    assert NativeAffineFieldResult.from_archive(complete.archive()).archive() == complete.archive()
    assert complete.input_uses()[0]['instances']['occurrences'][1] == tuple(range(8))
    assert complete.input_uses()[0]['use'] == 801
    selected = complete.select(2,((0,0),(0,1)),coordinate_limit=5)
    assert selected[0]['particular']['values'] == lower[:2] and all(selected[0]['identified'])


def test_partial_whole_stalk_evidence_keeps_correlated_unknowns_complete_residual_and_parent_map(model):
    owner,parent = model; theta = (Q(2),Q(-1),Q(3),Q(0),Q(0)); first = actual_asset(900111,111,b'first input')
    values = (Q(1),Q(2),Q(3),Q(4)); target = reference(values,theta)[0]
    child = train_model(parent,owner,field_program_batch(parent,(observation(1,first,111,values,target,((0,0),)),)))
    native = inspect_field_learning_model(child,owner); info = native.inspect()
    assert native.identity()['free_parameter_count'] == 3
    assert native.observation(0)['target'] == tensor(target)
    residual = native.residual(0).field_archive(); assert residual['direction_nonzeros'] > 0
    assert read_matrix(residual['directions'])[:2] == ((Q(0),)*3,)*2
    held = actual_asset(999991,801,b'independent input'); family = native.execute(native_input(native,held,801,HELD_INPUT))
    assert family.field_archive(2)['direction_nonzeros'] > 0
    particular, kernel = read_vector(info['particular']),read_matrix(info['kernel'])
    lower, flags, later = reference(HELD_INPUT,particular)
    for node, expected in ((1,HELD_INPUT),(2,lower),(3,later)):
        assert family.field_archive(node)['particular']==tensor(expected)
    assert family.field_archive(2,occurrences=True)['particular']==tensor(flags)
    baseline = reference(HELD_INPUT,(Q(0),)*5)
    directions = [reference(HELD_INPUT,tuple(row[j] for row in kernel)) for j in range(3)]
    for node, part in ((2,0),(3,2)):
        expected = tuple(tuple(direction[part][i]-baseline[part][i] for direction in directions) for i in range(len(baseline[part])))
        assert read_matrix(family.field_archive(node)['directions'])==expected
    expected_flags = tuple(tuple(direction[1][i]-baseline[1][i] for direction in directions) for i in range(len(baseline[1])))
    assert read_matrix(family.field_archive(2,occurrences=True)['directions'])==expected_flags
    choice = tuple(map(Q,(7,11,13))); theta_choice = tuple(p+q for p,q in zip(particular,action(kernel,choice),strict=True))
    prospective = family.evaluate(choice).result(); response_choice = reference(HELD_INPUT,theta_choice)
    for node,expected in ((2,response_choice[0]),(3,response_choice[2])): assert prospective.field_archive(node)==tensor(expected)
    assert prospective.field_archive(2,occurrences=True)==tensor(response_choice[1])
    second = actual_asset(900112,112,b'second source use'); inputs = (Q(4),Q(-1),Q(2),Q(3)); target2 = reference(inputs,theta)[0]
    grandchild = train_model(child,owner,field_program_batch(child,(observation(2,second,112,inputs,target2),)))
    updated = inspect_field_learning_model(grandchild,owner); relation = updated.inspect()['parent_relation']
    assert updated.parent().archive() == native.archive() and read_vector(relation['shift'])
    assert len(read_matrix(relation['map'])) == 3 and updated.identity()['free_parameter_count'] == 2
    updated_info = updated.inspect(); shift, transport = read_vector(relation['shift']),read_matrix(relation['map'])
    assert read_vector(updated_info['particular'])==tuple(a+b for a,b in zip(particular,action(kernel,shift),strict=True))
    child_kernel = read_matrix(updated_info['kernel'])
    assert child_kernel==tuple(tuple(sum((row[k]*transport[k][j] for k in range(3)),Q(0)) for j in range(2)) for row in kernel)
    clear_field_learning_cache()
    assert inspect_field_learning_model(grandchild,owner).archive() == updated.archive()
    with pytest.raises(ValueError): inspect_field_learning_model(replace(grandchild,parent='0'*64),owner)


@pytest.mark.parametrize('held_values', [(2,3,5,7), (11,-2,3,1), (-3,0,2,5), (0,4,-1,2),
                                        (1,0,0,1), (0,0,3,7), (5,7,0,0), (-2,-3,-5,-7)])
def test_same_learned_structural_relations_transfer_across_unseen_original_uses_and_inputs(model,held_values):
    owner,parent = model; theta = tuple(map(Q,(2,-1,3,0,0)))
    training = actual_asset(900111,111,b'observed use'); inputs = tuple(map(Q,(1,2,3,4)))
    child = train_model(parent,owner,field_program_batch(parent,(observation(1,training,111,inputs,reference(inputs,theta)[0]),)))
    held = actual_asset(999992,802,b'new original use and wording'); values = tuple(map(Q,held_values))
    output = infer_model(child,owner,field_learning_query(child,(input_row(held,802,values),)))
    family = read_field_learning_result(output,state=child); lower,flags,later = reference(values,theta)
    assert family.field_archive(1)['particular']==tensor(values) and family.field_archive(2)['particular']==tensor(lower)
    assert family.field_archive(3)['particular']==tensor(later) and family.field_archive(2,occurrences=True)['particular']==tensor(flags)
    original = family.input_uses()[0]
    assert original['use']==802 and original['instances']['cells']==((0,1,2),(0,1))
    assert NativeInputUseAsset.from_archive(original['asset']).recover(802)[0]==b'new original use and wording'
    assert family.evidence().identity()['free_parameter_count']==2


def test_explicit_prospective_choices_retain_unknown_map_origins_and_whole_family(model):
    owner,state = model; asset = actual_asset(900111,111,b'evidence'); values = (Q(1),Q(2),Q(3),Q(4)); theta = (Q(2),Q(-1),Q(3),Q(0),Q(0))
    child = train_model(state,owner,field_program_batch(state,(observation(1,asset,111,values,reference(values,theta)[0]),)))
    held = actual_asset(999991,801,b'held'); output = infer_model(child,owner,field_learning_query(child,(input_row(held,801,HELD_INPUT),)))
    selected = evaluate_field_program(output,(Q(7),Q(11)),state=child); native = read_field_learning_result(selected,state=child)
    assert native.family().archive() == read_field_learning_result(output).archive()
    assert read_vector(native.free_values_archive()) == (7,11)
    assert native.result().program().identity()['revision'] == child.configuration['revision']+child.payload['revision']
    assert native.field_archive(2) == native.family().field_archive(2)['particular']
    assert NativeAffineEvaluation.from_archive(native.archive()).archive() == native.archive()
    assert read_field_program_result(selected).archive() == native.archive()
    with pytest.raises(ValueError): evaluate_field_program(output,(Q(0),),state=child)
    with pytest.raises(TypeError): output.tensor()


def test_actual_owner_boundaries_lifetime_concurrency_and_inconsistent_updates(model):
    owner,state = model; evidence = inspect_field_learning_model(state,owner); asset = actual_asset(900111,111,b'input')
    actual = native_input(evidence,asset,111,HELD_INPUT)
    foreign = PreparedFieldEvidence.from_archive(evidence.archive())
    with pytest.raises(ValueError): foreign.execute(actual)
    target = actual_asset(910001,9001,b'target')
    row = {'key':1,'input':actual,'target_asset':target,'target_use':9001,'target':tensor((1,)*7),'observed':((0,2),)}
    with pytest.raises(ValueError): evidence.append((row,))
    with ThreadPoolExecutor(4) as pool: results = list(pool.map(evidence.execute,[actual]*8))
    del evidence, actual, asset
    assert len({r.archive() for r in results}) == 1
    for cls,method in ((PreparedFieldEvidence,'archive'),(NativeAffineFieldResult,'archive'),(NativeAffineEvaluation,'archive'),
                      (NativeFieldResidual,'field_archive'),(NativeOccurrenceMapFamily,'inspect')):
        with pytest.raises(RuntimeError): getattr(cls.__new__(cls),method)()


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_cold_learning_model_batch_and_standalone_correlated_result(model,tmp_path,suffix):
    owner,parent = model; asset = actual_asset(900111,111,b'train'); values = (Q(1),Q(2),Q(3),Q(4)); theta = (Q(2),Q(-1),Q(3),Q(0),Q(0))
    batch = field_program_batch(parent,(observation(1,asset,111,values,reference(values,theta)[0]),))
    assert unpack_model(pack_model(batch,native=True)).coefficient_digest == batch.coefficient_digest
    child = train_model(parent,owner,batch); held = actual_asset(999991,801,b'held input')
    query = field_learning_query(child,(input_row(held,801,HELD_INPUT),)); output = infer_model(child,owner,query)
    record = model_record(owner,child); record.attach_metadata(1,0,'response',output)
    path = tmp_path/('learning'+suffix); save_model(path,record); clear_field_learning_cache()
    restored,actual = read_model_record(load_model(path)); assert restored.coefficient_digest == child.coefficient_digest
    assert infer_model(restored,actual,query).field_archive['sha256'] == output.field_archive['sha256']
    packet = tmp_path/'response.value'; packet.write_bytes(pack_value(pack_model(output,native=True)))
    code = '''from pathlib import Path
import sys
from rexgraph.value_codec import unpack_value
from rexgraph.model_codec import unpack_model
from rexgraph.cpp_relational_field_learning import read_field_learning_result
o=unpack_model(unpack_value(Path(sys.argv[1]).read_bytes())); r=read_field_learning_result(o)
assert r.evidence().identity()['free_parameter_count']==2
assert r.evidence().inspect()['free_parameters']==(7004,7005)
assert r.input_uses()[0]['use']==801 and len(r.fields())==3
assert o.dependencies[-1].native.namespace==999991
'''
    subprocess.run([sys.executable,'-I','-c',code,str(packet)],cwd=tmp_path,check=True)


@pytest.mark.parametrize('kind',['dependency','program','evidence','receipt','bytes','model'])
def test_false_whole_field_execution_envelopes_reject(model,kind):
    owner,state = model; held = actual_asset(999991,801,b'held'); output = infer_model(state,owner,field_learning_query(state,(input_row(held,801,HELD_INPUT),)))
    a,r = thaw_tree(output.field_archive),thaw_tree(output.execution_receipt); changes = {}
    if kind == 'dependency': changes['dependencies'] = ()
    elif kind == 'program': a['program_sha256'] = '0'*64
    elif kind == 'evidence': r['evidence_sha256'] = '0'*64
    elif kind == 'receipt': r['interpretation'] = 'identified-target'
    elif kind == 'bytes': a['data'] = a['data'][:-1]; a['sha256'] = sha256(a['data']).hexdigest()
    else: changes['model_digest'] = '0'*64
    candidate = replace(output,field_archive=a,execution_receipt=r,**changes); clear_field_learning_cache()
    with pytest.raises((ValueError,RuntimeError)): read_field_learning_result(candidate,state=state if kind=='model' else None)


def test_rcql_creation_training_intermediary_views_and_system_summary(model):
    from rcql import Executor,parse
    from system.serialize import json_value
    owner,parent = model; record = model_record(owner,parent); asset = actual_asset(900111,111,b'train'); values = (Q(1),Q(2),Q(3),Q(4)); theta = (Q(2),Q(-1),Q(3),Q(0),Q(0))
    batch = field_program_batch(parent,(observation(1,asset,111,values,reference(values,theta)[0]),)); child = train_model(parent,owner,batch)
    held = actual_asset(999991,801,b'held'); query = field_learning_query(child,(input_row(held,801,HELD_INPUT),)); output = infer_model(child,owner,query)
    config = {'source_backed':True,'learning':True,'declaration':learning_declaration(),'parameters':parameters()}
    result = Executor(sources={'r':record},params={'c':config,'b':batch,'o':output}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_relational_field_program",$c),MODEL_TRAIN(MODEL_STATE(),$b),'
        'MODEL_STALK($o,[0,0],node=2,max_coordinates=2),MODEL_VALUES($o),MODEL_INFO(MODEL_STATE())')).values
    assert result[0].coefficient_digest == parent.coefficient_digest and result[1].coefficient_digest == child.coefficient_digest
    assert result[2]['particular']['basis']==1001 and result[2]['identified']==(True,True)
    assert len(result[3]['fields'])==3 and result[4]['trainable'] and result[4]['retains_correlated_families']
    assert json_value(child)['trainable'] and json_value(child)['resumable'] and json_value(output)['retains_whole_fields']


def test_rcql_batch_construction_and_system_preview_keep_complete_encoded_measurements(model):
    from rcql import Executor,parse
    from system.serialize import json_value
    owner,state = model; record = model_record(owner,state); asset = actual_asset(900111,111,b'train')
    values = tuple(map(Q,(1,2,3,4))); theta = tuple(map(Q,(2,-1,3,0,0)))
    batch = field_program_batch(state,(observation(1,asset,111,values,reference(values,theta)[0],((0,0),)),))
    result = Executor(sources={'r':record},params={'t':batch.targets,'m':batch.observed}).execute(parse(
        'FROM $r RETURN MODEL_BATCH(MODEL_STATE(),$t,$m)')).values[0]
    assert result.coefficient_digest==batch.coefficient_digest
    preview = json_value(batch,max_values=2)
    assert preview['targets']['samples']==1 and preview['targets']['observed_stalks']==1
    assert preview['targets']['retains_complete_measurements'] and preview['targets']['native_target_bytes']==len(batch.targets['observations'][0]['target'])
    assert 'observations' not in preview['targets'] and 'target' not in preview['targets']


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_checked_record_copy_and_agent_worker_preserve_full_learning_family(model,tmp_path,backend):
    from rcdb import open_store,copy_record
    from agent.hive import Hive
    owner,parent = model; asset = actual_asset(900111,111,b'train'); values = (Q(1),Q(2),Q(3),Q(4)); theta = (Q(2),Q(-1),Q(3),Q(0),Q(0))
    child = train_model(parent,owner,field_program_batch(parent,(observation(1,asset,111,values,reference(values,theta)[0]),)))
    held = actual_asset(999991,801,b'held'); query = field_learning_query(child,(input_row(held,801,HELD_INPUT),)); output = infer_model(child,owner,query)
    record = model_record(owner,child); record.attach_metadata(1,0,'response',output)
    stores = [open_store('sqlite:///'+str(tmp_path/(n+'.sqlite'))) if backend=='sqlite' else open_store(backend+'://'+str(tmp_path/n),read_only=False) for n in ('first','second')]
    try:
        stored = stores[0].put('learning',record); copy_record(stores[0],stores[1],stored); copied = stores[1].get('learning')
        clear_field_learning_cache(); restored,actual = read_model_record(copied)
        assert read_field_learning_result(copied.get_metadata(1,0,'response'),state=restored).archive() == read_field_learning_result(output).archive()
        hive = Hive(); hive.add_worker('correlated-fields',lambda value:infer_model(restored,actual,value),capability='analyze')
        result = hive.dispatch_capability('analyze',query)['result']; assert result.field_archive['sha256']==output.field_archive['sha256']
    finally:
        for store in stores: store.close()

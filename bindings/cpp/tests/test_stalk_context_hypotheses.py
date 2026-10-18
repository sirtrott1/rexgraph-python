"""Observed complete star hypotheses, explicit frames and unresolved readings."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import pytest

from rexgraph_cpp import NativeSource, PreparedStalkTransport
from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model, stalk_transport_observation_batch, _prepare
from rexgraph.cpp_stalk_parameter_relations import stalk_context_query, stalk_context_hypotheses
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.model_codec import unpack_model
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record


def fixture(*, repeated=True, inactive=False, alternate=None, dimension=1):
    lower=tuple(range(3 if repeated else 6));relations=[];key=0
    for row in range(3):
        flags=[]
        for slot in range(2):
            key+=1;flag=dict(key=key,lower=row if repeated else 2*row+slot,exists=not(inactive and slot==1),
                negative=False,magnitude=1,aspects=(slot,))
            if row==1 and slot==1 and alternate is not None:flag.update(alternate)
            flags.append(flag)
        relations.append(dict(grade=1,key=row+1,type=0,occurrences=tuple(flags)))
    source=NativeSource.from_declaration(dict(profile='declared-role-source-v1',namespace=926184,
        semantic_scale=2,max_grade=1,role_aspects=('first-role','second-role'),
        types=(dict(name='context',min_arity=0,max_arity=0),),witnesses=lower,
        relations=tuple(relations),fields=(),metrics=(),carriers=()))
    axes=(tuple(tuple('lower:'+str(i) for i in range(dimension)) for _ in lower),
          tuple(tuple('upper:'+str(i) for i in range(dimension)) for _ in range(3)))
    slots=tuple((p,i,j) for p in range(6) for i in range(dimension) for j in range(dimension))
    state=cpp_stalk_transport_model(source,axes,((),tuple(() for _ in range(6))),slots,tuple(range(len(lower))))
    frames=(tuple(('lower-frame',tuple(range(dimension))) for _ in lower),
            tuple(('upper-frame',tuple(range(dimension))) for _ in range(3)))
    return source,state,frames


def acquire(state,source,row,target):
    field=tuple((1 if r==row else 0,) for r in range(3))
    return train_model(state,source,stalk_transport_observation_batch(state,field,((row,(target,)),)))


def learned():
    source,state,frames=fixture()
    return source,acquire(state,source,0,4),frames


def test_seed_and_whole_roles_come_from_actual_observations():
    source,state,frames=learned();before=state.coefficient_digest
    output=infer_model(state,source,stalk_context_query(state,(0,1,2),frames))
    acquisition=stalk_context_hypotheses(output)
    assert acquisition['seed_rows']==(0,) and acquisition['attempts']==3
    assert acquisition['rejections']=={'original-row':1}
    assert len(acquisition['blocks'])==2 and len(acquisition['relations'])==4
    block=acquisition['blocks'][0]
    assert block['cell_map']==((0,1),) and block['occurrence_map']==((0,2),(1,3))
    assert block['evidence_samples']==(0,) and block['equations']==(0,1)
    assert block['status']=='open' and block['joint_consistent']
    assert set(output.execution_receipt['certificate']['statuses'])=={'open'}
    assert output.execution_receipt['conditional_family'] is None and state.coefficient_digest==before
    assert output.values.shape==(4,6) and output.axes[0].keys==('origin','free:1','free:2','free:3','free:4','free:5')


def test_original_utf8_accession_is_distinct_from_structural_transfer(monkeypatch):
    from rexgraph.linguistic_source import structural_language_source
    from rexgraph_cpp import PreparedRelationGuards
    documents=[]
    for name,text in (('en','cat'),('zh','猫'),('el','γάτα')):
        raw=text.encode('utf8')
        documents.append(dict(id=name,raw=raw,spans=((0,len(raw)),),assertions=(dict(
            id='reading',type='observed-reading',participants=(dict(token=0,concept=None,role='subject',
                exists=True,negative=False,magnitude=1),)),)))
    corpus=structural_language_source(documents,types=('observed-reading',),roles=('subject',))
    source=corpus.source;guard=PreparedRelationGuards(source.archive)
    rows=tuple(corpus.relation(name,'reading') for name in ('en','zh','el'))
    lower=tuple(corpus.lower.index(('use',name,0)) for name in ('en','zh','el'))
    axes=(tuple(('reading',) if i in lower else ('bytes',) for i in range(len(corpus.lower))),
          tuple(('interpretation',) if r in rows else ('encoding',) for r in range(len(source.identity['coordinates'][1]))))
    occurrences=sum(len(guard.boundary(r)['occurrences']) for r in range(len(axes[1])))
    slots=tuple((guard.boundary(r)['occurrences'][0]['slot'],0,0) for r in rows)
    state=cpp_stalk_transport_model(source,axes,((),tuple(() for _ in range(occurrences))),slots,lower,
        source_accession=corpus.accession())
    state=train_model(state,source,stalk_transport_observation_batch(state,
        tuple((1 if r==rows[0] else 0,) for r in range(len(axes[1]))),((lower[0],(2,)),)))
    frames=(tuple(('reading-frame',(0,)) if i in lower else None for i in range(len(axes[0]))),
            tuple(('interpretation-frame',(0,)) if r in rows else None for r in range(len(axes[1]))))
    output=infer_model(state,source,stalk_context_query(state,rows[1:],frames))
    assert len(stalk_context_hypotheses(output)['blocks'])==2
    assert all(b['status']=='open' for b in stalk_context_hypotheses(output)['blocks'])
    assert output.execution_receipt['source_archive']==source.archive
    assert tuple(bytes.fromhex(c['raw_utf8_hex']) for c in state.configuration['source_accession']['charts'])==(
        b'cat','猫'.encode('utf8'),'γάτα'.encode('utf8'))


def test_complete_block_can_be_excluded_when_every_entry_equation_is_open():
    source,state,frames=learned();state=acquire(state,source,1,6)
    output=infer_model(state,source,stalk_context_query(state,(1,),frames))
    assert output.execution_receipt['certificate']['statuses']==('open','open')
    block=stalk_context_hypotheses(output)['blocks'][0]
    assert block['status']=='excluded' and not block['joint_consistent']
    selected=infer_model(state,source,stalk_context_query(state,(1,),frames,selected_blocks=((0,1),)))
    assert not selected.execution_receipt['conditional_family']['consistent']


def test_selection_preserves_complete_evidence_and_unresolved_alternatives():
    source,state,frames=learned()
    output=infer_model(state,source,stalk_context_query(state,(1,2),frames,selected_blocks=((0,1),)))
    family=output.execution_receipt['conditional_family']
    assert family['consistent'] and family['kernel_shape']==(6,3)
    assert output.execution_receipt['certificate']['evidence_fit']==state.payload['fit']
    assert len(stalk_context_hypotheses(output)['blocks'])==2


def test_actual_separate_field_observations_can_force_complete_blocks():
    source,state,frames=fixture(repeated=False)
    for row in (0,1):
        state=train_model(state,source,stalk_transport_observation_batch(state,
            tuple((1 if r==row else 0,) for r in range(3)),((2*row,(Q(2,3),)),(2*row+1,(Q(7,5),)))))
    output=infer_model(state,source,stalk_context_query(state,(1,),frames))
    assert stalk_context_hypotheses(output)['blocks'][0]['status']=='forced'
    assert output.execution_receipt['certificate']['statuses']==('forced','forced')


def test_no_observations_produce_no_hypotheses_or_false_certificates():
    source,state,frames=fixture()
    output=infer_model(state,source,stalk_context_query(state,(1,),frames))
    assert output.values.shape==(0,7) and output.execution_receipt['certificate'] is None
    assert stalk_context_hypotheses(output)['blocks']==() and stalk_context_hypotheses(output)['attempts']==0


def test_block_statuses_under_a_supplied_basis_remain_explicitly_conditional():
    source,independent,frames=fixture();c=independent.configuration
    state=cpp_stalk_transport_model(source,c['stalk_axes'],c['fixed_maps'],c['parameter_slots'],(0,1,2),
        parameter_space={'name':'supplied-tie','keys':('shared',)},parameter_basis=tuple((i,0,1) for i in range(6)))
    state=acquire(state,source,0,4)
    output=infer_model(state,source,stalk_context_query(state,(1,),frames))
    acquisition=stalk_context_hypotheses(output)
    assert acquisition['blocks'][0]['status']=='forced'
    assert acquisition['evidence_scope']=='conditional-on-declared-parameter-basis'


@pytest.mark.parametrize('value',[True,Q(1),1.0])
def test_context_source_reference_scalar_kinds_are_not_coerced(value):
    source,state,frames=learned();query=stalk_context_query(state,(1,),frames)
    query['source_reference']['schema_version']=value
    with pytest.raises(ValueError,match='original source'):infer_model(state,source,query)


def test_no_frame_map_is_inferred_from_equal_axes_or_dimensions():
    source,state,frames=learned();frames=(tuple(None for _ in frames[0]),frames[1])
    output=infer_model(state,source,stalk_context_query(state,(1,),frames))
    assert stalk_context_hypotheses(output)['blocks']==()
    assert stalk_context_hypotheses(output)['rejections']=={'undeclared-lower-frame':1}


def test_frame_permutations_move_complete_block_entries():
    source,state,frames=fixture(dimension=2)
    state=train_model(state,source,stalk_transport_observation_batch(state,((1,0),(0,0),(0,0)),((0,(4,6)),)))
    frames=((frames[0][0],('lower-frame',(1,0)),frames[0][2]),
            (frames[1][0],('upper-frame',(1,0)),frames[1][2]))
    output=infer_model(state,source,stalk_context_query(state,(1,),frames))
    a=stalk_context_hypotheses(output)
    assert len(a['relations'])==8 and a['relations'][0][1]==((0,Q(1)),(11,Q(-1)))


def test_inactive_flags_retain_unidentified_supported_map_directions():
    source,state,frames=fixture(inactive=True);state=acquire(state,source,0,4)
    output=infer_model(state,source,stalk_context_query(state,(1,),frames))
    assert stalk_context_hypotheses(output)['blocks'][0]['occurrence_map']==((0,2),(1,3))
    assert output.execution_receipt['certificate']['statuses']==('open','open')
    assert output.execution_receipt['certificate']['evidence_fit']['free_columns']==(1,2,3,4,5)


@pytest.mark.parametrize('alteration',[{'exists':False},{'negative':True},{'magnitude':Q(3,2)},{'aspects':(0,)}])
def test_context_map_does_not_erase_roles_activity_orientation_or_coefficients(alteration):
    source,state,frames=fixture(alternate=alteration);state=acquire(state,source,0,4)
    output=infer_model(state,source,stalk_context_query(state,(1,),frames))
    assert stalk_context_hypotheses(output)['rejections']=={'typed-retained-map':1}
    assert stalk_context_hypotheses(output)['blocks']==()


def test_target_values_do_not_choose_the_structural_candidate_family():
    source,state,frames=learned()
    first=infer_model(state,source,stalk_context_query(state,(1,),frames))
    state=acquire(state,source,1,9)
    second=infer_model(state,source,stalk_context_query(state,(1,),frames))
    assert stalk_context_hypotheses(first)['relations']==stalk_context_hypotheses(second)['relations']
    assert stalk_context_hypotheses(first)['blocks'][0]['status']=='open'
    assert stalk_context_hypotheses(second)['blocks'][0]['status']=='excluded'


def test_native_discovery_uses_owned_samples_and_detached_parallel_receipts():
    source,state,frames=learned();owner=_prepare(state,source)
    expected=owner.contextual_hypotheses((1,),frames)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(r==expected for r in pool.map(lambda _:owner.contextual_hypotheses((1,),frames),range(8)))
    expected['blocks'][0]['cell_map']=();expected['relations']=()
    assert owner.contextual_hypotheses((1,),frames)['blocks'][0]['cell_map']==((0,1),)
    empty=PreparedStalkTransport.__new__(PreparedStalkTransport)
    with pytest.raises(RuntimeError):empty.contextual_hypotheses((1,),frames)


@pytest.mark.parametrize('targets',[(True,),(Q(1),),(1.0,),(1,1),(3,),'1'])
def test_malformed_target_rows_reject(targets):
    source,state,frames=learned()
    with pytest.raises((ValueError,TypeError,OverflowError)):
        q=stalk_context_query(state,targets,frames);infer_model(state,source,q)


@pytest.mark.parametrize('frame',[('lower-frame',(0,0)),('lower-frame',(True,)),('lower-frame',()),('',(0,))])
def test_incomplete_or_coerced_frame_maps_reject(frame):
    source,state,frames=learned();frames=((frame,*frames[0][1:]),frames[1])
    with pytest.raises((ValueError,TypeError,OverflowError)):
        q=stalk_context_query(state,(1,),frames);infer_model(state,source,q)


@pytest.mark.parametrize('selected',[((2,1),),((0,1),(0,1)),((True,1),),((0,True),),'x'])
def test_selected_blocks_require_actual_acquired_correspondences(selected):
    source,state,frames=learned()
    with pytest.raises((ValueError,TypeError)):
        q=stalk_context_query(state,(1,),frames,selected_blocks=selected);infer_model(state,source,q)


@pytest.mark.parametrize('changed',['blocks','seed_rows','relations','frame_atlas','attempts'])
def test_changed_acquisition_receipts_reject(changed):
    source,state,frames=learned();output=infer_model(state,source,stalk_context_query(state,(1,),frames))
    receipt=thaw_tree(output.execution_receipt);receipt['acquisition'][changed]=0 if changed=='attempts' else ()
    with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=receipt)


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_acquired_hypotheses_roundtrip_with_original_evidence(tmp_path,suffix):
    source,state,frames=learned();q=stalk_context_query(state,(1,2),frames,selected_blocks=((0,1),))
    output=infer_model(state,source,q);record=model_record(source,state)
    record.attach_metadata(1,0,'query',q);record.attach_metadata(1,0,'output',output)
    path=tmp_path/('contexts'+suffix);save_model(path,record);loaded=load_model(path);model,native=read_model_record(loaded)
    assert loaded.get_metadata(1,0,'output').coefficient_digest==output.coefficient_digest
    assert infer_model(model,native,loaded.get_metadata(1,0,'query')).coefficient_digest==output.coefficient_digest
    assert unpack_model(model.payload['observations'][0]['batch_checkpoint']).coefficient_digest==unpack_model(
        state.payload['observations'][0]['batch_checkpoint']).coefficient_digest
    assert native.archive==source.archive


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_cold_five_package_hypothesis_replay(tmp_path,backend):
    from rcdb import open_store
    source,state,frames=learned();q=stalk_context_query(state,(1,2),frames,selected_blocks=((0,1),))
    output=infer_model(state,source,q);record=model_record(source,state)
    record.attach_metadata(1,0,'query',q);record.attach_metadata(1,0,'output',output)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('contexts',record,expected_version=0);assert store.verify_commits('contexts');store.close()
    script='''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import read_model_record
store=open_store(sys.argv[1]);record=store.read_record('contexts',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
output=result.values[0]
assert output.coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert result.values[1]['context_hypotheses'] and json_value(state)['context_hypotheses']
assert output.execution_receipt['acquisition']['blocks'][0]['status']=='open'
assert output.execution_receipt['conditional_family']['kernel_shape']==(6,3)
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('contexts');hive.add_worker('contexts',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('contexts',query,record=False).coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    result=subprocess.run([sys.executable,'-c',script,url,output.coefficient_digest],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr

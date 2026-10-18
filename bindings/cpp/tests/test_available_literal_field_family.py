"""Owning joint availability, complete recipes, decisions and native streaming."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
import gc
import pytest
import rexgraph_cpp as cpp
from test_known_tensor_graph import small
from test_relational_field_evidence import input_accession
from test_relational_field_program import matrix, source

POLICY={'limits':(4096,16000000,64000000,16384),'max_bytes':256*1024*1024}


def context(origins, remove=(), cutoffs=None, namespace=None):
    return cpp.NativeAvailabilityContext(tuple({'origin':origin,
        'namespace':(namespace or {}).get(key,origin.inspect()['namespace']),
        'sequence':(cutoffs or {}).get(key,origin.inspect()['sequence'])}
        for key,origin in origins.items() if key not in remove))


def fixture(raw=b'bank'):
    f=small(raw)
    origins={}
    def publish(key,role,payload):
        origins[key]=cpp.NativeValueOrigin(key,f['owner'].archive,f['owner'].identity['sequence'],role,payload,**POLICY)
        return key
    def program_keys(program,base):
        return {name:publish(base+i,name,program.archive()) for i,name in enumerate(('program','maps','metrics'))}
    parameters={'profile':'rexgraph.relational-field-parameters.v1','key':8201,'revision':1,'node':2,
                'variables':tuple((0,r,c) for r in range(2) for c in range(3)),
                'parameters':tuple(range(8301,8307)),
                'basis':{'shape':(6,6),'entries':tuple((i,i,1) for i in range(6))}}
    pkeys=program_keys(f['target_program'],1)
    available=cpp.PreparedAvailableFieldProgram(f['target_program'],pkeys,context(origins))
    root=publish(4,'evidence',cpp.field_evidence_origin_payload(f['target_program'],parameters))
    member=cpp.PreparedAvailableFieldEvidence(available,parameters,root,context(origins))
    family=cpp.PreparedFieldFamily({'profile':'rexgraph.relational-field-family.v1','key':8401,'revision':1,'interfaces':()},(member.evidence(),))
    def input_keys(program,declaration,field,base):
        payloads=cpp.field_input_origin_payloads(program,{'asset':f['asset'].archive(),'accession':declaration,
            'use':f['admitted'].input().inspect()['use'],'field':field})
        return {name:publish(base+i,{'asset':'input-asset','value':'input-value','accession':'input-accession','reconstruction':'reconstruction'}[name],payload)
                for i,(name,payload) in enumerate(payloads.items())}
    selected=input_keys(f['target_program'],f['target_spec'],f['graph'].lifted_field(),10)
    original_program=program_keys(f['parent_program'],20)
    original=input_keys(f['parent_program'],input_accession(),f['literal'].input().inspect()['field'],30)
    reader=publish(40,'interface',f['admitted'].archive())
    row={'input':f['admitted'],'origins':{'input':selected,'original_program':original_program,'original_input':original,'reader':reader}}
    ledger=((),)
    payload=cpp.joint_family_origin_payload(family,(member,),ledger,(),context(origins))
    origins[41]=cpp.joint_packet_value_origin(41,f['owner'].archive,f['owner'].identity['sequence'],'interface',payload,context(origins))
    model=cpp.PreparedAvailableLiteralFieldFamily(family,(member,),ledger,{'interfaces':(),'family':41},context(origins))
    return {**f,'origins':origins,'publish':publish,'member':member,'family':family,'model':model,'row':row,
            'recipe':{'program':pkeys,'root':root,'targets':()},'ledger':ledger}


def decision(f, values=(0,0,2,0,0,3)):
    model=f['model'];origins=f['origins'];ctx=context(origins)
    result=model.execute(0,(f['row'],),ctx)
    origins[42]=cpp.joint_packet_value_origin(42,f['owner'].archive,f['owner'].identity['sequence'],'prospective',model.setting_payload(values,ctx),ctx)
    ctx=context(origins)
    origins[43]=cpp.joint_packet_value_origin(43,f['owner'].archive,f['owner'].identity['sequence'],'prospective',result.completion_payload(values,42,ctx),ctx)
    ctx=context(origins)
    return result,result.evaluate(values,42,43,ctx),ctx


def test_actual_owning_model_result_and_decision_canonical_fields():
    f=fixture();result,evaluation,ctx=decision(f)
    assert result.model(ctx).same_owner(f['model'])
    assert result.raw_result(ctx).family().same_owner(f['family'])
    assert result.inputs(ctx)[0]['input'].graph().literal_parent().same_owner(f['literal'])
    assert evaluation.raw_evaluation(ctx).field_archive(2)==result.raw_result(ctx).evaluate((0,0,2,0,0,3)).field_archive(2)
    assert evaluation.decision_origins(ctx)[0].same_owner(f['origins'][42])
    assert evaluation.decision_origins(ctx)[1].same_owner(f['origins'][43])
    for value in (f['model'],result,evaluation):
        image=value.archive(ctx);cold=type(value).from_archive(image,ctx)
        assert cold.archive(ctx)==image
    assert f['model'].training_readers(ctx)==((),)
    assert f['model'].origins(ctx)=={'interfaces':(),'family':41}


def test_every_consumed_origin_denies_cached_fields_packets_and_cold_replay():
    f=fixture();result,evaluation,ctx=decision(f);image=evaluation.archive(ctx);packet=evaluation.prepare(ctx)
    consumed=(*result.obligations(ctx),*evaluation.decision_origins(ctx))
    for origin in consumed:
        key=origin.inspect()['key']
        for denied in (context(f['origins'],remove=(key,)),context(f['origins'],namespace={key:999999999})):
            for call in (lambda:evaluation.admit(denied),lambda:evaluation.field_archive(2,denied),
                         lambda:packet.to_bytes(denied),lambda:cpp.NativeAvailableLiteralJointFieldEvaluation.from_archive(image,denied)):
                with pytest.raises(ValueError):call()
    assert evaluation.archive(ctx)==image


def test_equal_receipt_replaced_live_owner_refuses_and_cold_can_reconstruct():
    f=fixture();result,evaluation,ctx=decision(f);image=evaluation.archive(ctx)
    old=f['origins'][40]
    replacement=cpp.NativeValueOrigin(40,old.source_archive(),old.inspect()['sequence'],old.inspect()['role'],old.payload())
    assert old.receipt()==replacement.receipt() and not old.same_owner(replacement)
    f['origins'][40]=replacement;changed=context(f['origins'])
    with pytest.raises(ValueError):evaluation.admit(changed)
    assert cpp.NativeAvailableLiteralJointFieldEvaluation.from_archive(image,changed).archive(changed)==image


def test_prepared_packet_streams_bounded_chunks_and_pins_deleted_owners():
    f=fixture();result,evaluation,ctx=decision(f);packet=evaluation.prepare(ctx);expected=evaluation.archive(ctx)
    del f,result,evaluation;gc.collect()
    chunks=[]
    assert packet.stream(chunks.append,ctx)==len(expected)==packet.size
    assert b''.join(chunks)==expected and all(len(chunk)<=65536 for chunk in chunks)
    sink=BytesIO();assert packet.stream(sink.write,ctx)==len(expected) and sink.getvalue()==expected
    with pytest.raises(OSError):packet.stream(lambda chunk:0,ctx)


def test_original_callback_baseexceptions_cancel_stream_and_cold_safely():
    f=fixture();result,evaluation,ctx=decision(f);image=evaluation.archive(ctx)
    class Stop(BaseException):pass
    error=Stop('stop')
    def stop(*args):raise error
    for call in (lambda:evaluation.archive(ctx,check=stop),lambda:evaluation.prepare(ctx).stream(stop,ctx),
                 lambda:cpp.NativeAvailableLiteralJointFieldEvaluation.from_archive(image,ctx,check=stop),
                 lambda:f['model'].execute(0,(f['row'],),ctx,check=stop)):
        with pytest.raises(Stop) as exc:call()
        assert exc.value is error
    assert evaluation.archive(ctx)==image


@pytest.mark.parametrize('kind',('model','result','evaluation'))
def test_cold_initialized_subclass_cannot_replace_immutable_owner(kind):
    f=fixture();result,evaluation,ctx=decision(f);value={'model':f['model'],'result':result,'evaluation':evaluation}[kind];image=value.archive(ctx)
    class Reused(type(value)):
        def __new__(cls):return value
    with pytest.raises(TypeError):Reused.from_archive(image,ctx)
    assert value.archive(ctx)==image


def test_bulk_native_replay_rebuilds_members_without_owner_substitution():
    f=fixture();ctx=context(f['origins'])
    family,members=cpp.replay_available_joint_family(f['family'],(f['recipe'],),f['ledger'],ctx)
    assert family.archive()==f['family'].archive() and not family.same_owner(f['family'])
    assert members[0].evidence().archive()==f['member'].evidence().archive()
    with pytest.raises(ValueError):
        cpp.PreparedAvailableLiteralFieldFamily(f['family'],members,f['ledger'],{'interfaces':(),'family':41},ctx)
    model=cpp.PreparedAvailableLiteralFieldFamily(family,members,f['ledger'],{'interfaces':(),'family':41},ctx)
    assert model.archive(ctx)==f['model'].archive(ctx)


def test_source_conditioned_reader_completion_cannot_authorize_another_source():
    f=fixture();result,evaluation,ctx=decision(f)
    changed=small(b'bark',parent_program=f['parent_program'],target_program=f['target_program'])
    row=deepcopy(f['row']['origins'])
    # Missing receipts are refused before a reader can authorize changed bytes.
    with pytest.raises(ValueError):f['model'].execute(0,({'input':changed['admitted'],'origins':row},),ctx)
    assert evaluation.archive(ctx)


def test_immutable_concurrent_native_decisions_and_policy_refusals():
    f=fixture();result,evaluation,ctx=decision(f);expected=evaluation.archive(ctx)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _:result.evaluate((0,0,2,0,0,3),42,43,ctx).archive(ctx),range(4)))==[expected]*4
    for data in (expected[:-1],expected+b'x',expected[:-1]+bytes((expected[-1]^1,))):
        with pytest.raises((ValueError,RuntimeError)):cpp.NativeAvailableLiteralJointFieldEvaluation.from_archive(data,ctx)
    with pytest.raises(ValueError):cpp.NativeAvailableLiteralJointFieldEvaluation.from_archive(expected,ctx,max_bytes=len(expected)-1)
    with pytest.raises(RuntimeError,match='metadata bound'):f['model'].training_readers(ctx,max_records=0)


@pytest.mark.parametrize('kind',('model','result','evaluation'))
def test_exposed_cold_owner_is_reserved_before_every_callback(kind):
    f=fixture();result,evaluation,ctx=decision(f);value={'model':f['model'],'result':result,'evaluation':evaluation}[kind]
    exposed=[]
    class Exposed(type(value)):
        def __new__(cls):
            obj=super().__new__(cls);exposed.append(obj);return obj
    def inspect():
        with pytest.raises(RuntimeError):exposed[0].admission()
        if kind=='model':
            with pytest.raises(TypeError):exposed[0].__init__(f['family'],(f['member'],),f['ledger'],{'interfaces':(),'family':41},ctx)
    image=value.archive(ctx)
    assert Exposed.from_archive(image,ctx,check=inspect).archive(ctx)==image


def test_cancelled_constructor_cannot_publish_or_replace_an_owner():
    f=fixture();ctx=context(f['origins']);obj=cpp.PreparedAvailableLiteralFieldFamily.__new__(cpp.PreparedAvailableLiteralFieldFamily)
    class Stop(Exception):pass
    error=Stop('cancel')
    def stop():
        with pytest.raises(RuntimeError):obj.admission()
        raise error
    with pytest.raises(Stop):obj.__init__(f['family'],(f['member'],),f['ledger'],{'interfaces':(),'family':41},ctx,check=stop)
    obj.__init__(f['family'],(f['member'],),f['ledger'],{'interfaces':(),'family':41},ctx)
    with pytest.raises(TypeError):obj.__init__(f['family'],(f['member'],),f['ledger'],{'interfaces':(),'family':41},ctx)
    assert obj.archive(ctx)==f['model'].archive(ctx)


def test_packet_publication_requires_current_origins_and_explicit_bounds():
    f=fixture();ctx=context(f['origins']);packet=f['model'].prepare(ctx)
    denied=context(f['origins'],remove=(41,))
    with pytest.raises(ValueError):cpp.joint_packet_value_origin(50,f['owner'].archive,f['owner'].identity['sequence'],'interface',packet,denied)
    with pytest.raises(RuntimeError,match='payload byte bound'):cpp.joint_packet_value_origin(50,f['owner'].archive,f['owner'].identity['sequence'],'interface',packet,ctx,max_bytes=packet.size-1)
    assert packet.to_bytes(ctx)==f['model'].archive(ctx)


def test_explicit_reader_replay_keeps_complete_program_and_actual_selected_owner():
    f=fixture();ctx=context(f['origins']);foreign=cpp.NativeKnownTensorGraphInput.from_archive(f['admitted'].archive())
    with pytest.raises(ValueError):f['model'].execute(0,({**f['row'],'input':foreign},),ctx)
    value=cpp.replay_joint_reader_input(foreign,f['family'].member(0).program())
    assert value.archive()==foreign.archive() and value.graph().literal_parent().asset().archive()==f['asset'].archive()
    result=f['model'].execute(0,({**f['row'],'input':value},),ctx)
    assert result.raw_result(ctx).field_archive(1)['particular']==f['graph'].lifted_field()
    other=small(b'bank')['parent_program']
    with pytest.raises(ValueError):cpp.replay_joint_reader_input(value,other)


def test_explicit_direct_literal_replay_retains_recipe_and_actual_program():
    f=fixture();foreign=cpp.NativeLiteralInput.from_archive(f['literal'].archive())
    actual=cpp.replay_joint_reader_input(foreign,f['parent_program'])
    assert actual.archive()==foreign.archive() and actual.asset().archive()==f['asset'].archive()
    raw=cpp.PreparedFieldEvidence(f['parent_program'],{'profile':'rexgraph.relational-field-parameters.v1','key':8201,'revision':1,'node':2,
        'variables':tuple((0,r,c) for r in range(2) for c in range(2)),'parameters':tuple(range(8301,8305)),
        'basis':{'shape':(4,4),'entries':tuple((i,i,1) for i in range(4))}})
    family=cpp.PreparedFieldFamily({'profile':'rexgraph.relational-field-family.v1','key':8401,'revision':1,'interfaces':()},(raw,))
    with pytest.raises(ValueError):cpp.NativeLiteralJointFieldResult(family,0,(foreign,))
    assert cpp.NativeLiteralJointFieldResult(family,0,(actual,)).inputs()[0].archive()==actual.archive()


@pytest.mark.parametrize('change',('source','basis','map','metric'))
def test_reader_replay_refuses_changed_complete_source_basis_map_or_metric(change):
    f=fixture();spec=deepcopy(f['spec']);owner=f['owner']
    if change=='source':owner=source(992002)
    elif change=='basis':spec['stalks'][0]['basis']+=10000
    elif change=='map':spec['occurrences'][0]['map']['entries']=((0,0,1),)
    else:
        width=len(spec['stalks'][0]['directions'])
        spec['stalks'][0]['metric']=matrix(tuple(tuple((2 if i==0 else 1) if i==j else 0 for j in range(width)) for i in range(width)))
    changed=cpp.PreparedAccessionFieldProgram(cpp.PreparedFieldProgram(owner.archive,spec).archive())
    with pytest.raises(ValueError,match='same complete'):cpp.replay_joint_reader_input(f['admitted'],changed)

"""Whole reader obligations survive native execution, choices and cold adoption."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gc
import importlib.util
import json
import os
import pytest
import rexgraph_cpp as cpp

P={'limits':(4096,16000000,64000000,16384),'max_bytes':64*1024*1024}


def context(origins,remove=(),cutoffs=None):
    cutoffs={} if cutoffs is None else cutoffs
    return cpp.NativeAvailabilityContext(tuple({'origin':v,'namespace':v.inspect()['namespace'],
        'sequence':cutoffs.get(k,v.inspect()['sequence'])} for k,v in origins.items() if k not in remove))


def fixture(n=3, future_key=None):
    policy=dict(P)
    future=None
    if future_key is not None:
        policy["limits"]=(4096,16000000,64000000,1<<24)
        future=cpp.NativeSource(Path(os.environ["REXGRAPH_CPP_AVAILABLE_READER_FUTURE_HISTORY"]).read_bytes())
        assert future.identity["sequence"]>future.identity["first_sequence"]
    root=os.environ.get('REXGRAPH_CPP_LITERAL_READER_PREFLIGHT')
    if not root:pytest.skip('complete original source preflight required')
    root=Path(root);spec=importlib.util.spec_from_file_location('available_reader_source',root/'preflight.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    selection=json.loads((root/'source-selection-frozen.json').read_bytes());row=selection['cases'][str(n)]
    inventory=tuple(bytes.fromhex(x) for x in selection['inventory_hex']);obj=module.reader(row,210000+n,inventory)
    d=deepcopy(obj['finite_spec']);d['key']+=100;d['revision']+=1
    for symbol in d['symbols']:
        b=next(b for b in d['bindings'] if symbol['key'] in d['variables'][b['cell'][1]]['domain'])
        symbol['definition']=cpp.encode_literal_use_value(obj['asset'],b['use'],**policy)
    origins={}
    def publish(key,role,payload):
        publisher=future if key==future_key else obj['owner']
        origins[key]=cpp.NativeValueOrigin(key,publisher.archive,publisher.identity['sequence'],role,payload,**policy)
    publish(1,'input-asset',obj['asset'].archive());publish(2,'evidence',cpp.finite_constraint_origin_payload(obj['asset'],d,'observed',**policy))
    finite=cpp.PreparedAvailableFiniteSource(obj['asset'],d,'observed',{'asset':1,'constraints':2},context(origins),**policy)
    for key,role in ((3,'program'),(4,'maps'),(5,'metrics')):publish(key,role,obj['program'].archive())
    program=cpp.PreparedAvailableFieldProgram(obj['program'],{'program':3,'maps':4,'metrics':5},context(origins),**policy)
    column=inventory.index(obj['asset'].recover(row['uses'][0]['original_token_id'])[0])
    variables=tuple((p['slots'][1],0,column) for p in obj['maps'])
    parameters={'profile':'rexgraph.relational-field-parameters.v1','key':56001,'revision':1,'node':2,'variables':variables,
                'parameters':(56101,),'basis':{'shape':(len(variables),1),'entries':tuple((i,0,1) for i in range(len(variables)))}}
    publish(6,'evidence',cpp.field_evidence_origin_payload(obj['program'],parameters,**policy))
    evidence=cpp.PreparedAvailableFieldEvidence(program,parameters,6,context(origins),**policy)
    submission={'asset':obj['asset'],'accession':obj['access'],'use':row['uses'][0]['original_token_id'],
                'field':obj['literal'].input().inspect()['field'],'origins':{'asset':1,'value':7,'accession':8,'reconstruction':9}}
    payloads=cpp.field_input_origin_payloads(obj['program'],{k:(obj['asset'].archive() if k=='asset' else v) for k,v in submission.items() if k!='origins'},**policy)
    for key,name,role in ((7,'value','input-value'),(8,'accession','input-accession'),(9,'reconstruction','reconstruction')):publish(key,role,payloads[name])
    publish(10,'interface',obj['literal'].archive())
    bindings=tuple(b['key'] for b in d['bindings']);symbols=tuple(v['domain'][0] for v in d['variables'])
    spec={'profile':'rexgraph.available-relational-finite-field-map.v1','key':55001,'revision':1,'rules':tuple(
        {'key':55002+i,'bindings':bindings,'symbols':symbols,'target':i,'inputs':(submission,),'unsupported':''} for i in (0,1))}
    targets=(program,evidence)
    payload=cpp.finite_interpretation_origin_payload(finite,targets,spec,'observed',context(origins),**policy)
    publish(11,'maps',payload);publish(12,'evidence',payload)
    base=cpp.PreparedAvailableFiniteFieldMap(finite,targets,spec,'observed',{'maps':11,'relationship':12},context(origins),**policy)
    request={'profile':'rexgraph.relational-literal-finite-field-map.v1','key':55001,'revision':1,'rules':tuple(
        dict(r,inputs=(obj['literal'],)) for r in spec['rules'])}
    reader=cpp.PreparedLiteralFiniteFieldMap(finite.raw(),(program.program(),evidence.evidence()),request,**policy)
    keys={'inputs':((10,),(10,)),'interface':13}
    publish(13,'interface',cpp.literal_finite_interface_origin_payload(base,reader,keys['inputs'],context(origins),**policy))
    model=cpp.PreparedAvailableLiteralFiniteFieldMap(base,reader,keys,context(origins),**policy)
    return locals()


@pytest.mark.parametrize('n',(1,3,5,8))
def test_complete_live_and_cold_fields_and_actual_owners(n):
    f=fixture(n);ctx=context(f['origins']);model=f['model'];family=model.execute(ctx)
    assert model.reader(ctx).same_owner(f['reader']) and model.base(ctx).same_owner(f['base'])
    assert family.model(ctx).same_owner(model) and model.origins(ctx)==f['keys']
    assert family.raw(ctx).archive()==f['reader'].execute().family().archive()
    assert tuple(b.inspect(ctx)['affine'] for b in family.select(0,2,ctx))==(False,True)
    assert len(model.reader(ctx).source().asset().uses())==n
    assert cpp.PreparedAvailableLiteralFiniteFieldMap.from_archive(model.archive(),ctx,**P).archive()==model.archive()
    assert cpp.NativeAvailableLiteralFiniteFieldResult.from_archive(family.archive(),ctx,**P).archive()==family.archive()


@pytest.mark.parametrize('key',range(1,14))
def test_every_retained_dependency_is_rechecked_live_and_cold(key):
    f=fixture();ctx=context(f['origins']);family=f['model'].execute(ctx);denied=context(f['origins'],(key,))
    for call in (lambda:f['model'].execute(denied),lambda:family.admit(denied),lambda:family.select(0,1,denied),
                 lambda:cpp.NativeAvailableLiteralFiniteFieldResult.from_archive(family.archive(),denied,**P)):
        with pytest.raises(ValueError,match='missing actual value origin'):call()


def chosen(f,ordinal=1):
    origins=f['origins'];family=f['model'].execute(context(origins));free=(7,) if ordinal else ();setting=14 if ordinal else 0
    if setting:f['publish'](14,'prospective',f['evidence'].parameter_setting_payload(free))
    f['publish'](15,'prospective',family.completion_payload(ordinal,free,setting,context(origins)))
    return family,family.evaluate(ordinal,free,setting,15,context(origins))


@pytest.mark.parametrize('ordinal',(0,1))
def test_reader_specific_prospective_choice_retains_entire_family(ordinal):
    f=fixture();family,value=chosen(f,ordinal);ctx=context(f['origins'])
    assert value.family(ctx).archive()==family.archive() and value.inspect(ctx)['ordinal']==ordinal
    assert value.inspect(ctx)['choice_receipt']==f['origins'][15].receipt()
    assert cpp.NativeAvailableLiteralFiniteFieldEvaluation.from_archive(value.archive(),ctx,**P).archive()==value.archive()
    for key in ((10,13,14,15) if ordinal else (10,13,15)):
        with pytest.raises(ValueError):value.admit(context(f['origins'],(key,)))
    generic=f['base'].execute(ctx);f['publish'](16,'prospective',generic.completion_payload(ordinal,(7,) if ordinal else (),14 if ordinal else 0,ctx))
    with pytest.raises(ValueError,match='role or complete content'):family.evaluate(ordinal,(7,) if ordinal else (),14 if ordinal else 0,16,context(f['origins']))


def test_changed_recipe_equal_field_and_generic_packets_need_new_whole_interface():
    f=fixture();d=deepcopy(f['obj']['literal_spec']);rows=[dict(c) for c in d['coordinates']]
    index=next(i for i,c in enumerate(rows) if c['operand']!=f['obj']['asset'].recover(c['use'])[c['span_index']])
    rows[index]['operand']=b'\0different-unobserved-predicate';d['coordinates']=tuple(rows)
    literal=cpp.NativeLiteralInput(f['obj']['accession'],d,**P)
    request=dict(f['request'],rules=tuple(dict(r,inputs=(literal,)) for r in f['request']['rules']))
    reader=cpp.PreparedLiteralFiniteFieldMap(f['finite'].raw(),(f['program'].program(),f['evidence'].evidence()),request,**P)
    assert reader.raw_map().archive()==f['reader'].raw_map().archive()
    with pytest.raises(ValueError):cpp.literal_finite_interface_origin_payload(f['base'],reader,((10,),(10,)),context(f['origins']),**P)
    f['publish'](17,'interface',literal.archive());payload=cpp.literal_finite_interface_origin_payload(f['base'],reader,((17,),(17,)),context(f['origins']),**P)
    assert payload!=f['origins'][13].payload()
    with pytest.raises(ValueError):cpp.PreparedAvailableLiteralFiniteFieldMap(f['base'],reader,{'inputs':((17,),(17,)),'interface':13},context(f['origins']),**P)


def test_foreign_equal_owners_require_explicit_cold_adoption():
    f=fixture();model=f['model'];origins=f['origins'];clones={k:cpp.NativeValueOrigin(k,v.source_archive(),v.inspect()['sequence'],v.inspect()['role'],v.payload(),**P) for k,v in origins.items()}
    with pytest.raises(ValueError,match='live value origin owner'):model.admit(context(clones))
    assert cpp.PreparedAvailableLiteralFiniteFieldMap.from_archive(model.archive(),context(clones),**P).archive()==model.archive()
    foreign=cpp.PreparedLiteralFiniteFieldMap.from_archive(f['reader'].archive(),**P)
    with pytest.raises(ValueError):cpp.literal_finite_interface_origin_payload(f['base'],foreign,((10,),(10,)),context(origins),**P)


def test_escaped_owner_concurrency_immutability_and_bounds():
    f=fixture();ctx=context(f['origins']);family,value=chosen(f);ctx=context(f['origins']);packet=value.archive();branch=family.select(1,1,ctx)[0]
    with pytest.raises(TypeError):f['model'].__init__(f['base'],f['reader'],f['keys'],ctx)
    with pytest.raises((ValueError,RuntimeError)):f['model'].origins(ctx,max_records=1)
    with pytest.raises((ValueError,RuntimeError)):f['model'].dependency_archives(ctx,max_records=1)
    del f,family,value;gc.collect();assert branch.family(ctx).inspect(ctx)['branch_count']==2
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert tuple(pool.map(lambda _:cpp.NativeAvailableLiteralFiniteFieldEvaluation.from_archive(packet,ctx,**P).archive(),range(4)))==(packet,)*4
    for cls in (cpp.PreparedAvailableLiteralFiniteFieldMap,cpp.NativeAvailableLiteralFiniteFieldResult,cpp.NativeAvailableLiteralFiniteFieldBranch,cpp.NativeAvailableLiteralFiniteFieldEvaluation):
        with pytest.raises(RuntimeError):cls.__new__(cls).admit(ctx)
    for bad in (packet[:-1],packet+b'trailing'):
        with pytest.raises(ValueError):cpp.NativeAvailableLiteralFiniteFieldEvaluation.from_archive(bad,ctx,**P)


def test_reentrant_origin_rows_cannot_publish_a_partial_owner():
    f=fixture();ctx=context(f['origins']);owner=cpp.PreparedAvailableLiteralFiniteFieldMap.__new__(cpp.PreparedAvailableLiteralFiniteFieldMap)
    class Rows:
        def __iter__(self):
            with pytest.raises(TypeError):owner.__init__(f['base'],f['reader'],f['keys'],ctx,**P)
            with pytest.raises(RuntimeError):owner.admission()
            return iter(((10,),(10,)))
    owner.__init__(f['base'],f['reader'],{'inputs':Rows(),'interface':13},ctx,**P)
    assert owner.archive()==f['model'].archive() and not owner.same_owner(f['model'])


def test_cold_subclass_exposure_never_observes_partial_admission():
    from threading import Event
    f=fixture();ctx=context(f['origins']);packet=f['model'].execute(ctx).archive();published=Event();owners=[]
    class Published(cpp.NativeAvailableLiteralFiniteFieldResult):
        def __new__(cls):
            value=super().__new__(cls);owners.append(value);published.set();return value
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(Published.from_archive,packet,ctx,**P)
        assert published.wait(10)
        while not future.done():
            try:admission=owners[0].admission()
            except RuntimeError:continue
            assert admission==P
        result=future.result()
    assert result is owners[0] and result.admission()==P and result.archive()==packet


@pytest.mark.parametrize('kind',('model','family','choice'))
def test_cold_subclass_cannot_rebind_existing_reader_owners(kind):
    f=fixture();family,choice=chosen(f);ctx=context(f['origins'])
    value={'model':f['model'],'family':family,'choice':choice}[kind];packet=value.archive()
    class Reused(type(value)):
        def __new__(cls):return value
    with pytest.raises(TypeError):Reused.from_archive(packet,ctx,**P)
    assert value.archive()==packet


def test_cold_resource_parsing_cannot_reinitialize_an_exposed_reader():
    f=fixture();ctx=context(f['origins']);owners=[]
    class Exposed(cpp.PreparedAvailableLiteralFiniteFieldMap):
        def __new__(cls):
            value=super().__new__(cls);owners.append(value);return value
    class Bounds:
        def __len__(self):return 4
        def __iter__(self):
            with pytest.raises(TypeError):owners[0].__init__(f['base'],f['reader'],f['keys'],ctx,**P)
            with pytest.raises(RuntimeError):owners[0].admission()
            return iter(P['limits'])
    value=Exposed.from_archive(f['model'].archive(),ctx,limits=Bounds(),max_bytes=P['max_bytes'])
    assert value.archive()==f['model'].archive()

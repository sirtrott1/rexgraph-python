"""Available native finite fields retain complete owners and independent authority."""
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
from rexgraph_cpp import (
    NativeSource, NativeValueOrigin, NativeAvailabilityContext, NativeInputUseAsset,
    PreparedAvailableFieldProgram, PreparedAvailableFiniteSource,
    NativeAvailableFiniteSections, NativeAvailableFiniteSection,
    PreparedAvailableFiniteFieldMap, NativeAvailableFiniteFieldResult,
    NativeAvailableFiniteFieldBranch, NativeAvailableFiniteFieldEvaluation,
    PreparedAvailableFieldEvidence, field_evidence_origin_payload,
    finite_constraint_origin_payload, finite_interpretation_origin_payload,
    field_input_origin_payloads,
)
from test_relational_field_admission import publish, context
from test_relational_field_evidence_admission import learned, prepared, measured_observation
from test_relational_field_evidence import input_accession, tensor, matrix, parameters
from test_relational_finite_field import asset, declaration


def fixture(values=(2,3,5,7), *, prospective=False, namespace=900001, identified=False):
    if identified:
        owner,program,old_root,origins=prepared();d=parameters();d['parameters']=d['parameters'][:3]
        d['basis']={'shape':(12,3),'entries':tuple(row for row in d['basis']['entries'] if row[1]<3)}
        origins[4]=publish(4,owner,'evidence',field_evidence_origin_payload(program,d))
        root=PreparedAvailableFieldEvidence(old_root.program(),d,4,context(origins));theta=(2,-1,3,0,0)
        first=measured_observation(owner,program,origins,11,(1,2,0,0),theta)
        evidence=root.append_inputs((first,),context(origins))
        second=measured_observation(owner,program,origins,12,(0,1,0,0),theta)
        evidence=evidence.append_inputs((second,),context(origins))
    else:
        owner, program, root, evidence, origins, theta=learned(mask=((0,0),))
    literal=asset(namespace);d=declaration();status='prospective' if prospective else 'observed'
    role='prospective' if prospective else 'evidence'
    origins[400]=publish(400,owner,'input-asset',literal.archive())
    origins[401]=publish(401,owner,role,finite_constraint_origin_payload(literal,d,status))
    finite=PreparedAvailableFiniteSource(literal,d,status,{'asset':400,'constraints':401},context(origins))
    rows=[]
    for use,start in ((111,411),(112,431),(111,443)):
        row={'asset':literal.archive(),'accession':input_accession(),'use':use,'field':tensor(values)}
        payloads=field_input_origin_payloads(program,row)
        keys={'asset':400,'value':start,'accession':start+1,'reconstruction':start+2}
        for name,label in (('value','input-value'),('accession','input-accession'),('reconstruction','reconstruction')):
            origins[keys[name]]=publish(keys[name],owner,label,payloads[name])
        row['asset']=literal;row['origins']=keys;rows.append(row)
    available=evidence.program()
    for key,label in ((80,'program'),(81,'maps'),(82,'metrics')):origins[key]=publish(key,owner,label,program.archive())
    unused=PreparedAvailableFieldProgram(program,{'program':80,'maps':81,'metrics':82},context(origins))
    targets=(available,evidence,unused)
    spec={'profile':'rexgraph.available-relational-finite-field-map.v1','key':6301,'revision':1,'rules':(
        {'key':6401,'bindings':(6201,6202),'symbols':(101,101),'target':0,'inputs':(rows[0],),'unsupported':''},
        {'key':6402,'bindings':(6201,6202),'symbols':(102,102),'target':0,'inputs':(rows[0],),'unsupported':''},
        {'key':6403,'bindings':(6203,6204),'symbols':(101,101),'target':1,'inputs':(rows[1],),'unsupported':''},
        {'key':6404,'bindings':(6201,),'symbols':(103,),'target':None,'inputs':(),'unsupported':'no declared interpretation'},
        {'key':6405,'bindings':(6201,6202),'symbols':(101,102),'target':0,'inputs':(rows[2],),'unsupported':''})}
    payload=finite_interpretation_origin_payload(finite,targets,spec,status,context(origins))
    origins[500]=publish(500,owner,'maps',payload);origins[501]=publish(501,owner,role,payload)
    model=PreparedAvailableFiniteFieldMap(finite,targets,spec,status,{'maps':500,'relationship':501},context(origins))
    return owner,literal,finite,evidence,targets,spec,model,origins


def selected(f, ordinal=1, free=(7,11,13), *, key=601):
    owner,literal,finite,evidence,targets,spec,model,origins=f
    family=model.execute(context(origins));setting=600 if ordinal==1 else 0
    if setting:origins[setting]=publish(setting,owner,'prospective',evidence.parameter_setting_payload(free))
    origins[key]=publish(key,owner,'prospective',family.completion_payload(ordinal,free,setting,context(origins)))
    return family,family.evaluate(ordinal,free,setting,key,context(origins))


def clone_origins(origins):
    return {key:publish(key,NativeSource(origin.source_archive()),origin.inspect()['role'],origin.payload()) for key,origin in origins.items()}


def test_complete_available_carriers_fields_and_independent_completion_receipts():
    f=fixture();owner,literal,finite,evidence,targets,spec,model,origins=f;authority=context(origins)
    sections=finite.sections(authority);assert sections.inspect(authority)=={'candidate_count':27,'section_count':3}
    assert finite.admission()==model.admission()==sections.admission()=={'limits':(4096,16000000,64000000,16384),'max_bytes':256*1024*1024}
    choices=sections.select(0,3,authority)
    assert tuple(s.inspect(authority)['assignment'] for s in choices)==((101,101,101),(102,102,102),(103,103,103))
    assert sections.source(authority).same_owner(finite) and finite.raw().asset().same_owner(literal)
    assert finite.inspect(authority)['status']=='observed'
    family,value=selected(f);authority=context(origins)
    assert family.inspect(authority)=={'branch_count':4}
    branches=family.select(0,4,authority)
    assert tuple(b.inspect(authority)['status'] for b in branches)==('interpreted','interpreted','interpreted','unsupported')
    expected=evidence.execute_inputs((dict(spec['rules'][2]['inputs'][0],asset=literal.archive()),),authority).result().evaluate((7,11,13))
    actual=value.raw().result()
    for node in (1,2,3):assert actual.field_archive(node)==expected.field_archive(node)
    assert actual.field_archive(2,occurrences=True)==expected.field_archive(2,occurrences=True)
    assert value.raw().affine_evaluation().archive()==expected.archive()
    info=value.inspect(authority)
    assert value.admission()==family.admission()==branches[0].admission()==choices[0].admission()==model.admission()
    assert info['parameter_setting_receipt']==origins[600].receipt() and info['choice_receipt']==origins[601].receipt()
    assert value.family(authority).archive()==family.archive() and len(model.targets(authority))==3
    assert len(value.dependency_archives(authority))>len(model.dependency_archives(authority))
    assert PreparedAvailableFiniteSource.from_archive(finite.archive(),authority).archive()==finite.archive()
    assert NativeAvailableFiniteSections.from_archive(sections.archive(),authority).archive()==sections.archive()
    assert PreparedAvailableFiniteFieldMap.from_archive(model.archive(),authority).archive()==model.archive()
    assert NativeAvailableFiniteFieldResult.from_archive(family.archive(),authority).archive()==family.archive()
    assert NativeAvailableFiniteFieldEvaluation.from_archive(value.archive(),authority).archive()==value.archive()


@pytest.mark.parametrize('key',(1,2,3,4,110,111,112,113,114,115,400,401,411,412,413,431,432,433,443,444,445,80,81,82,500,501))
def test_all_targets_and_unmatched_rule_roles_recheck_current_authority(key):
    f=fixture();model,origins=f[-2:];family=model.execute(context(origins))
    denied=context(origins,remove=(key,))
    with pytest.raises(ValueError):model.execute(denied)
    with pytest.raises(ValueError):family.admit(denied)
    with pytest.raises(ValueError):family.select(0,1,denied)
    with pytest.raises(ValueError):NativeAvailableFiniteFieldResult.from_archive(family.archive(),denied,limits=(1,16000000,64000000,16384))


def test_whole_rule_preflight_precedes_first_invalid_reconstruction():
    owner,literal,finite,evidence,targets,spec,model,origins=fixture()
    rules=[dict(r) for r in spec['rules']];first=dict(rules[0]['inputs'][0]);first['accession']=input_accession()
    first['accession']['accession']=matrix(((0,0,0,0),)*4)
    raw=dict(first,asset=literal.archive());raw.pop('origins')
    payloads=field_input_origin_payloads(targets[0].program(),raw)
    for name,label in (('value','input-value'),('accession','input-accession'),('reconstruction','reconstruction')):
        key=first['origins'][name];origins[key]=publish(key,owner,label,payloads[name])
    rules[0]['inputs']=(first,);rules[1]['inputs']=(first,);bad=dict(spec,rules=tuple(rules))
    with pytest.raises(ValueError,match='missing actual value origin'):
        finite_interpretation_origin_payload(finite,targets,bad,'observed',context(origins,remove=(443,)))
    payload=finite_interpretation_origin_payload(finite,targets,bad,'observed',context(origins))
    origins[500]=publish(500,owner,'maps',payload);origins[501]=publish(501,owner,'evidence',payload)
    with pytest.raises(ValueError):PreparedAvailableFiniteFieldMap(finite,targets,bad,'observed',{'maps':500,'relationship':501},context(origins))


def test_live_owner_cold_adoption_and_passive_escape_are_explicit():
    f=fixture();owner,literal,finite,evidence,targets,spec,model,origins=f
    other=context(clone_origins(origins))
    with pytest.raises(ValueError,match='live value origin owner'):model.admit(other)
    assert PreparedAvailableFiniteFieldMap.from_archive(model.archive(),other).archive()==model.archive()
    assert model.raw().execute().inspect()['branch_count']==4
    rules=[dict(r) for r in spec['rules']];row=dict(rules[0]['inputs'][0]);row['asset']=NativeInputUseAsset.from_archive(literal.archive());rules[0]['inputs']=(row,)
    with pytest.raises(ValueError,match='foreign actual source asset owner'):
        finite_interpretation_origin_payload(finite,targets,dict(spec,rules=tuple(rules)),'observed',context(origins))
    row['asset']=literal.archive()
    with pytest.raises(TypeError,match='actual finite asset owner'):
        finite_interpretation_origin_payload(finite,targets,dict(spec,rules=tuple(rules)),'observed',context(origins))


def test_fixed_affine_choices_and_input_specific_provenance_remain_separate():
    f=fixture();family,value=selected(f);origins=f[-1];authority=context(origins)
    for key in (600,601):
        denied=context(origins,remove=(key,))
        with pytest.raises(ValueError):value.admit(denied)
        with pytest.raises(ValueError):NativeAvailableFiniteFieldEvaluation.from_archive(value.archive(),denied)
    with pytest.raises(ValueError):family.evaluate(0,(),600,601,authority)
    with pytest.raises(ValueError):family.evaluate(0,(1,),0,601,authority)
    with pytest.raises(ValueError):family.evaluate(1,(7,11),600,601,authority)
    with pytest.raises(ValueError):family.evaluate(3,(),0,601,authority)
    _,fixed=selected(f,0,(),key=602);authority=context(origins)
    assert fixed.inspect(authority)['parameter_setting_receipt'] is None
    assert fixed.inspect(authority)['choice_receipt']==origins[602].receipt()
    second=fixture(namespace=900002);new_family=second[-2].execute(context(second[-1]));second[-1][600]=publish(600,second[0],'prospective',second[3].parameter_setting_payload((7,11,13)))
    second[-1][601]=publish(601,second[0],'prospective',origins[601].payload())
    with pytest.raises(ValueError,match='role or complete content'):new_family.evaluate(1,(7,11,13),600,601,context(second[-1]))
    assert family.archive()!=new_family.archive()


def test_identified_affine_empty_xi_still_requires_its_global_setting_receipt():
    f=fixture(identified=True);origins=f[-1];family=f[-2].execute(context(origins))
    assert f[3].evidence().identity()['free_parameter_count']==0
    with pytest.raises(ValueError):family.completion_payload(1,(),0,context(origins))
    _,value=selected(f,1,());authority=context(origins)
    assert value.inspect(authority)['parameter_setting_receipt']==origins[600].receipt()
    assert value.raw().free_values_archive()
    assert NativeAvailableFiniteFieldEvaluation.from_archive(value.archive(),authority).archive()==value.archive()


def test_escaped_sections_branches_choices_concurrent_cold_replay_and_bounds():
    f=fixture(prospective=True);origins=f[-1];family,value=selected(f);authority=context(origins)
    section=f[2].sections(authority).select(0,1,authority)[0];branch=family.select(1,1,authority)[0]
    family_bytes=family.archive();choice_bytes=value.archive();del f,family;gc.collect()
    assert section.family(authority).inspect(authority)['section_count']==3
    assert branch.family(authority).archive()==family_bytes and value.family(authority).archive()==family_bytes
    with ThreadPoolExecutor(max_workers=6) as pool:
        replay=tuple(pool.map(lambda _:NativeAvailableFiniteFieldEvaluation.from_archive(choice_bytes,authority).archive(),range(12)))
    assert replay==(choice_bytes,)*12
    with pytest.raises((ValueError,RuntimeError,OverflowError)):section.inspect(authority,max_records=2)
    with pytest.raises((ValueError,RuntimeError,OverflowError)):value.dependency_archives(authority,max_records=1)
    with pytest.raises((ValueError,RuntimeError)):NativeAvailableFiniteFieldEvaluation.from_archive(choice_bytes,authority,max_bytes=32)
    corrupt=bytearray(choice_bytes);corrupt[-1]^=1
    with pytest.raises(ValueError):NativeAvailableFiniteFieldEvaluation.from_archive(bytes(corrupt),authority)
    for cls in (PreparedAvailableFiniteSource,NativeAvailableFiniteSections,NativeAvailableFiniteSection,PreparedAvailableFiniteFieldMap,
                NativeAvailableFiniteFieldResult,NativeAvailableFiniteFieldBranch,NativeAvailableFiniteFieldEvaluation):
        with pytest.raises(RuntimeError):cls.__new__(cls).admit(authority)


def authority_from_fixture(path, *, withhold=None):
    data=path.read_bytes();offset=0;roles=('program','maps','metrics','input-asset','input-value','input-accession',
        'reconstruction','target-asset','target-value','interface','evidence','prospective')
    def u64():
        nonlocal offset
        value=struct.unpack_from('<Q',data,offset)[0];offset+=8;return value
    def blob():
        nonlocal offset
        size=u64();value=data[offset:offset+size];offset+=size;return value
    count=u64();grants=[]
    for _ in range(count):
        key=u64();role=data[offset];offset+=1;history=blob();payload=blob();namespace=u64();sequence=u64()
        source=NativeSource(history);origin=NativeValueOrigin(key,history,source.identity['sequence'],roles[role],payload)
        if key==withhold:sequence=origin.inspect()['sequence']-1
        grants.append({'origin':origin,'namespace':namespace,'sequence':sequence})
    assert offset==len(data)
    return NativeAvailabilityContext(tuple(grants))


def test_native_fixture_full_available_packets_and_independent_registry_cold_reader(tmp_path):
    binary=os.environ['REXGRAPH_CPP_AVAILABLE_FINITE_FIXTURE'];subprocess.run((binary,'--write',str(tmp_path)),check=True)
    authority=authority_from_fixture(tmp_path/'authority.bin');family=(tmp_path/'result.packet').read_bytes();choice=(tmp_path/'choice.packet').read_bytes()
    # The native fixture records its explicitly larger exact integer policy.
    limits=(4096,16000000,64000000,1<<24)
    assert NativeAvailableFiniteFieldResult.from_archive(family,authority,limits=limits).archive()==family
    selected=NativeAvailableFiniteFieldEvaluation.from_archive(choice,authority,limits=limits);assert selected.archive()==choice and selected.inspect(authority)['ordinal']==1
    # This native fixture's authentic publishers have positive sequence cutoffs.
    for key in (1,2,3,4,10,11,12,13,30,31,400,401,411,412,413,431,432,433,443,444,445,80,81,82,500,501,600,601):
        denied=authority_from_fixture(tmp_path/'authority.bin',withhold=key)
        with pytest.raises(ValueError):NativeAvailableFiniteFieldEvaluation.from_archive(choice,denied,limits=limits)
    code='from pathlib import Path;import sys;from test_relational_finite_field_admission import authority_from_fixture;from rexgraph_cpp import NativeAvailableFiniteFieldEvaluation;p=Path(sys.argv[1]);a=authority_from_fixture(p/"authority.bin");b=(p/"choice.packet").read_bytes();r=NativeAvailableFiniteFieldEvaluation.from_archive(b,a,limits=(4096,16000000,64000000,1<<24));assert r.archive()==b and r.family(a).inspect(a)["branch_count"]==4'
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent))
    subprocess.run((sys.executable,'-c',code,str(tmp_path)),cwd=tmp_path,env=env,check=True)

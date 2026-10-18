"""Independent whole inputs retain complete operator authority and source lineage."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
from functools import lru_cache
from pathlib import Path
import gc
import os
import subprocess
import sys

import pytest
from rexgraph_cpp import (
    NativeInputUseAsset, PreparedAvailableFiniteSource,
    PreparedAvailableFiniteFieldOperator, NativeAvailableFiniteOperatorInstance,
    NativeAvailableFiniteOperatorResult, NativeAvailableFiniteOperatorBranch,
    NativeAvailableFiniteOperatorEvaluation, finite_operator_origin_payload,
    finite_operator_input_origin_payloads, finite_constraint_origin_payload,
    finite_operator_admission_build_info,
)
from test_relational_field_admission import publish, context
from test_relational_field_program import source, tensor
from test_relational_field_evidence import reference, read_vector, read_matrix
from test_relational_finite_field import values
from test_relational_finite_field_admission import (
    fixture as map_fixture, clone_origins, authority_from_fixture,
)

_NATIVE_LIMITS=(4096,16000000,64000000,1<<24)
_KEYS=('transport_maps','transport_relationship','maps','relationship','instance')


@lru_cache(maxsize=4)
def reference_fixture(identified=False,prospective=False):
    f=map_fixture(identified=identified,prospective=prospective)
    owner,literal,finite,evidence,targets,spec,model,base=f
    origins=dict(base);status='prospective' if prospective else 'observed'
    ports=tuple({'key':key,'rule':rule,'input':0} for key,rule in ((9003,6403),(9004,6405),(9001,6401),(9002,6402)))
    d={'profile':'rexgraph.relational-finite-operator.v1','key':8001,'revision':1,'ports':ports}
    payload=finite_operator_origin_payload(model,d,status,context(origins))
    origins[1000]=publish(1000,owner,'maps',payload)
    origins[1001]=publish(1001,owner,'prospective' if prospective else 'evidence',payload)
    op=PreparedAvailableFiniteFieldOperator(model,d,status,{'maps':1000,'relationship':1001},context(origins))
    return f,op,origins,d


def independent(x=(2,3,5,7),*,namespace=990002,identified=False,prospective=False,instance_key=8101):
    f,op,base,d=reference_fixture(identified,prospective);origins=dict(base)
    held_owner=source(namespace)
    asset=NativeInputUseAsset(held_owner.archive,{'profile':'rexgraph.relational-input-use-asset.v1',
        'key':801,'channel':901,'encoding':'UTF-8','bytes':b'oak / cedar',
        'uses':tuple({'key':key,'cells':((0,10),(0,20),(0,30),(1,100),(1,101)),
            'occurrences':tuple((1,i) for i in range(1,9)),'spans':spans}
            for key,spans in ((311,((0,3),(0,0),(1,3))),(312,((6,11),(6,6),(8,11)))))})
    transport={'profile':'rexgraph.relational-finite-source-transport.v1','key':8201,'revision':1,
        'source_map':{'cells':((0,1,2),(0,1)),'occurrences':((),tuple(range(8)))},
        'uses':({'reference':111,'input':311},{'reference':112,'input':312})}
    status='prospective' if prospective else 'observed';role='prospective' if prospective else 'evidence'
    schema=op.source_declaration(asset,transport)
    origins[1100]=publish(1100,held_owner,'input-asset',asset.archive())
    origins[1101]=publish(1101,held_owner,role,finite_constraint_origin_payload(asset,schema,status))
    finite=PreparedAvailableFiniteSource(asset,schema,status,{'asset':1100,'constraints':1101},context(origins))
    rows=[]
    for i,port in enumerate(d['ports']):
        use=312 if port['rule']==6403 else 311
        actual=op.input_schema(asset,transport,port['key']);field=tensor(x)
        payloads=finite_operator_input_origin_payloads(actual,use,field)
        assert payloads['asset']==asset.archive()
        keys={'asset':1100,'value':1110+3*i,'accession':1111+3*i,'reconstruction':1112+3*i}
        for name,label in (('value','input-value'),('accession','input-accession'),('reconstruction','reconstruction')):
            origins[keys[name]]=publish(keys[name],held_owner,label,payloads[name])
        rows.append({'port':port['key'],'schema':actual,'use':use,'field':field,'origins':keys})
    rows=tuple(reversed(rows))
    declaration={'profile':'rexgraph.relational-finite-operator-instance.v1','key':instance_key,'revision':1,'transport':transport}
    keys=dict(zip(_KEYS,range(1140,1145),strict=True))
    payload=op.transport_payload(finite,transport,status,context(origins))
    for name,label in (('transport_maps','maps'),('transport_relationship',role)):
        origins[keys[name]]=publish(keys[name],held_owner,label,payload)
    payload=op.child_map_payload(finite,declaration,rows,status,context(origins))
    for name,label in (('maps','maps'),('relationship',role)):
        origins[keys[name]]=publish(keys[name],held_owner,label,payload)
    payload=op.instance_payload(finite,declaration,rows,status,keys,context(origins))
    origins[1144]=publish(1144,held_owner,'prospective',payload)
    instance=op.instantiate(finite,declaration,rows,status,keys,context(origins))
    return f,op,instance,origins,finite,rows,declaration,keys,held_owner


def complete(family,f,origins,held_owner,xi=(7,11,13),ordinal=1,key=1501):
    setting=1500 if ordinal==1 else 0
    if setting:origins[setting]=publish(setting,held_owner,'prospective',f[3].parameter_setting_payload(xi))
    payload=family.completion_payload(ordinal,xi,setting,context(origins))
    origins[key]=publish(key,held_owner,'prospective',payload)
    return family.evaluate(ordinal,xi,setting,key,context(origins))


@pytest.mark.parametrize('trial',range(12))
def test_independent_whole_fields_correlated_maps_original_occurrences_and_unknowns(trial):
    x=(Q(trial+1,3),Q(trial-2,5),Q(2*trial+1,7),Q(3-trial,2));xi=(Q(trial-2),Q(trial+3),Q(2*trial))
    f,op,instance,origins,finite,rows,d,keys,held=independent(x,instance_key=8300+trial)
    authority=context(origins);before=op.reference(authority).archive();family=instance.execute(authority)
    assert instance.source(authority).same_owner(finite) and instance.operator_program(authority).same_owner(op)
    assert tuple(b.inspect(authority)['status'] for b in family.select(0,4,authority))==('interpreted','interpreted','interpreted','unsupported')
    affine=family.select(1,1,authority)[0].raw().raw().result()
    assert affine.field_archive(2)['free_parameter_count']==3 and affine.field_archive(2)['direction_nonzeros']>0
    choice=complete(family,f,origins,held,xi);whole=choice.raw().raw().result()
    evidence=f[3].evidence().inspect();particular=read_vector(evidence['particular']);kernel=read_matrix(evidence['kernel'])
    theta=tuple(p+sum(q*v for q,v in zip(row,xi,strict=True)) for p,row in zip(particular,kernel,strict=True))
    lower,flags,later=reference(x,theta)
    for node,expected in ((1,x),(2,lower),(3,later)):assert values(whole.field_archive(node))==tuple(expected)
    assert values(whole.field_archive(2,occurrences=True))==flags and len(flags)==19
    assert op.reference(context(origins)).archive()==before
    assert instance.select_inputs(0,4,context(origins))[0]['schema'].asset().same_owner(finite.raw().asset())
    assert finite.raw().asset().recover(311)==(b'oak',b'',b'ak')
    assert f[1].recover(111)!=finite.raw().asset().recover(311)
    assert family.select(3,1,context(origins))[0].raw().raw().result() is None


@pytest.mark.parametrize('key',(1,2,3,4,110,111,112,113,114,115,400,401,411,412,413,431,432,433,443,444,445,80,81,82,500,501,1000,1001,1100,1101,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1140,1141,1142,1143,1144))
def test_whole_parent_unused_targets_unmatched_ports_and_instance_recheck_current_authority(key):
    f,op,instance,origins,*_=independent();family=instance.execute(context(origins))
    denied=context(origins,remove=(key,))
    with pytest.raises(ValueError):instance.execute(denied)
    with pytest.raises(ValueError):family.select(0,1,denied)
    with pytest.raises(ValueError,match='missing actual value origin'):
        NativeAvailableFiniteOperatorResult.from_archive(family.archive(),denied)


def test_separate_global_setting_and_input_specific_choice_including_empty_xi():
    f,op,instance,origins,finite,rows,d,keys,held=independent(identified=True)
    family=instance.execute(context(origins));assert f[3].evidence().identity()['free_parameter_count']==0
    with pytest.raises(ValueError):family.completion_payload(1,(),0,context(origins))
    value=complete(family,f,origins,held,())
    for key in (1500,1501):
        with pytest.raises(ValueError):value.admit(context(origins,remove=(key,)))
        with pytest.raises(ValueError):NativeAvailableFiniteOperatorEvaluation.from_archive(value.archive(),context(origins,remove=(key,)))
    info=value.inspect(context(origins));assert info['parameter_setting_receipt']==origins[1500].receipt()
    assert info['choice_receipt']==origins[1501].receipt()
    fixed=complete(family,f,origins,held,(),ordinal=0,key=1502)
    assert fixed.inspect(context(origins))['parameter_setting_receipt'] is None
    with pytest.raises(ValueError):family.evaluate(3,(),0,1502,context(origins))


def test_live_owner_withdrawal_cutoffs_and_explicit_cold_adoption():
    f,op,instance,origins,finite,rows,d,keys,held=independent();family=instance.execute(context(origins))
    clone=context(clone_origins(origins))
    with pytest.raises(ValueError,match='live value origin owner'):instance.admit(clone)
    assert NativeAvailableFiniteOperatorResult.from_archive(family.archive(),clone).archive()==family.archive()
    for key in (1001,1101,1113,1144):
        cutoff=origins[key].inspect()['sequence']-1
        with pytest.raises(ValueError):instance.execute(context(origins,limits={key:cutoff}))
        with pytest.raises(ValueError):instance.execute(context(origins,namespaces={key:123456}))
    raw=instance.raw();assert raw.execute().inspect()['branch_count']==4
    foreign=PreparedAvailableFiniteFieldOperator.from_archive(op.archive(),clone)
    wrong=tuple(dict(row,schema=foreign.input_schema(finite.raw().asset(),d['transport'],row['port'])) for row in rows)
    with pytest.raises(ValueError,match='its operator'):op.child_map_declaration(d,wrong)


def test_complete_ports_native_payloads_positive_instance_and_bounded_metadata():
    f,op,instance,origins,finite,rows,d,keys,held=independent();authority=context(origins)
    for bad in (rows[:-1],(*rows[:-1],rows[0]),tuple(dict(r,schema=rows[1]['schema']) for r in rows)):
        with pytest.raises(ValueError):op.child_map_declaration(d,bad)
    with pytest.raises(ValueError):op.instantiate(finite,d,rows,'observed',dict(keys,instance=0),authority)
    with pytest.raises(ValueError):op.instantiate(finite,d,rows,'observed',keys,context(origins,remove=(1144,)))
    with pytest.raises((ValueError,RuntimeError)):op.child_map_declaration(d,rows,max_records=1)
    with pytest.raises((ValueError,RuntimeError)):instance.dependency_archives(authority,max_records=1)
    assert op.child_map_declaration(d,rows)['authority']=='raw-none'
    assert instance.use_correspondences(0,2,authority)==({'reference':111,'input':311},{'reference':112,'input':312})
    assert len(instance.dependency_archives(authority))>len(op.dependency_archives(authority))


def test_passive_input_payloads_do_not_authorize_missing_late_ports_before_bad_shape():
    f,op,instance,origins,finite,rows,d,keys,held=independent()
    bad=[dict(r) for r in rows];bad[-1]['field']=tensor((1,2,3));row=bad[-1]
    payloads=finite_operator_input_origin_payloads(row['schema'],row['use'],row['field'])
    for name,label in (('value','input-value'),('accession','input-accession'),('reconstruction','reconstruction')):
        origins[row['origins'][name]]=publish(row['origins'][name],held,label,payloads[name])
    with pytest.raises(ValueError,match='missing actual value origin'):
        op.child_map_payload(finite,d,tuple(bad),'observed',context(origins,remove=(1113,)))
    with pytest.raises(ValueError):op.instantiate(finite,d,tuple(bad),'observed',keys,context(origins))


def test_full_canonical_codecs_provenance_tamper_mutation_and_escaped_readers():
    f,op,instance,origins,finite,rows,d,keys,held=independent();family=instance.execute(context(origins))
    choice=complete(family,f,origins,held);authority=context(origins)
    for value in (op,instance,family,choice):
        image=value.archive();assert type(value).from_archive(image,authority).archive()==image
        with pytest.raises(ValueError):type(value).from_archive(image+b'x',authority)
        with pytest.raises(ValueError):type(value).from_archive(image,authority,max_bytes=32)
    other=independent(namespace=990003);second=other[2].execute(context(other[3]))
    other[3][1500]=publish(1500,other[-1],'prospective',f[3].parameter_setting_payload((7,11,13)))
    other[3][1501]=publish(1501,other[-1],'prospective',origins[1501].payload())
    with pytest.raises(ValueError):second.evaluate(1,(7,11,13),1500,1501,context(other[3]))
    assert second.archive()!=family.archive()
    with pytest.raises(TypeError,match='immutable'):op.__init__(f[-2],reference_fixture()[3],'observed',{'maps':1000,'relationship':1001},authority)
    with pytest.raises((AttributeError,TypeError)):op.fields=(1,2)
    image=choice.archive()
    def read(_):
        value=NativeAvailableFiniteOperatorEvaluation.from_archive(image,authority)
        return value.archive(),value.family(authority).instance(authority).source(authority).raw().asset().recover(312)
    with ThreadPoolExecutor(max_workers=6) as pool:assert tuple(pool.map(read,range(6)))==((image,(b'cedar',b'',b'dar')),)*6
    del op,instance,family,rows,finite;gc.collect()
    assert values(choice.raw().raw().result().field_archive(2))
    for cls in (PreparedAvailableFiniteFieldOperator,NativeAvailableFiniteOperatorInstance,NativeAvailableFiniteOperatorResult,NativeAvailableFiniteOperatorBranch,NativeAvailableFiniteOperatorEvaluation):
        with pytest.raises(RuntimeError):cls.__new__(cls).admit(authority)
    assert finite_operator_admission_build_info()['scalar_policy']=='gmp-rational-exact'


def test_actual_native_packets_and_separate_trusted_registry_cold_process():
    root=Path(os.environ['REXGRAPH_CPP_AVAILABLE_OPERATOR_FIXTURE_DIR'])
    authority=authority_from_fixture(root/'authority.bin')
    family=(root/'family.packet').read_bytes();choice=(root/'choice.packet').read_bytes()
    assert NativeAvailableFiniteOperatorResult.from_archive(family,authority,limits=_NATIVE_LIMITS).archive()==family
    value=NativeAvailableFiniteOperatorEvaluation.from_archive(choice,authority,limits=_NATIVE_LIMITS)
    assert value.archive()==choice and value.family(authority).inspect(authority)['branch_count']==4
    code='from pathlib import Path;import sys;from test_relational_finite_field_admission import authority_from_fixture;from rexgraph_cpp import NativeAvailableFiniteOperatorEvaluation;p=Path(sys.argv[1]);a=authority_from_fixture(p/"authority.bin");b=(p/"choice.packet").read_bytes();v=NativeAvailableFiniteOperatorEvaluation.from_archive(b,a,limits=(4096,16000000,64000000,1<<24));assert v.archive()==b and v.family(a).instance(a).source(a).raw().asset().recover(311)==(b"oak",)'
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run((sys.executable,'-c',code,str(root)),cwd=root,env=env,check=True)

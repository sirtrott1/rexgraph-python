"""Raw native finite carriers retain correlated alternatives and whole fields."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
import gc
import os
import subprocess
import sys

import pytest
from rexgraph_cpp import (NativeFiniteSource, NativeFiniteSections, PreparedFiniteFieldMap,
    NativeFiniteFieldResult, NativeFiniteFieldBranch, NativeFiniteFieldEvaluation,
    NativeInputUseAsset, NativeInputAccession, PreparedFieldProgram, PreparedAccessionFieldProgram,
    PreparedFieldEvidence)
from test_relational_field_program import source,tensor
from test_relational_field_evidence import learning_declaration,input_accession,parameters,actual_asset,reference
from test_relational_field_local import values


def asset(namespace=900001):
    owner=source(namespace)
    declaration={'profile':'rexgraph.relational-input-use-asset.v1','key':601,'channel':602,'encoding':'UTF-8',
        'bytes':b'caf\xc3\xa9 / caf\xc3\xa9','uses':tuple({'key':key,'cells':((0,10),(0,20),(0,30),(1,100),(1,101)),
        'occurrences':tuple((1,i) for i in range(1,9)),'spans':spans} for key,spans in ((111,((0,5),(0,0),(3,5))),(112,((8,13),(0,3),(10,13)))))}
    return NativeInputUseAsset(owner.archive,declaration)


def declaration():
    return {'profile':'rexgraph.relational-finite-source.v1','key':6001,'revision':1,
        'symbols':({'key':101,'definition':b'same literal meaning alternative'},{'key':102,'definition':b'same literal meaning alternative'},{'key':103,'definition':b''}),
        'variables':tuple({'cell':(0,i),'domain':(101,102,103)} for i in range(3)),
        'charts':tuple({'key':6101+i,'name':'shared chart '+str(i),'patch':((0,0),(0,1),(0,2),(1,i)),
                       'support':((0,i),(0,i+1)),'allowed':((101,101),(102,102),(103,103))} for i in range(2)),
        'bindings':({'key':6201,'use':111,'cell':(0,0)},{'key':6202,'use':111,'cell':(0,1)},
                    {'key':6203,'use':112,'cell':(0,0)},{'key':6204,'use':112,'cell':(0,1)},{'key':6205,'use':111,'cell':(0,2)})}


def fixture(v=(Q(2),Q(3),Q(5),Q(7)), *, namespace=900001,unsupported=True):
    literal=asset(namespace);finite=NativeFiniteSource(literal,declaration())
    program=PreparedAccessionFieldProgram(PreparedFieldProgram(source().archive,learning_declaration()).archive())
    root=PreparedFieldEvidence(program,parameters())
    measured=actual_asset(910001,9001,b'whole historical field measurement');training=asset(910002)
    train_input=NativeInputAccession(training,program,input_accession());train=program.execute((train_input.admit(111,tensor((1,2,0,0))),))
    learned=root.append(({'key':1,'input':train,'target_asset':measured,'target_use':9001,'target':tensor(reference((1,2,0,0),(2,-1,3,0,0))[0]),'observed':((0,0),(0,1),(0,2))},))
    accession=NativeInputAccession(literal,program,input_accession());first=accession.admit(111,tensor(v));second=accession.admit(112,tensor(v))
    rules=[{'key':6401,'bindings':(6201,6202),'symbols':(101,101),'target':0,'inputs':(first,),'unsupported':''},
           {'key':6402,'bindings':(6201,6202),'symbols':(102,102),'target':0,'inputs':(first,),'unsupported':''},
           {'key':6403,'bindings':(6203,6204),'symbols':(101,101),'target':1,'inputs':(second,),'unsupported':''}]
    if unsupported:rules.append({'key':6404,'bindings':(6201,),'symbols':(103,),'target':None,'inputs':(),'unsupported':'no declared continuous interpretation'})
    spec={'profile':'rexgraph.relational-finite-field-map.v1','key':6301,'revision':1,'rules':tuple(rules)}
    return literal,finite,program,learned,PreparedFiniteFieldMap(finite,(program,learned),spec),spec


def test_correlated_shared_cell_constraints_retain_actual_bytes_and_opaque_symbols():
    literal,finite,program,learned,model,spec=fixture();sections=finite.sections()
    assert sections.inspect()=={'candidate_count':27,'section_count':3}
    assert sections.select(0,3)==((101,101,101),(102,102,102),(103,103,103))
    assert sections.source().same_owner(finite) and finite.asset().same_owner(literal)
    assert finite.asset().recover(111)==literal.recover(111) and finite.asset().recover(112)==literal.recover(112)
    info=finite.inspect();assert info['symbols'][0]['definition']==info['symbols'][1]['definition'] and info['symbols'][0]['key']!=info['symbols'][1]['key']
    assert info['bindings'][0]['cell']==info['bindings'][2]['cell'] and info['bindings'][0]['use']!=info['bindings'][2]['use']
    assert NativeFiniteSource.from_archive(finite.archive()).archive()==finite.archive()
    assert not NativeFiniteSource.from_archive(finite.archive()).same_owner(finite)
    with pytest.raises((ValueError,RuntimeError,OverflowError)):finite.inspect(max_records=2)
    with pytest.raises((ValueError,IndexError)):sections.select(2,2)


@pytest.mark.parametrize('v',((2,3,5,7),(11,-2,3,1),(-3,0,2,5),(0,0,0,0)))
def test_complete_native_branches_preserve_full_fixed_affine_fields_and_choices(v):
    literal,finite,program,learned,model,spec=fixture(v);family=model.execute();branches=family.select(0,4)
    assert family.sections().inspect()=={'candidate_count':27,'section_count':3}
    assert tuple(b.inspect()['status'] for b in branches)==('interpreted','interpreted','interpreted','unsupported')
    assert tuple(b.inspect()['rule'] for b in branches)==(6401,6403,6402,6404)
    assert branches[0].result().field_archive(1)==tensor(v) and branches[3].result() is None
    assert branches[0].result().archive()==branches[2].result().archive()
    fixed=family.evaluate(0,());affine_branch=branches[1].result();axes=learned.identity()['free_parameter_count'] if 'free_parameter_count' in learned.identity() else learned.identity()['free_parameters']
    # The native family's exact named free axes determine the full choice.
    choice=tuple(Q(i+3) for i in range(axes));selected=family.evaluate(1,choice)
    expected=affine_branch.evaluate(choice)
    for node in (1,2,3):assert selected.result().field_archive(node)==expected.field_archive(node)
    assert selected.result().field_archive(2,occurrences=True)==expected.field_archive(2,occurrences=True)
    assert selected.affine_evaluation().archive()==expected.archive()
    assert selected.family().archive()==family.archive() and selected.ordinal()==1
    assert values(selected.free_values_archive())==choice
    assert NativeFiniteFieldResult.from_archive(family.archive()).archive()==family.archive()
    assert NativeFiniteFieldEvaluation.from_archive(selected.archive()).archive()==selected.archive()
    assert fixed.affine_evaluation() is None and fixed.result().archive()==branches[0].result().archive()
    assert family.evaluate(0,()).result().archive()==family.evaluate(2,()).result().archive()
    assert family.evaluate(0,()).archive()!=family.evaluate(2,()).archive()
    with pytest.raises((ValueError,RuntimeError)):family.evaluate(0,(Q(1),))
    with pytest.raises((ValueError,RuntimeError)):family.evaluate(1,choice[:-1])
    with pytest.raises((ValueError,RuntimeError)):family.evaluate(3,())
    del model,family,branches,finite,literal,program,learned;gc.collect()
    assert selected.family().inspect()['branch_count']==4
    assert selected.affine_evaluation().family().field_archive(2)['free_parameter_count']==axes


def test_unmapped_empty_complete_and_resource_refusal_are_distinct():
    literal,finite,program,learned,model,spec=fixture(unsupported=False);family=model.execute()
    missing=family.select(3,1)[0];assert missing.inspect()['status']=='unmapped' and missing.inspect()['rule'] is None and missing.result() is None
    d=declaration();d['charts'][0]['allowed']=((101,101),);d['charts'][1]['allowed']=((102,102),)
    empty=NativeFiniteSource(literal,d).sections();assert empty.inspect()['candidate_count']==27 and empty.inspect()['section_count']==0
    assert empty.select(0,0)==()
    with pytest.raises((ValueError,RuntimeError,OverflowError)):NativeFiniteSource(literal,declaration(),limits=(4096,16000000,1,16384)).sections()


def test_chart_tuple_inspection_respects_the_complete_record_bound():
    finite=NativeFiniteSource(asset(),declaration())
    d=declaration()
    count=len(d['symbols'])+len(d['variables'])+len(d['charts'])+len(d['bindings'])
    count+=sum(len(s['definition']) for s in d['symbols'])
    count+=sum(len(v['domain']) for v in d['variables'])
    count+=sum(len(c['name'])+len(c['patch'])+len(c['support'])+sum(len(t)+1 for t in c['allowed']) for c in d['charts'])
    assert finite.inspect(max_records=count)['charts']==finite.inspect()['charts']
    with pytest.raises((ValueError,RuntimeError,OverflowError)):finite.inspect(max_records=count-1)


@pytest.mark.parametrize('bad',('asset','program','missing_node','duplicate_node','use','symbol'))
def test_live_equal_bytes_foreign_owners_and_invalid_actual_rule_images_reject(bad):
    literal,finite,program,learned,model,spec=fixture();rules=[dict(r) for r in spec['rules']]
    if bad=='asset':
        foreign=NativeInputUseAsset.from_archive(literal.archive());rules[0]['inputs']=(NativeInputAccession(foreign,program,input_accession()).admit(111,tensor((2,3,5,7))),)
    elif bad=='program':
        foreign=PreparedAccessionFieldProgram(program.archive());rules[0]['inputs']=(NativeInputAccession(literal,foreign,input_accession()).admit(111,tensor((2,3,5,7))),)
    elif bad=='missing_node':rules[0]['inputs']=()
    elif bad=='duplicate_node':rules[0]['inputs']*=2
    elif bad=='use':rules[0]['bindings']=(6203,6204)
    else:rules[0]['symbols']=(999,999)
    with pytest.raises((ValueError,RuntimeError)):PreparedFiniteFieldMap(finite,(program,learned),dict(spec,rules=tuple(rules)))


def test_equal_fields_and_literal_definition_bytes_keep_original_use_and_cut_history():
    first=fixture(namespace=900001)[4].execute();second=fixture(namespace=900002)[4].execute()
    assert first.select(0,1)[0].result().field_archive(2)==second.select(0,1)[0].result().field_archive(2)
    assert first.archive()!=second.archive() and first.evaluate(0,()).archive()!=second.evaluate(0,()).archive()
    escaped=first.select(1,1)[0];original=first.archive();del first;gc.collect()
    assert escaped.family().archive()==original and escaped.inspect()['ordinal']==1


def test_raw_immutable_handles_native_bulk_readers_and_canonical_tamper_rejection():
    literal,finite,program,learned,model,spec=fixture();family=model.execute();expected=family.archive()
    with ThreadPoolExecutor(max_workers=6) as pool:images=tuple(pool.map(lambda _:NativeFiniteFieldResult.from_archive(expected).archive(),range(12)))
    assert images==(expected,)*12
    bad=bytearray(expected);bad[-1]^=1
    with pytest.raises((ValueError,RuntimeError)):NativeFiniteFieldResult.from_archive(bytes(bad))
    with pytest.raises((ValueError,RuntimeError)):NativeFiniteFieldResult.from_archive(expected,max_bytes=32)
    with pytest.raises(TypeError):finite.__init__(literal,declaration())
    with pytest.raises(TypeError):model.__init__(finite,(program,),spec)
    for cls in (NativeFiniteSource,NativeFiniteSections,PreparedFiniteFieldMap,NativeFiniteFieldResult,NativeFiniteFieldBranch,NativeFiniteFieldEvaluation):
        value=cls.__new__(cls)
        with pytest.raises(RuntimeError):value.inspect() if hasattr(value,'inspect') else value.archive()
    with pytest.raises((AttributeError,TypeError)):finite.hidden_symbols=(1,2,3)


def test_native_fixture_raw_packets_and_standalone_cold_reader(tmp_path):
    executable=os.environ['REXGRAPH_CPP_FINITE_FIXTURE'];family=tmp_path/'family.packet';choice=tmp_path/'choice.packet'
    subprocess.run((executable,'--write',str(family)),check=True);subprocess.run((executable,'--write-choice',str(choice)),check=True)
    # The native fixture stores its explicitly larger integer policy.
    limits=(4096,16000000,64000000,1<<24)
    result=NativeFiniteFieldResult.from_archive(family.read_bytes(),limits=limits);evaluation=NativeFiniteFieldEvaluation.from_archive(choice.read_bytes(),limits=limits)
    assert result.archive()==family.read_bytes() and evaluation.archive()==choice.read_bytes()
    assert evaluation.family().archive()==result.archive() and evaluation.ordinal()==1
    code='from pathlib import Path;from rexgraph_cpp import NativeFiniteFieldEvaluation;import sys;p=Path(sys.argv[1]);r=NativeFiniteFieldEvaluation.from_archive(p.read_bytes(),limits=(4096,16000000,64000000,1<<24));assert r.archive()==p.read_bytes() and r.family().inspect()["branch_count"]==4'
    subprocess.run((sys.executable,'-c',code,str(choice)),cwd=tmp_path,check=True)

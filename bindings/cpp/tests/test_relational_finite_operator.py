"""Installed raw operators retain independent source and whole field ancestry."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
from pathlib import Path
import copy
import gc
import os

import pytest
from rexgraph_cpp import (PreparedFiniteFieldOperator, NativeFiniteSource,
    NativeFiniteSourceTransport, NativeFiniteOperatorInstance, NativeFiniteOperatorResult,
    NativeFiniteOperatorEvaluation, NativeInputUseAsset, finite_operator_build_info)
from test_relational_finite_field import tensor, values

_NATIVE_LIMITS=(4096,16000000,64000000,1<<24)


def fixture():
    path=os.environ.get('REXGRAPH_CPP_OPERATOR_FIXTURE_DIR')
    if not path:pytest.skip('native operator fixture export not provided')
    family=NativeFiniteOperatorResult.from_archive((Path(path)/'family.packet').read_bytes(),limits=_NATIVE_LIMITS)
    instance=family.instance();operator=instance.operator_program()
    return operator,instance,instance.source().asset(),instance.source_transport().inspect()


def declaration(key,transport):
    return {'profile':'rexgraph.relational-finite-operator-instance.v1','key':key,'revision':1,'transport':transport}


def independent(operator,asset,transport,x,key=8801):
    source=NativeFiniteSource(asset,operator.source_declaration(asset,transport))
    ports=operator.select_ports(0,operator.inspect()['port_count'])
    field=tensor((x[2],x[3],x[0],x[1]))
    rows=tuple({'port':p['key'],'schema':operator.input_schema(asset,transport,p['key']),
                'use':312 if p['rule']==6403 else 311,'field':field} for p in reversed(ports))
    return source,declaration(key,transport),rows


def response(x,theta):
    a,b,c,d=x;u,v,w=theta
    return (-a-2*b+2*c+d+u*a+v*b,-b+c+3*d+v*a+w*b,
            -a+(w*c+u*d)/2,-b+(u*c+v*d)/2,
            -a-b+(v*c+w*d)/2,-c+d,2*c-2*d)


def occurrence_response(x,theta):
    a,b,c,d=x;u,v,w=theta
    return (u*a+v*b,v*a+w*b,-a,-b,-a-b,-a-2*b,-b,
            0,0,0,0,0,2*c+d,c+3*d,(w*c+u*d)/2,
            (u*c+v*d)/2,(v*c+w*d)/2,-c+d,2*c-2*d)


@pytest.mark.parametrize('trial',range(12))
def test_independently_supplied_whole_fields_change_complete_correlated_response(trial):
    operator,template,asset,transport=fixture();before=operator.reference().archive()
    x=(Q(trial+1,3),Q(trial-2,5),Q(2*trial+1,7),Q(3-trial,2))
    xi=(Q(trial-2),Q(trial+3),Q(2*trial))
    source,d,rows=independent(operator,asset,transport,x,8300+trial)
    instance=operator.instantiate(source,d,rows);family=instance.execute()
    assert instance.source().same_owner(source) and instance.operator_program().same_owner(operator)
    assert tuple(r['port'] for r in instance.select_inputs(0,4))==tuple(p['key'] for p in operator.select_ports(0,4))
    assert all(r['schema'].asset().same_owner(asset) for r in instance.select_inputs(0,4))
    assert tuple(b.raw().inspect()['status'] for b in family.select(0,4))==('interpreted','interpreted','interpreted','unsupported')
    affine=family.select(1,1)[0].raw().result()
    theta=(-10+4*xi[0],5-2*xi[0],xi[0])
    assert tuple(values(affine.field_archive(2)['particular']))==response(x,(-10,5,0))
    assert affine.field_archive(2)['free_parameter_count']==3
    choice=family.evaluate(1,xi);whole=choice.raw().result()
    expected=response(x,theta)
    assert tuple(values(whole.field_archive(1)))==x
    assert tuple(values(whole.field_archive(2)))==expected
    assert tuple(values(whole.field_archive(3)))==(expected[0]+expected[2],expected[1]+expected[3],
                                                expected[4]+expected[5],expected[0]+expected[6])
    assert tuple(values(whole.field_archive(2,occurrences=True)))==occurrence_response(x,theta)
    assert len(values(whole.field_archive(2,occurrences=True)))==19
    assert operator.reference().archive()==before
    assert family.instance().operator_program().same_owner(operator)
    assert choice.family().archive()==family.archive() and choice.ordinal()==1
    assert family.select(1,1)[0].family().archive()==family.archive()
    assert family.raw().model().source().asset().same_owner(asset)
    assert choice.raw().affine_evaluation().archive()==affine.evaluate(xi).archive()


def test_explicit_variable_use_transport_and_passive_native_schema_keep_provenance():
    operator,template,asset,transport=fixture();source,d,rows=independent(operator,asset,transport,(2,3,5,7))
    owned=NativeFiniteSourceTransport(operator.reference().source(),source,transport)
    assert owned.inspect()==transport and owned.source().same_owner(source)
    assert owned.reference().same_owner(operator.reference().source())
    positions=owned.variable_positions(0,3)
    assert tuple(p['input'] for p in positions)==(2,0,1)
    assert owned.use_correspondences(0,2)==({'reference':111,'input':311},{'reference':112,'input':312})
    assert asset.recover(311)==(b'oak',) and asset.recover(312)==(b'cedar',)
    assert asset.recover(311)!=operator.reference().source().asset().recover(111)
    info=rows[0]['schema'].inspect()
    assert isinstance(info['accession_archive'],bytes) and isinstance(info['reconstruction_archive'],bytes)
    assert all(isinstance(s['metric_archive'],bytes) for s in info['stalks'])
    assert info['authority']=='raw-none' and rows[0]['schema'].asset().same_owner(asset)
    with pytest.raises((ValueError,RuntimeError)):owned.variable_positions(3,1)
    with pytest.raises((ValueError,RuntimeError)):rows[0]['schema'].inspect(max_records=1)


def test_native_bulk_port_matching_rejects_missing_duplicate_foreign_and_wrong_schema():
    operator,template,asset,transport=fixture();source,d,rows=independent(operator,asset,transport,(2,3,5,7))
    with pytest.raises(ValueError):operator.instantiate(source,d,rows[:-1])
    with pytest.raises(ValueError):operator.instantiate(source,d,(*rows[:-1],rows[0]))
    wrong=tuple(dict(r) for r in rows);wrong[0]['use']=999
    with pytest.raises(ValueError):operator.instantiate(source,d,wrong)
    foreign=NativeInputUseAsset.from_archive(asset.archive());foreign_source,d,other=independent(operator,foreign,transport,(2,3,5,7))
    assert not foreign.same_owner(asset)
    with pytest.raises(ValueError):operator.instantiate(source,d,other)
    wrong=tuple(dict(r) for r in rows);wrong[0]['schema']=rows[1]['schema']
    with pytest.raises(ValueError,match='named port'):operator.instantiate(source,d,wrong)
    wrong=tuple(dict(r) for r in rows);wrong[0]['field']=tensor((1,2,3))
    with pytest.raises(ValueError):operator.instantiate(source,d,wrong)
    with pytest.raises(TypeError):operator.instantiate(source,d,tuple(dict(r,field=(1,2,3,4)) for r in rows))


@pytest.mark.parametrize('graft',('cells','uses','schema'))
def test_complete_declared_bijection_and_symbolic_schema_grafts_reject(graft):
    operator,template,asset,transport=fixture();source,d,rows=independent(operator,asset,transport,(2,3,5,7))
    bad=copy.deepcopy(transport)
    if graft=='cells':
        bad['source_map']['cells']=((0,0),*bad['source_map']['cells'][1:])
        with pytest.raises(ValueError):operator.source_declaration(asset,bad)
    elif graft=='uses':
        bad['uses']=(*bad['uses'][:-1],dict(bad['uses'][0]))
        with pytest.raises(ValueError):operator.source_declaration(asset,bad)
    else:
        schema=operator.source_declaration(asset,transport);schema['symbols']=tuple(dict(s,definition=b'foreign') for s in schema['symbols'])
        foreign=NativeFiniteSource(asset,schema)
        with pytest.raises(ValueError):operator.instantiate(foreign,d,rows)


def test_full_parent_codecs_and_distinct_choices_replay_actual_owners():
    operator,template,asset,transport=fixture();source,d,rows=independent(operator,asset,transport,(2,3,5,7))
    instance=operator.instantiate(source,d,rows);family=instance.execute();choice=family.evaluate(1,(3,5,7))
    for value,cls in ((operator,PreparedFiniteFieldOperator),(instance,NativeFiniteOperatorInstance),
                      (family,NativeFiniteOperatorResult),(choice,NativeFiniteOperatorEvaluation)):
        image=value.archive();restored=cls.from_archive(image,limits=_NATIVE_LIMITS)
        assert restored.archive()==image
        with pytest.raises((ValueError,RuntimeError)):cls.from_archive(image+b'x',limits=_NATIVE_LIMITS)
    cold=NativeFiniteOperatorInstance.from_archive(instance.archive(),limits=_NATIVE_LIMITS);cold_asset=cold.source().asset()
    assert all(r['schema'].asset().same_owner(cold_asset) for r in cold.select_inputs(0,4))
    assert not cold_asset.same_owner(asset) and not cold.same_owner(instance)
    fixed=family.evaluate(0,());equal=family.evaluate(2,())
    assert fixed.raw().result().archive()==equal.raw().result().archive() and fixed.archive()!=equal.archive()
    with pytest.raises(ValueError):family.evaluate(3,())
    with pytest.raises(ValueError):family.evaluate(1,(3,5))
    with pytest.raises(ValueError):family.evaluate(0,(1,))
    with pytest.raises(TypeError):family.evaluate(1,(1.0,2,3))
    with pytest.raises((ValueError,RuntimeError)):NativeFiniteOperatorResult.from_archive(family.archive(),limits=(4096,16000000,1,16384))
    with pytest.raises(ValueError):NativeFiniteOperatorResult.from_archive(family.archive(),max_bytes=32)
    del operator,template,instance,family,source,asset,rows;gc.collect()
    assert choice.family().instance().source().asset().recover(312)==(b'cedar',)
    assert len(values(choice.raw().result().field_archive(2)))==7


def test_port_constructor_and_reentrant_mutation_reject_before_owner_rebinding():
    operator,template,asset,transport=fixture();ports=operator.select_ports(0,4)
    d={'profile':'rexgraph.relational-finite-operator.v1','key':777,'revision':1,'ports':ports[:-1]}
    with pytest.raises(ValueError):PreparedFiniteFieldOperator(operator.reference(),d)
    d['ports']=ports
    owner=PreparedFiniteFieldOperator.__new__(PreparedFiniteFieldOperator)
    class Reentrant:
        def __iter__(self):
            with pytest.raises(TypeError,match='immutable'):owner.__init__(operator.reference(),d)
            return iter(ports)
    owner.__init__(operator.reference(),dict(d,ports=Reentrant()))
    with pytest.raises(TypeError,match='immutable'):owner.__init__(operator.reference(),d)
    assert owner.inspect()['key']==777 and owner.reference().archive()==operator.reference().archive()
    d['key']=999;assert owner.inspect()['key']==777
    assert finite_operator_build_info()['authority']=='raw-none'


def test_concurrent_cold_readers_keep_whole_input_and_response_ancestry():
    operator,template,asset,transport=fixture();source,d,rows=independent(operator,asset,transport,(2,3,5,7))
    image=operator.instantiate(source,d,rows).execute().evaluate(1,(3,5,7)).archive()
    def read(_):
        value=NativeFiniteOperatorEvaluation.from_archive(image,limits=_NATIVE_LIMITS)
        return value.archive(),value.family().instance().source().asset().recover(312)
    with ThreadPoolExecutor(max_workers=6) as executor:
        assert tuple(executor.map(read,range(12)))==((image,(b'cedar',)),)*12


def test_actual_native_packets_interoperate_and_reject_common_available_provider_routes():
    operator,template,asset,transport=fixture();root=Path(os.environ['REXGRAPH_CPP_OPERATOR_FIXTURE_DIR'])
    family=NativeFiniteOperatorResult.from_archive((root/'family.packet').read_bytes(),limits=_NATIVE_LIMITS)
    choice=NativeFiniteOperatorEvaluation.from_archive((root/'choice.packet').read_bytes(),limits=_NATIVE_LIMITS)
    assert family.archive()==(root/'family.packet').read_bytes() and choice.archive()==(root/'choice.packet').read_bytes()
    from rexgraph.cpp_relational_finite_field import cpp_available_finite_field_model
    with pytest.raises(TypeError):cpp_available_finite_field_model(None,available_map=operator,targets=(),execution_context=None)

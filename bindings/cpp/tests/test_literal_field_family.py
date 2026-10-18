"""Whole literal family composition, retained recipes and prospective choices."""
from concurrent.futures import ThreadPoolExecutor
import gc
import pytest
import rexgraph_cpp as cpp
from test_known_tensor_graph import small
from test_relational_field_program import tensor


def result_fixture(raw=b'bank'):
    f=small(raw)
    parameters={'profile':'rexgraph.relational-field-parameters.v1','key':8201,'revision':1,'node':2,
                'variables':tuple((0,r,c) for r in range(2) for c in range(3)),
                'parameters':tuple(range(8301,8307)),
                'basis':{'shape':(6,6),'entries':tuple((i,i,1) for i in range(6))}}
    evidence=cpp.PreparedFieldEvidence(f['target_program'],parameters)
    family=cpp.PreparedFieldFamily({'profile':'rexgraph.relational-field-family.v1','key':8401,'revision':1,'interfaces':()},(evidence,))
    result=cpp.NativeLiteralJointFieldResult(family,0,(f['admitted'],))
    return {**f,'evidence':evidence,'family':family,'result':result}


def test_owning_live_literal_family_conditioned_whole_fields_and_choices():
    f=result_fixture();result=f['result']
    assert result.family().same_owner(f['family'])
    assert result.inputs()[0].graph().literal_parent().same_owner(f['literal'])
    assert result.raw_result().field_archive(1)['particular']==f['graph'].lifted_field()
    assert result.fields()[1]['dimension']==7
    assert result.field_archive(2)['free_parameter_count']==6
    setting=(0,0,2,0,0,3)
    evaluation=result.evaluate(setting)
    assert evaluation.field_archive(2)==tensor((2,3,0,0,0,0,0))
    moved=small(b'bark',parent_program=f['parent_program'],target_program=f['target_program'])
    changed=cpp.NativeLiteralJointFieldResult(f['family'],0,(moved['admitted'],))
    assert changed.field_archive(1)['particular']!=result.field_archive(1)['particular']
    assert changed.evaluate(setting).field_archive(2)==tensor((0,0,0,0,0,0,0))
    assert changed.field_archive(2)['free_parameter_count']==6
    packet=evaluation.archive();del f,result,evaluation,moved,changed;gc.collect()
    cold=cpp.NativeLiteralJointFieldEvaluation.from_archive(packet)
    assert cold.archive()==packet
    assert cold.parent().inputs()[0].graph().literal_parent().asset().recover(111)==(b'bank',b'',b'ban')
    assert cold.field_archive(2)==tensor((2,3,0,0,0,0,0))


def test_cold_owner_adoption_canonical_raw_and_full_stalk_fields():
    f=result_fixture();r=f['result'];packet=r.archive()
    cold=cpp.NativeLiteralJointFieldResult.from_archive(packet)
    assert cold.archive()==packet
    assert not cold.family().same_owner(r.family())
    # Reusing the restored actual owners succeeds; independently decoded equal owners refuse.
    again=cpp.NativeLiteralJointFieldResult(cold.family(),cold.member_index,cold.inputs())
    assert again.archive()==packet
    foreign=cpp.NativeKnownTensorGraphInput.from_archive(cold.inputs()[0].archive())
    with pytest.raises(ValueError):cpp.NativeLiteralJointFieldResult(cold.family(),0,(foreign,))
    assert cold.field_archive(2,occurrences=True)==r.field_archive(2,occurrences=True)
    assert cold.select(2,((0,0),),coordinate_limit=2)==r.select(2,((0,0),),coordinate_limit=2)
    with pytest.raises(TypeError):r.__init__(f['family'],0,(f['admitted'],))
    with pytest.raises((ValueError,RuntimeError)):r.inputs(max_records=0)
    for values in ((),(1,),tuple(range(7))):
        with pytest.raises(ValueError):r.evaluate(values)
    for inputs in ((),(f['admitted'],f['admitted']),('unsupported',)):
        with pytest.raises((ValueError,TypeError)):cpp.NativeLiteralJointFieldResult(f['family'],0,inputs)
    generic=small(generic=True,target_program=f['target_program'])
    with pytest.raises(ValueError):cpp.NativeLiteralJointFieldResult(f['family'],0,(generic['admitted'],))


def test_callback_cancel_does_not_publish_constructor_or_cold_handles():
    f=result_fixture();value=cpp.NativeLiteralJointFieldResult.__new__(cpp.NativeLiteralJointFieldResult)
    class Stop(Exception):pass
    error=Stop('stop')
    def cancel():
        with pytest.raises(RuntimeError):value.admission()
        raise error
    with pytest.raises(Stop):value.__init__(f['family'],0,(f['admitted'],),check=cancel)
    value.__init__(f['family'],0,(f['admitted'],))
    def stop():raise error
    with pytest.raises(Stop):value.evaluate((0,)*6,check=stop)
    with pytest.raises(Stop):value.archive(check=stop)
    with pytest.raises(Stop):cpp.NativeLiteralJointFieldResult.from_archive(value.archive(),check=stop)
    evaluation=value.evaluate((0,)*6)
    with pytest.raises(Stop):evaluation.archive(check=stop)
    with pytest.raises(Stop):cpp.NativeLiteralJointFieldEvaluation.from_archive(evaluation.archive(),check=stop)
    assert value.archive()==f['result'].archive()


@pytest.mark.parametrize('kind',('result','evaluation'))
def test_full_codec_caps_truncation_and_corruption(kind):
    r=result_fixture()['result'];value=r if kind=='result' else r.evaluate((0,)*6)
    cls=type(value);image=value.archive()
    for data in (image[:-1],image[:-1]+bytes((image[-1]^1,)),image+b'x'):
        with pytest.raises((ValueError,RuntimeError)):cls.from_archive(data)
    with pytest.raises(ValueError):cls.from_archive(image,max_bytes=len(image)-1)
    with pytest.raises(ValueError):cls.from_archive(image,limits=(4096,16000001,64000000,16384))


def test_concurrent_prospective_evaluations_and_whole_field_lifetimes():
    r=result_fixture()['result'];expected=r.evaluate((0,0,2,0,0,3)).archive()
    with ThreadPoolExecutor(max_workers=2) as pool:
        packets=list(pool.map(lambda _:r.evaluate((0,0,2,0,0,3)).archive(),range(4)))
    assert packets==[expected]*4
    evaluation=cpp.NativeLiteralJointFieldEvaluation.from_archive(expected);parent=evaluation.parent()
    assert parent.raw_result().archive()==r.raw_result().archive()
    assert parent.input_uses()[0]['asset']==r.input_uses()[0]['asset']


@pytest.mark.parametrize('kind',('result','evaluation'))
def test_cold_subclass_cannot_rebind_an_existing_immutable_owner(kind):
    result=result_fixture()['result'];value=result if kind=='result' else result.evaluate((0,)*6);packet=value.archive()
    class Reused(type(value)):
        def __new__(cls):return value
    with pytest.raises(TypeError):Reused.from_archive(packet)
    assert value.archive()==packet


def test_cold_callbacks_cannot_reinitialize_an_exposed_result():
    f=result_fixture();owners=[]
    class Exposed(cpp.NativeLiteralJointFieldResult):
        def __new__(cls):
            value=super().__new__(cls);owners.append(value);return value
    def inspect():
        with pytest.raises(TypeError):owners[0].__init__(f['family'],0,(f['admitted'],))
        with pytest.raises(RuntimeError):owners[0].admission()
    value=Exposed.from_archive(f['result'].archive(),check=inspect)
    assert value.archive()==f['result'].archive()

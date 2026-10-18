"""Cached seals preserve canonical bytes and detect changes in retained model data."""
from fractions import Fraction as Q
from types import MappingProxyType
import gc

import numpy as np
import pytest

from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_state import ModelBatch, ModelInput, ModelOutput, ModelState, ModelTimeline
from rexgraph.tensor_field import FieldSource
from rexgraph.type_accession import CoordinateSpace


def state(configuration=None, payload=None):
    return ModelState('integrity-fixture','1',FieldSource(None,state_digest='a'*64),
        CoordinateSpace('rows',('a','b')),(),{} if configuration is None else configuration,
        {} if payload is None else payload,'rational')


def value(kind):
    initial=state({'names':('meaning','register'),'signed_zero':-0.0},
        {'exact':Q(1,3),'array':np.asarray([Q(2,5),Q(7,11)],dtype=object)})
    if kind=='model':return initial
    if kind=='timeline':
        from rexgraph.graph import RexGraph, TemporalRex
        from rexgraph.temporal_state import to_temporal_state
        history=TemporalRex([])
        history.append_snapshot(RexGraph.from_hypergraph([0,1],[0]),at=0)
        temporal=to_temporal_state(history)
        return ModelTimeline((Q(1,3),),(initial.coefficient_digest,),(initial.source.state_digest,),
                             temporal.header,temporal.tensors)
    output=ModelOutput([Q(1,3),Q(2,7)],initial.space,(),initial.source,initial.coefficient_digest,
        'rational','integrity-fixture',execution_receipt={'exact':Q(1,3),'array':np.asarray([2,3],dtype=np.int64)})
    if kind=='output':return output
    converted=ModelInput(output.values,np.asarray(output.values,dtype=np.float64),output.space,(),output.source,
        output.coefficient_digest,'rational')
    if kind=='input':return converted
    return ModelBatch(initial.source,initial.space,[Q(1,3),Q(2,7)],[True,False],
        {'field':converted,'exact':Q(1,3)})


@pytest.mark.parametrize('kind',['model','output','input','batch','timeline'])
def test_warm_content_checks_reuse_the_seal_without_encoding_again(kind,monkeypatch):
    import rexgraph.model_state as module
    selected=value(kind);original=module._digest
    def forbidden(_):raise AssertionError('unchanged model was encoded again')
    monkeypatch.setattr(module,'_digest',forbidden)
    for _ in range(5):selected.check_state();assert selected.coefficient_digest==selected._digest
    monkeypatch.setattr(module,'_digest',original)
    assert original(selected)==selected.coefficient_digest


@pytest.mark.parametrize('kind',['model','output','input','batch','timeline'])
@pytest.mark.parametrize('native',[False,True])
def test_warm_and_recomputed_seals_keep_identical_complete_checkpoint_bytes(kind,native):
    selected=value(kind);before=pack_model(selected,native=native)
    object.__setattr__(selected,'_integrity_witness',None)
    after=pack_model(selected,native=native)
    assert set(before)==set(after)
    for key in before:
        assert before[key].dtype==after[key].dtype and before[key].shape==after[key].shape
        if before[key].dtype.hasobject:assert np.array_equal(before[key],after[key])
        else:assert before[key].tobytes()==after[key].tobytes()
    assert unpack_model(after).coefficient_digest==selected.coefficient_digest


@pytest.mark.parametrize('kind',['model','output','input','batch'])
@pytest.mark.parametrize('change',['digest','source','axis','array','fraction'])
def test_mutations_still_fail_after_multiple_warm_checks(kind,change):
    selected=value(kind);selected.check_state();selected.check_state()
    if change=='digest':object.__setattr__(selected,'_digest','0'*64)
    elif change=='source':object.__setattr__(selected.source,'record_id','foreign')
    elif change=='axis':object.__setattr__(selected.space,'keys',('b','a'))
    else:
        arrays={'model':lambda:selected.payload['array'],'output':lambda:selected.values,
                'input':lambda:selected.original,'batch':lambda:selected.targets}
        array=arrays[kind]()
        if change=='array':array.flags.writeable=True;array[0]=Q(99)
        else:array[0]._numerator=99
    with pytest.raises(ValueError):selected.check_state()
    with pytest.raises(ValueError):pack_model(selected)


@pytest.mark.parametrize('change',['bool_integer','negative_zero','key_kind','added','removed','backing_map'])
def test_metadata_type_sign_and_original_keys_remain_checked(change):
    selected=state({1:1,'zero':0.0,'nested':{'coefficient':Q(1,3)}});selected.check_state()
    if change=='backing_map':
        underlying=next(v for v in gc.get_referents(selected.configuration['nested']) if isinstance(v,dict))
        underlying['coefficient']=Q(2,3)
    else:
        data=dict(selected.configuration)
        if change=='bool_integer':data[1]=True
        elif change=='negative_zero':data['zero']=-0.0
        elif change=='key_kind':data['1']=data.pop(1)
        elif change=='added':data['extra']=0
        else:data.pop('zero')
        object.__setattr__(selected,'configuration',MappingProxyType(data))
    with pytest.raises(ValueError):selected.check_state()


@pytest.mark.parametrize('change',['shape','dtype','signed_zero'])
def test_array_shapes_dtypes_and_ieee_zero_bits_are_checked(change):
    selected=state(payload={'array':np.asarray([0.0,1.0],np.float64)})
    selected.check_state();array=selected.payload['array']
    if change=='shape':object.__setattr__(selected,'payload',MappingProxyType({'array':array.reshape(1,2)}))
    elif change=='dtype':object.__setattr__(selected,'payload',MappingProxyType({'array':array.view(np.int64)}))
    else:array.flags.writeable=True;array[0]=-0.0
    with pytest.raises(ValueError):selected.check_state()


def test_nested_declared_input_mutation_is_detected_by_its_batch():
    selected=value('batch');selected.check_state();selected.inputs['field'].original[0]._numerator=99
    with pytest.raises(ValueError):selected.check_state()


@pytest.mark.parametrize('change',['time','model','source','header','tensor'])
def test_warm_timeline_checks_retain_exact_times_references_and_temporal_content(change):
    selected=value('timeline');selected.check_state();selected.check_state()
    if change=='time':selected.times[0]._numerator=2
    elif change=='model':object.__setattr__(selected,'model_digests',('0'*64,))
    elif change=='source':object.__setattr__(selected,'source_digests',('0'*64,))
    elif change=='header':
        data=dict(selected.temporal_header);data['T']=2
        object.__setattr__(selected,'temporal_header',MappingProxyType(data))
    else:
        array=next(v for v in selected.temporal_tensors.values() if v.size)
        array.flags.writeable=True;array.view(np.uint8).flat[0]^=1
    with pytest.raises(ValueError):selected.check_state()
    with pytest.raises(ValueError):pack_model(selected)


def test_equivalent_replacement_rebuilds_the_witness_without_changing_the_seal():
    selected=state({'a':1,'b':2},payload={'array':np.asarray([1,2],np.int64)})
    old=selected.coefficient_digest
    object.__setattr__(selected,'configuration',MappingProxyType({'b':2,'a':1}))
    object.__setattr__(selected,'payload',MappingProxyType({'array':np.asarray([1,2],np.int64)}))
    assert selected.coefficient_digest==old
    selected.payload['array'][0]=9
    with pytest.raises(ValueError):selected.check_state()


@pytest.mark.parametrize('bound',['_MAX_NODES','_MAX_BYTES'])
def test_exhausted_witness_budget_keeps_full_canonical_validation(bound,monkeypatch):
    import rexgraph.model_integrity as module
    monkeypatch.setattr(module,bound,0)
    selected=state(payload={'array':np.asarray([1,2],np.int64)})
    assert selected._integrity_witness is None
    selected.check_state();selected.payload['array'].flags.writeable=True;selected.payload['array'][0]=9
    with pytest.raises(ValueError):selected.check_state()


def test_mutation_during_initial_canonical_sealing_is_rejected(monkeypatch):
    import rexgraph.model_state as module
    original=module._digest
    def changed(selected):
        digest=original(selected);object.__setattr__(selected,'step',selected.step+1);return digest
    monkeypatch.setattr(module,'_digest',changed)
    with pytest.raises(ValueError,match='changed after construction'):state()


def test_shared_mutable_fraction_is_checked_even_when_it_has_the_same_identity():
    q=Q(1,3);selected=state(payload={'values':(q,q,q)});selected.check_state()
    selected.payload['values'][0]._denominator=7
    with pytest.raises(ValueError):selected.check_state()


@pytest.mark.parametrize('part',['numerator','denominator'])
@pytest.mark.parametrize('kind',[np.int64,bool])
def test_fraction_component_types_cannot_be_resealed_from_equal_canonical_numbers(part,kind):
    q=Q(1);selected=state(payload={'exact':q})
    setattr(q,'_'+part,kind(1))
    with pytest.raises(ValueError):selected.check_state()
    with pytest.raises(ValueError):pack_model(selected)


@pytest.mark.parametrize('container',['tuple','array','mixed'])
@pytest.mark.parametrize('change',['numerator','denominator','integer'])
def test_fraction_blocks_keep_exact_content_and_integer_kind(container,change):
    q=Q(1,3)
    values=(q,Q(2,7)) if container!='mixed' else (q,1)
    if container=='array':values=np.asarray(values,dtype=object)
    selected=state(payload={'values':values});selected.check_state()
    retained=selected.payload['values']
    if change=='numerator':retained[0]._numerator=9
    elif change=='denominator':retained[0]._denominator=9
    else:
        if container=='array':
            retained.flags.writeable=True;retained[0]=1
        else:
            object.__setattr__(selected,'payload',MappingProxyType({'values':(1,*retained[1:])}))
    with pytest.raises(ValueError):selected.check_state()
    with pytest.raises(ValueError):pack_model(selected)


@pytest.mark.parametrize('array',[False,True])
def test_fraction_block_admission_precedes_content_capture(array,monkeypatch):
    from rexgraph import model_integrity
    values=(Q(1,3),)*3
    if array:values=np.asarray(values,dtype=object)
    monkeypatch.setattr(model_integrity,'_MAX_NODES',2)
    def forbidden(_):raise AssertionError('rejected block was captured')
    monkeypatch.setattr(model_integrity,'_fraction_parts',forbidden)
    assert model_integrity.capture_content(values) is None

"""Compiled model storage preserves content checks and checkpoint identities."""
from fractions import Fraction as Q
from types import MappingProxyType
import subprocess
import sys

import numpy as np
import pytest

from rexgraph import model_codec, model_integrity
from rexgraph.model_state import ModelState, ModelInput, ModelOutput, ModelBatch, ModelTimeline, freeze_tree
from rexgraph.tensor_field import FieldSource
from rexgraph.type_accession import CoordinateSpace
from rexgraph_cpp._model_storage import capture_content, NativeContentWitness, PayloadEncoder


def model():
    return ModelState('storage-fixture', '1', FieldSource(None, state_digest='a'*64),
        CoordinateSpace('rows', ('a', 'b')), (), {'names': ('syntax', 'meaning'), 'zero': -0.0},
        {'fractions': (Q(1,3), Q(7,11)), 'exact': np.asarray([Q(2,5), 1], dtype=object),
         'data': np.asarray([[1,2], [3,4]], dtype=np.int64)}, 'rational')


def model_value(kind):
    value=model()
    if kind == 'model': return value
    if kind == 'timeline':
        from rexgraph.graph import RexGraph, TemporalRex
        from rexgraph.temporal_state import to_temporal_state
        history=TemporalRex([]);history.append_snapshot(RexGraph.from_hypergraph([0,1], [0]), at=0)
        temporal=to_temporal_state(history)
        return ModelTimeline((Q(1,3),), (value.coefficient_digest,), (value.source.state_digest,),
                             temporal.header, temporal.tensors)
    output=ModelOutput([Q(1,3), Q(2,7)], value.space, (), value.source, value.coefficient_digest,
        'rational', 'storage-fixture', execution_receipt={'values': (Q(1,3),), 'data': np.arange(3)})
    if kind == 'output': return output
    input=ModelInput(output.values, np.asarray(output.values, dtype=np.float64), output.space, (), output.source,
                     output.coefficient_digest, 'rational')
    if kind == 'input': return input
    return ModelBatch(value.source, value.space, [Q(1,3), Q(2,7)], [True, False], {'field': input})


@pytest.mark.parametrize('order',['core_first','native_first'])
def test_native_storage_activation_preserves_both_import_orders(order):
    code = ("import rexgraph\nassert 'rexgraph_cpp' not in sys.modules\nimport rexgraph_cpp\n" if order == 'core_first'
            else "import rexgraph_cpp\nassert 'rexgraph' not in sys.modules\nimport rexgraph\n")
    code = 'import sys\n' + code + "from rexgraph import model_integrity,model_codec,model_state\n" \
        + "assert model_integrity._native_capture is not None and model_state._native_freeze is not None\n" \
        + "assert model_codec._PayloadEncoder is not None and model_codec._PayloadDecoder is not None\n"
    subprocess.run([sys.executable,'-I','-c',code],check=True)


@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed', 'fortran', 'scalar', 'empty'])
@pytest.mark.parametrize('dtype', [np.int64, np.uint8, np.float64, bool])
def test_native_tensor_bytes_match_equal_storage_and_detect_changes(layout, dtype):
    a = np.arange(24).reshape(4,6).astype(dtype)
    if layout == 'strided': a = a[:, ::2]
    elif layout == 'reversed': a = a[::-1, ::-1]
    elif layout == 'fortran': a = np.asfortranarray(a)
    elif layout == 'scalar': a = np.asarray(a[1,1])
    elif layout == 'empty': a = a[:0]
    witness = capture_content(a)
    assert type(witness) is NativeContentWitness
    assert witness.matches(a) and witness.matches(a.copy(order='C')) and witness.matches(a.copy(order='F'))
    assert not witness.matches(a.reshape(-1))
    if a.size:
        a.flat[0] = not a.flat[0] if dtype is bool else a.flat[0]+1
        assert not witness.matches(a)


@pytest.mark.parametrize('container', ['tuple', 'array', 'mixed'])
@pytest.mark.parametrize('change', ['numerator', 'denominator', 'kind', 'integer_kind'])
def test_native_fraction_content_and_types_are_checked(container, change):
    q = Q(1,3); data = (q, Q(2,7)) if container != 'mixed' else (q, 'meaning')
    if container == 'array': data = np.asarray(data, dtype=object)
    witness = capture_content(data)
    assert witness.matches(data)
    if change == 'numerator': q._numerator = 9
    elif change == 'denominator': q._denominator = 9
    elif change == 'integer_kind': q._numerator = np.int64(1)
    elif container == 'array': data[0] = 1
    else: data = (1, data[1])
    assert not witness.matches(data)


def test_native_witness_retains_signed_zero_and_ordered_mapping_key_types():
    v = {1: -0.0, '1': Q(1,3)}; witness = capture_content(v)
    assert witness.matches(v)
    assert not witness.matches({1: 0.0, '1': Q(1,3)})
    assert not witness.matches({'1': Q(1,3), 1: -0.0})
    assert not witness.matches({True: -0.0, '1': Q(1,3)})


@pytest.mark.parametrize('limit', ['nodes', 'bytes', 'depth', 'cycle', 'unsupported'])
def test_native_witness_admission_falls_back_without_partial_witnesses(limit):
    if limit == 'nodes': assert capture_content((Q(1,3),)*3, max_nodes=2) is None
    elif limit == 'bytes': assert capture_content(np.arange(3), max_bytes=0) is None
    elif limit == 'cycle':
        v=[];v.append(v);assert capture_content(v) is None
    elif limit == 'unsupported': assert capture_content(object()) is None
    else:
        v=Q(1,3)
        for _ in range(140): v=(v,)
        assert capture_content(v) is None


def test_native_witness_tree_and_bytes_are_not_mutable_python_attributes():
    witness = capture_content(np.arange(4))
    with pytest.raises(AttributeError): witness.tree = ()
    with pytest.raises(TypeError): NativeContentWitness()


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('kind', ['model', 'output', 'input', 'batch', 'timeline'])
def test_compiled_payload_encoding_is_identical_to_existing_checkpoint_bytes(native, kind, monkeypatch):
    value = model_value(kind); assert type(value._integrity_witness.tree) is NativeContentWitness
    compiled = model_codec.pack_model(value, native=native)
    monkeypatch.setattr(model_codec, '_PayloadEncoder', None)
    monkeypatch.setattr(model_codec, '_PayloadDecoder', None)
    legacy = model_codec.pack_model(value, native=native)
    assert set(compiled) == set(legacy)
    for key in compiled:
        assert compiled[key].dtype == legacy[key].dtype and compiled[key].shape == legacy[key].shape
        assert np.array_equal(compiled[key], legacy[key])
    assert model_codec.unpack_model(compiled).coefficient_digest == value.coefficient_digest


@pytest.mark.parametrize('kind', ['model', 'output', 'input', 'batch', 'timeline'])
def test_compiled_freezing_keeps_full_model_identities(kind, monkeypatch):
    from rexgraph import model_state
    compiled=model_value(kind)
    monkeypatch.setattr(model_state, '_native_freeze', None)
    monkeypatch.setattr(model_integrity, '_native_capture', None)
    monkeypatch.setattr(model_codec, '_PayloadEncoder', None)
    monkeypatch.setattr(model_codec, '_PayloadDecoder', None)
    legacy=model_value(kind)
    assert compiled.coefficient_digest == legacy.coefficient_digest


@pytest.mark.parametrize('value', [np.asarray([1.0],dtype=object), np.asarray([True],dtype=object),
    np.asarray([1+2j]), np.asarray([float('inf')]), float('nan'), {True: 1}, {Q(1,3): 1}, object()])
def test_compiled_freezing_preserves_rejection_of_invalid_metadata(value, monkeypatch):
    from rexgraph import model_state
    with pytest.raises((TypeError,ValueError)):freeze_tree(value)
    monkeypatch.setattr(model_state, '_native_freeze', None)
    with pytest.raises((TypeError,ValueError)):freeze_tree(value)


def test_compiled_freezing_retains_array_ownership_and_scalar_map_sharing():
    array=np.arange(4);q=Q(1,3)
    result=freeze_tree({'data':array,'fraction':q,'maps':({'index':1},{'index':1})})
    assert result['maps'][0] is result['maps'][1]
    assert result['fraction'] is q
    assert not np.shares_memory(result['data'],array) and not result['data'].flags.writeable
    array[0]=99;assert result['data'][0] == 0


def test_compiled_freezing_and_encoding_reject_cycles_with_recursion_checks():
    v=[];v.append(v)
    with pytest.raises(RecursionError):freeze_tree(v)
    with pytest.raises(RecursionError):PayloadEncoder().put(v)


def test_deeper_compatible_metadata_uses_the_existing_complete_codec(monkeypatch):
    v=Q(1,3)
    for _ in range(150):v=(v,)
    selected=ModelState('storage-fixture','1',FieldSource(None,state_digest='a'*64),
                        CoordinateSpace('rows',('a',)),(),{}, {'deep':v},'rational')
    checkpoint=model_codec.pack_model(selected)
    assert model_codec.unpack_model(checkpoint).coefficient_digest == selected.coefficient_digest
    monkeypatch.setattr(model_codec,'_PayloadEncoder',None)
    monkeypatch.setattr(model_codec,'_PayloadDecoder',None)
    before=model_codec.pack_model(selected)
    assert all(np.array_equal(checkpoint[k],before[k]) for k in before)
    assert model_codec.unpack_model(before).coefficient_digest == selected.coefficient_digest


@pytest.mark.parametrize('change', ['tag', 'denominator', 'duplicate', 'tensor_dtype', 'boolean'])
def test_compiled_decoder_preserves_invalid_payload_rejection(change, monkeypatch):
    from rexgraph.value_codec import pack_spec, unpack_spec
    data=model_codec.pack_model(model());spec=unpack_spec(data)
    if change == 'tag': spec['payload']=['unknown']
    elif change == 'denominator': spec['payload']=['rational', '0x1', '0x0']
    elif change == 'duplicate': spec['payload']=['map', [[['str','a'],['none']], [['str','a'],['none']]]]
    elif change == 'tensor_dtype':
        name=next(k for k,v in data.items() if v.dtype == np.int64)
        data[name]=data[name].view(np.uint64)
    else: spec['payload']=['bool',1]
    data['spec']=pack_spec(spec, set(data)-{'spec'}, native=False)
    with pytest.raises(ValueError):model_codec.unpack_model(data)
    monkeypatch.setattr(model_codec, '_PayloadDecoder', None)
    with pytest.raises(ValueError):model_codec.unpack_model(data)


def test_compiled_payload_sharing_keeps_arrays_detached_and_exact_values_typed():
    metadata = MappingProxyType({'index': 1, 'name': 'meaning'})
    array = np.asarray([1,2], np.int64)
    encoder = PayloadEncoder()
    tag = encoder.put((metadata, metadata, array, array, Q(1,3), Q(1,3)))
    assert tag[1][0] is tag[1][1] and tag[1][4] is tag[1][5]
    assert len(encoder.tensors) == 2
    assert not np.shares_memory(encoder.tensors['tensor/0'], encoder.tensors['tensor/1'])
    array[0] = 99
    assert encoder.tensors['tensor/0'][0] == 1


def test_native_witness_checks_equal_replacements_and_rebuilds_after_reordering(monkeypatch):
    value=model();original=value.coefficient_digest
    object.__setattr__(value, 'configuration', MappingProxyType(dict(reversed(tuple(value.configuration.items())))))
    assert value.coefficient_digest == original
    assert type(value._integrity_witness.tree) is NativeContentWitness
    monkeypatch.setattr(model_integrity, '_native_capture', None)
    fallback=model();assert type(fallback._integrity_witness.tree) is tuple
    assert fallback.coefficient_digest == original
    fallback.payload['fractions'][0]._denominator=99
    with pytest.raises(ValueError):fallback.check_state()


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('part', ['numerator','denominator'])
def test_fraction_component_type_changes_cannot_replace_a_known_model_witness(compiled, part, monkeypatch):
    if not compiled:monkeypatch.setattr(model_integrity, '_native_capture', None)
    value=model();q=value.payload['fractions'][0]
    if part == 'numerator':q._numerator=np.int64(q.numerator)
    else:q._denominator=np.int64(q.denominator)
    with pytest.raises(ValueError):value.check_state()
    with pytest.raises(ValueError):model_codec.pack_model(value)

"""Compiled value framing preserves canonical bytes and malformed input rejection."""
from fractions import Fraction as Q
import random
import struct
import subprocess
import sys

import numpy as np
import pytest

from rexgraph import value_codec as codec
from rexgraph._binary import Cursor
from rexgraph.exact_array import ExactArray
from rexgraph.value import Absent, Approx, ExactTime, TimeRange
from rexgraph_cpp._model_storage import pack_native_value, unpack_native_value


def python_pack(value):
    return codec._MAGIC+codec._encode(value,0)


def python_unpack(raw):
    c=Cursor(raw)
    if c.take(5)!=codec._MAGIC:raise ValueError('unsupported native value format')
    value=codec._decode(c,0);c.finish();return value


VALUES=[Absent,None,False,True,-0.0,0.0,0,127,128,-128,-129,-32768,1<<512,-(1<<512),
    Q(0),Q(1),Q(-7,11),Q(-(1<<63),3),Q(1,(1<<63)-1),Q(1,1<<200),Q(1<<512,7),
    '猫 Ελληνικά café',b'\x00\xff',(),[],{},(1,[Q(2,3),Absent]),{1:'a','1':'b',('x',2):[None,Q(3)]},
    Approx(1.25,'declared'),ExactTime(Q(2,7)),TimeRange(ExactTime(Q(1,3)),ExactTime(Q(2,3))),
    ExactArray.from_values([1,Q(2,3),Absent,1<<200]),np.asarray([Q(2,3),1,Absent],dtype=object),
    np.int64(-3),np.uint64(1<<63),np.float32(1.5),np.complex128(1+2j),np.str_('猫'),np.bytes_(b'abc'),
    np.bool_(True)]


@pytest.mark.parametrize('value',VALUES)
def test_closed_values_are_byte_identical_in_both_directions(value):
    expected=python_pack(value)
    assert pack_native_value(value)==expected
    assert python_pack(unpack_native_value(expected))==expected
    assert pack_native_value(python_unpack(expected))==expected
    assert codec.pack_value(value)==expected


@pytest.mark.parametrize('dtype',['i1','i8','>i8','u8','f4','>f8','c16','S4','U4','?','longdouble','clongdouble'])
@pytest.mark.parametrize('layout',['scalar','empty','contiguous','strided','fortran'])
def test_tensor_framing_retains_dtype_shape_layout_and_padding(dtype,layout):
    a=np.arange(12).reshape(3,4).astype(dtype)
    if layout=='scalar':a=a[0,0:1].reshape(())
    elif layout=='empty':a=a[:0]
    elif layout=='strided':a=a[::-1,::2]
    elif layout=='fortran':a=np.asfortranarray(a)
    expected=python_pack(a)
    assert pack_native_value(a)==expected
    value=unpack_native_value(expected)
    assert value.shape==a.shape and value.dtype.str==a.dtype.newbyteorder('<').str
    assert python_pack(value)==expected


@pytest.mark.parametrize('order',['core_first','native_first'])
def test_optional_native_codec_activation_and_cold_core_remain_independent(order):
    code='import sys\n'
    if order=='core_first':code+='from rexgraph import value_codec as c\nassert c._native_pack is None\nimport rexgraph_cpp\n'
    else:code+='import rexgraph_cpp\nfrom rexgraph import value_codec as c\n'
    code+='assert c._native_pack is not None and c._native_unpack is not None\nassert c.unpack_value(c.pack_value(-128))==-128\n'
    subprocess.run([sys.executable,'-I','-c',code],check=True)


def result(decoder,raw):
    try:return ('value',python_pack(decoder(raw)))
    except Exception as exc:return ('error',type(exc))


def test_truncated_mutated_and_random_values_match_the_reference_admission():
    rng=random.Random(93519)
    originals=[python_pack(v) for v in VALUES]
    for raw in originals:
        for end in range(len(raw)):
            fragment=raw[:end]
            assert result(unpack_native_value,fragment)==result(python_unpack,fragment)
        for _ in range(12):
            value=bytearray(raw);index=rng.randrange(len(value));value[index]=rng.randrange(256);value=bytes(value)
            assert result(unpack_native_value,value)==result(python_unpack,value)
    for _ in range(1000):
        raw=codec._MAGIC+rng.randbytes(rng.randrange(0,32))
        assert result(unpack_native_value,raw)==result(python_unpack,raw)


@pytest.mark.parametrize('suffix',[b'I\x00',b'I\x02\x80\xff',b'L\x80\x00',b'U\xff'*10,
    b'Y\x03|b1\x01\x01\x01\x02',b'M\x02I\x01\x01NI\x01\x01N',
    b'M\x02I\x01\x01NTN',b'M\x01L\x00N',b'GNT',b'ZN',b'D'+struct.pack('<d',float('inf'))])
def test_noncanonical_closed_records_are_refused(suffix):
    raw=codec._MAGIC+suffix
    assert result(unpack_native_value,raw)==result(python_unpack,raw)
    assert result(unpack_native_value,raw)[0]=='error'


@pytest.mark.parametrize('depth',[64,65,66])
def test_nesting_limit_is_identical(depth):
    value=None
    for _ in range(depth):value=(value,)
    if depth<=64:assert pack_native_value(value)==python_pack(value)
    else:
        with pytest.raises(ValueError,match='nesting'):pack_native_value(value)
        with pytest.raises(ValueError,match='nesting'):python_pack(value)
    raw=codec._MAGIC+b'U\x01'*depth+b'N'
    assert result(unpack_native_value,raw)==result(python_unpack,raw)


def test_cycles_unsupported_values_and_mutated_fraction_slots_match_reference_errors():
    cycle=[];cycle.append(cycle)
    bad_fraction=Q(1,3);bad_fraction._denominator=-3
    unreduced=Q(1,3);unreduced._numerator=2;unreduced._denominator=6
    for value in (cycle,object(),bad_fraction,unreduced,np.asarray([float('nan')]),np.zeros(2,dtype='datetime64[D]')):
        errors=[]
        for encoder in (pack_native_value,python_pack):
            try:encoder(value)
            except Exception as exc:errors.append(type(exc))
        assert len(errors)==2 and errors[0] is errors[1]


def test_bytes_subclasses_do_not_replace_the_retained_bytes():
    class OriginalBytes(bytes):
        def __bytes__(self):return b'changed'
    value=OriginalBytes(b'original')
    assert pack_native_value(value)==python_pack(value)
    for raw in (OriginalBytes(python_pack(value)),np.bytes_(python_pack(value))):
        assert unpack_native_value(raw)==python_unpack(raw)

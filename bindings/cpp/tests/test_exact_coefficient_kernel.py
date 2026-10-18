"""Compiled coefficient bytes preserve the existing exact tensor format."""
from fractions import Fraction as Q
from random import Random

import numpy as np
import pytest

from rexgraph import state
from rexgraph.core._state import encode_exact_coefficients


def original(array,monkeypatch):
    with monkeypatch.context() as changed:
        changed.setattr(state,'_encode_exact_kernel',None)
        return state._encode_exact(array)


@pytest.mark.parametrize('layout',['contiguous','strided','reversed','fortran','scalar','empty','cube'])
def test_original_exact_bytes_and_shape_roundtrip_for_every_layout(layout,monkeypatch):
    values=[Q(-17,20),1<<16000,Q(3),-(1<<16001),np.int64(-4),np.uint64(7)]*4
    array=np.asarray(values,dtype=object).reshape(4,6)
    if layout=='strided':array=array[:,::2]
    elif layout=='reversed':array=array[::-1,::-1]
    elif layout=='fortran':array=np.asfortranarray(array)
    elif layout=='scalar':array=np.asarray(array[0,0],dtype=object)
    elif layout=='empty':array=array[:0]
    elif layout=='cube':array=array.reshape(2,3,4)
    expected=original(array,monkeypatch);actual=encode_exact_coefficients(array)
    assert actual.dtype==np.uint8 and actual.ndim==1 and actual.flags.owndata
    assert actual.tobytes()==expected.tobytes()
    assert state._encode_exact_kernel is encode_exact_coefficients
    tensors={'field':array};spec=state.encode_tensors(tensors)
    state.decode_tensors(tensors,spec)
    assert tensors['field'].shape==array.shape
    assert all(type(a) is (Q if isinstance(b,Q) else int) and a==b for a,b in zip(tensors['field'].flat,array.flat,strict=True))


def test_bounded_scalar_cache_keeps_distinct_tags_and_all_large_values(monkeypatch):
    rng=Random(918);values=[]
    for i in range(4096):
        n=rng.randrange(-(1<<180),(1<<180));d=rng.randrange(1,1<<90)
        values.extend((n,Q(n,d),Q(1),1,Q(0),0))
    array=np.asarray(values,dtype=object)
    assert encode_exact_coefficients(array).tobytes()==original(array,monkeypatch).tobytes()


def test_cache_does_not_reuse_previously_mutated_fraction_components(monkeypatch):
    q=Q(2,3);array=np.asarray([q,Q(2,3),3],dtype=object)
    before=encode_exact_coefficients(array).tobytes();q._numerator=7
    after=encode_exact_coefficients(array).tobytes()
    assert before!=after and after==original(array,monkeypatch).tobytes()


@pytest.mark.parametrize('value',[True,np.bool_(False),1.0,np.float64(2),'3',None,object(),complex(1,2)])
def test_nonexact_carriers_reject_without_changing_the_input(value,monkeypatch):
    array=np.asarray([Q(1,3),value],dtype=object)
    for encode in (encode_exact_coefficients,lambda a:original(a,monkeypatch)):
        with pytest.raises(TypeError):encode(array)
    assert array[1] is value


def test_fraction_subclass_uses_its_actual_original_format(monkeypatch):
    class Alternate(Q):
        @property
        def numerator(self):return 7
    array=np.asarray([Q(1,3),Alternate(1,3),Q(1,3)],dtype=object)
    assert encode_exact_coefficients(array).tobytes()==original(array,monkeypatch).tobytes()

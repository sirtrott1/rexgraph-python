"""Installed owning context product binding, current origins and cold replay."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q

import pytest
from rexgraph_cpp import (
    NativeSource, NativeInputUseAsset, NativeValueOrigin, NativeAvailabilityContext,
    PreparedFieldProgram, PreparedAccessionFieldProgram, PreparedAvailableFieldProgram,
    PreparedAvailableKnownTensorSquare, NativeAvailableKnownTensorSquareInput,
    NativeAvailableKnownTensorSquareResult, field_input_origin_payloads,
    known_tensor_square_origin_payload, encode_stalk_column_tensor,
)


def matrix(n, entries=()):
    return {"shape": (n,n), "entries": tuple(entries)}


def identity(n):
    return matrix(n, ((i,i,Q(1)) for i in range(n)))


def tensor(values):
    return encode_stalk_column_tensor(len(values), tuple((i,Q(v)) for i,v in enumerate(values) if v))


def source():
    flags = ((10,True,False,1),(20,True,True,1),(10,True,True,1),
             (20,False,False,1),(30,True,False,0))
    return NativeSource.from_declaration({"profile": "declared-role-source-v1", "namespace": 998201,
        "semantic_scale": 2, "max_grade": 1, "role_aspects": ("observed-use",),
        "types": ({"name": "context", "min_arity": 1, "max_arity": 8},),
        "witnesses": (10,20,30), "relations": ({"grade": 1, "key": 100, "type": 0,
            "occurrences": tuple({"key": i+1,"lower": lower,"exists": active,"negative": negative,
                                  "magnitude": magnitude,"aspects": (0,)}
                                 for i,(lower,active,negative,magnitude) in enumerate(flags))},),
        "carriers": (), "fields": (), "metrics": ()})


def program(owner, lifted=False):
    n = 6 if lifted else 2
    spec = {"profile": "rexgraph.relational-field-program.v1", "key": 42 if lifted else 41,
        "revision": 1, "differential": True,
        "stalks": tuple({"cell": (0,i), "basis": 101, "directions": (102,), "metric": identity(1)}
                         for i in range(3)) + ({"cell": (1,0), "basis": 400 if lifted else 303,
                            "directions": (31,32,501,502,503,504) if lifted else (31,32), "metric": identity(n)},),
        "occurrences": tuple({"grade": 1,"slot": i,"role": "coefficient_independent",
            "map": {"shape": (1,n),"entries": ((0,0,Q(1)),(0,1,Q(2))) + (((0,3,Q(1)),) if lifted else ())}}
                             for i in range(5)),
        "nodes": ({"key": 1,"kind": "input","grade": 1},
                  {"key": 2,"kind": "boundary","grade": 0,"input": 1})}
    fixed = PreparedFieldProgram(owner.archive,spec)
    return PreparedAccessionFieldProgram(fixed.archive())


def accession(lifted=False):
    n = 6 if lifted else 2
    return {"profile": "rexgraph.relational-input-accession.v1", "key": 902 if lifted else 901,
        "node": 1,"grade": 1,"isometric": True,
        "stalks": ({"cell": (1,0),"basis": 400 if lifted else 303,
                    "directions": (31,32,501,502,503,504) if lifted else (31,32),"metric": identity(n)},),
        "instances": {"cells": ((0,1,2),(0,)), "occurrences": ((),(0,1,2,3,4))},
        "accession": identity(n), "reconstruction": identity(n)}


DECLARATION = {"profile": "rexgraph.known-context-tensor-square.v1", "key": 701,
               "revision": 1,"lifted_basis": 400,"context": (1,0),"pair_directions": (501,502,503,504)}
KEY_NAMES = ('asset','value','accession','reconstruction')


def context(origins, remove=()):
    return NativeAvailabilityContext(tuple({"origin": origin,"namespace": origin.inspect()["namespace"],
        "sequence": origin.inspect()["sequence"]} for key,origin in origins.items() if key not in remove))


def fixture(values=(2,3)):
    history = source(); parent = program(history); target = program(history,True)
    asset = NativeInputUseAsset(history.archive,{"profile": "rexgraph.relational-input-use-asset.v1",
        "key": 600,"channel": 700,"encoding": "UTF-8","bytes": b"caf\xc3\xa9 / caf\xc3\xa9",
        "uses": ({"key": 111,"cells": ((1,100),),"occurrences": tuple((1,i+1) for i in range(5)),
                  "spans": ((0,5),(8,13),(0,0),(0,5))},)})
    row = {"asset": asset.archive(),"accession": accession(),"use": 111,"field": tensor(values)}
    origins = {}
    def publish(key,role,payload):
        origins[key] = NativeValueOrigin(key,history.archive,history.identity['sequence'],role,payload)
    for key,role in enumerate(('program','maps','metrics'),1): publish(key,role,parent.archive())
    for key,(name,role) in enumerate(zip(KEY_NAMES,('input-asset','input-value','input-accession','reconstruction'),strict=True),4):
        publish(key,role,field_input_origin_payloads(parent,row)[name])
    row['origins'] = dict(zip(KEY_NAMES,range(4,8),strict=True))
    publish(8,'interface',known_tensor_square_origin_payload(parent,DECLARATION))
    authority = context(origins)
    available = PreparedAvailableFieldProgram(parent,{'program':1,'maps':2,'metrics':3},authority)
    producer = PreparedAvailableKnownTensorSquare(available,DECLARATION,8,authority)
    square = producer.lift(row,authority)
    for key,role in enumerate(('program','maps','metrics'),9): publish(key,role,target.archive())
    payloads = square.input_origin_payloads(target,accession(True))
    for key,(name,role) in enumerate(zip(KEY_NAMES,('input-asset','input-value','input-accession','reconstruction'),strict=True),12):
        publish(key,role,payloads[name])
    authority = context(origins)
    downstream = PreparedAvailableFieldProgram(target,{'program':9,'maps':10,'metrics':11},authority)
    value = square.admit_input(downstream,accession(True),dict(zip(KEY_NAMES,range(12,16),strict=True)),authority)
    return producer,row,square,value,origins,authority


def test_whole_fields_native_pair_provenance_and_owned_cold_execution():
    _,_,square,value,_,authority = fixture()
    assert square.field_archive('context') == tensor((2,3))
    assert square.field_archive() == tensor((2,3,4,6,6,9))
    assert square.pair_coordinate(1) == {'context': (1,0),'original_basis': 303,'left': 31,'right': 32,'lifted_direction': 502}
    square.require_image(square.field_archive())
    assert value.input()['field'] == square.field_archive()
    assert value.parent() == square.parent()
    assert len(value.dependency_receipts()) == 15
    with pytest.raises(RuntimeError,match='complete product dependency bound'):
        value.dependency_receipts(max_records=14)
    result = value.execute(authority)
    assert result.input().archive() == value.archive()
    assert NativeAvailableKnownTensorSquareInput.from_archive(value.archive(),authority).archive() == value.archive()
    restored = NativeAvailableKnownTensorSquareResult.from_archive(result.archive(),authority)
    assert restored.archive() == result.archive()
    assert restored.result().archive() == result.result().archive()
    assert restored.result().field_archive(1) == square.field_archive()
    assert restored.result().field_archive(2,occurrences=True) == result.result().field_archive(2,occurrences=True)
    assert restored.input().parent()['asset'] == value.parent()['asset']
    interface = restored.result().interface({'profile': 'rexgraph.relational-field-interface.v1',
        'key': 811,'branch': 812,'cells': ((2,0,1),)})
    reading = interface.observe({'profile': 'rexgraph.relational-field-view.v1','key': 813,'basis': 814,
        'directions': (1,),'accession': identity(1),'reconstruction': identity(1),'physical': None})
    assert reading.inspect()['visible'] == tensor((-14,))


@pytest.mark.parametrize('key',range(1,16))
def test_each_original_declaration_and_derived_origin_remains_required(key):
    producer,row,_,value,origins,authority = fixture(); result = value.execute(authority)
    missing = context(origins,(key,))
    if key <= 8:
        with pytest.raises(ValueError): producer.lift(row,missing)
    for operation in (lambda:value.execute(missing),lambda:result.admit(missing),
                      lambda:NativeAvailableKnownTensorSquareInput.from_archive(value.archive(),missing,limits=(1,16000000,64000000,16384)),
                      lambda:NativeAvailableKnownTensorSquareResult.from_archive(result.archive(),missing,limits=(1,16000000,64000000,16384))):
        with pytest.raises(ValueError,match='origin|availability'): operation()


def test_off_image_supported_zeros_and_false_independent_publications_reject():
    _,_,square,value,origins,authority = fixture((0,Q(3,2)))
    assert square.field_archive() == tensor((0,Q(3,2),0,0,0,Q(9,4)))
    square.require_image(square.field_archive())
    with pytest.raises(ValueError): square.require_image(tensor((0,Q(3,2),0,1,0,Q(9,4))))
    copied = {key:NativeValueOrigin(key,origin.source_archive(),origin.inspect()['sequence'],origin.inspect()['role'],origin.payload())
              for key,origin in origins.items()}
    with pytest.raises(ValueError,match='live value origin owner'): value.admit(context(copied))
    assert NativeAvailableKnownTensorSquareInput.from_archive(value.archive(),context(copied)).archive() == value.archive()
    with pytest.raises((ValueError,IndexError)): NativeAvailableKnownTensorSquareInput.from_archive(value.archive(),authority,max_bytes=16)


def test_immutable_owners_concurrent_replay_and_closed_declarations():
    producer,_,square,value,_,authority = fixture()
    with pytest.raises(TypeError): producer.__init__(None,DECLARATION,8,authority)
    with pytest.raises((AttributeError,TypeError)): value.saved_authority = True
    with pytest.raises(RuntimeError): NativeAvailableKnownTensorSquareInput.__new__(NativeAvailableKnownTensorSquareInput).archive()
    with pytest.raises((ValueError,IndexError)): square.pair_coordinate(4)
    expected = value.execute(authority).archive()
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert tuple(pool.map(lambda _:NativeAvailableKnownTensorSquareInput.from_archive(value.archive(),authority).execute(authority).archive(),range(6))) == (expected,)*6

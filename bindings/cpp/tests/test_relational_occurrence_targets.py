"""Owning typed targets, canonical native histories and actual external grants."""
from pathlib import Path
from copy import deepcopy
import hashlib
import os

import pytest

POLICY = {'limits':(4096,16000000,64000000,1 << 24),'max_bytes':64 << 20}

cpp = pytest.importorskip('rexgraph_cpp')
if not hasattr(cpp, 'NativeOccurrenceTargetPort'):
    pytest.skip('typed native occurrence SDK required', allow_module_level=True)


@pytest.fixture(scope='module')
def occurrence_root():
    value = os.environ.get('REXGRAPH_CPP_OCCURRENCE_QUALIFIED')
    if not value: pytest.skip('qualified native occurrence packet fixtures not supplied')
    return Path(value)


def external_context(root, *, remove=(), cutoff=None):
    import json
    directory = root / 'external-origin-context'
    records = json.loads((directory / 'context.json').read_bytes())
    origins = {}; grants = []
    for row in records['origins']:
        contents = {}
        for key in ('payload', 'publication-history', 'receipt'):
            data = (directory / row[key+'_file']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row[key+'_sha256']
            contents[key] = data
        origin = cpp.NativeValueOrigin(row['key'], contents['publication-history'], row['publication'], row['role'], contents['payload'])
        assert origin.receipt() == contents['receipt']
        origins[row['key']] = origin
        if row['key'] not in remove:
            grants.append({'origin': origin, 'namespace': row['availability_namespace'],
                           'sequence': row['availability_sequence'] if cutoff is None else min(cutoff, row['availability_sequence'])})
    return cpp.NativeAvailabilityContext(tuple(grants)), origins


@pytest.mark.parametrize('directory,name,kind', [
    ('unit','port','NativeOccurrenceTargetPort'), ('unit','mixed-evidence','PreparedFieldEvidence'),
    ('unit','result','NativeAffineFieldResult'), ('unit','evaluation','NativeAffineEvaluation'),
    ('actual','port','NativeOccurrenceTargetPort'), ('actual','evidence','PreparedFieldEvidence'),
    ('actual','result','NativeAffineFieldResult'), ('actual','evaluation','NativeAffineEvaluation'),
    ('actual','family','PreparedFieldFamily'), ('actual','joint','NativeJointFieldResult'),
    ('actual','higher-evidence','PreparedFieldEvidence'), ('actual','higher-result','NativeAffineFieldResult')])
def test_native_canonical_packets_preserve_whole_typed_owners(occurrence_root, directory, name, kind):
    packet = (occurrence_root / (directory+'-canonical-packets') / (name+'.packet')).read_bytes()
    value = getattr(cpp, kind).from_archive(packet,**POLICY)
    assert value.archive() == packet
    if kind == 'PreparedFieldEvidence':
        with pytest.raises(ValueError, match='typed'): value.inspect()
        details = value.inspect(typed=True, max_records=4096)
        assert all(len(row) == 4 for row in details['observation_coordinates'])
        assert any(row[1] == 1 for row in details['observation_coordinates'])
        assert value.parent() is not None
        index = value.identity()['observation_count']-1
        observation = value.observation(index, max_records=4096)
        assert observation['domain'] == 'original-occurrence' and observation['observed'] == ()
        residual = value.residual(index)
        assert residual.target_port().archive() == observation['port'].archive()
        assert residual.response().evidence().archive() == value.archive()


def test_actual_port24_declaration_owner_rebind_caps_and_immutability(occurrence_root):
    packet = (occurrence_root / 'actual-canonical-packets/port.packet').read_bytes()
    port = cpp.NativeOccurrenceTargetPort.from_archive(packet,**POLICY)
    detail = port.inspect(max_records=24)
    assert len(detail['coordinates']) == 24
    assert tuple(c['occurrence'] for c in detail['coordinates']) == tuple((1,6+2*j) for j in range(12) for _ in range(2))
    with pytest.raises((ValueError,RuntimeError)): port.inspect(max_records=23)
    cloned = cpp.NativeOccurrenceTargetPort.from_archive(packet,**POLICY)
    assert not cloned.same_owner(port)
    rebound = cloned.rebind(port.program())
    assert rebound.archive() == packet and not rebound.same_owner(port)
    with pytest.raises(TypeError): port.__init__(port.program(), {})
    from test_relational_field_evidence import read_matrix
    declaration = {'profile':'rexgraph.relational-occurrence-target-port.v1', **{k:detail[k] for k in ('key','revision','node','frame','coordinates')}}
    for key in ('accession','reconstruction'):
        matrix = read_matrix(detail[key])
        declaration[key] = {'shape':(len(matrix),len(matrix[0])), 'entries':tuple((i,j,q) for i,row in enumerate(matrix) for j,q in enumerate(row) if q)}
    assert cpp.NativeOccurrenceTargetPort(port.program(), declaration).archive() == packet
    for key in ('basis','direction','occurrence'):
        changed = deepcopy(declaration); rows = list(changed['coordinates']); row = dict(rows[0])
        row[key] = (2,row[key][1]) if key == 'occurrence' else row[key]+999
        rows[0] = row;changed['coordinates'] = rows
        with pytest.raises((ValueError,RuntimeError)): cpp.NativeOccurrenceTargetPort(port.program(),changed)


@pytest.mark.parametrize('name,kind', [
    ('available-root','PreparedAvailableFieldEvidence'),('available-lower-v1','PreparedAvailableFieldEvidence'),
    ('available-mixed-v2','PreparedAvailableFieldEvidence'),('available-result','NativeAvailableAffineResult'),
    ('available-evaluation','NativeAvailableAffineEvaluation')])
def test_external_actual_authorities_restore_canonical_available_packets(occurrence_root,name,kind):
    context, _ = external_context(occurrence_root)
    packet = (occurrence_root / 'available-canonical-packets' / (name+'.packet')).read_bytes()
    value = getattr(cpp,kind).from_archive(packet,context,**POLICY)
    assert value.archive() == packet
    value.admit(context)


def test_kind_aware_native_origin_enumeration_and_caps(occurrence_root):
    context, origins = external_context(occurrence_root)
    packet = (occurrence_root / 'available-canonical-packets/available-mixed-v2.packet').read_bytes()
    value = cpp.PreparedAvailableFieldEvidence.from_archive(packet,context,**POLICY)
    for index, count in ((0,9),(1,10)):
        actual = value.observation_origins(index,context,max_records=count)
        assert len(actual) == count
        assert all(o.receipt() == origins[o.inspect()['key']].receipt() for o in actual)
        with pytest.raises((ValueError,RuntimeError)): value.observation_origins(index,context,max_records=count-1)
    assert [o.inspect()['role'] for o in value.observation_origins(1,context)][-1] == 'interface'


@pytest.mark.parametrize('key',(1,2,3,4,10,11,12,13,20,21,22,31))
def test_every_input_target_port_and_historical_grant_rechecked(occurrence_root,key):
    good, _ = external_context(occurrence_root)
    bad, _ = external_context(occurrence_root,remove=(key,))
    packet = (occurrence_root / 'available-canonical-packets/available-mixed-v2.packet').read_bytes()
    value = cpp.PreparedAvailableFieldEvidence.from_archive(packet,good,**POLICY)
    with pytest.raises(ValueError): value.admit(bad)
    with pytest.raises(ValueError): value.observation_origins(1,bad)
    with pytest.raises(ValueError): cpp.PreparedAvailableFieldEvidence.from_archive(packet,bad,**POLICY)


def test_actual_later_publication8_rejected_at_cutoff7(occurrence_root):
    good, _ = external_context(occurrence_root)
    past, _ = external_context(occurrence_root,cutoff=7)
    packets = occurrence_root / 'available-canonical-packets'
    for name,kind in (('available-mixed-v2','PreparedAvailableFieldEvidence'),('available-result','NativeAvailableAffineResult'),('available-evaluation','NativeAvailableAffineEvaluation')):
        data = (packets/(name+'.packet')).read_bytes()
        value = getattr(cpp,kind).from_archive(data,good,**POLICY)
        with pytest.raises(ValueError): value.admit(past)
        with pytest.raises(ValueError): getattr(cpp,kind).from_archive(data,past,**POLICY)

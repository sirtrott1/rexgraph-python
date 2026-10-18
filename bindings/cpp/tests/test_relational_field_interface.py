"""Actual source uses, faithful images, shared interfaces and native cold replay."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import pytest
from rexgraph_cpp import (
    PreparedFieldProgram, PreparedAccessionFieldProgram, NativeInputUseAsset,
    NativeInputAccession, NativeAccessionInput, NativeAccessionFieldResult,
    NativeFieldInterface, NativeFieldReading, NativeFieldSection,
)
from test_relational_field_program import source, declaration, matrix, tensor, boundary, action, LATER, MAPS
from rexgraph.cpp_relational_field_program import (
    cpp_accession_field_program_model, accession_field_program_query,
    read_field_program_result, clear_field_program_cache,
)
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import model_record, read_model_record, save_model, load_model
from rexgraph.model_codec import pack_model
from rexgraph.value_codec import pack_value


def asset_declaration():
    return {'profile': 'rexgraph.relational-input-use-asset.v1', 'key': 600, 'channel': 700,
            'encoding': 'UTF-8', 'bytes': 'café / café'.encode(), 'uses': (
                {'key': 111, 'cells': ((1, 100),), 'occurrences': tuple((1, k) for k in range(1, 9)), 'spans': ((0, 5),)},
                {'key': 112, 'cells': ((1, 101),), 'occurrences': (), 'spans': ((8, 13),)},
                {'key': 113, 'cells': ((1, 100), (1, 100)), 'occurrences': (), 'spans': ((8, 13), (1, 4), (0, 0), (0, 5))})}


def accession_declaration():
    return {'profile': 'rexgraph.relational-input-accession.v1', 'key': 901, 'node': 1, 'grade': 1,
            'stalks': ({'cell': (1, 0), 'basis': 910, 'directions': (911,), 'metric': matrix(((3,),))},
                       {'cell': (1, 1), 'basis': 920, 'directions': (921,), 'metric': matrix(((1,),))}),
            'instances': {'cells': ((0, 1, 2), (0, 1)), 'occurrences': ((), tuple(range(8)))},
            'accession': matrix(((1, 1), (0, 1), (1, 2), (0, 0))),
            'reconstruction': matrix(((1, -1, 0, 0), (0, 1, 0, 0))), 'isometric': False}


def patch_declaration():
    return {'profile': 'rexgraph.relational-field-interface.v1', 'key': 2001, 'branch': 2002, 'cells': ((2, 0, 0),)}


def view_declaration():
    return {'profile': 'rexgraph.relational-field-view.v1', 'key': 3001, 'basis': 3101, 'directions': (3111, 3112, 3113),
            'accession': matrix(((1, 1), (0, 1), (1, 2))),
            'reconstruction': matrix(((1, -1, 0), (0, 1, 0))), 'physical': None}


@pytest.fixture
def prepared():
    owner = source(); original = source(993003)
    program = PreparedAccessionFieldProgram(PreparedFieldProgram(owner.archive, declaration()).archive())
    asset = NativeInputUseAsset(original.archive, asset_declaration())
    accession = NativeInputAccession(asset, program, accession_declaration())
    return owner, original, program, asset, accession


@pytest.mark.parametrize('trial', range(16))
def test_complete_original_inputs_and_all_native_intermediates_replay(prepared, trial):
    _, original, program, asset, accession = prepared
    a, b = Q(trial - 5, 3), Q(trial % 9 - 3, 5)
    result = program.execute((accession.admit(111, tensor((a, b))),))
    represented = (a+b, b, a+2*b, Q(0))
    lower, occurrences = boundary(represented); following = action(LATER, lower); last, last_occurrences = boundary(following)
    for node, values in ((1, represented), (2, lower), (3, following), (4, last)):
        assert result.field_archive(node) == tensor(values)
    assert result.field_archive(2, occurrences=True) == tensor(occurrences)
    assert result.field_archive(4, occurrences=True) == tensor(last_occurrences)
    captured = result.input_uses()[0]
    assert captured['field'] == tensor((a, b)) and captured['represented'] == tensor(represented)
    assert captured['instances']['occurrences'][1] == tuple(range(8))
    retained = NativeInputUseAsset.from_archive(captured['asset'])
    assert retained.source_archive() == original.archive and retained.inspect()['source']['namespace'] == 993003
    assert retained.archive() == asset.archive() and retained.recover(111) == ('café'.encode(),)
    cold = NativeAccessionFieldResult.from_archive(result.archive())
    assert cold.archive() == result.archive() and cold.input_uses() == result.input_uses()
    assert cold.field_archive(4) == tensor(last)


def test_original_spans_repetitions_empty_cuts_and_equal_encodings_stay_distinct(prepared):
    _, original, program, asset, accession = prepared
    assert asset.recover(111) == asset.recover(112)
    raw = asset_declaration()['bytes']
    assert asset.recover(113) == (raw[8:13], raw[1:4], b'', raw[:5])
    assert asset.uses()[2]['cells'] == ((1, 100), (1, 100))
    assert not asset.same_owner(NativeInputUseAsset.from_archive(asset.archive()))
    first, second = (program.execute((accession.admit(use, tensor((2, 3))),)) for use in (111, 112))
    assert first.field_archive(4) == second.field_archive(4) and first.archive() != second.archive()
    assert first.input_uses()[0]['use'] == 111 and second.input_uses()[0]['use'] == 112
    assert NativeInputUseAsset.from_archive(asset.archive()).source_archive() == original.archive
    with pytest.raises(RuntimeError): asset.recover(113, max_bytes=2)
    with pytest.raises(RuntimeError): asset.uses(max_records=1)


def test_faithful_views_whole_original_stalks_compatible_glue_and_distinct_contributions(prepared):
    _, _, program, _, accession = prepared
    result = program.execute((accession.admit(111, tensor((2, 3))),))
    interface = result.interface(patch_declaration()); view = view_declaration()
    derived = interface.observe(view)
    assert derived.inspect()['status'] == 'derived' and derived.inspect()['physical'] is None
    original = result.select(2, ((0, 0),), coordinate_limit=2)[0]['values']; u, v = original
    visible = (u+v, v, u+2*v)
    assert derived.inspect()['visible'] == tensor(visible) and derived.inspect()['reconstructed'] == tensor(original)
    assert derived.interface().same_owner(interface)
    observed = interface.admit(view, tensor(visible), status='observed')
    compatible = NativeFieldSection((derived, observed))
    assert compatible.kind == 'compatible' and compatible.select(((0, 0),), coordinate_limit=2)[0]['values'] == original
    prospective = interface.admit(view, tensor((-u+v, v, -u+2*v)), status='prospective')
    with pytest.raises(ValueError): NativeFieldSection((derived, prospective))
    summed = NativeFieldSection((derived, prospective), kind='contribution_sum')
    assert summed.kind == 'contribution_sum' and summed.select(((0, 0),), coordinate_limit=2)[0]['values'] == (0, 2*v)
    for section in (compatible, summed):
        restored = NativeFieldSection.from_archive(section.archive())
        assert restored.archive() == section.archive() and restored.kind == section.kind
        readings = restored.readings()
        assert readings[0].interface().same_owner(readings[1].interface())
        assert not readings[0].interface().same_owner(interface)
        assert restored.parent().archive() == result.archive()
        assert restored.select(((0, 0),), coordinate_limit=2) == section.select(((0, 0),), coordinate_limit=2)


@pytest.mark.parametrize('kind', ['image', 'reconstruction', 'directions', 'physical', 'derived', 'owner', 'closure', 'instance', 'isometry'])
def test_unsupported_images_maps_owners_metrics_and_patch_bindings_reject(prepared, kind):
    _, _, program, asset, accession = prepared
    result = program.execute((accession.admit(111, tensor((2, 3))),))
    interface = result.interface(patch_declaration()); view = view_declaration()
    if kind in {'instance', 'isometry'}:
        spec = accession_declaration()
        if kind == 'instance': spec['instances']['occurrences'] = ((), tuple(range(7)))
        else: spec['isometric'] = True
        with pytest.raises(ValueError): NativeInputAccession(asset, program, spec)
    elif kind == 'closure':
        patch = patch_declaration(); patch['cells'] = ((1, 1, 0),)
        with pytest.raises(ValueError): result.interface(patch)
    elif kind == 'owner':
        other = result.interface(patch_declaration())
        assert not other.same_owner(interface)
        with pytest.raises(ValueError): NativeFieldSection((interface.observe(view), other.observe(view)))
    else:
        data = interface.observe(view).inspect()['visible']
        if kind == 'image': data = tensor((1, 2, 4))
        elif kind == 'reconstruction': view['reconstruction'] = matrix(((1, 0, 0), (0, 1, 0)))
        elif kind == 'directions': view['directions'] = (3111, 3111, 3113)
        elif kind == 'physical': view['physical'] = matrix(((1,),))
        else: data = tensor((3, 2, 5))
        with pytest.raises(ValueError): interface.admit(view, data, status='derived' if kind == 'derived' else 'observed')


def test_live_program_owner_required_and_native_handles_outlive_python_producers(prepared):
    _, _, program, asset, accession = prepared
    other = PreparedAccessionFieldProgram(program.archive())
    input_ = accession.admit(111, tensor((2, 3)))
    with pytest.raises(ValueError): other.execute((input_,))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(program.execute, [(input_,)]*8))
    del program, asset, accession
    assert len({r.archive() for r in results}) == 1
    for cls, method in ((NativeInputUseAsset, 'archive'), (PreparedAccessionFieldProgram, 'archive'),
                        (NativeAccessionInput, 'inspect'), (NativeAccessionFieldResult, 'archive'),
                        (NativeFieldInterface, 'inspect'), (NativeFieldReading, 'inspect'), (NativeFieldSection, 'archive')):
        with pytest.raises(RuntimeError): getattr(cls.__new__(cls), method)()


def test_equal_literal_bytes_with_distinct_declared_context_fields_glue_only_on_their_own_interfaces(prepared):
    _, _, program, asset, accession = prepared
    results = (program.execute((accession.admit(use, tensor(values)),)) for use, values in ((111, (2, 3)), (112, (5, -1))))
    first, second = results
    assert asset.recover(111) == asset.recover(112)
    assert first.field_archive(2) != second.field_archive(2)
    interfaces = [r.interface(patch_declaration()) for r in (first, second)]
    readings = [i.observe(view_declaration()) for i in interfaces]
    assert readings[0].inspect()['visible'] != readings[1].inspect()['visible']
    for reading in readings:
        section = NativeFieldSection((reading, reading))
        assert section.readings()[0].interface().same_owner(reading.interface())
    with pytest.raises(ValueError): NativeFieldSection(readings)


def test_oblique_original_basis_changes_preserve_complete_fields_and_view_metrics(prepared):
    owner, _, program, asset, accession = prepared
    original = program.execute((accession.admit(111, tensor((2, 3))),))
    spec = deepcopy(declaration())
    spec['stalks'][0].update(basis=777, directions=(778, 779), metric=matrix(((2, -1), (-1, 3))))
    for slot in (0, 2, 5):
        rows = MAPS[slot]
        spec['occurrences'][slot]['map'] = matrix((tuple(a+b for a, b in zip(rows[0], rows[1], strict=True)), rows[1]))
    spec['nodes'][2]['map'] = matrix(tuple((row[0], row[1]-row[0], *row[2:]) for row in LATER))
    framed_program = PreparedAccessionFieldProgram(PreparedFieldProgram(owner.archive, spec).archive())
    framed_accession = NativeInputAccession(asset, framed_program, accession_declaration())
    framed = framed_program.execute((framed_accession.admit(111, tensor((2, 3))),))
    for node in (2, 4):
        selected = original.select(node, ((0, 0), (0, 1), (0, 2)), coordinate_limit=7)
        rows = tuple(v for stalk in selected for v in stalk['values'])
        assert framed.field_archive(node) == tensor((rows[0]+rows[1], rows[1], *rows[2:]))
    assert framed.field_archive(1) == original.field_archive(1) and framed.field_archive(3) == original.field_archive(3)
    visible = original.interface(patch_declaration()).observe(view_declaration()).inspect()
    view = view_declaration(); view.update(accession=matrix(((1, 0), (0, 1), (1, 1))), reconstruction=matrix(((1, 0, 0), (0, 1, 0))))
    transformed = framed.interface(patch_declaration()).observe(view).inspect()
    assert visible['visible'] == transformed['visible'] and visible['metric'] == transformed['metric']
    assert framed.select(2, ((0, 0),), coordinate_limit=2)[0]['directions'] == (778, 779)


@pytest.mark.parametrize('suffix', ['.rcbd', '.safetensors'])
def test_source_backed_provider_receipts_dependencies_and_cold_model_output_replay(prepared, tmp_path, suffix):
    owner, original, _, asset, _ = prepared
    state = cpp_accession_field_program_model(owner, declaration())
    query = accession_field_program_query(state, ({'asset': asset.archive(), 'accession': accession_declaration(), 'use': 111, 'field': tensor((2, 3))},))
    output = infer_model(state, owner, query)
    assert state.adapter_version == '2' and len(output.dependencies) == 1
    assert output.dependencies[0].native.namespace == 993003
    assert output.dependencies[0].native.archive_digest == original.identity['archive_digest']
    result = read_field_program_result(output, state=state)
    record = model_record(owner, state); record.attach_metadata(1, 0, 'response', output)
    path = tmp_path / ('source-backed' + suffix); save_model(path, record)
    clear_field_program_cache(); restored, actual = read_model_record(load_model(path))
    assert restored.coefficient_digest == state.coefficient_digest
    assert infer_model(restored, actual, query).field_archive['sha256'] == output.field_archive['sha256']
    packet = tmp_path / 'output.value'; packet.write_bytes(pack_value(pack_model(output, native=True)))
    code = '''
from pathlib import Path
import sys
from rexgraph.value_codec import unpack_value
from rexgraph.model_codec import unpack_model
from rexgraph.cpp_relational_field_program import read_field_program_result
output=unpack_model(unpack_value(Path(sys.argv[1]).read_bytes()))
result=read_field_program_result(output)
assert result.input_uses()[0]['use']==111
assert len(result.fields())==4
assert output.dependencies[0].native.namespace==993003
'''
    subprocess.run([sys.executable, '-I', '-c', code, str(packet)], cwd=tmp_path, check=True)
    with pytest.raises(ValueError): read_field_program_result(replace(output, dependencies=()))
    assert result.archive() == read_field_program_result(output).archive()


def test_rcql_selects_the_explicit_source_backed_profile_and_system_retains_archive(prepared):
    from rcql import Executor, parse
    from system.serialize import json_value
    owner, _, _, asset, _ = prepared
    state = cpp_accession_field_program_model(owner, declaration())
    query = accession_field_program_query(state, ({'asset': asset.archive(), 'accession': accession_declaration(), 'use': 111, 'field': tensor((2, 3))},))
    output = infer_model(state, owner, query); record = model_record(owner, state)
    values = Executor(sources={'r': record}, params={'c': {'source_backed': True, 'declaration': declaration()}, 'q': query, 'o': output}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_relational_field_program",$c),MODEL_INFER(MODEL_STATE(),$q),'
        'MODEL_STALK($o,[0,0],node=2,max_coordinates=2),MODEL_VALUES($o),MODEL_INFO(MODEL_STATE())')).values
    assert values[0].adapter_version == '2' and values[0].coefficient_digest == state.coefficient_digest
    assert values[1].field_archive['sha256'] == output.field_archive['sha256']
    assert len(values[2]['values']) == 2 and len(values[3]['fields']) == 4 and values[4]['source_backed']
    assert json_value(state)['source_backed'] and json_value(output)['retains_whole_fields']


@pytest.mark.parametrize('backend', ['file', 'rex', 'sqlite'])
def test_original_source_dependencies_survive_checked_store_copies_and_agent_workers(prepared, tmp_path, backend):
    from rcdb import open_store, copy_record
    from agent.hive import Hive
    owner, original, _, asset, _ = prepared
    state = cpp_accession_field_program_model(owner, declaration())
    query = accession_field_program_query(state, ({'asset': asset.archive(), 'accession': accession_declaration(), 'use': 113, 'field': tensor((2, 3))},))
    output = infer_model(state, owner, query)
    record = model_record(owner, state); record.attach_metadata(1, 0, 'response', output)
    def store(name):
        if backend == 'sqlite': return open_store('sqlite:///' + str(tmp_path / (name + '.sqlite')))
        return open_store(backend + '://' + str(tmp_path / name), read_only=False)
    stores = [store(name) for name in ('first', 'second')]
    try:
        stored = stores[0].put('source-backed', record)
        copy_record(stores[0], stores[1], stored)
        copied = stores[1].get('source-backed')
        restored, actual = read_model_record(copied)
        response = copied.get_metadata(1, 0, 'response')
        assert response.dependencies[0].native.archive_digest == original.identity['archive_digest']
        assert read_field_program_result(response, state=restored).archive() == read_field_program_result(output).archive()
        assert infer_model(restored, actual, query).field_archive['sha256'] == output.field_archive['sha256']
        hive = Hive()
        hive.add_worker('source-backed-fields', lambda value: infer_model(restored, actual, value), capability='analyze')
        assert hive.dispatch_capability('analyze', query)['result'].field_archive['sha256'] == output.field_archive['sha256']
    finally:
        for item in stores: item.close()

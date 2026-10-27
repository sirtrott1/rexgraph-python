"""Exact measured assignment learning retains native owners and whole responses."""
import io
import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

import rexgraph_cpp as cpp

POLICY = dict(limits=(4096, 16000000, 64000000, 1 << 24), max_bytes=256*1024*1024)


def rationals(value):
    if isinstance(value, dict): return {k: rationals(v) for k, v in value.items()}
    if isinstance(value, list): return tuple(rationals(v) for v in value)
    return value


def binding(value):
    out = rationals(value)
    view = dict(out['view'])
    for name in ('accession', 'reconstruction', 'physical'):
        matrix = view[name]
        if matrix is not None:
            view[name] = dict(shape=matrix['shape'], entries=tuple((i, j, Fraction(v)) for i, j, v in matrix['entries']))
    return dict(out, view=view)


class Fixture:
    def __init__(self):
        folder = os.environ.get('REXGRAPH_ASSIGNMENT_FIXTURE_DIR')
        if folder is None: pytest.skip('native exact assignment fixture required')
        self.root = Path(folder)
        self.data = json.loads((self.root / 'fixture.json').read_text())
        self.origins = {row['key']: cpp.NativeValueOrigin(row['key'], self.read(row['history']), row['cutoff'], row['role'],
            self.read(row['payload']), **POLICY) for row in self.data['origins']}

    def read(self, name): return (self.root / name).read_bytes()

    def context(self, remove=(), limits=None, only=None):
        limits = {} if limits is None else limits
        return cpp.NativeAvailabilityContext(tuple(dict(origin=self.origins[row['key']],
            namespace=row['availability']['namespace'], sequence=limits.get(row['key'], row['availability']['sequence']))
            for row in self.data['origins'] if row['key'] not in remove and (only is None or row['key'] in only)))

    def candidates(self):
        return tuple(dict(key=row['key'], assignment=tuple(row['assignment']), binding=binding(row['binding']),
            evidence=cpp.PreparedAvailableFieldEvidence.from_archive(self.read(row['evidence']), self.context(), **POLICY),
            shift=tuple(Fraction(v) for v in row['shift']), injection=dict(shape=tuple(row['injection']['shape']),
                entries=tuple((i, j, Fraction(v)) for i, j, v in row['injection']['entries']))) for row in self.data['candidates'])

    def model(self):
        finite = cpp.NativeFiniteSource.from_archive(self.read(self.data['finite_source']), **POLICY)
        declaration = dict(self.data['declaration']); declaration['parameters'] = tuple(declaration['parameters'])
        candidates = self.candidates()
        assert cpp.assignment_root_origin_payload(declaration, finite, candidates, **POLICY) == self.origins[declaration['origin']].payload()
        return cpp.NativeAssignmentEvidence(declaration, finite, candidates, self.context(), **POLICY)

    def observation(self, model, index):
        row = self.data['observations'][index]
        indices = {model.candidate(i, self.context())['key']: i for i in range(model.inspect(self.context())['candidate_count'])}
        return dict(key=row['key'], origin=row['origin'], binding=binding(row['binding']),
            measured=cpp.NativeAvailableFieldResult.from_archive(self.read(row['measured']), self.context(), **POLICY),
            inputs=tuple(dict(candidate=item['candidate'], input=model.restore_input(indices[item['candidate']],
                self.read(item['archive']), self.context())) for item in row['inputs']))

    def learned(self):
        root = self.model(); versions = [root]
        for i in range(2): versions.append(versions[-1].append((self.observation(versions[-1], i),), self.context()))
        return tuple(versions)


@pytest.fixture(scope='module')
def case(): return Fixture()


def test_original_root_and_each_immutable_child_replay_native_bytes(case):
    versions = case.learned()
    for i, model in enumerate(versions, 1):
        assert model.archive(case.context()) == case.read(f'model-v{i}.bin')
        sink = io.BytesIO(); count = model.stream(sink.write, case.context())
        assert count == len(sink.getvalue()) and sink.getvalue() == model.archive(case.context())
        cold = cpp.NativeAssignmentEvidence.from_archive(sink.getvalue(), case.context(), **POLICY)
        assert not cold.same_owner(model) and cold.archive(case.context()) == sink.getvalue()
        assert cold.inspect(case.context()) == model.inspect(case.context())
        if i > 1: assert model.parent(case.context()).same_owner(versions[i - 2])
    assert versions[-1].inspect(case.context())['identified_candidate'] == 0
    assert versions[-1].branch(0, case.context())['coordinates']['free_dimension'] == 1
    assert versions[-1].solve(0, 0, case.context()).same_owner(versions[1].solve(0, 0, case.context()))
    for i in range(2):
        observation = versions[-1].observation(i, case.context())
        assert cpp.assignment_measurement_origin_payload(observation, **POLICY) == case.origins[observation['origin']].payload()
        assert observation['measured'].archive() == case.read(case.data['observations'][i]['measured'])


@pytest.mark.parametrize('ordinal', (0, 1))
def test_conditioned_whole_held_fields_retain_source_and_inactive_directions(case, ordinal):
    model = case.learned()[-1]; held = case.data['held'][ordinal]
    input = model.restore_input(0, case.read(held['input']), case.context())
    required = {v.inspect()['key'] for v in (*model.origins(case.context()), *cpp.assignment_input_origins(input, case.context()))}
    authority = case.context(only=required)
    response = model.execute(0, input, authority)
    assert response.archive(authority) == case.read(held['response'])
    assert response.model(authority).same_owner(model)
    assert response.inspect(authority)['coordinates']['free_dimension'] == 1
    actual = cpp.NativeAvailableFieldResult.from_archive(case.read(held['actual']), case.context(), **POLICY)
    assert response.field_archive(3, authority)['particular'] == actual.result().field_archive(3)
    assert response.field_archive(3, authority)['free_parameter_count'] == 1
    assert response.field_archive(3, authority)['direction_nonzeros'] == 0
    assert response.original(authority).result().field_archive(3)['free_parameter_count'] == 4
    assert len(response.coefficient_independent_map(1, 2, authority)['directions']) == 1
    sink = io.BytesIO(); response.stream(sink.write, authority)
    assert sink.getvalue() == case.read(held['response'])
    cold = cpp.NativeAssignmentResponse.from_archive(sink.getvalue(), authority, **POLICY)
    assert cold.field_archive(3, authority) == response.field_archive(3, authority)
    assert {v.inspect()['key'] for v in response.origins(authority)} == required


def test_uncovered_catalog_alternative_is_retained_without_guessed_fields(case):
    model = cpp.NativeAssignmentEvidence.from_archive(case.read('incomplete-catalog.bin'), case.context(), **POLICY)
    assert model.inspect(case.context())['status'] == 'unresolved'
    assert model.inspect(case.context())['identified_candidate'] is None
    assert model.remainder(0, 1, case.context()) == ((999,),)


def test_live_foreign_input_requires_explicit_full_archive_adoption(case):
    model = case.learned()[-1]
    foreign = cpp.NativeAssignmentEvidence.from_archive(model.archive(case.context()), case.context(), **POLICY)
    image = case.read(case.data['held'][0]['input'])
    other_input = foreign.restore_input(0, image, case.context())
    with pytest.raises(ValueError): model.execute(0, other_input, case.context())
    assert model.execute(0, model.restore_input(0, other_input.archive(), case.context()), case.context()).archive(case.context()) == case.read(case.data['held'][0]['response'])


def test_current_authority_preflight_cancellation_and_stream_failures(case):
    model = case.learned()[-1]; root_key = model.inspect(case.context())['origin']
    denied = case.context(remove=(root_key,))
    for action in (lambda: model.admit(denied), lambda: model.archive(denied), lambda: model.origins(denied),
                   lambda: cpp.NativeAssignmentEvidence.from_archive(model.archive(case.context()), denied, **POLICY)):
        with pytest.raises(ValueError): action()
    class Cancelled(Exception): pass
    def cancel(): raise Cancelled('stopped')
    with pytest.raises(Cancelled): model.archive(case.context(), check=cancel)
    with pytest.raises(OSError): model.stream(lambda part: 0, case.context())
    assert model.archive(case.context()) == case.read('model-v3.bin')

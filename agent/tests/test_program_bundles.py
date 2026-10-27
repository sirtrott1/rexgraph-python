"""Explicit bundle ports retain actual field owners through Hive and RCDB."""
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from agent import RelationalSession
from agent.hive import Hive
from agent.record_access import PinnedRecords
from rcdb import LocalStore, MemoryStore, copy_record
from rcql import OutputRef, Program, ProgramInput, ProgramStep, Result, parse
from rexgraph.cochain import Chain
from rexgraph.graph import RexGraph
from rexgraph.weighted_dirac import GradedChain


def bundle():
    a = RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2])
    b = RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2])
    def field(source):
        base = GradedChain(source)
        return GradedChain(source, tuple(Chain(g, np.array([
            [Fraction(i + 1, 7), Fraction(2 * i + 3, 7)] for i in range(c.n_cells)], dtype=object),
            source=source) for g, c in enumerate(base.components)))
    return Result((a, b, field(a), field(b)), aliases=("r0", "r1", "x0", "x1"))


def program():
    return Program("bundle", (ProgramStep("whole", parse("FROM $r0 RETURN APPLY(DIRAC(),$x0,true)")),),
        (ProgramInput("x0", "GradedChain"),), (("answer", OutputRef("whole")),))


@pytest.mark.parametrize('backend', ['memory', 'local'])
def test_bundle_worker_preserves_both_owners_complete_fields_history_and_copy(backend, tmp_path):
    selected = MemoryStore() if backend == 'memory' else LocalStore(tmp_path / 'records')
    with closing(selected) as store, closing(LocalStore(tmp_path / 'copy')) as destination:
        session = RelationalSession(store)
        session.save_result('inputs', bundle())
        session.publish_program('program', program(), expected_version=0)
        pin = PinnedRecords(session).pin('inputs')
        sources, parameters = {'r0': 0, 'r1': 1}, {'x0': 2}
        hive = Hive('bundle-program')
        try:
            hive.add_program_bundle('run', session, 'program', version=1, bundle=pin,
                source_outputs=sources, parameter_outputs=parameters)
            pin['version'] = 99; sources['r0'] = 1; parameters['x0'] = 3
            value = hive.invoke('run', {})
            a, b, x, y = value.native_plan['bundle_values']
            assert a is not b and x.source is a and y.source is b
            assert value.values[0].source is a
            assert value.native_plan['step_results'][0]['values'][0].source is a
            assert all(not c.values.flags.writeable for c in value.values[0].components)
            session.save_result('whole', value)
            restored = session.read('whole').value
            assert restored.to_bytes() == value.to_bytes()
            a, b, x, y = restored.native_plan['bundle_values']
            assert a is not b and x.source is a and y.source is b and restored.values[0].source is a
            copy_record(store, destination, store.get_record('whole'), destination_id='copied')
            assert destination.read_record('copied').value.to_bytes() == value.to_bytes()
            session.save_result('inputs', bundle(), expected_version=1)
            assert hive.invoke('run', {}).to_bytes() == value.to_bytes()
            with pytest.raises(ValueError, match='overlap'):
                hive.invoke('run', {'x0': x})
        finally:
            hive.stop_all()


@pytest.mark.parametrize('defect', ['foreign-owner', 'negative', 'bool', 'outside', 'reserved', 'foreign-store'])
def test_bundle_does_not_infer_transport_or_admit_invalid_ports(defect):
    with closing(MemoryStore()) as store:
        session = RelationalSession(store)
        session.save_result('inputs', bundle())
        session.publish_program('program', program(), expected_version=0)
        pin = PinnedRecords(session).pin('inputs')
        sources, parameters = {'r0': 0, 'r1': 1}, {'x0': 2}
        if defect == 'foreign-owner': parameters['x0'] = 3
        elif defect == 'negative': sources['r0'] = -1
        elif defect == 'bool': parameters['x0'] = True
        elif defect == 'outside': sources['r0'] = 10
        elif defect == 'reserved': sources['db'] = 0
        elif defect == 'foreign-store': pin['store_id'] = 'elsewhere'
        with pytest.raises((ValueError, TypeError)):
            session.run_program_bundle('program', version=1, bundle=pin,
                source_outputs=sources, parameter_outputs=parameters, save_as='forbidden')
        assert store.get_record('forbidden') is None


def test_program_worker_keeps_an_owned_native_publication(tmp_path, monkeypatch):
    native=pytest.importorskip('rexgraph_cpp')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'bindings/cpp/tests'))
    from test_construction import declaration
    from rexgraph.cpp_model import cpp_model, cpp_rational_batch_input
    from rexgraph.io.model_record import model_record
    from rexgraph.model_runtime import infer_model
    from rexgraph.source_binding import CppSourceReference
    from rcql.result_codec import bind_sources
    source=native.NativeSource.from_declaration(declaration())
    from rexgraph.type_accession import CoordinateSpace
    state=cpp_model(source,(10,11,12))
    field=cpp_rational_batch_input(state,source,(1001,),CoordinateSpace('samples',('one',)))
    bound=CppSourceReference(source)
    assert bind_sources(bound,{},strict=True) is bound
    with pytest.raises(ValueError,match='unbound source'):
        bind_sources(state.source.detached(),{},strict=True)
    identity=(state.source.state_digest,state.source.record_id,state.source.version)
    with pytest.raises(TypeError):bind_sources(state.source,{identity:RexGraph.from_cells([1,[]])},strict=True)
    assert isinstance(state.source,CppSourceReference)
    p=Program('native_publication',(ProgramStep('whole',parse('FROM $r RETURN MODEL_INFER($model,$field)')),),
        (ProgramInput('model','ModelState'),ProgramInput('field')))
    with closing(LocalStore(tmp_path/'native')) as store:
        session=RelationalSession(store)
        session.publish('source',model_record(source,state),expected_version=0)
        session.publish_program('program',p,expected_version=0)
        hive=Hive('native-publication')
        try:
            hive.add_program('run',session,'program',version=1,sources={'r':{'record_id':'source','version':1}},
                runtime_parameters={'model':lambda:state,'field':lambda:field})
            result=hive.invoke('run',{})
            expected=infer_model(state,source,field)
            assert result.values[0].source.matches(state.source)
            assert result.values[0].values.tolist()==expected.values.tolist()
            session.save_result('result',result)
            assert session.read('result').value.to_bytes()==result.to_bytes()
        finally:hive.stop_all()

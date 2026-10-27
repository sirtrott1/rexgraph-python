"""Ontology publication and RCQL execution through the existing Agent session."""
from __future__ import annotations

from fractions import Fraction


def _session(session):
    from .relational import RelationalSession
    if not isinstance(session, RelationalSession):
        raise TypeError("ontology access requires the existing RelationalSession")
    return session


def read_ontology(session, record_id, *, version=None, cutoff, anchor=(1, 0)):
    """Check both actual publication time and the declared knowledge horizon."""
    from rexgraph.ontology import OntologyState
    from rexgraph.value import ExactTime
    if not isinstance(cutoff, ExactTime):
        raise TypeError("ontology read requires an exact knowledge cutoff")
    snapshot = _session(session).read(record_id, version)
    # RCDB transaction clocks are binary floats. Compare their actual exact
    # values instead of rounding an exact cutoff into the store clock.
    if Fraction(snapshot.record.tx_from) > cutoff.seconds:
        raise ValueError("ontology publication lies beyond the knowledge cutoff")
    value = (snapshot.value if isinstance(snapshot.value, OntologyState)
             else OntologyState.from_record(snapshot.value, anchor=anchor))
    value.require_horizon(cutoff)
    if value.dependencies:
        from rcql import BoundSource, SourcePolicy
        from rcql.source_context import SnapshotContext, SourceSelection
        selections = tuple(SourceSelection("dependency_" + str(i), d.record_id, version=d.version)
                           for i, d in enumerate(value.dependencies))
        context = SnapshotContext.select(BoundSource(session.store, SourcePolicy.allow("read", "identity")),
                                         selections, cutoff=cutoff.seconds)
        context.validate_values((value,))
    return value


def publish_ontology(session, record_id, ontology, *, expected_version, meta=None):
    """Publish complete graph state with checked lineage and version concurrency."""
    from rexgraph.ontology import OntologyState
    if not isinstance(ontology, OntologyState):
        raise TypeError("ontology publication requires an OntologyState")
    session = _session(session)
    if type(expected_version) is not int or expected_version < 0:
        raise ValueError("ontology publication requires an explicit nonnegative version")
    if expected_version:
        snapshot = session.read(record_id, expected_version)
        parent = (snapshot.value if isinstance(snapshot.value, OntologyState)
                  else OntologyState.from_record(snapshot.value, anchor=ontology._anchor))
        if not ontology.is_child_of(parent):
            raise ValueError("ontology revision requires the actual published parent")
    elif ontology.declaration["parent"] is not None:
        raise ValueError("a child ontology cannot replace an unbound root publication")
    return session.publish(record_id, ontology.record(), expected_version=expected_version, meta=meta)


def load_ontology_file(path, **_options):
    """Return the full graph, including ontology declarations and native archives."""
    from rexgraph.io.rcbo import load_rcbo
    return load_rcbo(path).record()


def native_revision_program(name, *, operation="append", source="r"):
    """Declare a checked RCQL revision worker without capturing live resources.

    Register it with ``Hive.add_program(..., allow_write=True)`` and an explicit
    parent record version. The returned child is published separately through
    ``publish_ontology`` with that parent's expected version.
    """
    from rcql import OutputRef, Program, ProgramInput, ProgramStep, parse
    if type(source) is not str or not source.isidentifier() or not source.isascii():
        raise ValueError("ontology program source requires an ASCII identifier")
    if operation not in ("append", "add"):
        raise ValueError("ontology program operation requires append or add")
    payload = "frame" if operation == "append" else "specification"
    query = ('FROM $' + source + ' RETURN ONTOLOGY_NATIVE_' + operation.upper() +
             '(ONTOLOGY_READ(), $component, $' + payload + ', $cutoff)')
    return Program(name, (ProgramStep("revision", parse(query)),),
        (ProgramInput("component", "Text"),
         ProgramInput(payload, "Bytes" if operation == "append" else None),
         ProgramInput("cutoff", "ExactInteger")),
        (("ontology", OutputRef("revision")),))


def register(register_reader=None):
    if register_reader is None:
        from .adapters.formats import register_reader
    register_reader("relational_ontology", load_ontology_file, extensions=(".rcbo",))


__all__ = ["read_ontology", "publish_ontology", "load_ontology_file", "native_revision_program", "register"]

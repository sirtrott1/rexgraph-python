"""Ontology snapshots and explicit validation through existing query authority."""
from fractions import Fraction

from .layer_operators import _authorize
from .execution_trace import record_method


def _time(value):
    from rexgraph.value import ExactTime
    if type(value) is int or isinstance(value, Fraction):
        return ExactTime(Fraction(value))
    if not isinstance(value, ExactTime):
        raise TypeError("ontology horizon requires an exact timestamp")
    return value


def ontology_read(source, grade=1, index=0):
    from rexgraph.ontology import OntologyState
    return OntologyState.from_record(source, anchor=(grade, index))


def ontology_import(source, archive):
    from rexgraph.ontology import OntologyState
    return OntologyState.from_bytes(archive)


def ontology_export(source, ontology):
    return ontology.to_bytes()


def ontology_record(source, ontology):
    return ontology.record()


def ontology_info(source, ontology):
    return ontology.inspect()


def ontology_declaration(source, ontology):
    return ontology.declaration


def ontology_at(source, ontology, cutoff):
    ontology.require_horizon(_time(cutoff))
    return ontology


def ontology_revise(source, ontology, graph, changes, sources=None):
    if type(changes) is not dict or not {"known_at", "kind", "maps"} <= changes.keys():
        raise ValueError("ontology changes require known_at, kind and maps")
    _authorize(ontology, sources, training=True)
    changes = dict(changes, known_at=_time(changes["known_at"]))
    child = ontology.revise(graph, **changes)
    record_method("declared-ontology-revision", parent=ontology.digest, child=child.digest,
                  mathematical_validation="not-evaluated")
    return child


def ontology_verify(source, ontology, cutoff, sources=None, parent=None):
    ontology.require_horizon(_time(cutoff))
    bindings = _authorize(ontology, sources)
    if parent is not None:
        parent.require_horizon(_time(cutoff))
        _authorize(parent, bindings)
    record_method("explicit-ontology-provider-validation")
    result = ontology.verify(sources=bindings, parent=parent)
    if type(result) is not dict:
        raise TypeError("ontology provider must return a closed validation record")
    return result


def _native_revision(ontology, component, payload, cutoff, sources, operation):
    from .execution_trace import current_binding
    from rexgraph.tensor_field import FieldSource
    _authorize(ontology, sources, training=True)
    binding = current_binding()
    if binding is None:
        raise PermissionError("native ontology revision requires its current source binding")
    parent = FieldSource(binding.value, binding.ref.record_id, binding.ref.record_version)
    if parent.state_digest != ontology.source_digest:
        raise ValueError("native ontology parent differs from the selected query source")
    child = operation(ontology, component, payload, known_at=_time(cutoff), parent_source=parent)
    record_method("native-ontology-source-revision", parent=ontology.digest, child=child.digest,
                  component=component, axiom_validation="not-evaluated")
    return child


def ontology_native_append(source, ontology, component, frame, cutoff, sources=None):
    from rexgraph.native_ontology import append_native_ontology
    return _native_revision(ontology, component, frame, cutoff, sources, append_native_ontology)


def ontology_native_add(source, ontology, component, specification, cutoff, sources=None):
    from rexgraph.native_ontology import add_native_ontology_declarations
    return _native_revision(ontology, component, specification, cutoff, sources, add_native_ontology_declarations)


def install(register):
    from .ontology_contracts import ARGUMENTS
    for name in ARGUMENTS:
        register(name)(globals()[name.lower()])
